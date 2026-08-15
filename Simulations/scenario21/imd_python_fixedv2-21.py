import argparse
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    Image = None
    ImageDraw = None
    ImageFont = None


def norm_pdf(z):
    """Standard normal density, vectorized for scalars or NumPy arrays."""
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def norm_cdf(z):
    """Standard normal distribution function, vectorized without SciPy."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def rho_H(z, scale):
    """
    Smoothed absolute-value loss used by the R code:
    z * (2 * pnorm(z / scale) - 1) + 2 * scale * dnorm(z / scale)
    """
    z = np.asarray(z, dtype=float)
    return z * (2.0 * norm_cdf(z / scale) - 1.0) + 2.0 * scale * norm_pdf(z / scale)


def psi_H(z, scale):
    """First derivative of rho_H."""
    z = np.asarray(z, dtype=float)
    return 2.0 * norm_cdf(z / scale) - 1.0


def psi_prime_H(z, scale):
    """Derivative of psi_H."""
    z = np.asarray(z, dtype=float)
    return 2.0 * norm_pdf(z / scale) / scale


def epanechnikov(v):
    """Epanechnikov kernel."""
    v = np.asarray(v, dtype=float)
    return 0.75 * np.maximum(1.0 - v**2, 0.0) * (np.abs(v) <= 1.0)


def periodic_extend(values, grid):
    """Periodically extend values and keep the extended grid in [-0.5, 1.5]."""
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    ext_grid = np.concatenate([grid - 1.0, grid, grid + 1.0])
    ext_values = np.tile(values, 3)
    keep = (ext_grid >= -0.5) & (ext_grid <= 1.5)
    return ext_grid[keep], ext_values[keep]


def golden_section_minimize(func, lower, upper, tol=1e-6, max_iter=200):
    """Bounded scalar minimizer, matching R's optimize closely enough here."""
    lower = float(lower)
    upper = float(upper)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - math.sqrt(5.0)) / 2.0

    interval = upper - lower
    if interval <= tol:
        return (lower + upper) / 2.0

    n_iter = min(max_iter, int(math.ceil(math.log(tol / interval) / math.log(inv_phi))))
    c = lower + inv_phi_sq * interval
    d = lower + inv_phi * interval
    yc = func(c)
    yd = func(d)

    for _ in range(n_iter):
        if not np.isfinite(yc):
            yc = float("inf")
        if not np.isfinite(yd):
            yd = float("inf")

        if yc < yd:
            upper = d
            d = c
            yd = yc
            interval *= inv_phi
            c = lower + inv_phi_sq * interval
            yc = func(c)
        else:
            lower = c
            c = d
            yc = yd
            interval *= inv_phi
            d = lower + inv_phi * interval
            yd = func(d)

    return (lower + upper) / 2.0


def weighted_minimizer(target_grid, obs_grid, objective_at_a, bandwidth, lower, upper):
    """Find the weighted objective minimizer for each target-grid point."""
    target_grid = np.asarray(target_grid, dtype=float)
    obs_grid = np.asarray(obs_grid, dtype=float)
    results = np.empty(len(target_grid), dtype=float)

    for i, t0 in enumerate(target_grid):
        weights = epanechnikov((t0 - obs_grid) / bandwidth) / bandwidth
        active = weights > 0.0

        def objective_value(a):
            contributions = np.asarray(objective_at_a(a, active), dtype=float)
            finite = np.isfinite(contributions)
            if not np.any(finite):
                return float("inf")
            active_weights = weights[active]
            return float(np.sum(active_weights[finite] * contributions[finite]))

        results[i] = golden_section_minimize(objective_value, lower, upper, tol=1e-6)

    return results


def imd_decompose_with_linearization(
    Y,
    x_grid,
    true_signal,
    K_steps=8,
    h_1=0.2,
    H=1,
    epsi_prime_tol=1e-6,
):
    """
    Iterative multiscale decomposition plus the linearization diagnostics in combine2.R.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.ndim != 2:
        raise ValueError("Y must be a vector or a 2D matrix.")

    n_time, n_paths = Y.shape
    x_grid = np.asarray(x_grid, dtype=float)
    true_signal = np.asarray(true_signal, dtype=float)
    if len(x_grid) != n_time:
        raise ValueError("x_grid length must match the number of rows in Y.")
    if len(true_signal) != n_time:
        raise ValueError("true_signal length must match the number of rows in Y.")

    h_sequence = h_1 / (math.sqrt(2.0) ** np.arange(K_steps))
    tilde_S = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    Y_step = np.full((n_time, n_paths, K_steps + 1), np.nan, dtype=float)
    Y_step[:, :, 0] = Y
    S_empirical = np.full((n_time, K_steps), np.nan, dtype=float)

    epsilon_step = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    psi_epsilon = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    psi_prime_epsilon = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    Epsi_prime = np.full((n_time, K_steps), np.nan, dtype=float)
    N_pop_t = np.full((n_time, K_steps), np.nan, dtype=float)
    N_t = np.full((n_time, K_steps), np.nan, dtype=float)
    L_pop = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    L_emp = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    L_pop_raw = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    L_emp_raw = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    pdf_linear = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    simple_linear = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    X_step = np.full((n_time, K_steps + 1), np.nan, dtype=float)
    X_step[:, 0] = true_signal

    for k in range(K_steps):
        h_k = h_sequence[k]
        Y_current = Y_step[:, :, k]

        ext_list = [periodic_extend(Y_current[:, j], x_grid) for j in range(n_paths)]
        ext_grid = ext_list[0][0]
        Y_current_ext = np.column_stack([values for _, values in ext_list])

        for j in range(n_paths):
            def objective_path(a, active, path_index=j):
                return rho_H(Y_current_ext[active, path_index] - a, H)

            tilde_S[:, j, k] = weighted_minimizer(
                target_grid=x_grid,
                obs_grid=ext_grid,
                objective_at_a=objective_path,
                bandwidth=h_k,
                lower=np.nanmin(Y_current[:, j]) - 4.0 * H,
                upper=np.nanmax(Y_current[:, j]) + 4.0 * H,
            )

        def objective_pooled(a, active):
            values = rho_H(Y_current_ext[active, :] - a, H)
            return np.nanmean(values, axis=1)

        S_empirical[:, k] = weighted_minimizer(
            target_grid=x_grid,
            obs_grid=ext_grid,
            objective_at_a=objective_pooled,
            bandwidth=h_k,
            lower=np.nanmin(Y_current) - 4.0 * H,
            upper=np.nanmax(Y_current) + 4.0 * H,
        )

        X_step[:, k + 1] = X_step[:, k] - S_empirical[:, k]
        epsilon_step[:, :, k] = Y_current - X_step[:, k, np.newaxis]
        psi_epsilon[:, :, k] = psi_H(epsilon_step[:, :, k], H)
        psi_prime_epsilon[:, :, k] = psi_prime_H(epsilon_step[:, :, k], H)
        Epsi_prime[:, k] = np.nanmean(psi_prime_epsilon[:, :, k], axis=1)

        eps_ext_list = [periodic_extend(epsilon_step[:, j, k], x_grid) for j in range(n_paths)]
        eps_ext = np.column_stack([values for _, values in eps_ext_list])

        for i, t0 in enumerate(x_grid):
            weights = epanechnikov((t0 - ext_grid) / h_k) / h_k
            active = weights > 0.0
            active_weights = weights[active]

            psi_eps_active = psi_H(eps_ext[active, :], H)
            psi_prime_eps_active = psi_prime_H(eps_ext[active, :], H)
            Epsi_by_u = np.nanmean(psi_eps_active, axis=1)
            Epsi_prime_by_u = np.nanmean(psi_prime_eps_active, axis=1)
            N_pop_t[i, k] = np.nansum(active_weights * Epsi_prime_by_u)

            arg_pdf = Y_current_ext[active, :] - S_empirical[i, k]
            psi_pdf = psi_H(arg_pdf, H)
            psi_prime_pdf = psi_prime_H(arg_pdf, H)
            E_psi_pdf_by_u = np.nanmean(psi_pdf, axis=1)
            E_psi_prime_pdf_by_u = np.nanmean(psi_prime_pdf, axis=1)
            N_t[i, k] = np.nansum(active_weights * E_psi_prime_pdf_by_u)

            simple_denom = np.nansum(active_weights) * np.nanmean(psi_prime_eps_active)
            N_pop_safe = max(N_pop_t[i, k], epsi_prime_tol)
            N_emp_safe = max(N_t[i, k], epsi_prime_tol)
            simple_safe = simple_denom if abs(simple_denom) >= epsi_prime_tol else math.copysign(epsi_prime_tol, simple_denom or 1.0)

            for j in range(n_paths):
                raw_eps_score = psi_eps_active[:, j]
                centered_eps_score = raw_eps_score - Epsi_by_u
                weighted_raw = np.nansum(active_weights * raw_eps_score)
                weighted_centered = np.nansum(active_weights * centered_eps_score)

                L_pop_raw[i, j, k] = weighted_raw / N_pop_safe
                L_emp_raw[i, j, k] = weighted_raw / N_emp_safe
                L_pop[i, j, k] = weighted_centered / N_pop_safe
                L_emp[i, j, k] = weighted_centered / N_emp_safe

                diamond = psi_pdf[:, j] - E_psi_pdf_by_u
                pdf_linear[i, j, k] = np.nansum(active_weights * diamond) / N_emp_safe
                simple_linear[i, j, k] = weighted_centered / simple_safe

        Y_step[:, :, k + 1] = Y_current - tilde_S[:, :, k]

    tilde_error = tilde_S - S_empirical[:, np.newaxis, :]
    summary = []
    for k in range(K_steps):
        actual = tilde_error[:, :, k].ravel()
        approximations = {
            "L_pop": L_pop[:, :, k].ravel(),
            "L_emp": L_emp[:, :, k].ravel(),
            "L_pop_raw": L_pop_raw[:, :, k].ravel(),
            "L_emp_raw": L_emp_raw[:, :, k].ravel(),
            "pdf_linear": pdf_linear[:, :, k].ravel(),
            "simple_linear": simple_linear[:, :, k].ravel(),
        }
        row = {
            "k": k + 1,
            "h": h_sequence[k],
            "mean_Epsi_prime": np.nanmean(Epsi_prime[:, k]),
            "mean_N_pop_t": np.nanmean(N_pop_t[:, k]),
            "mean_N_t": np.nanmean(N_t[:, k]),
        }
        for name, approx in approximations.items():
            finite = np.isfinite(actual) & np.isfinite(approx)
            if np.count_nonzero(finite) >= 2:
                row[f"cor_{name}"] = np.corrcoef(actual[finite], approx[finite])[0, 1]
                row[f"rmse_{name}"] = math.sqrt(np.nanmean((actual[finite] - approx[finite]) ** 2))
            else:
                row[f"cor_{name}"] = np.nan
                row[f"rmse_{name}"] = np.nan
        summary.append(row)

    summary_dtype = [(key, float) for key in summary[0].keys()]
    summary_array = np.array([tuple(row[key] for key in row.keys()) for row in summary], dtype=summary_dtype)

    return {
        "tilde_S": tilde_S,
        "S_empirical": S_empirical,
        "epsilon": Y_step[:, :, 1:],
        "Y_final": Y_step[:, :, -1],
        "h_sequence": h_sequence,
        "Y_step": Y_step,
        "linear": {
            "tilde_error": tilde_error,
            "epsilon": epsilon_step,
            "psi_epsilon": psi_epsilon,
            "psi_prime_epsilon": psi_prime_epsilon,
            "Epsi_prime": Epsi_prime,
            "N_pop_t": N_pop_t,
            "N_t": N_t,
            "L_pop": L_pop,
            "L_emp": L_emp,
            "L_pop_raw": L_pop_raw,
            "L_emp_raw": L_emp_raw,
            "pdf_linear": pdf_linear,
            "simple_linear": simple_linear,
            "X_step": X_step,
            "summary": summary_array,
        },
    }


def generate_zero_contaminated_signal(
    n=None,
    x_grid=None,
    true_signal=None,
    signal_fun=None,
    case=1,
    type=2,
    prop=0.2,
    sd_o=4,
    sd_c=0.1,
    n_paths=1,
    seed=None,
):
    """Generate the contaminated signal used by the R script."""
    rng = np.random.default_rng(seed)

    if true_signal is None:
        if x_grid is None:
            if n is None:
                raise ValueError("Please provide at least one of n, x_grid, or true_signal.")
            x_grid = np.arange(n + 1, dtype=float) / n
        else:
            x_grid = np.asarray(x_grid, dtype=float)

        if signal_fun is not None:
            signal_function = signal_fun
        elif case == 1:
            signal_function = lambda x: 1.2 * np.sin(2.0 * np.pi * x) + 0.5 * np.sin(6.0 * np.pi * x)
        elif case == 2:
            signal_function = lambda x: 2.0 * np.sin(12.0 * np.pi * x)
        else:
            raise ValueError("case must be 1 or 2.")

        true_signal = np.asarray(signal_function(x_grid), dtype=float)
    else:
        true_signal = np.asarray(true_signal, dtype=float)
        if x_grid is None:
            if n is None:
                n = len(true_signal) - 1
            x_grid = np.arange(n + 1, dtype=float) / n
        else:
            x_grid = np.asarray(x_grid, dtype=float)
            if len(x_grid) != len(true_signal):
                raise ValueError("Length of x_grid must match length of true_signal.")

    n_time = len(true_signal)

    if type == 1:
        if n_paths == 1:
            is_cont = rng.random(n_time) < prop
            Y = true_signal.copy()
            Y[is_cont] = 0.0
            contamination_index = np.where(is_cont)[0]
        else:
            Y = np.tile(true_signal[:, np.newaxis], (1, n_paths))
            contamination_index = {}
            for j in range(n_paths):
                is_cont = rng.random(n_time) < prop
                Y[is_cont, j] = 0.0
                contamination_index[f"path_{j + 1}"] = np.where(is_cont)[0]
    elif type == 2:
        if n_paths == 1:
            is_cont = rng.random(n_time) < prop
            eps = rng.normal(0.0, np.where(is_cont, sd_o, 1.0), size=n_time)
            Y = true_signal + sd_c * eps
            contamination_index = np.where(is_cont)[0]
        else:
            is_cont = rng.random((n_time, n_paths)) < prop
            eps = rng.normal(0.0, np.where(is_cont, sd_o, 1.0), size=(n_time, n_paths))
            Y = true_signal[:, np.newaxis] + sd_c * eps
            contamination_index = {
                f"path_{j + 1}": np.where(is_cont[:, j])[0] for j in range(n_paths)
            }
    else:
        raise ValueError("type must be 1 for zero contamination or 2 for mixture normal error.")

    return {
        "x_grid": x_grid,
        "true_signal": true_signal,
        "Y": Y,
        "contamination_index": contamination_index,
        "case": case,
        "type": type,
        "prop": prop,
        "sd_o": sd_o,
        "sd_c": sd_c,
    }


def normal_quantiles(n):
    """Approximate R qqnorm theoretical quantiles without SciPy."""
    p = (np.arange(1, n + 1) - 0.5) / n
    return np.array([math.sqrt(2.0) * inverse_erf(2.0 * q - 1.0) for q in p])


def inverse_erf(x):
    """Accurate enough inverse erf for QQ plots; uses Winitzki approximation plus Newton steps."""
    x = min(max(float(x), -0.999999999999), 0.999999999999)
    a = 0.147
    sign = -1.0 if x < 0 else 1.0
    ln_term = math.log(1.0 - x * x)
    first = 2.0 / (math.pi * a) + ln_term / 2.0
    y = sign * math.sqrt(math.sqrt(first * first - ln_term / a) - first)
    for _ in range(3):
        err = math.erf(y) - x
        y -= err / ((2.0 / math.sqrt(math.pi)) * math.exp(-y * y))
    return y


def format_param(x):
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x).replace(".", "p")


def make_output_dir(case, type, tc=None, H=1, n=None, m=None):
    parts = [
        f"c{format_param(case)}",
        f"t{format_param(type)}",
    ]
    if tc is not None:
        parts.append(f"tc{format_param(tc)}")
    if n is not None:
        parts.append(f"n{format_param(n)}")
    if m is not None:
        parts.append(f"m{format_param(m)}")
    parts.append(f"H{format_param(H)}")
    return "_".join(parts)


def compute_cumulative(dat, res):
    cumulative_signal = np.cumsum(res["tilde_S"], axis=2)
    cumulative_error = cumulative_signal - dat["true_signal"][:, np.newaxis, np.newaxis]
    return {
        "cumulative_signal": cumulative_signal,
        "cumulative_error": cumulative_error,
    }


def finite_range(values, default=(0.0, 1.0)):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return default
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = abs(lo) * 0.1 + 1.0
        return lo - pad, hi + pad
    return lo, hi


def draw_pil_boxplot(filename, data, ylim, title, width=500, height=500):
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    left, right, top, bottom = 55, width - 18, 42, height - 42
    blue = (0, 114, 178, 255)
    fill = (86, 180, 233, 190)
    black = (30, 30, 30, 255)
    gray = (150, 150, 150, 255)

    def x_map(index):
        return left + (index + 0.5) * (right - left) / data.shape[1]

    def y_map(value):
        lo, hi = ylim
        return bottom - (float(value) - lo) / (hi - lo) * (bottom - top)

    draw.text((left, 16), title, fill=black, font=font)
    draw.line((left, bottom, right, bottom), fill=black, width=1)
    draw.line((left, top, left, bottom), fill=black, width=1)
    for tick in np.linspace(ylim[0], ylim[1], 5):
        y = y_map(tick)
        draw.line((left - 4, y, left, y), fill=black, width=1)
        draw.text((6, y - 5), f"{tick:g}", fill=black, font=font)
    for k in range(data.shape[1]):
        x = x_map(k)
        vals = np.asarray(data[:, k], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        low, high = np.min(vals), np.max(vals)
        box_w = max(12, (right - left) / data.shape[1] * 0.42)
        y_q1, y_med, y_q3 = y_map(q1), y_map(med), y_map(q3)
        y_low, y_high = y_map(low), y_map(high)
        draw.line((x, y_high, x, y_q3), fill=blue, width=1)
        draw.line((x, y_q1, x, y_low), fill=blue, width=1)
        draw.line((x - box_w / 3, y_high, x + box_w / 3, y_high), fill=blue, width=1)
        draw.line((x - box_w / 3, y_low, x + box_w / 3, y_low), fill=blue, width=1)
        draw.rectangle((x - box_w / 2, y_q3, x + box_w / 2, y_q1), fill=fill, outline=blue, width=1)
        draw.line((x - box_w / 2, y_med, x + box_w / 2, y_med), fill=blue, width=2)
        draw.text((x - 3, bottom + 10), str(k + 1), fill=black, font=font)
    draw.rectangle((left, top, right, bottom), outline=gray, width=1)
    img.save(filename)


def draw_pil_qq_grid(filename, tilde_error_all, x_grid, h_sequence):
    K_steps = tilde_error_all.shape[2]
    cols = min(4, K_steps)
    rows = int(math.ceil(K_steps / cols))
    width, height = 150 * cols, 200 * rows
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    blue = (0, 114, 178, 255)
    orange = (213, 94, 0, 255)
    black = (30, 30, 30, 255)
    gray = (150, 150, 150, 255)
    target_index = int(np.argmin(np.abs(x_grid - 0.4)))
    panel_w, panel_h = width / cols, height / rows

    for k in range(K_steps):
        col = k % cols
        row = k // cols
        px = int(col * panel_w)
        py = int(row * panel_h)
        left, right = px + 28, int((col + 1) * panel_w) - 12
        top, bottom = py + 24, int((row + 1) * panel_h) - 20

        def map_xy(x, y):
            xx = left + (float(x) + 3.0) / 6.0 * (right - left)
            yy = bottom - (float(y) + 3.0) / 6.0 * (bottom - top)
            return xx, yy

        draw.rectangle((left, top, right, bottom), outline=gray, width=1)
        draw.text((left, py + 7), f"h_{k + 1}={h_sequence[k]:.3f}", fill=black, font=font)
        for tick in (-3, 0, 3):
            x0, y0 = map_xy(tick, tick)
            draw.line((x0, bottom, x0, bottom + 3), fill=black, width=1)
            draw.line((left - 3, y0, left, y0), fill=black, width=1)

        z = tilde_error_all[target_index, :, k]
        z_sd = np.nanstd(z, ddof=1)
        z_std = z - np.nanmean(z) if (not np.isfinite(z_sd) or z_sd == 0) else (z - np.nanmean(z)) / z_sd
        sample = np.sort(z_std[np.isfinite(z_std)])
        theory = normal_quantiles(len(sample))
        for tx, sy in zip(theory, sample):
            x, y = map_xy(tx, sy)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=blue)
        if len(sample) >= 2:
            q1_sample, q3_sample = np.percentile(sample, [25, 75])
            q1_theory, q3_theory = np.percentile(theory, [25, 75])
            slope = (q3_sample - q1_sample) / (q3_theory - q1_theory)
            intercept = q1_sample - slope * q1_theory
            x1, y1 = map_xy(-3.0, intercept - 3.0 * slope)
            x2, y2 = map_xy(3.0, intercept + 3.0 * slope)
            draw.line((x1, y1, x2, y2), fill=orange, width=2)
    img.save(filename)


def compute_plot_metrics(tilde_error_all, cumulative_signal_error_all, h_sequence):
    K_steps = tilde_error_all.shape[2]
    n_paths = tilde_error_all.shape[1]
    n_time = tilde_error_all.shape[0]
    temp_sup = np.empty((n_paths, K_steps))
    temp_imse = np.empty((n_paths, K_steps))
    temp_cum_sup = np.empty((n_paths, K_steps))
    temp_cum_imse = np.empty((n_paths, K_steps))

    for k in range(K_steps):
        temp_sup[:, k] = np.nanmax(np.abs(tilde_error_all[:, :, k]), axis=0)
        temp_imse[:, k] = np.nanmean(tilde_error_all[:, :, k] ** 2, axis=0) * (n_time * h_sequence[k])
        temp_cum_sup[:, k] = np.nanmax(np.abs(cumulative_signal_error_all[:, :, k]), axis=0)
        temp_cum_imse[:, k] = np.nanmean(cumulative_signal_error_all[:, :, k] ** 2, axis=0) * (n_time * h_sequence[k])

    return {
        "estimated_error_sup": temp_sup,
        "estimated_error_imse": temp_imse,
        "cumulative_error_sup": temp_cum_sup,
        "cumulative_error_imse": temp_cum_imse,
    }


def compute_plot_ylims(metrics, pad_fraction=0.05):
    ylims = {}
    for key, data in metrics.items():
        finite = np.asarray(data, dtype=float)
        finite = finite[np.isfinite(finite)]
        upper = float(np.max(finite)) if len(finite) else 1.0
        if upper <= 0:
            upper = 1.0
        ylims[key] = (0.0, upper * (1.0 + pad_fraction))
    return ylims


def save_outputs_pil(output_dir, tilde_error_all, cumulative_signal_error_all, x_grid, h_sequence, plot_ylims):
    metrics = compute_plot_metrics(tilde_error_all, cumulative_signal_error_all, h_sequence)

    draw_pil_boxplot(output_dir / "03_estimated_error_sup.png", metrics["estimated_error_sup"], plot_ylims["estimated_error_sup"], "sup |tilde_S - S|")
    draw_pil_boxplot(output_dir / "03_estimated_error_imse.png", metrics["estimated_error_imse"], plot_ylims["estimated_error_imse"], "n h_k mean((tilde_S - S)^2)")
    draw_pil_boxplot(output_dir / "04_cumulative_error_sup.png", metrics["cumulative_error_sup"], plot_ylims["cumulative_error_sup"], "sup |sum tilde_S - X|")
    draw_pil_boxplot(output_dir / "04_cumulative_error_imse.png", metrics["cumulative_error_imse"], plot_ylims["cumulative_error_imse"], "n h_k mean((sum tilde_S - X)^2)")
    draw_pil_qq_grid(output_dir / "smooth_u_04all.png", tilde_error_all, x_grid, h_sequence)


def save_numeric_results(dat, res, cumulative, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tilde_error_all = res["linear"]["tilde_error"]
    cumulative_signal_error_all = cumulative["cumulative_error"]

    np.savez_compressed(
        output_dir / filename,
        case=np.array(dat["case"]),
        type=np.array(dat["type"]),
        prop=np.array(dat["prop"]),
        sd_o=np.array(dat["sd_o"]),
        sd_c=np.array(dat["sd_c"]),
        x_grid=dat["x_grid"],
        true_signal=dat["true_signal"],
        Y=dat["Y"],
        tilde_S=res["tilde_S"],
        S_empirical=res["S_empirical"],
        Y_step=res["Y_step"],
        epsilon=res["epsilon"],
        Y_final=res["Y_final"],
        h_sequence=res["h_sequence"],
        tilde_error=tilde_error_all,
        linear_epsilon=res["linear"]["epsilon"],
        psi_epsilon=res["linear"]["psi_epsilon"],
        psi_prime_epsilon=res["linear"]["psi_prime_epsilon"],
        Epsi_prime=res["linear"]["Epsi_prime"],
        N_pop_t=res["linear"]["N_pop_t"],
        N_t=res["linear"]["N_t"],
        L_pop=res["linear"]["L_pop"],
        L_emp=res["linear"]["L_emp"],
        L_pop_raw=res["linear"]["L_pop_raw"],
        L_emp_raw=res["linear"]["L_emp_raw"],
        pdf_linear=res["linear"]["pdf_linear"],
        simple_linear=res["linear"]["simple_linear"],
        X_step=res["linear"]["X_step"],
        summary=res["linear"]["summary"],
        cumulative_signal=cumulative["cumulative_signal"],
        cumulative_error=cumulative_signal_error_all,
    )


def draw_pil_boxplot_panel(draw, data, bounds, ylim, title, font):
    left, top, right, bottom = bounds
    blue = (0, 114, 178, 255)
    fill = (86, 180, 233, 190)
    black = (30, 30, 30, 255)
    gray = (150, 150, 150, 255)

    def x_map(index):
        return left + (index + 0.5) * (right - left) / data.shape[1]

    def y_map(value):
        lo, hi = ylim
        return bottom - (float(value) - lo) / (hi - lo) * (bottom - top)

    draw.text((left, top - 28), title, fill=black, font=font)
    draw.line((left, bottom, right, bottom), fill=black, width=1)
    draw.line((left, top, left, bottom), fill=black, width=1)
    for tick in np.linspace(ylim[0], ylim[1], 5):
        y = y_map(tick)
        draw.line((left - 4, y, left, y), fill=black, width=1)
        draw.text((left - 48, y - 5), f"{tick:g}", fill=black, font=font)
    for k in range(data.shape[1]):
        x = x_map(k)
        vals = np.asarray(data[:, k], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        low, high = np.min(vals), np.max(vals)
        box_w = max(12, (right - left) / data.shape[1] * 0.42)
        y_q1, y_med, y_q3 = y_map(q1), y_map(med), y_map(q3)
        y_low, y_high = y_map(low), y_map(high)
        draw.line((x, y_high, x, y_q3), fill=blue, width=1)
        draw.line((x, y_q1, x, y_low), fill=blue, width=1)
        draw.line((x - box_w / 3, y_high, x + box_w / 3, y_high), fill=blue, width=1)
        draw.line((x - box_w / 3, y_low, x + box_w / 3, y_low), fill=blue, width=1)
        draw.rectangle((x - box_w / 2, y_q3, x + box_w / 2, y_q1), fill=fill, outline=blue, width=1)
        draw.line((x - box_w / 2, y_med, x + box_w / 2, y_med), fill=blue, width=2)
        draw.text((x - 3, bottom + 10), str(k + 1), fill=black, font=font)
    draw.rectangle((left, top, right, bottom), outline=gray, width=1)


def draw_pil_boxplot_pair(filename, tc1_data, tc2_data, ylim, title):
    width, height = 1000, 500
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw_pil_boxplot_panel(draw, tc1_data, (60, 58, 480, 445), ylim, title, font)
    draw_pil_boxplot_panel(draw, tc2_data, (560, 58, 980, 445), ylim, title, font)
    img.save(filename)


def save_combined_pair_plots_pil(output_dir, metrics_by_tc, experiments, plot_ylims):
    specs = [
        ("03_estimated_error_sup.png", "estimated_error_sup", "sup |tilde_S - S|"),
        ("03_estimated_error_imse.png", "estimated_error_imse", "n h_k mean((tilde_S - S)^2)"),
        ("04_cumulative_error_sup.png", "cumulative_error_sup", "sup |sum tilde_S - X|"),
        ("04_cumulative_error_imse.png", "cumulative_error_imse", "n h_k mean((sum tilde_S - X)^2)"),
    ]
    for filename, key, title in specs:
        draw_pil_boxplot_pair(
            output_dir / filename,
            metrics_by_tc[1][key],
            metrics_by_tc[2][key],
            plot_ylims[key],
            title,
        )

    for experiment in experiments:
        tc_value = experiment["tc_value"]
        draw_pil_qq_grid(
            output_dir / f"smooth_u_04all_tc{tc_value}.png",
            experiment["res"]["linear"]["tilde_error"],
            experiment["dat"]["x_grid"],
            experiment["res"]["h_sequence"],
        )


def save_combined_pair_plots_matplotlib(output_dir, metrics_by_tc, experiments, plot_ylims):
    specs = [
        ("03_estimated_error_sup.png", "estimated_error_sup", r"$\sup_t |\tilde{S}_t^{(k)} - S_t^{(k)}|$"),
        ("03_estimated_error_imse.png", "estimated_error_imse", r"$n h_k \int (\tilde{S}_t^{(k)} - S_t^{(k)})^2 dt$"),
        ("04_cumulative_error_sup.png", "cumulative_error_sup", r"$\sup_t |\sum_{l=1}^{k}\tilde{S}_t^{(l)} - X_t|$"),
        ("04_cumulative_error_imse.png", "cumulative_error_imse", r"$n h_k \int (\sum_{l=1}^{k}\tilde{S}_t^{(l)} - X_t)^2 dt$"),
    ]
    for filename, key, title in specs:
        fig, axes = plt.subplots(1, 2, figsize=(1000 / 180, 500 / 180), sharey=True)
        for ax, tc_value in zip(axes, (1, 2)):
            data = metrics_by_tc[tc_value][key]
            bp = ax.boxplot(
                [data[:, k] for k in range(data.shape[1])],
                tick_labels=[str(k + 1) for k in range(data.shape[1])],
                patch_artist=True,
            )
            for box in bp["boxes"]:
                box.set(facecolor="#56B4E9", edgecolor="#0072B2")
            for artist_key in ("whiskers", "caps", "medians"):
                for artist in bp[artist_key]:
                    artist.set(color="#0072B2")
            ax.set_ylim(*plot_ylims[key])
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180, transparent=True)
        plt.close(fig)

    for experiment in experiments:
        save_outputs(
            experiment["dat"],
            experiment["res"],
            experiment["cumulative"],
            output_dir,
            plot_ylims=plot_ylims,
            qq_filename=f"smooth_u_04all_tc{experiment['tc_value']}.png",
            boxplots=False,
            npz_filename=None,
        )


def save_outputs(
    dat,
    res,
    cumulative,
    output_dir=".",
    plot_ylims=None,
    qq_filename="smooth_u_04all.png",
    boxplots=True,
    npz_filename="imd_results.npz",
):
    """Save numeric results and the five plots defined in newplot11.R."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tilde_error_all = res["linear"]["tilde_error"]
    cumulative_signal_error_all = cumulative["cumulative_error"]
    h_sequence = res["h_sequence"]
    if plot_ylims is None:
        plot_ylims = compute_plot_ylims(compute_plot_metrics(tilde_error_all, cumulative_signal_error_all, h_sequence))

    if npz_filename is not None:
        save_numeric_results(dat, res, cumulative, output_dir, npz_filename)

    if plt is None:
        if Image is None:
            print("matplotlib and PIL are not installed; saved numeric results to imd_results.npz only.")
            return
        if boxplots:
            save_outputs_pil(output_dir, tilde_error_all, cumulative_signal_error_all, dat["x_grid"], h_sequence, plot_ylims)
        else:
            draw_pil_qq_grid(output_dir / qq_filename, tilde_error_all, dat["x_grid"], h_sequence)
        print("matplotlib is not installed; used PIL fallback for the five PNG plots.")
        return

    x_grid = dat["x_grid"]
    K_steps = tilde_error_all.shape[2]
    metrics = compute_plot_metrics(tilde_error_all, cumulative_signal_error_all, h_sequence)

    if boxplots:
        boxplot_specs = [
            ("03_estimated_error_sup.png", metrics["estimated_error_sup"], plot_ylims["estimated_error_sup"], r"$\sup_t |\tilde{S}_t^{(k)} - S_t^{(k)}|$"),
            ("03_estimated_error_imse.png", metrics["estimated_error_imse"], plot_ylims["estimated_error_imse"], r"$n h_k \int (\tilde{S}_t^{(k)} - S_t^{(k)})^2 dt$"),
            ("04_cumulative_error_sup.png", metrics["cumulative_error_sup"], plot_ylims["cumulative_error_sup"], r"$\sup_t |\sum_{l=1}^{k}\tilde{S}_t^{(l)} - X_t|$"),
            (
                "04_cumulative_error_imse.png",
                metrics["cumulative_error_imse"],
                plot_ylims["cumulative_error_imse"],
                r"$n h_k \int (\sum_{l=1}^{k}\tilde{S}_t^{(l)} - X_t)^2 dt$",
            ),
        ]

        for filename, data, ylim, title in boxplot_specs:
            plt.figure(figsize=(500 / 180, 500 / 180))
            bp = plt.boxplot(
                [data[:, k] for k in range(K_steps)],
                tick_labels=[str(k + 1) for k in range(K_steps)],
                patch_artist=True,
            )
            for box in bp["boxes"]:
                box.set(facecolor="#56B4E9", edgecolor="#0072B2")
            for key in ("whiskers", "caps", "medians"):
                for artist in bp[key]:
                    artist.set(color="#0072B2")
            plt.ylim(*ylim)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=180, transparent=True)
            plt.close()

    target_index = int(np.argmin(np.abs(x_grid - 0.4)))
    qq_cols = min(4, K_steps)
    qq_rows = int(math.ceil(K_steps / qq_cols))
    fig, axes = plt.subplots(qq_rows, qq_cols, figsize=(150 * qq_cols / 180, 200 * qq_rows / 180))
    axes = np.asarray(axes).ravel()
    for k, ax in enumerate(axes):
        if k >= K_steps:
            ax.axis("off")
            continue
        z = tilde_error_all[target_index, :, k]
        z_sd = np.nanstd(z, ddof=1)
        if not np.isfinite(z_sd) or z_sd == 0:
            z_std = z - np.nanmean(z)
        else:
            z_std = (z - np.nanmean(z)) / z_sd
        z_std = z_std[np.isfinite(z_std)]
        sample = np.sort(z_std)
        theory = normal_quantiles(len(sample))

        ax.scatter(theory, sample, s=10, color="#0072B2")
        if len(sample) >= 2:
            q1_sample, q3_sample = np.percentile(sample, [25, 75])
            q1_theory, q3_theory = np.percentile(theory, [25, 75])
            slope = (q3_sample - q1_sample) / (q3_theory - q1_theory)
            intercept = q1_sample - slope * q1_theory
            xs = np.array([-3.0, 3.0])
            ax.plot(xs, intercept + slope * xs, color="#D55E00", linewidth=1.2)
        ax.set_title(f"h_{k + 1}={h_sequence[k]:.3f}", fontsize=7)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6, pad=1)
        ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.88, wspace=0.35, hspace=0.45)
    fig.savefig(output_dir / qq_filename, dpi=180, transparent=True)
    plt.close(fig)


def resolve_contamination_parameters(type_value, tc_value, prop_value=None, sd_o_value=None):
    if prop_value is None:
        prop_value = 0.2
    if sd_o_value is None:
        sd_o_value = 4

    if type_value == 1 and tc_value == 2 and prop_value == 0.2:
        prop_value = 0.4
    if type_value == 2 and tc_value == 2 and sd_o_value == 4:
        sd_o_value = 7
    return prop_value, sd_o_value


def build_experiment(
    case_value=1,
    type_value=1,
    tc_value=1,
    n_value=200,
    m_value=5,
    seed_value=2026,
    K_steps_value=8,
    h_1_value=0.2,
    H_value=1.0,
    sigma_value=0.1,
    prop_value=None,
    sd_o_value=None,
    output_dir=None,
):
    prop_value, sd_o_value = resolve_contamination_parameters(type_value, tc_value, prop_value, sd_o_value)

    dat = generate_zero_contaminated_signal(
        n=n_value,
        n_paths=m_value,
        seed=seed_value,
        type=type_value,
        case=case_value,
        prop=prop_value,
        sd_o=sd_o_value,
        sd_c=sigma_value,
    )
    if output_dir is None:
        output_dir = make_output_dir(
            case=case_value,
            type=type_value,
            tc=tc_value,
            n=n_value,
            m=m_value,
            H=H_value,
        )
    res = imd_decompose_with_linearization(
        dat["Y"],
        x_grid=dat["x_grid"],
        true_signal=dat["true_signal"],
        K_steps=K_steps_value,
        h_1=h_1_value,
        H=H_value,
        epsi_prime_tol=1e-6,
    )
    cumulative = compute_cumulative(dat, res)
    return {
        "dat": dat,
        "res": res,
        "cumulative": cumulative,
        "output_dir": Path(output_dir),
        "prop_value": prop_value,
        "sd_o_value": sd_o_value,
        "case_value": case_value,
        "type_value": type_value,
        "tc_value": tc_value,
        "n_value": n_value,
        "m_value": m_value,
    }


def print_finished_message(experiment):
    print(
        "Finished. "
        f"c={experiment['case_value']}, t={experiment['type_value']}, "
        f"tc={experiment['tc_value']}, n={experiment['n_value']}, m={experiment['m_value']}, "
        f"prop={experiment['prop_value']}, sd_o={experiment['sd_o_value']}. "
        f"Results saved in {experiment['output_dir']}."
    )


def run_experiment(**kwargs):
    experiment = build_experiment(**kwargs)
    save_outputs(
        experiment["dat"],
        experiment["res"],
        experiment["cumulative"],
        experiment["output_dir"],
    )
    print_finished_message(experiment)
    return experiment["output_dir"]


def run_tc_pair(
    case_value=1,
    type_value=1,
    n_value=200,
    m_value=5,
    seed_value=2026,
    K_steps_value=8,
    h_1_value=0.2,
    H_value=1.0,
    sigma_value=0.1,
    prop_value=None,
    sd_o_value=None,
    output_dir=None,
):
    experiments = []
    if output_dir is None:
        output_dir = make_output_dir(
            case=case_value,
            type=type_value,
            tc=None,
            n=n_value,
            m=m_value,
            H=H_value,
        )
    output_dir = Path(output_dir)
    for tc_value in (1, 2):
        experiments.append(
            build_experiment(
                case_value=case_value,
                type_value=type_value,
                tc_value=tc_value,
                n_value=n_value,
                m_value=m_value,
                seed_value=seed_value,
                K_steps_value=K_steps_value,
                h_1_value=h_1_value,
                H_value=H_value,
                sigma_value=sigma_value,
                prop_value=prop_value,
                sd_o_value=sd_o_value,
                output_dir=output_dir,
            )
        )

    tc2_experiment = experiments[1]
    tc2_metrics = compute_plot_metrics(
        tc2_experiment["res"]["linear"]["tilde_error"],
        tc2_experiment["cumulative"]["cumulative_error"],
        tc2_experiment["res"]["h_sequence"],
    )
    shared_plot_ylims = compute_plot_ylims(tc2_metrics)
    metrics_by_tc = {}

    for experiment in experiments:
        tc_value = experiment["tc_value"]
        metrics_by_tc[tc_value] = compute_plot_metrics(
            experiment["res"]["linear"]["tilde_error"],
            experiment["cumulative"]["cumulative_error"],
            experiment["res"]["h_sequence"],
        )
        save_numeric_results(
            experiment["dat"],
            experiment["res"],
            experiment["cumulative"],
            output_dir,
            f"imd_results_tc{tc_value}.npz",
        )
        print_finished_message(experiment)

    if plt is None:
        if Image is None:
            print("matplotlib and PIL are not installed; saved numeric results only.")
        else:
            save_combined_pair_plots_pil(output_dir, metrics_by_tc, experiments, shared_plot_ylims)
            print("matplotlib is not installed; used PIL fallback for combined PNG plots.")
    else:
        save_combined_pair_plots_matplotlib(output_dir, metrics_by_tc, experiments, shared_plot_ylims)

    print("Shared y-axis limits for tc=1 and tc=2 were determined from tc=2 data.")
    print(f"Combined results saved in {output_dir}.")
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the IMD simulation and save five diagnostic plots."
    )
    parser.add_argument("--c", "--case", dest="case_value", type=int, default=2, choices=[1, 2])
    parser.add_argument("--t", "--type", dest="type_value", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tc", dest="tc_value", type=int, default=None, choices=[1, 2])
    parser.add_argument("--n", dest="n_value", type=int, default=2000)
    parser.add_argument("--m", dest="m_value", type=int, default=1000)
    parser.add_argument("--seed", dest="seed_value", type=int, default=2026)
    parser.add_argument("--K", dest="K_steps_value", type=int, default=8)
    parser.add_argument("--h1", dest="h_1_value", type=float, default=0.2)
    parser.add_argument("--H", dest="H_value", type=float, default=1.0)
    parser.add_argument("--sigma", dest="sigma_value", type=float, default=0.1)
    parser.add_argument("--prop", dest="prop_value", type=float, default=None)
    parser.add_argument("--sd-o", dest="sd_o_value", type=float, default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    args = parser.parse_args()
    if args.n_value < 2:
        parser.error("--n must be at least 2.")
    if args.m_value < 1:
        parser.error("--m must be at least 1.")
    if args.K_steps_value < 1:
        parser.error("--K must be at least 1.")
    return args


def main():
    args = parse_args()
    kwargs = vars(args)
    tc_value = kwargs.pop("tc_value")
    if tc_value is None:
        run_tc_pair(**kwargs)
    else:
        run_experiment(tc_value=tc_value, **kwargs)


if __name__ == "__main__":
    main()
