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


def imd_decompose_all(Y, x_grid=None, K_steps=8, h_1=0.2, H=math.sqrt(0.1)):
    """
    Iterative multiscale decomposition translated from mainfile4.R.

    Returns a dictionary with:
    tilde_S: shape (n_time, n_paths, K_steps)
    S_empirical: shape (n_time, K_steps)
    epsilon: shape (n_time, n_paths, K_steps)
    Y_final: shape (n_time, n_paths)
    h_sequence: shape (K_steps,)
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.ndim != 2:
        raise ValueError("Y must be a vector or a 2D matrix.")

    n_time, n_paths = Y.shape
    if x_grid is None:
        x_grid = np.linspace(0.0, 1.0, n_time)
    else:
        x_grid = np.asarray(x_grid, dtype=float)
        if len(x_grid) != n_time:
            raise ValueError("x_grid length must match the number of rows in Y.")

    h_sequence = h_1 / (math.sqrt(2.0) ** np.arange(K_steps))
    tilde_S_all = np.full((n_time, n_paths, K_steps), np.nan, dtype=float)
    Y_step = np.full((n_time, n_paths, K_steps + 1), np.nan, dtype=float)
    Y_step[:, :, 0] = Y
    S_empirical = np.full((n_time, K_steps), np.nan, dtype=float)

    for k in range(K_steps):
        h_k = h_sequence[k]
        Y_current = Y_step[:, :, k]

        ext_list = [periodic_extend(Y_current[:, j], x_grid) for j in range(n_paths)]
        ext_grid = ext_list[0][0]
        Y_current_ext = np.column_stack([values for _, values in ext_list])

        for j in range(n_paths):
            def objective_path(a, active, path_index=j):
                return rho_H(Y_current_ext[active, path_index] - a, H)

            tilde_S_all[:, j, k] = weighted_minimizer(
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

        Y_step[:, :, k + 1] = Y_current - tilde_S_all[:, :, k]

    return {
        "tilde_S": tilde_S_all,
        "S_empirical": S_empirical,
        "epsilon": Y_step[:, :, 1:],
        "Y_final": Y_step[:, :, -1],
        "h_sequence": h_sequence,
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


def save_outputs(dat, res, output_dir="."):
    """Create the same diagnostics as the R script when matplotlib is installed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tilde_error_all = res["tilde_S"] - res["S_empirical"][:, np.newaxis, :]
    cumulative_signal = np.cumsum(res["tilde_S"], axis=2)
    cumulative_signal_error_all = cumulative_signal - dat["true_signal"][:, np.newaxis, np.newaxis]

    np.savez_compressed(
        output_dir / "imd_results.npz",
        x_grid=dat["x_grid"],
        true_signal=dat["true_signal"],
        Y=dat["Y"],
        tilde_S=res["tilde_S"],
        S_empirical=res["S_empirical"],
        epsilon=res["epsilon"],
        Y_final=res["Y_final"],
        h_sequence=res["h_sequence"],
        tilde_error_all=tilde_error_all,
        cumulative_signal_error_all=cumulative_signal_error_all,
    )

    if plt is None:
        print("matplotlib is not installed; saved numeric results to imd_results.npz only.")
        return

    x_grid = dat["x_grid"]
    true_signal = dat["true_signal"]
    h_sequence = res["h_sequence"]
    K_steps = tilde_error_all.shape[2]
    n_paths = tilde_error_all.shape[1]

    sum_S_emp = np.sum(res["S_empirical"][:, :K_steps], axis=1)
    sum_tilde_S_path1 = np.sum(res["tilde_S"][:, 0, :K_steps], axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, true_signal, "b-", label="True signal")
    plt.plot(x_grid, sum_S_emp, "r-", label="Sum S_empirical")
    plt.plot(x_grid, sum_tilde_S_path1, "k-", label="Sum tilde_S, path 1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "decomposition_overview.png", dpi=180, transparent=True)
    plt.close()

    target_index = int(np.argmin(np.abs(x_grid - 0.4)))
    for k in range(K_steps):
        z = tilde_error_all[target_index, :, k]
        z_sd = np.std(z, ddof=1)
        if not np.isfinite(z_sd) or z_sd == 0:
            z_std = z - np.mean(z)
        else:
            z_std = (z - np.mean(z)) / z_sd
        sample = np.sort(z_std)
        theory = normal_quantiles(len(sample))

        plt.figure(figsize=(5, 5))
        plt.scatter(theory, sample, s=18, color="#0072B2")
        if len(sample) >= 2:
            q1_sample, q3_sample = np.percentile(sample, [25, 75])
            q1_theory, q3_theory = np.percentile(theory, [25, 75])
            slope = (q3_sample - q1_sample) / (q3_theory - q1_theory)
            intercept = q1_sample - slope * q1_theory
            xs = np.array([theory.min(), theory.max()])
            plt.plot(xs, intercept + slope * xs, color="#D55E00", linewidth=2)
        plt.title(f"k={k + 1}, h_k={h_sequence[k]:.4g}")
        plt.xlabel("")
        plt.ylabel("")
        ax = plt.gca()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.savefig(output_dir / f"{k + 1}_smooth_u_0.4.png", dpi=180, transparent=True)
        plt.close()

    temp_sup = np.empty((n_paths, K_steps))
    temp_imse = np.empty((n_paths, K_steps))
    temp_cum_sup = np.empty((n_paths, K_steps))
    temp_cum_imse = np.empty((n_paths, K_steps))
    n_time = tilde_error_all.shape[0]

    for k in range(K_steps):
        temp_sup[:, k] = np.nanmax(np.abs(tilde_error_all[:, :, k]), axis=0)
        temp_imse[:, k] = np.nanmean(tilde_error_all[:, :, k] ** 2, axis=0) * (n_time * h_sequence[k])
        temp_cum_sup[:, k] = np.nanmax(np.abs(cumulative_signal_error_all[:, :, k]), axis=0)
        temp_cum_imse[:, k] = (
            np.nanmean(cumulative_signal_error_all[:, :, k] ** 2, axis=0) * (n_time * h_sequence[k])
        )

    boxplot_specs = [
        ("03_estimated_error_sup.png", temp_sup, r"$\sup_t |\tilde{S}_t^{(k)} - S_t^{(k)}|$"),
        ("03_estimated_error_imse.png", temp_imse, r"$n h_k \int (\tilde{S}_t^{(k)} - S_t^{(k)})^2 dt$"),
        ("04_cumulative_error_sup.png", temp_cum_sup, r"$\sup_t |\sum_{l=1}^{k}\tilde{S}_t^{(l)} - X_t|$"),
        (
            "04_cumulative_error_imse.png",
            temp_cum_imse,
            r"$n h_k \int (\sum_{l=1}^{k}\tilde{S}_t^{(l)} - X_t)^2 dt$",
        ),
    ]

    for filename, data, title in boxplot_specs:
        plt.figure(figsize=(10, 6))
        plt.boxplot([data[:, k] for k in range(K_steps)], tick_labels=[str(k + 1) for k in range(K_steps)])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180, transparent=True)
        plt.close()


def main():
    dat = generate_zero_contaminated_signal(
        n=200,
        n_paths=5,
        seed=2026,
        type=1,
        case=1,
        prop=0.2,
    )
    res = imd_decompose_all(dat["Y"], x_grid=dat["x_grid"], K_steps=8, H=1.0)
    save_outputs(dat, res)
    print("Finished. Numeric results saved to imd_results.npz.")


if __name__ == "__main__":
    main()
