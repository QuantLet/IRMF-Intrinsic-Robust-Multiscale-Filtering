rho_H <- function(z, scale) {
  z * (2 * pnorm(z / scale) - 1) + 2 * scale * dnorm(z / scale)
}

psi_H <- function(z, scale) {
  2 * pnorm(z / scale) - 1
}

psi_prime_H <- function(z, scale) {
  2 * dnorm(z / scale) / scale
}

epanechnikov <- function(v) {
  0.75 * pmax(1 - v^2, 0) * (abs(v) <= 1)
}

periodic_extend <- function(values, grid) {
  ext_grid <- c(grid - 1, grid, grid + 1)
  ext_values <- rep(values, 3)
  keep <- ext_grid >= -0.5 & ext_grid <= 1.5
  list(grid = ext_grid[keep], values = ext_values[keep])
}

weighted_minimizer <- function(target_grid, obs_grid, objective_at_a,
                               bandwidth, lower, upper) {
  vapply(target_grid, function(t0) {
    weights <- epanechnikov((t0 - obs_grid) / bandwidth) / bandwidth
    active <- weights > 0
    objective_value <- function(a) {
      contributions <- objective_at_a(a, active)
      if (!any(is.finite(contributions))) return(Inf)
      sum(weights[active] * contributions, na.rm = TRUE)
    }
    optimize(objective_value, interval = c(lower, upper), tol = 1e-6)$minimum
  }, numeric(1))
}

generate_zero_contaminated_signal <- function(n = NULL,
                                              x_grid = NULL,
                                              true_signal = NULL,
                                              signal_fun = NULL,
                                              case = 1,
                                              type = 2,
                                              prop = 0.2,
                                              sd_o = 4,
                                              sd_c = 0.1,
                                              n_paths = 1,
                                              seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  if (is.null(true_signal)) {
    if (is.null(x_grid)) {
      if (is.null(n)) stop("Please provide at least one of n, x_grid, or true_signal.")
      x_grid <- (0:n) / n
    }

    if (!is.null(signal_fun)) {
      f <- signal_fun
    } else {
      f <- switch(as.character(case),
                  "1" = function(x) 1.2 * sin(2 * pi * x) + 0.5 * sin(6 * pi * x),
                  "2" = function(x) 2 * sin(12 * pi * x),
                  stop("case must be 1 or 2"))
    }
    true_signal <- f(x_grid)
  } else {
    if (is.null(x_grid)) {
      if (is.null(n)) n <- length(true_signal) - 1
      x_grid <- (0:n) / n
    } else if (length(x_grid) != length(true_signal)) {
      stop("Length of x_grid must match length of true_signal.")
    }
  }

  n_time <- length(true_signal)

  if (type == 1) {
    if (n_paths == 1) {
      is_cont <- runif(n_time) < prop
      Y <- true_signal
      Y[is_cont] <- 0
      contam_index <- which(is_cont)
    } else {
      Y <- matrix(true_signal, nrow = n_time, ncol = n_paths)
      contam_index <- vector("list", n_paths)
      for (j in seq_len(n_paths)) {
        is_cont <- runif(n_time) < prop
        Y[is_cont, j] <- 0
        contam_index[[j]] <- which(is_cont)
      }
      names(contam_index) <- paste0("path_", seq_len(n_paths))
    }
  } else if (type == 2) {
    if (n_paths == 1) {
      is_cont <- runif(n_time) < prop
      eps <- rnorm(n_time, mean = 0, sd = ifelse(is_cont, sd_o, 1))
      Y <- true_signal + sd_c * eps
      contam_index <- which(is_cont)
    } else {
      U <- matrix(runif(n_time * n_paths), nrow = n_time, ncol = n_paths)
      is_cont <- U < prop
      eps <- matrix(rnorm(n_time * n_paths, mean = 0,
                          sd = ifelse(is_cont, sd_o, 1)),
                    nrow = n_time, ncol = n_paths)
      Y <- true_signal + sd_c * eps
      contam_index <- lapply(seq_len(n_paths), function(j) which(is_cont[, j]))
      names(contam_index) <- paste0("path_", seq_len(n_paths))
    }
  } else {
    stop("type must be 1 or 2.")
  }

  list(
    x_grid = x_grid,
    true_signal = true_signal,
    Y = Y,
    contamination_index = contam_index,
    case = case,
    type = type,
    prop = prop,
    sd_o = sd_o,
    sd_c = sd_c
  )
}

imd_decompose_with_linearization <- function(Y, x_grid, true_signal,
                                             K_steps = 8,
                                             h_1 = 0.2,
                                             H = sqrt(0.1),
                                             epsi_prime_tol = 1e-6) {
  if (is.vector(Y)) Y <- as.matrix(Y, ncol = 1)
  stopifnot(is.matrix(Y))
  n_time <- nrow(Y)
  n_paths <- ncol(Y)
  stopifnot(length(x_grid) == n_time)
  stopifnot(length(true_signal) == n_time)

  h_sequence <- h_1 / sqrt(2)^(0:(K_steps - 1))

  tilde_S <- array(NA_real_, dim = c(n_time, n_paths, K_steps))
  Y_step <- array(NA_real_, dim = c(n_time, n_paths, K_steps + 1))
  Y_step[, , 1] <- Y
  S_empirical <- matrix(NA_real_, nrow = n_time, ncol = K_steps)

  epsilon_step <- array(NA_real_, dim = c(n_time, n_paths, K_steps))
  psi_epsilon <- array(NA_real_, dim = c(n_time, n_paths, K_steps))
  psi_prime_epsilon <- array(NA_real_, dim = c(n_time, n_paths, K_steps))
  Epsi_prime <- matrix(NA_real_, nrow = n_time, ncol = K_steps)
  N_pop_t <- matrix(NA_real_, nrow = n_time, ncol = K_steps)
  N_t <- matrix(NA_real_, nrow = n_time, ncol = K_steps)
  L_pop <- array(NA_real_, dim = c(n_time, n_paths, K_steps))
  L_emp <- array(NA_real_, dim = c(n_time, n_paths, K_steps))
  L_pop_raw <- array(NA_real_, dim = c(n_time, n_paths, K_steps))
  L_emp_raw <- array(NA_real_, dim = c(n_time, n_paths, K_steps))
  pdf_linear <- array(NA_real_, dim = c(n_time, n_paths, K_steps))
  simple_linear <- array(NA_real_, dim = c(n_time, n_paths, K_steps))
  X_step <- matrix(NA_real_, nrow = n_time, ncol = K_steps + 1)
  X_step[, 1] <- true_signal

  for (k in seq_len(K_steps)) {
    h_k <- h_sequence[k]
    Y_current <- Y_step[, , k]

    ext_list <- lapply(seq_len(n_paths), function(j) {
      periodic_extend(Y_current[, j], grid = x_grid)
    })
    ext_grid <- ext_list[[1]]$grid
    Y_current_ext <- vapply(ext_list, `[[`, numeric(length(ext_grid)), "values")

    for (j in seq_len(n_paths)) {
      tilde_S[, j, k] <- weighted_minimizer(
        target_grid = x_grid,
        obs_grid = ext_grid,
        objective_at_a = function(a, active) {
          rho_H(Y_current_ext[active, j] - a, H)
        },
        bandwidth = h_k,
        lower = min(Y_current[, j], na.rm = TRUE) - 4 * H,
        upper = max(Y_current[, j], na.rm = TRUE) + 4 * H
      )
    }

    S_empirical[, k] <- weighted_minimizer(
      target_grid = x_grid,
      obs_grid = ext_grid,
      objective_at_a = function(a, active) {
        rowMeans(rho_H(Y_current_ext[active, , drop = FALSE] - a, H), na.rm = TRUE)
      },
      bandwidth = h_k,
      lower = min(Y_current, na.rm = TRUE) - 4 * H,
      upper = max(Y_current, na.rm = TRUE) + 4 * H
    )

    X_step[, k + 1] <- X_step[, k] - S_empirical[, k]
    epsilon_step[, , k] <- sweep(Y_current, MARGIN = 1, STATS = X_step[, k], FUN = "-")
    psi_epsilon[, , k] <- psi_H(epsilon_step[, , k], H)
    psi_prime_epsilon[, , k] <- psi_prime_H(epsilon_step[, , k], H)
    Epsi_prime[, k] <- rowMeans(psi_prime_epsilon[, , k], na.rm = TRUE)

    eps_ext_list <- lapply(seq_len(n_paths), function(j) {
      periodic_extend(epsilon_step[, j, k], x_grid)
    })
    eps_ext <- vapply(eps_ext_list, `[[`, numeric(length(ext_grid)), "values")

    for (i in seq_len(n_time)) {
      weights <- epanechnikov((x_grid[i] - ext_grid) / h_k) / h_k
      active <- weights > 0
      active_weights <- weights[active]

      psi_eps_active <- psi_H(eps_ext[active, , drop = FALSE], H)
      psi_prime_eps_active <- psi_prime_H(eps_ext[active, , drop = FALSE], H)
      Epsi_by_u <- rowMeans(psi_eps_active, na.rm = TRUE)
      Epsi_prime_by_u <- rowMeans(psi_prime_eps_active, na.rm = TRUE)
      N_pop_t[i, k] <- sum(active_weights * Epsi_prime_by_u, na.rm = TRUE)

      arg_pdf <- Y_current_ext[active, , drop = FALSE] - S_empirical[i, k]
      psi_pdf <- psi_H(arg_pdf, H)
      psi_prime_pdf <- psi_prime_H(arg_pdf, H)
      E_psi_pdf_by_u <- rowMeans(psi_pdf, na.rm = TRUE)
      E_psi_prime_pdf_by_u <- rowMeans(psi_prime_pdf, na.rm = TRUE)

      N_t[i, k] <- sum(active_weights * E_psi_prime_pdf_by_u, na.rm = TRUE)
      simple_denom <- sum(active_weights, na.rm = TRUE) *
        mean(as.vector(psi_prime_eps_active), na.rm = TRUE)

      for (j in seq_len(n_paths)) {
        raw_eps_score <- psi_eps_active[, j]
        centered_eps_score <- psi_eps_active[, j] - Epsi_by_u
        N_pop_safe <- pmax(N_pop_t[i, k], epsi_prime_tol)
        N_emp_safe <- pmax(N_t[i, k], epsi_prime_tol)

        L_pop_raw[i, j, k] <- sum(active_weights * raw_eps_score, na.rm = TRUE) / N_pop_safe
        L_emp_raw[i, j, k] <- sum(active_weights * raw_eps_score, na.rm = TRUE) / N_emp_safe

        L_pop[i, j, k] <- sum(active_weights * centered_eps_score, na.rm = TRUE) / N_pop_safe
        L_emp[i, j, k] <- sum(active_weights * centered_eps_score, na.rm = TRUE) / N_emp_safe

        diamond <- psi_pdf[, j] - E_psi_pdf_by_u
        pdf_linear[i, j, k] <- sum(active_weights * diamond, na.rm = TRUE) / N_emp_safe
        simple_linear[i, j, k] <- sum(active_weights * centered_eps_score, na.rm = TRUE) /
          simple_denom
      }
    }

    Y_step[, , k + 1] <- Y_current - tilde_S[, , k]
  }

  tilde_error <- sweep(tilde_S, MARGIN = c(1, 3), STATS = S_empirical, FUN = "-")

  summary <- do.call(rbind, lapply(seq_len(K_steps), function(k) {
    actual <- as.vector(tilde_error[, , k])
    approx_pop <- as.vector(L_pop[, , k])
    approx_emp <- as.vector(L_emp[, , k])
    approx_pop_raw <- as.vector(L_pop_raw[, , k])
    approx_emp_raw <- as.vector(L_emp_raw[, , k])
    approx_pdf <- as.vector(pdf_linear[, , k])
    approx_simple <- as.vector(simple_linear[, , k])
    data.frame(
      k = k,
      h = h_sequence[k],
      cor_L_pop = cor(actual, approx_pop, use = "complete.obs"),
      rmse_L_pop = sqrt(mean((actual - approx_pop)^2, na.rm = TRUE)),
      cor_L_emp = cor(actual, approx_emp, use = "complete.obs"),
      rmse_L_emp = sqrt(mean((actual - approx_emp)^2, na.rm = TRUE)),
      cor_L_pop_raw = cor(actual, approx_pop_raw, use = "complete.obs"),
      rmse_L_pop_raw = sqrt(mean((actual - approx_pop_raw)^2, na.rm = TRUE)),
      cor_L_emp_raw = cor(actual, approx_emp_raw, use = "complete.obs"),
      rmse_L_emp_raw = sqrt(mean((actual - approx_emp_raw)^2, na.rm = TRUE)),
      cor_pdf_linear = cor(actual, approx_pdf, use = "complete.obs"),
      rmse_pdf_linear = sqrt(mean((actual - approx_pdf)^2, na.rm = TRUE)),
      cor_simple_linear = cor(actual, approx_simple, use = "complete.obs"),
      rmse_simple_linear = sqrt(mean((actual - approx_simple)^2, na.rm = TRUE)),
      mean_Epsi_prime = mean(Epsi_prime[, k], na.rm = TRUE),
      mean_N_pop_t = mean(N_pop_t[, k], na.rm = TRUE),
      mean_N_t = mean(N_t[, k], na.rm = TRUE)
    )
  }))

  list(
    tilde_S = tilde_S,
    S_empirical = S_empirical,
    epsilon = Y_step[, , -1, drop = FALSE],
    Y_final = Y_step[, , K_steps + 1],
    h_sequence = h_sequence,
    Y_step = Y_step,
    linear = list(
      tilde_error = tilde_error,
      epsilon = epsilon_step,
      psi_epsilon = psi_epsilon,
      psi_prime_epsilon = psi_prime_epsilon,
      Epsi_prime = Epsi_prime,
      N_pop_t = N_pop_t,
      N_t = N_t,
      L_pop = L_pop,
      L_emp = L_emp,
      L_pop_raw = L_pop_raw,
      L_emp_raw = L_emp_raw,
      pdf_linear = pdf_linear,
      simple_linear = simple_linear,
      X_step = X_step,
      summary = summary
    )
  )
}

format_param <- function(x) {
  gsub("\\.", "p", format(x, scientific = FALSE, trim = TRUE))
}

make_output_dir <- function(case, type, tc, H) {
  dirname <- paste0(
    "c", format_param(case),
    "_t", format_param(type),
    "_tc", format_param(tc),
    "_H", format_param(H)
  )
  dir.create(dirname, showWarnings = FALSE, recursive = TRUE)
  dirname
}

compute_cumulative <- function(dat, result) {
  K_steps <- dim(result$tilde_S)[3]
  cumulative_signal <- array(0, dim = dim(result$tilde_S))
  cumulative_signal[, , 1] <- result$tilde_S[, , 1]
  for (k in 2:K_steps) {
    cumulative_signal[, , k] <- cumulative_signal[, , k - 1] + result$tilde_S[, , k]
  }
  cumulative_error <- sweep(cumulative_signal, MARGIN = 1,
                            STATS = dat$true_signal, FUN = "-")
  list(cumulative_signal = cumulative_signal, cumulative_error = cumulative_error)
}

save_original_plots <- function(dat, result, cumulative, output_dir) {
  x_grid <- dat$x_grid
  true_signal <- dat$true_signal
  tilde_error_all <- result$linear$tilde_error
  cumulative_error <- cumulative$cumulative_error
  K_steps <- dim(tilde_error_all)[3]
  n_time <- dim(tilde_error_all)[1]
  n_paths <- dim(tilde_error_all)[2]
  h_sequence <- result$h_sequence
  bg <- "transparent"

  png(file.path(output_dir, "decomposition_overview.png"),
      width = 800, height = 600, res = 180, bg = bg)
  matplot(x_grid,
          cbind(true_signal,
                rowSums(result$S_empirical[, seq_len(K_steps), drop = FALSE]),
                rowSums(result$tilde_S[, 1, seq_len(K_steps), drop = FALSE])),
          type = "l", lty = 1, col = c("blue", "red", "black"))
  dev.off()

  target_index <- which.min(abs(x_grid - 0.4))
  for (k in seq_len(K_steps)) {
    z <- tilde_error_all[target_index, , k]
    z_std <- (z - mean(z, na.rm = TRUE)) / sd(z, na.rm = TRUE)

    png(file.path(output_dir, paste0(k, "_smooth_u_0.4.png")),
        width = 800, height = 800, res = 180, bg = bg)
    par(pty = "s")
    qqnorm(z_std,
           main = paste0("k=", k, ", h_k=", format(h_sequence[k], digits = 4)),
           xlab = "", ylab = "", pch = 19, col = "#0072B2",
           xlim = c(-3, 3), ylim = c(-3, 3))
    qqline(z_std, col = "#D55E00", lwd = 2)
    dev.off()
  }

  temp1 <- matrix(NA_real_, n_paths, K_steps)
  for (k in seq_len(K_steps)) temp1[, k] <- apply(abs(tilde_error_all[, , k]), 2, max)
  png(file.path(output_dir, "03_estimated_error_sup.png"),
      width = 800, height = 600, res = 180, bg = bg)
  boxplot(as.data.frame(temp1), names = seq_len(ncol(temp1)),
          col = "#56B4E9", border = "#0072B2",
          main = bquote(sup[t] ~ "|" ~ tilde(S)[t]^{(k)} - S[t]^{(k)} ~ "|"))
  dev.off()

  temp2 <- matrix(NA_real_, n_paths, K_steps)
  for (k in seq_len(K_steps)) {
    temp2[, k] <- apply((tilde_error_all[, , k])^2, 2, mean) * (n_time * h_sequence[k])
  }
  png(file.path(output_dir, "03_estimated_error_imse.png"),
      width = 800, height = 600, res = 180, bg = bg)
  boxplot(as.data.frame(temp2), names = seq_len(ncol(temp2)),
          col = "#56B4E9", border = "#0072B2",
          main = bquote(n ~ h[k] ~ integral((tilde(S)[t]^{(k)} - S[t]^{(k)})^2 ~ dt)))
  dev.off()

  temp1 <- matrix(NA_real_, n_paths, K_steps)
  for (k in seq_len(K_steps)) temp1[, k] <- apply(abs(cumulative_error[, , k]), 2, max)
  png(file.path(output_dir, "04_cumulative_error_sup.png"),
      width = 800, height = 600, res = 180, bg = bg)
  boxplot(as.data.frame(temp1), names = seq_len(ncol(temp1)),
          col = "#56B4E9", border = "#0072B2",
          main = bquote(sup[t] ~ "|" ~ sum(tilde(S)[t]^{(l)}, l == 1, k) - X[t] ~ "|"))
  dev.off()

  temp2 <- matrix(NA_real_, n_paths, K_steps)
  for (k in seq_len(K_steps)) {
    temp2[, k] <- apply((cumulative_error[, , k])^2, 2, mean) * (n_time * h_sequence[k])
  }
  png(file.path(output_dir, "04_cumulative_error_imse.png"),
      width = 800, height = 600, res = 180, bg = bg)
  boxplot(as.data.frame(temp2), names = seq_len(ncol(temp2)),
          col = "#56B4E9", border = "#0072B2",
          main = bquote(n ~ h[k] ~ integral((sum(tilde(S)[t]^{(l)}, l == 1, k) - X[t])^2 ~ dt)))
  dev.off()
}

save_linearization_path_plots <- function(dat, result, output_dir, path_id = 1) {
  K_steps <- dim(result$linear$tilde_error)[3]
  x_grid <- dat$x_grid

  for (k in seq_len(K_steps)) {
    y_plot <- c(result$linear$tilde_error[, path_id, k],
                result$linear$L_pop[, path_id, k],
                result$linear$L_emp[, path_id, k])
    ylim <- range(y_plot[is.finite(y_plot)])

    png(file.path(output_dir, paste0("linearization_path", path_id, "_k", k, ".png")),
        width = 900, height = 600, res = 180, bg = "transparent")
    matplot(
            x_grid,
            cbind(result$linear$tilde_error[, path_id, k],
                  result$linear$L_pop[, path_id, k],
                  result$linear$L_emp[, path_id, k]),
            type = "l", lty = 1, lwd = 2,
            col = c("black", "#0072B2", "#D55E00"),
            ylim = ylim,
            xlab = "t", ylab = "",
            main = paste0("Linearization check, path=", path_id, ", k=", k))
    legend("topright",
           legend = c("tilde_S - S_empirical", "L_pop demean", "L_emp demean"),
           col = c("black", "#0072B2", "#D55E00"),
           lty = 1, lwd = 2, bty = "n")
    dev.off()
  }
}

save_outputs <- function(dat, result, cumulative, output_dir, writep=0) {
if(writep==1) {
  write.csv(result$linear$summary,
            file = file.path(output_dir, "linearization_summary.csv"),
            row.names = FALSE)

  write.csv(data.frame(
    time = rep(dat$x_grid, times = ncol(dat$Y)),
    path = rep(seq_len(ncol(dat$Y)), each = length(dat$x_grid)),
    epsilon_k1 = as.vector(result$linear$epsilon[, , 1]),
    psi_epsilon_k1 = as.vector(result$linear$psi_epsilon[, , 1]),
    psi_prime_epsilon_k1 = as.vector(result$linear$psi_prime_epsilon[, , 1]),
    tilde_error_k1 = as.vector(result$linear$tilde_error[, , 1]),
    L_pop_k1 = as.vector(result$linear$L_pop[, , 1]),
    L_emp_k1 = as.vector(result$linear$L_emp[, , 1]),
    L_pop_raw_k1 = as.vector(result$linear$L_pop_raw[, , 1]),
    L_emp_raw_k1 = as.vector(result$linear$L_emp_raw[, , 1])
  ), file = file.path(output_dir, "pointwise_terms_k1.csv"), row.names = FALSE)
}

  results <- list(
    dat = dat,
    tilde_S = result$tilde_S,
    S_empirical = result$S_empirical,
    Y_step = result$Y_step,
    h_sequence = result$h_sequence,
    tilde_error = result$linear$tilde_error,
    epsilon = result$linear$epsilon,
    psi_epsilon = result$linear$psi_epsilon,
    psi_prime_epsilon = result$linear$psi_prime_epsilon,
    Epsi_prime = result$linear$Epsi_prime,
    N_pop_t = result$linear$N_pop_t,
    N_t = result$linear$N_t,
    L_pop = result$linear$L_pop,
    L_emp = result$linear$L_emp,
    L_pop_raw = result$linear$L_pop_raw,
    L_emp_raw = result$linear$L_emp_raw,
    pdf_linear = result$linear$pdf_linear,
    simple_linear = result$linear$simple_linear,
    cumulative_signal = cumulative$cumulative_signal,
    cumulative_error = cumulative$cumulative_error,
    summary = result$linear$summary
  )
  #saveRDS(results, file = file.path(output_dir, "combined_results_all.rds"))
}


### main values 
sigma_value <- 0.1

case_value <- 1
type_value <- 1
tc=1 
H_value <- 1            #sigma_value         #1 
prop_value <- 0.2
sd_o_value <- 4

if (type_value==1 & tc==2) {prop_value <- 0.4}
if (type_value==2 & tc==2) {sd_o_value <- 7}

n_value <- 2000
n_paths_value <- 500
seed_value <- 2026
K_steps_value <- 8
h_1_value <- 0.2

dat <- generate_zero_contaminated_signal(
  n = n_value,
  n_paths = n_paths_value,
  seed = seed_value,
  type = type_value,
  case = case_value,
  prop = prop_value,
  sd_o = sd_o_value,
  sd_c = sigma_value
)

output_dir <- make_output_dir(
  case = case_value,
  type = type_value,
  tc=tc,
  H = 1*(H_value==1)
)

result <- imd_decompose_with_linearization(
  Y = dat$Y,
  x_grid = dat$x_grid,
  true_signal = dat$true_signal,
  K_steps = K_steps_value,
  h_1 = h_1_value,
  H = H_value
  ,
  epsi_prime_tol = 1e-6
)

cumulative <- compute_cumulative(dat, result)
save_original_plots(dat, result, cumulative, output_dir)

#save_linearization_path_plots(dat, result, output_dir, path_id = 1)
save_outputs(dat, result, cumulative, output_dir)

#print(result$linear$summary)
#cat("\nSaved plots and linearization results in:", normalizePath(output_dir), "\n")
























