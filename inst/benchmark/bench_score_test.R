#!/usr/bin/env Rscript
# Score test performance benchmark: count and zero components
# Usage: Rscript inst/benchmark/bench_score_test.R

library(fasthurdle)
library(bench)

# ---- Data generator ----
simulate_hurdle <- function(n, kx, theta = 2, zero_prob = 0.5,
                            beta_test = 0, offset_sd = 0.5) {
  X <- cbind(1, matrix(rnorm(n * (kx - 1)), n, kx - 1))
  colnames(X) <- c("(Intercept)", paste0("x", seq_len(kx - 2)), "x_test")
  beta <- c(-0.5, rep(0, kx - 2), beta_test)
  off <- rnorm(n, sd = offset_sd)
  mu <- exp(X %*% beta + off)
  y <- rnbinom(n, size = theta, mu = as.numeric(mu))
  zero_mask <- rbinom(n, 1, zero_prob)
  y[zero_mask == 1] <- 0L
  list(X = X, y = y, offset = off)
}

# ---- Benchmark function ----
run_benchmarks <- function(n_vals = c(1e4, 1e5, 5e5, 1e6),
                           kx_vals = c(3, 6, 20),
                           n_reps = 5) {
  results <- list()

  for (n in n_vals) {
    for (kx in kx_vals) {
      for (scenario in c("null", "alt")) {
        beta_test <- if (scenario == "null") 0 else 0.3
        set.seed(42)
        dat <- simulate_hurdle(n, kx, beta_test = beta_test)

        X_null <- dat$X[, -kx, drop = FALSE]
        x_test <- dat$X[, kx, drop = TRUE]

        # Pre-fit null models (excluded from timing)
        null_count <- fit_null_count(X_null, dat$y,
          offsetx = dat$offset, dist = "negbin"
        )
        null_zero <- fit_null_zero(X_null, dat$y)

        label <- sprintf(
          "n=%s kx=%d %s",
          format(n, big.mark = ","), kx, scenario
        )
        cat(sprintf("Running: %s ...\n", label))

        bm <- bench::mark(
          count_spa = score_test_count(
            X_null, x_test, dat$y,
            offsetx = dat$offset,
            dist = "negbin",
            null_fit_count = null_count,
            spa_cutoff = 2
          ),
          count_nospa = score_test_count(
            X_null, x_test, dat$y,
            offsetx = dat$offset,
            dist = "negbin",
            null_fit_count = null_count,
            spa_cutoff = Inf
          ),
          zero_spa = score_test_zero(
            X_null, x_test, dat$y,
            null_fit_zero = null_zero,
            spa_cutoff = 2
          ),
          zero_nospa = score_test_zero(
            X_null, x_test, dat$y,
            null_fit_zero = null_zero,
            spa_cutoff = Inf
          ),
          full_spa = fast_negbin_hurdle(
            dat$X, dat$y,
            offsetx = dat$offset,
            score_test = "x_test",
            null_fit_count = null_count,
            null_fit_zero = null_zero,
            spa_cutoff = 2
          ),
          full_nospa = fast_negbin_hurdle(
            dat$X, dat$y,
            offsetx = dat$offset,
            score_test = "x_test",
            null_fit_count = null_count,
            null_fit_zero = null_zero,
            spa_cutoff = Inf
          ),
          min_iterations = n_reps,
          check = FALSE,
          memory = TRUE
        )
        bm$config <- label
        results[[length(results) + 1]] <- bm
      }
    }
  }

  res <- do.call(rbind, results)
  # Print summary
  cat("\n\n=== BENCHMARK RESULTS ===\n\n")
  print(res[, c(
    "config", "expression", "median", "mem_alloc",
    "n_itr", "n_gc"
  )])
  invisible(res)
}

# ---- Run ----
res <- run_benchmarks(
  n_vals = c(1e4, 1e5, 5e5),
  kx_vals = c(6),
  n_reps = 5
)
