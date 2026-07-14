#!/usr/bin/env Rscript

# fasthurdle vs pscl::hurdle across the count x zero-hurdle grid, swept over sample size
# and zero fraction. Backs the model-fitting table in the README; the peak-gene scan
# table comes from benchmark_scan_scale.R.
#
#   OMP_NUM_THREADS=1 Rscript benchmark/benchmark_readme.R

# Single-threaded, so the reported speedup reflects the implementation rather than the
# core count (pscl is single-threaded; fasthurdle uses OpenMP). This has to be set in the
# environment before R starts: R here links a threaded OpenBLAS, which reads its thread
# count at load time, so Sys.setenv() from inside the script is too late. It is not a
# cosmetic difference -- on the small matrices these models produce, a 16-thread BLAS is
# ~2.6x SLOWER than a single-threaded one, which silently distorts every timing.
if (!identical(Sys.getenv("OMP_NUM_THREADS"), "1")) {
  stop(
    "Set OMP_NUM_THREADS=1 in the environment before starting R:\n",
    "  OMP_NUM_THREADS=1 Rscript benchmark/benchmark_readme.R"
  )
}

suppressMessages({
  library(fasthurdle)
  library(pscl)
  library(microbenchmark)
})

out_dir <- file.path("benchmark", "benchmark_results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

# --- Data generation ----------------------------------------------------------
# True hurdle DGP: y = 0 with prob 1 - p, else y ~ zero-truncated NB(mu, theta).
# The zero-model intercept is solved to hit the requested zero fraction exactly.
make_data <- function(n, zero_frac, theta = 2, seed = 123) {
  set.seed(seed)
  x <- rnorm(n)
  z <- rnorm(n)

  # solve intercept a such that mean(1 - plogis(a - 0.5 * z)) == zero_frac
  a <- uniroot(
    function(a) mean(1 - plogis(a - 0.5 * z)) - zero_frac,
    interval = c(-20, 20)
  )$root
  p <- plogis(a - 0.5 * z)

  mu <- exp(1 + 0.5 * x)
  # zero-truncated NB via inverse-CDF on the conditional-on-positive support
  u <- runif(n, pnbinom(0, size = theta, mu = mu), 1)
  y_pos <- qnbinom(u, size = theta, mu = mu)

  y <- rbinom(n, size = 1, prob = p) * y_pos
  data.frame(y = y, x = x, z = z)
}

# --- Part A: count x zero grid -------------------------------------------------
count_dists <- c("poisson", "negbin", "geometric")
zero_dists <- c("binomial", "poisson", "negbin", "geometric")
sample_sizes <- c(1e3, 1e4, 1e5)
zero_fracs <- c(0.4, 0.8, 0.95)

# fewer reps at large n -- the fits dominate and the variance is small there
reps_for <- function(n) if (n <= 1e3) 20 else if (n <= 1e4) 10 else 5

grid <- expand.grid(
  count_dist = count_dists,
  zero_dist = zero_dists,
  n = sample_sizes,
  zero_frac = zero_fracs,
  stringsAsFactors = FALSE
)

cat("Part A:", nrow(grid), "configurations\n")
rows <- vector("list", nrow(grid))

for (i in seq_len(nrow(grid))) {
  g <- grid[i, ]
  df <- make_data(g$n, g$zero_frac)
  times <- reps_for(g$n)

  cat(sprintf(
    "[%3d/%3d] n=%-6g zf=%.2f %s/%s ... ",
    i, nrow(grid), g$n, g$zero_frac, g$count_dist, g$zero_dist
  ))

  res <- tryCatch(
    {
      fit_f <- fasthurdle(y ~ x | z,
        data = df, dist = g$count_dist, zero.dist = g$zero_dist
      )
      fit_p <- pscl::hurdle(y ~ x | z,
        data = df, dist = g$count_dist, zero.dist = g$zero_dist
      )
      # Same MLE? A speedup is meaningless if the two disagree.
      coef_diff <- max(abs(unlist(fit_f$coefficients) - unlist(fit_p$coefficients)))

      mb <- microbenchmark(
        fasthurdle = fasthurdle(y ~ x | z,
          data = df, dist = g$count_dist, zero.dist = g$zero_dist
        ),
        pscl = pscl::hurdle(y ~ x | z,
          data = df, dist = g$count_dist, zero.dist = g$zero_dist
        ),
        times = times
      )
      s <- summary(mb, unit = "ms")
      t_f <- s$median[s$expr == "fasthurdle"]
      t_p <- s$median[s$expr == "pscl"]

      list(t_fast = t_f, t_pscl = t_p, speedup = t_p / t_f, coef_diff = coef_diff)
    },
    error = function(e) {
      cat("ERROR:", conditionMessage(e), "\n")
      list(t_fast = NA, t_pscl = NA, speedup = NA, coef_diff = NA)
    }
  )

  if (!is.na(res$speedup)) {
    cat(sprintf(
      "%6.1fms vs %7.1fms = %5.1fx (coef diff %.1e)\n",
      res$t_fast, res$t_pscl, res$speedup, res$coef_diff
    ))
  }

  rows[[i]] <- cbind(g, as.data.frame(res))
}

grid_results <- do.call(rbind, rows)
write.csv(grid_results,
  file.path(out_dir, sprintf("readme_grid_%s.csv", stamp)),
  row.names = FALSE
)

cat("\nWrote grid results to", out_dir, "with stamp", stamp, "\n")
