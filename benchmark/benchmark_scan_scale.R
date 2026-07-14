#!/usr/bin/env Rscript

# Peak-gene scan at analysis scale: how hurdle_scan() compares to refitting the full
# model once per peak, as pscl requires.
#
# Two configurations, both NB count + binomial zero hurdle, both single-threaded:
#   small - 5,000 nuclei, 200 peaks, 2 covariates
#   large - 500,000 nuclei, 500 peaks, 20 covariates (a realistic cis-window scan)
#
# hurdle_scan() is timed in full. The per-peak refit baselines are timed on a handful of
# peaks and scaled by the peak count: they are independent refits over identical data, so
# the loop is linear, but the totals are extrapolated rather than measured. (Checked
# against a full 200-peak pscl run in the small config: 45.2s extrapolated vs 45.4s
# measured.) Running the large pscl baseline for real would take over four hours.
#
#   OMP_NUM_THREADS=1 Rscript benchmark/benchmark_scan_scale.R

# OMP_NUM_THREADS must be set in the environment before R starts. R links a threaded
# OpenBLAS, which reads its thread count at load time, so Sys.setenv() here would be too
# late -- and on matrices this small a 16-thread BLAS is ~2.6x SLOWER than a single
# thread, which silently distorts every timing.
if (!identical(Sys.getenv("OMP_NUM_THREADS"), "1")) {
  stop(
    "Set OMP_NUM_THREADS=1 in the environment before starting R:\n",
    "  OMP_NUM_THREADS=1 Rscript benchmark/benchmark_scan_scale.R"
  )
}

suppressMessages({
  library(fasthurdle)
  library(pscl)
})

out_dir <- file.path("benchmark", "benchmark_results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

# True hurdle DGP: zero with prob 1 - p, else zero-truncated NB. Only peak1 is causal.
simulate_gene <- function(n, n_peaks, n_cov, zero_frac, theta = 2, seed = 1) {
  set.seed(seed)
  peaks <- matrix(rpois(n * n_peaks, 2), n)
  colnames(peaks) <- paste0("peak", seq_len(n_peaks))
  covs <- matrix(rnorm(n * n_cov), n, n_cov)
  colnames(covs) <- paste0("c", seq_len(n_cov))

  log_atac <- rnorm(n, 8, 1)
  pct_mito <- runif(n, 0, 0.2)
  log_rna <- rnorm(n, 8, 1)

  # solve the zero-model intercept so the gene hits the requested zero fraction
  a <- uniroot(
    function(a) mean(1 - plogis(a - 0.4 * peaks[, 1])) - zero_frac,
    interval = c(-30, 30)
  )$root
  p <- plogis(a - 0.4 * peaks[, 1])
  mu <- exp(-0.5 + 0.3 * peaks[, 1] + log_rna - 8)
  u <- runif(n, pnbinom(0, size = theta, mu = mu), 1)
  y <- rbinom(n, 1, p) * qnbinom(u, size = theta, mu = mu)

  df <- data.frame(
    y = y, log_atac = log_atac, pct_mito = pct_mito, log_rna = log_rna
  )
  df <- cbind(df, as.data.frame(covs), as.data.frame(peaks))
  list(df = df, y = y, peak_names = colnames(peaks), cov_names = colnames(covs))
}

# median seconds per peak over the first `k` peaks
per_peak <- function(k, f) {
  median(vapply(seq_len(k), function(i) {
    system.time(f(paste0("peak", i)))[3]
  }, numeric(1)))
}

run_config <- function(label, n, n_peaks, n_cov, zero_frac, k_pscl, k_fast) {
  d <- simulate_gene(n, n_peaks, n_cov, zero_frac)
  df <- d$df
  y <- d$y
  cv <- paste(d$cov_names, collapse = " + ")

  cat(sprintf(
    "\n=== %s: n=%d, peaks=%d, covariates=%d, zero frac=%.3f, expressing=%d\n",
    label, n, n_peaks, n_cov + 3, mean(y == 0), sum(y > 0)
  ))

  t_pscl <- per_peak(k_pscl, function(pk) {
    f <- as.formula(sprintf(
      "y ~ %s + %s + log_atac + pct_mito + offset(log_rna) | %s + %s + log_atac + pct_mito",
      pk, cv, pk, cv
    ))
    pscl::hurdle(f, data = df, dist = "negbin", zero.dist = "binomial")
  })

  design <- function(pk) {
    list(
      X = model.matrix(as.formula(sprintf("~ %s + %s + pct_mito", pk, cv)), df),
      Z = model.matrix(
        as.formula(sprintf("~ %s + %s + log_atac + pct_mito", pk, cv)), df
      )
    )
  }

  t_wald <- per_peak(k_fast, function(pk) {
    m <- design(pk)
    fast_negbin_hurdle(m$X, y, Z = m$Z, offsetx = df$log_rna)
  })

  t_score <- per_peak(k_fast, function(pk) {
    m <- design(pk)
    fast_negbin_hurdle(m$X, y, Z = m$Z, offsetx = df$log_rna, score_test = pk)
  })

  # hurdle_scan: null models fit once, every peak scored against them. Timed in full.
  all_peaks <- paste(d$peak_names, collapse = " + ")
  X_all <- model.matrix(
    as.formula(sprintf("~ %s + pct_mito + %s", cv, all_peaks)), df
  )
  Z_all <- model.matrix(
    as.formula(sprintf("~ %s + log_atac + pct_mito + %s", cv, all_peaks)), df
  )
  t_scan <- system.time(
    res <- hurdle_scan(X_all, y, peaks = d$peak_names, Z = Z_all, offsetx = df$log_rna)
  )[3]

  top <- res$peak[which.max(res$nlog10p_count)]
  cat(sprintf("  hurdle_scan recovered %s as top peak (peak1 is the causal one)\n", top))

  # per-peak costs scaled to the full peak set; hurdle_scan is the measured total
  tot_pscl <- t_pscl * n_peaks
  tot_wald <- t_wald * n_peaks
  tot_score <- t_score * n_peaks

  cat(sprintf("  pscl        refit/peak: %8.2f s -> %8.2f s total (extrapolated)\n", t_pscl, tot_pscl))
  cat(sprintf("  fasthurdle  refit/peak: %8.2f s -> %8.2f s total (extrapolated), Wald\n", t_wald, tot_wald))
  cat(sprintf("  fasthurdle  refit/peak: %8.2f s -> %8.2f s total (extrapolated), score test\n", t_score, tot_score))
  cat(sprintf("  hurdle_scan           : %8.2f s total (measured)  = %.0fx vs pscl\n", t_scan, tot_pscl / t_scan))

  data.frame(
    config = label, n = n, n_peaks = n_peaks, n_cov = n_cov + 3,
    zero_frac = mean(y == 0),
    pscl_per_peak_s = t_pscl, pscl_total_s = tot_pscl,
    fast_wald_per_peak_s = t_wald, fast_wald_total_s = tot_wald,
    fast_score_per_peak_s = t_score, fast_score_total_s = tot_score,
    scan_total_s = t_scan,
    speedup_scan_vs_pscl = tot_pscl / t_scan,
    top_peak = top
  )
}

results <- rbind(
  run_config("small",
    n = 5e3, n_peaks = 200, n_cov = 2, zero_frac = 0.85,
    k_pscl = 5, k_fast = 5
  ),
  run_config("large",
    n = 5e5, n_peaks = 500, n_cov = 17, zero_frac = 0.85,
    k_pscl = 3, k_fast = 5
  )
)

write.csv(results, file.path(out_dir, sprintf("scan_scale_%s.csv", stamp)), row.names = FALSE)
cat("\nWrote", file.path(out_dir, sprintf("scan_scale_%s.csv", stamp)), "\n")
