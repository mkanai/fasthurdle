library(testthat)
library(fasthurdle)

# Helper: generate hurdle NB dataset
gen_data <- function(n = 500, beta_x = 0.3, seed = 42) {
  set.seed(seed)
  x <- rnorm(n)
  y <- rbinom(n, 1, 0.5) * rnbinom(n, size = 2, mu = exp(0.5 + beta_x * x))
  X_null <- matrix(1, n, 1)
  colnames(X_null) <- "(Intercept)"
  list(y = y, x = x, X_null = X_null, n = n)
}

# ==========================================================================
# Expected FIM
# ==========================================================================

test_that("expected FIM is symmetric positive definite", {
  d <- gen_data(200)
  pos <- d$y > 0
  fim <- compute_ztnb_fisher_info_cpp(
    c(0.5, 0.3), 2.0,
    cbind(1, d$x)[pos, ], rep(0, sum(pos)), rep(1, sum(pos))
  )
  expect_equal(dim(fim), c(3L, 3L))
  expect_equal(fim, t(fim), tolerance = 1e-10)
  eig <- eigen(fim, symmetric = TRUE, only.values = TRUE)$values
  expect_true(all(eig > 0))
})

# ==========================================================================
# Basic functionality across distributions
# ==========================================================================

test_that("score_test_count works with negbin/poisson/geometric", {
  d <- gen_data(500, beta_x = 0.3)
  for (dist in c("negbin", "poisson", "geometric")) {
    r <- score_test_count(d$X_null, d$x, d$y, dist = dist, spa_cutoff = NULL)
    expect_true(r$pvalue < 0.05, info = dist)
    expect_true(r$statistic > 0, info = dist)
    expect_length(r$beta, 1)
    expect_true(r$se[1] > 0, info = dist)
  }
})

test_that("score_test_count rejects multi-column x_test", {
  d <- gen_data(200, beta_x = 0.3)
  x_multi <- cbind(d$x, rnorm(200))
  expect_error(
    score_test_count(d$X_null, x_multi, d$y, dist = "negbin"),
    "single test variable"
  )
})

# ==========================================================================
# Statistical accuracy
# ==========================================================================

test_that("z = beta/SE exactly recovers p-value (chi-squared and SPA)", {
  d <- gen_data(2000, beta_x = 0.3)

  # Chi-squared path: no SPA, no refine — ratio estimator + back-computed SE
  r <- score_test_count(d$X_null, d$x, d$y, dist = "negbin", spa_cutoff = NULL)
  p_from_z <- 2 * pnorm(-abs(r$beta[1] / r$se[1]))
  expect_equal(p_from_z, r$pvalue, tolerance = 1e-10)

  # SPA path: SPA + refine — refined beta + SE back-computed from SPA p-value
  # Consistency holds because SE = |beta_refined| / |qnorm(p/2)|
  r_spa <- score_test_count(d$X_null, d$x, d$y, dist = "negbin", spa_cutoff = 2)
  if (r_spa$spa_applied) {
    p_from_z_spa <- 2 * pnorm(-abs(r_spa$beta[1] / r_spa$se[1]))
    expect_equal(p_from_z_spa, r_spa$pvalue, tolerance = 1e-10)
  }
})

test_that("score beta is close to true beta and Wald MLE at large n", {
  set.seed(42)
  n <- 5000
  true_beta <- 0.3
  x <- rnorm(n)
  y <- rbinom(n, 1, 0.5) * rnbinom(n, size = 2, mu = exp(0.5 + true_beta * x))
  X_null <- matrix(1, n, 1)
  colnames(X_null) <- "(Intercept)"
  X_full <- cbind(X_null, x)
  colnames(X_full) <- c("(Intercept)", "x")

  m <- fast_negbin_hurdle(X_full, y)
  wald_beta <- unname(coef(m, model = "count")["x"])
  wald_se <- unname(sqrt(diag(m$vcov))["count_x"])

  # Beta is now refined via 5-iter BFGS for all significant tests (p < 0.05),
  # regardless of SPA. This corrects the ratio estimator bias from observed info.
  r_nospa <- score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = NULL)
  expect_equal(r_nospa$beta[1], true_beta, tolerance = 0.1)
  expect_equal(r_nospa$beta[1], wald_beta, tolerance = 0.1)

  # With SPA: same refined beta (refinement triggers on significance, not SPA)
  r_spa <- score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = 2)
  if (r_spa$spa_applied) {
    expect_equal(r_spa$beta[1], wald_beta, tolerance = 0.1)
  }

  # Both score and Wald should be highly significant for this effect size.
  # With observed info, exact p-values can differ substantially from Wald
  # (observed info test can be more powerful), so we check direction + significance.
  p_wald <- summary(m)$coefficients$count["x", "Pr(>|z|)"]
  expect_true(r_nospa$pvalue < 1e-10)
  expect_true(p_wald < 1e-10)
})

test_that("Schur complement gives correct beta with correlated covariates", {
  set.seed(42)
  n <- 5000
  x <- rnorm(n)
  cov1 <- x + rnorm(n) # correlated with x (r ~ 0.7)
  y <- rbinom(n, 1, 0.5) * rnbinom(n, size = 2, mu = exp(0.5 + 0.3 * x + 0.1 * cov1))
  X_null <- cbind(1, cov1)
  colnames(X_null) <- c("(Intercept)", "cov1")
  X_full <- cbind(X_null, x)
  colnames(X_full) <- c("(Intercept)", "cov1", "x")

  # Without SPA: ratio estimator with Schur complement
  r <- score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = NULL)
  m <- fast_negbin_hurdle(X_full, y)
  wald_beta <- unname(coef(m, model = "count")["x"])

  # Beta refined via BFGS for significant tests. Should be close to Wald.
  # Key check: Schur complement is working (beta not halved by covariate confounding).
  expect_equal(r$beta[1], wald_beta, tolerance = 0.15)
  expect_true(abs(r$beta[1]) > abs(wald_beta) * 0.5)
})

test_that("score test is calibrated under null", {
  n_sim <- 200
  pvals <- numeric(n_sim)
  for (i in seq_len(n_sim)) {
    set.seed(i + 5000)
    n <- 2000
    x <- rnorm(n)
    y <- rbinom(n, 1, 0.5) * rnbinom(n, size = 2, mu = exp(0.5))
    X_null <- matrix(1, n, 1)
    colnames(X_null) <- "(Intercept)"
    r <- score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = NULL)
    pvals[i] <- r$pvalue
  }
  # 95% CI for FPR=0.05 at n_sim=200 is roughly [0.02, 0.08]
  fpr <- mean(pvals < 0.05)
  expect_true(fpr < 0.10, info = paste("FPR at 0.05:", fpr))
  expect_true(fpr > 0.01, info = paste("FPR at 0.05:", fpr, "(too conservative)"))
})

# ==========================================================================
# SPA + beta refinement
# ==========================================================================

test_that("spa_cutoff = NULL or Inf disables SPA and refinement", {
  d <- gen_data(1000, beta_x = 0.3)
  r1 <- score_test_count(d$X_null, d$x, d$y, dist = "negbin", spa_cutoff = NULL)
  r2 <- score_test_count(d$X_null, d$x, d$y, dist = "negbin", spa_cutoff = Inf)
  expect_false(r1$spa_applied)
  expect_false(r2$spa_applied)
  # Both give identical results (same ratio estimator, same chi-squared p)
  expect_equal(r1$pvalue, r2$pvalue)
  expect_equal(r1$beta, r2$beta)
})

test_that("SPA does not fire for non-significant tests", {
  set.seed(99)
  n <- 500
  x <- rnorm(n)
  y <- rbinom(n, 1, 0.5) * rnbinom(n, size = 2, mu = exp(0.5)) # null
  X_null <- matrix(1, n, 1)
  colnames(X_null) <- "(Intercept)"
  r <- score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = 2)
  expect_false(r$spa_applied)
})

test_that("SPA refines beta and adjusts p-value vs chi-squared", {
  d <- gen_data(5000, beta_x = 0.3)
  r_chi2 <- score_test_count(d$X_null, d$x, d$y, dist = "negbin", spa_cutoff = NULL)
  r_spa <- score_test_count(d$X_null, d$x, d$y, dist = "negbin", spa_cutoff = 2)
  expect_true(r_spa$spa_applied)
  # Both paths now refine beta via BFGS when significant (p < 0.05).
  # Refined beta within 10% of true for N(0,1) predictor.
  expect_equal(r_spa$beta[1], 0.3, tolerance = 0.1)
  expect_equal(r_chi2$beta[1], 0.3, tolerance = 0.1)
  # P-values differ (SPA adjusts tail)
  expect_true(r_spa$pvalue != r_chi2$pvalue)
})

# ==========================================================================
# Cached null model
# ==========================================================================

test_that("fit_null_count returns correct class and cached null is reproducible", {
  d <- gen_data(500, beta_x = 0.3)
  nf <- fit_null_count(d$X_null, d$y, dist = "negbin")
  expect_s3_class(nf, "fasthurdle_null")
  expect_equal(nf$convergence, 0L)
  expect_equal(nf$dist, "negbin")

  # Cached gives identical results (same path, no SPA/refine)
  r1 <- score_test_count(d$X_null, d$x, d$y, dist = "negbin", spa_cutoff = NULL)
  r2 <- score_test_count(d$X_null, d$x, d$y,
    dist = "negbin",
    null_fit_count = nf, spa_cutoff = NULL
  )
  expect_equal(r1$pvalue, r2$pvalue)
  expect_equal(r1$beta, r2$beta)
})

# ==========================================================================
# Integration: fast_negbin_hurdle score test mode
# ==========================================================================

test_that("fast_negbin_hurdle score_test mode returns valid summary", {
  d <- gen_data(500, beta_x = 0.3)
  X_full <- cbind(d$X_null, d$x)
  colnames(X_full) <- c("(Intercept)", "x")
  m <- fast_negbin_hurdle(X_full, d$y, score_test = "x")

  s <- summary(m)
  # Test variable: score-derived beta/SE/z/p (all present)
  expect_true(!is.na(s$coefficients$count["x", "Estimate"]))
  expect_true(!is.na(s$coefficients$count["x", "Pr(>|z|)"]))

  # Covariate: null MLE beta, NA for SE/z/p
  expect_true(!is.na(s$coefficients$count["(Intercept)", "Estimate"]))
  expect_true(is.na(s$coefficients$count["(Intercept)", "Std. Error"]))

  # Zero model: fully computed (Wald)
  expect_true(!is.na(s$coefficients$zero["x", "Pr(>|z|)"]))

  # Integer index is rejected (must use character name)
  expect_error(
    fast_negbin_hurdle(X_full, d$y, score_test = 2L),
    "character"
  )
})

test_that("fast_negbin_hurdle accepts cached null_fit_count", {
  d <- gen_data(500, beta_x = 0.3)
  X_full <- cbind(d$X_null, d$x)
  colnames(X_full) <- c("(Intercept)", "x")
  nf <- fit_null_count(d$X_null, d$y, dist = "negbin")
  m <- fast_negbin_hurdle(X_full, d$y, score_test = "x", null_fit_count = nf)
  expect_true(m$score_test_count$pvalue < 0.05)
  expect_true(m$converged)
})

test_that("fasthurdle score_test works via formula", {
  set.seed(42)
  n <- 500
  df <- data.frame(
    y = rbinom(n, 1, 0.5) * rnbinom(n, 2, mu = exp(0.5 + 0.3 * rnorm(n))),
    x = rnorm(n)
  )
  m <- fasthurdle(y ~ x, data = df, dist = "negbin", score_test = "x")
  expect_true(!is.null(m$score_test_count))
  expect_length(m$score_test_count$beta, 1)
})

# ==========================================================================
# Edge cases
# ==========================================================================

test_that("score test works with offset", {
  set.seed(42)
  n <- 500
  x <- rnorm(n)
  off <- rnorm(n, 5)
  y <- rbinom(n, 1, 0.5) * rnbinom(n, size = 2, mu = exp(0.3 * x + off))
  X_null <- matrix(1, n, 1)
  colnames(X_null) <- "(Intercept)"
  r <- score_test_count(X_null, x, y, offsetx = off, dist = "negbin")
  expect_true(r$pvalue < 0.05)
})

test_that("score test is non-significant for null effect", {
  set.seed(99)
  n <- 500
  x <- rnorm(n)
  y <- rbinom(n, 1, 0.5) * rnbinom(n, size = 2, mu = exp(0.5))
  X_null <- matrix(1, n, 1)
  colnames(X_null) <- "(Intercept)"
  r <- score_test_count(X_null, x, y, dist = "negbin")
  expect_true(r$pvalue > 0.001)
})

test_that("score_test column not found gives clear error", {
  d <- gen_data(100, beta_x = 0.3)
  X_full <- cbind(d$X_null, d$x)
  colnames(X_full) <- c("(Intercept)", "x")
  expect_error(
    fast_negbin_hurdle(X_full, d$y, score_test = "nonexistent"),
    "score_test column"
  )
})

# ==========================================================================
# Zero model score test
# ==========================================================================

test_that("score_test_zero works and matches Wald", {
  set.seed(42)
  n <- 2000
  x <- rpois(n, 0.5)
  cov1 <- rnorm(n)
  y <- rbinom(n, 1, plogis(0.5 + 0.3 * x - 0.2 * cov1)) *
    rnbinom(n, size = 2, mu = exp(0.5 + 0.3 * x))

  Z_null <- cbind(1, cov1)
  colnames(Z_null) <- c("(Intercept)", "cov1")
  Z_full <- cbind(Z_null, x)
  colnames(Z_full) <- c("(Intercept)", "cov1", "x")

  # Score test
  r <- score_test_zero(Z_null, x, y, spa_cutoff = NULL)
  expect_true(r$pvalue < 0.05)
  expect_length(r$beta, 1)
  expect_true(r$se[1] > 0)

  # Wald comparison
  X_full <- cbind(1, x, cov1)
  colnames(X_full) <- c("(Intercept)", "x", "cov1")
  m <- fast_negbin_hurdle(X_full, y, Z = Z_full)
  wald_beta <- summary(m)$coefficients$zero["x", "Estimate"]
  # Score beta within 15% of Wald (ratio estimator for logit)
  expect_equal(r$beta[1], wald_beta, tolerance = 0.15)
})

test_that("fit_null_zero returns correct class", {
  set.seed(42)
  n <- 500
  y <- rbinom(n, 1, 0.3) * rpois(n, 2)
  Z_null <- matrix(1, n, 1)
  colnames(Z_null) <- "(Intercept)"
  nfz <- fit_null_zero(Z_null, y)
  expect_s3_class(nfz, "fasthurdle_null_zero")
  expect_equal(nfz$convergence, 0L)
  expect_equal(nfz$link, "logit")
})

test_that("fast_negbin_hurdle with null_fit_zero caches both models", {
  set.seed(42)
  n <- 1000
  peak_acc <- rpois(n, 0.5)
  pct_mito <- runif(n, 0, 0.2)
  log_tc <- rnorm(n, 8, 0.3)
  mu <- exp(-7 + 0.3 * peak_acc - 0.5 * pct_mito + log_tc)
  y <- rbinom(n, 1, plogis(0.5 + 0.2 * peak_acc)) * rnbinom(n, size = 2, mu = mu)

  X <- cbind(1, peak_acc, pct_mito)
  colnames(X) <- c("(Intercept)", "peak_acc", "pct_mito")
  Z <- cbind(1, peak_acc, log_tc, pct_mito)
  colnames(Z) <- c("(Intercept)", "peak_acc", "log_total_counts", "pct_mito")
  X_null <- X[, c(1, 3), drop = FALSE]
  Z_null <- Z[, c(1, 3, 4), drop = FALSE]

  nf <- fit_null_count(X_null, y, offsetx = log_tc, dist = "negbin")
  nfz <- fit_null_zero(Z_null, y)
  m <- fast_negbin_hurdle(X, y,
    Z = Z, offsetx = log_tc,
    score_test = "peak_acc", null_fit_count = nf, null_fit_zero = nfz
  )

  s <- summary(m)
  # Both count and zero have score test results
  expect_true(!is.na(s$coefficients$count["peak_acc", "Pr(>|z|)"]))
  expect_true(!is.na(s$coefficients$zero["peak_acc", "Pr(>|z|)"]))
  # Covariates have NA SE (null model only)
  expect_true(is.na(s$coefficients$zero["(Intercept)", "Std. Error"]))
  # score_test_zero slot exists
  expect_true(!is.null(m$score_test_zero))
})

test_that("zero score test z = beta/SE recovers p-value", {
  set.seed(42)
  n <- 2000
  x <- rpois(n, 0.5)
  y <- rbinom(n, 1, plogis(0.5 + 0.3 * x)) * rnbinom(n, size = 2, mu = exp(0.5))
  Z_null <- matrix(1, n, 1)
  colnames(Z_null) <- "(Intercept)"

  r <- score_test_zero(Z_null, x, y, spa_cutoff = NULL)
  p_from_z <- 2 * pnorm(-abs(r$beta[1] / r$se[1]))
  expect_equal(p_from_z, r$pvalue, tolerance = 1e-10)
})

# ==========================================================================
# Model misspecification: spike-at-1 (ambient RNA contamination)
# ==========================================================================

test_that("observed info score test is calibrated under spike-at-1 misspecification", {
  # Spike-at-1 model: 50% of "expressed" cells get count=1 (ambient RNA),
  # rest follow ZTNB. The expected FIM is severely inflated under this
  # misspecification; the observed Hessian should match Wald calibration.
  n_sim <- 100
  n <- 10000
  pvals_score <- pvals_wald <- numeric(n_sim)
  for (i in seq_len(n_sim)) {
    set.seed(i)
    peak_acc <- rpois(n, 0.5)
    log_tc <- rnorm(n, 8, 0.3)
    pct_mito <- runif(n, 0, 0.2)
    mu <- exp(-7 + 0 * peak_acc - 0.5 * pct_mito + log_tc)
    p_expr <- plogis(qlogis(0.05) + 0.2 * peak_acc - 0.3 * pct_mito)
    is_spike <- rbinom(n, 1, 0.5)
    nb_count <- pmax(rnbinom(n, size = 3, mu = mu), 1L)
    y <- as.integer(rbinom(n, 1, p_expr) * ifelse(is_spike, 1L, nb_count))
    X <- model.matrix(~ peak_acc + pct_mito)
    Z <- model.matrix(~ peak_acc + log_tc + pct_mito)
    fw <- tryCatch(
      suppressWarnings(fast_negbin_hurdle(X, y, Z = Z, offsetx = log_tc)),
      error = function(e) NULL)
    fs <- tryCatch(
      suppressWarnings(fast_negbin_hurdle(X, y, Z = Z, offsetx = log_tc,
                                          score_test = "peak_acc",
                                          spa_cutoff = NULL)),
      error = function(e) NULL)
    pvals_wald[i] <- if (!is.null(fw)) {
      summary(fw)$coefficients$count["peak_acc", "Pr(>|z|)"]
    } else NA
    pvals_score[i] <- if (!is.null(fs)) {
      summary(fs)$coefficients$count["peak_acc", "Pr(>|z|)"]
    } else NA
  }
  fpr_wald <- mean(pvals_wald[!is.na(pvals_wald)] < 0.05)
  fpr_score <- mean(pvals_score[!is.na(pvals_score)] < 0.05)
  # Score FPR should be within 10% of Wald FPR (both may be mildly inflated)
  expect_true(abs(fpr_score - fpr_wald) < 0.10,
              info = sprintf("Score FPR=%.3f, Wald FPR=%.3f", fpr_score, fpr_wald))
  # Neither should be catastrophically inflated (< 25%)
  expect_true(fpr_score < 0.25,
              info = sprintf("Score FPR=%.3f too high", fpr_score))
})

# ==========================================================================
# Edge cases
# ==========================================================================

test_that("score test handles small n_pos gracefully", {
  # Very sparse data: only ~10-20 positive observations
  set.seed(42)
  n <- 5000
  x <- rnorm(n)
  # Very sparse: ~1% expression
  y <- rbinom(n, 1, 0.01) * rnbinom(n, size = 2, mu = exp(0.5 + 0.3 * x))
  X_null <- matrix(1, n, 1)
  colnames(X_null) <- "(Intercept)"
  n_pos <- sum(y > 0)
  # Should have ~50 positive obs at n=5000, 1% expr
  expect_true(n_pos >= 5)
  r <- score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = NULL)
  # Should return a valid result (not error), possibly NA if too few
  expect_true(is.numeric(r$pvalue))
  expect_true(is.numeric(r$beta))
  if (!is.na(r$pvalue)) {
    expect_true(r$pvalue >= 0 && r$pvalue <= 1)
  }
})

test_that("score test handles extreme theta (near-Poisson)", {
  # theta = 100 → near-Poisson, observed Hessian should still work
  set.seed(42)
  n <- 2000
  x <- rnorm(n)
  y <- rbinom(n, 1, 0.5) * rnbinom(n, size = 100, mu = exp(0.5 + 0.3 * x))
  X_null <- matrix(1, n, 1)
  colnames(X_null) <- "(Intercept)"
  r <- score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = NULL)
  expect_true(!is.na(r$pvalue))
  expect_true(r$pvalue < 0.05)  # true effect, should detect
  expect_true(r$beta[1] > 0)    # correct sign
})

test_that("score test handles high overdispersion (theta near 0)", {
  # theta = 0.1 → very overdispersed, lots of large counts + many 1s
  set.seed(42)
  n <- 2000
  x <- rnorm(n)
  y <- rbinom(n, 1, 0.3) * rnbinom(n, size = 0.1, mu = exp(0.5 + 0.3 * x))
  X_null <- matrix(1, n, 1)
  colnames(X_null) <- "(Intercept)"
  r <- score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = NULL)
  expect_true(is.numeric(r$pvalue))
  if (!is.na(r$pvalue)) {
    expect_true(r$pvalue >= 0 && r$pvalue <= 1)
  }
})

test_that("observed info falls back to expected FIM when Hessian is non-finite", {
  # Degenerate case: all positive counts are 1 (spike-at-1 with 100% spike)
  set.seed(42)
  n <- 500
  x <- rnorm(n)
  y <- rbinom(n, 1, 0.3)  # all positives are exactly 1
  X_null <- matrix(1, n, 1)
  colnames(X_null) <- "(Intercept)"
  # Should not error — falls back to expected FIM or returns NA
  r <- tryCatch(
    score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = NULL),
    error = function(e) list(pvalue = NA)
  )
  expect_true(is.numeric(r$pvalue))
})
