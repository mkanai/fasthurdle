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

  # Without SPA: ratio estimator, ~3% bias for N(0,1) predictor
  r_nospa <- score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = NULL)
  expect_equal(r_nospa$beta[1], true_beta, tolerance = 0.05)
  expect_equal(r_nospa$beta[1], wald_beta, tolerance = 0.1)

  # With SPA: refined beta via 5-iter BFGS, <4% bias
  r_spa <- score_test_count(X_null, x, y, dist = "negbin", spa_cutoff = 2)
  if (r_spa$spa_applied) {
    # Refined beta should be closer to Wald MLE than ratio estimator
    expect_equal(r_spa$beta[1], wald_beta, tolerance = 0.05)
  }

  # log10(p) within 2 orders of magnitude (chi-squared vs Wald)
  p_wald <- summary(m)$coefficients$count["x", "Pr(>|z|)"]
  expect_true(abs(log10(r_nospa$pvalue) - log10(p_wald)) < 2)
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

  # Ratio estimator within 15% of Wald (Schur corrects for correlation)
  expect_equal(r$beta[1], wald_beta, tolerance = 0.15)
  # Must NOT be ~half the Wald (= missing Schur complement)
  expect_true(abs(r$beta[1]) > abs(wald_beta) * 0.7)
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
  # SPA path refines beta (5-iter BFGS from score warm start)
  # Refined beta within 5% of true for N(0,1) predictor
  # Ratio estimator (chi2 path) within 10%
  expect_equal(r_spa$beta[1], 0.3, tolerance = 0.05)
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
    null_fit = nf, spa_cutoff = NULL
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

test_that("fast_negbin_hurdle accepts cached null_fit", {
  d <- gen_data(500, beta_x = 0.3)
  X_full <- cbind(d$X_null, d$x)
  colnames(X_full) <- c("(Intercept)", "x")
  nf <- fit_null_count(d$X_null, d$y, dist = "negbin")
  m <- fast_negbin_hurdle(X_full, d$y, score_test = "x", null_fit = nf)
  expect_true(m$score_test$pvalue < 0.05)
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
  expect_true(!is.null(m$score_test))
  expect_length(m$score_test$beta, 1)
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
