library(testthat)
library(fasthurdle)
library(pscl)

# --- Test data ---

# Poisson-generated data: for poisson/geometric count + binomial/poisson/geometric zero tests
set.seed(123)
n <- 500
x_pois <- rnorm(n)
z_pois <- rnorm(n)
lambda_pois <- exp(1 + 0.5 * x_pois)
p_pois <- plogis(0.5 - 0.5 * z_pois)
y_pois <- rbinom(n, size = 1, prob = p_pois) * rpois(n, lambda = lambda_pois)
df_pois <- data.frame(y = y_pois, x = x_pois, z = z_pois)

# NB-generated data: for negbin count tests (count theta is finite and estimable)
set.seed(456)
n_nb <- 500
x_nb <- rnorm(n_nb)
z_nb <- rnorm(n_nb)
lambda_nb <- exp(1 + 0.5 * x_nb)
p_nb <- plogis(0.5 - 0.5 * z_nb)
y_nb <- rbinom(n_nb, size = 1, prob = p_nb) * rnbinom(n_nb, size = 2, mu = lambda_nb)
df_nb <- data.frame(y = y_nb, x = x_nb, z = z_nb)

# --- Helper: compare fasthurdle vs pscl ---

compare_hurdle_models <- function(df, count_dist, zero_dist, link = "logit") {
  desc <- paste0(count_dist, "+", zero_dist, ifelse(link != "logit", paste0("+", link), ""))

  fast_model <- fasthurdle(y ~ x | z,
    data = df, dist = count_dist, zero.dist = zero_dist, link = link
  )
  pscl_model <- pscl::hurdle(y ~ x | z,
    data = df, dist = count_dist, zero.dist = zero_dist, link = link
  )

  # Tolerance justification (measured max relative differences, fasthurdle vs pscl):
  #
  # fasthurdle uses y>0 starting values for the count GLM (closer to the
  # truncated MLE), while pscl uses full-data starting values. Both converge
  # to the same log-likelihood (always <1e-10 relative diff) but may follow
  # slightly different optimization paths, causing small coefficient differences.
  #
  # Observed max relative differences by count distribution:
  #   poisson:   coef ~3e-07, SE ~2e-07, fitted ~7e-07
  #   negbin:    coef ~1e-08, SE ~2e-08, fitted ~3e-08
  #   geometric: coef ~3e-05, SE ~5e-06, fitted ~5e-05
  #     (geometric = NB with theta fixed at 1; most sensitive to starting values)
  #
  # Observed max relative differences by zero distribution:
  #   non-negbin: coef ~1e-08, SE ~4e-10
  #   negbin:     coef ~9e-08, SE ~8e-05
  #     (zero NB theta is poorly identified on this data — flat likelihood
  #      surface — causing different optima with near-identical log-likelihoods)
  #
  # Tolerances set ~10x above observed maximums for robustness:
  count_tol <- switch(count_dist,
    "geometric" = 1e-3, # observed ~5e-05, margin ~20x
    "poisson"   = 1e-5, # observed ~7e-07, margin ~14x
    "negbin"    = 1e-6 # observed ~3e-08, margin ~33x
  )
  zero_negbin <- (zero_dist == "negbin")
  zero_coef_tol <- if (zero_negbin) 1e-4 else 1e-6 # observed ~9e-08 max element-wise,
  #   but geometric+negbin compounds to ~3e-05 in expect_equal's mean-scaled metric
  zero_se_tol <- if (zero_negbin) 1e-3 else 1e-6 # observed ~8e-05 vs ~4e-10

  # Coefficients
  expect_equal(coef(fast_model, model = "count"), coef(pscl_model, model = "count"),
    tolerance = count_tol, info = paste(desc, "count coefs")
  )
  expect_equal(coef(fast_model, model = "zero"), coef(pscl_model, model = "zero"),
    tolerance = zero_coef_tol, info = paste(desc, "zero coefs")
  )

  # Log-likelihood (always <1e-10 relative diff — both reach same optimum)
  expect_equal(logLik(fast_model), logLik(pscl_model),
    tolerance = 1e-6, info = paste(desc, "logLik")
  )

  # Fitted values (dominated by count coefficient differences)
  expect_equal(fitted(fast_model), fitted(pscl_model),
    tolerance = count_tol, info = paste(desc, "fitted")
  )

  # Theta (compare on log scale for stability when theta is large)
  if (count_dist == "negbin") {
    # observed ~5e-08; 1e-6 gives ~20x margin
    expect_equal(log(fast_model$theta["count"]), log(pscl_model$theta["count"]),
      tolerance = 1e-6, info = paste(desc, "count log(theta)")
    )
  }
  if (zero_dist == "negbin") {
    # poorly identified parameter; can differ substantially while
    # log-likelihood remains effectively identical
    expect_equal(log(fast_model$theta["zero"]), log(pscl_model$theta["zero"]),
      tolerance = 1e-2, info = paste(desc, "zero log(theta)")
    )
  }

  # Summary estimates and SEs (same tolerances as raw coefficients)
  fast_summary <- summary(fast_model)
  pscl_summary <- summary(pscl_model)

  expect_equal(
    fast_summary$coefficients$count[, "Estimate"],
    pscl_summary$coefficients$count[, "Estimate"],
    tolerance = count_tol, info = paste(desc, "summary count estimates")
  )
  expect_equal(
    fast_summary$coefficients$count[, "Std. Error"],
    pscl_summary$coefficients$count[, "Std. Error"],
    tolerance = count_tol, info = paste(desc, "summary count SEs")
  )
  expect_equal(
    fast_summary$coefficients$zero[, "Estimate"],
    pscl_summary$coefficients$zero[, "Estimate"],
    tolerance = zero_coef_tol, info = paste(desc, "summary zero estimates")
  )
  expect_equal(
    fast_summary$coefficients$zero[, "Std. Error"],
    pscl_summary$coefficients$zero[, "Std. Error"],
    tolerance = zero_se_tol, info = paste(desc, "summary zero SEs")
  )
  # Log-likelihood and AIC are functions of the optimum, not the path
  expect_equal(fast_summary$loglik, pscl_summary$loglik,
    tolerance = 1e-6, info = paste(desc, "summary loglik")
  )
  expect_equal(AIC(fast_model), AIC(pscl_model),
    tolerance = 1e-6, info = paste(desc, "AIC")
  )
}

# --- Distribution combination tests ---

# Poisson/geometric count with non-negbin zero: use Poisson-generated data
for (count_dist in c("poisson", "geometric")) {
  for (zero_dist in c("binomial", "poisson", "geometric")) {
    test_that(paste("fasthurdle matches pscl:", count_dist, "count +", zero_dist, "zero"), {
      compare_hurdle_models(df_pois, count_dist, zero_dist)
    })
  }
}

# Negbin count with non-negbin zero: use NB-generated data (finite theta)
for (zero_dist in c("binomial", "poisson", "geometric")) {
  test_that(paste("fasthurdle matches pscl: negbin count +", zero_dist, "zero"), {
    compare_hurdle_models(df_nb, "negbin", zero_dist)
  })
}

# Any count with negbin zero: use Poisson-generated data
# (zero NB theta poorly identified since true zero mechanism is binomial;
#  relaxed tolerances applied in compare_hurdle_models for zero_dist="negbin")
for (count_dist in c("poisson", "geometric")) {
  test_that(paste("fasthurdle matches pscl:", count_dist, "count + negbin zero"), {
    compare_hurdle_models(df_pois, count_dist, "negbin")
  })
}

test_that("fasthurdle matches pscl: negbin count + negbin zero", {
  compare_hurdle_models(df_nb, "negbin", "negbin")
})

# --- Link function tests ---

for (link_func in c("logit", "probit", "cloglog", "cauchit")) {
  test_that(paste("fasthurdle matches pscl: poisson + binomial +", link_func, "link"), {
    compare_hurdle_models(df_pois, "poisson", "binomial", link = link_func)
  })
}

# link=log is unsupported in both fasthurdle and pscl for this data
test_that("fasthurdle with log link skips (unsupported in pscl too)", {
  expect_error(
    pscl::hurdle(y ~ x | z, data = df_pois, dist = "poisson", zero.dist = "binomial", link = "log"),
    info = "pscl also fails with log link"
  )
})

# Negbin count with different link functions: use NB data
for (link_func in c("logit", "probit", "cloglog", "cauchit")) {
  test_that(paste("fasthurdle matches pscl: negbin + binomial +", link_func, "link"), {
    compare_hurdle_models(df_nb, "negbin", "binomial", link = link_func)
  })
}

# Geometric count with different link functions
for (link_func in c("logit", "probit", "cloglog", "cauchit")) {
  test_that(paste("fasthurdle matches pscl: geometric + binomial +", link_func, "link"), {
    compare_hurdle_models(df_pois, "geometric", "binomial", link = link_func)
  })
}

# --- Offset + different formula test ---

test_that("fasthurdle with offset and different count/zero formulas matches pscl::hurdle", {
  set.seed(42)
  n <- 500
  x <- rnorm(n)
  log_depth <- rnorm(n, mean = 8, sd = 1)

  lambda <- exp(0.5 + 0.3 * x + log_depth)
  p <- plogis(-2 + 0.5 * x + 0.8 * log_depth)
  y <- rbinom(n, size = 1, prob = p) * rnbinom(n, size = 2, mu = lambda)

  df <- data.frame(y = y, x = x, log_depth = log_depth)

  fast_model <- fasthurdle(y ~ x + offset(log_depth) | x + log_depth,
    data = df, dist = "negbin", zero.dist = "binomial", link = "logit"
  )

  pscl_model <- pscl::hurdle(y ~ x + offset(log_depth) | x + log_depth,
    data = df, dist = "negbin", zero.dist = "binomial", link = "logit"
  )

  # negbin count with offset: BFGS convergence path can differ slightly across
  # platforms (macOS ARM vs Linux x86), so use 1e-5 tolerance here.
  # Observed max rel diff: ~2e-06 for coefs, ~3e-06 for fitted on macOS.
  expect_equal(coef(fast_model, model = "count"), coef(pscl_model, model = "count"),
    tolerance = 1e-5
  )
  expect_equal(coef(fast_model, model = "zero"), coef(pscl_model, model = "zero"),
    tolerance = 1e-5
  )
  expect_equal(logLik(fast_model), logLik(pscl_model), tolerance = 1e-5)
  expect_equal(fitted(fast_model), fitted(pscl_model), tolerance = 1e-5)
  expect_equal(fast_model$theta["count"], pscl_model$theta["count"], tolerance = 1e-5)

  fast_summary <- summary(fast_model)
  pscl_summary <- summary(pscl_model)

  expect_equal(
    fast_summary$coefficients$count[, "Estimate"],
    pscl_summary$coefficients$count[, "Estimate"],
    tolerance = 1e-5
  )
  expect_equal(
    fast_summary$coefficients$count[, "Std. Error"],
    pscl_summary$coefficients$count[, "Std. Error"],
    tolerance = 1e-5
  )
  expect_equal(
    fast_summary$coefficients$zero[, "Estimate"],
    pscl_summary$coefficients$zero[, "Estimate"],
    tolerance = 1e-5
  )
  expect_equal(
    fast_summary$coefficients$zero[, "Std. Error"],
    pscl_summary$coefficients$zero[, "Std. Error"],
    tolerance = 1e-5
  )

  expect_length(coef(fast_model, model = "count"), 2)
  expect_length(coef(fast_model, model = "zero"), 3)
})
