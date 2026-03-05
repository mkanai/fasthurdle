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

  # When zero_dist="negbin", the zero NB theta is often poorly identified
  # (flat likelihood surface), so different optimizers converge to different
  # parameter combinations that yield nearly identical log-likelihoods.
  # Use relaxed tolerances for such cases.
  zero_negbin <- (zero_dist == "negbin")
  coef_tol <- if (zero_negbin) 0.1 else 1e-3
  fitted_tol <- if (zero_negbin) 1e-2 else 1e-3
  se_tol <- if (zero_negbin) 0.1 else 1e-2

  # Coefficients
  expect_equal(coef(fast_model, model = "count"), coef(pscl_model, model = "count"),
    tolerance = 1e-3, info = paste(desc, "count coefs")
  )
  expect_equal(coef(fast_model, model = "zero"), coef(pscl_model, model = "zero"),
    tolerance = coef_tol, info = paste(desc, "zero coefs")
  )

  # Log-likelihood (should always match closely, even when parameters differ)
  expect_equal(logLik(fast_model), logLik(pscl_model),
    tolerance = 1e-3, info = paste(desc, "logLik")
  )

  # Fitted values
  expect_equal(fitted(fast_model), fitted(pscl_model),
    tolerance = fitted_tol, info = paste(desc, "fitted")
  )

  # Theta (compare on log scale for stability when theta is large)
  if (count_dist == "negbin") {
    expect_equal(log(fast_model$theta["count"]), log(pscl_model$theta["count"]),
      tolerance = 0.1, info = paste(desc, "count log(theta)")
    )
  }
  if (zero_dist == "negbin") {
    # Zero NB theta can vary widely on flat surface; just check same order of magnitude
    expect_equal(log(fast_model$theta["zero"]), log(pscl_model$theta["zero"]),
      tolerance = 1.0, info = paste(desc, "zero log(theta)")
    )
  }

  # Summary
  fast_summary <- summary(fast_model)
  pscl_summary <- summary(pscl_model)

  expect_equal(
    fast_summary$coefficients$count[, "Estimate"],
    pscl_summary$coefficients$count[, "Estimate"],
    tolerance = 1e-3, info = paste(desc, "summary count estimates")
  )
  expect_equal(
    fast_summary$coefficients$count[, "Std. Error"],
    pscl_summary$coefficients$count[, "Std. Error"],
    tolerance = 1e-2, info = paste(desc, "summary count SEs")
  )
  expect_equal(
    fast_summary$coefficients$zero[, "Estimate"],
    pscl_summary$coefficients$zero[, "Estimate"],
    tolerance = coef_tol, info = paste(desc, "summary zero estimates")
  )
  expect_equal(
    fast_summary$coefficients$zero[, "Std. Error"],
    pscl_summary$coefficients$zero[, "Std. Error"],
    tolerance = se_tol, info = paste(desc, "summary zero SEs")
  )
  expect_equal(fast_summary$loglik, pscl_summary$loglik,
    tolerance = 1e-3, info = paste(desc, "summary loglik")
  )
  expect_equal(AIC(fast_model), AIC(pscl_model),
    tolerance = 1e-3, info = paste(desc, "AIC")
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

  expect_equal(coef(fast_model, model = "count"), coef(pscl_model, model = "count"),
    tolerance = 1e-3
  )
  expect_equal(coef(fast_model, model = "zero"), coef(pscl_model, model = "zero"),
    tolerance = 1e-3
  )
  expect_equal(logLik(fast_model), logLik(pscl_model), tolerance = 1e-3)
  expect_equal(fitted(fast_model), fitted(pscl_model), tolerance = 1e-3)
  expect_equal(fast_model$theta["count"], pscl_model$theta["count"], tolerance = 1e-2)

  fast_summary <- summary(fast_model)
  pscl_summary <- summary(pscl_model)

  expect_equal(
    fast_summary$coefficients$count[, "Estimate"],
    pscl_summary$coefficients$count[, "Estimate"],
    tolerance = 1e-3
  )
  expect_equal(
    fast_summary$coefficients$count[, "Std. Error"],
    pscl_summary$coefficients$count[, "Std. Error"],
    tolerance = 1e-2
  )
  expect_equal(
    fast_summary$coefficients$zero[, "Estimate"],
    pscl_summary$coefficients$zero[, "Estimate"],
    tolerance = 1e-3
  )
  expect_equal(
    fast_summary$coefficients$zero[, "Std. Error"],
    pscl_summary$coefficients$zero[, "Std. Error"],
    tolerance = 1e-2
  )

  expect_length(coef(fast_model, model = "count"), 2)
  expect_length(coef(fast_model, model = "zero"), 3)
})
