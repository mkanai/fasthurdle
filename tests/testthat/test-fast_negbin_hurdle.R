library(testthat)
library(fasthurdle)

test_that("fast_negbin_hurdle produces the same results as fasthurdle with negbin count and binomial zero", {
  # Generate some sample data
  set.seed(123)
  n <- 500
  x <- rnorm(n)
  z <- rnorm(n)
  lambda <- exp(1 + 0.5 * x)
  p <- plogis(0.5 - 0.5 * z)
  y <- rbinom(n, size = 1, prob = p) * rpois(n, lambda = lambda)

  # Create a data frame and model matrix
  df <- data.frame(y = y, x = x, z = z)
  X <- model.matrix(~ x + z, data = df)

  # Fit models with both functions
  fast_model <- fast_negbin_hurdle(X, y)
  regular_model <- fasthurdle(y ~ x + z,
    data = df,
    dist = "negbin",
    zero.dist = "binomial",
    link = "logit"
  )

  # Check that coefficients are identical (same optimization path)
  expect_equal(coef(fast_model, model = "count"),
    coef(regular_model, model = "count"),
    tolerance = 1e-6
  )

  expect_equal(coef(fast_model, model = "zero"),
    coef(regular_model, model = "zero"),
    tolerance = 1e-6
  )

  # Check that log-likelihood is the same
  expect_equal(logLik(fast_model),
    logLik(regular_model),
    tolerance = 1e-6
  )

  # Check that fitted values are the same
  expect_equal(fitted(fast_model),
    fitted(regular_model),
    tolerance = 1e-6
  )

  # Check count theta parameter
  expect_equal(fast_model$theta["count"],
    regular_model$theta["count"],
    tolerance = 1e-6
  )
})

test_that("fast_negbin_hurdle with Z matrix produces the same results as fasthurdle with different formulas", {
  set.seed(42)
  n <- 500
  x <- rnorm(n)
  log_depth <- rnorm(n, mean = 8, sd = 1)

  lambda <- exp(0.5 + 0.3 * x + log_depth)
  p <- plogis(-2 + 0.5 * x + 0.8 * log_depth)
  y <- rbinom(n, size = 1, prob = p) * rnbinom(n, size = 2, mu = lambda)

  df <- data.frame(y = y, x = x, log_depth = log_depth)

  # Fit fast_negbin_hurdle with separate Z and offsetx
  X <- model.matrix(~x, data = df)
  Z <- model.matrix(~ x + log_depth, data = df)
  fast_model <- fast_negbin_hurdle(X, y, Z = Z, offsetx = log_depth)

  # Fit fasthurdle with equivalent formula specification
  regular_model <- fasthurdle(y ~ x + offset(log_depth) | x + log_depth,
    data = df,
    dist = "negbin",
    zero.dist = "binomial",
    link = "logit"
  )

  # Check count coefficients (identical optimization path)
  expect_equal(coef(fast_model, model = "count"),
    coef(regular_model, model = "count"),
    tolerance = 1e-6
  )

  # Check zero coefficients
  expect_equal(coef(fast_model, model = "zero"),
    coef(regular_model, model = "zero"),
    tolerance = 1e-6
  )

  # Check log-likelihood
  expect_equal(logLik(fast_model),
    logLik(regular_model),
    tolerance = 1e-6
  )

  # Check fitted values
  expect_equal(fitted(fast_model),
    fitted(regular_model),
    tolerance = 1e-6
  )

  # Check theta
  expect_equal(fast_model$theta["count"],
    regular_model$theta["count"],
    tolerance = 1e-6
  )

  # Verify dimensions: count has 2 coefs, zero has 3 coefs
  expect_length(coef(fast_model, model = "count"), ncol(X))
  expect_length(coef(fast_model, model = "zero"), ncol(Z))
  expect_equal(nrow(fast_model$vcov), ncol(X) + ncol(Z))
})
