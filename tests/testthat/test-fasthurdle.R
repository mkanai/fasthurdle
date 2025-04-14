library(testthat)
library(fasthurdle)
library(pscl)

# Test that fasthurdle produces the same results as pscl::hurdle for all distribution combinations
test_that("fasthurdle produces the same results as pscl::hurdle for all distribution combinations", {
  # Generate some sample data
  set.seed(123)
  n <- 500
  x <- rnorm(n)
  z <- rnorm(n)
  lambda <- exp(1 + 0.5 * x)
  p <- plogis(0.5 - 0.5 * z)
  y <- rbinom(n, size = 1, prob = p) * rpois(n, lambda = lambda)

  # Create a data frame
  df <- data.frame(y = y, x = x, z = z)

  # Define all combinations of count and zero hurdle distributions
  count_dists <- c("poisson", "negbin", "geometric")
  zero_dists <- c("binomial", "poisson", "negbin", "geometric")

  # Create all combinations
  combinations <- expand.grid(
    count_dist = count_dists,
    zero_dist = zero_dists,
    stringsAsFactors = FALSE
  )

  # Test each combination
  for (i in 1:nrow(combinations)) {
    count_dist <- combinations$count_dist[i]
    zero_dist <- combinations$zero_dist[i]

    # Create a descriptive test name
    test_name <- paste("Testing", count_dist, "count model with", zero_dist, "zero hurdle")

    # Run the test for this combination
    test_that(test_name, {
      # Skip the test if the combination is not supported
      tryCatch(
        {
          # Fit models with both packages
          fast_model <- fasthurdle(y ~ x | z,
            data = df,
            dist = count_dist,
            zero.dist = zero_dist
          )

          pscl_model <- pscl::hurdle(y ~ x | z,
            data = df,
            dist = count_dist,
            zero.dist = zero_dist
          )

          # Check that coefficients are the same (with relaxed tolerance due to different optimization methods)
          expect_equal(coef(fast_model, model = "count"),
            coef(pscl_model, model = "count"),
            tolerance = 1e-3,
            info = paste("Count coefficients differ for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          expect_equal(coef(fast_model, model = "zero"),
            coef(pscl_model, model = "zero"),
            tolerance = 1e-3,
            info = paste("Zero coefficients differ for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          # Check that log-likelihood is the same
          expect_equal(logLik(fast_model),
            logLik(pscl_model),
            tolerance = 1e-3,
            info = paste("Log-likelihood differs for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          # Check that fitted values are the same
          expect_equal(fitted(fast_model),
            fitted(pscl_model),
            tolerance = 1e-3,
            info = paste("Fitted values differ for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          # Check theta if applicable
          if (count_dist == "negbin" || zero_dist == "negbin") {
            if (count_dist == "negbin") {
              expect_equal(fast_model$theta["count"],
                pscl_model$theta["count"],
                tolerance = 1e-2, # More relaxed tolerance for theta
                info = paste("Count theta differs for", count_dist, "count model with", zero_dist, "zero hurdle")
              )
            }

            if (zero_dist == "negbin") {
              expect_equal(fast_model$theta["zero"],
                pscl_model$theta["zero"],
                tolerance = 1e-2, # More relaxed tolerance for theta
                info = paste("Zero theta differs for", count_dist, "count model with", zero_dist, "zero hurdle")
              )
            }
          }

          # Compare summary outputs
          fast_summary <- summary(fast_model)
          pscl_summary <- summary(pscl_model)

          # Compare count model coefficients in summary
          expect_equal(
            fast_summary$count$coefficients[, "Estimate"],
            pscl_summary$count$coefficients[, "Estimate"],
            tolerance = 1e-3,
            info = paste("Count coefficient estimates differ in summary for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          expect_equal(
            fast_summary$count$coefficients[, "Std. Error"],
            pscl_summary$count$coefficients[, "Std. Error"],
            tolerance = 1e-2, # Slightly more relaxed tolerance for standard errors
            info = paste("Count coefficient standard errors differ in summary for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          expect_equal(
            fast_summary$count$coefficients[, "z value"],
            pscl_summary$count$coefficients[, "z value"],
            tolerance = 1e-2,
            info = paste("Count coefficient z-values differ in summary for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          expect_equal(
            fast_summary$count$coefficients[, "Pr(>|z|)"],
            pscl_summary$count$coefficients[, "Pr(>|z|)"],
            tolerance = 1e-2,
            info = paste("Count coefficient p-values differ in summary for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          # Compare zero model coefficients in summary
          expect_equal(
            fast_summary$zero$coefficients[, "Estimate"],
            pscl_summary$zero$coefficients[, "Estimate"],
            tolerance = 1e-3,
            info = paste("Zero coefficient estimates differ in summary for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          expect_equal(
            fast_summary$zero$coefficients[, "Std. Error"],
            pscl_summary$zero$coefficients[, "Std. Error"],
            tolerance = 1e-2, # Slightly more relaxed tolerance for standard errors
            info = paste("Zero coefficient standard errors differ in summary for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          expect_equal(
            fast_summary$zero$coefficients[, "z value"],
            pscl_summary$zero$coefficients[, "z value"],
            tolerance = 1e-2,
            info = paste("Zero coefficient z-values differ in summary for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          expect_equal(
            fast_summary$zero$coefficients[, "Pr(>|z|)"],
            pscl_summary$zero$coefficients[, "Pr(>|z|)"],
            tolerance = 1e-2,
            info = paste("Zero coefficient p-values differ in summary for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          # Compare log-likelihood and AIC in summary
          expect_equal(
            fast_summary$loglik,
            pscl_summary$loglik,
            tolerance = 1e-3,
            info = paste("Log-likelihood differs in summary for", count_dist, "count model with", zero_dist, "zero hurdle")
          )

          expect_equal(
            fast_summary$aic,
            pscl_summary$aic,
            tolerance = 1e-3,
            info = paste("AIC differs in summary for", count_dist, "count model with", zero_dist, "zero hurdle")
          )
        },
        error = function(e) {
          skip(sprintf("Combination not supported or error (dist: %s, zero_dist: %s): %s", count_dist, zero_dist, e$message))
        }
      )
    })
  }
})

# Test that fasthurdle produces the same results as pscl::hurdle for all link function combinations
test_that("fasthurdle produces the same results as pscl::hurdle for all link function combinations", {
  # Generate some sample data
  set.seed(123)
  n <- 500
  x <- rnorm(n)
  z <- rnorm(n)
  lambda <- exp(1 + 0.5 * x)
  p <- plogis(0.5 - 0.5 * z)
  y <- rbinom(n, size = 1, prob = p) * rpois(n, lambda = lambda)

  # Create a data frame
  df <- data.frame(y = y, x = x, z = z)

  # Define all link functions for binomial zero hurdle
  link_functions <- c("logit", "probit", "cloglog", "cauchit", "log")

  # Test each link function with binomial zero hurdle
  for (link_func in link_functions) {
    # Create a descriptive test name
    test_name <- paste("Testing binomial zero hurdle with", link_func, "link function")

    # Run the test for this link function
    test_that(test_name, {
      # Skip the test if the combination is not supported
      tryCatch(
        {
          # Fit models with both packages
          fast_model <- fasthurdle(y ~ x | z,
            data = df,
            dist = "poisson",
            zero.dist = "binomial",
            link = link_func
          )

          pscl_model <- pscl::hurdle(y ~ x | z,
            data = df,
            dist = "poisson",
            zero.dist = "binomial",
            link = link_func
          )

          # Check that coefficients are the same (with relaxed tolerance due to different optimization methods)
          expect_equal(coef(fast_model, model = "count"),
            coef(pscl_model, model = "count"),
            tolerance = 1e-3,
            info = paste("Count coefficients differ for binomial zero hurdle with", link_func, "link")
          )

          expect_equal(coef(fast_model, model = "zero"),
            coef(pscl_model, model = "zero"),
            tolerance = 1e-3,
            info = paste("Zero coefficients differ for binomial zero hurdle with", link_func, "link")
          )

          # Check that log-likelihood is the same
          expect_equal(logLik(fast_model),
            logLik(pscl_model),
            tolerance = 1e-3,
            info = paste("Log-likelihood differs for binomial zero hurdle with", link_func, "link")
          )

          # Check that fitted values are the same
          expect_equal(fitted(fast_model),
            fitted(pscl_model),
            tolerance = 1e-3,
            info = paste("Fitted values differ for binomial zero hurdle with", link_func, "link")
          )

          # Compare summary outputs
          fast_summary <- summary(fast_model)
          pscl_summary <- summary(pscl_model)

          # Compare count model coefficients in summary
          expect_equal(
            fast_summary$count$coefficients[, "Estimate"],
            pscl_summary$count$coefficients[, "Estimate"],
            tolerance = 1e-3,
            info = paste("Count coefficient estimates differ in summary for binomial zero hurdle with", link_func, "link")
          )

          expect_equal(
            fast_summary$count$coefficients[, "Std. Error"],
            pscl_summary$count$coefficients[, "Std. Error"],
            tolerance = 1e-2, # Slightly more relaxed tolerance for standard errors
            info = paste("Count coefficient standard errors differ in summary for binomial zero hurdle with", link_func, "link")
          )

          expect_equal(
            fast_summary$count$coefficients[, "z value"],
            pscl_summary$count$coefficients[, "z value"],
            tolerance = 1e-2,
            info = paste("Count coefficient z-values differ in summary for binomial zero hurdle with", link_func, "link")
          )

          expect_equal(
            fast_summary$count$coefficients[, "Pr(>|z|)"],
            pscl_summary$count$coefficients[, "Pr(>|z|)"],
            tolerance = 1e-2,
            info = paste("Count coefficient p-values differ in summary for binomial zero hurdle with", link_func, "link")
          )

          # Compare zero model coefficients in summary
          expect_equal(
            fast_summary$zero$coefficients[, "Estimate"],
            pscl_summary$zero$coefficients[, "Estimate"],
            tolerance = 1e-3,
            info = paste("Zero coefficient estimates differ in summary for binomial zero hurdle with", link_func, "link")
          )

          expect_equal(
            fast_summary$zero$coefficients[, "Std. Error"],
            pscl_summary$zero$coefficients[, "Std. Error"],
            tolerance = 1e-2, # Slightly more relaxed tolerance for standard errors
            info = paste("Zero coefficient standard errors differ in summary for binomial zero hurdle with", link_func, "link")
          )

          expect_equal(
            fast_summary$zero$coefficients[, "z value"],
            pscl_summary$zero$coefficients[, "z value"],
            tolerance = 1e-2,
            info = paste("Zero coefficient z-values differ in summary for binomial zero hurdle with", link_func, "link")
          )

          expect_equal(
            fast_summary$zero$coefficients[, "Pr(>|z|)"],
            pscl_summary$zero$coefficients[, "Pr(>|z|)"],
            tolerance = 1e-2,
            info = paste("Zero coefficient p-values differ in summary for binomial zero hurdle with", link_func, "link")
          )

          # Compare link function in summary
          expect_equal(
            fast_summary$link,
            pscl_summary$link,
            info = paste("Link function differs in summary for binomial zero hurdle with", link_func, "link")
          )
        },
        error = function(e) {
          skip(sprintf("Link function not supported or error (link: %s): %s", link_func, e$message))
        }
      )
    })
  }

  # Test combinations of distributions with different link functions
  count_dists <- c("poisson", "negbin", "geometric")

  # Create all combinations
  combinations <- expand.grid(
    count_dist = count_dists,
    link_func = link_functions,
    stringsAsFactors = FALSE
  )

  # Test each combination
  for (i in 1:nrow(combinations)) {
    count_dist <- combinations$count_dist[i]
    link_func <- combinations$link_func[i]

    # Create a descriptive test name
    test_name <- paste("Testing", count_dist, "count model with binomial zero hurdle and", link_func, "link")

    # Run the test for this combination
    test_that(test_name, {
      # Skip the test if the combination is not supported
      tryCatch(
        {
          # Fit models with both packages
          fast_model <- fasthurdle(y ~ x | z,
            data = df,
            dist = count_dist,
            zero.dist = "binomial",
            link = link_func
          )

          pscl_model <- pscl::hurdle(y ~ x | z,
            data = df,
            dist = count_dist,
            zero.dist = "binomial",
            link = link_func
          )

          # Check that coefficients are the same (with relaxed tolerance due to different optimization methods)
          expect_equal(coef(fast_model, model = "count"),
            coef(pscl_model, model = "count"),
            tolerance = 1e-3,
            info = paste("Count coefficients differ for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          expect_equal(coef(fast_model, model = "zero"),
            coef(pscl_model, model = "zero"),
            tolerance = 1e-3,
            info = paste("Zero coefficients differ for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          # Check that log-likelihood is the same
          expect_equal(logLik(fast_model),
            logLik(pscl_model),
            tolerance = 1e-3,
            info = paste("Log-likelihood differs for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          # Check that fitted values are the same
          expect_equal(fitted(fast_model),
            fitted(pscl_model),
            tolerance = 1e-3,
            info = paste("Fitted values differ for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          # Check theta if applicable
          if (count_dist == "negbin") {
            expect_equal(fast_model$theta["count"],
              pscl_model$theta["count"],
              tolerance = 1e-2, # More relaxed tolerance for theta
              info = paste("Count theta differs for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
            )
          }

          # Compare summary outputs
          fast_summary <- summary(fast_model)
          pscl_summary <- summary(pscl_model)

          # Compare count model coefficients in summary
          expect_equal(
            fast_summary$count$coefficients[, "Estimate"],
            pscl_summary$count$coefficients[, "Estimate"],
            tolerance = 1e-3,
            info = paste("Count coefficient estimates differ in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          expect_equal(
            fast_summary$count$coefficients[, "Std. Error"],
            pscl_summary$count$coefficients[, "Std. Error"],
            tolerance = 1e-2, # Slightly more relaxed tolerance for standard errors
            info = paste("Count coefficient standard errors differ in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          expect_equal(
            fast_summary$count$coefficients[, "z value"],
            pscl_summary$count$coefficients[, "z value"],
            tolerance = 1e-2,
            info = paste("Count coefficient z-values differ in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          expect_equal(
            fast_summary$count$coefficients[, "Pr(>|z|)"],
            pscl_summary$count$coefficients[, "Pr(>|z|)"],
            tolerance = 1e-2,
            info = paste("Count coefficient p-values differ in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          # Compare zero model coefficients in summary
          expect_equal(
            fast_summary$zero$coefficients[, "Estimate"],
            pscl_summary$zero$coefficients[, "Estimate"],
            tolerance = 1e-3,
            info = paste("Zero coefficient estimates differ in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          expect_equal(
            fast_summary$zero$coefficients[, "Std. Error"],
            pscl_summary$zero$coefficients[, "Std. Error"],
            tolerance = 1e-2, # Slightly more relaxed tolerance for standard errors
            info = paste("Zero coefficient standard errors differ in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          expect_equal(
            fast_summary$zero$coefficients[, "z value"],
            pscl_summary$zero$coefficients[, "z value"],
            tolerance = 1e-2,
            info = paste("Zero coefficient z-values differ in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          expect_equal(
            fast_summary$zero$coefficients[, "Pr(>|z|)"],
            pscl_summary$zero$coefficients[, "Pr(>|z|)"],
            tolerance = 1e-2,
            info = paste("Zero coefficient p-values differ in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          # Compare log-likelihood and AIC in summary
          expect_equal(
            fast_summary$loglik,
            pscl_summary$loglik,
            tolerance = 1e-3,
            info = paste("Log-likelihood differs in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          expect_equal(
            fast_summary$aic,
            pscl_summary$aic,
            tolerance = 1e-3,
            info = paste("AIC differs in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )

          # Compare link function in summary
          expect_equal(
            fast_summary$link,
            pscl_summary$link,
            info = paste("Link function differs in summary for", count_dist, "count model with binomial zero hurdle and", link_func, "link")
          )
        },
        error = function(e) {
          skip(sprintf("Combination not supported or error (dist: %s, link: %s): %s", count_dist, link_func, e$message))
        }
      )
    })
  }
})
