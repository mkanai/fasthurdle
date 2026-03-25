#' Fit a Null Count Model for Score Testing
#'
#' @description
#' Fits the null (reduced) count model for use with \code{\link{score_test_count}}.
#' The fitted null can be cached (e.g., via \code{saveRDS}) and reused across
#' multiple test variables, avoiding redundant model fitting.
#'
#' @param X_null Model matrix for the null model (intercept + covariates, no test variable).
#' @param y Response vector of counts.
#' @param offsetx Optional offset vector. Default is NULL (no offset).
#' @param weights Optional weight vector. Default is NULL (unit weights).
#' @param dist Count distribution: "negbin", "poisson", or "geometric".
#' @param method Optimization method. Default is "BFGS".
#' @param maxit Maximum iterations. Default is 10000.
#'
#' @return An object of class "fasthurdle_null" containing the null MLE parameters,
#'   convergence status, and metadata needed for score testing. This object can be
#'   saved with \code{saveRDS} and reloaded for reuse.
#'
#' @examples
#' \dontrun{
#' # Fit null model once per gene (covariates only)
#' null_fit_count <- fit_null_count(X_null, y, offsetx = off, dist = "negbin")
#' saveRDS(null_fit_count, "null_fit_gene1.rds")
#'
#' # Reuse for each peak
#' null_fit_count <- readRDS("null_fit_gene1.rds")
#' for (peak in peaks) {
#'   model <- fast_negbin_hurdle(X, y,
#'     offsetx = off, score_test = "peak_acc",
#'     null_fit_count = null_fit_count
#'   )
#'   results[[peak]] <- model$score_test_count$pvalue
#' }
#' }
#'
#' @export
fit_null_count <- function(X_null, y, offsetx = NULL, weights = NULL,
                           dist = c("negbin", "poisson", "geometric"),
                           method = "BFGS", maxit = 10000) {
  dist <- match.arg(dist)
  n <- length(y)
  if (is.null(offsetx)) offsetx <- rep.int(0, n)
  if (is.null(weights)) weights <- rep.int(1, n)

  reltol <- .Machine$double.eps^(1 / 1.6)
  pos_idx <- y > 0
  pos_mean <- if (any(pos_idx)) mean(log(y[pos_idx] + 0.5) - offsetx[pos_idx]) else 0

  null_fit <- switch(dist,
    "negbin" = optim_count_negbin_cpp(
      start = c(pos_mean, rep(0, ncol(X_null) - 1), log(2)),
      Y = y, X = X_null, offsetx = offsetx, weights = weights,
      method = method, hessian = FALSE, maxit = maxit, reltol = reltol
    ),
    "poisson" = optim_count_poisson_cpp(
      start = c(pos_mean, rep(0, ncol(X_null) - 1)),
      Y = y, X = X_null, offsetx = offsetx, weights = weights,
      method = method, hessian = FALSE, maxit = maxit, reltol = reltol
    ),
    "geometric" = optim_count_geom_cpp(
      start = c(pos_mean, rep(0, ncol(X_null) - 1)),
      Y = y, X = X_null, offsetx = offsetx, weights = weights,
      method = method, hessian = FALSE, maxit = maxit, reltol = reltol
    )
  )

  structure(
    list(
      par = null_fit$par,
      value = null_fit$value,
      convergence = null_fit$convergence,
      dist = dist,
      kx_null = ncol(X_null)
    ),
    class = "fasthurdle_null"
  )
}

#' Fit Null Zero Model for Caching
#'
#' @description
#' Fits the zero model (binomial/logit) without the test variable, for reuse
#' across multiple score tests on the same gene. Only logit link is supported.
#'
#' @param Z_null Zero model design matrix without the test variable.
#' @param y Response vector of counts.
#' @param offsetz Optional offset vector for zero model. Default is NULL.
#' @param weights Optional weight vector. Default is NULL (unit weights).
#' @param method Optimization method. Default is "BFGS".
#' @param maxit Maximum iterations. Default is 10000.
#'
#' @return An object of class "fasthurdle_null_zero" for use with
#'   \code{fast_negbin_hurdle(..., null_fit_zero = ...)}.
#'
#' @export
fit_null_zero <- function(Z_null, y, offsetz = NULL, weights = NULL,
                          method = "BFGS", maxit = 10000) {
  n <- length(y)
  if (is.null(offsetz)) offsetz <- rep.int(0, n)
  if (is.null(weights)) weights <- rep.int(1, n)

  reltol <- .Machine$double.eps^(1 / 1.6)

  # Starting values from logistic GLM
  y_bin <- as.integer(y > 0)
  glm_start <- tryCatch(
    suppressWarnings(fastglm::fastglm(
      Z_null, y_bin,
      family = binomial(link = "logit"),
      weights = weights, offset = offsetz
    )),
    error = function(e) NULL
  )
  start <- if (!is.null(glm_start) && all(is.finite(glm_start$coefficients))) {
    glm_start$coefficients
  } else {
    rep(0, ncol(Z_null))
  }

  null_fit <- optim_zero_binom_cpp(
    start = start, Y = y, X = Z_null, offsetx = offsetz, weights = weights,
    link = "logit", method = method, hessian = FALSE, maxit = maxit,
    reltol = reltol
  )

  structure(
    list(
      par = null_fit$par,
      value = null_fit$value,
      convergence = null_fit$convergence,
      kz_null = ncol(Z_null),
      link = "logit"
    ),
    class = "fasthurdle_null_zero"
  )
}

#' Score Test for Zero (Binomial/Logit) Component
#'
#' @param Z_null Zero model design matrix without the test variable.
#' @param z_test Test variable vector or single-column matrix.
#' @param y Response vector.
#' @param offsetz Optional offset vector. Default is NULL.
#' @param weights Optional weight vector. Default is NULL.
#' @param null_fit_zero Optional cached null zero model from \code{fit_null_zero}.
#' @param spa_cutoff SPA cutoff. Default is NULL (disabled). Set to 2 to enable
#'   SPA for improved tail accuracy at small sample sizes.
#' @param method Optimization method for null model. Default is "BFGS".
#' @param maxit Maximum iterations for null model. Default is 10000.
#'
#' @return A list with beta, se, statistic, pvalue, spa_applied.
#'
#' @export
score_test_zero <- function(Z_null, z_test, y, offsetz = NULL, weights = NULL,
                            null_fit_zero = NULL, spa_cutoff = NULL,
                            method = "BFGS", maxit = 10000) {
  n <- length(y)
  if (is.null(offsetz)) offsetz <- rep.int(0, n)
  if (is.null(weights)) weights <- rep.int(1, n)

  if (is.vector(z_test)) z_test <- matrix(z_test, ncol = 1)
  if (ncol(z_test) != 1) {
    stop("score_test_zero currently supports only a single test variable")
  }
  Z_full <- cbind(Z_null, z_test)
  if (is.null(colnames(Z_full))) {
    colnames(Z_full) <- paste0("V", seq_len(ncol(Z_full)))
  }

  # Fit or reuse null model
  if (is.null(null_fit_zero)) {
    null_fit_zero <- fit_null_zero(Z_null, y,
      offsetz = offsetz, weights = weights,
      method = method, maxit = maxit
    )
  } else {
    if (null_fit_zero$link != "logit") {
      stop("Zero score test only supports logit link")
    }
    if (null_fit_zero$kz_null != ncol(Z_null)) {
      stop(
        "null_fit_zero has ", null_fit_zero$kz_null,
        " covariates but Z_null has ", ncol(Z_null)
      )
    }
  }

  if (null_fit_zero$convergence != 0) {
    warning("Zero null model did not converge; returning NA")
    return(list(
      beta = NA_real_, se = NA_real_, statistic = NA_real_,
      pvalue = NA_real_, spa_applied = FALSE,
      null_par = null_fit_zero$par, null_convergence = null_fit_zero$convergence
    ))
  }

  use_spa <- !is.null(spa_cutoff) && is.finite(spa_cutoff)
  spa_cutoff_val <- if (use_spa) spa_cutoff else 1e30

  result <- score_test_zero_cpp(
    null_par = null_fit_zero$par, Y = y, Z_null = Z_null, Z_full = Z_full,
    offsetz = offsetz, weights = weights,
    use_spa = use_spa, spa_cutoff = spa_cutoff_val
  )

  result$null_par <- null_fit_zero$par
  result$null_convergence <- null_fit_zero$convergence
  result
}


#' Score Test for Count Component of Hurdle Model
#'
#' @description
#' Computes the score test for a single predictor in the count component
#' of a hurdle model. The score test evaluates the score statistic at the null
#' MLE using the observed information (negative Hessian) instead of the expected
#' Fisher information, making it robust to model misspecification.
#'
#' @param X_null Model matrix for the null model (intercept + covariates).
#' @param x_test Test variable vector or single-column matrix.
#' @param y Response vector of counts.
#' @param offsetx Optional offset vector. Default is NULL (no offset).
#' @param weights Optional weight vector. Default is NULL (unit weights).
#' @param dist Count distribution: "negbin", "poisson", or "geometric".
#' @param null_fit_count Optional. A pre-fitted null model from \code{\link{fit_null_count}}.
#'   If provided, the null model is not re-fitted, saving computation time.
#' @param spa_cutoff Numeric or NULL. Apply saddlepoint approximation (SPA) for
#'   p-values when |z| exceeds this cutoff. Default is \code{NULL} (disabled).
#'   Set to \code{2} to enable SPA for improved tail accuracy at small sample sizes.
#' @param method Optimization method for fitting the null model (ignored if null_fit_count
#'   is provided). Default is "BFGS".
#' @param maxit Maximum iterations for the null model (ignored if null_fit_count is provided).
#'   Default is 10000.
#'
#' @return A list with components:
#'   \item{beta}{Effect size estimate. For significant tests (|z| > \code{spa_cutoff},
#'     or |z| > 2 when SPA is disabled), refined via 5-iteration BFGS from the score
#'     estimate (within ~3\% of full MLE). For non-significant tests, uses the ratio
#'     estimator (approximate).}
#'   \item{se}{Standard error, back-computed from the p-value for consistency.}
#'   \item{statistic}{The score test statistic (chi-squared, using observed information).}
#'   \item{pvalue}{The p-value. SPA-adjusted when |z| > \code{spa_cutoff}.}
#'   \item{spa_applied}{Logical. Whether SPA was applied.}
#'   \item{null_par}{The null model MLE parameters.}
#'   \item{null_convergence}{Convergence status of the null model.}
#'
#' @examples
#' \dontrun{
#' # One-shot usage
#' result <- score_test_count(X_null, x_test, y, dist = "negbin")
#'
#' # With cached null model (for testing many x_test variables)
#' null_fit_count <- fit_null_count(X_null, y, dist = "negbin")
#' result1 <- score_test_count(X_null, x_test1, y, null_fit_count = null_fit_count)
#' result2 <- score_test_count(X_null, x_test2, y, null_fit_count = null_fit_count)
#' }
#'
#' @export
score_test_count <- function(X_null, x_test, y, offsetx = NULL, weights = NULL,
                             dist = c("negbin", "poisson", "geometric"),
                             null_fit_count = NULL, spa_cutoff = NULL,
                             method = "BFGS", maxit = 10000) {
  dist <- match.arg(dist)
  n <- length(y)
  if (is.null(offsetx)) offsetx <- rep.int(0, n)
  if (is.null(weights)) weights <- rep.int(1, n)

  # Ensure x_test is a single-column matrix
  if (is.vector(x_test)) x_test <- matrix(x_test, ncol = 1)
  if (ncol(x_test) != 1) {
    stop("score_test_count currently supports only a single test variable")
  }

  # Build full model matrix
  X_full <- cbind(X_null, x_test)
  if (is.null(colnames(X_full))) {
    colnames(X_full) <- paste0("V", seq_len(ncol(X_full)))
  }

  # Fit or reuse null model
  if (is.null(null_fit_count)) {
    null_fit_count <- fit_null_count(X_null, y,
      offsetx = offsetx, weights = weights,
      dist = dist, method = method, maxit = maxit
    )
  } else {
    if (null_fit_count$dist != dist) {
      stop("null_fit_count distribution (", null_fit_count$dist, ") does not match dist (", dist, ")")
    }
    if (null_fit_count$kx_null != ncol(X_null)) {
      stop("null_fit_count has ", null_fit_count$kx_null, " covariates but X_null has ", ncol(X_null))
    }
  }

  # Resolve SPA: NULL or Inf disables, numeric enables with that cutoff
  use_spa <- !is.null(spa_cutoff) && is.finite(spa_cutoff)
  spa_cutoff_val <- if (use_spa) spa_cutoff else 1e30

  # Compute score test via C++
  result <- score_test_count_cpp(
    null_par = null_fit_count$par, Y = y, X_null = X_null, X_full = X_full,
    offsetx = offsetx, weights = weights, dist = dist,
    use_spa = use_spa, spa_cutoff = spa_cutoff_val
  )

  result$null_par <- null_fit_count$par
  result$null_convergence <- null_fit_count$convergence
  result
}
