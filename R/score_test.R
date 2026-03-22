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
#' null_fit <- fit_null_count(X_null, y, offsetx = off, dist = "negbin")
#' saveRDS(null_fit, "null_fit_gene1.rds")
#'
#' # Reuse for each peak
#' null_fit <- readRDS("null_fit_gene1.rds")
#' for (peak in peaks) {
#'   model <- fast_negbin_hurdle(X, y,
#'     offsetx = off, score_test = "peak_acc",
#'     null_fit = null_fit
#'   )
#'   results[[peak]] <- model$score_test$pvalue
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
      convergence = null_fit$convergence,
      dist = dist,
      kx_null = ncol(X_null)
    ),
    class = "fasthurdle_null"
  )
}


#' Score Test for Count Component of Hurdle Model
#'
#' @description
#' Computes the score test for one or more predictors in the count component
#' of a hurdle model. The score test evaluates the score statistic at the null
#' MLE, avoiding the beta-theta confounding that can inflate NB Wald tests.
#'
#' @param X_null Model matrix for the null model (intercept + covariates).
#' @param x_test Test variable vector or matrix.
#' @param y Response vector of counts.
#' @param offsetx Optional offset vector. Default is NULL (no offset).
#' @param weights Optional weight vector. Default is NULL (unit weights).
#' @param dist Count distribution: "negbin", "poisson", or "geometric".
#' @param null_fit Optional. A pre-fitted null model from \code{\link{fit_null_count}}.
#'   If provided, the null model is not re-fitted, saving computation time.
#' @param method Optimization method for fitting the null model (ignored if null_fit
#'   is provided). Default is "BFGS".
#' @param maxit Maximum iterations for the null model (ignored if null_fit is provided).
#'   Default is 10000.
#'
#' @return A list with components:
#'   \item{beta}{Ratio estimator of the test coefficient(s): score / information.
#'     Asymptotically equivalent to the MLE but computed at the null.}
#'   \item{se}{Standard error(s): 1 / sqrt(information).}
#'   \item{statistic}{The score test statistic (chi-squared).}
#'   \item{pvalue}{The p-value from chi-squared distribution.}
#'   \item{null_par}{The null model MLE parameters.}
#'   \item{null_convergence}{Convergence status of the null model.}
#'
#' @examples
#' \dontrun{
#' # One-shot usage
#' result <- score_test_count(X_null, x_test, y, dist = "negbin")
#'
#' # With cached null model (for testing many x_test variables)
#' null_fit <- fit_null_count(X_null, y, dist = "negbin")
#' result1 <- score_test_count(X_null, x_test1, y, null_fit = null_fit)
#' result2 <- score_test_count(X_null, x_test2, y, null_fit = null_fit)
#' }
#'
#' @export
score_test_count <- function(X_null, x_test, y, offsetx = NULL, weights = NULL,
                             dist = c("negbin", "poisson", "geometric"),
                             null_fit = NULL, spa_cutoff = 2,
                             method = "BFGS", maxit = 10000) {
  dist <- match.arg(dist)
  n <- length(y)
  if (is.null(offsetx)) offsetx <- rep.int(0, n)
  if (is.null(weights)) weights <- rep.int(1, n)

  # Ensure x_test is a matrix
  if (is.vector(x_test)) x_test <- matrix(x_test, ncol = 1)

  # Build full model matrix
  X_full <- cbind(X_null, x_test)
  if (is.null(colnames(X_full))) {
    colnames(X_full) <- paste0("V", seq_len(ncol(X_full)))
  }

  # Fit or reuse null model
  if (is.null(null_fit)) {
    null_fit <- fit_null_count(X_null, y,
      offsetx = offsetx, weights = weights,
      dist = dist, method = method, maxit = maxit
    )
  } else if (null_fit$dist != dist) {
    stop("null_fit distribution (", null_fit$dist, ") does not match dist (", dist, ")")
  }

  # Resolve SPA: NULL or Inf disables, numeric enables with that cutoff
  use_spa <- !is.null(spa_cutoff) && is.finite(spa_cutoff)
  spa_cutoff_val <- if (use_spa) spa_cutoff else 1e30

  # Compute score test via C++
  result <- score_test_count_cpp(
    null_par = null_fit$par, Y = y, X_null = X_null, X_full = X_full,
    offsetx = offsetx, weights = weights, dist = dist,
    use_spa = use_spa, spa_cutoff = spa_cutoff_val
  )

  result$null_par <- null_fit$par
  result$null_convergence <- null_fit$convergence
  result
}
