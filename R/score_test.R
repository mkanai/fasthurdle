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
      converged = null_fit$convergence == 0,
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
      converged = null_fit$convergence == 0,
      kz_null = ncol(Z_null),
      link = "logit"
    ),
    class = "fasthurdle_null_zero"
  )
}

#' Prepare cached quantities for fast per-peak zero score tests
#'
#' Pre-computes null FIM inverse, score weights, and SPA intermediates for
#' the zero (binomial/logistic) component. Analogous to
#' \code{\link{prepare_score_cache_count}} for the count component.
#'
#' @param null_fit_zero Fitted null model from \code{\link{fit_null_zero}}.
#' @param y Response vector.
#' @param Z_null Null zero model matrix.
#' @param offsetz Offset vector.
#' @param weights Weight vector.
#' @return The null_fit_zero object with an attached score_cache element.
#' @export
prepare_score_cache_zero <- function(null_fit_zero, y, Z_null, offsetz = NULL,
                                     weights = NULL) {
  n <- length(y)
  if (is.null(offsetz)) offsetz <- rep.int(0, n)
  if (is.null(weights)) weights <- rep.int(1, n)
  cache <- prepare_score_cache_zero_cpp(
    null_fit_zero$par, y, Z_null, offsetz, weights
  )
  if (isTRUE(cache$valid)) {
    null_fit_zero$score_cache <- cache
  }
  null_fit_zero
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

  # Prepare cache if not present
  if (is.null(null_fit_zero$score_cache) ||
    !isTRUE(null_fit_zero$score_cache$valid)) {
    null_fit_zero <- prepare_score_cache_zero(
      null_fit_zero, y, Z_null,
      offsetz = offsetz, weights = weights
    )
  }
  sc <- null_fit_zero$score_cache

  if (is.null(sc) || !isTRUE(sc$valid)) {
    warning("Zero score cache invalid (null FIM singular; complete separation?); returning NA")
    return(list(
      beta = NA_real_, se = NA_real_, statistic = NA_real_,
      pvalue = NA_real_, spa_applied = FALSE,
      null_par = null_fit_zero$par, null_convergence = null_fit_zero$convergence
    ))
  }

  z_vec <- as.numeric(z_test)
  result <- score_test_zero_cpp(
    z_vec, sc$W_resid, sc$W_diag, sc$I_nn_inv, sc$Znull_W_t,
    sc$p_null, y, Z_null, offsetz, weights, null_fit_zero$par, sc$kz_null,
    use_spa, spa_cutoff_val
  )
  result$null_par <- null_fit_zero$par
  result$null_convergence <- null_fit_zero$convergence
  result
}


#' Prepare cached quantities for fast per-peak count score tests
#'
#' Pre-computes null-only Hessian weights, FIM inverse, and SPA intermediates.
#' When stored in the null_fit_count object, subsequent score_test_count calls
#' use the cached quantities to avoid redundant O(n_pos) computation per peak.
#' Supports all count distributions (negbin, poisson, geometric) via ZTNB
#' formulas with appropriate theta parameterization.
#'
#' @param null_fit_count Fitted null model from \code{\link{fit_null_count}}.
#' @param y Response vector.
#' @param X_null Null model matrix.
#' @param offsetx Offset vector.
#' @param weights Weight vector.
#' @return The null_fit_count object with an attached score_cache element.
#' @export
prepare_score_cache_count <- function(null_fit_count, y, X_null, offsetx = NULL,
                                      weights = NULL) {
  n <- length(y)
  if (is.null(offsetx)) offsetx <- rep.int(0, n)
  if (is.null(weights)) weights <- rep.int(1, n)
  cache <- prepare_score_cache_count_cpp(
    null_fit_count$par, y, X_null, offsetx, weights,
    dist = null_fit_count$dist
  )
  if (isTRUE(cache$valid)) {
    null_fit_count$score_cache <- cache
  }
  null_fit_count
}

#' Score test for the count component of a hurdle model
#'
#' Tests whether a single test variable has a significant effect on the count
#' component, using a score test with observed information. Uses pre-cached
#' null quantities for O(n_pos) per-peak computation. Supports all count
#' distributions (negbin, poisson, geometric).
#'
#' @param X_null Null model matrix (n x kx_null).
#' @param x_test Test variable vector (length n).
#' @param y Response vector.
#' @param offsetx Offset vector (default: zeros).
#' @param weights Weight vector (default: ones).
#' @param dist Count distribution: \code{"negbin"}, \code{"poisson"}, or
#'   \code{"geometric"}.
#' @param null_fit_count Fitted null model from \code{\link{fit_null_count}}.
#'   If NULL, fitted internally.
#' @param spa_cutoff Saddlepoint approximation cutoff. NULL or Inf disables SPA.
#' @param method Optimization method for null model fitting (default: "BFGS").
#' @param maxit Maximum iterations for null model fitting.
#' @return A list with components \code{beta}, \code{se}, \code{statistic},
#'   \code{pvalue}, \code{spa_applied}, \code{null_par}, \code{null_convergence}.
#' @export
score_test_count <- function(X_null, x_test, y, offsetx = NULL, weights = NULL,
                             dist = c("negbin", "poisson", "geometric"),
                             null_fit_count = NULL, spa_cutoff = NULL,
                             method = "BFGS", maxit = 10000) {
  dist <- match.arg(dist)
  n <- length(y)
  if (is.null(offsetx)) offsetx <- rep.int(0, n)
  if (is.null(weights)) weights <- rep.int(1, n)

  # Ensure x_test is a single test variable
  if (is.matrix(x_test) || is.data.frame(x_test)) {
    if (ncol(x_test) != 1) {
      stop("score_test_count currently supports only a single test variable")
    }
    x_test <- as.numeric(x_test)
  }

  # Fit or reuse null model
  if (is.null(null_fit_count)) {
    null_fit_count <- fit_null_count(X_null, y,
      offsetx = offsetx, weights = weights,
      dist = dist, method = method, maxit = maxit
    )
  } else {
    if (null_fit_count$dist != dist) {
      stop(
        "null_fit_count distribution (", null_fit_count$dist,
        ") does not match dist (", dist, ")"
      )
    }
    if (null_fit_count$kx_null != ncol(X_null)) {
      stop(
        "null_fit_count has ", null_fit_count$kx_null,
        " covariates but X_null has ", ncol(X_null)
      )
    }
  }

  # Prepare cache if not already present (works for all distributions)
  if (is.null(null_fit_count$score_cache) ||
    !isTRUE(null_fit_count$score_cache$valid)) {
    null_fit_count <- prepare_score_cache_count(null_fit_count, y, X_null,
      offsetx = offsetx, weights = weights
    )
  }
  sc <- null_fit_count$score_cache
  if (is.null(sc) || !isTRUE(sc$valid)) {
    warning("Count score cache invalid (null FIM singular?); returning NA")
    return(list(
      beta = NA_real_, se = NA_real_, statistic = NA_real_,
      pvalue = NA_real_, spa_applied = FALSE,
      null_par = null_fit_count$par, null_convergence = null_fit_count$convergence
    ))
  }

  # Resolve SPA: NULL or Inf disables, numeric enables with that cutoff
  use_spa <- !is.null(spa_cutoff) && is.finite(spa_cutoff)
  spa_cutoff_val <- if (use_spa) spa_cutoff else 1e30

  # Unified cached path for all distributions
  has_theta <- isTRUE(sc$has_theta)
  result <- score_test_count_cpp(
    x_test = x_test, Y1 = sc$Y1,
    grad_weights = sc$grad_weights, v_ee = sc$v_ee,
    Y_pos = sc$Y_pos,
    I_nn_inv = sc$I_nn_inv, I_nn_beta_inv = sc$I_nn_beta_inv,
    beta_inv_ok = sc$beta_inv_ok,
    Xnull_vee_t = sc$Xnull_vee_t, X_null_pos = sc$X_null_pos,
    w_pos = sc$w_pos, theta = sc$theta,
    beta_null = sc$beta_null, eta_null_pos = sc$eta_null_pos,
    mu_pos = sc$mu_pos, p0_pos = sc$p0_pos, log_p1_pos = sc$log_p1_pos,
    kx_null = sc$kx_null, has_theta = has_theta,
    use_spa = use_spa, spa_cutoff = spa_cutoff_val,
    v_et_nullable = if (has_theta) sc$v_et else NULL
  )
  result$null_par <- null_fit_count$par
  result$null_convergence <- null_fit_count$convergence
  result
}

# ============================================================================
# Internal helpers: ensure null models are fitted and cached
# ============================================================================

# Ensure count null is fitted with score cache.
# Returns a ready-to-use null fit object.
ensure_null_count <- function(null_fit_count, X_null, y,
                              offsetx = NULL, weights = NULL,
                              dist = "negbin") {
  n <- length(y)
  if (is.null(offsetx)) offsetx <- rep.int(0, n)
  if (is.null(weights)) weights <- rep.int(1, n)

  if (is.null(null_fit_count) ||
    !inherits(null_fit_count, "fasthurdle_null")) {
    null_fit_count <- fit_null_count(
      X_null, y,
      offsetx = offsetx, weights = weights,
      dist = dist
    )
  }
  if (is.null(null_fit_count$score_cache)) {
    null_fit_count <- prepare_score_cache_count(
      null_fit_count, y, X_null,
      offsetx = offsetx, weights = weights
    )
  }
  null_fit_count
}

# Ensure zero null is fitted with score cache.
ensure_null_zero <- function(null_fit_zero, Z_null, y,
                             offsetz = NULL, weights = NULL) {
  n <- length(y)
  if (is.null(offsetz)) offsetz <- rep.int(0, n)
  if (is.null(weights)) weights <- rep.int(1, n)

  if (is.null(null_fit_zero) ||
    !inherits(null_fit_zero, "fasthurdle_null_zero")) {
    null_fit_zero <- fit_null_zero(
      Z_null, y,
      offsetz = offsetz, weights = weights
    )
  }
  null_fit_zero
}
