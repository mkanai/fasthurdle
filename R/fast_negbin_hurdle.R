#' Fast Negative Binomial Hurdle Model with Binomial Zero Hurdle
#'
#' @description
#' A specialized version of fasthurdle that only handles negative binomial count model
#' with binomial zero hurdle model and logit link. This function is optimized
#' for speed by skipping all unnecessary parameter checks and validations.
#'
#' @param X Model matrix for the count component (and zero component if Z is NULL)
#' @param y Response vector of counts
#' @param Z Optional model matrix for the zero component. Default is NULL (use X).
#' @param offsetx Optional offset vector for the count model. Default is NULL (no offset).
#' @param offsetz Optional offset vector for the zero model. Default is NULL (no offset).
#' @param method Optimization method used by optim. Default is "BFGS".
#' @param maxit Maximum number of iterations. Default is 10000.
#' @param separate Logical. If TRUE, count and zero components are estimated separately. Default is TRUE.
#' @param score_test Optional. Column name(s) or integer index(es) in X to compute
#'   a score test for in the count component. When specified, the full count model is
#'   NOT fitted — only the null model (X without the test columns). This is faster
#'   and gives better-calibrated p-values. See Details and \code{\link{fasthurdle}}
#'   for background on the score test.
#' @param null_fit Optional. A pre-fitted null model from \code{\link{fit_null_count}}.
#'   When provided with \code{score_test}, the null model is not re-fitted. This is
#'   useful for testing many predictors against the same null (e.g., many peaks per gene).
#' @param spa_cutoff Numeric or NULL. When \code{score_test} is used, apply saddlepoint
#'   approximation (SPA) for p-values when |z| exceeds this cutoff. Default is 2.
#'   Set to \code{NULL} or \code{Inf} to disable SPA.
#' @param compute_fitted Logical. If FALSE (default), skip computing fitted values
#'   and residuals for speed. Set to TRUE if you need fitted.values, residuals, y, or x
#'   in the returned object. Not available when \code{score_test} is used.
#'
#' @return An object of class "fasthurdle" representing the fitted model.
#'
#' @details
#' ## Score test mode
#'
#' When \code{score_test} is specified, the function operates differently:
#' \itemize{
#'   \item The full count model is NOT fitted — only the null model (covariates only).
#'   \item For significant tests (|z| > \code{spa_cutoff}), beta is refined via a short
#'     BFGS optimization from the score estimate (within ~1\% of the full MLE). For
#'     non-significant tests, beta uses the ratio estimator (approximate).
#'   \item Covariate coefficients are the null model MLEs (valid under H0).
#'   \item \code{loglik} is the null model's log-likelihood, not the full model's.
#'   \item \code{vcov} contains the null model's covariance for covariates and the
#'     score-based variance for the test variable. Cross-covariances are NA.
#'   \item \code{AIC()} and \code{BIC()} reflect the null model.
#'   \item \code{theta} is estimated from the null model.
#'   \item \code{compute_fitted} is not available (fitted values require the full model).
#' }
#'
#' @examples
#' \dontrun{
#' # Load example data
#' data(bioChemists, package = "pscl")
#'
#' # Create model matrix and response vector
#' X <- model.matrix(~ fem + mar + kid5 + phd + ment, data = bioChemists)
#' y <- bioChemists$art
#'
#' # Fit the model (Wald test)
#' m <- fast_negbin_hurdle(X, y)
#' summary(m)
#'
#' # Fit with score test for 'ment'
#' m <- fast_negbin_hurdle(X, y, score_test = "ment")
#' summary(m)
#'
#' # High-throughput: cache null model, test many variables
#' X_null <- model.matrix(~ fem + mar + kid5 + phd, data = bioChemists)
#' null_fit <- fit_null_count(X_null, y, dist = "negbin")
#' m <- fast_negbin_hurdle(X, y, score_test = "ment", null_fit = null_fit)
#' summary(m)
#' }
#'
#' @export
fast_negbin_hurdle <- function(X, y, Z = NULL, offsetx = NULL, offsetz = NULL,
                               method = "BFGS", maxit = 10000, separate = TRUE,
                               score_test = NULL, null_fit = NULL, spa_cutoff = 2,
                               compute_fitted = FALSE) {
  # Fixed parameters
  dist <- "negbin"
  zero.dist <- "binomial"
  linkstr <- "logit"
  linkinv <- plogis

  # Set up model dimensions
  n <- length(y)
  if (is.null(Z)) Z <- X
  kx <- NCOL(X)
  kz <- NCOL(Z)

  # Set up weights and offsets
  weights <- rep.int(1, n)
  if (is.null(offsetx)) offsetx <- rep.int(0, n)
  if (is.null(offsetz)) offsetz <- rep.int(0, n)

  # Resolve score_test column indices
  test_idx <- NULL
  if (!is.null(score_test)) {
    if (is.character(score_test)) {
      test_idx <- match(score_test, colnames(X))
      if (any(is.na(test_idx))) {
        stop(
          "score_test column(s) not found in X: ",
          paste(score_test[is.na(test_idx)], collapse = ", ")
        )
      }
    } else {
      test_idx <- as.integer(score_test)
    }
    if (length(test_idx) > 1) {
      stop("score_test currently supports only a single test variable")
    }
    if (!separate) {
      warning("separate = FALSE is ignored when score_test is used")
    }
    if (compute_fitted) {
      warning("compute_fitted is not available with score_test; ignoring")
      compute_fitted <- FALSE
    }
  }

  reltol <- .Machine$double.eps^(1 / 1.6)
  control <- list(
    method = method, maxit = maxit, fnscale = -1,
    reltol = reltol, hessian = TRUE
  )

  # ====================================================================
  # Count component estimation
  # ====================================================================
  if (!is.null(test_idx)) {
    # --- Score test mode: fit null model only ---
    null_idx <- setdiff(seq_len(kx), test_idx)
    X_null <- X[, null_idx, drop = FALSE]
    x_test <- X[, test_idx, drop = FALSE]

    # Fit or reuse null model
    if (is.null(null_fit)) {
      null_fit <- fit_null_count(X_null, y,
        offsetx = offsetx, weights = weights,
        dist = "negbin", method = method, maxit = maxit
      )
    }

    # Compute score test
    st_result <- score_test_count(
      X_null, x_test, y,
      offsetx = offsetx, weights = weights,
      dist = "negbin", null_fit = null_fit, spa_cutoff = spa_cutoff
    )

    # Build count coefficients: null MLE for covariates, score for test
    kx_null <- length(null_idx)
    coefc <- numeric(kx)
    coefc[null_idx] <- null_fit$par[seq_len(kx_null)]
    coefc[test_idx] <- st_result$beta

    # Theta from null model
    theta <- c(count = as.vector(exp(null_fit$par[kx_null + 1])))
    SE.logtheta <- c(count = NA_real_)

    # Use null model loglik (covariate vcov is NA in score test mode)
    null_loglik <- null_fit$value
    vc_count <- matrix(NA_real_, kx, kx)
    for (j in seq_along(test_idx)) {
      vc_count[test_idx[j], test_idx[j]] <- st_result$se[j]^2
    }

    fit <- list(
      count = list(par = null_fit$par, convergence = null_fit$convergence),
      zero = NULL
    )
    converged_count <- null_fit$convergence == 0
  } else {
    # --- Standard Wald mode: fit full count model ---
    pos_idx <- y > 0
    model_count <- tryCatch(
      fastglm::fastglm(
        X[pos_idx, , drop = FALSE], y[pos_idx],
        family = poisson(),
        weights = weights[pos_idx], offset = offsetx[pos_idx]
      ),
      error = function(e) NULL
    )
    if (is.null(model_count) || any(!is.finite(model_count$coefficients))) {
      pos_mean <- if (any(pos_idx)) mean(log(y[pos_idx] + 0.5) - offsetx[pos_idx]) else 0
      start_count <- c(pos_mean, rep(0, kx - 1), log(1))
    } else {
      start_count <- c(model_count$coefficients, log(1))
    }

    if (separate) {
      fit_count <- optim_count_negbin_cpp(
        start = start_count, Y = y, X = X, offsetx = offsetx, weights = weights,
        method = method, hessian = TRUE, maxit = maxit, reltol = reltol
      )
    } else {
      model_zero <- suppressWarnings(fastglm::fastglm(
        Z, as.integer(y > 0),
        family = binomial(link = linkstr),
        weights = weights, offset = offsetz
      ))
      fit_result <- optim_joint_cpp(
        start = c(model_count$coefficients, log(1), model_zero$coefficients),
        Y = y, X = X, offsetx = offsetx, Z = Z, offsetz = offsetz, weights = weights,
        dist = dist, zero_dist = zero.dist, link = linkstr,
        method = method, hessian = TRUE, maxit = maxit, reltol = reltol
      )
      if (fit_result$convergence > 0) warning("optimization failed to converge")
      # Extract from joint fit
      coefc <- fit_result$par[1:kx]
      coefz_joint <- fit_result$par[(kx + 2):(kx + kz + 1)]
      vc_joint <- tryCatch(solve(as.matrix(fit_result$hessian)),
        error = function(e) {
          warning(e$message, call. = FALSE)
          matrix(NA, nrow(as.matrix(fit_result$hessian)), ncol(as.matrix(fit_result$hessian)))
        }
      )
      np <- kx + 1
      theta <- c(count = as.vector(exp(fit_result$par[np])))
      diag_val <- diag(vc_joint)[np]
      SE.logtheta <- c(count = if (is.na(diag_val) || diag_val <= 0) NA_real_ else as.vector(sqrt(diag_val)))
      vc_joint <- vc_joint[-np, -np, drop = FALSE]
      names(coefc) <- colnames(X)
      names(coefz_joint) <- colnames(Z)
      vc_count <- vc_joint[1:kx, 1:kx, drop = FALSE]
      loglik <- fit_result$value
      converged_count <- fit_result$convergence < 1
      # Will assemble full vc below after zero model
    }

    if (separate) {
      coefc <- fit_count$par[1:kx]
      theta <- c(count = as.vector(exp(fit_count$par[kx + 1])))
      vc_count_full <- tryCatch(solve(as.matrix(fit_count$hessian)),
        error = function(e) {
          warning(e$message, call. = FALSE)
          matrix(NA, nrow(as.matrix(fit_count$hessian)), ncol(as.matrix(fit_count$hessian)))
        }
      )
      diag_val <- diag(vc_count_full)[kx + 1]
      SE.logtheta <- c(count = if (is.na(diag_val) || diag_val <= 0) NA_real_ else as.vector(sqrt(diag_val)))
      vc_count <- vc_count_full[-(kx + 1), -(kx + 1), drop = FALSE]
      null_loglik <- fit_count$value
      converged_count <- fit_count$convergence < 1
    }

    if (separate) {
      fit <- list(count = fit_count, zero = NULL)
    } else {
      # Mark zero as done so we don't re-fit below (coefz extracted from joint)
      fit <- list(joint = fit_result, zero = list(convergence = fit_result$convergence))
    }
    st_result <- NULL
  }

  # ====================================================================
  # Zero component estimation (always Wald)
  # ====================================================================
  if (is.null(fit$zero)) {
    model_zero_start <- tryCatch(
      suppressWarnings(fastglm::fastglm(
        Z, as.integer(y > 0),
        family = binomial(link = linkstr),
        weights = weights, offset = offsetz
      )),
      error = function(e) NULL
    )
    zero_start <- if (!is.null(model_zero_start) &&
      all(is.finite(model_zero_start$coefficients))) {
      model_zero_start$coefficients
    } else {
      rep(0, kz)
    }
    fit_zero <- optim_zero_binom_cpp(
      start = zero_start,
      Y = y, X = Z, offsetx = offsetz, weights = weights, link = linkstr,
      method = method, hessian = TRUE, maxit = maxit, reltol = reltol
    )
    fit$zero <- fit_zero
  }

  if (!is.null(fit$joint)) {
    # Joint mode: extract zero coefs from joint fit (already computed as coefz_joint)
    coefz <- coefz_joint
    vc_zero <- vc_joint[(kx + 1):(kx + kz), (kx + 1):(kx + kz), drop = FALSE]
  } else {
    coefz <- fit$zero$par[1:kz]
    vc_zero <- tryCatch(solve(as.matrix(fit$zero$hessian)),
      error = function(e) {
        warning(e$message, call. = FALSE)
        matrix(NA, nrow(as.matrix(fit$zero$hessian)), ncol(as.matrix(fit$zero$hessian)))
      }
    )
  }

  # ====================================================================
  # Assemble return object
  # ====================================================================
  names(coefc) <- colnames(X)
  names(coefz) <- colnames(Z)

  # Combined vcov
  if (!is.null(fit$joint)) {
    # Joint mode: vcov already includes both count and zero
    vc <- vc_joint
  } else {
    vc <- rbind(
      cbind(vc_count, matrix(NA_real_, kx, kz)),
      cbind(matrix(NA_real_, kz, kx), vc_zero)
    )
    # For Wald mode (no score test), use 0 for cross-terms (independent estimation)
    if (is.null(test_idx)) {
      vc[1:kx, (kx + 1):(kx + kz)] <- 0
      vc[(kx + 1):(kx + kz), 1:kx] <- 0
    }
  }
  colnames(vc) <- rownames(vc) <- c(
    paste("count", colnames(X), sep = "_"),
    paste("zero", colnames(Z), sep = "_")
  )

  # loglik
  if (!is.null(test_idx)) {
    loglik <- null_loglik + fit$zero$value
  } else if (separate) {
    loglik <- null_loglik + fit$zero$value
  }
  # (joint mode loglik already set above)

  converged <- converged_count & (fit$zero$convergence < 1)

  # Fitted values
  Yhat <- res <- NULL
  if (compute_fitted) {
    fitted_result <- compute_negbin_hurdle_fitted_cpp(
      coefc, coefz, X, Z, offsetx, offsetz, theta["count"], y
    )
    Yhat <- fitted_result$fitted.values
    names(Yhat) <- rownames(X)
    res <- fitted_result$residuals
  }

  nobs <- sum(weights > 0)

  rval <- list(
    coefficients = list(count = coefc, zero = coefz),
    residuals = res,
    fitted.values = Yhat,
    optim = fit,
    method = method,
    control = control,
    start = NULL,
    weights = NULL,
    offset = list(
      count = if (all(offsetx == 0)) NULL else offsetx,
      zero = if (all(offsetz == 0)) NULL else offsetz
    ),
    n = nobs,
    df.null = nobs - 2,
    df.residual = if (!is.null(test_idx)) nobs - (length(null_idx) + kz + 1) else nobs - (kx + kz + 1),
    theta = theta,
    SE.logtheta = SE.logtheta,
    loglik = loglik,
    vcov = vc,
    dist = list(count = dist, zero = zero.dist),
    link = linkstr,
    linkinv = linkinv,
    separate = if (!is.null(fit$joint)) FALSE else TRUE,
    score_test = st_result,
    converged = converged,
    call = match.call(),
    y = if (compute_fitted) y else NULL,
    x = if (compute_fitted) list(count = X, zero = Z) else NULL
  )

  class(rval) <- "fasthurdle"
  rval
}
