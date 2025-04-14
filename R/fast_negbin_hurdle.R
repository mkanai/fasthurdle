#' Fast Negative Binomial Hurdle Model with Binomial Zero Hurdle
#'
#' @description
#' A specialized version of fasthurdle that only handles negative binomial count model
#' with binomial zero hurdle model and logit link. This function is optimized
#' for speed by skipping all unnecessary parameter checks and validations.
#'
#' @param X Model matrix for both count and zero components
#' @param y Response vector of counts
#' @param method Optimization method used by optim. Default is "BFGS".
#' @param maxit Maximum number of iterations. Default is 10000.
#' @param separate Logical. If TRUE, count and zero components are estimated separately. Default is TRUE.
#'
#' @return An object of class "fasthurdle" representing the fitted model.
#'
#' @details
#' This function is a specialized version of fasthurdle that only handles the specific
#' parameter combination: dist = "negbin", zero.dist = "binomial", link = "logit".
#' It takes a model matrix and response vector directly, skipping formula processing
#' and other parameter validations to improve performance.
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
#' # Fit the model
#' m <- fast_negbin_hurdle(X, y)
#' summary(m)
#' }
#'
#' @export
fast_negbin_hurdle <- function(X, y, method = "BFGS", maxit = 10000, separate = TRUE) {
  # Fixed parameters
  dist <- "negbin"
  zero.dist <- "binomial"
  linkstr <- "logit"

  # Create link function
  linkobj <- make.link(linkstr)
  linkinv <- linkobj$linkinv

  # Set up model dimensions
  n <- length(y)
  kx <- NCOL(X)
  kz <- NCOL(X) # Using same matrix for both components

  # Set up weights and offsets (default to 1 and 0)
  weights <- rep.int(1, n)
  offsetx <- rep.int(0, n)
  offsetz <- rep.int(0, n)

  # Generate starting values
  # For count model
  model_count <- fastglm::fastglm(X, y, family = poisson(), weights = weights, offset = offsetx)
  # For zero model - for binomial zero model, we use binomial family with logit link
  model_zero <- suppressWarnings(fastglm::fastglm(X, as.integer(y > 0), family = binomial(link = linkstr), weights = weights, offset = offsetz))

  # Combine results
  start <- list(
    count = model_count$coefficients,
    zero = model_zero$coefficients,
    theta = c(count = 1, zero = 1)
  )

  # Set up control parameters
  control <- list(
    method = method,
    maxit = maxit,
    fnscale = -1,
    reltol = .Machine$double.eps^(1 / 1.6),
    hessian = TRUE
  )

  # Perform model estimation
  if (separate) {
    # Estimate count component
    fit_count <- optim_count_negbin_cpp(
      start = c(start$count, log(start$theta["count"])),
      Y = y, X = X, offsetx = offsetx, weights = weights,
      method = method, hessian = TRUE
    )

    # Estimate zero component
    fit_zero <- optim_zero_binom_cpp(
      start = c(start$zero),
      Y = y, X = X, offsetx = offsetz, weights = weights,
      link = linkstr,
      method = method, hessian = TRUE
    )

    # Convert to compatible format
    fit_count_compat <- list(
      par = fit_count$par,
      value = fit_count$value,
      counts = fit_count$counts,
      convergence = fit_count$convergence,
      message = fit_count$message,
      hessian = fit_count$hessian
    )

    fit_zero_compat <- list(
      par = fit_zero$par,
      value = fit_zero$value,
      counts = fit_zero$counts,
      convergence = fit_zero$convergence,
      message = fit_zero$message,
      hessian = fit_zero$hessian
    )

    fit <- list(count = fit_count_compat, zero = fit_zero_compat)

    # Extract coefficients
    coefc <- fit_count$par[1:kx]
    coefz <- fit_zero$par[1:kz]
    theta <- c(
      count = as.vector(exp(fit_count$par[kx + 1])),
      zero = NULL
    )

    # Calculate covariance matrices
    vc_count <- tryCatch(solve(as.matrix(fit_count$hessian)),
      error = function(e) {
        warning(e$message, call = FALSE)
        k <- nrow(as.matrix(fit_count$hessian))
        return(matrix(NA, k, k))
      }
    )
    vc_zero <- tryCatch(solve(as.matrix(fit_zero$hessian)),
      error = function(e) {
        warning(e$message, call = FALSE)
        k <- nrow(as.matrix(fit_zero$hessian))
        return(matrix(NA, k, k))
      }
    )

    # Extract standard errors for theta
    SE.logtheta <- list()
    diag_val <- diag(vc_count)[kx + 1]
    SE.logtheta$count <- if (is.na(diag_val) || diag_val <= 0) NA_real_ else as.vector(sqrt(diag_val))
    vc_count <- vc_count[-(kx + 1), -(kx + 1), drop = FALSE]

    # Combine covariance matrices
    vc <- rbind(cbind(vc_count, matrix(0, kx, kz)), cbind(matrix(0, kz, kx), vc_zero))
    SE.logtheta <- unlist(SE.logtheta)

    # Calculate loglik
    loglik <- fit_count$value + fit_zero$value
    converged <- fit_count$convergence < 1 & fit_zero$convergence < 1
  } else {
    # Estimate joint model
    fit_result <- optim_joint_cpp(
      start = c(
        start$count, log(start$theta["count"]),
        start$zero
      ),
      Y = y, X = X, offsetx = offsetx, Z = X, offsetz = offsetz, weights = weights,
      dist = dist, zero_dist = zero.dist,
      link = linkstr,
      method = method, hessian = TRUE
    )

    # Convert to compatible format
    fit <- list(
      par = fit_result$par,
      value = fit_result$value,
      counts = fit_result$counts,
      convergence = fit_result$convergence,
      message = fit_result$message,
      hessian = fit_result$hessian
    )

    if (fit$convergence > 0) warning("optimization failed to converge")

    # Extract coefficients
    coefc <- fit$par[1:kx]
    coefz <- fit$par[(kx + 1 + 1):(kx + kz + 1)]

    # Calculate covariance matrix
    vc <- tryCatch(solve(as.matrix(fit$hessian)),
      error = function(e) {
        warning(e$message, call = FALSE)
        k <- nrow(as.matrix(fit$hessian))
        return(matrix(NA, k, k))
      }
    )

    # Extract theta parameters and standard errors
    np <- kx + 1

    theta <- c(count = as.vector(exp(fit$par[np])))
    diag_val <- diag(vc)[np]
    SE.logtheta <- if (is.na(diag_val) || diag_val <= 0) NA_real_ else as.vector(sqrt(diag_val))
    names(SE.logtheta) <- "count"
    vc <- vc[-np, -np, drop = FALSE]

    # Calculate loglik and convergence
    loglik <- fit$value
    converged <- fit$convergence < 1
  }

  # Set coefficient names
  names(coefc) <- colnames(X)
  names(coefz) <- colnames(X)
  colnames(vc) <- rownames(vc) <- c(
    paste("count", colnames(X), sep = "_"),
    paste("zero", colnames(X), sep = "_")
  )

  # Calculate fitted values
  # Calculate zero component
  phi <- linkinv(X %*% coefz + offsetz)[, 1]

  # Calculate probability of zero
  p0_zero <- log(phi)

  # Calculate count component
  mu <- exp(X %*% coefc + offsetx)[, 1]

  # Calculate probability of zero in count model
  p0_count <- pnbinom(0, size = theta["count"], mu = mu, lower.tail = FALSE, log.p = TRUE)

  # Calculate fitted values
  Yhat <- exp((p0_zero - p0_count) + log(mu))

  # Calculate residuals
  res <- y - Yhat

  # Calculate effective observations
  nobs <- sum(weights > 0)

  # Create return object
  rval <- list(
    coefficients = list(count = coefc, zero = coefz),
    residuals = res,
    fitted.values = Yhat,
    optim = fit,
    method = method,
    control = control,
    start = start,
    weights = NULL,
    offset = list(
      count = NULL,
      zero = NULL
    ),
    n = nobs,
    df.null = nobs - 2,
    df.residual = nobs - (kx + kz + 1), # +1 for the count theta parameter
    theta = theta,
    SE.logtheta = SE.logtheta,
    loglik = loglik,
    vcov = vc,
    dist = list(count = dist, zero = zero.dist),
    link = linkstr,
    linkinv = linkinv,
    separate = separate,
    converged = converged,
    call = match.call(),
    y = y,
    x = list(count = X, zero = X)
  )

  class(rval) <- "fasthurdle"
  return(rval)
}
