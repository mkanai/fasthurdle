#' Fit Hurdle Regression Models for Count Data
#'
#' @description
#' Fits hurdle regression models for count data. The hurdle model combines a count data model
#' (such as Poisson, negative binomial, or geometric) for positive counts with a hurdle component
#' that models the zero counts. This implementation uses C++ for the core computations, making it
#' faster than the original pscl::hurdle implementation.
#'
#' @param formula A formula expression of the form y ~ x | z where y is the response
#'   and x and z are regressor variables for the count and zero components, respectively.
#'   If the | operator is not specified, the same regressors are used for both components.
#' @param data An optional data frame containing the variables in the model.
#' @param subset An optional vector specifying a subset of observations to be used.
#' @param na.action A function which indicates what should happen when the data contain NAs.
#' @param weights An optional vector of weights to be used in the fitting process.
#' @param offset An optional offset for the count model. Can also be specified via the offset
#'   argument in the formula.
#' @param dist Character string specifying the count distribution. Currently, "poisson",
#'   "negbin" (negative binomial), and "geometric" are supported.
#' @param zero.dist Character string specifying the zero hurdle distribution. Currently,
#'   "binomial", "poisson", "negbin" (negative binomial), and "geometric" are supported.
#' @param link Character string specifying the link function for the binomial zero hurdle model.
#'   Currently, "logit", "probit", "cloglog", "cauchit", and "log" are supported.
#' @param control A list of control parameters passed to the optimizer. See \code{\link{hurdle.control}}.
#' @param model Logical. If TRUE, the model frame is included in the returned object.
#' @param y Logical. If TRUE, the response vector is included in the returned object.
#' @param x Logical. If TRUE, the model matrices are included in the returned object.
#' @param ... Additional arguments passed to \code{\link{hurdle.control}}.
#'
#' @return An object of class "fasthurdle" representing the fitted model.
#'
#' @details
#' The hurdle model combines two components: a truncated count component for positive counts
#' and a zero hurdle component that models the zeros. The probability mass function is given by:
#'
#' \deqn{f(y) = \begin{cases}
#' f_{zero}(0) & \text{if } y = 0 \\
#' (1 - f_{zero}(0)) \cdot \frac{f_{count}(y)}{1 - f_{count}(0)} & \text{if } y > 0
#' \end{cases}}
#'
#' where \eqn{f_{zero}} is the zero hurdle distribution and \eqn{f_{count}} is the count distribution.
#'
#' @examples
#' \dontrun{
#' # Load example data
#' data(bioChemists, package = "pscl")
#'
#' # Fit a hurdle model with Poisson count component and binomial zero component
#' m1 <- fasthurdle(art ~ fem + mar + kid5 + phd + ment | fem + mar + kid5 + phd + ment,
#'   data = bioChemists, dist = "poisson", zero.dist = "binomial"
#' )
#' summary(m1)
#'
#' # Fit a hurdle model with negative binomial count component
#' m2 <- fasthurdle(art ~ fem + mar + kid5 + phd + ment | fem + mar + kid5 + phd + ment,
#'   data = bioChemists, dist = "negbin", zero.dist = "binomial"
#' )
#' summary(m2)
#' }
#'
#' @export
fasthurdle <- function(formula, data, subset, na.action, weights, offset,
                       dist = c("poisson", "negbin", "geometric"),
                       zero.dist = c("binomial", "poisson", "negbin", "geometric"),
                       link = c("logit", "probit", "cloglog", "cauchit", "log"),
                       control = hurdle.control(...),
                       model = TRUE, y = TRUE, x = FALSE, ...) {
  # Match arguments
  dist <- match.arg(dist)
  zero.dist <- match.arg(zero.dist)
  linkstr <- match.arg(link)

  # Process link function
  linkobj <- make.link(linkstr)
  linkinv <- linkobj$linkinv

  # Print trace information if requested
  if (control$trace) {
    cat("Hurdle Count Model\n",
      paste("count model:", dist, "with log link\n"),
      paste(
        "zero hurdle model:", zero.dist, "with",
        ifelse(zero.dist == "binomial", linkstr, "log"), "link\n"
      ),
      sep = ""
    )
  }

  # Process call and formula
  cl <- match.call()
  if (missing(data)) data <- environment(formula)
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "na.action", "weights", "offset"), names(mf), 0)
  mf <- mf[c(1, m)]
  mf$drop.unused.levels <- TRUE

  # Process extended formula with | operator
  if (length(formula[[3]]) > 1 && identical(formula[[3]][[1]], as.name("|"))) {
    ff <- formula
    formula[[3]][1] <- call("+")
    mf$formula <- formula
    ffc <- . ~ .
    ffz <- ~.
    ffc[[2]] <- ff[[2]]
    ffc[[3]] <- ff[[3]][[2]]
    ffz[[3]] <- ff[[3]][[3]]
    ffz[[2]] <- NULL
  } else {
    ffz <- ffc <- ff <- formula
    ffz[[2]] <- NULL
  }

  # Handle special case for zero model formula
  if (inherits(try(terms(ffz), silent = TRUE), "try-error")) {
    ffz <- eval(parse(text = sprintf(paste("%s -", deparse(ffc[[2]])), deparse(ffz))))
  }

  # Create model frame
  mf[[1]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())

  # Extract model components
  mt <- attr(mf, "terms")
  mtX <- terms(ffc, data = data)
  X <- model.matrix(mtX, mf)
  mtZ <- terms(ffz, data = data)
  mtZ <- terms(update(mtZ, ~.), data = data)
  Z <- model.matrix(mtZ, mf)
  Y <- model.response(mf, "numeric")

  # Perform data validation
  validate_data(Y, Z, zero.dist)

  # Print dependent variable summary if requested
  if (control$trace) {
    cat("dependent variable:\n")
    tab <- table(factor(Y, levels = 0:max(Y)), exclude = NULL)
    names(dimnames(tab)) <- NULL
    print(tab)
  }

  # Set up model dimensions
  n <- length(Y)
  kx <- NCOL(X)
  kz <- NCOL(Z)

  # Process weights
  weights <- process_weights(mf, n)

  # Process offsets
  offsets <- process_offsets(mf, mtX, mtZ, n)
  offsetx <- offsets$offsetx
  offsetz <- offsets$offsetz

  # Generate starting values
  start <- generate_starting_values(
    control$start, Y, X, Z, offsetx, offsetz,
    weights, dist, zero.dist, linkstr, kx, kz, control$trace
  )

  # Extract control parameters
  method <- control$method
  hessian <- control$hessian
  separate <- control$separate
  control$method <- control$hessian <- control$separate <- control$start <- NULL

  # Perform model estimation
  if (separate) {
    fit_result <- estimate_separate_components(
      Y, X, Z, offsetx, offsetz, weights,
      dist, zero.dist, linkstr, start,
      method, hessian, control, kx, kz
    )
  } else {
    fit_result <- estimate_joint_components(
      Y, X, Z, offsetx, offsetz, weights,
      dist, zero.dist, linkstr, start,
      method, hessian, control, kx, kz
    )
  }

  # Extract results
  fit <- fit_result$fit
  coefc <- fit_result$coefc
  coefz <- fit_result$coefz
  theta <- fit_result$theta
  vc <- fit_result$vc
  SE.logtheta <- fit_result$SE.logtheta

  # Set coefficient names
  names(coefc) <- names(start$count) <- colnames(X)
  names(coefz) <- names(start$zero) <- colnames(Z)
  colnames(vc) <- rownames(vc) <- c(
    paste("count", colnames(X), sep = "_"),
    paste("zero", colnames(Z), sep = "_")
  )

  # Calculate fitted values and residuals
  fitted_values <- calculate_fitted_values(
    Y, X, Z, coefc, coefz, offsetx, offsetz,
    dist, zero.dist, theta, linkinv
  )

  # Calculate effective observations
  nobs <- sum(weights > 0)

  # Create return object
  rval <- list(
    coefficients = list(count = coefc, zero = coefz),
    residuals = fitted_values$res,
    fitted.values = fitted_values$Yhat,
    optim = fit,
    method = method,
    control = control,
    start = start,
    weights = if (identical(as.vector(weights), rep.int(1L, n))) NULL else weights,
    offset = list(
      count = if (identical(offsetx, rep.int(0, n))) NULL else offsetx,
      zero = if (identical(offsetz, rep.int(0, n))) NULL else offsetz
    ),
    n = nobs,
    df.null = nobs - 2,
    df.residual = nobs - (kx + kz + (dist == "negbin") + (zero.dist == "negbin")),
    terms = list(count = mtX, zero = mtZ, full = mt),
    theta = theta,
    SE.logtheta = SE.logtheta,
    loglik = if (separate) fit_result$loglik else fit$value,
    vcov = vc,
    dist = list(count = dist, zero = zero.dist),
    link = if (zero.dist == "binomial") linkstr else NULL,
    linkinv = if (zero.dist == "binomial") linkinv else NULL,
    separate = separate,
    converged = fit_result$converged,
    call = cl,
    formula = ff,
    levels = .getXlevels(mt, mf),
    contrasts = list(count = attr(X, "contrasts"), zero = attr(Z, "contrasts"))
  )

  # Add optional components
  if (model) rval$model <- mf
  if (y) rval$y <- Y
  if (x) rval$x <- list(count = X, zero = Z)

  class(rval) <- "fasthurdle"
  return(rval)
}

#' Validate Input Data for Hurdle Models
#'
#' @param Y Response variable
#' @param Z Zero model matrix
#' @param zero.dist Zero distribution type
#'
#' @return NULL, but stops with error if validation fails
#' @keywords internal
validate_data <- function(Y, Z, zero.dist) {
  if (length(Y) < 1) stop("empty model")
  if (all(Y > 0)) stop("invalid dependent variable, minimum count is not zero")
  if (!isTRUE(all.equal(as.vector(Y), as.integer(round(Y + 0.001))))) {
    stop("invalid dependent variable, non-integer values")
  }
  if (any(Y < 0)) stop("invalid dependent variable, negative counts")
  if (zero.dist == "negbin" & isTRUE(all.equal(as.vector(Z), rep.int(Z[1], length(Z))))) {
    stop("negative binomial zero hurdle model is not identified with only an intercept")
  }
}

#' Process Weights for Hurdle Models
#'
#' @param mf Model frame
#' @param n Number of observations
#'
#' @return Processed weights vector
#' @keywords internal
process_weights <- function(mf, n) {
  weights <- model.weights(mf)
  if (is.null(weights)) weights <- 1
  if (length(weights) == 1) weights <- rep.int(weights, n)
  weights <- as.vector(weights)
  names(weights) <- rownames(mf)
  return(weights)
}

#' Process Offsets for Hurdle Models
#'
#' @param mf Model frame
#' @param mtX Count model terms
#' @param mtZ Zero model terms
#' @param n Number of observations
#'
#' @return List with offsetx and offsetz
#' @keywords internal
process_offsets <- function(mf, mtX, mtZ, n) {
  offsetx <- model_offset_2(mf, terms = mtX, offset = TRUE)
  if (is.null(offsetx)) offsetx <- 0
  if (length(offsetx) == 1) offsetx <- rep.int(offsetx, n)
  offsetx <- as.vector(offsetx)

  offsetz <- model_offset_2(mf, terms = mtZ, offset = FALSE)
  if (is.null(offsetz)) offsetz <- 0
  if (length(offsetz) == 1) offsetz <- rep.int(offsetz, n)
  offsetz <- as.vector(offsetz)

  return(list(offsetx = offsetx, offsetz = offsetz))
}

#' Generate Starting Values for Hurdle Models
#'
#' @param start User-provided starting values
#' @param Y Response variable
#' @param X Count model matrix
#' @param Z Zero model matrix
#' @param offsetx Count model offset
#' @param offsetz Zero model offset
#' @param weights Observation weights
#' @param dist Count distribution
#' @param zero.dist Zero distribution
#' @param linkstr Link function
#' @param kx Number of count model parameters
#' @param kz Number of zero model parameters
#' @param trace Logical for printing trace information
#'
#' @return List of starting values
#' @keywords internal
generate_starting_values <- function(start, Y, X, Z, offsetx, offsetz, weights,
                                     dist, zero.dist, linkstr, kx, kz, trace) {
  if (!is.null(start)) {
    valid <- TRUE

    # Validate count coefficients
    if (!("count" %in% names(start))) {
      valid <- FALSE
      warning("invalid starting values, count model coefficients not specified")
      start$count <- rep.int(0, kx)
    }

    # Validate zero coefficients
    if (!("zero" %in% names(start))) {
      valid <- FALSE
      warning("invalid starting values, zero-inflation model coefficients not specified")
      start$zero <- rep.int(0, kz)
    }

    # Check dimensions
    if (length(start$count) != kx) {
      valid <- FALSE
      warning("invalid starting values, wrong number of count model coefficients")
    }
    if (length(start$zero) != kz) {
      valid <- FALSE
      warning("invalid starting values, wrong number of zero-inflation model coefficients")
    }

    # Process theta parameters
    if (dist == "negbin" | zero.dist == "negbin") {
      if (!("theta" %in% names(start))) start$theta <- c(1, 1)
      start <- list(count = start$count, zero = start$zero, theta = rep(start$theta, length.out = 2))
      if (is.null(names(start$theta))) names(start$theta) <- c("count", "zero")
      if (dist != "negbin") start$theta <- start$theta["zero"]
      if (zero.dist != "negbin") start$theta <- start$theta["count"]
    } else {
      start <- list(count = start$count, zero = start$zero)
    }

    if (!valid) start <- NULL
  }

  # Generate starting values if not provided or invalid
  if (is.null(start)) {
    if (trace) cat("generating starting values...")

    # Fit count model
    model_count <- fastglm::fastglm(X, Y, family = poisson(), weights = weights, offset = offsetx)

    # Fit zero model
    model_zero <- switch(zero.dist,
      "poisson" = fastglm::fastglm(Z, Y, family = poisson(), weights = weights, offset = offsetz),
      "negbin" = fastglm::fastglm(Z, Y, family = poisson(), weights = weights, offset = offsetz),
      "geometric" = suppressWarnings(fastglm::fastglm(Z, as.integer(Y > 0), family = binomial(), weights = weights, offset = offsetz)),
      "binomial" = suppressWarnings(fastglm::fastglm(Z, as.integer(Y > 0), family = binomial(link = linkstr), weights = weights, offset = offsetz))
    )

    # Combine results
    start <- list(count = model_count$coefficients, zero = model_zero$coefficients)
    start$theta <- c(
      count = if (dist == "negbin") 1 else NULL,
      zero = if (zero.dist == "negbin") 1 else NULL
    )

    if (trace) cat("done\n")
  }

  return(start)
}

#' Estimate Separate Components for Hurdle Models
#'
#' @param Y Response variable
#' @param X Count model matrix
#' @param Z Zero model matrix
#' @param offsetx Count model offset
#' @param offsetz Zero model offset
#' @param weights Observation weights
#' @param dist Count distribution
#' @param zero.dist Zero distribution
#' @param linkstr Link function
#' @param start Starting values
#' @param method Optimization method
#' @param hessian Logical for computing hessian
#' @param control Control parameters
#' @param kx Number of count model parameters
#' @param kz Number of zero model parameters
#'
#' @return List of estimation results
#' @keywords internal
estimate_separate_components <- function(Y, X, Z, offsetx, offsetz, weights,
                                         dist, zero.dist, linkstr, start,
                                         method, hessian, control, kx, kz) {
  if (control$trace) cat("calling roptim for count component estimation:\n")

  # Estimate count component
  fit_count <- switch(dist,
    "poisson" = optim_count_poisson_cpp(
      start = c(start$count),
      Y = Y, X = X, offsetx = offsetx, weights = weights,
      method = method, hessian = hessian
    ),
    "negbin" = optim_count_negbin_cpp(
      start = c(start$count, log(start$theta["count"])),
      Y = Y, X = X, offsetx = offsetx, weights = weights,
      method = method, hessian = hessian
    ),
    "geometric" = optim_count_geom_cpp(
      start = c(start$count),
      Y = Y, X = X, offsetx = offsetx, weights = weights,
      method = method, hessian = hessian
    )
  )

  if (control$trace) cat("calling roptim for zero hurdle component estimation:\n")

  # Estimate zero component
  fit_zero <- switch(zero.dist,
    "poisson" = optim_zero_poisson_cpp(
      start = c(start$zero),
      Y = Y, X = Z, offsetx = offsetz, weights = weights,
      method = method, hessian = hessian
    ),
    "negbin" = optim_zero_negbin_cpp(
      start = c(start$zero, log(start$theta["zero"])),
      Y = Y, X = Z, offsetx = offsetz, weights = weights,
      method = method, hessian = hessian
    ),
    "geometric" = optim_zero_geom_cpp(
      start = c(start$zero),
      Y = Y, X = Z, offsetx = offsetz, weights = weights,
      method = method, hessian = hessian
    ),
    "binomial" = optim_zero_binom_cpp(
      start = c(start$zero),
      Y = Y, X = Z, offsetx = offsetz, weights = weights,
      link = linkstr,
      method = method, hessian = hessian
    )
  )

  if (control$trace) cat("done\n")

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
    count = if (dist == "negbin") as.vector(exp(fit_count$par[kx + 1])) else NULL,
    zero = if (zero.dist == "negbin") as.vector(exp(fit_zero$par[kz + 1])) else NULL
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
  if (dist == "negbin") {
    diag_val <- diag(vc_count)[kx + 1]
    SE.logtheta$count <- if (is.na(diag_val) || diag_val <= 0) NA_real_ else as.vector(sqrt(diag_val))
    vc_count <- vc_count[-(kx + 1), -(kx + 1), drop = FALSE]
  }
  if (zero.dist == "negbin") {
    diag_val <- diag(vc_zero)[kz + 1]
    SE.logtheta$zero <- if (is.na(diag_val) || diag_val <= 0) NA_real_ else as.vector(sqrt(diag_val))
    vc_zero <- vc_zero[-(kz + 1), -(kz + 1), drop = FALSE]
  }

  # Combine covariance matrices
  vc <- rbind(cbind(vc_count, matrix(0, kx, kz)), cbind(matrix(0, kz, kx), vc_zero))
  SE.logtheta <- unlist(SE.logtheta)

  # Return results
  return(list(
    fit = fit,
    coefc = coefc,
    coefz = coefz,
    theta = theta,
    vc = vc,
    SE.logtheta = SE.logtheta,
    loglik = fit_count$value + fit_zero$value,
    converged = fit_count$convergence < 1 & fit_zero$convergence < 1
  ))
}

#' Estimate Joint Components for Hurdle Models
#'
#' @param Y Response variable
#' @param X Count model matrix
#' @param Z Zero model matrix
#' @param offsetx Count model offset
#' @param offsetz Zero model offset
#' @param weights Observation weights
#' @param dist Count distribution
#' @param zero.dist Zero distribution
#' @param linkstr Link function
#' @param start Starting values
#' @param method Optimization method
#' @param hessian Logical for computing hessian
#' @param control Control parameters
#' @param kx Number of count model parameters
#' @param kz Number of zero model parameters
#'
#' @return List of estimation results
#' @keywords internal
estimate_joint_components <- function(Y, X, Z, offsetx, offsetz, weights,
                                      dist, zero.dist, linkstr, start,
                                      method, hessian, control, kx, kz) {
  if (control$trace) cat("calling roptim for joint count and zero hurdle estimation:\n")

  # Estimate joint model
  fit_result <- optim_joint_cpp(
    start = c(
      start$count, if (dist == "negbin") log(start$theta["count"]) else NULL,
      start$zero, if (zero.dist == "negbin") log(start$theta["zero"]) else NULL
    ),
    Y = Y, X = X, offsetx = offsetx, Z = Z, offsetz = offsetz, weights = weights,
    dist = dist, zero_dist = zero.dist,
    link = linkstr,
    method = method, hessian = hessian
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
  if (control$trace) cat("done\n")

  # Extract coefficients
  coefc <- fit$par[1:kx]
  coefz <- fit$par[(kx + (dist == "negbin") + 1):(kx + kz + (dist == "negbin"))]

  # Calculate covariance matrix
  vc <- tryCatch(solve(as.matrix(fit$hessian)),
    error = function(e) {
      warning(e$message, call = FALSE)
      k <- nrow(as.matrix(fit$hessian))
      return(matrix(NA, k, k))
    }
  )

  # Extract theta parameters and standard errors
  np <- c(
    if (dist == "negbin") kx + 1 else NULL,
    if (zero.dist == "negbin") kx + kz + 1 + (dist == "negbin") else NULL
  )

  if (length(np) > 0) {
    theta <- as.vector(exp(fit$par[np]))
    diag_vals <- diag(vc)[np]
    SE.logtheta <- sapply(diag_vals, function(val) {
      if (is.na(val) || val <= 0) NA_real_ else sqrt(val)
    })
    names(theta) <- names(SE.logtheta) <- c(
      if (dist == "negbin") "count" else NULL,
      if (zero.dist == "negbin") "zero" else NULL
    )
    vc <- vc[-np, -np, drop = FALSE]
  } else {
    theta <- NULL
    SE.logtheta <- NULL
  }

  # Return results
  return(list(
    fit = fit,
    coefc = coefc,
    coefz = coefz,
    theta = theta,
    vc = vc,
    SE.logtheta = SE.logtheta,
    loglik = fit$value,
    converged = fit$convergence < 1
  ))
}

#' Calculate Fitted Values for Hurdle Models
#'
#' @param Y Response variable
#' @param X Count model matrix
#' @param Z Zero model matrix
#' @param coefc Count model coefficients
#' @param coefz Zero model coefficients
#' @param offsetx Count model offset
#' @param offsetz Zero model offset
#' @param dist Count distribution
#' @param zero.dist Zero distribution
#' @param theta Dispersion parameters
#' @param linkinv Link inverse function
#'
#' @return List with fitted values and residuals
#' @keywords internal
calculate_fitted_values <- function(Y, X, Z, coefc, coefz, offsetx, offsetz,
                                    dist, zero.dist, theta, linkinv) {
  # Calculate zero component
  phi <- if (zero.dist == "binomial") {
    linkinv(Z %*% coefz + offsetz)[, 1]
  } else {
    exp(Z %*% coefz + offsetz)[, 1]
  }

  # Calculate probability of zero
  p0_zero <- switch(zero.dist,
    "binomial" = log(phi),
    "poisson" = ppois(0, lambda = phi, lower.tail = FALSE, log.p = TRUE),
    "negbin" = pnbinom(0, size = theta["zero"], mu = phi, lower.tail = FALSE, log.p = TRUE),
    "geometric" = pnbinom(0, size = 1, mu = phi, lower.tail = FALSE, log.p = TRUE)
  )

  # Calculate count component
  mu <- exp(X %*% coefc + offsetx)[, 1]

  # Calculate probability of zero in count model
  p0_count <- switch(dist,
    "poisson" = ppois(0, lambda = mu, lower.tail = FALSE, log.p = TRUE),
    "negbin" = pnbinom(0, size = theta["count"], mu = mu, lower.tail = FALSE, log.p = TRUE),
    "geometric" = pnbinom(0, size = 1, mu = mu, lower.tail = FALSE, log.p = TRUE)
  )

  # Calculate fitted values
  Yhat <- exp((p0_zero - p0_count) + log(mu))

  # Calculate residuals
  weights_sqrt <- if (is.null(attr(Y, "weights"))) rep(1, length(Y)) else sqrt(attr(Y, "weights"))
  res <- weights_sqrt * (Y - Yhat)

  # Return fitted values and residuals
  return(list(Yhat = Yhat, res = res))
}

#' Control Parameters for Hurdle Models
#'
#' @description
#' Set control parameters for fitting hurdle models.
#'
#' @param method Optimization method used by optim. Default is "BFGS".
#' @param maxit Maximum number of iterations. Default is 10000.
#' @param trace Logical. If TRUE, information about the fitting process is printed. Default is FALSE.
#' @param separate Logical. If TRUE, count and zero components are estimated separately. Default is TRUE.
#' @param start Optional list with starting values for the parameters.
#' @param ... Additional control parameters passed to optim.
#'
#' @return A list of control parameters.
#'
#' @export
hurdle.control <- function(method = "BFGS", maxit = 10000, trace = FALSE, separate = TRUE, start = NULL, ...) {
  rval <- list(method = method, maxit = maxit, trace = trace, separate = separate, start = start)
  rval <- c(rval, list(...))
  if (!is.null(rval$fnscale)) warning("fnscale must not be modified")
  rval$fnscale <- -1
  if (!is.null(rval$hessian)) warning("hessian must not be modified")
  rval$hessian <- TRUE
  if (is.null(rval$reltol)) rval$reltol <- .Machine$double.eps^(1 / 1.6)
  rval
}

#' Extract Model Coefficients from a Hurdle Model
#'
#' @description
#' Extract the estimated coefficients from a fitted hurdle model.
#'
#' @param object A fitted model object of class "fasthurdle".
#' @param model Character string specifying which model coefficients to extract.
#'   "full" extracts all coefficients, "count" extracts count model coefficients,
#'   and "zero" extracts zero hurdle model coefficients.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A named vector of coefficients.
#'
#' @export
coef.fasthurdle <- function(object, model = c("full", "count", "zero"), ...) {
  model <- match.arg(model)
  rval <- object$coefficients
  rval <- switch(model,
    "full" = structure(c(rval$count, rval$zero),
      .Names = c(
        paste("count", names(rval$count), sep = "_"),
        paste("zero", names(rval$zero), sep = "_")
      )
    ),
    "count" = rval$count,
    "zero" = rval$zero
  )
  rval
}

#' Extract Variance-Covariance Matrix from a Hurdle Model
#'
#' @description
#' Extract the variance-covariance matrix from a fitted hurdle model.
#'
#' @param object A fitted model object of class "fasthurdle".
#' @param model Character string specifying which model coefficients to extract.
#'   "full" extracts the full variance-covariance matrix, "count" extracts the count model
#'   variance-covariance matrix, and "zero" extracts the zero hurdle model variance-covariance matrix.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A variance-covariance matrix.
#'
#' @export
vcov.fasthurdle <- function(object, model = c("full", "count", "zero"), ...) {
  model <- match.arg(model)
  rval <- object$vcov
  if (model == "full") {
    return(rval)
  }

  cf <- object$coefficients[[model]]
  wi <- seq(along = object$coefficients$count)
  rval <- if (model == "count") rval[wi, wi, drop = FALSE] else rval[-wi, -wi, drop = FALSE]
  colnames(rval) <- rownames(rval) <- names(cf)
  return(rval)
}

#' Extract Log-Likelihood from a Hurdle Model
#'
#' @description
#' Extract the log-likelihood from a fitted hurdle model.
#'
#' @param object A fitted model object of class "fasthurdle".
#' @param ... Additional arguments (currently ignored).
#'
#' @return An object of class "logLik".
#'
#' @export
logLik.fasthurdle <- function(object, ...) {
  structure(object$loglik, df = object$n - object$df.residual, nobs = object$n, class = "logLik")
}

#' Print Method for Hurdle Models
#'
#' @description
#' Print a summary of a fitted hurdle model.
#'
#' @param x A fitted model object of class "fasthurdle".
#' @param digits Number of significant digits to use for printing.
#' @param ... Additional arguments passed to print methods.
#'
#' @return The fitted model object (invisibly).
#'
#' @export
print.fasthurdle <- function(x, digits = max(3, getOption("digits") - 3), ...) {
  cat("\nCall:", deparse(x$call, width.cutoff = floor(getOption("width") * 0.85)), "", sep = "\n")

  if (!x$converged) {
    cat("model did not converge\n")
  } else {
    cat(paste("Count model coefficients (truncated ", x$dist$count, " with log link):\n", sep = ""))
    print.default(format(x$coefficients$count, digits = digits), print.gap = 2, quote = FALSE)
    if (x$dist$count == "negbin") cat(paste("Theta =", round(x$theta["count"], digits), "\n"))

    zero_dist <- if (x$dist$zero != "binomial") {
      paste("censored", x$dist$zero, "with log link")
    } else {
      paste("binomial with", x$link, "link")
    }
    cat(paste("\nZero hurdle model coefficients (", zero_dist, "):\n", sep = ""))
    print.default(format(x$coefficients$zero, digits = digits), print.gap = 2, quote = FALSE)
    if (x$dist$zero == "negbin") cat(paste("Theta =", round(x$theta["zero"], digits), "\n"))
    cat("\n")
  }

  invisible(x)
}

#' Summary Method for Hurdle Models
#'
#' @description
#' Compute a summary of a fitted hurdle model.
#'
#' @param object A fitted model object of class "fasthurdle".
#' @param ... Additional arguments (currently ignored).
#'
#' @return An object of class "summary.fasthurdle".
#'
#' @export
summary.fasthurdle <- function(object, ...) {
  ## residuals
  object$residuals <- residuals(object, type = "pearson")

  ## compute z statistics
  kc <- length(object$coefficients$count)
  kz <- length(object$coefficients$zero)
  se <- sqrt(diag(object$vcov))
  coef <- c(object$coefficients$count, object$coefficients$zero)
  if (object$dist$count == "negbin") {
    coef <- c(coef[1:kc], "Log(theta)" = as.vector(log(object$theta["count"])), coef[(kc + 1):(kc + kz)])
    se <- c(se[1:kc], object$SE.logtheta["count"], se[(kc + 1):(kc + kz)])
    kc <- kc + 1
  }
  if (object$dist$zero == "negbin") {
    coef <- c(coef, "Log(theta)" = as.vector(log(object$theta["zero"])))
    se <- c(se, object$SE.logtheta["zero"])
    kz <- kz + 1
  }
  zstat <- coef / se
  pval <- 2 * pnorm(-abs(zstat))
  coef <- cbind(coef, se, zstat, pval)
  colnames(coef) <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)")
  object$coefficients$count <- coef[1:kc, , drop = FALSE]
  object$coefficients$zero <- coef[(kc + 1):(kc + kz), , drop = FALSE]

  ## number of iterations
  object$iterations <- if (!object$separate) {
    tail(na.omit(object$optim$counts$`function`), 1)
  } else {
    tail(na.omit(object$optim$count$counts$`function`), 1) +
      tail(na.omit(object$optim$zero$counts$`function`), 1)
  }

  ## delete some slots
  object$fitted.values <- object$terms <- object$model <- object$y <-
    object$x <- object$levels <- object$contrasts <- object$start <- object$separate <- NULL

  ## return
  class(object) <- "summary.fasthurdle"
  object
}

#' Print Method for Summary of Hurdle Models
#'
#' @description
#' Print a summary of a fitted hurdle model.
#'
#' @param x An object of class "summary.fasthurdle".
#' @param digits Number of significant digits to use for printing.
#' @param ... Additional arguments passed to print methods.
#'
#' @return The summary object (invisibly).
#'
#' @export
print.summary.fasthurdle <- function(x, digits = max(3, getOption("digits") - 3), ...) {
  cat("\nCall:", deparse(x$call, width.cutoff = floor(getOption("width") * 0.85)), "", sep = "\n")

  if (!x$converged) {
    cat("model did not converge\n")
  } else {
    cat("Pearson residuals:\n")
    print(structure(quantile(x$residuals),
      names = c("Min", "1Q", "Median", "3Q", "Max")
    ), digits = digits, ...)

    cat(paste("\nCount model coefficients (truncated ", x$dist$count, " with log link):\n", sep = ""))
    printCoefmat(x$coefficients$count, digits = digits, signif.legend = FALSE)

    zero_dist <- if (x$dist$zero != "binomial") {
      paste("censored", x$dist$zero, "with log link")
    } else {
      paste("binomial with", x$link, "link")
    }
    cat(paste("Zero hurdle model coefficients (", zero_dist, "):\n", sep = ""))
    printCoefmat(x$coefficients$zero, digits = digits, signif.legend = FALSE)

    if (getOption("show.signif.stars") & any(rbind(x$coefficients$count, x$coefficients$zero)[, 4] < 0.1, na.rm = TRUE)) {
      cat("---\nSignif. codes: ", "0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1", "\n")
    }

    if (!is.null(x$theta)) cat(paste("\nTheta:", paste(names(x$theta), round(x$theta, digits), sep = " = ", collapse = ", ")))
    cat(paste("\nNumber of iterations in", x$method, "optimization:", x$iterations, "\n"))
    cat("Log-likelihood:", formatC(x$loglik, digits = digits), "on", x$n - x$df.residual, "Df\n")
  }

  invisible(x)
}

#' Extract Terms from a Hurdle Model
#'
#' @description
#' Extract the terms object from a fitted hurdle model.
#'
#' @param x A fitted model object of class "fasthurdle".
#' @param model Character string specifying which model terms to extract.
#'   "count" extracts count model terms, and "zero" extracts zero hurdle model terms.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A terms object.
#'
#' @export
terms.fasthurdle <- function(x, model = c("count", "zero"), ...) {
  x$terms[[match.arg(model)]]
}

#' Extract Model Matrix from a Hurdle Model
#'
#' @description
#' Extract the model matrix from a fitted hurdle model.
#'
#' @param object A fitted model object of class "fasthurdle".
#' @param model Character string specifying which model matrix to extract.
#'   "count" extracts count model matrix, and "zero" extracts zero hurdle model matrix.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A model matrix.
#'
#' @export
model.matrix.fasthurdle <- function(object, model = c("count", "zero"), ...) {
  model <- match.arg(model)
  if (!is.null(object$x)) {
    rval <- object$x[[model]]
  } else if (!is.null(object$model)) {
    rval <- model.matrix(object$terms[[model]], object$model, contrasts = object$contrasts[[model]])
  } else {
    stop("not enough information in fitted model to return model.matrix")
  }
  return(rval)
}

#' Predict Method for Hurdle Models
#'
#' @description
#' Predict values from a fitted hurdle model.
#'
#' @param object A fitted model object of class "fasthurdle".
#' @param newdata An optional data frame containing the variables needed for prediction.
#' @param type Character string specifying the type of prediction. "response" returns the
#'   expected value, "prob" returns the probability mass function, "count" returns the
#'   expected value of the count component, and "zero" returns the expected value of the
#'   zero hurdle component.
#' @param na.action Function determining what to do with missing values in newdata.
#' @param at Optional vector of counts for which to compute probabilities.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A vector or matrix of predictions.
#'
#' @export
predict.fasthurdle <- function(
    object, newdata, type = c("response", "prob", "count", "zero"),
    na.action = na.pass, at = NULL, ...) {
  type <- match.arg(type)

  ## if no new data supplied
  if (missing(newdata)) {
    if (type != "response") {
      if (!is.null(object$x)) {
        X <- object$x$count
        Z <- object$x$zero
      } else if (!is.null(object$model)) {
        X <- model.matrix(object$terms$count, object$model, contrasts = object$contrasts$count)
        Z <- model.matrix(object$terms$zero, object$model, contrasts = object$contrasts$zero)
      } else {
        stop("predicted probabilities cannot be computed with missing newdata")
      }
      offsetx <- if (is.null(object$offset$count)) rep.int(0, NROW(X)) else object$offset$count
      offsetz <- if (is.null(object$offset$zero)) rep.int(0, NROW(Z)) else object$offset$zero
    } else {
      return(object$fitted.values)
    }
  } else {
    mf <- model.frame(delete.response(object$terms$full), newdata, na.action = na.action, xlev = object$levels)
    X <- model.matrix(delete.response(object$terms$count), mf, contrasts = object$contrasts$count)
    Z <- model.matrix(delete.response(object$terms$zero), mf, contrasts = object$contrasts$zero)
    offsetx <- model_offset_2(mf, terms = object$terms$count, offset = FALSE)
    offsetz <- model_offset_2(mf, terms = object$terms$zero, offset = FALSE)
    if (is.null(offsetx)) offsetx <- rep.int(0, NROW(X))
    if (is.null(offsetz)) offsetz <- rep.int(0, NROW(Z))
    if (!is.null(object$call$offset)) offsetx <- offsetx + eval(object$call$offset, newdata)
  }

  phi <- if (object$dist$zero == "binomial") {
    object$linkinv(Z %*% object$coefficients$zero + offsetz)[, 1]
  } else {
    exp(Z %*% object$coefficients$zero + offsetz)[, 1]
  }
  p0_zero <- switch(object$dist$zero,
    "binomial" = log(phi),
    "poisson" = ppois(0, lambda = phi, lower.tail = FALSE, log.p = TRUE),
    "negbin" = pnbinom(0, size = object$theta["zero"], mu = phi, lower.tail = FALSE, log.p = TRUE),
    "geometric" = pnbinom(0, size = 1, mu = phi, lower.tail = FALSE, log.p = TRUE)
  )

  mu <- exp(X %*% object$coefficients$count + offsetx)[, 1]
  p0_count <- switch(object$dist$count,
    "poisson" = ppois(0, lambda = mu, lower.tail = FALSE, log.p = TRUE),
    "negbin" = pnbinom(0, size = object$theta["count"], mu = mu, lower.tail = FALSE, log.p = TRUE),
    "geometric" = pnbinom(0, size = 1, mu = mu, lower.tail = FALSE, log.p = TRUE)
  )
  logphi <- p0_zero - p0_count

  if (type == "response") rval <- exp(logphi + log(mu))
  if (type == "count") rval <- mu
  if (type == "zero") rval <- exp(logphi)

  ## predicted probabilities
  if (type == "prob") {
    if (!is.null(object$y)) {
      y <- object$y
    } else if (!is.null(object$model)) {
      y <- model.response(object$model)
    } else {
      stop("predicted probabilities cannot be computed for fits with y = FALSE and model = FALSE")
    }

    yUnique <- if (is.null(at)) 0:max(y) else at
    nUnique <- length(yUnique)
    rval <- matrix(NA, nrow = length(mu), ncol = nUnique)
    dimnames(rval) <- list(rownames(X), yUnique)

    rval[, 1] <- 1 - exp(p0_zero)
    switch(object$dist$count,
      "poisson" = {
        for (i in 2:nUnique) rval[, i] <- exp(logphi + dpois(yUnique[i], lambda = mu, log = TRUE))
      },
      "negbin" = {
        for (i in 2:nUnique) rval[, i] <- exp(logphi + dnbinom(yUnique[i], mu = mu, size = object$theta["count"], log = TRUE))
      },
      "geometric" = {
        for (i in 2:nUnique) rval[, i] <- exp(logphi + dnbinom(yUnique[i], mu = mu, size = 1, log = TRUE))
      }
    )
  }

  rval
}

#' Extract Fitted Values from a Hurdle Model
#'
#' @description
#' Extract fitted values from a fitted hurdle model.
#'
#' @param object A fitted model object of class "fasthurdle".
#' @param ... Additional arguments (currently ignored).
#'
#' @return A vector of fitted values.
#'
#' @export
fitted.fasthurdle <- function(object, ...) {
  object$fitted.values
}

#' Extract Residuals from a Hurdle Model
#'
#' @description
#' Extract residuals from a fitted hurdle model.
#'
#' @param object A fitted model object of class "fasthurdle".
#' @param type Character string specifying the type of residuals. "pearson" returns
#'   Pearson residuals, and "response" returns response residuals.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A vector of residuals.
#'
#' @export
residuals.fasthurdle <- function(object, type = c("pearson", "response"), ...) {
  type <- match.arg(type)
  res <- object$residuals

  switch(type,
    "response" = {
      return(res)
    },
    "pearson" = {
      mu <- predict(object, type = "count")
      phi <- predict(object, type = "zero")
      theta1 <- switch(object$dist$count,
        "poisson" = 0,
        "geometric" = 1,
        "negbin" = 1 / object$theta["count"]
      )
      vv <- object$fitted.values * (1 + ((1 - phi) + theta1) * mu)
      return(res / sqrt(vv))
    }
  )
}

#' Predict Probabilities from a Hurdle Model
#'
#' @description
#' Predict probabilities from a fitted hurdle model.
#'
#' @param obj A fitted model object of class "fasthurdle".
#' @param ... Additional arguments passed to predict.fasthurdle.
#'
#' @return A matrix of predicted probabilities.
#'
#' @export
predprob.fasthurdle <- function(obj, ...) {
  predict(obj, type = "prob", ...)
}

#' Extract AIC from a Hurdle Model
#'
#' @description
#' Extract AIC from a fitted hurdle model.
#'
#' @param fit A fitted model object of class "fasthurdle".
#' @param scale Scale parameter for the AIC. Not used.
#' @param k Penalty per parameter to be used in AIC.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A numeric vector of length 2, with the degrees of freedom and the AIC.
#'
#' @export
extractAIC.fasthurdle <- function(fit, scale = NULL, k = 2, ...) {
  c(attr(logLik(fit), "df"), AIC(fit, k = k))
}

#' Test for Equality of Count and Zero Model Coefficients
#'
#' @description
#' Test for equality of count and zero model coefficients in a hurdle model.
#'
#' @param object A fitted model object of class "fasthurdle".
#' @param ... Additional arguments passed to car::linearHypothesis.
#'
#' @return An object of class "anova" containing the results of the test.
#'
#' @export
hurdletest <- function(object, ...) {
  stopifnot(inherits(object, "fasthurdle"))
  stopifnot(object$dist$count == object$dist$zero)
  stopifnot(all(sort(names(object$coefficients$count)) == sort(names(object$coefficients$zero))))
  stopifnot(requireNamespace("car"))
  nam <- names(object$coefficients$count)
  lh <- paste("count_", nam, " = ", "zero_", nam, sep = "")
  rval <- car::linearHypothesis(object, lh, ...)
  attr(rval, "heading")[1] <- "Wald test for hurdle models\n\nRestrictions:"
  return(rval)
}

#' Extract Offset from a Model Frame
#'
#' @description
#' Extract offset from a model frame, optionally using different terms.
#'
#' @param x A model frame.
#' @param terms Terms object. If NULL, the terms attribute of x is used.
#' @param offset Logical. If TRUE, the "(offset)" column is included.
#'
#' @return A numeric vector of offsets.
#'
#' @keywords internal
model_offset_2 <- function(x, terms = NULL, offset = TRUE) {
  if (is.null(terms)) terms <- attr(x, "terms")
  offsets <- attr(terms, "offset")
  if (length(offsets) > 0) {
    ans <- if (offset) x$"(offset)" else NULL
    if (is.null(ans)) ans <- 0
    for (i in offsets) ans <- ans + x[[deparse(attr(terms, "variables")[[i + 1]])]]
    ans
  } else {
    ans <- if (offset) x$"(offset)" else NULL
  }
  if (!is.null(ans) && !is.numeric(ans)) stop("'offset' must be numeric")
  ans
}

#' Null-coalescing operator
#'
#' @description
#' Return the first non-NULL value.
#'
#' @param x First value.
#' @param y Second value.
#'
#' @return x if x is not NULL, otherwise y.
#'
#' @keywords internal
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
