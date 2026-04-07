#' Batch Score Test Scan for Hurdle Models
#'
#' @description
#' Tests multiple peaks (test variables) against a hurdle model in a single call.
#' This is the primary production entry point for high-throughput cis-QTL mapping.
#'
#' P-values are returned as \code{-log10(p)} to prevent underflow at genome-wide
#' significance levels.
#'
#' @param X Model matrix (n x p) containing both null covariates and test variables.
#' @param y Response vector of counts.
#' @param peaks Character vector of column names in \code{X} to test (required).
#' @param Z Optional model matrix for the zero component. Default is NULL (use X).
#' @param offsetx Optional offset vector for the count model. Default is NULL.
#' @param offsetz Optional offset vector for the zero model. Default is NULL.
#' @param null_fit_count Optional pre-fitted count null model (from
#'   \code{\link{fit_null_count}}).
#' @param null_fit_zero Optional pre-fitted zero null model (from
#'   \code{\link{fit_null_zero}}).
#' @param spa_cutoff Numeric or NULL. Apply saddlepoint approximation when |z|
#'   exceeds this cutoff. Default is NULL (disabled).
#'
#' @return A \code{data.frame} with columns:
#'   \item{peak}{Peak/test variable name.}
#'   \item{nlog10p_count}{-log10(p-value) for the count component.}
#'   \item{beta_count}{Effect size estimate for the count component.}
#'   \item{se_count}{Standard error for the count component.}
#'   \item{stat_count}{Test statistic for the count component.}
#'   \item{nlog10p_zero}{-log10(p-value) for the zero component (if test vars
#'     are found in Z).}
#'   \item{beta_zero}{Effect size estimate for the zero component.}
#'   \item{se_zero}{Standard error for the zero component.}
#'   \item{stat_zero}{Test statistic for the zero component.}
#'
#'   Attributes \code{null_fit_count} and \code{null_fit_zero} are attached for
#'   reuse across multiple genes.
#'
#' @examples
#' \dontrun{
#' # Standard batch score test
#' result <- hurdle_scan(X, y, peaks = c("peak1", "peak2", "peak3"))
#' }
#'
#' @export
hurdle_scan <- function(X, y, peaks, Z = NULL,
                        offsetx = NULL, offsetz = NULL,
                        null_fit_count = NULL, null_fit_zero = NULL,
                        spa_cutoff = NULL) {
  # Validate peaks
  if (!is.character(peaks) || length(peaks) == 0) {
    stop("peaks must be a non-empty character vector of column names in X")
  }
  test_idx <- match(peaks, colnames(X))
  bad <- is.na(test_idx)
  if (any(bad)) {
    stop("peaks not found in X: ", paste(peaks[bad], collapse = ", "))
  }
  if (!is.null(Z)) {
    bad_z <- is.na(match(peaks, colnames(Z)))
    if (any(bad_z)) {
      stop("peaks not found in Z: ", paste(peaks[bad_z], collapse = ", "))
    }
  }

  # Setup
  n <- length(y)
  if (is.null(Z)) Z <- X
  kx <- NCOL(X)
  kz <- NCOL(Z)
  weights <- rep.int(1, n)
  if (is.null(offsetx)) offsetx <- rep.int(0, n)
  if (is.null(offsetz)) offsetz <- rep.int(0, n)

  null_idx <- setdiff(seq_len(kx), test_idx)
  X_null <- X[, null_idx, drop = FALSE]

  # ---- Count component ----
  null_fit_count <- ensure_null_count(
    null_fit_count, X_null, y,
    offsetx = offsetx, weights = weights,
    dist = "negbin"
  )

  sc <- null_fit_count$score_cache
  use_spa <- !is.null(spa_cutoff) && is.finite(spa_cutoff)
  spa_cutoff_val <- if (use_spa) spa_cutoff else 1e30
  has_theta <- isTRUE(sc$has_theta)

  X_test_pos <- X[sc$Y1 + 1L, test_idx, drop = FALSE]

  batch_count <- score_test_count_batch_cpp(
    X_test_pos, sc$Y1, sc$grad_weights, sc$v_ee, sc$Y_pos,
    sc$I_nn_inv, sc$I_nn_beta_inv, sc$beta_inv_ok,
    sc$Xnull_vee_t, sc$X_null_pos, sc$w_pos, sc$theta,
    sc$beta_null, sc$eta_null_pos,
    sc$mu_pos, sc$p0_pos, sc$log_p1_pos, sc$kx_null,
    has_theta, use_spa, spa_cutoff_val,
    if (has_theta) sc$v_et else NULL
  )

  nlog10p_count <- stat_nlog10p_chisq(
    (batch_count$beta / batch_count$se)^2
  )

  # ---- Zero component ----
  zero_test_idx <- match(peaks, colnames(Z))
  zero_null_idx <- setdiff(seq_len(kz), zero_test_idx)
  Z_null_zero <- Z[, zero_null_idx, drop = FALSE]

  null_fit_zero <- ensure_null_zero(
    null_fit_zero, Z_null_zero, y,
    offsetz = offsetz, weights = weights
  )

  n_peaks_z <- length(zero_test_idx)
  pv <- bv <- sv <- stv <- numeric(n_peaks_z)
  for (k in seq_len(n_peaks_z)) {
    rz <- score_test_zero(
      Z_null_zero, Z[, zero_test_idx[k]], y,
      offsetz = offsetz, weights = weights,
      null_fit_zero = null_fit_zero, spa_cutoff = spa_cutoff
    )
    pv[k] <- rz$pvalue
    bv[k] <- rz$beta[1]
    sv[k] <- rz$se[1]
    stv[k] <- rz$statistic
  }
  batch_zero <- list(pvalue = pv, beta = bv, se = sv, statistic = stv)
  nlog10p_zero <- stat_nlog10p_chisq((bv / sv)^2)

  # ---- Assemble result ----
  result <- data.frame(
    peak = peaks,
    nlog10p_count = nlog10p_count,
    beta_count = batch_count$beta,
    se_count = batch_count$se,
    stat_count = batch_count$statistic,
    nlog10p_zero = nlog10p_zero,
    beta_zero = batch_zero$beta,
    se_zero = batch_zero$se,
    stat_zero = batch_zero$statistic,
    stringsAsFactors = FALSE
  )
  attr(result, "null_fit_count") <- null_fit_count
  attr(result, "null_fit_zero") <- null_fit_zero
  result
}

# Compute exact -log10(p) from chi-squared(df=1) statistic using log.p=TRUE.
stat_nlog10p_chisq <- function(stat) {
  nlog10p <- -pchisq(stat, df = 1, lower.tail = FALSE, log.p = TRUE) / log(10)
  nlog10p[is.na(stat)] <- NA_real_
  nlog10p
}
