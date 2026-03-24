#' Cauchy Combination Test (CCT / ACAT)
#'
#' Combines p-values using the Cauchy distribution. This is an analytical
#' p-value combination method that is valid under arbitrary dependency
#' structures between the input p-values.
#'
#' Based on the STAAR package implementation. When any p-value equals 1, the
#' original STAAR implementation returns 1. We adopt the SAIGE-QTL modification
#' that instead returns a Bonferroni-corrected p-value (\code{min(1, min(p)*n)}),
#' which is more conservative and informative in this edge case.
#'
#' @param pvals A numeric vector of p-values between 0 and 1.
#' @param weights A numeric vector of non-negative weights. If \code{NULL},
#'   equal weights are used.
#' @return A single combined p-value.
#' @references
#' Liu, Y., & Xie, J. (2020). Cauchy combination test: a powerful test with
#' analytic p-value calculation under arbitrary dependency structures.
#' \emph{Journal of the American Statistical Association 115}(529), 393-402.
#' @importFrom stats pcauchy
#' @export
CCT <- function(pvals, weights = NULL) {
  if (any(is.na(pvals))) stop("NAs not allowed in 'pvals'")
  if (any(pvals < 0) || any(pvals > 1)) stop("all 'pvals' must be between 0 and 1")

  if (any(pvals == 0)) {
    return(0)
  }
  # When any p-value is exactly 1, the Cauchy statistic is undefined.
  # STAAR returns 1; SAIGE-QTL instead returns Bonferroni min(1, min(p)*n).
  if (any(pvals == 1)) {
    return(min(1, min(pvals) * length(pvals)))
  }

  # Default: equal weights; otherwise validate and standardize
  if (is.null(weights)) {
    weights <- rep(1 / length(pvals), length(pvals))
  } else {
    if (length(weights) != length(pvals)) {
      stop("'weights' must have the same length as 'pvals'")
    }
    if (any(weights < 0)) stop("all 'weights' must be non-negative")
    weights <- weights / sum(weights)
  }

  # For very small p-values, use the asymptotic approximation tan((0.5-p)*pi) ~ 1/(p*pi)
  is.small <- (pvals < 1e-16)
  if (sum(is.small) == 0) {
    cct.stat <- sum(weights * tan((0.5 - pvals) * pi))
  } else {
    cct.stat <- sum((weights[is.small] / pvals[is.small]) / pi) +
      sum(weights[!is.small] * tan((0.5 - pvals[!is.small]) * pi))
  }

  if (cct.stat > 1e+15) (1 / cct.stat) / pi else 1 - stats::pcauchy(cct.stat)
}

#' Jiang-Doerge two-stage FDR procedure for hurdle models
#'
#' Performs a two-stage FDR procedure where stage 1 screens on one set of
#' p-values (e.g., zero-inflation) and stage 2 confirms using a second set
#' (e.g., count model), with FDR adjustment accounting for the selection step.
#'
#' @param p_stage1 Numeric vector of stage 1 p-values (screening).
#' @param p_stage2 Numeric vector of stage 2 p-values (confirmation).
#' @param alpha1 FDR threshold for stage 1 screening (default: 0.1).
#' @param alpha2 FDR threshold for stage 2 confirmation (default: 0.05).
#' @return A list with components:
#'   \item{selected}{Logical vector of pairs passing stage 1.}
#'   \item{significant}{Logical vector of pairs significant after both stages.}
#'   \item{n_selected}{Number of pairs selected in stage 1.}
#'   \item{n_significant}{Number of significant pairs.}
#'   \item{pi0_selected}{Estimated proportion of true nulls among selected.}
#'   \item{alpha2_adjusted}{Adjusted alpha for stage 2.}
#'   \item{q_stage1}{BH-adjusted q-values from stage 1.}
#'   \item{q_stage2}{BH-adjusted q-values from stage 2 (1 for unselected).}
#' @references
#' Jiang, H. & Doerge, R.W. (2006). A two-step multiple comparison procedure
#' for a large number of tests and multiple treatments. \emph{Statistical
#' Applications in Genetics and Molecular Biology}, 5, Article28.
#' @importFrom stats p.adjust
#' @export
jiang_doerge_fdr <- function(p_stage1, p_stage2, alpha1 = 0.1, alpha2 = 0.05) {
  n <- length(p_stage1)
  if (length(p_stage2) != n) stop("'p_stage1' and 'p_stage2' must have the same length")

  # Stage 1: Liberal screening
  q_stage1 <- stats::p.adjust(p_stage1, method = "BH")
  selected <- q_stage1 < alpha1
  n_selected <- sum(selected)

  if (n_selected == 0) {
    return(list(
      selected = selected, significant = rep(FALSE, n),
      n_selected = 0L, n_significant = 0L,
      pi0_selected = NA_real_, alpha2_adjusted = NA_real_,
      q_stage1 = q_stage1, q_stage2 = rep(1, n)
    ))
  }

  # Estimate proportion of true nulls among selected (Jiang-Doerge estimator)
  pi0_selected <- min(1, sum(p_stage2[selected] > 0.5) * 2 / n_selected)

  # Adjust alpha2 to account for selection
  alpha2_adjusted <- alpha2 * (1 - pi0_selected * alpha1)

  # Stage 2: BH on selected pairs with adjusted threshold
  q_stage2 <- rep(1, n)
  q_stage2[selected] <- stats::p.adjust(p_stage2[selected], method = "BH")
  significant <- selected & (q_stage2 < alpha2_adjusted)

  list(
    selected = selected, significant = significant,
    n_selected = n_selected, n_significant = sum(significant),
    pi0_selected = pi0_selected, alpha2_adjusted = alpha2_adjusted,
    q_stage1 = q_stage1, q_stage2 = q_stage2
  )
}

#' ACAT stage-wise testing for hurdle model FDR control
#'
#' Combines zero-inflation and count model p-values using the Cauchy
#' combination test (CCT/ACAT) for omnibus screening, then applies stage-wise
#' confirmation via the Holm procedure to classify the regulatory mode.
#' This provides overall FDR control at the screening level with family-wise
#' error rate control within each selected pair.
#'
#' @param p_zero Numeric vector of zero-inflation model p-values.
#' @param p_count Numeric vector of count model p-values.
#' @param alpha Significance threshold for both screening and confirmation
#'   (default: 0.05).
#' @param pi0.method Method for estimating the proportion of true nulls in
#'   \code{\link[qvalue]{qvalue}}. Either \code{"smoother"} (default) or
#'   \code{"bootstrap"}.
#' @return A data.frame with columns:
#'   \item{p_omnibus}{CCT-combined omnibus p-value.}
#'   \item{q_omnibus}{qvalue-based FDR for the omnibus test.}
#'   \item{selected}{Logical; TRUE if q_omnibus < alpha.}
#'   \item{p_adj_zero}{Holm-adjusted zero-model p-value (NA if not selected).}
#'   \item{p_adj_count}{Holm-adjusted count-model p-value (NA if not selected).}
#'   \item{sig_zero}{Logical; zero component significant after Holm correction.}
#'   \item{sig_count}{Logical; count component significant after Holm correction.}
#'   \item{mode}{Factor classifying the regulatory mode as "dual", "switch",
#'     "rheostat", "joint_only", or "not_significant".}
#'   \item{sig}{Logical; TRUE if mode is not "not_significant".}
#' @references
#' Van den Berge, K., et al. (2017). stageR: a general stage-wise method for
#' controlling the gene-level false discovery rate in differential expression
#' and differential transcript usage. \emph{Genome Biology}, 18, 151.
#'
#' Liu, Y., & Xie, J. (2020). Cauchy combination test: a powerful test with
#' analytic p-value calculation under arbitrary dependency structures.
#' \emph{Journal of the American Statistical Association}, 115(529), 393-402.
#'
#' Storey, J.D. & Tibshirani, R. (2003). Statistical significance for
#' genome-wide experiments. \emph{Proceedings of the National Academy of
#' Sciences}, 100, 9440-9445.
#' @export
acat_stagewise <- function(p_zero, p_count, alpha = 0.05,
                           pi0.method = c("smoother", "bootstrap")) {
  pi0.method <- match.arg(pi0.method)
  n <- length(p_zero)
  if (length(p_count) != n) stop("'p_zero' and 'p_count' must have the same length")
  if (!requireNamespace("qvalue", quietly = TRUE)) {
    stop("package 'qvalue' is required; install with: BiocManager::install(\"qvalue\")")
  }

  # Screening: CCT omnibus test
  p_omnibus <- mapply(function(pz, pc) {
    if (is.na(pz) || is.na(pc)) NA_real_ else CCT(c(pz, pc))
  }, p_zero, p_count)

  # FDR control via qvalue with bootstrap pi0 estimation
  q_omnibus <- rep(NA_real_, n)
  ok <- !is.na(p_omnibus)
  if (sum(ok) > 1) {
    q_omnibus[ok] <- qvalue::qvalue(p_omnibus[ok], pi0.method = pi0.method)$qvalues
  }

  selected <- !is.na(q_omnibus) & q_omnibus < alpha

  # Confirmation: Holm procedure within each selected pair
  p_adj_zero <- rep(NA_real_, n)
  p_adj_count <- rep(NA_real_, n)
  idx <- which(selected)
  for (i in idx) {
    holm <- stats::p.adjust(c(p_zero[i], p_count[i]), method = "holm")
    p_adj_zero[i] <- holm[1]
    p_adj_count[i] <- holm[2]
  }

  sig_zero <- !is.na(p_adj_zero) & p_adj_zero < alpha
  sig_count <- !is.na(p_adj_count) & p_adj_count < alpha

  mode <- rep("not_significant", n)
  mode[selected & sig_zero & sig_count] <- "dual"
  mode[selected & sig_zero & !sig_count] <- "switch"
  mode[selected & !sig_zero & sig_count] <- "rheostat"
  mode[selected & !sig_zero & !sig_count] <- "joint_only"
  mode <- factor(mode,
    levels = c("dual", "switch", "rheostat", "joint_only", "not_significant")
  )

  data.frame(
    p_omnibus = p_omnibus,
    q_omnibus = q_omnibus,
    selected = selected,
    p_adj_zero = p_adj_zero,
    p_adj_count = p_adj_count,
    sig_zero = sig_zero,
    sig_count = sig_count,
    mode = mode,
    sig = mode != "not_significant"
  )
}

#' Joint 2-df score test with stage-wise mode classification
#'
#' Combines zero and count model score test statistics via a joint chi-squared(2)
#' test for omnibus screening, then applies Holm step-down for mode classification.
#' Under the factorized hurdle likelihood, the zero and count score statistics are
#' independent, so the joint statistic is simply their sum: T_joint = T_zero + T_count.
#' This is more principled than ACAT (Cauchy combination) which uses an ad-hoc
#' heavy-tailed combination.
#'
#' Takes chi-squared statistics directly (not p-values) to avoid underflow at
#' large sample sizes where p-values can be numerically zero.
#'
#' @param chisq_zero Numeric vector of zero-model chi-squared statistics (1 df).
#'   From \code{score_test_zero()$statistic} or \code{summary()$coefficients$zero[,"z value"]^2}.
#' @param chisq_count Numeric vector of count-model chi-squared statistics (1 df).
#'   From \code{score_test_count()$statistic} or \code{summary()$coefficients$count[,"z value"]^2}.
#' @param alpha Significance threshold for both screening and confirmation
#'   (default: 0.05).
#' @param pi0.method Method for estimating the proportion of true nulls in
#'   \code{\link[qvalue]{qvalue}}. Either \code{"smoother"} (default) or
#'   \code{"bootstrap"}.
#' @return A data.frame with columns:
#'   \item{chisq_joint}{Joint chi-squared(2) statistic (T_zero + T_count).}
#'   \item{p_joint}{Joint p-value from chi-squared(2) distribution.}
#'   \item{q_joint}{qvalue-based FDR for the joint test.}
#'   \item{selected}{Logical; TRUE if q_joint < alpha.}
#'   \item{p_adj_zero}{Holm-adjusted zero-model p-value (NA if not selected).}
#'   \item{p_adj_count}{Holm-adjusted count-model p-value (NA if not selected).}
#'   \item{sig_zero}{Logical; zero component significant after Holm correction.}
#'   \item{sig_count}{Logical; count component significant after Holm correction.}
#'   \item{mode}{Factor classifying the regulatory mode as "dual", "switch",
#'     "rheostat", "joint_only", or "not_significant".}
#'   \item{sig}{Logical; TRUE if mode is not "not_significant".}
#' @references
#' Van den Berge, K., et al. (2017). stageR: a general stage-wise method for
#' controlling the gene-level false discovery rate in differential expression
#' and differential transcript usage. \emph{Genome Biology}, 18, 151.
#' @importFrom stats pchisq p.adjust
#' @export
joint_score_test <- function(chisq_zero, chisq_count,
                              alpha = 0.05,
                              pi0.method = c("smoother", "bootstrap")) {
  pi0.method <- match.arg(pi0.method)
  n <- length(chisq_zero)
  if (length(chisq_count) != n) {
    stop("'chisq_zero' and 'chisq_count' must have the same length")
  }
  if (!requireNamespace("qvalue", quietly = TRUE)) {
    stop("package 'qvalue' is required; install with: BiocManager::install(\"qvalue\")")
  }

  # Joint statistic: T_zero + T_count ~ chi2(2)
  # Valid because zero and count scores are independent under factorized hurdle
  chisq_joint <- chisq_zero + chisq_count
  p_joint <- ifelse(is.na(chisq_joint), NA_real_,
                     pchisq(chisq_joint, df = 2, lower.tail = FALSE))

  # FDR control via qvalue (fall back to BH if qvalue fails, e.g., all p < lambda)
  q_joint <- rep(NA_real_, n)
  ok <- !is.na(p_joint)
  if (sum(ok) > 1) {
    q_joint[ok] <- tryCatch(
      qvalue::qvalue(p_joint[ok], pi0.method = pi0.method)$qvalues,
      error = function(e) stats::p.adjust(p_joint[ok], method = "BH")
    )
  }

  selected <- !is.na(q_joint) & q_joint < alpha

  # Confirmation: Holm procedure within each selected pair
  # Compute component p-values from chi-squared stats (avoids requiring separate p inputs)
  p_zero <- pchisq(chisq_zero, df = 1, lower.tail = FALSE)
  p_count <- pchisq(chisq_count, df = 1, lower.tail = FALSE)

  p_adj_zero <- rep(NA_real_, n)
  p_adj_count <- rep(NA_real_, n)
  idx <- which(selected)
  for (i in idx) {
    holm <- stats::p.adjust(c(p_zero[i], p_count[i]), method = "holm")
    p_adj_zero[i] <- holm[1]
    p_adj_count[i] <- holm[2]
  }

  sig_zero <- !is.na(p_adj_zero) & p_adj_zero < alpha
  sig_count <- !is.na(p_adj_count) & p_adj_count < alpha

  mode <- rep("not_significant", n)
  mode[selected & sig_zero & sig_count] <- "dual"
  mode[selected & sig_zero & !sig_count] <- "switch"
  mode[selected & !sig_zero & sig_count] <- "rheostat"
  mode[selected & !sig_zero & !sig_count] <- "joint_only"
  mode <- factor(mode,
    levels = c("dual", "switch", "rheostat", "joint_only", "not_significant")
  )

  data.frame(
    chisq_joint = chisq_joint,
    p_joint = p_joint,
    q_joint = q_joint,
    selected = selected,
    p_adj_zero = p_adj_zero,
    p_adj_count = p_adj_count,
    sig_zero = sig_zero,
    sig_count = sig_count,
    mode = mode,
    sig = mode != "not_significant"
  )
}
