#' @import Rcpp
#' @useDynLib fasthurdle, .registration = TRUE
#' @importFrom stats .getXlevels AIC binomial delete.response dnbinom dpois
#'   logLik make.link model.frame model.matrix model.response model.weights
#'   na.omit na.pass plogis pnbinom pnorm poisson ppois predict printCoefmat
#'   quantile residuals terms update
#' @importFrom utils tail
NULL
