% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fast_negbin_hurdle.R
\name{fast_negbin_hurdle}
\alias{fast_negbin_hurdle}
\title{Fast Negative Binomial Hurdle Model with Binomial Zero Hurdle}
\usage{
fast_negbin_hurdle(X, y, method = "BFGS", maxit = 10000, separate = TRUE)
}
\arguments{
\item{X}{Model matrix for both count and zero components}

\item{y}{Response vector of counts}

\item{method}{Optimization method used by optim. Default is "BFGS".}

\item{maxit}{Maximum number of iterations. Default is 10000.}

\item{separate}{Logical. If TRUE, count and zero components are estimated separately. Default is TRUE.}
}
\value{
An object of class "fasthurdle" representing the fitted model.
}
\description{
A specialized version of fasthurdle that only handles negative binomial count model
with binomial zero hurdle model and logit link. This function is optimized
for speed by skipping all unnecessary parameter checks and validations.
}
\details{
This function is a specialized version of fasthurdle that only handles the specific
parameter combination: dist = "negbin", zero.dist = "binomial", link = "logit".
It takes a model matrix and response vector directly, skipping formula processing
and other parameter validations to improve performance.
}
\examples{
\dontrun{
# Load example data
data(bioChemists, package = "pscl")

# Create model matrix and response vector
X <- model.matrix(~ fem + mar + kid5 + phd + ment, data = bioChemists)
y <- bioChemists$art

# Fit the model
m <- fast_negbin_hurdle(X, y)
summary(m)
}

}
