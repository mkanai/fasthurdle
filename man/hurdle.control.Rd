% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fasthurdle.R
\name{hurdle.control}
\alias{hurdle.control}
\title{Control Parameters for Hurdle Models}
\usage{
hurdle.control(
  method = "BFGS",
  maxit = 10000,
  trace = FALSE,
  separate = TRUE,
  start = NULL,
  ...
)
}
\arguments{
\item{method}{Optimization method used by optim. Default is "BFGS".}

\item{maxit}{Maximum number of iterations. Default is 10000.}

\item{trace}{Logical. If TRUE, information about the fitting process is printed. Default is FALSE.}

\item{separate}{Logical. If TRUE, count and zero components are estimated separately. Default is TRUE.}

\item{start}{Optional list with starting values for the parameters.}

\item{...}{Additional control parameters passed to optim.}
}
\value{
A list of control parameters.
}
\description{
Set control parameters for fitting hurdle models.
}
