# fasthurdle

A fast implementation of hurdle models using Rcpp. This package provides the same functionality as the `hurdle` function in the `pscl` package, but with improved performance through C++ implementations of key functions.

## Installation

You can install the development version of fasthurdle from GitHub with:

```r
# install.packages("remotes")
remotes::install_github("mkanai/fasthurdle")
```

## Usage

```r
library(fasthurdle)

# Generate some sample data
set.seed(123)
n <- 500
x <- rnorm(n)
z <- rnorm(n)
lambda <- exp(1 + 0.5 * x)
p <- plogis(0.5 - 0.5 * z)
y <- rbinom(n, size = 1, prob = p) * rpois(n, lambda = lambda)

# Create a data frame
df <- data.frame(y = y, x = x, z = z)

# Fit a hurdle model
model <- fasthurdle(y ~ x | z, data = df, dist = "poisson", zero.dist = "binomial")

# Print the model summary
summary(model)
```

## Features

- Supports the same models as `pscl::hurdle`:
  - Count distributions: Poisson, Negative Binomial, Geometric
  - Zero hurdle distributions: Binomial, Poisson, Negative Binomial, Geometric
- Compatible API with `pscl::hurdle`
- Improved performance through C++ implementations

## License

GPL-2
