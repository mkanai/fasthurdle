# fasthurdle

A fast implementation of hurdle models using Rcpp. This package provides the same functionality as [the `hurdle` function](https://www.rdocumentation.org/packages/pscl/versions/1.5.9/topics/hurdle) in [the `pscl` package](https://github.com/atahk/pscl), but with improved performance through C++ implementations of key functions.

## Installation

You can install the development version of fasthurdle from GitHub with:

```r
# install.packages("pak")
pak::pkg_install("mkanai/fasthurdle")
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

## Benchmark Results

Average speedup of `fasthurdle` compared to `pscl::hurdle`:

| Count Model | Zero Hurdle | Speedup Factor |
| ----------- | ----------- | -------------- |
| geometric   | binomial    | 1.8x           |
| geometric   | geometric   | 1.9x           |
| geometric   | negbin      | 2.8x           |
| geometric   | poisson     | 2.6x           |
| negbin      | binomial    | 3.6x           |
| negbin      | geometric   | 3.5x           |
| negbin      | negbin      | 3.5x           |
| negbin      | poisson     | 3.8x           |
| poisson     | binomial    | 2.8x           |
| poisson     | geometric   | 2.6x           |
| poisson     | negbin      | 3.2x           |
| poisson     | poisson     | 3.4x           |

_Note: Benchmarks run with sample sizes 1,000, 10,000, 100,000, and 1,000,000. Speedup factor is the ratio of pscl execution time to fasthurdle execution time._

## Features

- Supports the same models as `pscl::hurdle`:
  - Count distributions: Poisson, Negative Binomial, Geometric
  - Zero hurdle distributions: Binomial, Poisson, Negative Binomial, Geometric
- Compatible API with `pscl::hurdle`
- Improved performance through C++ implementations

## Acknowledgements

[The `pscl` package](https://github.com/atahk/pscl), where the original hurdle function was implemented, was developed at the Political Science Computational Laboratory, led by Simon Jackman at Stanford University. The hurdle and count data models in the package were re-written by Achim Zeileis and Christian Kleiber.

## License

GPL-2

## Contact

Masahiro Kanai (mkanai@broadinstitute.org)
