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
|------------|------------|---------------|
| geometric  | binomial   | 2.2x          |
| geometric  | geometric  | 2.3x          |
| geometric  | negbin     | 5.1x          |
| geometric  | poisson    | 2.9x          |
| negbin     | binomial   | 13x           |
| negbin     | geometric  | 12.1x         |
| negbin     | negbin     | 11x           |
| negbin     | poisson    | 11.3x         |
| poisson    | binomial   | 3.4x          |
| poisson    | geometric  | 3.1x          |
| poisson    | negbin     | 5.8x          |
| poisson    | poisson    | 3.7x          |

*Note: Benchmarks run with sample sizes 1000, 10000, 1e+05. Speedup factor is the ratio of pscl execution time to fasthurdle execution time.*

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

Masahiro Kanai (<mkanai@broadinstitute.org>)
