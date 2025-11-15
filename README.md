# fasthurdle

A fast implementation of hurdle models using Rcpp. This package provides the same functionality as [the `hurdle` function](https://www.rdocumentation.org/packages/pscl/versions/1.5.9/topics/hurdle) in [the `pscl` package](https://github.com/atahk/pscl), but with improved performance through C++ implementations of key functions. This package is optimized for efficient peak-gene link analysis in large-scale single-nucleus multiome datasets with millions of cells.

## Installation

You can install the development version of fasthurdle from GitHub with:

```r
# install.packages("pak")
pak::pkg_install("mkanai/fasthurdle")
```

## General usage

```r
library(fasthurdle)

# Generate some sample data
set.seed(42)
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

**Note:** `fasthurdle` uses OpenMP multithreading for improved performance. Control the number of threads using the `OMP_NUM_THREADS` environment variable. When fitting many models, avoid additional parallelization (e.g., with `parallel` or `future`) to prevent oversubscription.

## Peak-gene link analysis

Hurdle models are well-suited for analyzing peak-gene associations in single-nucleus multiome data (scRNA-seq + scATAC-seq), as originally introduced in [Open4Gene](https://github.com/hbliu/Open4Gene). The two-part hurdle model independently fits: (1) a binomial zero-inflation model testing whether peak accessibility affects the probability of nonzero expression, and (2) a negative binomial count model testing whether peak accessibility affects expression magnitude among expressing cells. This approach explicitly accounts for technical and biological sparsity while modeling the count-based nature of expression measurements with overdispersion.

### Using `fasthurdle` with formula interface

```r
library(fasthurdle)

# Generate sample data
set.seed(42)
n <- 500
peak_acc <- rpois(n, lambda = 2)
log_total_counts <- rnorm(n, mean = 8, sd = 1)
pct_counts_mito <- runif(n, min = 0, max = 0.2)
lambda <- exp(0.5 + 0.3 * peak_acc + 0.2 * log_total_counts - 0.5 * pct_counts_mito)
p <- plogis(1 - 0.4 * peak_acc - 0.3 * pct_counts_mito)
gene_expr <- rbinom(n, size = 1, prob = p) * rpois(n, lambda = lambda)

# Create a data frame
df <- data.frame(
  gene_expr = gene_expr,
  peak_acc = peak_acc,
  log_total_counts = log_total_counts,
  pct_counts_mito = pct_counts_mito
)

# Fit hurdle model with negative binomial count model and binomial zero hurdle
# Note: You can also use other distributions as needed (dist: "poisson", "geometric"; zero.dist: "poisson", "negbin", "geometric")
model <- fasthurdle(
  gene_expr ~ peak_acc + log_total_counts + pct_counts_mito,
  data = df,
  dist = "negbin",
  zero.dist = "binomial"
)

# Extract results
model_summary <- summary(model)

# Get coefficients for peak accessibility
peak_coef_zero <- model_summary$coefficients$zero["peak_acc", ]    # Zero hurdle component
peak_coef_count <- model_summary$coefficients$count["peak_acc", ]  # Count component

# Extract specific statistics
beta_zero <- peak_coef_zero[1]    # Coefficient (log-odds of nonzero expression per unit increase in peak accessibility)
se_zero <- peak_coef_zero[2]      # Standard error
z_zero <- peak_coef_zero[3]       # Z-statistic
p_zero <- peak_coef_zero[4]       # P-value

beta_count <- peak_coef_count[1]  # Coefficient (change in log gene expression counts per unit increase in peak accessibility)
se_count <- peak_coef_count[2]    # Standard error
z_count <- peak_coef_count[3]     # Z-statistic
p_count <- peak_coef_count[4]     # P-value

# Model fit statistics
aic <- AIC(model)
bic <- BIC(model)
```

### Using `fast_negbin_hurdle` for high-performance analysis

For large-scale peak-gene pair testing, use `fast_negbin_hurdle`, which provides the best performance by directly accepting a model matrix and skipping formula processing:

```r
library(fasthurdle)

# Prepare model matrix and response
# Note: Include intercept and all covariates in the design
X <- model.matrix(~ peak_acc + log_total_counts + pct_counts_mito, data = df)
y <- df$gene_expr

# Fit the model (same result extraction as fasthurdle above)
model <- fast_negbin_hurdle(X, y)
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

*Note: Benchmarks run with sample sizes of 1,000, 10,000, and 100,000. Speedup factor is the ratio of pscl execution time to fasthurdle execution time.*

## Features

- Supports the same models as `pscl::hurdle`:
  - Count distributions: Poisson, Negative Binomial, Geometric
  - Zero hurdle distributions: Binomial, Poisson, Negative Binomial, Geometric
- Compatible API with `pscl::hurdle`
- Improved performance through C++ implementations

## Acknowledgements

[The `pscl` package](https://github.com/atahk/pscl), where the original hurdle function was implemented, was developed at the Political Science Computational Laboratory, led by Simon Jackman at Stanford University. The hurdle and count data models in the `pscl` package were re-written by Achim Zeileis and Christian Kleiber.

The use of hurdle models for peak-gene link analysis in single-nucleus multiome data was originally introduced in [Open4Gene](https://github.com/hbliu/Open4Gene) (Liu, H. et al., 2025).

## License

GPL-2

## Contact

Masahiro Kanai (<mkanai@broadinstitute.org>)
