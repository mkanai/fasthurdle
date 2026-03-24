# fasthurdle

A fast implementation of hurdle models using Rcpp. This package provides the same functionality as [the `hurdle` function](https://www.rdocumentation.org/packages/pscl/versions/1.5.9/topics/hurdle) in [the `pscl` package](https://github.com/atahk/pscl), but with improved performance through C++ implementations of key functions. This package is optimized for efficient peak-gene link analysis in large-scale single-nucleus multiome datasets with millions of cells.

## Installation

### Pre-built binaries from R-universe (recommended)

Pre-built binaries are available from [R-universe](https://mkanai.r-universe.dev/fasthurdle), which does not require a C++ compiler or Fortran toolchain:

```r
install.packages("fasthurdle", repos = c("https://mkanai.r-universe.dev", "https://cloud.r-project.org"))
```

### From source

Installing from source requires a C++ compiler and GNU Fortran (`gfortran`, required by the RcppArmadillo dependency):

```r
# install.packages("pak")
pak::pkg_install("mkanai/fasthurdle")
```

### Docker / Singularity

For environments where installing compiler toolchains is difficult, a pre-built Docker image is available:

```bash
docker run --rm -it masakanai/fasthurdle
```

On HPC clusters that support [Singularity](https://docs.sylabs.io/guides/latest/user-guide/) / [Apptainer](https://apptainer.org/), you can convert the Docker image:

```bash
singularity build fasthurdle.sif docker://masakanai/fasthurdle
singularity exec fasthurdle.sif R
```

To build the image locally, a Dockerfile is provided under [`docker/`](docker/):

```bash
docker build -t masakanai/fasthurdle docker/
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

## Score test

By default, `fasthurdle` uses the **Wald test** for inference, same as `pscl::hurdle`. The **score test** is an alternative that evaluates significance at the null model — it does not fit the full count model, making it both faster (~1.5x with cached nulls) and robust to model misspecification. The score test is available for all count distributions (negbin, poisson, geometric).

The count component uses the **observed information** (negative Hessian at the null MLE) instead of the expected Fisher information. This makes the score test robust to distributional misspecification — it matches Wald test calibration even when the NB model is not perfectly specified (e.g., ambient RNA contamination, non-NB count distributions). The zero component uses the expected FIM with SPA, which is already well-calibrated.

For significant tests (|z| > `spa_cutoff`, or |z| > 2 when SPA is disabled), beta is refined via a short BFGS optimization, giving accuracy within ~3% of the full MLE. SPA provides tail p-value correction for sparse genes at small sample sizes. The `summary()` output format is unchanged.

```r
# Wald (default)
model <- fasthurdle(y ~ x | z, data = df, dist = "negbin", zero.dist = "binomial")

# Score test for x — just add score_test
model <- fasthurdle(y ~ x | z, data = df, dist = "negbin", zero.dist = "binomial",
                    score_test = "x")
summary(model)  # same format, better-calibrated p-value for x
```

## Peak-gene link analysis

Hurdle models are well-suited for analyzing peak-gene associations in single-nucleus multiome data (scRNA-seq + scATAC-seq), as originally introduced in [Open4Gene](https://github.com/hbliu/Open4Gene). The two-part hurdle model independently fits: (1) a binomial zero-inflation model testing whether peak accessibility affects the probability of nonzero expression, and (2) a negative binomial count model testing whether peak accessibility affects expression magnitude among expressing cells. This approach explicitly accounts for technical and biological sparsity while modeling the count-based nature of expression measurements with overdispersion.

### Wald test (Open4Gene default)

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

# Fit hurdle model with NB count and binomial zero hurdle
# Use log_total_counts as offset in count model and covariate in zero model
model <- fasthurdle(
  gene_expr ~ peak_acc + pct_counts_mito + offset(log_total_counts) | peak_acc + log_total_counts + pct_counts_mito,
  data = df,
  dist = "negbin",
  zero.dist = "binomial"
)

# Extract results
s <- summary(model)
s$coefficients$count["peak_acc", ]  # Count: beta, SE, z, p-value
s$coefficients$zero["peak_acc", ]   # Zero: beta, SE, z, p-value
```

For high-performance analysis, `fast_negbin_hurdle` accepts model matrices directly and skips formula processing:

```r
X <- model.matrix(~ peak_acc + pct_counts_mito, data = df)
y <- df$gene_expr
offsetx <- df$log_total_counts
Z <- model.matrix(~ peak_acc + log_total_counts + pct_counts_mito, data = df)

model <- fast_negbin_hurdle(X, y, Z = Z, offsetx = offsetx)
```

### Score test (recommended)

The score test with SPA gives better-calibrated p-values and is faster. Just add `score_test`:

```r
model <- fast_negbin_hurdle(X, y, Z = Z, offsetx = offsetx, score_test = "peak_acc")
s <- summary(model)
s$coefficients$count["peak_acc", ]  # Score test: beta, SE, z, p-value
s$coefficients$zero["peak_acc", ]   # Score test: beta, SE, z, p-value
```

For high-throughput testing (many peaks per gene), both null models can be fitted once and reused:

```r
# Fit nulls once per gene (covariates only, no peak_acc)
X_null <- model.matrix(~ pct_counts_mito, data = df)
Z_null <- model.matrix(~ log_total_counts + pct_counts_mito, data = df)
null_fit_count <- fit_null_count(X_null, y, offsetx = offsetx, dist = "negbin")
null_fit_zero <- fit_null_zero(Z_null, y)

# Test each peak against cached nulls
for (peak in peaks) {
  X <- cbind(X_null, df[[peak]])
  colnames(X)[ncol(X)] <- peak
  Z <- cbind(Z_null, df[[peak]])
  colnames(Z)[ncol(Z)] <- peak
  model <- fast_negbin_hurdle(X, y, Z = Z, offsetx = offsetx,
                               score_test = peak,
                               null_fit_count = null_fit_count,
                               null_fit_zero = null_fit_zero)
  # Extract results from summary(model)
}
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

## Changelog

### v1.2.0 (2026-03-23)

- **New feature**: Score test with saddlepoint approximation (SPA) for both count and zero components (`score_test` parameter in `fast_negbin_hurdle()` and `fasthurdle()`).
  - Count component uses **observed information** (analytical negative Hessian) instead of expected FIM, making it robust to model misspecification. Matches Wald calibration under both correct NB and misspecified distributions (e.g., ambient RNA contamination).
  - Zero component uses expected FIM with SPA (closed-form binomial CGF), which is already well-calibrated.
  - Beta for significant tests (|z| > `spa_cutoff`) is refined via 5-iteration BFGS (within ~3% of full MLE).
  - Both null models can be cached via `fit_null_count()` and `fit_null_zero()` for high-throughput testing (~1.5x faster than Wald per peak with cached nulls).

### v1.1.1 (2026-03-09)

- **New feature**: Added statistical utilities for hurdle model p-value combination and FDR control:
  - `CCT()`: Cauchy Combination Test (ACAT) for combining p-values under arbitrary dependency structures.
  - `jiang_doerge_fdr()`: Two-stage FDR procedure for hurdle models, screening on one component and confirming on the other.
  - `acat_stagewise()`: ACAT-based omnibus screening with stage-wise Holm confirmation, classifying regulatory mechanisms as "dual", "switch", "rheostat", or "omnibus_only".

### v1.1.0 (2026-03-05)

- **Bug fix**: Fixed multiple convergence issues in the negative binomial count model:
  - Fixed missing `theta*log(theta)` term in the count model log-likelihood (`CountNegBinFunctor`). The analytical gradient was correct, but the inconsistency with the function value caused BFGS line search failures and incorrect theta/coefficient estimates.
  - Fixed `maxit` and `reltol` from `hurdle.control()` not being forwarded to the C++ optimizer. The roptim library defaulted to `maxit=100` instead of the intended `10000`, causing premature convergence.
  - Fit count model starting values on `y > 0` subset only. The count component models a zero-truncated distribution, but starting values were previously computed from a Poisson GLM on the full dataset including zeros. Fitting on expressing cells provides starting values closer to the truncated MLE, reducing optimizer iterations by 40–75% at high zero fractions and avoiding degenerate local optima at >95% zeros.
- **New feature**: Extended `fast_negbin_hurdle()` to support flexible model specification:
  - Added `Z` parameter for specifying a separate design matrix for the zero component. This enables use cases like scRNA-seq depth correction, where `log(library_size)` is an offset in the count model but a covariate in the zero model.
  - Added `offsetx` and `offsetz` parameters for specifying offsets in the count and zero components.

### v1.0.0 (2025-07-28)

- Initial release.

## License

GPL-2

## Contact

Masahiro Kanai (<mkanai@broadinstitute.org>)
