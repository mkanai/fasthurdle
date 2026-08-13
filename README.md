# fasthurdle

A fast implementation of hurdle models using Rcpp. This package provides the same functionality as [the `hurdle` function](https://www.rdocumentation.org/packages/pscl/versions/1.5.9/topics/hurdle) in [the `pscl` package](https://github.com/atahk/pscl), but with improved performance through C++ implementations of key functions. This package is optimized for efficient peak-gene link analysis in large-scale single-nucleus multiome datasets with millions of cells.

## Installation

Requires R 4.4 or newer.

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

The **score test** evaluates significance at the null model: it does not fit the full count model, making it both faster and robust to model misspecification. The score test is available for all count distributions (negbin, poisson, geometric).

The count component uses the **observed information** (analytical negative Hessian) instead of the expected Fisher information. This makes the score test robust to distributional misspecification: it matches Wald test calibration even when the NB model is not perfectly specified (e.g., ambient RNA contamination, non-NB count distributions). The zero component uses the expected FIM, which is identical to the observed information for the binomial/logit model (a property of canonical GLMs).

For significant tests (|z| > 2), beta is refined via a short BFGS optimization, giving accuracy within ~3% of the full MLE. The `summary()` output format is unchanged.

**SPA** (saddlepoint approximation) is available via `spa_cutoff = 2` for improved tail p-value accuracy, primarily useful for sparse genes at small sample sizes (n < 50K).

```r
# Score test for x: just add score_test
model <- fasthurdle(y ~ x | z, data = df, dist = "negbin", zero.dist = "binomial",
                    score_test = "x")
summary(model)  # same format, score-test p-value for x

# With SPA for small-n studies
model <- fasthurdle(y ~ x | z, data = df, dist = "negbin", zero.dist = "binomial",
                    score_test = "x", spa_cutoff = 2)
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

The score test gives better-calibrated p-values and is faster. Just add `score_test`:

```r
model <- fast_negbin_hurdle(X, y, Z = Z, offsetx = offsetx, score_test = "peak_acc")
s <- summary(model)
s$coefficients$count["peak_acc", ]  # Score test: beta, SE, z, p-value
s$coefficients$zero["peak_acc", ]   # Score test: beta, SE, z, p-value
```

### Batch scanning with `hurdle_scan()`

For testing many peaks against the same gene, `hurdle_scan()` fits the null models once and scans all peaks efficiently. The design matrix includes all peaks as columns, but each peak is tested marginally (one at a time) against a null model containing only the covariates:

```r
# Build a matrix with covariates + all peaks to test
X <- model.matrix(~ pct_counts_mito + peak1 + peak2 + peak3, data = df)
Z <- model.matrix(~ log_total_counts + pct_counts_mito + peak1 + peak2 + peak3, data = df)

results <- hurdle_scan(X, y, peaks = c("peak1", "peak2", "peak3"),
                       Z = Z, offsetx = offsetx)
# Returns data.frame: peak, nlog10p_count, beta_count, se_count, stat_count,
#                      nlog10p_zero, beta_zero, se_zero, stat_zero
```

## Benchmark Results

### Model fitting vs `pscl::hurdle`

Speedup of `fasthurdle()` over `pscl::hurdle()` on the same data, by sample size:

| Count Model | Zero Hurdle | n=1,000 | n=10,000 | n=100,000 |
|------------|------------|---------|----------|-----------|
| geometric  | binomial   | 2.2x    | 3.1x     | 3.4x      |
| geometric  | geometric  | 2.4x    | 3.4x     | 3.6x      |
| geometric  | negbin     | 5.6x    | 6.5x     | 5.3x      |
| geometric  | poisson    | 3.1x    | 4.7x     | 4.3x      |
| negbin     | binomial   | 3.2x    | 4.9x     | 4.3x      |
| negbin     | geometric  | 3.2x    | 4.9x     | 4.5x      |
| negbin     | negbin     | 6.4x    | 6.6x     | 5.8x      |
| negbin     | poisson    | 3.9x    | 5.0x     | 5.1x      |
| poisson    | binomial   | 2.5x    | 4.0x     | 4.2x      |
| poisson    | geometric  | 2.7x    | 3.9x     | 4.3x      |
| poisson    | negbin     | 6.0x    | 6.6x     | 5.3x      |
| poisson    | poisson    | 3.4x    | 4.8x     | 4.9x      |

Speedup is mildly higher on sparser data: across the grid at n=100,000 the median is 4.0x at 40% zeros, 4.3x at 80%, and 4.8x at 95%.

### Peak-gene scan

Scanning all candidate peaks against one gene, with an NB count model and a binomial zero hurdle at 85% zeros. This is the workload `hurdle_scan()` exists for: the null models are fit once per gene and every peak is scored against them, rather than refitting the full model for each peak (which is the only option in `pscl`).

Total time per gene:

| Method | 5,000 cells, 200 peaks | 500,000 cells, 500 peaks |
|--------|-----------------------|--------------------------|
| `pscl::hurdle()`, refit per peak | 18s * | 4.3h * |
| `fast_negbin_hurdle()`, refit per peak (Wald) | 3.2s * | 46min * |
| `fast_negbin_hurdle()`, refit per peak (score test) | 2.8s * | 35min * |
| `hurdle_scan()`, shared null + score test | **0.10s** | **85s** |
| | **175x** | **180x** |

The larger configuration (~20 covariates) is representative of a cis-window scan on a single-cell multiome dataset. Note that the advantage of amortizing the null fit narrows as the individual fits get more expensive, so quoting the small-scale ratio alone would overstate it.

*\* Per-peak refit totals are extrapolated: the per-peak cost is timed on a handful of peaks and multiplied by the peak count. These are independent refits over identical data, so the loop is linear (verified against a full 200-peak `pscl` run: 45.2s extrapolated vs 45.4s measured). Running the 500,000-cell `pscl` baseline in full would take over four hours. `hurdle_scan()` is timed in full in both columns.*

*Benchmarks run single-threaded (`OMP_NUM_THREADS=1` set in the environment before R starts) so the speedup reflects the implementation rather than the core count; `fasthurdle` uses OpenMP and will go faster with more threads, while `pscl` is single-threaded. Note that R is often linked against a threaded BLAS, which on matrices this small is markedly slower than a single thread -- the benchmark scripts refuse to run unless `OMP_NUM_THREADS=1` is set. Data are simulated from a true hurdle process (zero-truncated NB counts). Across all 108 grid configurations, `fasthurdle` and `pscl` coefficient estimates agree to within 2e-5, so the two are solving the same problem to the same optimum. Reproduce with `benchmark/benchmark_readme.R` (grid) and `benchmark/benchmark_scan_scale.R` (scan).*

## Features

- Supports the same models as `pscl::hurdle`:
  - Count distributions: Poisson, Negative Binomial, Geometric
  - Zero hurdle distributions: Binomial, Poisson, Negative Binomial, Geometric
- Compatible API with `pscl::hurdle`
- Score test with observed information for robust inference
- `hurdle_scan()` for high-throughput peak-gene link analysis
- Joint 2-df chi-squared test and ACAT stage-wise mode classification
- Saddlepoint approximation (SPA) for accurate tail p-values
- C++ backend via Rcpp/RcppArmadillo

## Acknowledgements

[The `pscl` package](https://github.com/atahk/pscl), where the original hurdle function was implemented, was developed at the Political Science Computational Laboratory, led by Simon Jackman at Stanford University. The hurdle and count data models in the `pscl` package were re-written by Achim Zeileis and Christian Kleiber.

The use of hurdle models for peak-gene link analysis in single-nucleus multiome data was originally introduced in [Open4Gene](https://github.com/hbliu/Open4Gene) (Liu, H. et al., 2025).

## Changelog

### v1.2.0 (2026-04-06)

- **New feature**: Score test with observed information for count and zero components.
- **New feature**: `hurdle_scan()` for batch score-testing many peaks against the same gene.
- **New feature**: Joint 2-df chi-squared score test (`joint_score_test()`) for omnibus peak-gene testing.

### v1.1.1 (2026-03-09)

- **New feature**: Added statistical utilities for hurdle model p-value combination and FDR control:
  - `CCT()`: Cauchy Combination Test (ACAT) for combining p-values under arbitrary dependency structures.
  - `jiang_doerge_fdr()`: Two-stage FDR procedure for hurdle models, screening on one component and confirming on the other.
  - `acat_stagewise()`: ACAT-based omnibus screening with stage-wise Holm confirmation, classifying regulatory modes as "dual", "switch", "rheostat", or "omnibus_only".

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

## Citation

Kanai, M. et al. [Population-scale multiome immune cell atlas reveals complex disease drivers](https://doi.org/10.1101/2025.11.25.25340489). medRxiv (2025)

## Contact

Masahiro Kanai (<mkanai@broadinstitute.org>)
