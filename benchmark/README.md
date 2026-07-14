# Fasthurdle Benchmarking System

Benchmarks for fasthurdle. This directory is excluded from the built package via
`.Rbuildignore`, so nothing here is installed with the package.

## Files

- `benchmark_readme.R` - Produces the tables in the top-level README: the count x zero
  grid against `pscl::hurdle` (swept over sample size and zero fraction) and the
  peak-gene scan comparison. Run it from the repository root:
  `OMP_NUM_THREADS=1 Rscript benchmark/benchmark_readme.R`. Results land in
  `benchmark_results/readme_{grid,scan}_<timestamp>.csv`; the checked-in pair backs the
  numbers currently published in the README.
- `bench_score_test.R` - Score test performance, count and zero components.
- `benchmark_commits.sh` - Runs benchmarks across multiple git commits (see below)
- `Dockerfile.benchmark` - Docker image definition for consistent benchmarking environment
- `run_benchmark.R` - R script that runs the actual benchmarks inside Docker containers
- `compare_results.py` - Compares the JSON output of several commits

Benchmark output (`benchmark_logs/`, `comparison_results/`, and the per-commit JSON in
`benchmark_results/`) is gitignored; only the `readme_*.csv` files are tracked.

## Cross-commit benchmarking

### Basic Usage

Benchmark specific commits:
```bash
benchmark/benchmark_commits.sh abc1234 def5678 master
```

### Using a Commits File

Create a file with commit hashes (one per line):
```bash
echo "abc1234" > commits.txt
echo "def5678" >> commits.txt
echo "master" >> commits.txt

benchmark/benchmark_commits.sh --file commits.txt
```

### Options

- `-f, --file FILE` - Read commits from file (one per line)
- `-p, --parallel NUM` - Number of parallel builds (default: 2)
- `-o, --output-dir DIR` - Output directory for results (default: benchmark/benchmark_results)
- `-n, --num-runs NUM` - Number of runs per benchmark (default: 10)
- `-s, --sample-sizes SIZES` - Comma-separated sample sizes (default: 1000,10000,100000)
- `--dockerfile FILE` - Dockerfile to use (default: benchmark/Dockerfile.benchmark)
- `--force-rebuild` - Force rebuild of Docker images
- `-h, --help` - Display help message

### Examples

Benchmark with custom sample sizes:
```bash
benchmark/benchmark_commits.sh --sample-sizes "1000,5000,10000,50000" master develop
```

Benchmark with more runs for higher precision:
```bash
benchmark/benchmark_commits.sh --num-runs 20 master
```

Run benchmarks with parallel builds:
```bash
benchmark/benchmark_commits.sh --parallel 4 --file commits.txt
```

Each commit is built into its own Docker image and benchmarked against `pscl` over the
count x zero grid at several sample sizes, so results are comparable across commits and
machines. Requires Docker, git, and bash; GNU parallel is optional and enables
`--parallel`.

## Output

- `benchmark/benchmark_results/benchmark_<commit>_<timestamp>.json` - per-commit timings
- `benchmark/benchmark_results/benchmark_summary_<timestamp>.txt` - summary across commits
- `benchmark/benchmark_logs/` - build and run logs

Each JSON holds a `metadata` block (commit, timestamp, R and package versions) and a
`benchmarks` array with one entry per (sample size, count dist, zero dist), each carrying
`median/mean/min/max_time_ns` for both `pscl` and `fasthurdle` plus the resulting
`speedup`.

To compare several commits and surface regressions:

```bash
python3 benchmark/compare_results.py benchmark/benchmark_results/*.json
```