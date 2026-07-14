#!/usr/bin/env Rscript

# Benchmarking script for fasthurdle
# This script is designed to be run inside Docker containers

library(fasthurdle)
library(pscl)
library(microbenchmark)
library(jsonlite)

# Get environment variables
git_commit <- Sys.getenv("GIT_COMMIT", "unknown")
num_runs <- as.integer(Sys.getenv("NUM_RUNS", "10"))
sample_sizes_str <- Sys.getenv("SAMPLE_SIZES", "1000,10000,100000")
results_dir <- Sys.getenv("RESULTS_DIR", "/results")

# Parse sample sizes
sample_sizes <- as.numeric(unlist(strsplit(sample_sizes_str, ",")))

# Function to run benchmarks for different sample sizes and model combinations
run_benchmarks <- function(sample_sizes, times = num_runs) {
  results <- list()

  # Define all combinations of count and zero hurdle distributions
  count_dists <- c("poisson", "negbin", "geometric")
  zero_dists <- c("binomial", "poisson", "negbin", "geometric")

  # Create all combinations
  combinations <- expand.grid(
    count_dist = count_dists,
    zero_dist = zero_dists,
    stringsAsFactors = FALSE
  )

  for (n in sample_sizes) {
    cat("Running benchmark for sample size:", n, "\n")

    # Generate sample data
    set.seed(123)
    x <- rnorm(n)
    z <- rnorm(n)
    lambda <- exp(1 + 0.5 * x)
    p <- plogis(0.5 - 0.5 * z)
    y <- rbinom(n, size = 1, prob = p) * rpois(n, lambda = lambda)

    # Create a data frame
    df <- data.frame(y = y, x = x, z = z)

    # Initialize results for this sample size
    results[[as.character(n)]] <- list()

    # Run benchmarks for each combination
    for (i in 1:nrow(combinations)) {
      count_dist <- combinations$count_dist[i]
      zero_dist <- combinations$zero_dist[i]

      # Create a unique key for this combination
      combo_key <- paste(count_dist, zero_dist, sep = "_")

      cat("  Benchmarking:", count_dist, "count model with", zero_dist, "zero hurdle\n")

      # Run the benchmark
      benchmark_result <- tryCatch(
        {
          # Untimed warm-up. Without this the first expression evaluated absorbs
          # one-off costs (dynamic loading, JIT) and its timing is meaningless --
          # it can report a speedup below 1x at small n.
          invisible(fasthurdle(y ~ x | z,
            data = df, dist = count_dist, zero.dist = zero_dist
          ))
          invisible(pscl::hurdle(y ~ x | z,
            data = df, dist = count_dist, zero.dist = zero_dist
          ))

          microbenchmark(
            fasthurdle = fasthurdle(y ~ x | z,
              data = df,
              dist = count_dist,
              zero.dist = zero_dist
            ),
            pscl = pscl::hurdle(y ~ x | z,
              data = df,
              dist = count_dist,
              zero.dist = zero_dist
            ),
            times = times
          )
        },
        error = function(e) {
          cat("    Error:", e$message, "\n")
          NULL
        }
      )

      # Store the result if successful
      if (!is.null(benchmark_result)) {
        results[[as.character(n)]][[combo_key]] <- benchmark_result
      }
    }
  }

  return(results)
}

# Function to convert results to JSON format
results_to_json <- function(results, git_commit) {
  json_data <- list(
    metadata = list(
      git_commit = git_commit,
      timestamp = as.character(Sys.time()),
      R_version = R.version.string,
      fasthurdle_version = as.character(packageVersion("fasthurdle")),
      pscl_version = as.character(packageVersion("pscl"))
    ),
    benchmarks = list()
  )

  for (n in names(results)) {
    for (combo_key in names(results[[n]])) {
      # Extract count and zero distributions from the key
      parts <- strsplit(combo_key, "_")[[1]]
      count_dist <- parts[1]
      zero_dist <- parts[2]

      # Get benchmark data
      benchmark <- results[[n]][[combo_key]]
      summary_data <- summary(benchmark)

      # Calculate speedup
      pscl_median <- summary_data$median[summary_data$expr == "pscl"]
      fasthurdle_median <- summary_data$median[summary_data$expr == "fasthurdle"]
      speedup <- pscl_median / fasthurdle_median

      # Add to JSON data
      json_data$benchmarks[[length(json_data$benchmarks) + 1]] <- list(
        sample_size = as.numeric(n),
        count_dist = count_dist,
        zero_dist = zero_dist,
        pscl = list(
          median_time_ns = pscl_median,
          mean_time_ns = summary_data$mean[summary_data$expr == "pscl"],
          min_time_ns = summary_data$min[summary_data$expr == "pscl"],
          max_time_ns = summary_data$max[summary_data$expr == "pscl"]
        ),
        fasthurdle = list(
          median_time_ns = fasthurdle_median,
          mean_time_ns = summary_data$mean[summary_data$expr == "fasthurdle"],
          min_time_ns = summary_data$min[summary_data$expr == "fasthurdle"],
          max_time_ns = summary_data$max[summary_data$expr == "fasthurdle"]
        ),
        speedup = speedup
      )
    }
  }

  return(json_data)
}

# Main execution
cat("Starting benchmark for commit:", git_commit, "\n")
cat("Sample sizes:", paste(sample_sizes, collapse = ", "), "\n")
cat("Number of runs per benchmark:", num_runs, "\n")
cat("\n")

# Run benchmarks
results <- run_benchmarks(sample_sizes)

# Convert to JSON
json_data <- results_to_json(results, git_commit)

# Save results
output_file <- file.path(results_dir, sprintf(
  "benchmark_%s_%s.json",
  substr(git_commit, 1, 8),
  format(Sys.time(), "%Y%m%d_%H%M%S")
))

writeLines(toJSON(json_data, pretty = TRUE, auto_unbox = TRUE), output_file)

cat("\nBenchmark completed. Results saved to:", basename(output_file), "\n")

# Print summary
cat("\nSummary of speedup factors:\n")
cat("==========================================================\n")

for (i in seq_along(json_data$benchmarks)) {
  b <- json_data$benchmarks[[i]]
  cat(sprintf(
    "Sample size: %d, %s/%s: %.1fx speedup (vs pscl)\n",
    b$sample_size, b$count_dist, b$zero_dist, b$speedup
  ))
}
