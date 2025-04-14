library(fasthurdle)
library(microbenchmark)
library(ggplot2)

# Function to run benchmarks for different sample sizes
run_benchmarks <- function(sample_sizes = c(500, 1000, 5000, 10000, 100000), times = 10) {
  results <- list()

  for (n in sample_sizes) {
    cat("Running benchmark for sample size:", n, "\n")

    # Generate sample data
    set.seed(123)
    x <- rnorm(n)
    z <- rnorm(n)
    lambda <- exp(1 + 0.5 * x)
    p <- plogis(0.5 - 0.5 * z)
    y <- rbinom(n, size = 1, prob = p) * rpois(n, lambda = lambda)

    # Create a data frame and model matrix
    df <- data.frame(y = y, x = x, z = z)
    X <- model.matrix(~ x + z, data = df)

    # Run the benchmark
    benchmark_result <- microbenchmark(
      fasthurdle = fasthurdle(y ~ x + z,
        data = df,
        dist = "negbin",
        zero.dist = "binomial",
        link = "logit"
      ),
      fast_negbin_hurdle = fast_negbin_hurdle(X, y),
      times = times
    )

    # Store the result
    results[[as.character(n)]] <- benchmark_result
  }

  return(results)
}

# Run benchmarks
results <- run_benchmarks()

# Function to summarize and plot results
summarize_results <- function(results) {
  # Prepare data for plotting
  plot_data <- data.frame()

  # Process results
  for (n in names(results)) {
    # Get benchmark data
    benchmark <- results[[n]]
    summary_data <- summary(benchmark)

    # Calculate speedup
    fasthurdle_time <- summary_data$median[summary_data$expr == "fasthurdle"]
    fast_negbin_time <- summary_data$median[summary_data$expr == "fast_negbin_hurdle"]
    speedup <- fasthurdle_time / fast_negbin_time

    # Add to plot data
    plot_data <- rbind(plot_data, data.frame(
      sample_size = as.numeric(n),
      fasthurdle_time = fasthurdle_time / 1e9, # Convert to seconds
      fast_negbin_time = fast_negbin_time / 1e9, # Convert to seconds
      speedup = speedup
    ))
  }

  # Print summary
  cat("\nPerformance Summary:\n")
  cat("-------------------\n")
  print(plot_data[, c("sample_size", "fasthurdle_time", "fast_negbin_time", "speedup")])

  # Plot time comparison
  p1 <- ggplot(plot_data, aes(x = factor(sample_size))) +
    geom_bar(aes(y = fasthurdle_time, fill = "fasthurdle"), stat = "identity", position = "dodge", alpha = 0.7) +
    geom_bar(aes(y = fast_negbin_time, fill = "fast_negbin_hurdle"), stat = "identity", position = "dodge", alpha = 0.7) +
    labs(
      title = "Execution Time Comparison",
      x = "Sample Size",
      y = "Time (seconds)",
      fill = "Function"
    ) +
    theme_minimal()

  # Plot speedup
  p2 <- ggplot(plot_data, aes(x = factor(sample_size), y = speedup)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
    labs(
      title = "Speedup Factor (fasthurdle time / fast_negbin_hurdle time)",
      x = "Sample Size",
      y = "Speedup Factor"
    ) +
    theme_minimal()

  return(list(summary = plot_data, plots = list(time = p1, speedup = p2)))
}

# Summarize and plot results
summary_results <- summarize_results(results)

# Print plots
print(summary_results$plots$time)
print(summary_results$plots$speedup)

# Save plots
ggsave("man/figures/benchmark_fast_negbin_time.png", summary_results$plots$time, width = 10, height = 6)
ggsave("man/figures/benchmark_fast_negbin_speedup.png", summary_results$plots$speedup, width = 10, height = 6)

cat("\nBenchmark results saved to benchmark_fast_negbin_time.png and benchmark_fast_negbin_speedup.png\n")
