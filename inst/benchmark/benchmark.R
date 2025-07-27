library(fasthurdle)
library(pscl)
library(microbenchmark)
library(ggplot2)

# Function to run benchmarks for different sample sizes and model combinations
run_benchmarks <- function(sample_sizes = 10**seq(3, 5), times = 10) {
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

# Run benchmarks
results <- run_benchmarks()

# Function to summarize and plot results
summarize_results <- function(results) {
  # Prepare data for plotting
  plot_data <- data.frame()

  # Define all combinations of count and zero hurdle distributions
  count_dists <- c("poisson", "negbin", "geometric")
  zero_dists <- c("binomial", "poisson", "negbin", "geometric")

  # Process results
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
      pscl_time <- summary_data$median[summary_data$expr == "pscl"]
      fasthurdle_time <- summary_data$median[summary_data$expr == "fasthurdle"]
      speedup <- pscl_time / fasthurdle_time

      # Add to plot data
      plot_data <- rbind(plot_data, data.frame(
        sample_size = as.numeric(n),
        count_dist = count_dist,
        zero_dist = zero_dist,
        combo = combo_key,
        pscl_time = pscl_time / 1e9, # Convert to seconds
        fasthurdle_time = fasthurdle_time / 1e9, # Convert to seconds
        speedup = speedup
      ))
    }
  }

  # Print summary
  cat("\nPerformance Summary:\n")
  cat("-------------------\n")

  # Group by count distribution and zero distribution
  for (count_dist in count_dists) {
    for (zero_dist in zero_dists) {
      combo_key <- paste(count_dist, zero_dist, sep = "_")
      combo_data <- plot_data[plot_data$combo == combo_key, ]

      if (nrow(combo_data) > 0) {
        cat("\nCount Distribution:", count_dist, "| Zero Distribution:", zero_dist, "\n")
        print(combo_data[, c("sample_size", "pscl_time", "fasthurdle_time", "speedup")])
      }
    }
  }

  # Plot time comparison by count distribution
  p1 <- ggplot(plot_data, aes(x = factor(sample_size), y = pscl_time, fill = "pscl")) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
    geom_bar(aes(y = fasthurdle_time, fill = "fasthurdle"), stat = "identity", position = "dodge", alpha = 0.7) +
    facet_grid(zero_dist ~ count_dist, scales = "free_y") +
    labs(
      title = "Execution Time Comparison",
      x = "Sample Size",
      y = "Time (seconds)",
      fill = "Package"
    ) +
    theme_minimal() +
    scale_fill_brewer(palette = "Set1")

  # Plot speedup
  p2 <- ggplot(plot_data, aes(x = factor(sample_size), y = speedup, fill = zero_dist)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
    facet_wrap(~count_dist, scales = "free_y") +
    labs(
      title = "Speedup Factor (pscl time / fasthurdle time)",
      x = "Sample Size",
      y = "Speedup Factor",
      fill = "Zero Distribution"
    ) +
    theme_minimal() +
    scale_fill_brewer(palette = "Set1")

  return(list(summary = plot_data, plots = list(time = p1, speedup = p2)))
}

# Summarize and plot results
summary_results <- summarize_results(results)

# Print plots
print(summary_results$plots$time)
print(summary_results$plots$speedup)

# Function to create markdown table from benchmark results
create_markdown_table <- function(plot_data) {
  # Create a summary table with average speedup by distribution combination
  summary_table <- aggregate(
    speedup ~ count_dist + zero_dist,
    data = plot_data,
    FUN = function(x) round(mean(x), 1)
  )

  # Sort by count_dist and zero_dist for consistent presentation
  summary_table <- summary_table[order(summary_table$count_dist, summary_table$zero_dist), ]

  # Create markdown table header
  md_table <- "## Benchmark Results\n\n"
  md_table <- paste0(md_table, "Average speedup of fasthurdle compared to pscl:\n\n")
  md_table <- paste0(md_table, "| Count Model | Zero Hurdle | Speedup Factor |\n")
  md_table <- paste0(md_table, "|------------|------------|---------------|\n")

  # Add rows to the table
  for (i in 1:nrow(summary_table)) {
    row <- summary_table[i, ]
    md_table <- paste0(
      md_table,
      "| ", row$count_dist, " | ", row$zero_dist, " | ", row$speedup, "x |\n"
    )
  }

  # Add note about benchmark conditions
  md_table <- paste0(
    md_table,
    "\n*Note: Benchmarks run with sample sizes ",
    paste(unique(plot_data$sample_size), collapse = ", "),
    ". Speedup factor is the ratio of pscl execution time to fasthurdle execution time.*\n"
  )

  return(md_table)
}

# Save plots
ggsave("man/figures/benchmark_time.png", summary_results$plots$time, width = 10, height = 6, bg = "white")
ggsave("man/figures/benchmark_speedup.png", summary_results$plots$speedup, width = 10, height = 6, bg = "white")

# Create and save markdown table
md_table <- create_markdown_table(summary_results$summary)
writeLines(md_table, "benchmark_results.md")

cat("\nBenchmark results saved to benchmark_time.png, benchmark_speedup.png, and benchmark_results.md\n")
