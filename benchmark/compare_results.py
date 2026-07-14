#!/usr/bin/env python3
"""
Compare benchmark results from multiple fasthurdle commits.
Generates comparison tables, plots, and identifies performance regressions.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import sys

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.", file=sys.stderr)


class BenchmarkComparator:
    def __init__(self, result_files, output_dir="comparison_results", baseline_commit=None):
        self.result_files = result_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.comparison_data = []
        self.baseline_commit = baseline_commit
        
    def load_results(self):
        """Load all benchmark result files."""
        for file_path in self.result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    commit = data['metadata']['git_commit'][:8]  # Use short hash
                    self.results[commit] = data
                    print(f"Loaded results for commit {commit}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}", file=sys.stderr)
    
    def extract_metrics(self):
        """Extract key metrics for comparison."""
        for commit, data in self.results.items():
            metadata = data['metadata']
            
            for benchmark in data['benchmarks']:
                row = {
                    'commit': commit,
                    'timestamp': metadata['timestamp'],
                    'R_version': metadata['R_version'],
                    'fasthurdle_version': metadata['fasthurdle_version'],
                    'pscl_version': metadata['pscl_version'],
                    'sample_size': benchmark['sample_size'],
                    'count_dist': benchmark['count_dist'],
                    'zero_dist': benchmark['zero_dist'],
                    'config': f"{benchmark['count_dist']}_{benchmark['zero_dist']}_{benchmark['sample_size']}",
                    'pscl_median_time': benchmark['pscl']['median_time_ns'] / 1e9,  # Convert to seconds
                    'pscl_mean_time': benchmark['pscl']['mean_time_ns'] / 1e9,
                    'fasthurdle_median_time': benchmark['fasthurdle']['median_time_ns'] / 1e9,
                    'fasthurdle_mean_time': benchmark['fasthurdle']['mean_time_ns'] / 1e9,
                    'speedup': benchmark['speedup']
                }
                
                self.comparison_data.append(row)
    
    def create_comparison_df(self):
        """Create pandas DataFrame for analysis."""
        self.df = pd.DataFrame(self.comparison_data)
        
        # Sort by timestamp to get chronological order
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['config', 'timestamp'])
        
        # Determine baseline commit
        if self.baseline_commit:
            baseline_to_use = self.baseline_commit[:8]
        elif 'master' in self.df['commit'].values:
            baseline_to_use = 'master'
        else:
            baseline_to_use = None
        
        # Calculate percentage changes for speedup - per model type and sample size
        for (count_dist, zero_dist, sample_size), group_data in self.df.groupby(['count_dist', 'zero_dist', 'sample_size']):
            group_mask = (self.df['count_dist'] == count_dist) & \
                        (self.df['zero_dist'] == zero_dist) & \
                        (self.df['sample_size'] == sample_size)
            
            if len(group_data) > 0:
                # Find baseline row
                if baseline_to_use and baseline_to_use in group_data['commit'].values:
                    baseline = group_data[group_data['commit'] == baseline_to_use].iloc[0]
                else:
                    # Fallback to first commit chronologically
                    baseline = group_data.sort_values('timestamp').iloc[0]
                
                baseline_speedup = baseline['speedup']
                if baseline_speedup > 0:
                    self.df.loc[group_mask, 'speedup_pct_change'] = \
                        ((self.df.loc[group_mask, 'speedup'] - baseline_speedup) / baseline_speedup) * 100
    
    def generate_summary_table(self):
        """Generate summary comparison table."""
        summary_file = self.output_dir / "benchmark_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# Fasthurdle Benchmark Comparison Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            f.write("## Overall Summary\n\n")
            f.write(f"- Total commits compared: {len(self.results)}\n")
            f.write(f"- Total benchmark configurations: {self.df['config'].nunique()}\n")
            f.write(f"- Total data points: {len(self.df)}\n\n")
            
            # Commit information
            f.write("## Commits\n\n")
            for commit in sorted(self.df['commit'].unique()):
                commit_data = self.df[self.df['commit'] == commit].iloc[0]
                f.write(f"- **{commit}**: {commit_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Performance comparison organized by sample size
            f.write("## Performance Comparison by Sample Size\n\n")
            
            # Get unique sample sizes and sort them
            sample_sizes = sorted(self.df['sample_size'].unique())
            
            for sample_size in sample_sizes:
                f.write(f"### Sample Size: {sample_size:,}\n\n")
                
                # Filter data for this sample size
                size_data = self.df[self.df['sample_size'] == sample_size]
                
                # Sort by model type and commit
                size_data = size_data.sort_values(['count_dist', 'zero_dist', 'timestamp'])
                
                # Group by model type
                f.write("| Model Type | Commit | Fasthurdle | pscl | Speedup | Δ Speedup |\n")
                f.write("|------------|--------|------------|------|---------|----------|\n")
                
                for (count_dist, zero_dist), model_data in size_data.groupby(['count_dist', 'zero_dist']):
                    model_type = f"{count_dist}/{zero_dist}"
                    
                    for _, row in model_data.iterrows():
                        commit = row['commit']
                        fasthurdle_time = row['fasthurdle_median_time']
                        pscl_time = row['pscl_median_time']
                        speedup = row['speedup']
                        
                        # Calculate speedup change
                        if 'speedup_pct_change' in row and not pd.isna(row['speedup_pct_change']):
                            speedup_change = f"{row['speedup_pct_change']:+.1f}%"
                            if row['speedup_pct_change'] < -10:
                                speedup_change = f"**{speedup_change}** ⚠️"
                        else:
                            speedup_change = "baseline"
                        
                        # Format times with appropriate precision and units
                        def format_time(time_seconds):
                            if time_seconds < 1e-6:  # Less than 1 microsecond
                                return f"{time_seconds*1e9:.1f}ns"
                            elif time_seconds < 1e-3:  # Less than 1 millisecond
                                return f"{time_seconds*1e6:.1f}μs"
                            elif time_seconds < 1:  # Less than 1 second
                                return f"{time_seconds*1e3:.1f}ms"
                            else:
                                return f"{time_seconds:.3f}s"
                        
                        fh_time_str = format_time(fasthurdle_time)
                        pscl_time_str = format_time(pscl_time)
                        
                        f.write(f"| {model_type} | {commit} | {fh_time_str} | "
                               f"{pscl_time_str} | {speedup:.1f}x | {speedup_change} |\n")
                
                f.write("\n")
            
            # Average speedup summary
            f.write("## Average Speedup Summary\n\n")
            f.write("| Model Type | Avg Speedup | Min Speedup | Max Speedup |\n")
            f.write("|------------|-------------|-------------|-------------|\n")
            
            # Re-group by distribution types for summary
            dist_groups = self.df.groupby(['count_dist', 'zero_dist'])
            
            for (count_dist, zero_dist), group_data in dist_groups:
                avg_speedup = group_data['speedup'].mean()
                min_speedup = group_data['speedup'].min()
                max_speedup = group_data['speedup'].max()
                f.write(f"| {count_dist}/{zero_dist} | {avg_speedup:.1f}x | "
                       f"{min_speedup:.1f}x | {max_speedup:.1f}x |\n")
            
            f.write("\n")
            
            # Performance regressions
            f.write("## Performance Analysis\n\n")
            
            # Check for speedup regressions - compare by model type AND sample size
            regressions = []
            
            # Group by model type and sample size
            for (count_dist, zero_dist, sample_size), group_data in self.df.groupby(['count_dist', 'zero_dist', 'sample_size']):
                if len(group_data) > 1:
                    # Sort by timestamp to ensure chronological order
                    group_data = group_data.sort_values('timestamp')
                    
                    # Find baseline
                    if self.baseline_commit and self.baseline_commit[:8] in group_data['commit'].values:
                        baseline = group_data[group_data['commit'] == self.baseline_commit[:8]].iloc[0]
                    else:
                        # Use earliest commit as baseline
                        baseline = group_data.iloc[0]
                    
                    # Check all other commits against baseline
                    for _, row in group_data.iterrows():
                        if row['commit'] != baseline['commit']:
                            speedup_decrease = ((baseline['speedup'] - row['speedup']) / baseline['speedup']) * 100
                            
                            if speedup_decrease > 10:  # More than 10% decrease in speedup
                                regressions.append({
                                    'count_dist': count_dist,
                                    'zero_dist': zero_dist,
                                    'sample_size': sample_size,
                                    'baseline_commit': baseline['commit'],
                                    'tested_commit': row['commit'],
                                    'speedup_decrease': speedup_decrease,
                                    'baseline_speedup': baseline['speedup'],
                                    'tested_speedup': row['speedup']
                                })
            
            if regressions:
                f.write("⚠️ **Warning: Performance regressions detected!**\n\n")
                for reg in sorted(regressions, key=lambda x: x['speedup_decrease'], reverse=True):
                    model_desc = f"{reg['count_dist']}/{reg['zero_dist']} (n={reg['sample_size']:,})"
                    f.write(f"- **{model_desc}**: {reg['speedup_decrease']:.1f}% decrease in speedup "
                           f"({reg['baseline_speedup']:.1f}x → {reg['tested_speedup']:.1f}x) "
                           f"from {reg['baseline_commit']} to {reg['tested_commit']}\n")
            else:
                f.write("✅ No significant performance regressions detected.\n")
            
            f.write("\n")
        
        print(f"Summary saved to: {summary_file}")
        return summary_file
    
    def export_csv(self):
        """Export detailed results to CSV."""
        csv_file = self.output_dir / "benchmark_details.csv"
        self.df.to_csv(csv_file, index=False)
        print(f"Detailed results saved to: {csv_file}")
        return csv_file
    
    def export_json(self):
        """Export comparison data as JSON."""
        json_file = self.output_dir / "benchmark_comparison.json"
        
        comparison_json = {
            'generated': datetime.now().isoformat(),
            'commits': list(self.results.keys()),
            'model_types': list(self.df[['count_dist', 'zero_dist']].drop_duplicates().values.tolist()),
            'summary': {},
            'detailed_results': self.comparison_data
        }
        
        # Add summary statistics by model type
        for (count_dist, zero_dist), group_data in self.df.groupby(['count_dist', 'zero_dist']):
            key = f"{count_dist}_{zero_dist}"
            comparison_json['summary'][key] = {
                'avg_speedup': float(group_data['speedup'].mean()),
                'min_speedup': float(group_data['speedup'].min()),
                'max_speedup': float(group_data['speedup'].max()),
                'best_commit': group_data.loc[group_data['speedup'].idxmax(), 'commit'],
                'sample_sizes': sorted(group_data['sample_size'].unique().tolist())
            }
        
        with open(json_file, 'w') as f:
            json.dump(comparison_json, f, indent=2)
        
        print(f"JSON comparison saved to: {json_file}")
        return json_file
    
    def create_plots(self):
        """Create visualization plots."""
        if not PLOT_AVAILABLE:
            print("Skipping plots - matplotlib not available")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Speedup comparison across commits
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        dist_types = self.df[['count_dist', 'zero_dist']].drop_duplicates().values[:4]
        
        for idx, (count_dist, zero_dist) in enumerate(dist_types):
            ax = axes[idx]
            
            dist_data = self.df[(self.df['count_dist'] == count_dist) & 
                               (self.df['zero_dist'] == zero_dist)]
            
            for sample_size in sorted(dist_data['sample_size'].unique()):
                size_data = dist_data[dist_data['sample_size'] == sample_size].sort_values('timestamp')
                if len(size_data) > 1:
                    ax.plot(size_data['commit'], size_data['speedup'], 
                           marker='o', label=f'n={sample_size:,}', linewidth=2, markersize=8)
            
            ax.set_xlabel('Commit', fontsize=10)
            ax.set_ylabel('Speedup Factor', fontsize=10)
            ax.set_title(f'{count_dist} / {zero_dist}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Fasthurdle Speedup vs pscl Across Commits', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_file = self.output_dir / "speedup_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Speedup plot saved to: {plot_file}")
        
        # 2. Heatmap of speedup values
        if len(self.results) > 1:
            # Create pivot table for heatmap
            pivot_data = self.df.pivot_table(
                values='speedup',
                index=['count_dist', 'zero_dist'],
                columns='commit',
                aggfunc='mean'
            )
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(pivot_data, 
                       annot=True, 
                       fmt='.1f',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Speedup Factor'},
                       annot_kws={'size': 10})
            
            ax.set_title('Average Speedup Heatmap (fasthurdle vs pscl)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Commit', fontsize=12)
            ax.set_ylabel('Model Type', fontsize=12)
            plt.tight_layout()
            
            plot_file = self.output_dir / "speedup_heatmap.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Heatmap saved to: {plot_file}")
        
        # 3. Box plot of speedup by model type
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create model type labels
        self.df['model_type'] = self.df['count_dist'] + '/' + self.df['zero_dist']
        
        # Sort by median speedup
        model_order = self.df.groupby('model_type')['speedup'].median().sort_values(ascending=False).index
        
        self.df.boxplot(column='speedup', by='model_type', ax=ax, 
                       positions=range(len(model_order)),
                       widths=0.6)
        
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        ax.set_xlabel('Model Type', fontsize=12)
        ax.set_ylabel('Speedup Factor', fontsize=12)
        ax.set_title('Distribution of Speedup Factors by Model Type', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove automatic title
        plt.tight_layout()
        
        plot_file = self.output_dir / "speedup_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Distribution plot saved to: {plot_file}")
    
    def run_comparison(self):
        """Run the full comparison analysis."""
        print("Loading benchmark results...")
        self.load_results()
        
        if not self.results:
            print("No results loaded. Exiting.")
            return
        
        print("Extracting metrics...")
        self.extract_metrics()
        
        if not self.comparison_data:
            print("No comparison data extracted. Exiting.")
            return
        
        print("Creating comparison dataframe...")
        self.create_comparison_df()
        
        print("Generating outputs...")
        self.generate_summary_table()
        self.export_csv()
        self.export_json()
        self.create_plots()
        
        print(f"\nComparison complete! Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare fasthurdle benchmark results")
    parser.add_argument("results", nargs="+", help="Benchmark result JSON files")
    parser.add_argument("-o", "--output-dir", default="benchmark/comparison_results",
                        help="Output directory for comparison results")
    parser.add_argument("--baseline", default=None,
                        help="Baseline commit to compare against (default: master if available, otherwise earliest)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip creating plots")
    
    args = parser.parse_args()
    
    # Validate input files
    result_files = []
    for file_pattern in args.results:
        files = list(Path().glob(file_pattern))
        if not files:
            print(f"Warning: No files found matching '{file_pattern}'", file=sys.stderr)
        result_files.extend(files)
    
    if not result_files:
        print("Error: No valid result files found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(result_files)} result files to compare")
    
    # Run comparison
    comparator = BenchmarkComparator(result_files, args.output_dir, baseline_commit=args.baseline)
    
    if args.no_plots:
        global PLOT_AVAILABLE
        PLOT_AVAILABLE = False
    
    comparator.run_comparison()


if __name__ == "__main__":
    main()