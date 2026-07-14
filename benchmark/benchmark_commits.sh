#!/bin/bash
# Automated benchmarking script for multiple fasthurdle commits
# Usage: benchmark/benchmark_commits.sh [options] commit1 commit2 ...
#    or: benchmark/benchmark_commits.sh --file commits.txt

set -euo pipefail

# Check for required commands
if ! command -v timeout >/dev/null 2>&1; then
    echo "Warning: 'timeout' command not found. Install coreutils for build timeouts." >&2
fi

# Run from the repository root so the default paths below resolve consistently
# no matter where the script is invoked from.
cd "$(git rev-parse --show-toplevel)"

# Default configuration
PARALLEL_BUILDS=2
OUTPUT_DIR="benchmark/benchmark_results"
DOCKERFILE="benchmark/Dockerfile.benchmark"
LOG_DIR="benchmark/benchmark_logs"
FORCE_REBUILD=false
NUM_RUNS=10
SAMPLE_SIZES="1000,10000,100000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [COMMITS...]

Benchmark multiple fasthurdle commits by building Docker images and running benchmarks.

OPTIONS:
    -f, --file FILE          Read commits from file (one per line)
    -p, --parallel NUM       Number of parallel builds (default: $PARALLEL_BUILDS)
    -o, --output-dir DIR     Output directory for results (default: $OUTPUT_DIR)
    -n, --num-runs NUM       Number of runs per benchmark (default: $NUM_RUNS)
    -s, --sample-sizes SIZES Comma-separated sample sizes (default: $SAMPLE_SIZES)
    --dockerfile FILE        Dockerfile to use (default: $DOCKERFILE)
    --force-rebuild          Force rebuild of Docker images
    -h, --help               Display this help message

EXAMPLES:
    # Benchmark specific commits
    $0 abc1234 def5678 master

    # Benchmark commits from file
    $0 --file commits.txt

    # With custom options
    $0 --parallel 4 --output-dir results/ --sample-sizes "1000,5000,10000" abc1234 def5678

EOF
    exit 1
}

# Parse command line arguments
COMMITS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            if [[ -f "$2" ]]; then
                while IFS= read -r commit; do
                    # Skip empty lines and comments
                    [[ -z "$commit" || "$commit" =~ ^# ]] && continue
                    COMMITS+=("$commit")
                done < "$2"
            else
                print_error "File not found: $2"
                exit 1
            fi
            shift 2
            ;;
        -p|--parallel)
            PARALLEL_BUILDS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -s|--sample-sizes)
            SAMPLE_SIZES="$2"
            shift 2
            ;;
        --dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            print_error "Unknown option: $1"
            usage
            ;;
        *)
            COMMITS+=("$1")
            shift
            ;;
    esac
done

# Validate inputs
if [[ ${#COMMITS[@]} -eq 0 ]]; then
    print_error "No commits specified"
    usage
fi

# Create output directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Function to build Docker image for a commit
build_image() {
    local commit=$1
    local tag="fasthurdle:benchmark-${commit:0:8}"
    local log_file="$LOG_DIR/build_${commit:0:8}.log"
    
    print_status "Building image for commit $commit..."
    
    # Check if image already exists
    if [[ "$FORCE_REBUILD" != "true" ]] && docker image inspect "$tag" >/dev/null 2>&1; then
        print_warning "Image $tag already exists, skipping build (use --force-rebuild to rebuild)"
        return 0
    fi
    
    # Ensure we're in the repository root for proper build context
    local repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
    
    # Build the image from repository root with timeout
    local build_result
    local build_timeout=1800  # 10 minutes
    local exit_code
    
    # Build with or without timeout depending on availability
    if command -v timeout >/dev/null 2>&1; then
        print_status "Building with ${build_timeout}s timeout..."
        
        (cd "$repo_root" && timeout $build_timeout docker build \
            --build-arg GIT_COMMIT="$commit" \
            -f "$DOCKERFILE" \
            -t "$tag" \
            . > "$log_file" 2>&1)
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            print_status "Successfully built $tag"
            build_result=0
        elif [ $exit_code -eq 124 ]; then
            print_warning "Build timed out after ${build_timeout}s, checking if image exists..."
            # Check if image was actually built despite timeout (BuildKit issue)
            if docker image inspect "$tag" >/dev/null 2>&1; then
                print_warning "Build timed out but image $tag exists, considering it successful"
                build_result=0
            else
                print_error "Build timed out and no image found for commit $commit"
                build_result=1
            fi
        else
            print_error "Failed to build image for commit $commit. See $log_file for details"
            build_result=1
        fi
    else
        print_status "Building without timeout (timeout command not available)..."
        
        if (cd "$repo_root" && docker build \
            --build-arg GIT_COMMIT="$commit" \
            -f "$DOCKERFILE" \
            -t "$tag" \
            . > "$log_file" 2>&1); then
            print_status "Successfully built $tag"
            build_result=0
        else
            print_error "Failed to build image for commit $commit. See $log_file for details"
            build_result=1
        fi
    fi
    
    # Clean up any hanging docker build processes for this commit
    pkill -f "docker build.*$commit" 2>/dev/null || true
    pkill -f "docker-buildx.*$commit" 2>/dev/null || true
    
    return $build_result
}

# Function to run benchmark for a commit
run_benchmark() {
    local commit=$1
    local tag="fasthurdle:benchmark-${commit:0:8}"
    local results_dir="/results"
    local log_file="$LOG_DIR/run_${commit:0:8}.log"
    
    print_status "Running benchmark for commit $commit..."
    
    # Check if image exists
    if ! docker image inspect "$tag" >/dev/null 2>&1; then
        print_error "Image $tag not found, skipping benchmark"
        return 1
    fi
    
    # Get repository root for mounting benchmark script
    local repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
    
    # Make OUTPUT_DIR absolute
    local abs_output_dir=$(cd "$OUTPUT_DIR" && pwd)
    
    # Prepare docker run command
    local docker_cmd=(
        docker run
        --rm
        -v "$abs_output_dir:$results_dir"
        -v "$repo_root/benchmark:/app/benchmark:ro"
        -e GIT_COMMIT="$commit"
        -e NUM_RUNS="$NUM_RUNS"
        -e SAMPLE_SIZES="$SAMPLE_SIZES"
        -e RESULTS_DIR="$results_dir"
        "$tag"
    )
    
    # Run the benchmark
    if "${docker_cmd[@]}" > "$log_file" 2>&1; then
        print_status "Successfully completed benchmark for $commit"
        
        # Show summary
        local result_file=$(ls -t "$OUTPUT_DIR"/benchmark_${commit:0:8}_*.json 2>/dev/null | head -1)
        if [[ -f "$result_file" ]]; then
            print_status "Results saved to: $(basename "$result_file")"
        fi
        return 0
    else
        print_error "Failed to run benchmark for commit $commit. See $log_file for details"
        return 1
    fi
}

# Main execution
print_status "Starting benchmark process for ${#COMMITS[@]} commits"
print_status "Configuration:"
echo "  - Parallel builds: $PARALLEL_BUILDS"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Runs per benchmark: $NUM_RUNS"
echo "  - Sample sizes: $SAMPLE_SIZES"
echo "  - Force rebuild: $FORCE_REBUILD"
echo ""

# Track failures
failed_builds=()
failed_runs=()

# Build images in parallel
print_status "Building Docker images..."
export -f build_image print_status print_error print_warning
export LOG_DIR DOCKERFILE FORCE_REBUILD

# Use GNU parallel if available, otherwise fall back to sequential
if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${COMMITS[@]}" | \
        parallel -j "$PARALLEL_BUILDS" --line-buffer build_image {} || true
else
    print_warning "GNU parallel not found, building sequentially"
    for commit in "${COMMITS[@]}"; do
        if ! build_image "$commit"; then
            failed_builds+=("$commit")
        fi
    done
fi

# Check which builds succeeded
successful_builds=()
for commit in "${COMMITS[@]}"; do
    if docker image inspect "fasthurdle:benchmark-${commit:0:8}" >/dev/null 2>&1; then
        successful_builds+=("$commit")
    else
        failed_builds+=("$commit")
    fi
done

print_status "Built ${#successful_builds[@]} out of ${#COMMITS[@]} images successfully"

# Run benchmarks sequentially to avoid resource contention
print_status "Running benchmarks sequentially..."
for commit in "${successful_builds[@]}"; do
    if ! run_benchmark "$commit"; then
        failed_runs+=("$commit")
    fi
    
    # Add delay between runs to ensure clean state
    sleep 2
done

# Summary report
echo ""
print_status "Benchmark process completed!"
echo ""
echo "Summary:"
echo "--------"
echo "Total commits: ${#COMMITS[@]}"
echo "Successful builds: ${#successful_builds[@]}"
echo "Failed builds: ${#failed_builds[@]}"
echo "Successful benchmarks: $((${#successful_builds[@]} - ${#failed_runs[@]}))"
echo "Failed benchmarks: ${#failed_runs[@]}"

if [[ ${#failed_builds[@]} -gt 0 ]]; then
    echo ""
    print_error "Failed to build images for:"
    printf '  - %s\n' "${failed_builds[@]}"
fi

if [[ ${#failed_runs[@]} -gt 0 ]]; then
    echo ""
    print_error "Failed to run benchmarks for:"
    printf '  - %s\n' "${failed_runs[@]}"
fi

echo ""
print_status "Results saved to: $OUTPUT_DIR"
print_status "Logs saved to: $LOG_DIR"

# Create a summary file
summary_file="$OUTPUT_DIR/benchmark_summary_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Benchmark Summary - $(date)"
    echo "========================="
    echo ""
    echo "Commits benchmarked:"
    for commit in "${successful_builds[@]}"; do
        if [[ ! " ${failed_runs[@]} " =~ " ${commit} " ]]; then
            echo "  - $commit: SUCCESS"
            # Find the latest result file for this commit
            result_file=$(ls -t "$OUTPUT_DIR"/benchmark_${commit:0:8}_*.json 2>/dev/null | head -1)
            if [[ -f "$result_file" ]]; then
                echo "    Result: $(basename "$result_file")"
            fi
        else
            echo "  - $commit: FAILED"
        fi
    done
    echo ""
    echo "Failed builds:"
    for commit in "${failed_builds[@]}"; do
        echo "  - $commit"
    done
} > "$summary_file"

print_status "Summary saved to: $summary_file"

# Exit with appropriate code
if [[ ${#failed_builds[@]} -gt 0 || ${#failed_runs[@]} -gt 0 ]]; then
    exit 1
else
    exit 0
fi
