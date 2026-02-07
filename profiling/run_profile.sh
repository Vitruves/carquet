#!/bin/bash
#
# Carquet Profiling Script
#
# Comprehensive profiling with perf, flamegraphs, and statistical analysis.
# Requires: perf, FlameGraph (optional, for visualization)
#
# Usage:
#   ./run_profile.sh [mode] [options]
#
# Modes:
#   full        - Full profiling (record + report + annotate)
#   record      - Record only (for later analysis)
#   stat        - Quick statistics (cache, branches, IPC)
#   flamegraph  - Generate flamegraph SVG
#   micro       - Run micro-benchmarks
#   compare     - Compare scalar vs SIMD implementations
#
# Examples:
#   ./run_profile.sh full
#   ./run_profile.sh flamegraph --rows 10000000
#   ./run_profile.sh stat --component rle
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
PROFILE_OUTPUT_DIR="${SCRIPT_DIR}/output"
FLAMEGRAPH_DIR="${HOME}/FlameGraph"  # Clone from https://github.com/brendangregg/FlameGraph

# Default parameters
ROWS=5000000
ITERATIONS=5
COMPONENT="all"
MODE="${1:-full}"
shift 2>/dev/null || true

# Parse additional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rows) ROWS="$2"; shift 2;;
        --iterations) ITERATIONS="$2"; shift 2;;
        --component) COMPONENT="$2"; shift 2;;
        --output) PROFILE_OUTPUT_DIR="$2"; shift 2;;
        *) shift;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check dependencies
check_deps() {
    if ! command -v perf &>/dev/null; then
        error "perf not found. Install with: sudo apt install linux-tools-generic"
        exit 1
    fi

    if [[ "$MODE" == "flamegraph" ]] && [[ ! -d "$FLAMEGRAPH_DIR" ]]; then
        warn "FlameGraph not found at $FLAMEGRAPH_DIR"
        warn "Clone it: git clone https://github.com/brendangregg/FlameGraph $FLAMEGRAPH_DIR"
    fi
}

# Build profiling binaries
build_binaries() {
    info "Building profiling binaries with debug symbols..."

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Build with debug symbols and optimization
    cmake .. \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_C_FLAGS="-g -fno-omit-frame-pointer" \
        -DCARQUET_BUILD_BENCHMARKS=ON \
        -DCARQUET_BUILD_TESTS=OFF

    make -j$(nproc) profile_read profile_micro 2>/dev/null || make profile_read profile_micro

    success "Binaries built"
    cd "$SCRIPT_DIR"
}

# Create output directory
setup_output() {
    mkdir -p "$PROFILE_OUTPUT_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    export PROFILE_PREFIX="${PROFILE_OUTPUT_DIR}/carquet_${TIMESTAMP}"
}

# Run perf stat for quick overview
run_stat() {
    info "Running perf stat..."

    local binary="$BUILD_DIR/profile_read"
    local args="-r $ROWS -i $ITERATIONS"

    if [[ "$COMPONENT" != "all" ]]; then
        binary="$BUILD_DIR/profile_micro"
        args="--component $COMPONENT --count 1000000 --iterations 100"
    fi

    echo ""
    echo "=== CPU Statistics ==="

    # Basic stats
    perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
        "$binary" $args 2>&1 | tee "${PROFILE_PREFIX}_stat.txt"

    echo ""
    echo "=== Memory Statistics ==="

    # Memory hierarchy stats (if available)
    perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        "$binary" $args 2>&1 | tee -a "${PROFILE_PREFIX}_stat.txt" || true

    success "Stats saved to ${PROFILE_PREFIX}_stat.txt"
}

# Run perf record for detailed profiling
run_record() {
    info "Recording with perf (this may take a while)..."

    local binary="$BUILD_DIR/profile_read"
    local args="-r $ROWS -i $ITERATIONS -v"

    if [[ "$COMPONENT" != "all" && "$COMPONENT" != "read" ]]; then
        binary="$BUILD_DIR/profile_micro"
        args="--component $COMPONENT --count 2000000 --iterations 200"
    fi

    # Record call graph
    perf record -g --call-graph dwarf -F 999 \
        -o "${PROFILE_PREFIX}_perf.data" \
        "$binary" $args

    success "Recording saved to ${PROFILE_PREFIX}_perf.data"
}

# Generate perf report
run_report() {
    local data_file="${PROFILE_PREFIX}_perf.data"

    if [[ ! -f "$data_file" ]]; then
        error "No perf data found. Run 'record' first."
        exit 1
    fi

    info "Generating perf report..."

    # Hierarchical report
    perf report -i "$data_file" --hierarchy --stdio \
        --no-children --percent-limit 0.5 \
        > "${PROFILE_PREFIX}_report.txt"

    # Top functions
    perf report -i "$data_file" --stdio \
        --no-children --percent-limit 0.5 -n \
        > "${PROFILE_PREFIX}_top_functions.txt"

    success "Reports saved to ${PROFILE_PREFIX}_*.txt"

    echo ""
    echo "=== Top 20 Functions ==="
    head -50 "${PROFILE_PREFIX}_top_functions.txt" | grep -E "^\s+[0-9]"
}

# Generate annotated source for specific function
run_annotate() {
    local data_file="${PROFILE_PREFIX}_perf.data"

    if [[ ! -f "$data_file" ]]; then
        error "No perf data found. Run 'record' first."
        exit 1
    fi

    info "Generating source annotations for hot functions..."

    # Key functions to annotate
    local functions=(
        "carquet_rle_decoder_get"
        "carquet_rle_decoder_get_batch"
        "carquet_rle_decode_levels"
        "read_batch_loop"
        "carquet_dispatch_gather_i32"
        "carquet_sse_gather_i32"
        "carquet_lz4_decompress"
    )

    mkdir -p "${PROFILE_PREFIX}_annotations"

    for func in "${functions[@]}"; do
        perf annotate -i "$data_file" -s "$func" --stdio \
            > "${PROFILE_PREFIX}_annotations/${func}.txt" 2>/dev/null || true
    done

    success "Annotations saved to ${PROFILE_PREFIX}_annotations/"
}

# Generate flamegraph
run_flamegraph() {
    local data_file="${PROFILE_PREFIX}_perf.data"

    if [[ ! -f "$data_file" ]]; then
        run_record
    fi

    if [[ ! -d "$FLAMEGRAPH_DIR" ]]; then
        error "FlameGraph not found. Clone it first:"
        error "git clone https://github.com/brendangregg/FlameGraph $FLAMEGRAPH_DIR"
        exit 1
    fi

    info "Generating flamegraph..."

    # Convert perf data to collapsed stacks
    perf script -i "$data_file" | \
        "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" --all \
        > "${PROFILE_PREFIX}_collapsed.txt"

    # Generate SVG
    "$FLAMEGRAPH_DIR/flamegraph.pl" \
        --title "Carquet Read Path ($ROWS rows)" \
        --subtitle "$(date)" \
        --width 1800 \
        --colors hot \
        "${PROFILE_PREFIX}_collapsed.txt" \
        > "${PROFILE_PREFIX}_flamegraph.svg"

    success "Flamegraph saved to ${PROFILE_PREFIX}_flamegraph.svg"

    # Generate reverse flamegraph (icicle graph)
    "$FLAMEGRAPH_DIR/flamegraph.pl" \
        --title "Carquet Read Path (Reversed)" \
        --reverse --inverted \
        --width 1800 \
        --colors hot \
        "${PROFILE_PREFIX}_collapsed.txt" \
        > "${PROFILE_PREFIX}_icicle.svg"

    success "Icicle graph saved to ${PROFILE_PREFIX}_icicle.svg"

    echo ""
    info "Open in browser: file://${PROFILE_PREFIX}_flamegraph.svg"
}

# Run micro-benchmarks
run_micro() {
    info "Running micro-benchmarks..."

    "$BUILD_DIR/profile_micro" --component "$COMPONENT" \
        --count 2000000 --iterations 200 \
        | tee "${PROFILE_PREFIX}_micro.txt"

    success "Micro-benchmark results saved to ${PROFILE_PREFIX}_micro.txt"
}

# Compare implementations
run_compare() {
    info "Comparing scalar vs SIMD implementations..."

    echo "=== Gather Operations ===" | tee "${PROFILE_PREFIX}_compare.txt"
    "$BUILD_DIR/profile_micro" --component gather --count 2000000 --iterations 200 \
        | tee -a "${PROFILE_PREFIX}_compare.txt"

    echo "" | tee -a "${PROFILE_PREFIX}_compare.txt"
    echo "=== Null Bitmap ===" | tee -a "${PROFILE_PREFIX}_compare.txt"
    "$BUILD_DIR/profile_micro" --component null --count 2000000 --iterations 200 \
        | tee -a "${PROFILE_PREFIX}_compare.txt"

    echo "" | tee -a "${PROFILE_PREFIX}_compare.txt"
    echo "=== Dispatch Overhead ===" | tee -a "${PROFILE_PREFIX}_compare.txt"
    "$BUILD_DIR/profile_micro" --component dispatch --iterations 500000 \
        | tee -a "${PROFILE_PREFIX}_compare.txt"

    success "Comparison saved to ${PROFILE_PREFIX}_compare.txt"
}

# Full profiling run
run_full() {
    info "Running full profiling suite..."

    run_stat
    echo ""
    run_record
    echo ""
    run_report
    echo ""
    run_annotate
    echo ""

    if [[ -d "$FLAMEGRAPH_DIR" ]]; then
        run_flamegraph
    else
        warn "Skipping flamegraph (FlameGraph not installed)"
    fi

    echo ""
    echo "=== Profiling Complete ==="
    echo "Output directory: $PROFILE_OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    ls -la "${PROFILE_PREFIX}"* 2>/dev/null | awk '{print "  " $NF}'
}

# Main
main() {
    check_deps

    echo "========================================"
    echo "  Carquet Profiling Suite"
    echo "========================================"
    echo "Mode: $MODE"
    echo "Rows: $ROWS"
    echo "Iterations: $ITERATIONS"
    echo "Component: $COMPONENT"
    echo ""

    # Always build first
    build_binaries
    setup_output

    case "$MODE" in
        full)       run_full;;
        record)     run_record;;
        stat)       run_stat;;
        report)     run_report;;
        annotate)   run_annotate;;
        flamegraph) run_flamegraph;;
        micro)      run_micro;;
        compare)    run_compare;;
        *)
            error "Unknown mode: $MODE"
            echo "Valid modes: full, record, stat, report, annotate, flamegraph, micro, compare"
            exit 1
            ;;
    esac
}

main "$@"
