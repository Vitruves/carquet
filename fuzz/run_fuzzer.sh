#!/bin/bash
#
# Run carquet fuzzers with libFuzzer or AFL++
#
# Usage:
#   ./run_fuzzer.sh [target] [options]
#
# Targets: reader, compression, encodings, thrift, all
#
# Examples:
#   ./run_fuzzer.sh reader              # Run reader fuzzer
#   ./run_fuzzer.sh all -max_total_time=3600  # Run all for 1 hour
#   ./run_fuzzer.sh compression -jobs=4  # Run with 4 parallel jobs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build-fuzz"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [target] [libfuzzer_options...]"
    echo ""
    echo "Targets:"
    echo "  reader       - Fuzz the Parquet file reader"
    echo "  compression  - Fuzz compression decoders"
    echo "  encodings    - Fuzz encoding decoders"
    echo "  thrift       - Fuzz Thrift protocol decoder"
    echo "  all          - Run all fuzzers sequentially"
    echo "  build        - Just build the fuzzers"
    echo ""
    echo "Common libFuzzer options:"
    echo "  -max_total_time=N    Run for N seconds"
    echo "  -jobs=N              Run N parallel jobs"
    echo "  -workers=N           Use N worker processes"
    echo "  -dict=FILE           Use dictionary file"
    echo ""
    echo "Examples:"
    echo "  $0 reader -max_total_time=300"
    echo "  $0 all -jobs=4 -max_total_time=3600"
    exit 1
}

build_fuzzers() {
    echo -e "${YELLOW}Building fuzz targets...${NC}"

    # Check for clang
    if ! command -v clang &> /dev/null; then
        echo -e "${RED}Error: clang not found. libFuzzer requires clang.${NC}"
        exit 1
    fi

    # Configure with fuzzing enabled
    cmake -B "$BUILD_DIR" \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCARQUET_BUILD_FUZZ=ON \
        -DCARQUET_BUILD_TESTS=OFF \
        -DCARQUET_BUILD_EXAMPLES=OFF \
        -DCARQUET_BUILD_BENCHMARKS=OFF \
        "$PROJECT_DIR"

    # Build
    cmake --build "$BUILD_DIR" -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

    echo -e "${GREEN}Build complete!${NC}"
}

run_fuzzer() {
    local target=$1
    shift
    local fuzzer="$BUILD_DIR/fuzz/fuzz_$target"
    local corpus="$BUILD_DIR/fuzz/corpus_$target"

    if [ ! -f "$fuzzer" ]; then
        echo -e "${RED}Error: Fuzzer not found: $fuzzer${NC}"
        echo "Run '$0 build' first."
        exit 1
    fi

    # Create corpus directory
    mkdir -p "$corpus"

    # Seed corpus with minimal valid inputs if empty
    if [ -z "$(ls -A "$corpus" 2>/dev/null)" ]; then
        echo -e "${YELLOW}Creating seed corpus for $target...${NC}"
        case $target in
            reader)
                # Minimal Parquet magic bytes
                echo -n "PAR1" > "$corpus/seed_magic"
                printf 'PAR1\x00\x00\x00\x00PAR1' > "$corpus/seed_minimal"
                ;;
            compression)
                # Seeds for each codec type
                printf '\x00\x00' > "$corpus/seed_snappy"
                printf '\x01\x00' > "$corpus/seed_lz4"
                printf '\x02\x00' > "$corpus/seed_gzip"
                printf '\x03\x00' > "$corpus/seed_zstd"
                ;;
            encodings)
                # Seeds for each encoding
                printf '\x00\x08\x00' > "$corpus/seed_rle"
                printf '\x01\x10\x00' > "$corpus/seed_delta32"
                printf '\x02\x10\x00' > "$corpus/seed_delta64"
                ;;
            thrift)
                # Minimal Thrift structures
                printf '\x00\x00' > "$corpus/seed_empty"
                ;;
        esac
    fi

    echo -e "${GREEN}Running fuzzer: $target${NC}"
    echo "Corpus: $corpus"
    echo "Options: $@"
    echo ""

    "$fuzzer" "$corpus" "$@"
}

# Parse arguments
TARGET=${1:-help}
shift 2>/dev/null || true

case $TARGET in
    help|--help|-h)
        usage
        ;;
    build)
        build_fuzzers
        ;;
    reader|compression|encodings|thrift)
        if [ ! -d "$BUILD_DIR" ]; then
            build_fuzzers
        fi
        run_fuzzer "$TARGET" "$@"
        ;;
    all)
        if [ ! -d "$BUILD_DIR" ]; then
            build_fuzzers
        fi
        # Default: 5 minutes per target
        TIME=${1:--max_total_time=300}
        for t in reader compression encodings thrift; do
            echo ""
            echo -e "${YELLOW}========== Fuzzing: $t ==========${NC}"
            run_fuzzer "$t" "$TIME" || true
        done
        echo -e "${GREEN}All fuzzers complete!${NC}"
        ;;
    *)
        echo -e "${RED}Unknown target: $TARGET${NC}"
        usage
        ;;
esac
