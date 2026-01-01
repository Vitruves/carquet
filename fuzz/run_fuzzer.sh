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
    echo "  roundtrip    - Fuzz encode->decode roundtrips"
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

    # Find a clang with libFuzzer support
    # System clang on macOS doesn't have libFuzzer, need LLVM from Homebrew
    CLANG=""

    # Try Homebrew LLVM first (has libFuzzer)
    if [ -x "/opt/homebrew/opt/llvm/bin/clang" ]; then
        CLANG="/opt/homebrew/opt/llvm/bin/clang"
    elif [ -x "/usr/local/opt/llvm/bin/clang" ]; then
        CLANG="/usr/local/opt/llvm/bin/clang"
    elif command -v clang &> /dev/null; then
        # Check if system clang has libFuzzer
        if clang --print-runtime-dir 2>/dev/null | grep -q "lib/clang"; then
            CLANG="clang"
        fi
    fi

    if [ -z "$CLANG" ]; then
        echo -e "${RED}Error: No clang with libFuzzer support found.${NC}"
        echo ""
        echo "On macOS, install LLVM via Homebrew:"
        echo "  brew install llvm"
        echo ""
        echo "On Linux, install clang:"
        echo "  apt install clang  # Debian/Ubuntu"
        echo "  dnf install clang  # Fedora"
        exit 1
    fi

    echo -e "${GREEN}Using clang: $CLANG${NC}"

    # Configure with fuzzing enabled and sanitizers
    # ASan: memory errors (buffer overflows, use-after-free)
    # UBSan: undefined behavior (signed overflow, null deref, etc.)
    SANITIZER_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all"

    cmake -B "$BUILD_DIR" \
        -DCMAKE_C_COMPILER="$CLANG" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_FLAGS="$SANITIZER_FLAGS" \
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

    # Set library path for Homebrew/Linuxbrew OpenMP
    # Find libomp.so location dynamically
    local omp_lib=""
    if [ -f "/home/linuxbrew/.linuxbrew/lib/libomp.so" ]; then
        omp_lib="/home/linuxbrew/.linuxbrew/lib"
    elif omp_lib=$(dirname "$(find /home/linuxbrew -name 'libomp.so' 2>/dev/null | head -1)" 2>/dev/null) && [ -n "$omp_lib" ]; then
        : # Found via find
    fi

    if [ -n "$omp_lib" ]; then
        export LD_LIBRARY_PATH="$omp_lib:${LD_LIBRARY_PATH:-}"
    elif [ -d "/opt/homebrew/lib" ]; then
        export DYLD_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_LIBRARY_PATH:-}"
    elif [ -d "/usr/local/opt/llvm/lib" ]; then
        export DYLD_LIBRARY_PATH="/usr/local/opt/llvm/lib:${DYLD_LIBRARY_PATH:-}"
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
                # Seeds for each encoding (12 modes)
                printf '\x00\x08\x00' > "$corpus/seed_rle"
                printf '\x01\x10\x00' > "$corpus/seed_delta32"
                printf '\x02\x10\x00' > "$corpus/seed_delta64"
                printf '\x03\x10\x00\x00\x00\x00' > "$corpus/seed_plain32"
                printf '\x04\x10\x00\x00\x00\x00\x00\x00\x00\x00' > "$corpus/seed_plain64"
                printf '\x05\x10\x00\x00\x00\x00\x00\x00\x00\x00' > "$corpus/seed_plain_double"
                printf '\x06\x10\x01\x00\x00\x00\x02\x00' > "$corpus/seed_dict32"
                printf '\x07\x10\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00' > "$corpus/seed_dict64"
                printf '\x08\x10\x00\x00\x80\x3f\x02\x00' > "$corpus/seed_dict_float"
                printf '\x09\x10\x00\x00\x00\x00\x00\x00\xf0\x3f\x02\x00' > "$corpus/seed_dict_double"
                printf '\x0a\x10\x00\x00\x00\x00\x00\x00\x00\x00' > "$corpus/seed_bss_float"
                printf '\x0b\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' > "$corpus/seed_bss_double"
                ;;
            thrift)
                # Minimal Thrift structures
                printf '\x00\x00' > "$corpus/seed_empty"
                printf '\x15\x00\x00' > "$corpus/seed_struct"
                printf '\x19\x00\x00\x00' > "$corpus/seed_list"
                ;;
            roundtrip)
                # Seeds for roundtrip testing (mode byte + data)
                printf '\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00' > "$corpus/seed_delta32"
                printf '\x01\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00' > "$corpus/seed_delta64"
                printf '\x02hello world test data' > "$corpus/seed_lz4"
                printf '\x03\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40' > "$corpus/seed_bss_float"
                printf '\x04\x00\x00\x00\x00\x00\x00\xf0\x3f\x00\x00\x00\x00\x00\x00\x00\x40' > "$corpus/seed_bss_double"
                ;;
        esac
    fi

    # Use dictionary file if available
    DICT_FILE="$SCRIPT_DIR/parquet.dict"
    DICT_OPT=""
    if [ -f "$DICT_FILE" ]; then
        DICT_OPT="-dict=$DICT_FILE"
        echo "Using dictionary: $DICT_FILE"
    fi

    echo -e "${GREEN}Running fuzzer: $target${NC}"
    echo "Corpus: $corpus"
    echo "Options: $DICT_OPT $@"
    echo ""

    "$fuzzer" "$corpus" $DICT_OPT "$@"
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
    reader|compression|encodings|thrift|roundtrip)
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
        for t in reader compression encodings thrift roundtrip; do
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
