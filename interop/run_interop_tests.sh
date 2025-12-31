#!/bin/bash
#
# Run carquet interoperability tests
#
# This script:
# 1. Generates Parquet files using PyArrow, DuckDB, and fastparquet
# 2. Compiles the test_interop program
# 3. Tests that carquet can read all generated files
#
# Requirements:
#   - Python 3 with pyarrow, pandas, duckdb, fastparquet
#   - C compiler (gcc or clang)
#   - carquet built (../build/libcarquet.a)
#
# Usage:
#   ./run_interop_tests.sh [--generate-only] [--test-only] [-v]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
TEST_FILES_DIR="$SCRIPT_DIR/test_files"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
GENERATE_ONLY=0
TEST_ONLY=0
VERBOSE=""

for arg in "$@"; do
    case $arg in
        --generate-only)
            GENERATE_ONLY=1
            ;;
        --test-only)
            TEST_ONLY=1
            ;;
        -v|--verbose)
            VERBOSE="-v"
            ;;
    esac
done

# Step 1: Generate test files
generate_files() {
    echo -e "${YELLOW}Step 1: Generating test files...${NC}"

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 not found${NC}"
        exit 1
    fi

    # Check for required Python packages
    python3 -c "import pyarrow" 2>/dev/null || {
        echo -e "${YELLOW}Warning: pyarrow not installed. Install with: pip install pyarrow${NC}"
    }

    # Generate files
    python3 "$SCRIPT_DIR/generate_test_files.py" "$TEST_FILES_DIR"

    echo -e "${GREEN}Test files generated in: $TEST_FILES_DIR${NC}"
}

# Step 2: Build test program
build_test() {
    echo -e "${YELLOW}Step 2: Building test_interop...${NC}"

    # Check for library
    if [ ! -f "$BUILD_DIR/libcarquet.a" ]; then
        echo -e "${YELLOW}Building carquet library first...${NC}"
        cmake -B "$BUILD_DIR" "$PROJECT_DIR"
        cmake --build "$BUILD_DIR"
    fi

    # Compile test program
    gcc -O2 -I"$PROJECT_DIR/include" -I"$PROJECT_DIR/src" \
        -o "$SCRIPT_DIR/test_interop" \
        "$SCRIPT_DIR/test_interop.c" \
        "$BUILD_DIR/libcarquet.a" \
        -lzstd -lz

    echo -e "${GREEN}Built: $SCRIPT_DIR/test_interop${NC}"
}

# Step 3: Run tests
run_tests() {
    echo -e "${YELLOW}Step 3: Running interoperability tests...${NC}"

    if [ ! -f "$SCRIPT_DIR/test_interop" ]; then
        echo -e "${RED}Error: test_interop not found. Run without --test-only first.${NC}"
        exit 1
    fi

    if [ ! -d "$TEST_FILES_DIR" ]; then
        echo -e "${RED}Error: Test files not found. Run without --test-only first.${NC}"
        exit 1
    fi

    "$SCRIPT_DIR/test_interop" --dir "$TEST_FILES_DIR" $VERBOSE
}

# Main
echo "=========================================="
echo "  Carquet Interoperability Test Suite"
echo "=========================================="
echo ""

if [ $TEST_ONLY -eq 0 ]; then
    generate_files
    echo ""
fi

if [ $GENERATE_ONLY -eq 0 ]; then
    build_test
    echo ""
    run_tests
fi

echo ""
echo -e "${GREEN}Done!${NC}"
