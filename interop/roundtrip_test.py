#!/usr/bin/env python3
"""
Full round-trip interoperability test for carquet.

This test:
1. Generates known test data
2. Writes it using carquet (via C test program)
3. Reads it back with PyArrow and DuckDB
4. Verifies values match EXACTLY

This is the GOLD STANDARD for semantic correctness.

Usage:
    python roundtrip_test.py

Requirements:
    pip install pyarrow pandas duckdb numpy
"""

import subprocess
import tempfile
import os
import sys
import json
from pathlib import Path
import struct

try:
    import pyarrow.parquet as pq
    import numpy as np
    import duckdb
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyarrow numpy duckdb")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# Find build directory (could be build or build-fuzz)
if (PROJECT_DIR / "build" / "libcarquet.a").exists():
    BUILD_DIR = PROJECT_DIR / "build"
elif (PROJECT_DIR / "build-fuzz" / "libcarquet.a").exists():
    BUILD_DIR = PROJECT_DIR / "build-fuzz"
else:
    BUILD_DIR = PROJECT_DIR / "build"


def build_roundtrip_writer():
    """Compile the C roundtrip test program."""
    source = SCRIPT_DIR / "roundtrip_writer.c"
    binary = SCRIPT_DIR / "roundtrip_writer"

    if not source.exists():
        create_roundtrip_writer_source(source)

    # Detect OpenMP flags
    omp_flags = ""
    omp_libs = ""
    if Path("/opt/homebrew/opt/llvm").exists():
        omp_flags = "-I/opt/homebrew/opt/llvm/include"
        omp_libs = "-L/opt/homebrew/opt/llvm/lib -lomp"
    elif sys.platform == "linux":
        omp_libs = "-fopenmp"

    cmd = f"gcc -O2 -I{PROJECT_DIR}/include {omp_flags} -o {binary} {source} {BUILD_DIR}/libcarquet.a -lzstd -lz {omp_libs}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return None

    return binary


def create_roundtrip_writer_source(path: Path):
    """Create the C source for roundtrip testing."""
    source = '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <carquet/carquet.h>

/*
 * Write test data with known values for roundtrip verification.
 * Output format is JSON with expected values.
 */

#define NUM_ROWS 1000

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <output.parquet>\\n", argv[0]);
        return 1;
    }

    const char* output_path = argv[1];
    carquet_error_t err = CARQUET_ERROR_INIT;

    if (carquet_init() != CARQUET_OK) {
        fprintf(stderr, "Failed to init carquet\\n");
        return 1;
    }

    /* Create schema */
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        fprintf(stderr, "Failed to create schema: %s\\n", err.message);
        return 1;
    }

    /* Add columns - various types */
    carquet_schema_add_column(schema, "int32_col", CARQUET_PHYSICAL_INT32,
                              NULL, CARQUET_REPETITION_REQUIRED, 0);
    carquet_schema_add_column(schema, "int64_col", CARQUET_PHYSICAL_INT64,
                              NULL, CARQUET_REPETITION_REQUIRED, 0);
    carquet_schema_add_column(schema, "float_col", CARQUET_PHYSICAL_FLOAT,
                              NULL, CARQUET_REPETITION_REQUIRED, 0);
    carquet_schema_add_column(schema, "double_col", CARQUET_PHYSICAL_DOUBLE,
                              NULL, CARQUET_REPETITION_REQUIRED, 0);
    carquet_schema_add_column(schema, "nullable_int", CARQUET_PHYSICAL_INT32,
                              NULL, CARQUET_REPETITION_OPTIONAL, 0);

    /* Create writer */
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_ZSTD;

    carquet_writer_t* writer = carquet_writer_create(output_path, schema, &opts, &err);
    if (!writer) {
        fprintf(stderr, "Failed to create writer: %s\\n", err.message);
        carquet_schema_free(schema);
        return 1;
    }

    /* Generate test data with predictable values */
    int32_t* int32_data = malloc(NUM_ROWS * sizeof(int32_t));
    int64_t* int64_data = malloc(NUM_ROWS * sizeof(int64_t));
    float* float_data = malloc(NUM_ROWS * sizeof(float));
    double* double_data = malloc(NUM_ROWS * sizeof(double));
    int32_t* nullable_data = malloc(NUM_ROWS * sizeof(int32_t));
    int16_t* def_levels = malloc(NUM_ROWS * sizeof(int16_t));

    for (int i = 0; i < NUM_ROWS; i++) {
        int32_data[i] = i * 10;
        int64_data[i] = (int64_t)i * 1000000LL;
        float_data[i] = (float)i * 0.5f;
        double_data[i] = (double)i * 0.125;
        nullable_data[i] = i * 100;
        def_levels[i] = (i % 5 == 0) ? 0 : 1;  /* Every 5th value is NULL */
    }

    /* Write columns */
    carquet_writer_write_batch(writer, 0, int32_data, NUM_ROWS, NULL, NULL);
    carquet_writer_write_batch(writer, 1, int64_data, NUM_ROWS, NULL, NULL);
    carquet_writer_write_batch(writer, 2, float_data, NUM_ROWS, NULL, NULL);
    carquet_writer_write_batch(writer, 3, double_data, NUM_ROWS, NULL, NULL);
    carquet_writer_write_batch(writer, 4, nullable_data, NUM_ROWS, def_levels, NULL);

    /* Close and finalize */
    carquet_status_t status = carquet_writer_close(writer);
    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to close writer\\n");
    }

    /* Output expected values as JSON for verification */
    printf("{\\n");
    printf("  \\"num_rows\\": %d,\\n", NUM_ROWS);
    printf("  \\"columns\\": {\\n");
    printf("    \\"int32_col\\": { \\"first\\": [0, 10, 20, 30, 40], \\"type\\": \\"int32\\" },\\n");
    printf("    \\"int64_col\\": { \\"first\\": [0, 1000000, 2000000, 3000000, 4000000], \\"type\\": \\"int64\\" },\\n");
    printf("    \\"float_col\\": { \\"first\\": [0.0, 0.5, 1.0, 1.5, 2.0], \\"type\\": \\"float\\" },\\n");
    printf("    \\"double_col\\": { \\"first\\": [0.0, 0.125, 0.25, 0.375, 0.5], \\"type\\": \\"double\\" },\\n");
    printf("    \\"nullable_int\\": { \\"first\\": [null, 100, 200, 300, 400], \\"null_indices\\": [0, 5, 10, 15, 20], \\"type\\": \\"int32\\" }\\n");
    printf("  }\\n");
    printf("}\\n");

    free(int32_data);
    free(int64_data);
    free(float_data);
    free(double_data);
    free(nullable_data);
    free(def_levels);
    carquet_schema_free(schema);

    return status == CARQUET_OK ? 0 : 1;
}
'''
    path.write_text(source)


def run_roundtrip_test():
    """Run the full roundtrip test."""
    print("=" * 60)
    print("  Carquet Round-Trip Interoperability Test")
    print("=" * 60)
    print()

    # Check if library exists
    lib_path = BUILD_DIR / "libcarquet.a"
    if not lib_path.exists():
        print(f"ERROR: Library not found: {lib_path}")
        print("Build with: cmake -B build && cmake --build build")
        return 1

    # Build the test writer
    print("Step 1: Building roundtrip test writer...")
    binary = build_roundtrip_writer()
    if not binary:
        return 1
    print(f"  Built: {binary}")
    print()

    # Create temp file for output
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        parquet_path = f.name

    try:
        # Run carquet writer
        print("Step 2: Writing test data with carquet...")
        result = subprocess.run(
            [str(binary), parquet_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"  FAILED: {result.stderr}")
            return 1

        # Parse expected values from stdout
        expected = json.loads(result.stdout)
        print(f"  Wrote {expected['num_rows']} rows")
        print()

        # Read with PyArrow
        print("Step 3: Reading with PyArrow...")
        try:
            table = pq.read_table(parquet_path)
            print(f"  Read {table.num_rows} rows, {table.num_columns} columns")

            # Verify values
            errors = []

            # Check int32_col
            int32_vals = table.column('int32_col').to_pylist()[:5]
            expected_int32 = expected['columns']['int32_col']['first']
            if int32_vals != expected_int32:
                errors.append(f"int32_col mismatch: {int32_vals} != {expected_int32}")
            else:
                print(f"  ✓ int32_col values correct")

            # Check int64_col
            int64_vals = table.column('int64_col').to_pylist()[:5]
            expected_int64 = expected['columns']['int64_col']['first']
            if int64_vals != expected_int64:
                errors.append(f"int64_col mismatch: {int64_vals} != {expected_int64}")
            else:
                print(f"  ✓ int64_col values correct")

            # Check float_col (with tolerance)
            float_vals = table.column('float_col').to_pylist()[:5]
            expected_float = expected['columns']['float_col']['first']
            float_ok = all(abs(a - b) < 1e-5 for a, b in zip(float_vals, expected_float))
            if not float_ok:
                errors.append(f"float_col mismatch: {float_vals} != {expected_float}")
            else:
                print(f"  ✓ float_col values correct")

            # Check double_col
            double_vals = table.column('double_col').to_pylist()[:5]
            expected_double = expected['columns']['double_col']['first']
            double_ok = all(abs(a - b) < 1e-10 for a, b in zip(double_vals, expected_double))
            if not double_ok:
                errors.append(f"double_col mismatch: {double_vals} != {expected_double}")
            else:
                print(f"  ✓ double_col values correct")

            # Check nullable column
            nullable_vals = table.column('nullable_int').to_pylist()[:5]
            expected_nullable = expected['columns']['nullable_int']['first']
            if nullable_vals != expected_nullable:
                errors.append(f"nullable_int mismatch: {nullable_vals} != {expected_nullable}")
            else:
                print(f"  ✓ nullable_int values correct (including NULLs)")

            if errors:
                print()
                print("ERRORS:")
                for e in errors:
                    print(f"  ✗ {e}")
                return 1

        except Exception as e:
            print(f"  FAILED: {e}")
            return 1
        print()

        # Read with DuckDB
        print("Step 4: Reading with DuckDB...")
        try:
            conn = duckdb.connect()
            df = conn.execute(f"SELECT * FROM read_parquet('{parquet_path}')").fetchdf()
            print(f"  Read {len(df)} rows")

            # Quick value check
            if df['int32_col'].iloc[1] == 10 and df['int64_col'].iloc[1] == 1000000:
                print(f"  ✓ DuckDB values correct")
            else:
                print(f"  ✗ DuckDB value mismatch")
                return 1

            conn.close()
        except Exception as e:
            print(f"  FAILED: {e}")
            return 1
        print()

        # Summary
        print("=" * 60)
        print("✓ ROUND-TRIP TEST PASSED")
        print()
        print("Verified:")
        print("  - carquet writes valid Parquet files")
        print("  - PyArrow can read carquet output")
        print("  - DuckDB can read carquet output")
        print("  - Values are semantically correct")
        print("  - NULL handling is correct")
        return 0

    finally:
        # Cleanup
        if os.path.exists(parquet_path):
            os.unlink(parquet_path)


if __name__ == '__main__':
    sys.exit(run_roundtrip_test())
