#!/usr/bin/env python3
"""
Comprehensive verification of carquet output files.

Tests all physical types, all compression codecs, null handling,
and verifies exact values with PyArrow and DuckDB.
"""

import subprocess
import tempfile
import json
import sys
import os
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import numpy as np
    import duckdb
except ImportError as e:
    print(f"Missing: {e}. Install: pip install pyarrow numpy duckdb")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
BUILD_DIR = PROJECT_DIR / "build"


def build_comprehensive_writer():
    """Build the comprehensive test writer."""
    source = SCRIPT_DIR / "roundtrip_comprehensive.c"
    binary = SCRIPT_DIR / "roundtrip_comprehensive"

    if not source.exists():
        print(f"ERROR: {source} not found")
        return None

    # Detect OpenMP
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
        print(f"Compile failed: {result.stderr}")
        return None

    return binary


def verify_with_pyarrow(path, expected, file_info):
    """Verify file with PyArrow."""
    errors = []

    try:
        table = pq.read_table(path)
    except Exception as e:
        return [f"PyArrow failed to read: {e}"]

    # Check row count
    if table.num_rows != expected["num_rows"]:
        errors.append(f"Row count: {table.num_rows} != {expected['num_rows']}")

    cols = file_info["columns"]

    # bool_col
    bools = table.column("bool_col").to_pylist()[:5]
    exp_bools = cols["bool_col"]["first"]
    if bools != exp_bools:
        errors.append(f"bool_col: {bools} != {exp_bools}")

    # int32_col
    int32s = table.column("int32_col").to_pylist()[:5]
    exp_int32s = cols["int32_col"]["first"]
    if int32s != exp_int32s:
        errors.append(f"int32_col: {int32s} != {exp_int32s}")

    # int64_col
    int64s = table.column("int64_col").to_pylist()[:5]
    exp_int64s = cols["int64_col"]["first"]
    if int64s != exp_int64s:
        errors.append(f"int64_col: {int64s} != {exp_int64s}")

    # float_col (with tolerance)
    floats = table.column("float_col").to_pylist()[:5]
    exp_floats = cols["float_col"]["first"]
    for i, (a, b) in enumerate(zip(floats, exp_floats)):
        if abs(a - b) > 1e-4:
            errors.append(f"float_col[{i}]: {a} != {b}")
            break

    # double_col
    doubles = table.column("double_col").to_pylist()[:5]
    exp_doubles = cols["double_col"]["first"]
    for i, (a, b) in enumerate(zip(doubles, exp_doubles)):
        if abs(a - b) > 1e-10:
            errors.append(f"double_col[{i}]: {a} != {b}")
            break

    # string_col (with nulls) - PyArrow returns bytes, decode to str
    strings_raw = table.column("string_col").to_pylist()[:5]
    strings = [s.decode('utf-8') if s is not None else None for s in strings_raw]
    exp_strings = cols["string_col"]["first"]
    if strings != exp_strings:
        errors.append(f"string_col: {strings} != {exp_strings}")

    # nullable_int
    nullable = table.column("nullable_int").to_pylist()[:5]
    exp_nullable = cols["nullable_int"]["first"]
    if nullable != exp_nullable:
        errors.append(f"nullable_int: {nullable} != {exp_nullable}")

    # Verify null counts
    string_nulls = sum(1 for v in table.column("string_col").to_pylist() if v is None)
    exp_string_nulls = expected["verification"]["null_count_string_col"]
    if string_nulls != exp_string_nulls:
        errors.append(f"string null count: {string_nulls} != {exp_string_nulls}")

    nullable_nulls = sum(1 for v in table.column("nullable_int").to_pylist() if v is None)
    exp_nullable_nulls = expected["verification"]["null_count_nullable_int"]
    if nullable_nulls != exp_nullable_nulls:
        errors.append(f"nullable_int null count: {nullable_nulls} != {exp_nullable_nulls}")

    # Verify aggregates
    int32_sum = sum(table.column("int32_col").to_pylist())
    exp_sum = expected["verification"]["int32_sum"]
    if int32_sum != exp_sum:
        errors.append(f"int32 sum: {int32_sum} != {exp_sum}")

    last_int32 = table.column("int32_col").to_pylist()[-1]
    exp_last = expected["verification"]["last_int32"]
    if last_int32 != exp_last:
        errors.append(f"last int32: {last_int32} != {exp_last}")

    return errors


def verify_with_duckdb(path, expected):
    """Verify file with DuckDB."""
    errors = []

    try:
        conn = duckdb.connect()
        df = conn.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()
        conn.close()
    except Exception as e:
        return [f"DuckDB failed to read: {e}"]

    if len(df) != expected["num_rows"]:
        errors.append(f"DuckDB row count: {len(df)} != {expected['num_rows']}")

    # Check sum
    int32_sum = int(df["int32_col"].sum())
    exp_sum = expected["verification"]["int32_sum"]
    if int32_sum != exp_sum:
        errors.append(f"DuckDB int32 sum: {int32_sum} != {exp_sum}")

    return errors


def run_comprehensive_test():
    """Run the full comprehensive test."""
    print("=" * 70)
    print("  Comprehensive Carquet Interoperability Test")
    print("=" * 70)
    print()

    # Check library
    if not (BUILD_DIR / "libcarquet.a").exists():
        print(f"ERROR: Library not found. Build first.")
        return 1

    # Build writer
    print("Building comprehensive test writer...")
    binary = build_comprehensive_writer()
    if not binary:
        return 1
    print(f"  Built: {binary}")
    print()

    # Create temp dir for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run writer
        print("Generating test files (all compressions)...")
        result = subprocess.run(
            [str(binary), tmpdir],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"  FAILED: {result.stderr}")
            return 1

        expected = json.loads(result.stdout)
        print(f"  Generated {len(expected['files'])} files with {expected['num_rows']} rows each")
        print()

        # Test each file
        total_errors = 0
        for file_info in expected["files"]:
            path = file_info["path"]
            compression = file_info["compression"]
            print(f"Testing {compression}...")

            # PyArrow
            pa_errors = verify_with_pyarrow(path, expected, file_info)
            if pa_errors:
                print(f"  PyArrow ERRORS:")
                for e in pa_errors:
                    print(f"    - {e}")
                total_errors += len(pa_errors)
            else:
                print(f"  PyArrow: OK (all types, nulls, values)")

            # DuckDB
            db_errors = verify_with_duckdb(path, expected)
            if db_errors:
                print(f"  DuckDB ERRORS:")
                for e in db_errors:
                    print(f"    - {e}")
                total_errors += len(db_errors)
            else:
                print(f"  DuckDB: OK")

            print()

    # Cleanup binary
    if binary and Path(binary).exists():
        os.unlink(binary)
        dsym = Path(str(binary) + ".dSYM")
        if dsym.exists():
            import shutil
            shutil.rmtree(dsym)

    # Summary
    print("=" * 70)
    if total_errors == 0:
        print("COMPREHENSIVE TEST PASSED")
        print()
        print("Verified:")
        print("  - All physical types: BOOLEAN, INT32, INT64, FLOAT, DOUBLE, BYTE_ARRAY")
        print("  - All compressions: UNCOMPRESSED, SNAPPY, GZIP, LZ4, ZSTD")
        print("  - Null handling: string nulls (every 7th), int nulls (every 5th)")
        print("  - Value correctness: first values, last values, sums, null counts")
        print("  - Readers: PyArrow and DuckDB")
        return 0
    else:
        print(f"FAILED: {total_errors} errors")
        return 1


if __name__ == "__main__":
    sys.exit(run_comprehensive_test())
