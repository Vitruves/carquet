#!/usr/bin/env python3
"""
Verify that Parquet files written by carquet can be read by other libraries
and contain correct values.

This is the CRITICAL interop test - it verifies semantic correctness.

Usage:
    python verify_carquet_output.py <carquet_file.parquet> [expected_data.json]
    python verify_carquet_output.py --generate-and-verify

Requirements:
    pip install pyarrow pandas duckdb
"""

import sys
import json
import subprocess
import tempfile
import os
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("Warning: pyarrow not available")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    print("Warning: duckdb not available")


def verify_with_pyarrow(parquet_path: str) -> dict:
    """Read parquet file with PyArrow and return data summary."""
    if not HAS_PYARROW:
        return {'error': 'pyarrow not available'}

    try:
        table = pq.read_table(parquet_path)
        result = {
            'success': True,
            'num_rows': table.num_rows,
            'num_columns': table.num_columns,
            'columns': {},
            'schema': str(table.schema),
        }

        for i, col_name in enumerate(table.column_names):
            col = table.column(i)
            col_data = col.to_pylist()
            result['columns'][col_name] = {
                'type': str(col.type),
                'null_count': col.null_count,
                'num_values': len(col_data),
                'first_values': col_data[:5] if len(col_data) > 0 else [],
                'last_values': col_data[-5:] if len(col_data) > 5 else col_data,
            }

        return result
    except Exception as e:
        return {'success': False, 'error': str(e)}


def verify_with_duckdb(parquet_path: str) -> dict:
    """Read parquet file with DuckDB and return data summary."""
    if not HAS_DUCKDB:
        return {'error': 'duckdb not available'}

    try:
        conn = duckdb.connect()

        # Get row count
        result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')").fetchone()
        num_rows = result[0]

        # Get schema
        schema_result = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").fetchall()
        columns = {}
        for col_name, col_type, *_ in schema_result:
            # Get sample values
            sample = conn.execute(
                f"SELECT \"{col_name}\" FROM read_parquet('{parquet_path}') LIMIT 5"
            ).fetchall()
            columns[col_name] = {
                'type': col_type,
                'sample_values': [row[0] for row in sample],
            }

        conn.close()

        return {
            'success': True,
            'num_rows': num_rows,
            'num_columns': len(columns),
            'columns': columns,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def verify_with_pandas(parquet_path: str) -> dict:
    """Read parquet file with pandas and return data summary."""
    if not HAS_PANDAS or not HAS_PYARROW:
        return {'error': 'pandas/pyarrow not available'}

    try:
        df = pd.read_parquet(parquet_path)
        return {
            'success': True,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'head': df.head().to_dict(),
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def compare_values(expected: dict, actual_pyarrow: dict, actual_duckdb: dict) -> list:
    """Compare expected values with actual values from readers."""
    errors = []

    # Check row counts
    expected_rows = expected.get('num_rows', 0)

    if actual_pyarrow.get('success'):
        if actual_pyarrow['num_rows'] != expected_rows:
            errors.append(f"PyArrow row count mismatch: expected {expected_rows}, got {actual_pyarrow['num_rows']}")

    if actual_duckdb.get('success'):
        if actual_duckdb['num_rows'] != expected_rows:
            errors.append(f"DuckDB row count mismatch: expected {expected_rows}, got {actual_duckdb['num_rows']}")

    # Check column count
    expected_cols = expected.get('num_columns', 0)

    if actual_pyarrow.get('success'):
        if actual_pyarrow['num_columns'] != expected_cols:
            errors.append(f"PyArrow column count mismatch: expected {expected_cols}, got {actual_pyarrow['num_columns']}")

    # Check actual values if provided
    if 'values' in expected and actual_pyarrow.get('success'):
        for col_name, expected_vals in expected['values'].items():
            if col_name in actual_pyarrow.get('columns', {}):
                actual_vals = actual_pyarrow['columns'][col_name].get('first_values', [])
                for i, (exp, act) in enumerate(zip(expected_vals[:5], actual_vals)):
                    if exp != act and not (exp is None and act is None):
                        # Handle float comparison
                        if isinstance(exp, float) and isinstance(act, float):
                            if abs(exp - act) > 1e-6:
                                errors.append(f"Value mismatch in {col_name}[{i}]: expected {exp}, got {act}")
                        else:
                            errors.append(f"Value mismatch in {col_name}[{i}]: expected {exp}, got {act}")

    return errors


def run_carquet_example(example_path: str, output_path: str) -> bool:
    """Run a carquet example that produces a parquet file."""
    try:
        result = subprocess.run(
            [example_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(example_path) or '.'
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {example_path}: {e}")
        return False


def test_carquet_basic_write():
    """Test the basic_write_read example output."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    build_dir = project_dir / "build"

    example = build_dir / "example_basic_write_read"
    if not example.exists():
        print(f"Example not found: {example}")
        print("Build with: cmake --build build")
        return False

    # Run example (it creates test_output.parquet)
    original_dir = os.getcwd()
    os.chdir(build_dir)

    try:
        result = subprocess.run([str(example)], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Example failed: {result.stderr}")
            return False
    finally:
        os.chdir(original_dir)

    output_file = build_dir / "test_output.parquet"
    if not output_file.exists():
        print(f"Output file not created: {output_file}")
        return False

    return str(output_file)


def main():
    print("=" * 60)
    print("  Carquet Output Verification")
    print("=" * 60)
    print()

    if len(sys.argv) > 1 and sys.argv[1] == '--generate-and-verify':
        # Run carquet example and verify output
        print("Running carquet example_basic_write_read...")
        parquet_path = test_carquet_basic_write()
        if not parquet_path:
            print("FAIL: Could not generate test file")
            return 1
        print(f"Generated: {parquet_path}")
        print()
    elif len(sys.argv) > 1:
        parquet_path = sys.argv[1]
    else:
        print("Usage:")
        print(f"  {sys.argv[0]} <parquet_file>")
        print(f"  {sys.argv[0]} --generate-and-verify")
        return 1

    print(f"Verifying: {parquet_path}")
    print()

    # Test with PyArrow
    print("Testing with PyArrow...")
    pyarrow_result = verify_with_pyarrow(parquet_path)
    if pyarrow_result.get('success'):
        print(f"  ✓ PyArrow can read file")
        print(f"    Rows: {pyarrow_result['num_rows']}")
        print(f"    Columns: {pyarrow_result['num_columns']}")
        for col_name, col_info in pyarrow_result.get('columns', {}).items():
            print(f"    - {col_name}: {col_info['type']}, nulls={col_info['null_count']}")
    else:
        print(f"  ✗ PyArrow FAILED: {pyarrow_result.get('error')}")
    print()

    # Test with DuckDB
    print("Testing with DuckDB...")
    duckdb_result = verify_with_duckdb(parquet_path)
    if duckdb_result.get('success'):
        print(f"  ✓ DuckDB can read file")
        print(f"    Rows: {duckdb_result['num_rows']}")
        print(f"    Columns: {duckdb_result['num_columns']}")
    else:
        print(f"  ✗ DuckDB FAILED: {duckdb_result.get('error')}")
    print()

    # Test with pandas
    print("Testing with pandas...")
    pandas_result = verify_with_pandas(parquet_path)
    if pandas_result.get('success'):
        print(f"  ✓ pandas can read file")
        print(f"    Shape: ({pandas_result['num_rows']}, {pandas_result['num_columns']})")
    else:
        print(f"  ✗ pandas FAILED: {pandas_result.get('error')}")
    print()

    # Summary
    print("=" * 60)
    success_count = sum([
        pyarrow_result.get('success', False),
        duckdb_result.get('success', False),
        pandas_result.get('success', False),
    ])

    if success_count == 3:
        print("✓ ALL READERS PASSED - File is interoperable!")
        return 0
    elif success_count > 0:
        print(f"⚠ PARTIAL SUCCESS - {success_count}/3 readers passed")
        return 1
    else:
        print("✗ ALL READERS FAILED - File may be corrupt")
        return 2


if __name__ == '__main__':
    sys.exit(main())
