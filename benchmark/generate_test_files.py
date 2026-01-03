#!/usr/bin/env python3
"""
Generate test Parquet files for profiling Carquet read performance.

Usage:
    python generate_test_files.py [output_dir] [num_rows]

Default: ./profile_data/ with 10M rows
"""

import os
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "profile_data")
DEFAULT_NUM_ROWS = 10_000_000

COMPRESSIONS = ["NONE", "SNAPPY", "ZSTD"]


def generate_data(num_rows: int, seed: int = 42) -> pa.Table:
    """Generate test data with various patterns."""
    np.random.seed(seed)

    # Sequential INT64 with small noise (delta-encoding friendly)
    id_col = np.arange(num_rows, dtype=np.int64) * 1000 + np.random.randint(0, 100, num_rows)

    # Random INT32
    int32_col = np.random.randint(0, 1_000_000, num_rows, dtype=np.int32)

    # Double with pattern
    double_col = np.arange(num_rows, dtype=np.float64) * 0.001 + np.random.random(num_rows) * 0.01

    # Float random
    float_col = np.random.random(num_rows).astype(np.float32) * 100

    # Low cardinality INT32 (dictionary-friendly, 100 unique values)
    category_col = np.random.randint(0, 100, num_rows, dtype=np.int32)

    # Nullable double (10% nulls)
    nullable_col = np.random.random(num_rows) * 1000
    null_mask = np.random.random(num_rows) < 0.1
    nullable_col = pa.array(nullable_col, mask=null_mask)

    return pa.table({
        "id": id_col,
        "value_i32": int32_col,
        "value_f64": double_col,
        "value_f32": float_col,
        "category": category_col,
        "nullable_val": nullable_col,
    })


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_NUM_ROWS

    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_rows:,} rows of test data...")
    table = generate_data(num_rows)
    print(f"  Schema: {table.schema}")
    print(f"  Memory: {table.nbytes / 1024 / 1024:.1f} MB")

    for compression in COMPRESSIONS:
        filename = os.path.join(output_dir, f"test_{compression.lower()}.parquet")
        print(f"\nWriting {filename}...")

        pq.write_table(
            table,
            filename,
            compression=compression if compression != "NONE" else None,
            use_dictionary=True,
            write_statistics=True,
            row_group_size=1_000_000,  # 1M rows per row group
        )

        file_size = os.path.getsize(filename)
        print(f"  Size: {file_size / 1024 / 1024:.1f} MB")

        # Verify readability
        read_table = pq.read_table(filename)
        assert len(read_table) == num_rows, f"Row count mismatch: {len(read_table)} vs {num_rows}"
        print(f"  Verified: {len(read_table):,} rows, {read_table.num_columns} columns")

    # Also create a file list for the C benchmark
    list_file = os.path.join(output_dir, "files.txt")
    with open(list_file, "w") as f:
        for compression in COMPRESSIONS:
            f.write(f"test_{compression.lower()}.parquet\n")

    print(f"\nDone! Files written to {output_dir}/")
    print(f"File list: {list_file}")


if __name__ == "__main__":
    main()
