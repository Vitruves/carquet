#!/usr/bin/env python3
"""
Benchmark for PyArrow Parquet - compare with Carquet
"""

import os
import time
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

WARMUP_ITERATIONS = 2
BENCHMARK_ITERATIONS = 5


def benchmark_write(filename, num_rows, compression):
    """Write a Parquet file and return (time_ms, file_size)"""
    # Generate data
    ids = np.arange(num_rows, dtype=np.int64)
    values = ids.astype(np.float64) * 1.5 + 0.123456789
    categories = (ids % 100).astype(np.int32)

    table = pa.table({
        'id': ids,
        'value': values,
        'category': categories
    })

    start = time.perf_counter()
    pq.write_table(table, filename, compression=compression, row_group_size=100000)
    elapsed_ms = (time.perf_counter() - start) * 1000

    file_size = os.path.getsize(filename)
    return elapsed_ms, file_size


def benchmark_read(filename, expected_rows):
    """Read a Parquet file and return time_ms"""
    start = time.perf_counter()
    table = pq.read_table(filename)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if len(table) != expected_rows:
        raise ValueError(f"Row count mismatch: {len(table)} vs {expected_rows}")

    # Verify some data
    ids = table['id'].to_numpy()
    values = table['value'].to_numpy()
    categories = table['category'].to_numpy()

    for i in range(min(100, len(ids))):
        assert ids[i] == i, f"ID mismatch at {i}"
        expected = i * 1.5 + 0.123456789
        assert abs(values[i] - expected) < 1e-10, f"Value mismatch at {i}"
        assert categories[i] == i % 100, f"Category mismatch at {i}"

    return elapsed_ms


def run_benchmark(name, num_rows, compression, compression_name):
    """Run a single benchmark configuration"""
    filename = f"/tmp/benchmark_{name}_{compression_name}_pyarrow.parquet"

    print(f"\n=== {name} ({num_rows:,} rows, {compression_name}) ===")

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        benchmark_write(filename, num_rows, compression)
        benchmark_read(filename, num_rows)

    # Benchmark
    write_times = []
    read_times = []
    file_size = 0

    for _ in range(BENCHMARK_ITERATIONS):
        write_ms, file_size = benchmark_write(filename, num_rows, compression)
        read_ms = benchmark_read(filename, num_rows)
        write_times.append(write_ms)
        read_times.append(read_ms)

    write_avg = sum(write_times) / len(write_times)
    read_avg = sum(read_times) / len(read_times)
    rows_per_sec_write = (num_rows / write_avg) * 1000
    rows_per_sec_read = (num_rows / read_avg) * 1000

    print(f"  Write: {write_avg:.2f} ms ({rows_per_sec_write/1e6:.2f} M rows/sec)")
    print(f"  Read:  {read_avg:.2f} ms ({rows_per_sec_read/1e6:.2f} M rows/sec)")
    print(f"  File:  {file_size/(1024*1024):.2f} MB ({file_size/num_rows:.2f} bytes/row)")

    # Output CSV line for parsing
    print(f"CSV:pyarrow,{name},{compression_name},{num_rows},{write_avg:.2f},{read_avg:.2f},{file_size}")

    os.remove(filename)


def main():
    print("PyArrow Benchmark")
    print("=================")
    print(f"PyArrow version: {pa.__version__}")

    # Small dataset
    run_benchmark("small", 100_000, None, "none")
    run_benchmark("small", 100_000, "snappy", "snappy")
    run_benchmark("small", 100_000, "zstd", "zstd")

    # Medium dataset
    run_benchmark("medium", 1_000_000, None, "none")
    run_benchmark("medium", 1_000_000, "snappy", "snappy")
    run_benchmark("medium", 1_000_000, "zstd", "zstd")

    # Large dataset
    run_benchmark("large", 10_000_000, None, "none")
    run_benchmark("large", 10_000_000, "snappy", "snappy")
    run_benchmark("large", 10_000_000, "zstd", "zstd")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
