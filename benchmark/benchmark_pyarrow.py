#!/usr/bin/env python3
"""
Benchmark for PyArrow Parquet - compare with Carquet.

Methodology mirrors benchmark_carquet.c:
  - Write and read benchmarked separately
  - Page cache purged between write and read phases
  - 3 warmup + 11 measured iterations, trimmed median (drop min + max)
  - Data pre-generated outside timing loops
  - Arrow write/read options aligned to the Carquet benchmark where supported
"""

import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

from bench_common import (
    WARMUP_ITERATIONS,
    generate_columns,
    get_benchmark_zstd_level,
    get_iterations,
    get_temp_dir,
    purge_file_cache,
    trimmed_median,
)


def generate_data(num_rows):
    """Pre-generate test data using the same RNG sequence as benchmark_carquet.c."""
    return pa.table(generate_columns(num_rows))


def benchmark_write(filename, table, compression):
    """Write a Parquet file and return time_ms."""
    start = time.perf_counter()
    num_rows = len(table)
    rg_size = num_rows // 10 if num_rows > 1_000_000 else 100_000
    kwargs = {
        "compression": compression,
        "row_group_size": rg_size,
        "use_dictionary": False,
        "store_schema": False,
        "write_page_checksum": True,
    }
    if compression is not None:
        kwargs["use_byte_stream_split"] = ["value"]
    if compression == "zstd":
        kwargs["compression_level"] = get_benchmark_zstd_level()
    pq.write_table(table, filename, **kwargs)
    return (time.perf_counter() - start) * 1000


def benchmark_read(filename, expected_rows):
    """Read a Parquet file and return time_ms."""
    start = time.perf_counter()
    table = pq.read_table(
        filename,
        memory_map=True,
        use_threads=True,
        page_checksum_verification=False,
    )

    checksum = 0
    for col_name in table.column_names:
        arr = table[col_name].to_numpy()
        checksum += arr[0]

    elapsed_ms = (time.perf_counter() - start) * 1000

    if len(table) != expected_rows:
        raise ValueError(f"Row count mismatch: {len(table)} vs {expected_rows}")

    return elapsed_ms


def run_benchmark(name, num_rows, compression, compression_name):
    """Run a single benchmark configuration."""
    filename = os.path.join(
        get_temp_dir(),
        f"benchmark_{name}_{compression_name}_pyarrow.parquet",
    )

    print(f"\n=== {name} ({num_rows:,} rows, {compression_name}) ===")

    table = generate_data(num_rows)

    iters = get_iterations(name)

    # Warmup writes
    for _ in range(WARMUP_ITERATIONS):
        benchmark_write(filename, table, compression)

    # Benchmark writes
    write_times = []
    for _ in range(iters):
        write_times.append(benchmark_write(filename, table, compression))

    file_size = os.path.getsize(filename)

    # Purge cache once, then warmup reads (first read = cold, rest = warm)
    purge_file_cache(filename)

    for _ in range(WARMUP_ITERATIONS):
        benchmark_read(filename, num_rows)

    # Benchmark reads (warm cache — realistic for most workloads)
    read_times = []
    for _ in range(iters):
        read_times.append(benchmark_read(filename, num_rows))

    write_med = trimmed_median(write_times)
    read_med = trimmed_median(read_times)
    rows_per_sec_write = (num_rows / write_med) * 1000
    rows_per_sec_read = (num_rows / read_med) * 1000

    print(f"  Write: {write_med:.2f} ms ({rows_per_sec_write / 1e6:.2f} M rows/sec)")
    print(f"  Read:  {read_med:.2f} ms ({rows_per_sec_read / 1e6:.2f} M rows/sec)")
    print(f"  File:  {file_size / (1024 * 1024):.2f} MB ({file_size / num_rows:.2f} bytes/row)")

    print(f"CSV:pyarrow,{name},{compression_name},{num_rows},"
          f"{write_med:.2f},{read_med:.2f},{file_size}")

    os.remove(filename)


def main():
    print("PyArrow Benchmark")
    print(f"PyArrow version: {pa.__version__}")

    for name, rows in [("small", 100_000), ("medium", 1_000_000), ("large", 10_000_000),
                        ("xlarge", 100_000_000)]:
        for comp_name, comp in [("none", None), ("snappy", "snappy"), ("zstd", "zstd"), ("lz4", "lz4")]:
            run_benchmark(name, rows, comp, comp_name)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
