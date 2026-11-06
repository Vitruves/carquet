#!/usr/bin/env python3
"""
Benchmark for fastparquet - compare with Carquet.

Same data, iteration counts, trimmed median and cache purge as
benchmark_carquet.c / benchmark_pyarrow.py. Settings are aligned where
fastparquet allows it (no dictionary, same row group size, zstd level, LZ4_RAW
for "lz4"); fastparquet cannot write BYTE_STREAM_SPLIT or page checksums and
reads through pandas, so its files and read path differ from the others.
"""

import os
import time

import fastparquet
import pandas as pd

from bench_common import (
    WARMUP_ITERATIONS,
    generate_columns,
    get_benchmark_zstd_level,
    get_iterations,
    get_temp_dir,
    purge_file_cache,
    trimmed_median,
)

# Carquet's "lz4" is LZ4_RAW; fastparquet's legacy "LZ4" id is not.
CODECS = {"none": None, "snappy": "SNAPPY", "zstd": "ZSTD", "lz4": "LZ4_RAW"}


def generate_data(num_rows):
    """Pre-generate test data using the same RNG sequence as benchmark_carquet.c."""
    return pd.DataFrame(generate_columns(num_rows))


def write_compression(compression_name):
    codec = CODECS[compression_name]
    if codec == "ZSTD":
        return {"_default": {"type": "ZSTD", "args": {"level": get_benchmark_zstd_level()}}}
    return codec


def benchmark_write(filename, df, compression):
    """Write a Parquet file and return time_ms."""
    start = time.perf_counter()
    num_rows = len(df)
    rg_size = num_rows // 10 if num_rows > 1_000_000 else 100_000
    fastparquet.write(
        filename,
        df,
        row_group_offsets=rg_size,
        compression=compression,
        write_index=False,
        has_nulls=False,
    )
    return (time.perf_counter() - start) * 1000


def benchmark_read(filename, expected_rows):
    """Read a Parquet file and return time_ms."""
    start = time.perf_counter()
    df = fastparquet.ParquetFile(filename).to_pandas()

    checksum = 0
    for col_name in df.columns:
        checksum += df[col_name].to_numpy()[0]

    elapsed_ms = (time.perf_counter() - start) * 1000

    if len(df) != expected_rows:
        raise ValueError(f"Row count mismatch: {len(df)} vs {expected_rows}")

    return elapsed_ms


def run_benchmark(name, num_rows, compression_name):
    """Run a single benchmark configuration."""
    filename = os.path.join(
        get_temp_dir(),
        f"benchmark_{name}_{compression_name}_fastparquet.parquet",
    )
    compression = write_compression(compression_name)

    print(f"\n=== {name} ({num_rows:,} rows, {compression_name}) ===")

    df = generate_data(num_rows)
    iters = get_iterations(name)

    for _ in range(WARMUP_ITERATIONS):
        benchmark_write(filename, df, compression)

    write_times = []
    for _ in range(iters):
        write_times.append(benchmark_write(filename, df, compression))

    file_size = os.path.getsize(filename)

    # Purge cache once, then warmup reads (first read = cold, rest = warm)
    purge_file_cache(filename)

    for _ in range(WARMUP_ITERATIONS):
        benchmark_read(filename, num_rows)

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

    print(f"CSV:fastparquet,{name},{compression_name},{num_rows},"
          f"{write_med:.2f},{read_med:.2f},{file_size}")

    os.remove(filename)


def main():
    print("fastparquet Benchmark")
    print(f"fastparquet version: {fastparquet.__version__}")

    for name, rows in [("small", 100_000), ("medium", 1_000_000), ("large", 10_000_000),
                       ("xlarge", 100_000_000)]:
        for comp_name in CODECS:
            run_benchmark(name, rows, comp_name)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
