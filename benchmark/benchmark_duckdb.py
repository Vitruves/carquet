#!/usr/bin/env python3
"""
Benchmark for DuckDB's native Parquet reader / writer - compare with Carquet.

DuckDB reads and writes Parquet with its own C++ implementation (the `parquet`
extension), not through Arrow C++. The data sits in a DuckDB in-memory table
built outside the timed loops; a write is `COPY ... TO ... (FORMAT parquet)`,
a read is `SELECT * FROM read_parquet(...)` fetched as an Arrow table (DuckDB's
cheapest full materialization).

Same data, iteration counts, trimmed median and cache purge as
benchmark_carquet.c / benchmark_pyarrow.py. Settings are aligned where DuckDB
allows it (same row group size, zstd level, LZ4_RAW for "lz4"); DuckDB picks
its encodings itself (it dictionary-encodes `id` and `category` on this data)
and has no memory-map switch, so no read_nommap figure is reported.

"<size>_nullable" datasets carry the shared null pattern (bench_common.null_masks).
"""

import os
import time

import duckdb
import pyarrow as pa

from bench_common import (
    WARMUP_ITERATIONS,
    generate_columns,
    get_benchmark_zstd_level,
    get_iterations,
    get_temp_dir,
    is_nullable_dataset,
    null_masks,
    purge_file_cache,
    trimmed_median,
)

CODECS = {"none": "uncompressed", "snappy": "snappy", "zstd": "zstd", "lz4": "lz4_raw"}


def load_table(con, num_rows, nullable=False):
    """Fill the in-memory table `bench` with the shared benchmark data."""
    columns = generate_columns(num_rows)
    masks = null_masks(num_rows) if nullable else {}
    source = pa.table({
        name: pa.array(values, mask=masks.get(name))
        for name, values in columns.items()
    })
    con.register("bench_source", source)
    con.execute("CREATE OR REPLACE TABLE bench AS SELECT * FROM bench_source")
    con.unregister("bench_source")


def expected_null_counts(num_rows):
    return {name: int(mask.sum()) for name, mask in null_masks(num_rows).items()}


def benchmark_write(con, filename, num_rows, compression_name):
    """Write a Parquet file and return time_ms."""
    rg_size = num_rows // 10 if num_rows > 1_000_000 else 100_000
    options = f"FORMAT parquet, COMPRESSION {CODECS[compression_name]}, ROW_GROUP_SIZE {rg_size}"
    if compression_name == "zstd":
        options += f", COMPRESSION_LEVEL {get_benchmark_zstd_level()}"
    start = time.perf_counter()
    con.execute(f"COPY bench TO '{filename}' ({options})")
    return (time.perf_counter() - start) * 1000


def fetch_arrow(result):
    if hasattr(result, "to_arrow_table"):
        return result.to_arrow_table()
    return result.fetch_arrow_table()


def benchmark_read(con, filename, expected_rows, expected_nulls=None):
    """Read a Parquet file and return time_ms."""
    start = time.perf_counter()
    table = fetch_arrow(con.execute(f"SELECT * FROM read_parquet('{filename}')"))

    nulls = None
    checksum = 0
    if expected_nulls is None:
        for name in table.column_names:
            checksum += table.column(name)[0].as_py()
    else:
        nulls = {name: table.column(name).null_count for name in table.column_names}

    elapsed_ms = (time.perf_counter() - start) * 1000

    if table.num_rows != expected_rows:
        raise ValueError(f"Row count mismatch: {table.num_rows} vs {expected_rows}")
    if expected_nulls is not None and nulls != expected_nulls:
        raise ValueError(f"Null count mismatch: {nulls} vs {expected_nulls}")

    return elapsed_ms


def run_benchmark(name, num_rows, compression_name):
    """Run a single benchmark configuration."""
    filename = os.path.join(
        get_temp_dir(),
        f"benchmark_{name}_{compression_name}_duckdb.parquet",
    )
    nullable = is_nullable_dataset(name)

    print(f"\n=== {name} ({num_rows:,} rows, {compression_name}) ===")

    con = duckdb.connect()
    load_table(con, num_rows, nullable)
    expected_nulls = expected_null_counts(num_rows) if nullable else None
    iters = get_iterations(name)

    for _ in range(WARMUP_ITERATIONS):
        benchmark_write(con, filename, num_rows, compression_name)

    write_times = []
    for _ in range(iters):
        write_times.append(benchmark_write(con, filename, num_rows, compression_name))

    file_size = os.path.getsize(filename)

    # Purge cache once, then warmup reads (first read = cold, rest = warm)
    purge_file_cache(filename)

    for _ in range(WARMUP_ITERATIONS):
        benchmark_read(con, filename, num_rows, expected_nulls)

    read_times = []
    for _ in range(iters):
        read_times.append(benchmark_read(con, filename, num_rows, expected_nulls))

    write_med = trimmed_median(write_times)
    read_med = trimmed_median(read_times)
    rows_per_sec_write = (num_rows / write_med) * 1000
    rows_per_sec_read = (num_rows / read_med) * 1000

    print(f"  Write: {write_med:.2f} ms ({rows_per_sec_write / 1e6:.2f} M rows/sec)")
    print(f"  Read:  {read_med:.2f} ms ({rows_per_sec_read / 1e6:.2f} M rows/sec)")
    print(f"  File:  {file_size / (1024 * 1024):.2f} MB ({file_size / num_rows:.2f} bytes/row)")

    print(f"CSV:duckdb,{name},{compression_name},{num_rows},"
          f"{write_med:.2f},{read_med:.2f},{file_size}")

    con.close()
    os.remove(filename)


def main():
    print("DuckDB Benchmark")
    print(f"duckdb version: {duckdb.__version__}")

    for name, rows in [("small", 100_000), ("medium", 1_000_000), ("large", 10_000_000),
                       ("xlarge", 100_000_000)]:
        for comp_name in CODECS:
            run_benchmark(name, rows, comp_name)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
