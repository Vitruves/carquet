<p align="center">
  <img src="res/img/carquet_logo.png" alt="Carquet" width="280" />
</p>

<h1 align="center">Carquet</h1>

<p align="center">
  A fast, pure C library for reading and writing Apache Parquet files.
</p>

<p align="center">
  <a href="https://github.com/Vitruves/carquet/actions/workflows/cpp.yml"><img src="https://github.com/Vitruves/carquet/actions/workflows/cpp.yml/badge.svg" alt="Build" /></a>
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue" alt="Platform" />
  <img src="https://img.shields.io/badge/C-C11-blue" alt="C Standard" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License" /></a>
  <br/>
  <img src="https://img.shields.io/badge/SIMD-SSE4.2%20%7C%20AVX%20%7C%20AVX2%20%7C%20AVX--512-red" alt="x86 SIMD" />
  <img src="https://img.shields.io/badge/SIMD-NEON%20%7C%20SVE-orange" alt="ARM SIMD" />
</p>

<!-- ────────────────────────────  PRESENTATION  ──────────────────────────── -->

## Highlights

- **Pure C11** with three external dependencies (zstd, zlib, lz4) -- all auto-fetched by CMake
- **~200KB binary** vs ~50MB+ for Arrow
- **Built-in CLI** for file inspection (`schema`, `info`, `head`, `tail`, `stat`, ...) and C code generation (`codegen`)
- **70x faster reads** than Arrow C++ on uncompressed data (mmap zero-copy), **150x faster** than PyArrow
- **1.2-2.6x faster compressed reads** than Arrow C++ on the same file (cross-read benchmark)
- **Writes 1.0-2.3x faster** than Arrow C++ across codecs and platforms
- Reads 10M uncompressed rows in **0.25ms** (mmap zero-copy on Apple M3)
- Full Parquet spec: all types, encodings, compression codecs, nested schemas, bloom filters, page indexes
- SIMD-optimized (SSE4.2, AVX2, AVX-512, NEON, SVE) with runtime detection and scalar fallbacks
- PyArrow, DuckDB, Spark compatible out of the box

<!-- ────────────────────────────  BENCHMARKS  ──────────────────────────── -->

## Benchmarks

At 10M rows (the most representative size); higher ratio = Carquet faster. ARM (Apple M3): Carquet 0.6.0 vs Arrow C++ 24.0.0. x86 (Xeon D-1531): Carquet 0.4.4 vs Arrow C++ 23.0.1.

| | x86 (Xeon D-1531) | | ARM (Apple M3) | |
|---|---|---|---|---|
| **Codec** | **Write** | **Read** | **Write** | **Read** |
| snappy | **1.55x** | **1.25x** | **1.88x** | **1.59x** |
| zstd | **1.31x** | **1.04x** | **2.23x** | **1.30x** |
| lz4 | **1.02x** | 0.83x | **1.89x** | **1.59x** |
| none | **1.13x** | **40.6x**\* | **1.97x** | **52.9x**\* |

\* Uncompressed reads use mmap zero-copy -- see note below.

Compressed reads involve full decompression and decoding of every value, no shortcuts — and both libraries use the same system lz4/zstd shared libraries, so the raw codec speed is identical. The most meaningful comparison is the **same-file cross-read** table (below), where both libraries read the exact same Parquet file: Carquet reads compressed data **1.5-2.6x faster** than Arrow C++ on that apples-to-apples test.

To run the benchmarks yourself, see [Running Benchmarks](#running-benchmarks).

<details>
<summary>Benchmark methodology</summary>

All benchmarks use identical data (deterministic LCG PRNG), identical Parquet settings (no dictionary, BYTE_STREAM_SPLIT for floats, page checksums, mmap reads), trimmed median of 11-51 iterations, with OS page cache purged between write and read phases and cooldown between configurations. Schema: 3 columns (INT64, DOUBLE, INT32). Compared against Arrow C++ 23.0.1 low-level Parquet reader (bypassing Arrow Table materialization) and PyArrow 23.0.1.

The **same-file cross-read** benchmark is the fairest comparison: both libraries read the exact same Parquet file (written by one, read by both). This eliminates differences in page sizes, encoding choices, and row group layout.

**Uncompressed reads** marked with \* use Carquet's **mmap zero-copy path**: for PLAIN-encoded, uncompressed, fixed-size, required columns, the batch reader returns pointers directly into the memory-mapped file with no memcpy. Arrow always materializes into its own buffers. **The compressed read numbers are the most representative measure of end-to-end read throughput.**

</details>

<details>
<summary>Full x86 results (Intel Xeon D-1531, Linux)</summary>

*12 threads @ 2.2GHz, 32GB RAM, Ubuntu 24.04 -- ZSTD level 1*

#### 10M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio | Size |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | **1557ms** | 1766ms | **1.13x** | **1.25ms** | 50.8ms | **40.6x**\* | 190.7MB |
| snappy | **1002ms** | 1549ms | **1.55x** | **78ms** | 97.8ms | **1.25x** | 125.1MB |
| zstd | **1311ms** | 1714ms | **1.31x** | **76.8ms** | 80.2ms | **1.04x** | 95.3MB |
| lz4 | **1521ms** | 1554ms | **1.02x** | 59.1ms | **49.0ms** | 0.83x | 122.9MB |

#### 1M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **180ms** | 196ms | **1.09x** | **0.22ms** | 6.2ms | **28x**\* |
| snappy | **141ms** | 148ms | **1.05x** | **8.1ms** | 11.6ms | **1.44x** |
| zstd | **131ms** | 185ms | **1.41x** | 10.3ms | **9.1ms** | 0.88x |
| lz4 | **143ms** | 149ms | **1.04x** | 8.5ms | **6.1ms** | 0.72x |

#### 100K rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **14.1ms** | 18.4ms | **1.30x** | **0.11ms** | 2.18ms | **19.8x**\* |
| snappy | **10.1ms** | 10.6ms | **1.05x** | **1.27ms** | 5.97ms | **4.70x** |
| zstd | **8.7ms** | 14.1ms | **1.62x** | **1.58ms** | 3.88ms | **2.46x** |
| lz4 | **9.6ms** | 11.0ms | **1.14x** | **0.77ms** | 2.78ms | **3.61x** |

#### Same-file cross-read (10M rows)

Both libraries read the **same** Parquet file — the fairest apples-to-apples comparison.

| Codec | Writer | Carquet Read | Arrow C++ Read | Ratio |
|-------|--------|-------------|----------------|-------|
| none | Carquet | **0.99ms** | 73.6ms | **74x**\* |
| none | Arrow | **7.6ms** | 51.2ms | **6.8x**\* |
| snappy | Carquet | **41.0ms** | 107ms | **2.61x** |
| snappy | Arrow | **43.4ms** | 101ms | **2.33x** |
| zstd | Carquet | **46.1ms** | 88.4ms | **1.92x** |
| zstd | Arrow | **49.1ms** | 79.5ms | **1.62x** |
| lz4 | Carquet | **34.8ms** | 74.8ms | **2.15x** |
| lz4 | Arrow | **27.4ms** | 52.0ms | **1.90x** |

#### 10M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **1557ms** | 1806ms | **1.16x** | **1.25ms** | 213ms | **170x**\* |
| snappy | **1002ms** | 1649ms | **1.65x** | **78ms** | 384ms | **4.91x** |
| zstd | **1311ms** | 1796ms | **1.37x** | **76.8ms** | 369ms | **4.81x** |
| lz4 | **1521ms** | 1676ms | **1.10x** | **59.1ms** | 281ms | **4.76x** |

\* Zero-copy mmap path

</details>

<details>
<summary>Full ARM results (Apple M3, macOS)</summary>

*Carquet 0.6.0 -- MacBook Air M3, 16GB RAM, macOS 26.5, Arrow C++ 24.0.0, PyArrow 23.0.1 -- ZSTD level 1*

#### 10M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio | Size |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | **62.21ms** | 122.5ms | **1.97x** | **0.26ms** | 13.75ms | **52.9x**\* | 190.7MB |
| snappy | **129.0ms** | 242.6ms | **1.88x** | **14.15ms** | 22.47ms | **1.59x** | 125.1MB |
| zstd | **149.6ms** | 334.0ms | **2.23x** | **21.39ms** | 27.91ms | **1.30x** | 95.3MB |
| lz4 | **128.1ms** | 241.7ms | **1.89x** | **10.13ms** | 16.08ms | **1.59x** | 122.9MB |

#### 1M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **5.85ms** | 12.65ms | **2.16x** | **0.05ms** | 1.54ms | **30.8x**\* |
| snappy | **12.39ms** | 23.62ms | **1.91x** | **1.36ms** | 2.41ms | **1.77x** |
| zstd | **15.26ms** | 33.90ms | **2.22x** | **2.39ms** | 3.17ms | **1.33x** |
| lz4 | **12.28ms** | 24.51ms | **2.00x** | **0.97ms** | 1.77ms | **1.82x** |

#### 100K rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **1.02ms** | 1.54ms | **1.51x** | **0.02ms** | 0.22ms | **11.0x**\* |
| snappy | **1.51ms** | 2.45ms | **1.62x** | **0.36ms** | 0.90ms | **2.50x** |
| zstd | **1.65ms** | 3.58ms | **2.17x** | **0.63ms** | 1.23ms | **1.95x** |
| lz4 | **1.57ms** | 2.45ms | **1.56x** | **0.24ms** | 0.55ms | **2.29x** |

#### Same-file cross-read (10M rows)

Both libraries read the **same** Parquet file — the fairest apples-to-apples comparison.

| Codec | Writer | Carquet Read | Arrow C++ Read | Ratio |
|-------|--------|-------------|----------------|-------|
| none | Carquet | **0.27ms** | 15.60ms | **57.8x**\* |
| none | Arrow | **0.91ms** | 13.68ms | **15.0x**\* |
| snappy | Carquet | **14.63ms** | 23.29ms | **1.59x** |
| snappy | Arrow | **13.20ms** | 22.15ms | **1.68x** |
| zstd | Carquet | **20.96ms** | 28.04ms | **1.34x** |
| zstd | Arrow | **21.03ms** | 27.73ms | **1.32x** |
| lz4 | Carquet | **10.16ms** | 16.89ms | **1.66x** |
| lz4 | Arrow | **9.66ms** | 16.09ms | **1.67x** |

#### 10M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **62.21ms** | 176.4ms | **2.83x** | **0.26ms** | 36.49ms | **140.4x**\* |
| snappy | **129.0ms** | 294.2ms | **2.28x** | **14.15ms** | 44.19ms | **3.12x** |
| zstd | **149.6ms** | 396.3ms | **2.65x** | **21.39ms** | 55.96ms | **2.62x** |
| lz4 | **128.1ms** | 305.3ms | **2.38x** | **10.13ms** | 38.00ms | **3.75x** |

#### 1M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **5.85ms** | 17.13ms | **2.93x** | **0.05ms** | 2.55ms | **51.0x**\* |
| snappy | **12.39ms** | 29.62ms | **2.39x** | **1.36ms** | 3.57ms | **2.62x** |
| zstd | **15.26ms** | 40.26ms | **2.64x** | **2.39ms** | 4.41ms | **1.85x** |
| lz4 | **12.28ms** | 31.15ms | **2.54x** | **0.97ms** | 3.11ms | **3.21x** |

#### 100K rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **1.02ms** | 1.94ms | **1.90x** | **0.02ms** | 0.23ms | **11.5x**\* |
| snappy | **1.51ms** | 2.96ms | **1.96x** | **0.36ms** | 0.58ms | **1.61x** |
| zstd | **1.65ms** | 4.15ms | **2.52x** | **0.63ms** | 0.80ms | **1.27x** |
| lz4 | **1.57ms** | 3.01ms | **1.92x** | **0.24ms** | 0.41ms | **1.71x** |

\* Zero-copy mmap path

</details>

<!-- ────────────────────────────  INSTALLATION  ──────────────────────────── -->

## Installation

### Requirements

- C11 compiler (GCC 4.9+, Clang 3.4+, MSVC 2015+)
- CMake 3.16+ (or [xmake](https://xmake.io) — see [Building with xmake](#building-with-xmake))
- zstd, zlib, lz4 (auto-fetched if missing)
- OpenMP (optional, for parallel column reading)

### Quick Start (make)

The `make` wrapper drives an optimized CMake build and a `/usr/local` install:

```bash
git clone https://github.com/Vitruves/carquet.git
cd carquet
make                              # optimized build (run `make help` to list all targets)
sudo make install                 # install to /usr/local (override with PREFIX=/opt/carquet)
```

### Full Build & Install (CMake)

Invoke CMake directly when you need specific build options or a custom prefix:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # add options, e.g. -DCARQUET_BUILD_SHARED=ON
cmake --build build -j$(nproc)
sudo cmake --install build --prefix /usr/local
```

Either path installs:
- `libcarquet.a` (or `.so` / `.dylib` with `-DCARQUET_BUILD_SHARED=ON`)
- `include/carquet/` headers
- `carquet` CLI binary
- `carquet.pc` (pkg-config) and CMake package config for `find_package(carquet)`

After installation, link with `-lcarquet`, or resolve flags via `pkg-config --cflags --libs carquet`.

#### Build Options

Pass these to the CMake configure step:

| Option | Default | Description |
|--------|---------|-------------|
| `CARQUET_BUILD_DEV` | OFF | Build everything (tests, examples, benchmarks) |
| `CARQUET_BUILD_TESTS` | OFF | Build test suite only |
| `CARQUET_BUILD_CLI` | ON | Build `carquet` CLI tool |
| `CARQUET_BUILD_SHARED` | OFF | Build shared library instead of static |
| `CARQUET_NATIVE_ARCH` | OFF | `-march=native` for max performance |
| `CARQUET_ENABLE_SVE` | OFF | ARM SVE (experimental) |

All x86 SIMD (SSE, AVX, AVX2, AVX-512) and ARM NEON are auto-detected and enabled by default.

<details>
<summary>All build options</summary>

| Option | Default | Description |
|--------|---------|-------------|
| `CARQUET_BUILD_EXAMPLES` | OFF | Build example programs |
| `CARQUET_BUILD_BENCHMARKS` | OFF | Build benchmark and profiling programs |
| `CARQUET_BUILD_ARROW_CPP_BENCHMARK` | OFF | Optional Arrow C++ comparison benchmark |
| `CARQUET_BUILD_INTEROP` | OFF | Build interoperability tests |
| `CARQUET_BUILD_FUZZ` | OFF | Build fuzz targets |
| `CARQUET_ENABLE_SSE` | ON | SSE optimizations (x86, auto-detected) |
| `CARQUET_ENABLE_AVX` | ON | AVX optimizations (x86, auto-detected) |
| `CARQUET_ENABLE_AVX2` | ON | AVX2 optimizations (x86, auto-detected) |
| `CARQUET_ENABLE_AVX512` | ON | AVX-512 optimizations (x86, auto-detected) |
| `CARQUET_ENABLE_NEON` | ON | NEON optimizations (ARM, auto-detected) |

</details>

#### Development Build

```bash
cmake -B build -DCARQUET_BUILD_DEV=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

### Building with xmake

An [xmake](https://xmake.io) build (`xmake.lua`) is provided as an alternative to CMake, with the same options and defaults; zstd/zlib/lz4 are linked statically so binaries are self-contained.

```bash
xmake                    # build the static library + `carquet` CLI (release)
xmake f --dev=y && xmake # add tests, examples, benchmarks, interop
xmake test               # run the test suite
```

<details>
<summary>Configure options (<code>xmake f --option=y|n</code>)</summary>

Options mirror the CMake ones (drop the `CARQUET_` prefix, lower-case):

| xmake option | Default | Description |
|--------------|---------|-------------|
| `--dev` | n | Build tests, examples, benchmarks and interop |
| `--tests` / `--examples` / `--benchmarks` / `--interop` | n | Build one group individually |
| `--cli` | y | Build the `carquet` CLI tool |
| `--shared` | n | Build a shared library instead of static |
| `--openmp` | y | OpenMP parallel column reading (auto-disabled if unavailable) |
| `--native_arch` | n | `-march=native` for max performance (host-only binary) |
| `--sse` / `--avx` / `--avx2` / `--avx512` / `--neon` | y | SIMD instruction sets (auto-detected) |
| `--sve` | n | ARM SVE (experimental) |
| `--fuzz` | n | Build fuzz targets (use `--toolchain=clang`) |

```bash
xmake f -m release --shared=y            # shared library
xmake f --dev=y --avx512=n && xmake      # dev build, AVX-512 disabled
```

</details>

<!-- ────────────────────────────  C API  ──────────────────────────── -->

## C API

This README stays intentionally short — a Write and a Read example below, then the [manual in `docs/`](docs/README.md) for everything else.

### Write a Parquet File

```c
#include <carquet/carquet.h>

int main(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    // Define schema
    carquet_schema_t* schema = carquet_schema_create(&err);
    carquet_schema_add_column(schema, "id",    CARQUET_PHYSICAL_INT64,  NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);

    // Configure writer
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_ZSTD;

    // Write
    carquet_writer_t* w = carquet_writer_create("output.parquet", schema, &opts, &err);

    int64_t ids[]    = {1, 2, 3, 4, 5};
    double values[]  = {1.1, 2.2, 3.3, 4.4, 5.5};
    carquet_writer_write_batch(w, 0, ids, 5, NULL, NULL);
    carquet_writer_write_batch(w, 1, values, 5, NULL, NULL);
    carquet_writer_close(w);

    carquet_schema_free(schema);
    return 0;
}
```

### Read a Parquet File

```c
#include <carquet/carquet.h>
#include <stdio.h>

int main(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    // Open with mmap for best read performance
    carquet_reader_options_t opts;
    carquet_reader_options_init(&opts);
    opts.use_mmap = true;

    carquet_reader_t* r = carquet_reader_open("output.parquet", &opts, &err);
    if (!r) { printf("Error: %s\n", err.message); return 1; }

    printf("Rows: %lld, Columns: %d\n",
           (long long)carquet_reader_num_rows(r),
           carquet_reader_num_columns(r));

    // Batch reader for efficient iteration
    carquet_batch_reader_config_t cfg;
    carquet_batch_reader_config_init(&cfg);
    cfg.batch_size = 65536;

    carquet_batch_reader_t* br = carquet_batch_reader_create(r, &cfg, &err);
    carquet_row_batch_t* batch = NULL;

    while (carquet_batch_reader_next(br, &batch) == CARQUET_OK && batch) {
        const void* data;
        const uint8_t* nulls;
        int64_t n;
        carquet_row_batch_column(batch, 0, &data, &nulls, &n);
        const int64_t* ids = (const int64_t*)data;
        // process ids[0..n-1] ...
        carquet_row_batch_free(batch);
        batch = NULL;
    }

    carquet_batch_reader_free(br);
    carquet_reader_close(r);
    return 0;
}
```

### More Recipes

Everything beyond flat read/write lives in the manual — each links to a runnable example:

| You want to… | See |
|---|---|
| Nullable columns, row groups, buffer output | [`docs/writing.md`](docs/writing.md) |
| Lists, maps, groups, definition/repetition levels | [`docs/nested-data.md`](docs/nested-data.md) |
| Column projection, statistics, metadata inspection | [`docs/reading.md`](docs/reading.md) |
| Predicate pushdown, page-level filtering | [`docs/reading.md`](docs/reading.md) |
| Append row groups to an existing file | [`docs/writing.md`](docs/writing.md#append-to-an-existing-file) |
| Compression, custom codecs, writer tuning | [`docs/writing.md`](docs/writing.md), [`docs/performance.md`](docs/performance.md) |
| mmap, zero-copy, prebuffering, I/O coalescing | [`docs/performance.md`](docs/performance.md) |
| Error codes and recovery hints | [`docs/error-handling.md`](docs/error-handling.md) |

### API Reference

Full API is in [`include/carquet/carquet.h`](include/carquet/carquet.h). Key types:

| Type | Purpose |
|------|---------|
| `carquet_reader_t` | File reader (open from path, FILE*, or memory buffer) |
| `carquet_writer_t` | File writer |
| `carquet_batch_reader_t` | High-level batch iteration |
| `carquet_schema_t` | Schema definition and introspection |
| `carquet_error_t` | Rich error info (code, message, source location, recovery hint) |

Full signatures live in the [header](include/carquet/carquet.h); the [manual](docs/README.md) explains which surface to use when. The source layout and architecture are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

<!-- ────────────────────────────  CLI TOOL  ──────────────────────────── -->

## CLI Tool

Carquet ships with a command-line tool for inspecting Parquet files and generating C reader code. Built and installed by default alongside the library.

```
Commands:
  schema     Print file schema
  info       Print detailed file metadata
  head       Print first N rows
  tail       Print last N rows
  cat        Print rows with slicing/column/row filtering
  count      Print total row count
  columns    List column names (one per line)
  stat       Print column statistics
  validate   Verify file integrity
  sample     Print N random rows
  export     Write rows to stdout as CSV
  codegen    Generate C reader code
```

```bash
carquet schema data.parquet
carquet head -n 20 data.parquet
carquet stat data.parquet
carquet validate data.parquet
```

`cat`, `count`, `head`, and `export` accept `-p / --filter EXPR` to push a row predicate down to the page level — only pages whose column-index min/max can match the predicate are decompressed:

```bash
carquet cat -p "price > 100 AND status = 'active'" data.parquet
carquet count --filter "id >= 1000" data.parquet
carquet export --filter "ts IS NOT NULL" -c id,ts data.parquet
```

The grammar is `column OP value [AND column OP value]...` with `OP` ∈ {`=`, `==`, `!=`, `<>`, `<`, `<=`, `>`, `>=`}, plus `column IS NULL` / `column IS NOT NULL`. Filtering requires the file to have a page index (`write_page_index = true`).

### Code Generation

Generate a complete, compilable C reader from any Parquet file's schema:

```bash
carquet codegen -f data.parquet -o reader.c
# Generated: reader.c
# Compile:   clang -o reader reader.c -I.../include -L.../build -lcarquet ...

./reader                    # reads data.parquet (embedded as default)
./reader other.parquet      # override with different file
```

Options:

| Flag | Description |
|------|-------------|
| `-f`, `--file FILE` | Parquet file to inspect schema from |
| `-o`, `--output FILE` | Output source file (default: stdout) |
| `--mmap` | Use memory-mapped I/O in generated code |
| `--skeleton` | Generate empty `process_batch` for custom logic |
| `-c`, `--columns COLS` | Comma-separated column filter |
| `-b`, `--batch-size N` | Batch size (default: 1024) |

<!-- ────────────────────────────  INTEROP & REFERENCE  ──────────────────────────── -->

## Interoperability

Carquet files are fully compatible with PyArrow, DuckDB, Spark, and any Parquet reader:

```python
import pyarrow.parquet as pq
table = pq.read_table("carquet_output.parquet")  # just works
```

```sql
-- DuckDB
SELECT * FROM read_parquet('carquet_output.parquet');
```

Bidirectional interop testing:

```bash
cmake -B build -DCARQUET_BUILD_INTEROP=ON && cmake --build build
python3 interop/run_interop.py
```

## Parquet Feature Support

| Feature | Status |
|---------|--------|
| Physical types | All 8 (BOOLEAN through FIXED_LEN_BYTE_ARRAY) |
| Logical types | STRING, DATE, TIME, TIMESTAMP, DECIMAL, UUID, JSON, INTERVAL, FLOAT16, VARIANT, GEOMETRY, GEOGRAPHY |
| Encodings | PLAIN, RLE, DICTIONARY, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY, BYTE_STREAM_SPLIT (read + write) |
| Data Page versions | V1 (default) and V2 (read + write) |
| Compression | UNCOMPRESSED, SNAPPY, GZIP, LZ4, ZSTD |
| Nested schemas | Groups, lists, maps with definition/repetition levels |
| Bloom filters | Read, write, and query (`carquet_bloom_filter_check_*`) |
| Page indexes | Column index + offset index (read + write + per-page stats access) |
| Statistics | Min/max/null count per column chunk |
| Predicate pushdown | Row group filtering via statistics; page-level filtering via column index (`carquet_batch_reader_set_page_filter`) |
| Append | Add row groups to an existing file (`carquet_writer_open_append`) |
| Custom codecs | Register a custom compress/decompress impl per codec slot (`carquet_register_codec`) |
| Key-value metadata | Read and write arbitrary footer metadata |
| Per-column options | Per-column encoding, compression, statistics, bloom filter |
| Buffer writer | Write Parquet to in-memory buffer |
| CRC32 | Page-level verification (HW-accelerated on ARM) |
| Memory-mapped I/O | Zero-copy reads for uncompressed PLAIN data |
| Column projection | Read only selected columns |
| I/O coalescing | Pre-buffer multi-column reads in a single I/O |
| Speculative footer | Single-I/O file open for most files |
| OpenMP parallel reads | When available |
| Encryption | Not supported |

## Running Benchmarks

```bash
# Build with max optimizations
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCARQUET_NATIVE_ARCH=ON -DCARQUET_BUILD_DEV=ON
cmake --build build -j$(nproc)

cd build
./benchmark_carquet                     # Carquet standalone
python3 ../benchmark/run_benchmark.py   # Full comparison (+ PyArrow, + Arrow C++)

# Skip 100M-row (xlarge) configs — they write ~2GB files per codec
# and can take 30+ minutes depending on hardware
python3 ../benchmark/run_benchmark.py --skip-xlarge

# Override ZSTD level (default: 1)
CARQUET_BENCH_ZSTD_LEVEL=3 python3 ../benchmark/run_benchmark.py
```

<details>
<summary>Optional Arrow C++ benchmark</summary>

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCARQUET_NATIVE_ARCH=ON \
  -DCARQUET_BUILD_BENCHMARKS=ON \
  -DCARQUET_BUILD_ARROW_CPP_BENCHMARK=ON
cmake --build build -j$(nproc)

# Or point at a custom Arrow install
cmake -B build ... -DCARQUET_ARROW_CPP_ROOT=/path/to/arrow-prefix
```

The Arrow C++ benchmark uses the low-level `parquet::ParquetFileReader` API (bypassing Arrow Table materialization overhead) with parallel row group readers. The **same-file cross-read** mode has both libraries read the exact same Parquet file, eliminating differences in page sizes, encoding, and row group layout. Both benchmarks use identical data, row group sizing, no dictionary, page checksums, mmap reads, BYTE_STREAM_SPLIT for floats.

</details>

## License

MIT
