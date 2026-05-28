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

## Performance

Carquet v0.4.4 vs Arrow C++ 23.0.1 at 10M rows (the most representative size). Higher ratio = Carquet faster.

| | x86 (Xeon D-1531) | | ARM (Apple M3) | |
|---|---|---|---|---|
| **Codec** | **Write** | **Read** | **Write** | **Read** |
| snappy | **1.55x** | **1.25x** | **1.10x** | **1.69x** |
| zstd | **1.31x** | **1.04x** | **1.70x** | **1.29x** |
| lz4 | **1.02x** | 0.83x | **1.21x** | **1.38x** |
| none | **1.13x** | **40.6x**\* | **2.31x** | **58.4x**\* |

\* Uncompressed reads use mmap zero-copy -- see note below.

Compressed reads involve full decompression and decoding of every value, no shortcuts — and both libraries use the same system lz4/zstd shared libraries, so the raw codec speed is identical. The most meaningful comparison is the **same-file cross-read** table (below), where both libraries read the exact same Parquet file: Carquet reads compressed data **1.5-2.6x faster** than Arrow C++ on that apples-to-apples test.

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

*MacBook Air M3, 16GB RAM, macOS 26.2, Arrow C++ 23.0.1, PyArrow 23.0.1 -- ZSTD level 1*

#### 10M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio | Size |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | **71.71ms** | 165.9ms | **2.31x** | **0.29ms** | 16.94ms | **58.4x**\* | 190.7MB |
| snappy | **223.1ms** | 244.6ms | **1.10x** | **15.00ms** | 25.39ms | **1.69x** | 125.1MB |
| zstd | **239.9ms** | 406.9ms | **1.70x** | **21.99ms** | 28.44ms | **1.29x** | 95.3MB |
| lz4 | **201.4ms** | 243.8ms | **1.21x** | **12.08ms** | 16.65ms | **1.38x** | 122.9MB |

#### 1M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **8.31ms** | 15.64ms | **1.88x** | **0.07ms** | 1.87ms | **26.7x**\* |
| snappy | **15.06ms** | 24.96ms | **1.66x** | **2.44ms** | 2.76ms | **1.13x** |
| zstd | **17.24ms** | 34.31ms | **1.99x** | 3.40ms | **3.12ms** | 0.92x |
| lz4 | **15.92ms** | 32.32ms | **2.03x** | **1.55ms** | 3.32ms | **2.14x** |

#### 100K rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **1.00ms** | 1.63ms | **1.63x** | **0.02ms** | 0.23ms | **11.5x**\* |
| snappy | **1.51ms** | 2.62ms | **1.74x** | **0.36ms** | 0.90ms | **2.50x** |
| zstd | **2.39ms** | 4.86ms | **2.03x** | **0.74ms** | 1.38ms | **1.86x** |
| lz4 | **1.55ms** | 3.68ms | **2.37x** | **0.30ms** | 0.71ms | **2.37x** |

#### Same-file cross-read (10M rows)

Both libraries read the **same** Parquet file — the fairest apples-to-apples comparison.

| Codec | Writer | Carquet Read | Arrow C++ Read | Ratio |
|-------|--------|-------------|----------------|-------|
| none | Carquet | **0.28ms** | 17.62ms | **62.9x**\* |
| none | Arrow | **0.97ms** | 16.86ms | **17.4x**\* |
| snappy | Carquet | **15.66ms** | 24.19ms | **1.54x** |
| snappy | Arrow | **14.71ms** | 22.82ms | **1.55x** |
| zstd | Carquet | **22.14ms** | 29.30ms | **1.32x** |
| zstd | Arrow | **22.49ms** | 32.10ms | **1.43x** |
| lz4 | Carquet | **11.53ms** | 18.65ms | **1.62x** |
| lz4 | Arrow | **10.17ms** | 17.38ms | **1.71x** |

#### 10M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **71.71ms** | 183.4ms | **2.56x** | **0.29ms** | 36.99ms | **127.6x**\* |
| snappy | **223.1ms** | 303.1ms | **1.36x** | **15.00ms** | 48.68ms | **3.25x** |
| zstd | **239.9ms** | 406.7ms | **1.70x** | **21.99ms** | 57.71ms | **2.62x** |
| lz4 | **201.4ms** | 338.2ms | **1.68x** | **12.08ms** | 48.68ms | **4.03x** |

#### 1M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **8.31ms** | 18.53ms | **2.23x** | **0.07ms** | 2.67ms | **38.1x**\* |
| snappy | **15.06ms** | 30.54ms | **2.03x** | **2.44ms** | 3.75ms | **1.54x** |
| zstd | **17.24ms** | 40.43ms | **2.35x** | **3.40ms** | 4.48ms | **1.32x** |
| lz4 | **15.92ms** | 40.69ms | **2.56x** | **1.55ms** | 5.83ms | **3.76x** |

#### 100K rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **1.00ms** | 2.81ms | **2.81x** | **0.02ms** | 0.25ms | **12.5x**\* |
| snappy | **1.51ms** | 4.46ms | **2.95x** | **0.36ms** | 0.63ms | **1.75x** |
| zstd | **2.39ms** | 5.59ms | **2.34x** | **0.74ms** | 0.86ms | **1.16x** |
| lz4 | **1.55ms** | 4.33ms | **2.79x** | **0.30ms** | 0.45ms | **1.50x** |

\* Zero-copy mmap path

</details>

## Building

### Requirements

- C11 compiler (GCC 4.9+, Clang 3.4+, MSVC 2015+)
- CMake 3.16+
- zstd, zlib, lz4 (auto-fetched if missing)
- OpenMP (optional, for parallel column reading)

### Quick Start

```bash
git clone https://github.com/Vitruves/carquet.git
cd carquet
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Build Options

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

### Installation

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --install build
```

This installs:
- `libcarquet.a` (or `.so` / `.dylib` with `-DCARQUET_BUILD_SHARED=ON`)
- `include/carquet/` headers
- `carquet` CLI binary

After installation, link your project with `-lcarquet`.

You can use the CLI directly if you want to create a file reader:

```bash
carquet info data.parquet
carquet codegen -f data.parquet -o reader.c
```

### Development Build

```bash
cmake -B build -DCARQUET_BUILD_DEV=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

## CLI Tool

Carquet ships with a command-line tool for inspecting Parquet files and generating C reader code. Built and installed by default alongside the library.

```
Commands:
  schema     Print file schema
  info       Print detailed file metadata
  head       Print first N rows
  tail       Print last N rows
  count      Print total row count
  columns    List column names (one per line)
  stat       Print column statistics
  validate   Verify file integrity
  sample     Print N random rows
  codegen    Generate C reader code
```

```bash
carquet schema data.parquet
carquet head -n 20 data.parquet
carquet stat data.parquet
carquet validate data.parquet
```

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

## C API

### Manual

The top-level README is intentionally short. For day-to-day usage, prefer the versioned manual in [`docs/`](docs/README.md):

- [Manual index](docs/README.md)
- [Reading files](docs/reading.md)
- [Writing files](docs/writing.md)
- [Nested and nullable data](docs/nested-data.md)
- [Performance and tuning](docs/performance.md)
- [Error handling and type reference](docs/error-handling.md)

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

### Nullable Columns

```c
// Schema with nullable column
carquet_schema_add_column(schema, "name", CARQUET_PHYSICAL_BYTE_ARRAY,
                          NULL, CARQUET_REPETITION_OPTIONAL, 0, 0);

// Write with definition levels (1 = present, 0 = null)
carquet_byte_array_t names[] = {{(uint8_t*)"Alice", 5}, {(uint8_t*)"Bob", 3}};
int16_t def_levels[] = {1, 0, 1};  // Alice, NULL, Bob (3 rows, 2 values)
carquet_writer_write_batch(writer, col, names, 3, def_levels, NULL);
```

### Nested Types (Lists, Maps)

```c
// list<int32>
int32_t list_leaf = carquet_schema_add_list(
    schema, "tags", CARQUET_PHYSICAL_INT32, NULL,
    CARQUET_REPETITION_OPTIONAL, 0, 0);

// map<string, int32>
int32_t map_val = carquet_schema_add_map(
    schema, "props",
    CARQUET_PHYSICAL_BYTE_ARRAY, NULL, 0,   // key: string
    CARQUET_PHYSICAL_INT32, NULL, 0,         // value: int32
    CARQUET_REPETITION_OPTIONAL, 0);

// Write list data: row0=[100,200], row1=NULL, row2=[300]
int32_t vals[] = {100, 200, 300};
int16_t def[]  = {  3,   3,   0,   3};
int16_t rep[]  = {  0,   1,   0,   0};
carquet_writer_write_batch(writer, col, vals, 4, def, rep);
```

### Column Projection

```c
carquet_batch_reader_config_t cfg;
carquet_batch_reader_config_init(&cfg);

// Read only specific columns
const char* names[] = {"id", "timestamp"};
cfg.column_names = names;
cfg.num_column_names = 2;
```

### Predicate Pushdown

Skip entire row groups that cannot match a query, based on column statistics:

```c
// Filter callback: only read row groups where column 0 might have values > threshold
bool filter_fn(const carquet_reader_t* reader, int32_t rg, void* ctx) {
    int64_t threshold = *(int64_t*)ctx;
    bool might_match = true;
    carquet_reader_row_group_matches(reader, rg, 0,
        CARQUET_COMPARE_GT, &threshold, sizeof(threshold), &might_match);
    return might_match;
}

int64_t threshold = 1000;
cfg.row_group_filter = filter_fn;
cfg.row_group_filter_ctx = &threshold;
// Non-matching row groups are skipped with zero I/O
```

### I/O Coalescing

Pre-buffer multiple columns in a single read (reduces seeks for fread path, no-op for mmap):

```c
int32_t cols[] = {0, 2, 5};
carquet_reader_prebuffer(reader, 0, cols, 3, &err);
// Subsequent column reads from row group 0 use the cached data
```

### Compression

| Codec | Enum | Best For |
|-------|------|----------|
| ZSTD | `CARQUET_COMPRESSION_ZSTD` | Best overall (great ratio + speed) |
| LZ4 | `CARQUET_COMPRESSION_LZ4_RAW` | Read-heavy workloads (fastest decompression) |
| Snappy | `CARQUET_COMPRESSION_SNAPPY` | Wide compatibility |
| GZIP | `CARQUET_COMPRESSION_GZIP` | Maximum compatibility with older tools |

```c
opts.compression = CARQUET_COMPRESSION_ZSTD;
opts.compression_level = 1;  // 0 = codec default; ZSTD: 1-22, GZIP: 1-9
```

### Writer Options

```c
carquet_writer_options_t opts;
carquet_writer_options_init(&opts);
opts.compression        = CARQUET_COMPRESSION_ZSTD;
opts.row_group_size     = 128 * 1024 * 1024;  // 128 MB row groups
opts.write_statistics   = true;                // min/max for predicate pushdown
opts.write_crc          = true;                // CRC32 page verification
opts.write_bloom_filters = true;               // bloom filters per column
opts.write_page_index   = true;                // column/offset page indexes
```

### Error Handling

```c
carquet_error_t err = CARQUET_ERROR_INIT;
carquet_reader_t* r = carquet_reader_open("data.parquet", NULL, &err);
if (!r) {
    printf("[%s] %s\n", carquet_status_name(err.code), err.message);
    printf("Hint: %s\n", carquet_error_recovery_hint(err.code));
    return 1;
}
```

All functions return `carquet_status_t` or use `carquet_error_t*` out-parameters. Programming errors (NULL where a valid pointer is required) trigger assertions; runtime errors (bad files, OOM) return error codes.

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
| Predicate pushdown | Row group filtering via statistics; page-level via column index |
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

## API Reference

Full API is in [`include/carquet/carquet.h`](include/carquet/carquet.h). Key types:

| Type | Purpose |
|------|---------|
| `carquet_reader_t` | File reader (open from path, FILE*, or memory buffer) |
| `carquet_writer_t` | File writer |
| `carquet_batch_reader_t` | High-level batch iteration |
| `carquet_schema_t` | Schema definition and introspection |
| `carquet_error_t` | Rich error info (code, message, source location, recovery hint) |

<details>
<summary>Core API functions</summary>

**Reader**
```c
carquet_reader_t* carquet_reader_open(const char* path, const carquet_reader_options_t* opts, carquet_error_t* err);
carquet_reader_t* carquet_reader_open_buffer(const void* buf, size_t size, const carquet_reader_options_t* opts, carquet_error_t* err);
void              carquet_reader_close(carquet_reader_t* reader);
int64_t           carquet_reader_num_rows(const carquet_reader_t* reader);
int32_t           carquet_reader_num_columns(const carquet_reader_t* reader);
```

**Batch Reader**
```c
carquet_batch_reader_t* carquet_batch_reader_create(carquet_reader_t* reader, const carquet_batch_reader_config_t* cfg, carquet_error_t* err);
carquet_status_t        carquet_batch_reader_next(carquet_batch_reader_t* br, carquet_row_batch_t** batch);
carquet_status_t        carquet_row_batch_column(const carquet_row_batch_t* batch, int32_t col, const void** data, const uint8_t** nulls, int64_t* n);
```

**Writer**
```c
carquet_writer_t*  carquet_writer_create(const char* path, const carquet_schema_t* schema, const carquet_writer_options_t* opts, carquet_error_t* err);
carquet_status_t   carquet_writer_write_batch(carquet_writer_t* w, int32_t col, const void* values, int64_t n, const int16_t* def, const int16_t* rep);
carquet_status_t   carquet_writer_close(carquet_writer_t* w);
```

**Schema**
```c
carquet_schema_t* carquet_schema_create(carquet_error_t* err);
carquet_status_t  carquet_schema_add_column(carquet_schema_t* s, const char* name, carquet_physical_type_t type, const carquet_logical_type_t* logical, carquet_field_repetition_t rep, int32_t type_len, int32_t parent);
int32_t           carquet_schema_add_list(carquet_schema_t* s, const char* name, carquet_physical_type_t elem_type, const carquet_logical_type_t* elem_logical, carquet_field_repetition_t rep, int32_t type_len, int32_t parent);
int32_t           carquet_schema_add_map(carquet_schema_t* s, const char* name, carquet_physical_type_t key_type, const carquet_logical_type_t* key_logical, int32_t key_len, carquet_physical_type_t val_type, const carquet_logical_type_t* val_logical, int32_t val_len, carquet_field_repetition_t rep, int32_t parent);
```

**Filtering**
```c
int32_t carquet_reader_filter_row_groups(const carquet_reader_t* reader, int32_t col, carquet_compare_op_t op, const void* value, int32_t value_size, int32_t* matching, int32_t max);
```

</details>

## Project Structure

```
include/carquet/   Public API (carquet.h, types.h, error.h)
src/
  core/            Arena allocator, buffer, bitpack, endian
  encoding/        PLAIN, RLE, DELTA, DICTIONARY, BYTE_STREAM_SPLIT
  compression/     Snappy (internal), GZIP, ZSTD, LZ4 (wrappers)
  thrift/          Thrift compact protocol for Parquet metadata
  simd/            Runtime dispatch + x86 (SSE/AVX2/AVX-512) + ARM (NEON/SVE)
  reader/          File, row group, column, page, batch readers + mmap
  writer/          File, row group, column, page writers
  metadata/        Schema, statistics, bloom filters, page indexes
  cli/             CLI tool and code generator
  util/            CRC32, xxHash
tests/             18 test files
examples/          basic_write_read, data_types, compression_codecs, nullable_columns, advanced_features
benchmark/         Performance benchmarks and comparison tools
```

## License

MIT
