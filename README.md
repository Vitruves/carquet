# Carquet

[![Build](https://github.com/Vitruves/carquet/actions/workflows/cpp.yml/badge.svg)](https://github.com/Vitruves/carquet/actions/workflows/cpp.yml)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)
![C Standard](https://img.shields.io/badge/C-C11-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A pure C library for reading and writing Apache Parquet files.

<img width="1792" height="592" alt="carquet" src="https://github.com/user-attachments/assets/ef669e62-6cf0-4dc0-9de8-dd2ab5afaeb3" />


## Why Carquet?

**The primary goal of Carquet is to provide Parquet support in pure C.** Before Carquet, there was no production-ready C library for Parquet - only C++ (Arrow), Rust, Java, and Python implementations.

### Use Cases

- **Embedded systems** - No C++ runtime, no exceptions, minimal dependencies
- **C codebases** - Native integration without FFI or language bridges
- **Minimal binaries** - ~200KB vs ~50MB+ for Arrow
- **Constrained environments** - IoT, microcontrollers, legacy systems

### Carquet vs Apache Arrow

Carquet is **not** a replacement for Apache Arrow. Arrow is the industry standard with years of production use, full feature support, and a large community.

| Aspect | Arrow Parquet | Carquet |
|--------|---------------|---------|
| Language | C++ | **Pure C11** |
| Dependencies | Many (Boost, etc.) | **zstd + zlib only** |
| Binary size | ~50MB+ | **~200KB** |
| Write speed (ARM) | Baseline | **1.5-5x faster** |
| Write speed (x86) | Baseline | ~same |
| Read speed (ARM) | Baseline | ~same to 1.3x faster |
| Read speed (x86) | **Baseline** | 2.5-5x slower |
| ZSTD file size | Baseline | **~1.4x smaller** |
| Nested types | **Full support** | Basic |
| Encryption | **Yes** | No |
| Community | **Large, mature** | New |
| Production tested | **Extensive** | Limited |

**Choose Carquet if:** You need Parquet in a C-only environment, want minimal dependencies, or are building for embedded/constrained systems.

**Choose Arrow if:** You need full feature support, battle-tested reliability, or are in a C++/Python/Java environment.

## Features

- **Pure C11** - Only external dependencies are zstd and zlib (auto-fetched by CMake if missing). Snappy and LZ4 are internal implementations.
- **Portable** - Works on any architecture. SIMD optimizations (SSE4.2, AVX2, AVX-512, NEON, SVE) with automatic runtime detection and scalar fallbacks. ARM CRC32 hardware acceleration.
- **Big-Endian Support** - Proper byte-order handling for s390x, SPARC, PowerPC, etc.
- **Parquet Support**:
  - All physical types (BOOLEAN, INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY)
  - All encodings (PLAIN, RLE, DICTIONARY, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, BYTE_STREAM_SPLIT)
  - All compression codecs (UNCOMPRESSED, SNAPPY, GZIP, LZ4, ZSTD)
  - Nullable columns with definition levels
  - Basic nested schema support (groups, definition/repetition levels)
- **Production Features**:
  - CRC32 page verification for data integrity (hardware-accelerated on ARM)
  - Column statistics for predicate pushdown
  - Memory-mapped I/O with zero-copy reads
  - Column projection for efficient reads
  - OpenMP parallel column reading (when available)
- **Streaming API** - Read and write large files without loading everything into memory
- **PyArrow Compatible** - Full interoperability with Python's PyArrow library

### Current Limitations

- Complex nested types (deeply nested lists/maps) are not fully supported
- No encryption support
- Bloom filters are read-only
- ZSTD decompression is single-threaded (Arrow uses multi-threaded)

## Table of Contents

- [Features](#features)
- [Building](#building)
- [Quick Start](#quick-start)
- [Reading Parquet Files](#reading-parquet-files)
- [Writing Parquet Files](#writing-parquet-files)
- [Schema API](#schema-api)
- [Compression](#compression)
- [Batch Reading](#batch-reading)
- [Error Handling](#error-handling)
- [Memory Management](#memory-management)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Interoperability](#interoperability)
- [Performance](#performance)

## Building

### Requirements

- C11-compatible compiler (GCC 4.9+, Clang 3.4+, MSVC 2015+)
- CMake 3.16+
- zstd and zlib (automatically fetched via FetchContent if not found on system)
- OpenMP (optional, for parallel column reading)

Works on Linux, macOS, Windows, and any POSIX system. Tested on x86_64, ARM64, and should work on RISC-V, MIPS, PowerPC, s390x, etc.

### Basic Build

```bash
git clone https://github.com/user/carquet.git
cd carquet
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CARQUET_BUILD_TESTS` | ON | Build test suite |
| `CARQUET_BUILD_EXAMPLES` | ON | Build example programs |
| `CARQUET_BUILD_BENCHMARKS` | ON | Build benchmark programs |
| `CARQUET_BUILD_SHARED` | OFF | Build shared library instead of static |
| `CARQUET_ENABLE_SSE` | ON | Enable SSE optimizations (x86) |
| `CARQUET_ENABLE_AVX2` | ON | Enable AVX2 optimizations (x86) |
| `CARQUET_ENABLE_AVX512` | ON | Enable AVX-512 optimizations (x86) |
| `CARQUET_ENABLE_NEON` | ON | Enable NEON optimizations (ARM) |
| `CARQUET_ENABLE_SVE` | OFF | Enable SVE optimizations (ARM) |

### Example: Release Build with Shared Library

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCARQUET_BUILD_SHARED=ON
make -j$(nproc)
sudo make install
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

## Quick Start

### Include Header

```c
#include <carquet/carquet.h>
```

### Link Library

```bash
# Static linking
gcc myprogram.c -I/path/to/carquet/include -L/path/to/carquet/build -lcarquet -o myprogram

# Or with pkg-config (after install)
gcc myprogram.c $(pkg-config --cflags --libs carquet) -o myprogram
```

### Minimal Example

```c
#include <carquet/carquet.h>
#include <stdio.h>

int main(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    // Create schema with two columns
    carquet_schema_t* schema = carquet_schema_create(&err);
    carquet_schema_add_column(schema, "id", CARQUET_PHYSICAL_INT32,
                              NULL, CARQUET_REPETITION_REQUIRED, 0);
    carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_DOUBLE,
                              NULL, CARQUET_REPETITION_REQUIRED, 0);

    // Write data
    carquet_writer_t* writer = carquet_writer_create("test.parquet", schema, NULL, &err);

    int32_t ids[] = {1, 2, 3, 4, 5};
    double values[] = {1.1, 2.2, 3.3, 4.4, 5.5};

    carquet_writer_write_batch(writer, 0, ids, 5, NULL, NULL);
    carquet_writer_write_batch(writer, 1, values, 5, NULL, NULL);
    carquet_writer_close(writer);

    // Read data back
    carquet_reader_t* reader = carquet_reader_open("test.parquet", NULL, &err);
    printf("Rows: %lld\n", (long long)carquet_reader_num_rows(reader));
    carquet_reader_close(reader);

    carquet_schema_free(schema);
    return 0;
}
```

## Reading Parquet Files

### Opening a File

```c
carquet_error_t err = CARQUET_ERROR_INIT;

// Basic open
carquet_reader_t* reader = carquet_reader_open("data.parquet", NULL, &err);
if (!reader) {
    printf("Error: %s\n", err.message);
    return 1;
}

// With options
carquet_reader_options_t opts;
carquet_reader_options_init(&opts);
opts.use_mmap = true;  // Use memory-mapped I/O

carquet_reader_t* reader = carquet_reader_open("data.parquet", &opts, &err);
```

### Getting File Metadata

```c
int64_t num_rows = carquet_reader_num_rows(reader);
int32_t num_columns = carquet_reader_num_columns(reader);
int32_t num_row_groups = carquet_reader_num_row_groups(reader);

// Get schema
const carquet_schema_t* schema = carquet_reader_schema(reader);

// Get column info
for (int32_t i = 0; i < num_columns; i++) {
    const char* name = carquet_schema_column_name(schema, i);
    carquet_physical_type_t type = carquet_schema_column_type(schema, i);
    printf("Column %d: %s (type: %s)\n", i, name, carquet_physical_type_name(type));
}
```

### Reading Column Data (Low-Level API)

```c
// Get column reader for row group 0, column 0
carquet_column_reader_t* col = carquet_reader_get_column(reader, 0, 0, &err);
if (!col) {
    printf("Error: %s\n", err.message);
}

// Read values
int64_t values[1024];
int16_t def_levels[1024];  // For nullable columns
int16_t rep_levels[1024];  // For nested/repeated columns

int64_t count = carquet_column_read_batch(col, values, 1024, def_levels, rep_levels);
printf("Read %lld values\n", (long long)count);

carquet_column_reader_free(col);
```

### Reading with Batch Reader (High-Level API)

```c
// Configure batch reader
carquet_batch_reader_config_t config;
carquet_batch_reader_config_init(&config);
config.batch_size = 10000;  // Rows per batch

// Create batch reader
carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &config, &err);

// Read batches
carquet_row_batch_t* batch = NULL;
while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
    int64_t num_rows = carquet_row_batch_num_rows(batch);
    int32_t num_cols = carquet_row_batch_num_columns(batch);

    // Access column data
    const void* data;
    const uint8_t* null_bitmap;
    int64_t num_values;

    carquet_row_batch_column(batch, 0, &data, &null_bitmap, &num_values);
    const int32_t* ids = (const int32_t*)data;

    // Process data...

    carquet_row_batch_free(batch);
    batch = NULL;
}

carquet_batch_reader_free(batch_reader);
```

### Column Projection

Read only specific columns for better performance:

```c
carquet_batch_reader_config_t config;
carquet_batch_reader_config_init(&config);

// By column indices
int32_t columns[] = {0, 3, 5};  // Only columns 0, 3, and 5
config.column_indices = columns;
config.num_columns = 3;

// Or by column names
const char* names[] = {"id", "timestamp", "value"};
config.column_names = names;
config.num_column_names = 3;
```

### Row Group Filtering (Predicate Pushdown)

Filter row groups using statistics before reading:

```c
// Find row groups where column 0 (id) might contain value > 1000
int32_t search_value = 1000;
int32_t matching_rgs[100];

int32_t num_matching = carquet_reader_filter_row_groups(
    reader,
    0,                      // Column index
    CARQUET_COMPARE_GT,     // Greater than
    &search_value,
    sizeof(int32_t),
    matching_rgs,
    100                     // Max results
);

printf("Found %d row groups that might contain id > 1000\n", num_matching);
```

### Reading from Memory Buffer

```c
// Read file into memory (e.g., from network, embedded resource)
uint8_t* buffer = ...;
size_t size = ...;

carquet_reader_t* reader = carquet_reader_open_buffer(buffer, size, NULL, &err);
// Use reader as normal...
carquet_reader_close(reader);
```

### Closing the Reader

```c
carquet_reader_close(reader);
```

## Writing Parquet Files

### Creating a Schema

```c
carquet_error_t err = CARQUET_ERROR_INIT;
carquet_schema_t* schema = carquet_schema_create(&err);

// Add required column (non-nullable)
carquet_schema_add_column(schema, "id", CARQUET_PHYSICAL_INT64,
                          NULL, CARQUET_REPETITION_REQUIRED, 0);

// Add optional column (nullable)
carquet_schema_add_column(schema, "name", CARQUET_PHYSICAL_BYTE_ARRAY,
                          NULL, CARQUET_REPETITION_OPTIONAL, 0);

// Add column with logical type
carquet_logical_type_t timestamp_type = {
    .type = CARQUET_LOGICAL_TIMESTAMP,
    .timestamp = { .unit = CARQUET_TIME_MILLIS, .is_adjusted_to_utc = true }
};
carquet_schema_add_column(schema, "created_at", CARQUET_PHYSICAL_INT64,
                          &timestamp_type, CARQUET_REPETITION_REQUIRED, 0);
```

### Writer Options

```c
carquet_writer_options_t opts;
carquet_writer_options_init(&opts);

opts.compression = CARQUET_COMPRESSION_ZSTD;  // Compression codec
opts.compression_level = 3;                    // Codec-specific level
opts.row_group_size = 128 * 1024 * 1024;      // 128 MB row groups
opts.page_size = 1024 * 1024;                  // 1 MB pages
opts.write_statistics = true;                  // Enable min/max statistics
opts.write_page_checksums = true;              // Enable CRC32 verification
```

### Creating a Writer

```c
carquet_writer_t* writer = carquet_writer_create(
    "output.parquet",
    schema,
    &opts,  // NULL for defaults
    &err
);

if (!writer) {
    printf("Error: %s\n", err.message);
    carquet_schema_free(schema);
    return 1;
}
```

### Writing Data

```c
// Write column 0 (id)
int64_t ids[] = {1, 2, 3, 4, 5};
carquet_writer_write_batch(writer, 0, ids, 5, NULL, NULL);

// Write column 1 (name) - with nulls (sparse: only non-null values in array)
carquet_byte_array_t names[] = {
    {5, (uint8_t*)"Alice"},
    {3, (uint8_t*)"Bob"},
    {5, (uint8_t*)"David"},
    {3, (uint8_t*)"Eve"}
};
int16_t def_levels[] = {1, 1, 0, 1, 1};  // 5 rows: 0 = null, 1 = present
carquet_writer_write_batch(writer, 1, names, 5, def_levels, NULL);

// Write column 2 (timestamp)
int64_t timestamps[] = {1703980800000, 1703984400000, 1703988000000,
                         1703991600000, 1703995200000};
carquet_writer_write_batch(writer, 2, timestamps, 5, NULL, NULL);
```

### Starting a New Row Group

```c
// Manually start a new row group (optional - automatic based on row_group_size)
carquet_writer_new_row_group(writer);
```

### Closing the Writer

```c
carquet_status_t status = carquet_writer_close(writer);
if (status != CARQUET_OK) {
    printf("Error closing file\n");
}

carquet_schema_free(schema);
```

## Schema API

### Physical Types

| Type | C Type | Description |
|------|--------|-------------|
| `CARQUET_PHYSICAL_BOOLEAN` | `uint8_t` | Boolean (0 or 1) |
| `CARQUET_PHYSICAL_INT32` | `int32_t` | 32-bit signed integer |
| `CARQUET_PHYSICAL_INT64` | `int64_t` | 64-bit signed integer |
| `CARQUET_PHYSICAL_INT96` | `uint8_t[12]` | 96-bit integer (legacy timestamp) |
| `CARQUET_PHYSICAL_FLOAT` | `float` | 32-bit IEEE 754 |
| `CARQUET_PHYSICAL_DOUBLE` | `double` | 64-bit IEEE 754 |
| `CARQUET_PHYSICAL_BYTE_ARRAY` | `carquet_byte_array_t` | Variable-length bytes |
| `CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY` | `uint8_t[]` | Fixed-length bytes |

### Logical Types

| Logical Type | Physical Type | Description |
|--------------|---------------|-------------|
| `CARQUET_LOGICAL_STRING` | BYTE_ARRAY | UTF-8 string |
| `CARQUET_LOGICAL_DATE` | INT32 | Days since epoch |
| `CARQUET_LOGICAL_TIME` | INT32/INT64 | Time of day |
| `CARQUET_LOGICAL_TIMESTAMP` | INT64 | Timestamp with timezone |
| `CARQUET_LOGICAL_DECIMAL` | INT32/INT64/FIXED | Decimal with precision/scale |
| `CARQUET_LOGICAL_UUID` | FIXED[16] | UUID |
| `CARQUET_LOGICAL_JSON` | BYTE_ARRAY | JSON string |

### Repetition Types

| Type | Description |
|------|-------------|
| `CARQUET_REPETITION_REQUIRED` | Non-nullable, exactly one value |
| `CARQUET_REPETITION_OPTIONAL` | Nullable, zero or one value |
| `CARQUET_REPETITION_REPEATED` | Zero or more values (list) |

### Nested Schemas

```c
// Create a nested schema: person { name: string, address { street, city } }
carquet_schema_t* schema = carquet_schema_create(&err);

// Add group for person (root is implicit)
int32_t person_idx = carquet_schema_add_group(schema, "person",
                                               CARQUET_REPETITION_REQUIRED, 0);

// Add leaf columns under person
carquet_schema_add_column(schema, "name", CARQUET_PHYSICAL_BYTE_ARRAY,
                          NULL, CARQUET_REPETITION_REQUIRED, person_idx);

// Add nested group for address
int32_t address_idx = carquet_schema_add_group(schema, "address",
                                                CARQUET_REPETITION_OPTIONAL, person_idx);

// Add columns under address
carquet_schema_add_column(schema, "street", CARQUET_PHYSICAL_BYTE_ARRAY,
                          NULL, CARQUET_REPETITION_REQUIRED, address_idx);
carquet_schema_add_column(schema, "city", CARQUET_PHYSICAL_BYTE_ARRAY,
                          NULL, CARQUET_REPETITION_REQUIRED, address_idx);
```

## Compression

### Available Codecs

| Codec | Enum Value | Compression | Decompression | Ratio |
|-------|------------|-------------|---------------|-------|
| Uncompressed | `CARQUET_COMPRESSION_UNCOMPRESSED` | N/A | N/A | 1.0x |
| Snappy | `CARQUET_COMPRESSION_SNAPPY` | Very Fast | Very Fast | ~4x |
| LZ4 | `CARQUET_COMPRESSION_LZ4` | Very Fast | Fastest | ~4x |
| GZIP | `CARQUET_COMPRESSION_GZIP` | Slow | Medium | ~6x |
| ZSTD | `CARQUET_COMPRESSION_ZSTD` | Fast | Fast | ~7x |

### Choosing a Codec

- **ZSTD**: Best overall choice - excellent compression with good speed
- **LZ4**: Best for read-heavy workloads - fastest decompression
- **Snappy**: Good balance, widely compatible
- **GZIP**: Maximum compatibility with older tools

### Setting Compression Level

```c
opts.compression = CARQUET_COMPRESSION_ZSTD;
opts.compression_level = 3;  // ZSTD: 1-22, default 3
                              // GZIP: 1-9, default 6
```

## Batch Reading

The batch reader provides an efficient way to read data in chunks:

```c
carquet_batch_reader_config_t config;
carquet_batch_reader_config_init(&config);
config.batch_size = 65536;  // 64K rows per batch

carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &config, &err);

carquet_row_batch_t* batch = NULL;
int64_t total_rows = 0;

while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
    int64_t batch_rows = carquet_row_batch_num_rows(batch);
    total_rows += batch_rows;

    // Access data for each column
    for (int32_t col = 0; col < carquet_row_batch_num_columns(batch); col++) {
        const void* data;
        const uint8_t* null_bitmap;
        int64_t num_values;

        carquet_row_batch_column(batch, col, &data, &null_bitmap, &num_values);

        // Process column data...
        // null_bitmap: bit i is 1 if value i is NOT null
    }

    carquet_row_batch_free(batch);
    batch = NULL;
}

printf("Total rows: %lld\n", (long long)total_rows);
carquet_batch_reader_free(batch_reader);
```

## Error Handling

### Error Structure

```c
carquet_error_t err = CARQUET_ERROR_INIT;

carquet_reader_t* reader = carquet_reader_open("data.parquet", NULL, &err);
if (!reader) {
    printf("Error code: %d\n", err.code);
    printf("Message: %s\n", err.message);
    printf("Function: %s\n", err.function);
    printf("File: %s:%d\n", err.file, err.line);

    // Get recovery hint
    const char* hint = carquet_error_recovery_hint(err.code);
    if (hint) {
        printf("Hint: %s\n", hint);
    }
}
```

### Formatting Errors

```c
char error_buffer[1024];
carquet_error_format(&err, error_buffer, sizeof(error_buffer));
printf("%s\n", error_buffer);
// Output: [File not found] Failed to open data.parquet (file offset: 0)
//         Hint: Check that the file exists and is readable
```

### Error Codes

| Category | Codes | Description |
|----------|-------|-------------|
| Success | `CARQUET_OK` | Operation succeeded |
| General | `CARQUET_ERROR_OUT_OF_MEMORY` | Memory allocation failed |
| File I/O | `CARQUET_ERROR_FILE_*` | File operation errors |
| Format | `CARQUET_ERROR_INVALID_MAGIC`, `CARQUET_ERROR_INVALID_FOOTER` | Invalid file format |
| Encoding | `CARQUET_ERROR_DECODE`, `CARQUET_ERROR_INVALID_ENCODING` | Encoding errors |
| Compression | `CARQUET_ERROR_COMPRESSION`, `CARQUET_ERROR_UNSUPPORTED_CODEC` | Compression errors |
| Integrity | `CARQUET_ERROR_CRC_MISMATCH`, `CARQUET_ERROR_CHECKSUM` | Data corruption |

### API Design: Assertions vs Error Returns

Carquet distinguishes between **programming errors** (bugs) and **runtime errors** (expected failures):

| Error Type | Handling | Example |
|------------|----------|---------|
| Programming error | `assert()` | Passing NULL to `carquet_buffer_init()` |
| Runtime error | Return status | File not found, corrupted data, out of memory |

**Rationale**: If you pass NULL where a valid pointer is required, that's a bug in your code - not something to "handle" at runtime. Assertions catch these during development. Runtime errors (bad files, memory exhaustion) return proper error codes since they can legitimately occur in production.

```c
// These assert on NULL (programming errors - fix your code!)
carquet_buffer_init(&buf);      // buf must not be NULL
carquet_arena_destroy(&arena);  // arena must not be NULL

// These return errors (runtime failures - handle gracefully)
carquet_reader_t* r = carquet_reader_open("bad.parquet", NULL, &err);
if (!r) { /* file might not exist or be corrupted */ }
```

### Checking Recoverability

```c
if (!carquet_error_is_recoverable(err.code)) {
    printf("Fatal error - cannot continue\n");
} else {
    printf("Recoverable error - can retry or skip\n");
}
```

## Memory Management

### Arena Allocator

Carquet uses arena allocation internally for efficient memory management:

```c
// Arenas are used internally - you typically don't need to manage them directly
// The reader/writer handle all memory management automatically
```

### Custom Allocator

```c
// Set a custom allocator before any Carquet calls
carquet_allocator_t alloc = {
    .malloc = my_malloc,
    .realloc = my_realloc,
    .free = my_free,
    .ctx = my_context
};
carquet_set_allocator(&alloc);
```

### Memory Tips

1. **Use batch reading** - Reads data in chunks instead of loading entire file
2. **Use column projection** - Only read columns you need
3. **Use memory-mapped I/O** - Let OS handle paging for large files
4. **Close readers/writers promptly** - Free memory when done

## API Reference

### Initialization

```c
carquet_status_t carquet_init(void);
const char* carquet_version(void);
const carquet_cpu_info_t* carquet_get_cpu_info(void);
```

### Schema

```c
carquet_schema_t* carquet_schema_create(carquet_error_t* error);
void carquet_schema_free(carquet_schema_t* schema);
int32_t carquet_schema_add_column(carquet_schema_t* schema, const char* name,
                                   carquet_physical_type_t type,
                                   const carquet_logical_type_t* logical_type,
                                   carquet_field_repetition_t repetition,
                                   int32_t parent_index);
int32_t carquet_schema_add_group(carquet_schema_t* schema, const char* name,
                                  carquet_field_repetition_t repetition,
                                  int32_t parent_index);
int32_t carquet_schema_num_columns(const carquet_schema_t* schema);
const char* carquet_schema_column_name(const carquet_schema_t* schema, int32_t index);
carquet_physical_type_t carquet_schema_column_type(const carquet_schema_t* schema, int32_t index);
```

### Reader

```c
carquet_reader_t* carquet_reader_open(const char* path,
                                       const carquet_reader_options_t* options,
                                       carquet_error_t* error);
carquet_reader_t* carquet_reader_open_buffer(const void* buffer, size_t size,
                                              const carquet_reader_options_t* options,
                                              carquet_error_t* error);
void carquet_reader_close(carquet_reader_t* reader);
int64_t carquet_reader_num_rows(const carquet_reader_t* reader);
int32_t carquet_reader_num_columns(const carquet_reader_t* reader);
int32_t carquet_reader_num_row_groups(const carquet_reader_t* reader);
const carquet_schema_t* carquet_reader_schema(const carquet_reader_t* reader);
carquet_column_reader_t* carquet_reader_get_column(carquet_reader_t* reader,
                                                    int32_t row_group,
                                                    int32_t column,
                                                    carquet_error_t* error);
```

### Batch Reader

```c
void carquet_batch_reader_config_init(carquet_batch_reader_config_t* config);
carquet_batch_reader_t* carquet_batch_reader_create(carquet_reader_t* reader,
                                                     const carquet_batch_reader_config_t* config,
                                                     carquet_error_t* error);
void carquet_batch_reader_free(carquet_batch_reader_t* batch_reader);
carquet_status_t carquet_batch_reader_next(carquet_batch_reader_t* batch_reader,
                                            carquet_row_batch_t** batch);
int64_t carquet_row_batch_num_rows(const carquet_row_batch_t* batch);
int32_t carquet_row_batch_num_columns(const carquet_row_batch_t* batch);
carquet_status_t carquet_row_batch_column(const carquet_row_batch_t* batch,
                                           int32_t column,
                                           const void** data,
                                           const uint8_t** null_bitmap,
                                           int64_t* num_values);
void carquet_row_batch_free(carquet_row_batch_t* batch);
```

### Writer

```c
void carquet_writer_options_init(carquet_writer_options_t* options);
carquet_writer_t* carquet_writer_create(const char* path,
                                         const carquet_schema_t* schema,
                                         const carquet_writer_options_t* options,
                                         carquet_error_t* error);
carquet_status_t carquet_writer_write_batch(carquet_writer_t* writer,
                                             int32_t column,
                                             const void* values,
                                             int64_t num_values,
                                             const int16_t* def_levels,
                                             const int16_t* rep_levels);
carquet_status_t carquet_writer_new_row_group(carquet_writer_t* writer);
carquet_status_t carquet_writer_close(carquet_writer_t* writer);
```

### Statistics and Filtering

```c
carquet_status_t carquet_reader_column_statistics(const carquet_reader_t* reader,
                                                   int32_t row_group_index,
                                                   int32_t column_index,
                                                   carquet_column_statistics_t* stats);
int32_t carquet_reader_filter_row_groups(const carquet_reader_t* reader,
                                          int32_t column_index,
                                          carquet_compare_op_t op,
                                          const void* value,
                                          int32_t value_size,
                                          int32_t* matching_row_groups,
                                          int32_t max_results);
```

## Examples

Example programs are in the `examples/` directory:

- **basic_write_read.c** - Simple write and read example
- **data_types.c** - Using different data types
- **compression_codecs.c** - Comparing compression codecs
- **nullable_columns.c** - Working with NULL values

Build and run examples:

```bash
cd build
./example_basic_write_read
./example_compression
./example_data_types
./example_nullable
```

## Interoperability

Carquet is tested bidirectionally with PyArrow, DuckDB, and fastparquet. Run `./interop/run_interop_tests.sh` to verify both directions (carquet reads others' files, others read carquet's files).

### PyArrow Compatibility

Files written by Carquet can be read by PyArrow and vice versa:

```python
import pyarrow.parquet as pq

# Read Carquet-written file
table = pq.read_table("carquet_output.parquet")
print(table.to_pandas())

# Write file for Carquet to read
import pyarrow as pa
table = pa.table({'id': [1, 2, 3], 'value': [1.1, 2.2, 3.3]})
pq.write_table(table, "pyarrow_output.parquet", compression='snappy')
```

### Apache Spark

```scala
// Read Carquet file in Spark
val df = spark.read.parquet("carquet_output.parquet")
df.show()
```

### DuckDB

```sql
-- Read Carquet file in DuckDB
SELECT * FROM read_parquet('carquet_output.parquet');
```

## Project Structure

```
carquet/
├── include/carquet/     # Public headers
│   ├── carquet.h        # Main API
│   ├── types.h          # Type definitions
│   └── error.h          # Error codes
├── src/
│   ├── compression/     # Compression codecs (LZ4, Snappy, GZIP, ZSTD)
│   ├── core/            # Core utilities (arena, buffer, endian)
│   ├── encoding/        # Parquet encodings (PLAIN, RLE, DELTA, etc.)
│   ├── metadata/        # File metadata, schema, statistics
│   ├── reader/          # File reader, batch reader, column reader
│   ├── writer/          # File writer, page writer
│   ├── simd/            # SIMD implementations (SSE, AVX, NEON)
│   ├── thrift/          # Thrift compact protocol
│   └── util/            # Utilities (CRC32, xxHash)
├── tests/               # Test suite
├── examples/            # Example programs
├── benchmark/           # Performance benchmarks
└── CMakeLists.txt
```

## Performance

Carquet's performance varies by platform and use case. These benchmarks show where Carquet excels and where Arrow is faster.

**Test configuration:** 10M rows, 3 columns (INT64 + DOUBLE + INT32), fair comparison with both libraries reading actual data values and verifying CRC checksums.

### Apple M3 (ARM64, macOS)

*MacBook Air (13-inch, M3, 2024), macOS Tahoe 26.2, PyArrow 20.0.0*

#### Writing (Carquet Excels)

| Codec | Carquet | PyArrow | Speedup |
|-------|---------|---------|---------|
| UNCOMPRESSED | 83 M rows/sec | 17 M rows/sec | **5.0x faster** |
| SNAPPY | 44 M rows/sec | 15 M rows/sec | **3.0x faster** |
| ZSTD | 19 M rows/sec | 12 M rows/sec | **1.5x faster** |

#### Reading

| Codec | Carquet | PyArrow | Ratio |
|-------|---------|---------|-------|
| UNCOMPRESSED | 475 M rows/sec | 368 M rows/sec | 1.3x faster |
| SNAPPY | 345 M rows/sec | 313 M rows/sec | 1.1x faster |
| ZSTD | 108 M rows/sec | 198 M rows/sec | 0.55x slower |

#### Compression Ratio

| Codec | Carquet | PyArrow | Ratio |
|-------|---------|---------|-------|
| ZSTD | 107 MB | 150 MB | **1.4x smaller** |
| SNAPPY | 191 MB | 174 MB | 1.1x larger |
| UNCOMPRESSED | 191 MB | 201 MB | 1.05x smaller |

### Intel Xeon E5-2680 (x86_64, Linux)

*Dell Precision T7600, 2x Intel Xeon E5-2680 (32 cores @ 2.7GHz), 94GB RAM, Ubuntu 24.04*

#### Writing

| Codec | Carquet | PyArrow | Speedup |
|-------|---------|---------|---------|
| UNCOMPRESSED | 4.3 M rows/sec | 4.2 M rows/sec | ~same |
| SNAPPY | 3.4 M rows/sec | 4.1 M rows/sec | 0.84x slower |
| ZSTD | 3.1 M rows/sec | 3.4 M rows/sec | 0.93x slower |

#### Reading (PyArrow Faster)

| Codec | Carquet | PyArrow | Ratio |
|-------|---------|---------|-------|
| UNCOMPRESSED | 29 M rows/sec | 73 M rows/sec | 0.40x slower |
| SNAPPY | 18 M rows/sec | 83 M rows/sec | 0.22x slower |
| ZSTD | 12 M rows/sec | 70 M rows/sec | 0.17x slower |

#### Compression Ratio

| Codec | Carquet | PyArrow | Ratio |
|-------|---------|---------|-------|
| ZSTD | 107 MB | 150 MB | **1.4x smaller** |
| SNAPPY | 191 MB | 174 MB | 1.1x larger |
| UNCOMPRESSED | 191 MB | 201 MB | 1.05x smaller |

### Running Benchmarks

```bash
cd build
./benchmark_carquet                    # Carquet only
../benchmark/run_benchmark.sh          # Full comparison with PyArrow
```

## License

MIT License

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a pull request.
