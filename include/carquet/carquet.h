/**
 * @file carquet.h
 * @brief Carquet - High-Performance Pure C Parquet Library
 * @version 0.1.0
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * @license MIT License
 *
 * Carquet is a production-ready, minimal-dependency pure C11 library for reading
 * and writing Apache Parquet files. It features automatic SIMD optimization
 * for maximum performance across x86-64 (SSE4.2, AVX2, AVX-512) and ARM
 * (NEON, SVE) architectures.
 *
 * @section features Key Features
 *
 * - **Minimal Dependencies**: Pure C11 with optional zstd/zlib for compression
 * - **SIMD Optimized**: Automatic CPU feature detection and optimal code dispatch
 * - **Complete Parquet Support**: All physical types, encodings, and compression codecs
 * - **Production Ready**: CRC32 verification, statistics, predicate pushdown
 * - **Memory Efficient**: Streaming API, column projection, memory-mapped I/O
 * - **Thread Safe**: Concurrent reads supported, atomic initialization
 *
 * @section quickstart Quick Start
 *
 * @subsection reading Reading a Parquet File
 * @code{.c}
 * #include <carquet/carquet.h>
 *
 * carquet_error_t err = CARQUET_ERROR_INIT;
 *
 * // Open file
 * carquet_reader_t* reader = carquet_reader_open("data.parquet", NULL, &err);
 * if (!reader) {
 *     fprintf(stderr, "Error: %s\n", err.message);
 *     return 1;
 * }
 *
 * // Get metadata
 * int64_t num_rows = carquet_reader_num_rows(reader);
 * int32_t num_cols = carquet_reader_num_columns(reader);
 *
 * // Read column data using batch reader
 * carquet_batch_reader_config_t config;
 * carquet_batch_reader_config_init(&config);
 * config.batch_size = 10000;
 *
 * carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &config, &err);
 * carquet_row_batch_t* batch = NULL;
 *
 * while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
 *     const void* data;
 *     const uint8_t* nulls;
 *     int64_t count;
 *     carquet_row_batch_column(batch, 0, &data, &nulls, &count);
 *     // Process data...
 *     carquet_row_batch_free(batch);
 *     batch = NULL;
 * }
 *
 * carquet_batch_reader_free(batch_reader);
 * carquet_reader_close(reader);
 * @endcode
 *
 * @subsection writing Writing a Parquet File
 * @code{.c}
 * #include <carquet/carquet.h>
 *
 * carquet_error_t err = CARQUET_ERROR_INIT;
 *
 * // Create schema
 * carquet_schema_t* schema = carquet_schema_create(&err);
 * carquet_schema_add_column(schema, "id", CARQUET_PHYSICAL_INT64,
 *                           NULL, CARQUET_REPETITION_REQUIRED, 0);
 * carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_DOUBLE,
 *                           NULL, CARQUET_REPETITION_REQUIRED, 0);
 *
 * // Create writer with compression
 * carquet_writer_options_t opts;
 * carquet_writer_options_init(&opts);
 * opts.compression = CARQUET_COMPRESSION_ZSTD;
 *
 * carquet_writer_t* writer = carquet_writer_create("output.parquet", schema, &opts, &err);
 *
 * // Write data
 * int64_t ids[] = {1, 2, 3, 4, 5};
 * double values[] = {1.1, 2.2, 3.3, 4.4, 5.5};
 *
 * carquet_writer_write_batch(writer, 0, ids, 5, NULL, NULL);
 * carquet_writer_write_batch(writer, 1, values, 5, NULL, NULL);
 *
 * carquet_writer_close(writer);
 * carquet_schema_free(schema);
 * @endcode
 *
 * @section threading Thread Safety
 *
 * - Library initialization (carquet_init) is thread-safe and uses atomic operations
 * - Multiple readers can read the same file concurrently
 * - A single reader/writer instance must not be shared across threads without synchronization
 * - Schema objects are immutable after creation and can be shared
 *
 * @section memory Memory Management
 *
 * - All returned pointers remain valid until their parent object is freed
 * - Batch data pointers are valid until carquet_row_batch_free() is called
 * - Schema pointers from readers are valid until the reader is closed
 * - Use carquet_set_allocator() to provide custom memory allocation
 *
 * @see https://parquet.apache.org/docs/ Apache Parquet Documentation
 * @see https://github.com/apache/parquet-format Parquet Format Specification
 */

#ifndef CARQUET_H
#define CARQUET_H

/* ============================================================================
 * Standard Library Includes
 * ============================================================================ */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

/* ============================================================================
 * Carquet Headers
 * ============================================================================ */

#include "types.h"
#include "error.h"

/* ============================================================================
 * C++ Compatibility
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Compiler Attributes
 * ============================================================================ */

/** @brief Mark function as non-null return */
#if defined(__GNUC__) || defined(__clang__)
    #define CARQUET_RETURNS_NONNULL __attribute__((returns_nonnull))
    #define CARQUET_NONNULL(...) __attribute__((nonnull(__VA_ARGS__)))
    #define CARQUET_WARN_UNUSED_RESULT __attribute__((warn_unused_result))
    #define CARQUET_DEPRECATED(msg) __attribute__((deprecated(msg)))
    #define CARQUET_PURE __attribute__((pure))
    #define CARQUET_CONST __attribute__((const))
#else
    #define CARQUET_RETURNS_NONNULL
    #define CARQUET_NONNULL(...)
    #define CARQUET_WARN_UNUSED_RESULT
    #define CARQUET_DEPRECATED(msg)
    #define CARQUET_PURE
    #define CARQUET_CONST
#endif

/* ============================================================================
 * API Visibility
 * ============================================================================ */

#if defined(CARQUET_BUILD_SHARED)
    #if defined(_WIN32) || defined(__CYGWIN__)
        #ifdef CARQUET_BUILDING_DLL
            #define CARQUET_API __declspec(dllexport)
        #else
            #define CARQUET_API __declspec(dllimport)
        #endif
    #elif defined(__GNUC__) || defined(__clang__)
        #define CARQUET_API __attribute__((visibility("default")))
    #else
        #define CARQUET_API
    #endif
#else
    #define CARQUET_API
#endif

/* ============================================================================
 * Version Information
 * ============================================================================
 *
 * Carquet follows Semantic Versioning (https://semver.org/).
 *
 * - MAJOR: Incompatible API changes
 * - MINOR: Backwards-compatible functionality additions
 * - PATCH: Backwards-compatible bug fixes
 */

/** @brief Major version number */
#define CARQUET_VERSION_MAJOR 0

/** @brief Minor version number */
#define CARQUET_VERSION_MINOR 1

/** @brief Patch version number */
#define CARQUET_VERSION_PATCH 0

/** @brief Version string in "MAJOR.MINOR.PATCH" format */
#define CARQUET_VERSION_STRING "0.1.0"

/** @brief Numeric version for compile-time comparisons: (MAJOR * 10000 + MINOR * 100 + PATCH) */
#define CARQUET_VERSION_NUMBER (CARQUET_VERSION_MAJOR * 10000 + CARQUET_VERSION_MINOR * 100 + CARQUET_VERSION_PATCH)

/**
 * @brief Get the library version as a string.
 *
 * Returns the version string in "MAJOR.MINOR.PATCH" format.
 * This is useful for runtime version checking and logging.
 *
 * @return Version string (statically allocated, never NULL)
 *
 * @note Thread-safe: Yes
 *
 * @code{.c}
 * printf("Using Carquet version %s\n", carquet_version());
 * @endcode
 */
CARQUET_API CARQUET_CONST CARQUET_RETURNS_NONNULL
const char* carquet_version(void);

/**
 * @brief Get individual version components.
 *
 * Retrieves the major, minor, and patch version numbers separately.
 * Useful for runtime compatibility checks.
 *
 * @param[out] major Major version number (may be NULL)
 * @param[out] minor Minor version number (may be NULL)
 * @param[out] patch Patch version number (may be NULL)
 *
 * @note Thread-safe: Yes
 *
 * @code{.c}
 * int major, minor, patch;
 * carquet_version_components(&major, &minor, &patch);
 * if (major != CARQUET_VERSION_MAJOR) {
 *     fprintf(stderr, "Warning: Header/library version mismatch\n");
 * }
 * @endcode
 */
CARQUET_API
void carquet_version_components(int* major, int* minor, int* patch);

/* ============================================================================
 * Library Initialization
 * ============================================================================
 *
 * Carquet automatically initializes itself on first use. Explicit initialization
 * is optional but can be useful for:
 *
 * - Deterministic startup behavior
 * - Early detection of initialization errors
 * - Controlling when CPU feature detection occurs
 */

/**
 * @brief Initialize the Carquet library.
 *
 * Performs CPU feature detection and sets up optimal SIMD dispatch tables.
 * This function is automatically called on first use of any Carquet function,
 * but can be called explicitly for deterministic initialization timing.
 *
 * Calling this function multiple times is safe and has no effect after the
 * first successful initialization.
 *
 * @return CARQUET_OK on success, error code on failure
 *
 * @note Thread-safe: Yes (uses atomic initialization)
 * @note Idempotent: Yes (safe to call multiple times)
 *
 * @code{.c}
 * // Optional: explicit initialization at program start
 * carquet_status_t status = carquet_init();
 * if (status != CARQUET_OK) {
 *     fprintf(stderr, "Failed to initialize Carquet: %s\n",
 *             carquet_status_string(status));
 *     return 1;
 * }
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT
carquet_status_t carquet_init(void);

/**
 * @brief CPU feature information detected at runtime.
 *
 * This structure contains the results of CPU feature detection,
 * used to select optimal SIMD implementations.
 */
typedef struct carquet_cpu_info {
    /* x86-64 features */
    bool has_sse2;          /**< SSE2 support (baseline for x86-64) */
    bool has_sse41;         /**< SSE4.1 support */
    bool has_sse42;         /**< SSE4.2 support (includes POPCNT, CRC32) */
    bool has_avx;           /**< AVX support */
    bool has_avx2;          /**< AVX2 support */
    bool has_avx512f;       /**< AVX-512 Foundation */
    bool has_avx512bw;      /**< AVX-512 Byte/Word instructions */
    bool has_avx512vl;      /**< AVX-512 Vector Length extensions */
    bool has_avx512vbmi;    /**< AVX-512 Vector Byte Manipulation */

    /* ARM features */
    bool has_neon;          /**< ARM NEON support */
    bool has_sve;           /**< ARM SVE support */
    int sve_vector_length;  /**< SVE vector length in bits (0 if not available) */
} carquet_cpu_info_t;

/**
 * @brief Get detected CPU features.
 *
 * Returns information about CPU features detected during library initialization.
 * This is useful for diagnostics and understanding which SIMD optimizations
 * are being used.
 *
 * @return Pointer to CPU info structure (statically allocated, never NULL)
 *
 * @note Thread-safe: Yes
 * @note The returned pointer remains valid for the lifetime of the program.
 *
 * @code{.c}
 * const carquet_cpu_info_t* cpu = carquet_get_cpu_info();
 * printf("SIMD features:\n");
 * printf("  AVX2: %s\n", cpu->has_avx2 ? "yes" : "no");
 * printf("  NEON: %s\n", cpu->has_neon ? "yes" : "no");
 * @endcode
 */
CARQUET_API CARQUET_PURE CARQUET_RETURNS_NONNULL
const carquet_cpu_info_t* carquet_get_cpu_info(void);

/* ============================================================================
 * Memory Allocation
 * ============================================================================
 *
 * By default, Carquet uses the standard C library allocator (malloc/free).
 * Custom allocators can be provided for integration with application-specific
 * memory management systems.
 */

/**
 * @brief Custom memory allocator interface.
 *
 * Users can provide custom memory allocation functions for all Carquet
 * operations. This is useful for:
 *
 * - Memory tracking and debugging
 * - Custom memory pools
 * - Integration with game engines or other frameworks
 *
 * All three function pointers must be provided (non-NULL) when setting
 * a custom allocator.
 */
typedef struct carquet_allocator {
    /**
     * @brief Allocate memory.
     * @param size Number of bytes to allocate
     * @param ctx User context pointer
     * @return Pointer to allocated memory, or NULL on failure
     */
    void* (*malloc)(size_t size, void* ctx);

    /**
     * @brief Reallocate memory.
     * @param ptr Pointer to existing allocation (may be NULL)
     * @param size New size in bytes
     * @param ctx User context pointer
     * @return Pointer to reallocated memory, or NULL on failure
     */
    void* (*realloc)(void* ptr, size_t size, void* ctx);

    /**
     * @brief Free memory.
     * @param ptr Pointer to free (may be NULL)
     * @param ctx User context pointer
     */
    void (*free)(void* ptr, void* ctx);

    /** @brief User context passed to all allocation functions */
    void* ctx;
} carquet_allocator_t;

/**
 * @brief Set the global memory allocator.
 *
 * Must be called before any other Carquet function that allocates memory.
 * If not called, the standard C library allocator is used.
 *
 * @param[in] allocator Custom allocator (NULL to reset to default)
 *
 * @warning Not thread-safe. Must be called before any concurrent Carquet usage.
 * @warning All function pointers in the allocator must be non-NULL.
 *
 * @code{.c}
 * carquet_allocator_t my_alloc = {
 *     .malloc = my_malloc,
 *     .realloc = my_realloc,
 *     .free = my_free,
 *     .ctx = my_context
 * };
 * carquet_set_allocator(&my_alloc);
 * @endcode
 */
CARQUET_API
void carquet_set_allocator(const carquet_allocator_t* allocator);

/**
 * @brief Get the current memory allocator.
 *
 * @return Pointer to current allocator configuration
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE
const carquet_allocator_t* carquet_get_allocator(void);

/* ============================================================================
 * Opaque Type Declarations
 * ============================================================================
 *
 * These types are opaque handles to internal structures. They can only be
 * created and manipulated through the public API functions.
 */

/** @brief Schema definition for a Parquet file */
typedef struct carquet_schema carquet_schema_t;

/** @brief Individual node within a schema (column or group) */
typedef struct carquet_schema_node carquet_schema_node_t;

/** @brief File reader handle */
typedef struct carquet_reader carquet_reader_t;

/** @brief File writer handle */
typedef struct carquet_writer carquet_writer_t;

/** @brief Column reader for streaming column data */
typedef struct carquet_column_reader carquet_column_reader_t;

/** @brief Column writer for streaming column data */
typedef struct carquet_column_writer carquet_column_writer_t;

/** @brief Row group metadata handle */
typedef struct carquet_row_group carquet_row_group_t;

/** @brief Bloom filter for membership testing */
typedef struct carquet_bloom_filter carquet_bloom_filter_t;

/** @brief Row batch for batch reading */
typedef struct carquet_row_batch carquet_row_batch_t;

/** @brief Batch reader for efficient columnar reading */
typedef struct carquet_batch_reader carquet_batch_reader_t;

/* ============================================================================
 * Schema API
 * ============================================================================
 *
 * The schema defines the structure of a Parquet file, including column names,
 * types, and nesting structure. Schemas support:
 *
 * - Flat structures (simple column list)
 * - Nested structures (groups containing columns)
 * - Repeated fields (lists/arrays)
 * - Optional fields (nullable columns)
 *
 * Schema Lifecycle:
 * 1. Create schema with carquet_schema_create()
 * 2. Add columns/groups with carquet_schema_add_column() / carquet_schema_add_group()
 * 3. Pass to writer or compare with reader schema
 * 4. Free with carquet_schema_free() when done
 */

/**
 * @brief Create a new empty schema.
 *
 * Creates a schema builder that can be populated with columns and groups.
 * The schema must be freed with carquet_schema_free() when no longer needed.
 *
 * @param[out] error Error information (may be NULL)
 * @return New schema handle, or NULL on error
 *
 * @note Thread-safe: Yes
 *
 * @code{.c}
 * carquet_error_t err = CARQUET_ERROR_INIT;
 * carquet_schema_t* schema = carquet_schema_create(&err);
 * if (!schema) {
 *     fprintf(stderr, "Failed to create schema: %s\n", err.message);
 * }
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT
carquet_schema_t* carquet_schema_create(carquet_error_t* error);

/**
 * @brief Free a schema and all associated resources.
 *
 * @param[in] schema Schema to free (may be NULL)
 *
 * @note Thread-safe: Yes (for different schema instances)
 * @note Safe to call with NULL (no-op)
 */
CARQUET_API
void carquet_schema_free(carquet_schema_t* schema);

/**
 * @brief Add a primitive (leaf) column to the schema.
 *
 * Adds a column that stores actual data values. For nested schemas, specify
 * the parent group index; for flat schemas, use 0 for root-level columns.
 *
 * @param[in,out] schema Target schema
 * @param[in] name Column name (must be unique within parent)
 * @param[in] physical_type Physical storage type
 * @param[in] logical_type Logical type annotation (may be NULL)
 * @param[in] repetition Field repetition level
 * @param[in] type_length Byte length for FIXED_LEN_BYTE_ARRAY (0 otherwise)
 * @return CARQUET_OK on success, error code on failure
 *
 * @note Thread-safe: No (schema is mutable during construction)
 *
 * @par Physical Types
 * - CARQUET_PHYSICAL_BOOLEAN: 1-bit boolean
 * - CARQUET_PHYSICAL_INT32: 32-bit signed integer
 * - CARQUET_PHYSICAL_INT64: 64-bit signed integer
 * - CARQUET_PHYSICAL_FLOAT: 32-bit IEEE 754 float
 * - CARQUET_PHYSICAL_DOUBLE: 64-bit IEEE 754 double
 * - CARQUET_PHYSICAL_BYTE_ARRAY: Variable-length byte sequence
 * - CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: Fixed-length byte sequence
 *
 * @code{.c}
 * // Required INT64 column
 * carquet_schema_add_column(schema, "id", CARQUET_PHYSICAL_INT64,
 *                           NULL, CARQUET_REPETITION_REQUIRED, 0);
 *
 * // Optional string column
 * carquet_schema_add_column(schema, "name", CARQUET_PHYSICAL_BYTE_ARRAY,
 *                           NULL, CARQUET_REPETITION_OPTIONAL, 0);
 *
 * // Fixed-length UUID column
 * carquet_schema_add_column(schema, "uuid", CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY,
 *                           NULL, CARQUET_REPETITION_REQUIRED, 16);
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 2)
carquet_status_t carquet_schema_add_column(
    carquet_schema_t* schema,
    const char* name,
    carquet_physical_type_t physical_type,
    const carquet_logical_type_t* logical_type,
    carquet_field_repetition_t repetition,
    int32_t type_length);

/**
 * @brief Add a group (struct) to the schema for nested structures.
 *
 * Groups are containers for other columns or groups, enabling nested schemas.
 * Use the returned index as the parent_index when adding child elements.
 *
 * @param[in,out] schema Target schema
 * @param[in] name Group name
 * @param[in] repetition Field repetition level
 * @param[in] parent_index Parent group index (0 for root level)
 * @return Index of new group (>= 0), or -1 on error
 *
 * @note Thread-safe: No
 *
 * @code{.c}
 * // Create nested schema: { address: { street: string, city: string } }
 * int32_t address_idx = carquet_schema_add_group(schema, "address",
 *                                                 CARQUET_REPETITION_OPTIONAL, 0);
 * carquet_schema_add_column(schema, "street", CARQUET_PHYSICAL_BYTE_ARRAY,
 *                           NULL, CARQUET_REPETITION_REQUIRED, address_idx);
 * carquet_schema_add_column(schema, "city", CARQUET_PHYSICAL_BYTE_ARRAY,
 *                           NULL, CARQUET_REPETITION_REQUIRED, address_idx);
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 2)
int32_t carquet_schema_add_group(
    carquet_schema_t* schema,
    const char* name,
    carquet_field_repetition_t repetition,
    int32_t parent_index);

/**
 * @brief Get the number of leaf columns in the schema.
 *
 * Returns the count of primitive columns (not including groups).
 * This corresponds to the number of column chunks in each row group.
 *
 * @param[in] schema Schema to query
 * @return Number of leaf columns
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
int32_t carquet_schema_num_columns(const carquet_schema_t* schema);

/**
 * @brief Get the total number of schema elements (columns + groups).
 *
 * @param[in] schema Schema to query
 * @return Total number of schema elements
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
int32_t carquet_schema_num_elements(const carquet_schema_t* schema);

/**
 * @brief Get a schema element by index.
 *
 * @param[in] schema Schema to query
 * @param[in] index Element index (0 to num_elements - 1)
 * @return Schema node, or NULL if index is invalid
 *
 * @note Thread-safe: Yes (read-only)
 * @note The returned pointer is valid until the schema is freed.
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
const carquet_schema_node_t* carquet_schema_get_element(
    const carquet_schema_t* schema,
    int32_t index);

/**
 * @brief Find a column by name.
 *
 * Searches for a column with the given name. For nested schemas, use
 * dot-separated paths (e.g., "address.street").
 *
 * @param[in] schema Schema to search
 * @param[in] name Column name or path
 * @return Column index (>= 0), or -1 if not found
 *
 * @note Thread-safe: Yes (read-only)
 *
 * @code{.c}
 * int32_t idx = carquet_schema_find_column(schema, "address.city");
 * if (idx >= 0) {
 *     printf("Found column at index %d\n", idx);
 * }
 * @endcode
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1, 2)
int32_t carquet_schema_find_column(
    const carquet_schema_t* schema,
    const char* name);

/* ============================================================================
 * Schema Node Accessors
 * ============================================================================
 *
 * Functions for querying properties of individual schema elements.
 */

/**
 * @brief Get the name of a schema node.
 *
 * @param[in] node Schema node to query
 * @return Node name (never NULL)
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1) CARQUET_RETURNS_NONNULL
const char* carquet_schema_node_name(const carquet_schema_node_t* node);

/**
 * @brief Check if a schema node is a leaf (column) or group.
 *
 * @param[in] node Schema node to query
 * @return true if the node is a leaf column, false if it's a group
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
bool carquet_schema_node_is_leaf(const carquet_schema_node_t* node);

/**
 * @brief Get the physical type of a leaf node.
 *
 * @param[in] node Schema node (must be a leaf)
 * @return Physical type
 *
 * @note Thread-safe: Yes (read-only)
 * @warning Behavior is undefined if called on a group node.
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
carquet_physical_type_t carquet_schema_node_physical_type(
    const carquet_schema_node_t* node);

/**
 * @brief Get the logical type annotation of a node.
 *
 * @param[in] node Schema node to query
 * @return Logical type, or NULL if none
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
const carquet_logical_type_t* carquet_schema_node_logical_type(
    const carquet_schema_node_t* node);

/**
 * @brief Get the repetition level of a node.
 *
 * @param[in] node Schema node to query
 * @return Field repetition (REQUIRED, OPTIONAL, or REPEATED)
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
carquet_field_repetition_t carquet_schema_node_repetition(
    const carquet_schema_node_t* node);

/**
 * @brief Get the maximum definition level for a column.
 *
 * The definition level indicates how many optional/repeated ancestors
 * are defined for a value. Used for reconstructing nested structures.
 *
 * @param[in] node Schema node (must be a leaf)
 * @return Maximum definition level
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
int16_t carquet_schema_node_max_def_level(const carquet_schema_node_t* node);

/**
 * @brief Get the maximum repetition level for a column.
 *
 * The repetition level indicates which repeated ancestor started a new
 * list element. Used for reconstructing nested repeated structures.
 *
 * @param[in] node Schema node (must be a leaf)
 * @return Maximum repetition level
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
int16_t carquet_schema_node_max_rep_level(const carquet_schema_node_t* node);

/* ============================================================================
 * Reader API
 * ============================================================================
 *
 * The reader API provides access to Parquet file data. There are two levels:
 *
 * 1. Low-level API: Direct column reader access for maximum control
 * 2. High-level API: Batch reader for efficient columnar processing
 *
 * Reader Lifecycle:
 * 1. Open file with carquet_reader_open()
 * 2. Query metadata (schema, row counts, statistics)
 * 3. Read data using column readers or batch reader
 * 4. Close with carquet_reader_close()
 */

/**
 * @brief Configuration options for file reading.
 */
typedef struct carquet_reader_options {
    /**
     * @brief Use memory-mapped I/O.
     *
     * When enabled, the file is memory-mapped rather than read into buffers.
     * This can improve performance for large files by letting the OS handle
     * paging and caching.
     *
     * Default: false
     */
    bool use_mmap;

    /**
     * @brief Verify page checksums (CRC32).
     *
     * When enabled, CRC32 checksums are verified for each data page.
     * This adds overhead but ensures data integrity.
     *
     * Default: true
     */
    bool verify_checksums;

    /**
     * @brief Read buffer size in bytes.
     *
     * Size of internal buffers for reading file data. Larger buffers
     * can improve throughput at the cost of memory usage.
     *
     * Default: 65536 (64 KB)
     */
    size_t buffer_size;

    /**
     * @brief Number of threads for parallel decompression.
     *
     * Set to 0 for automatic detection (uses number of CPU cores).
     * Set to 1 to disable parallel decompression.
     *
     * Default: 0 (auto)
     */
    int32_t num_threads;
} carquet_reader_options_t;

/**
 * @brief Initialize reader options with default values.
 *
 * @param[out] options Options structure to initialize
 *
 * @note Thread-safe: Yes
 */
CARQUET_API CARQUET_NONNULL(1)
void carquet_reader_options_init(carquet_reader_options_t* options);

/**
 * @brief Open a Parquet file for reading.
 *
 * Opens the specified file and reads its metadata. The file must be a valid
 * Parquet file with the "PAR1" magic bytes at the beginning and end.
 *
 * @param[in] path File path (must be null-terminated)
 * @param[in] options Reader options (may be NULL for defaults)
 * @param[out] error Error information (may be NULL)
 * @return Reader handle, or NULL on error
 *
 * @note Thread-safe: Yes
 * @note The returned reader must be closed with carquet_reader_close().
 *
 * @code{.c}
 * carquet_error_t err = CARQUET_ERROR_INIT;
 * carquet_reader_t* reader = carquet_reader_open("data.parquet", NULL, &err);
 * if (!reader) {
 *     char buf[512];
 *     carquet_error_format(&err, buf, sizeof(buf));
 *     fprintf(stderr, "Failed to open file: %s\n", buf);
 *     return 1;
 * }
 * // Use reader...
 * carquet_reader_close(reader);
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1)
carquet_reader_t* carquet_reader_open(
    const char* path,
    const carquet_reader_options_t* options,
    carquet_error_t* error);

/**
 * @brief Open a Parquet file from a FILE handle.
 *
 * The FILE handle must be opened in binary read mode ("rb") and positioned
 * at the beginning of the Parquet data. The handle must remain valid and
 * must not be modified while the reader is in use.
 *
 * @param[in] file FILE handle (must be opened in binary read mode)
 * @param[in] options Reader options (may be NULL)
 * @param[out] error Error information (may be NULL)
 * @return Reader handle, or NULL on error
 *
 * @note Thread-safe: Yes
 * @note The caller retains ownership of the FILE handle and must close it
 *       after closing the reader.
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1)
carquet_reader_t* carquet_reader_open_file(
    FILE* file,
    const carquet_reader_options_t* options,
    carquet_error_t* error);

/**
 * @brief Open a Parquet file from a memory buffer.
 *
 * Reads Parquet data directly from memory. This is useful for:
 * - Embedded resources
 * - Network-received data
 * - Memory-mapped files from external sources
 *
 * @param[in] buffer Pointer to Parquet data
 * @param[in] size Size of buffer in bytes
 * @param[in] options Reader options (may be NULL)
 * @param[out] error Error information (may be NULL)
 * @return Reader handle, or NULL on error
 *
 * @note Thread-safe: Yes
 * @warning The buffer must remain valid and unmodified while the reader is in use.
 *
 * @code{.c}
 * // Read from embedded resource
 * extern const unsigned char parquet_data[];
 * extern const size_t parquet_data_size;
 *
 * carquet_reader_t* reader = carquet_reader_open_buffer(
 *     parquet_data, parquet_data_size, NULL, NULL);
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1)
carquet_reader_t* carquet_reader_open_buffer(
    const void* buffer,
    size_t size,
    const carquet_reader_options_t* options,
    carquet_error_t* error);

/**
 * @brief Close a reader and release all resources.
 *
 * Closes the file (if opened by carquet_reader_open) and frees all memory
 * associated with the reader. After calling this function, the reader
 * handle is invalid and must not be used.
 *
 * @param[in] reader Reader to close (may be NULL)
 *
 * @note Thread-safe: Yes (for different reader instances)
 * @note Safe to call with NULL (no-op)
 */
CARQUET_API
void carquet_reader_close(carquet_reader_t* reader);

/**
 * @brief Get the file schema.
 *
 * Returns the schema describing the structure of the Parquet file.
 * The returned pointer is valid until the reader is closed.
 *
 * @param[in] reader File reader
 * @return Schema handle (never NULL for valid reader)
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
const carquet_schema_t* carquet_reader_schema(const carquet_reader_t* reader);

/**
 * @brief Get the total number of rows in the file.
 *
 * @param[in] reader File reader
 * @return Total row count across all row groups
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
int64_t carquet_reader_num_rows(const carquet_reader_t* reader);

/**
 * @brief Get the number of row groups in the file.
 *
 * Row groups are independent units of data that can be read in parallel.
 * Each row group contains a subset of the total rows.
 *
 * @param[in] reader File reader
 * @return Number of row groups
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
int32_t carquet_reader_num_row_groups(const carquet_reader_t* reader);

/**
 * @brief Get the number of columns in the file.
 *
 * @param[in] reader File reader
 * @return Number of leaf columns
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
int32_t carquet_reader_num_columns(const carquet_reader_t* reader);

/**
 * @brief Check if reader is using memory-mapped I/O.
 *
 * When mmap is enabled, the reader can provide zero-copy access to data
 * for uncompressed columns with PLAIN encoding.
 *
 * @param[in] reader File reader
 * @return true if mmap is active, false otherwise
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
bool carquet_reader_is_mmap(const carquet_reader_t* reader);

/**
 * @brief Check if zero-copy reading is possible for a column.
 *
 * Zero-copy requires:
 * - Memory-mapped I/O enabled
 * - Uncompressed data (no compression codec)
 * - PLAIN encoding
 * - Fixed-size physical type (INT32, INT64, FLOAT, DOUBLE, INT96, FIXED_LEN_BYTE_ARRAY)
 * - No definition levels (REQUIRED column)
 *
 * @param[in] reader File reader
 * @param[in] row_group_index Row group index
 * @param[in] column_index Column index
 * @return true if zero-copy is possible, false otherwise
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
bool carquet_reader_can_zero_copy(
    const carquet_reader_t* reader,
    int32_t row_group_index,
    int32_t column_index);

/**
 * @brief Metadata for a row group.
 */
typedef struct carquet_row_group_metadata {
    int64_t num_rows;               /**< Number of rows in this row group */
    int64_t total_byte_size;        /**< Total uncompressed size in bytes */
    int64_t total_compressed_size;  /**< Total compressed size in bytes */
} carquet_row_group_metadata_t;

/**
 * @brief Get metadata for a specific row group.
 *
 * @param[in] reader File reader
 * @param[in] row_group_index Row group index (0 to num_row_groups - 1)
 * @param[out] metadata Output metadata structure
 * @return CARQUET_OK on success, error code on failure
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 3)
carquet_status_t carquet_reader_row_group_metadata(
    const carquet_reader_t* reader,
    int32_t row_group_index,
    carquet_row_group_metadata_t* metadata);

/**
 * @brief Get a column reader for a specific row group and column.
 *
 * Creates a reader for streaming values from a single column within a
 * single row group. The column reader must be freed with
 * carquet_column_reader_free() when no longer needed.
 *
 * @param[in] reader File reader
 * @param[in] row_group_index Row group index
 * @param[in] column_index Column index
 * @param[out] error Error information (may be NULL)
 * @return Column reader, or NULL on error
 *
 * @note Thread-safe: Yes (multiple column readers can be used concurrently)
 *
 * @code{.c}
 * carquet_column_reader_t* col = carquet_reader_get_column(reader, 0, 0, &err);
 * if (col) {
 *     int64_t values[1024];
 *     int64_t count;
 *     while ((count = carquet_column_read_batch(col, values, 1024, NULL, NULL)) > 0) {
 *         // Process values...
 *     }
 *     carquet_column_reader_free(col);
 * }
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1)
carquet_column_reader_t* carquet_reader_get_column(
    carquet_reader_t* reader,
    int32_t row_group_index,
    int32_t column_index,
    carquet_error_t* error);

/* ============================================================================
 * Column Reader API
 * ============================================================================
 *
 * The column reader provides low-level access to column data with full control
 * over definition and repetition levels for nested/nullable schemas.
 */

/**
 * @brief Read a batch of values from a column.
 *
 * Reads up to max_values from the column into the output buffer. For nullable
 * columns, definition levels indicate which values are null. For repeated
 * columns, repetition levels indicate list boundaries.
 *
 * @param[in] reader Column reader
 * @param[out] values Output buffer for values (sized for physical type)
 * @param[in] max_values Maximum number of values to read
 * @param[out] def_levels Definition levels buffer (may be NULL if not needed)
 * @param[out] rep_levels Repetition levels buffer (may be NULL if not needed)
 * @return Number of values read (0 at end of column), or negative on error
 *
 * @note Thread-safe: No (single column reader is not thread-safe)
 *
 * @par Value Buffer Sizing
 * The values buffer must be sized appropriately for the column's physical type:
 * - BOOLEAN: uint8_t (1 byte per value)
 * - INT32: int32_t (4 bytes per value)
 * - INT64: int64_t (8 bytes per value)
 * - FLOAT: float (4 bytes per value)
 * - DOUBLE: double (8 bytes per value)
 * - BYTE_ARRAY: carquet_byte_array_t (pointer + length)
 * - FIXED_LEN_BYTE_ARRAY: uint8_t[type_length]
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1)
int64_t carquet_column_read_batch(
    carquet_column_reader_t* reader,
    void* values,
    int64_t max_values,
    int16_t* def_levels,
    int16_t* rep_levels);

/**
 * @brief Skip values in a column without reading them.
 *
 * Efficiently skips over values in the column stream. This is faster than
 * reading and discarding values.
 *
 * @param[in] reader Column reader
 * @param[in] num_values Number of values to skip
 * @return Number of values actually skipped
 *
 * @note Thread-safe: No
 */
CARQUET_API CARQUET_NONNULL(1)
int64_t carquet_column_skip(
    carquet_column_reader_t* reader,
    int64_t num_values);

/**
 * @brief Check if there are more values to read.
 *
 * @param[in] reader Column reader
 * @return true if more values are available
 *
 * @note Thread-safe: No
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
bool carquet_column_has_next(const carquet_column_reader_t* reader);

/**
 * @brief Get the number of remaining values in the column.
 *
 * @param[in] reader Column reader
 * @return Number of values remaining
 *
 * @note Thread-safe: No
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
int64_t carquet_column_remaining(const carquet_column_reader_t* reader);

/**
 * @brief Free a column reader.
 *
 * @param[in] reader Column reader to free (may be NULL)
 *
 * @note Thread-safe: Yes (for different reader instances)
 */
CARQUET_API
void carquet_column_reader_free(carquet_column_reader_t* reader);

/* ============================================================================
 * Batch Reader API
 * ============================================================================
 *
 * The batch reader provides a high-level, efficient interface for reading
 * Parquet files. It supports:
 *
 * - Column projection (read only needed columns)
 * - Automatic batch sizing
 * - Parallel I/O (optional)
 *
 * This is the recommended API for most use cases.
 */

/**
 * @brief Batch reader configuration.
 */
typedef struct carquet_batch_reader_config {
    /**
     * @brief Number of rows per batch.
     *
     * Larger batches reduce overhead but use more memory.
     *
     * Default: 65536 (64K rows)
     */
    int32_t batch_size;

    /**
     * @brief Number of threads for parallel column reading.
     *
     * Set to 0 for automatic detection, 1 to disable parallelism.
     *
     * Default: 0 (auto)
     */
    int32_t num_threads;

    /**
     * @brief Use memory-mapped I/O.
     *
     * Default: false
     */
    bool use_mmap;

    /**
     * @brief Column projection by index.
     *
     * Array of column indices to read. If NULL, all columns are read.
     * Takes precedence over column_names if both are specified.
     */
    const int32_t* column_indices;

    /**
     * @brief Number of columns in column_indices array.
     */
    int32_t num_columns;

    /**
     * @brief Column projection by name.
     *
     * Array of column names to read. If NULL, all columns are read.
     * Ignored if column_indices is specified.
     */
    const char* const* column_names;

    /**
     * @brief Number of column names.
     */
    int32_t num_column_names;
} carquet_batch_reader_config_t;

/**
 * @brief Initialize batch reader configuration with defaults.
 *
 * @param[out] config Configuration to initialize
 *
 * @note Thread-safe: Yes
 */
CARQUET_API CARQUET_NONNULL(1)
void carquet_batch_reader_config_init(carquet_batch_reader_config_t* config);

/**
 * @brief Create a batch reader for efficient columnar reading.
 *
 * Creates a batch reader that iterates over the file in row batches.
 * Use column projection to read only the columns you need.
 *
 * @param[in] reader File reader
 * @param[in] config Batch reader configuration (may be NULL for defaults)
 * @param[out] error Error information (may be NULL)
 * @return Batch reader, or NULL on error
 *
 * @note Thread-safe: Yes
 *
 * @code{.c}
 * carquet_batch_reader_config_t config;
 * carquet_batch_reader_config_init(&config);
 *
 * // Project only two columns
 * const char* cols[] = {"id", "timestamp"};
 * config.column_names = cols;
 * config.num_column_names = 2;
 *
 * carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(
 *     reader, &config, &err);
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1)
carquet_batch_reader_t* carquet_batch_reader_create(
    carquet_reader_t* reader,
    const carquet_batch_reader_config_t* config,
    carquet_error_t* error);

/**
 * @brief Read the next batch of rows.
 *
 * Reads the next batch of rows from the file. The batch must be freed
 * with carquet_row_batch_free() when done.
 *
 * @param[in] batch_reader Batch reader
 * @param[out] batch Output batch (set to NULL when no more data)
 * @return CARQUET_OK on success, CARQUET_ERROR_END_OF_DATA when finished
 *
 * @note Thread-safe: No
 *
 * @code{.c}
 * carquet_row_batch_t* batch = NULL;
 * while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
 *     // Process batch...
 *     carquet_row_batch_free(batch);
 *     batch = NULL;
 * }
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 2)
carquet_status_t carquet_batch_reader_next(
    carquet_batch_reader_t* batch_reader,
    carquet_row_batch_t** batch);

/**
 * @brief Free a batch reader.
 *
 * @param[in] batch_reader Batch reader to free (may be NULL)
 *
 * @note Thread-safe: Yes (for different instances)
 */
CARQUET_API
void carquet_batch_reader_free(carquet_batch_reader_t* batch_reader);

/**
 * @brief Get the number of rows in a batch.
 *
 * @param[in] batch Row batch
 * @return Number of rows
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
int64_t carquet_row_batch_num_rows(const carquet_row_batch_t* batch);

/**
 * @brief Get the number of columns in a batch.
 *
 * This is the number of projected columns, not the total file columns.
 *
 * @param[in] batch Row batch
 * @return Number of columns in the batch
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_PURE CARQUET_NONNULL(1)
int32_t carquet_row_batch_num_columns(const carquet_row_batch_t* batch);

/**
 * @brief Get column data from a batch.
 *
 * Returns pointers to the raw column data within the batch. The pointers
 * remain valid until the batch is freed.
 *
 * @param[in] batch Row batch
 * @param[in] column_index Column index within the batch (0 to num_columns-1)
 * @param[out] data Pointer to column data (type depends on physical type)
 * @param[out] null_bitmap Null bitmap (1 bit per value, set = not null) or NULL
 * @param[out] num_values Number of values in the column
 * @return CARQUET_OK on success
 *
 * @note Thread-safe: Yes (read-only)
 *
 * @par Null Bitmap Format
 * The null bitmap uses 1 bit per value, with bit i set if value i is NOT null.
 * Use the following to check if value i is null:
 * @code{.c}
 * bool is_null = null_bitmap && !(null_bitmap[i / 8] & (1 << (i % 8)));
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 3, 4, 5)
carquet_status_t carquet_row_batch_column(
    const carquet_row_batch_t* batch,
    int32_t column_index,
    const void** data,
    const uint8_t** null_bitmap,
    int64_t* num_values);

/**
 * @brief Free a row batch.
 *
 * @param[in] batch Batch to free (may be NULL)
 *
 * @note Thread-safe: Yes (for different instances)
 */
CARQUET_API
void carquet_row_batch_free(carquet_row_batch_t* batch);

/* ============================================================================
 * Row Group Statistics API
 * ============================================================================
 *
 * Statistics enable predicate pushdown - skipping row groups that cannot
 * contain matching data based on min/max values.
 */

/**
 * @brief Column statistics for a row group.
 */
typedef struct carquet_column_statistics {
    bool has_min_max;           /**< Min/max values are available */
    bool has_null_count;        /**< Null count is available */
    bool has_distinct_count;    /**< Distinct count is available */

    int64_t null_count;         /**< Number of null values */
    int64_t distinct_count;     /**< Approximate distinct value count */
    int64_t num_values;         /**< Total number of values (including nulls) */

    const void* min_value;      /**< Minimum value (type depends on column) */
    const void* max_value;      /**< Maximum value (type depends on column) */
    int32_t min_value_size;     /**< Size of min_value in bytes */
    int32_t max_value_size;     /**< Size of max_value in bytes */
} carquet_column_statistics_t;

/**
 * @brief Get statistics for a column in a row group.
 *
 * @param[in] reader File reader
 * @param[in] row_group_index Row group index
 * @param[in] column_index Column index
 * @param[out] stats Output statistics
 * @return CARQUET_OK on success
 *
 * @note Thread-safe: Yes (read-only)
 * @note Statistics may not be available for all columns/row groups.
 *       Check the has_* flags before using values.
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 4)
carquet_status_t carquet_reader_column_statistics(
    const carquet_reader_t* reader,
    int32_t row_group_index,
    int32_t column_index,
    carquet_column_statistics_t* stats);

/**
 * @brief Comparison operators for predicate pushdown.
 */
typedef enum carquet_compare_op {
    CARQUET_COMPARE_EQ,     /**< Equal (==) */
    CARQUET_COMPARE_NE,     /**< Not equal (!=) */
    CARQUET_COMPARE_LT,     /**< Less than (<) */
    CARQUET_COMPARE_LE,     /**< Less than or equal (<=) */
    CARQUET_COMPARE_GT,     /**< Greater than (>) */
    CARQUET_COMPARE_GE      /**< Greater than or equal (>=) */
} carquet_compare_op_t;

/**
 * @brief Check if a row group might contain values matching a predicate.
 *
 * Uses min/max statistics to determine if a row group can be safely skipped.
 * A return of might_match=true does not guarantee matches exist, only that
 * they cannot be ruled out based on statistics.
 *
 * @param[in] reader File reader
 * @param[in] row_group_index Row group index
 * @param[in] column_index Column index
 * @param[in] op Comparison operator
 * @param[in] value Value to compare against
 * @param[in] value_size Size of value in bytes
 * @param[out] might_match Set to true if row group might contain matches
 * @return CARQUET_OK on success
 *
 * @note Thread-safe: Yes (read-only)
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 5, 7)
carquet_status_t carquet_reader_row_group_matches(
    const carquet_reader_t* reader,
    int32_t row_group_index,
    int32_t column_index,
    carquet_compare_op_t op,
    const void* value,
    int32_t value_size,
    bool* might_match);

/**
 * @brief Filter row groups based on a predicate.
 *
 * Returns indices of row groups that might contain matching data.
 * Use this to skip reading row groups that cannot match a query.
 *
 * @param[in] reader File reader
 * @param[in] column_index Column index
 * @param[in] op Comparison operator
 * @param[in] value Value to compare against
 * @param[in] value_size Size of value in bytes
 * @param[out] matching_indices Output array of matching row group indices
 * @param[in] max_indices Maximum number of indices to return
 * @return Number of matching row groups, or negative on error
 *
 * @note Thread-safe: Yes (read-only)
 *
 * @code{.c}
 * int32_t threshold = 1000;
 * int32_t matches[100];
 * int32_t count = carquet_reader_filter_row_groups(
 *     reader, 0, CARQUET_COMPARE_GT, &threshold, sizeof(threshold), matches, 100);
 *
 * printf("Found %d row groups with values > 1000\n", count);
 * for (int i = 0; i < count; i++) {
 *     // Read only matching row groups...
 * }
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 4, 6)
int32_t carquet_reader_filter_row_groups(
    const carquet_reader_t* reader,
    int32_t column_index,
    carquet_compare_op_t op,
    const void* value,
    int32_t value_size,
    int32_t* matching_indices,
    int32_t max_indices);

/* ============================================================================
 * Writer API
 * ============================================================================
 *
 * The writer API creates Parquet files with configurable compression,
 * encoding, and metadata options.
 *
 * Writer Lifecycle:
 * 1. Create schema
 * 2. Configure writer options
 * 3. Create writer with carquet_writer_create()
 * 4. Write data with carquet_writer_write_batch()
 * 5. Optionally start new row groups with carquet_writer_new_row_group()
 * 6. Close with carquet_writer_close()
 *
 * Important: All columns must be written the same number of rows before
 * closing or starting a new row group.
 */

/**
 * @brief Writer configuration options.
 */
typedef struct carquet_writer_options {
    /**
     * @brief Compression codec for all columns.
     *
     * Default: CARQUET_COMPRESSION_SNAPPY
     */
    carquet_compression_t compression;

    /**
     * @brief Compression level (codec-specific).
     *
     * - ZSTD: 1-22 (default: 3)
     * - GZIP: 1-9 (default: 6)
     * - Others: ignored
     *
     * Default: 0 (use codec default)
     */
    int32_t compression_level;

    /**
     * @brief Target row group size in bytes.
     *
     * Row groups are automatically flushed when this size is exceeded.
     *
     * Default: 128MB
     */
    int64_t row_group_size;

    /**
     * @brief Target page size in bytes.
     *
     * Default: 1MB
     */
    int64_t page_size;

    /**
     * @brief Write column statistics (min/max values).
     *
     * Statistics enable predicate pushdown when reading.
     *
     * Default: true
     */
    bool write_statistics;

    /**
     * @brief Write page index for efficient page skipping.
     *
     * Default: false
     */
    bool write_page_index;

    /**
     * @brief Write bloom filters for membership testing.
     *
     * Default: false
     */
    bool write_bloom_filters;

    /**
     * @brief Dictionary encoding mode.
     *
     * Default: CARQUET_ENCODING_PLAIN_DICTIONARY
     */
    carquet_encoding_t dictionary_encoding;

    /**
     * @brief Maximum dictionary page size.
     *
     * Dictionary encoding is disabled for columns exceeding this size.
     *
     * Default: 1MB
     */
    int64_t dictionary_page_size;

    /**
     * @brief Creator identification string.
     *
     * Stored in file metadata.
     *
     * Default: "Carquet"
     */
    const char* created_by;
} carquet_writer_options_t;

/**
 * @brief Initialize writer options with default values.
 *
 * @param[out] options Options to initialize
 *
 * @note Thread-safe: Yes
 */
CARQUET_API CARQUET_NONNULL(1)
void carquet_writer_options_init(carquet_writer_options_t* options);

/**
 * @brief Create a new Parquet file for writing.
 *
 * Creates a new file and prepares it for writing. The schema defines the
 * structure of the data to be written.
 *
 * @param[in] path Output file path
 * @param[in] schema File schema (copied, caller retains ownership)
 * @param[in] options Writer options (may be NULL for defaults)
 * @param[out] error Error information (may be NULL)
 * @return Writer handle, or NULL on error
 *
 * @note Thread-safe: Yes
 *
 * @code{.c}
 * carquet_writer_options_t opts;
 * carquet_writer_options_init(&opts);
 * opts.compression = CARQUET_COMPRESSION_ZSTD;
 *
 * carquet_writer_t* writer = carquet_writer_create(
 *     "output.parquet", schema, &opts, &err);
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 2)
carquet_writer_t* carquet_writer_create(
    const char* path,
    const carquet_schema_t* schema,
    const carquet_writer_options_t* options,
    carquet_error_t* error);

/**
 * @brief Create a writer to a FILE handle.
 *
 * @param[in] file FILE handle (must be opened in binary write mode)
 * @param[in] schema File schema
 * @param[in] options Writer options (may be NULL)
 * @param[out] error Error information (may be NULL)
 * @return Writer handle, or NULL on error
 *
 * @note Thread-safe: Yes
 * @note Caller retains ownership of FILE handle.
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 2)
carquet_writer_t* carquet_writer_create_file(
    FILE* file,
    const carquet_schema_t* schema,
    const carquet_writer_options_t* options,
    carquet_error_t* error);

/**
 * @brief Write a batch of values to a column.
 *
 * Writes values to the specified column. All columns must be written the
 * same number of rows before closing or starting a new row group.
 *
 * @param[in] writer File writer
 * @param[in] column_index Column index
 * @param[in] values Input values (type must match column physical type)
 * @param[in] num_values Number of values to write
 * @param[in] def_levels Definition levels (NULL if all values defined)
 * @param[in] rep_levels Repetition levels (NULL if no repetition)
 * @return CARQUET_OK on success
 *
 * @note Thread-safe: No
 *
 * @par Writing Nullable Columns
 * For nullable columns (OPTIONAL repetition), provide definition levels:
 * - def_level = max_def_level: value is present
 * - def_level < max_def_level: value is null
 *
 * @code{.c}
 * // Write non-nullable column
 * int64_t ids[] = {1, 2, 3, 4, 5};
 * carquet_writer_write_batch(writer, 0, ids, 5, NULL, NULL);
 *
 * // Write nullable column with some nulls
 * double values[] = {1.1, 0.0, 3.3, 0.0, 5.5};  // 0.0 will be null
 * int16_t def_levels[] = {1, 0, 1, 0, 1};       // 0 = null, 1 = present
 * carquet_writer_write_batch(writer, 1, values, 5, def_levels, NULL);
 * @endcode
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 3)
carquet_status_t carquet_writer_write_batch(
    carquet_writer_t* writer,
    int32_t column_index,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels);

/**
 * @brief Start a new row group.
 *
 * Flushes the current row group and starts a new one. This is called
 * automatically when the row group size exceeds the configured limit,
 * but can be called explicitly for finer control.
 *
 * @param[in] writer File writer
 * @return CARQUET_OK on success
 *
 * @note Thread-safe: No
 * @warning All columns must have the same number of rows when this is called.
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1)
carquet_status_t carquet_writer_new_row_group(carquet_writer_t* writer);

/**
 * @brief Close the writer and finalize the file.
 *
 * Writes any buffered data, the file footer, and closes the file.
 * The writer handle becomes invalid after this call.
 *
 * @param[in] writer Writer to close
 * @return CARQUET_OK on success
 *
 * @note Thread-safe: No
 * @warning All columns must have the same number of rows when this is called.
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1)
carquet_status_t carquet_writer_close(carquet_writer_t* writer);

/**
 * @brief Abort writing and clean up without finalizing the file.
 *
 * Closes the writer and releases resources without writing a valid
 * Parquet footer. The resulting file will be invalid/incomplete.
 *
 * @param[in] writer Writer to abort (may be NULL)
 *
 * @note Thread-safe: No
 */
CARQUET_API
void carquet_writer_abort(carquet_writer_t* writer);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief File information from metadata (without full parsing).
 */
typedef struct carquet_file_info {
    int64_t file_size;          /**< Total file size in bytes */
    int64_t num_rows;           /**< Total number of rows */
    int32_t num_row_groups;     /**< Number of row groups */
    int32_t num_columns;        /**< Number of columns */
    int32_t version;            /**< Parquet format version */
    const char* created_by;     /**< Creator identification (may be NULL) */
} carquet_file_info_t;

/**
 * @brief Get basic file information without fully opening the file.
 *
 * Reads only the file footer to extract basic metadata.
 * Faster than opening a full reader when only metadata is needed.
 *
 * @param[in] path File path
 * @param[out] info Output file information
 * @param[out] error Error information (may be NULL)
 * @return CARQUET_OK on success
 *
 * @note Thread-safe: Yes
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1, 2)
carquet_status_t carquet_get_file_info(
    const char* path,
    carquet_file_info_t* info,
    carquet_error_t* error);

/**
 * @brief Validate a Parquet file structure.
 *
 * Performs structural validation of the file:
 * - Checks magic bytes
 * - Validates footer
 * - Optionally verifies page checksums
 *
 * @param[in] path File path
 * @param[out] error Detailed error information (may be NULL)
 * @return CARQUET_OK if file is valid
 *
 * @note Thread-safe: Yes
 */
CARQUET_API CARQUET_WARN_UNUSED_RESULT CARQUET_NONNULL(1)
carquet_status_t carquet_validate_file(
    const char* path,
    carquet_error_t* error);

/* ============================================================================
 * C++ Compatibility - End
 * ============================================================================ */

#ifdef __cplusplus
}
#endif

#endif /* CARQUET_H */
