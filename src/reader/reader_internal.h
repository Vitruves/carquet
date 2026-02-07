/**
 * @file reader_internal.h
 * @brief Internal reader structures
 *
 * This header defines internal structures that are shared between
 * reader components but not exposed in the public API.
 */

#ifndef CARQUET_READER_INTERNAL_H
#define CARQUET_READER_INTERNAL_H

#include <carquet/carquet.h>
#include "thrift/parquet_types.h"
#include "core/arena.h"
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Memory Mapping Types
 * ============================================================================
 */

/**
 * Indicates whether data is owned (malloc'd) or a view (mmap pointer).
 */
typedef enum carquet_data_ownership {
    CARQUET_DATA_OWNED = 0,    /* Data is malloc'd, caller must free */
    CARQUET_DATA_VIEW = 1,     /* Data is view into mmap, do NOT free */
} carquet_data_ownership_t;

/**
 * Platform-specific memory mapping handle.
 */
typedef struct carquet_mmap_info {
    uint8_t* data;
    size_t size;
#ifdef _WIN32
    HANDLE file_handle;
    HANDLE mapping_handle;
#else
    int fd;
#endif
    bool is_valid;
} carquet_mmap_info_t;

/* ============================================================================
 * Internal Schema Structure
 * ============================================================================
 */

struct carquet_schema {
    carquet_arena_t arena;
    parquet_schema_element_t* elements;
    int32_t num_elements;
    int32_t capacity;           /* Capacity of elements/leaf arrays */

    /* Computed fields */
    int32_t* leaf_indices;      /* Map leaf index -> schema element index */
    int32_t num_leaves;         /* Number of leaf columns */
    int16_t* max_def_levels;    /* Max definition level per leaf */
    int16_t* max_rep_levels;    /* Max repetition level per leaf */
};

/* ============================================================================
 * Internal Reader Structure
 * ============================================================================
 */

struct carquet_reader {
    FILE* file;
    bool owns_file;

    /* Memory-mapped data */
    const uint8_t* mmap_data;
    size_t file_size;
    carquet_mmap_info_t* mmap_info;  /* Platform-specific mmap handle, NULL if not using mmap */

    /* Metadata */
    carquet_arena_t arena;
    parquet_file_metadata_t metadata;
    carquet_schema_t* schema;

    /* Options */
    carquet_reader_options_t options;

    /* State */
    bool is_open;
};

/* ============================================================================
 * Internal Column Reader Structure
 * ============================================================================
 */

struct carquet_column_reader {
    carquet_reader_t* file_reader;
    int32_t row_group_index;
    int32_t column_index;

    /* Column metadata */
    const parquet_column_chunk_t* chunk;
    const parquet_column_metadata_t* col_meta;

    /* Schema info */
    int16_t max_def_level;
    int16_t max_rep_level;
    carquet_physical_type_t type;
    int32_t type_length;

    /* Reading state */
    int64_t values_remaining;
    int64_t data_start_offset;    /* Actual offset of first data page in file */
    int64_t current_page;

    /* Page data */
    uint8_t* page_buffer;
    size_t page_buffer_size;
    size_t page_buffer_capacity;

    /* Dictionary */
    bool has_dictionary;
    uint8_t* dictionary_data;
    size_t dictionary_size;
    int32_t dictionary_count;
    uint32_t* dictionary_offsets;  /* Offset cache for O(1) BYTE_ARRAY lookup */

    /* Retained page data for BYTE_ARRAY value pointers */
    uint8_t* page_data_for_values;

    /* Current page state for partial reads */
    bool page_loaded;           /* Is a page currently loaded? */
    int32_t page_num_values;    /* Total values in current page */
    int32_t page_values_read;   /* Values already read from current page */
    int32_t page_header_size;   /* Size of current page header */
    int32_t page_compressed_size; /* Size of current page compressed data */
    uint8_t* decoded_values;    /* Buffer for decoded values from current page */
    int16_t* decoded_def_levels; /* Buffer for decoded definition levels */
    int16_t* decoded_rep_levels; /* Buffer for decoded repetition levels */
    size_t decoded_capacity;    /* Capacity of decoded buffers */
    carquet_data_ownership_t decoded_ownership; /* OWNED or VIEW (mmap) */

    /* Reusable buffers to reduce allocations */
    uint32_t* indices_buffer;   /* Reusable buffer for dictionary indices */
    size_t indices_capacity;    /* Capacity of indices buffer */
};

/* ============================================================================
 * Internal Functions
 * ============================================================================
 */

/**
 * Build schema structure from parsed metadata.
 */
carquet_schema_t* build_schema(
    carquet_arena_t* arena,
    const parquet_file_metadata_t* metadata,
    carquet_error_t* error);

/**
 * Open file with memory mapping.
 * Returns mmap_info on success, NULL on failure (fallback to fread).
 */
carquet_mmap_info_t* carquet_mmap_open(const char* path, carquet_error_t* error);

/**
 * Close memory mapping and release resources.
 */
void carquet_mmap_close(carquet_mmap_info_t* mmap_info);

/**
 * Check if a page is eligible for zero-copy reading.
 * Requires: uncompressed, PLAIN encoding, fixed-size type.
 */
bool carquet_page_is_zero_copy_eligible(
    carquet_compression_t codec,
    carquet_encoding_t encoding,
    carquet_physical_type_t type);

#ifdef __cplusplus
}
#endif

#endif /* CARQUET_READER_INTERNAL_H */
