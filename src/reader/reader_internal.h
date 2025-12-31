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

#ifdef __cplusplus
extern "C" {
#endif

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

    /* Memory-mapped or buffered data */
    const uint8_t* mmap_data;
    size_t file_size;

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

#ifdef __cplusplus
}
#endif

#endif /* CARQUET_READER_INTERNAL_H */
