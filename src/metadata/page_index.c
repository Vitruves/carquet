/**
 * @file page_index.c
 * @brief Page index (ColumnIndex and OffsetIndex) implementation
 *
 * Page indexes enable predicate pushdown by storing per-page statistics.
 * - ColumnIndex: min/max values and null counts for each page
 * - OffsetIndex: file offset, compressed/uncompressed size for each page
 *
 * Reference: https://parquet.apache.org/docs/file-format/
 */

#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/arena.h"
#include "core/buffer.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * ColumnIndex Structure
 * ============================================================================
 */

typedef struct carquet_column_index {
    int32_t num_pages;

    /* Per-page null counts */
    int64_t* null_counts;

    /* Per-page min/max values (packed binary) */
    uint8_t** min_values;
    int32_t* min_value_lens;
    uint8_t** max_values;
    int32_t* max_value_lens;

    /* Per-page null page flags */
    bool* null_pages;

    /* Boundary order for efficient range queries */
    int32_t boundary_order;  /* 0=UNORDERED, 1=ASCENDING, 2=DESCENDING */
} carquet_column_index_t;

/* ============================================================================
 * OffsetIndex Structure
 * ============================================================================
 */

typedef struct carquet_page_location {
    int64_t offset;             /* File offset of page */
    int32_t compressed_size;    /* Compressed page size */
    int64_t first_row_index;    /* Index of first row in page */
} carquet_page_location_t;

typedef struct carquet_offset_index {
    int32_t num_pages;
    carquet_page_location_t* page_locations;

    /* Optional: uncompressed page sizes */
    bool has_uncompressed_sizes;
    int32_t* uncompressed_page_sizes;
} carquet_offset_index_t;

/* ============================================================================
 * Forward Declarations
 * ============================================================================
 */

typedef struct carquet_column_index_builder carquet_column_index_builder_t;
typedef struct carquet_offset_index_builder carquet_offset_index_builder_t;

void carquet_column_index_builder_destroy(carquet_column_index_builder_t* builder);
void carquet_offset_index_builder_destroy(carquet_offset_index_builder_t* builder);

/* ============================================================================
 * Column Index Builder
 * ============================================================================
 */

struct carquet_column_index_builder {
    carquet_physical_type_t type;
    int32_t type_length;

    int32_t capacity;
    int32_t num_pages;

    int64_t* null_counts;
    uint8_t** min_values;
    int32_t* min_value_lens;
    uint8_t** max_values;
    int32_t* max_value_lens;
    bool* null_pages;

    int32_t boundary_order;
};

/**
 * Create a column index builder.
 */
carquet_column_index_builder_t* carquet_column_index_builder_create(
    carquet_physical_type_t type,
    int32_t type_length) {

    carquet_column_index_builder_t* builder = calloc(1, sizeof(*builder));
    if (!builder) return NULL;

    builder->type = type;
    builder->type_length = type_length;
    builder->capacity = 16;
    builder->boundary_order = 0;  /* UNORDERED by default */

    builder->null_counts = calloc(builder->capacity, sizeof(int64_t));
    builder->min_values = calloc(builder->capacity, sizeof(uint8_t*));
    builder->min_value_lens = calloc(builder->capacity, sizeof(int32_t));
    builder->max_values = calloc(builder->capacity, sizeof(uint8_t*));
    builder->max_value_lens = calloc(builder->capacity, sizeof(int32_t));
    builder->null_pages = calloc(builder->capacity, sizeof(bool));

    if (!builder->null_counts || !builder->min_values || !builder->max_values ||
        !builder->min_value_lens || !builder->max_value_lens || !builder->null_pages) {
        carquet_column_index_builder_destroy(builder);
        return NULL;
    }

    return builder;
}

/**
 * Destroy a column index builder.
 */
void carquet_column_index_builder_destroy(carquet_column_index_builder_t* builder) {
    if (!builder) return;

    if (builder->min_values) {
        for (int32_t i = 0; i < builder->num_pages; i++) {
            free(builder->min_values[i]);
        }
        free(builder->min_values);
    }

    if (builder->max_values) {
        for (int32_t i = 0; i < builder->num_pages; i++) {
            free(builder->max_values[i]);
        }
        free(builder->max_values);
    }

    free(builder->null_counts);
    free(builder->min_value_lens);
    free(builder->max_value_lens);
    free(builder->null_pages);
    free(builder);
}

/**
 * Ensure capacity for more pages.
 */
static carquet_status_t ensure_capacity(carquet_column_index_builder_t* builder) {
    if (builder->num_pages < builder->capacity) {
        return CARQUET_OK;
    }

    int32_t new_cap = builder->capacity * 2;

    int64_t* new_null_counts = realloc(builder->null_counts, new_cap * sizeof(int64_t));
    uint8_t** new_min_values = realloc(builder->min_values, new_cap * sizeof(uint8_t*));
    int32_t* new_min_lens = realloc(builder->min_value_lens, new_cap * sizeof(int32_t));
    uint8_t** new_max_values = realloc(builder->max_values, new_cap * sizeof(uint8_t*));
    int32_t* new_max_lens = realloc(builder->max_value_lens, new_cap * sizeof(int32_t));
    bool* new_null_pages = realloc(builder->null_pages, new_cap * sizeof(bool));

    if (!new_null_counts || !new_min_values || !new_max_values ||
        !new_min_lens || !new_max_lens || !new_null_pages) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    builder->null_counts = new_null_counts;
    builder->min_values = new_min_values;
    builder->min_value_lens = new_min_lens;
    builder->max_values = new_max_values;
    builder->max_value_lens = new_max_lens;
    builder->null_pages = new_null_pages;
    builder->capacity = new_cap;

    /* Initialize new entries */
    for (int32_t i = builder->num_pages; i < new_cap; i++) {
        builder->null_counts[i] = 0;
        builder->min_values[i] = NULL;
        builder->min_value_lens[i] = 0;
        builder->max_values[i] = NULL;
        builder->max_value_lens[i] = 0;
        builder->null_pages[i] = false;
    }

    return CARQUET_OK;
}

/**
 * Add a page's statistics to the column index.
 */
carquet_status_t carquet_column_index_add_page(
    carquet_column_index_builder_t* builder,
    int64_t null_count,
    const void* min_value,
    int32_t min_value_len,
    const void* max_value,
    int32_t max_value_len,
    bool is_null_page) {

    if (!builder) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_status_t status = ensure_capacity(builder);
    if (status != CARQUET_OK) return status;

    int32_t idx = builder->num_pages;

    builder->null_counts[idx] = null_count;
    builder->null_pages[idx] = is_null_page;

    /* Copy min value */
    if (min_value && min_value_len > 0) {
        builder->min_values[idx] = malloc(min_value_len);
        if (!builder->min_values[idx]) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        memcpy(builder->min_values[idx], min_value, min_value_len);
        builder->min_value_lens[idx] = min_value_len;
    }

    /* Copy max value */
    if (max_value && max_value_len > 0) {
        builder->max_values[idx] = malloc(max_value_len);
        if (!builder->max_values[idx]) {
            free(builder->min_values[idx]);
            builder->min_values[idx] = NULL;
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        memcpy(builder->max_values[idx], max_value, max_value_len);
        builder->max_value_lens[idx] = max_value_len;
    }

    builder->num_pages++;
    return CARQUET_OK;
}

/**
 * Set boundary order for the column index.
 */
void carquet_column_index_set_boundary_order(
    carquet_column_index_builder_t* builder,
    int32_t order) {
    if (builder) {
        builder->boundary_order = order;
    }
}

/* ============================================================================
 * Offset Index Builder
 * ============================================================================
 */

struct carquet_offset_index_builder {
    int32_t capacity;
    int32_t num_pages;

    int64_t* offsets;
    int32_t* compressed_sizes;
    int64_t* first_row_indices;
    int32_t* uncompressed_sizes;  /* Optional */
    bool track_uncompressed;
};

/**
 * Create an offset index builder.
 */
carquet_offset_index_builder_t* carquet_offset_index_builder_create(
    bool track_uncompressed) {

    carquet_offset_index_builder_t* builder = calloc(1, sizeof(*builder));
    if (!builder) return NULL;

    builder->capacity = 16;
    builder->track_uncompressed = track_uncompressed;

    builder->offsets = calloc(builder->capacity, sizeof(int64_t));
    builder->compressed_sizes = calloc(builder->capacity, sizeof(int32_t));
    builder->first_row_indices = calloc(builder->capacity, sizeof(int64_t));

    if (track_uncompressed) {
        builder->uncompressed_sizes = calloc(builder->capacity, sizeof(int32_t));
    }

    if (!builder->offsets || !builder->compressed_sizes || !builder->first_row_indices ||
        (track_uncompressed && !builder->uncompressed_sizes)) {
        carquet_offset_index_builder_destroy(builder);
        return NULL;
    }

    return builder;
}

/**
 * Destroy an offset index builder.
 */
void carquet_offset_index_builder_destroy(carquet_offset_index_builder_t* builder) {
    if (!builder) return;

    free(builder->offsets);
    free(builder->compressed_sizes);
    free(builder->first_row_indices);
    free(builder->uncompressed_sizes);
    free(builder);
}

/**
 * Ensure capacity for more pages.
 */
static carquet_status_t offset_ensure_capacity(carquet_offset_index_builder_t* builder) {
    if (builder->num_pages < builder->capacity) {
        return CARQUET_OK;
    }

    int32_t new_cap = builder->capacity * 2;

    int64_t* new_offsets = realloc(builder->offsets, new_cap * sizeof(int64_t));
    int32_t* new_compressed = realloc(builder->compressed_sizes, new_cap * sizeof(int32_t));
    int64_t* new_first_rows = realloc(builder->first_row_indices, new_cap * sizeof(int64_t));

    if (!new_offsets || !new_compressed || !new_first_rows) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    builder->offsets = new_offsets;
    builder->compressed_sizes = new_compressed;
    builder->first_row_indices = new_first_rows;

    if (builder->track_uncompressed) {
        int32_t* new_uncompressed = realloc(builder->uncompressed_sizes, new_cap * sizeof(int32_t));
        if (!new_uncompressed) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        builder->uncompressed_sizes = new_uncompressed;
    }

    builder->capacity = new_cap;
    return CARQUET_OK;
}

/**
 * Add a page's location to the offset index.
 */
carquet_status_t carquet_offset_index_add_page(
    carquet_offset_index_builder_t* builder,
    int64_t offset,
    int32_t compressed_size,
    int64_t first_row_index,
    int32_t uncompressed_size) {

    if (!builder) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_status_t status = offset_ensure_capacity(builder);
    if (status != CARQUET_OK) return status;

    int32_t idx = builder->num_pages;

    builder->offsets[idx] = offset;
    builder->compressed_sizes[idx] = compressed_size;
    builder->first_row_indices[idx] = first_row_index;

    if (builder->track_uncompressed) {
        builder->uncompressed_sizes[idx] = uncompressed_size;
    }

    builder->num_pages++;
    return CARQUET_OK;
}

/* ============================================================================
 * Serialization to Thrift
 * ============================================================================
 */

/**
 * Serialize column index to buffer.
 */
carquet_status_t carquet_column_index_serialize(
    const carquet_column_index_builder_t* builder,
    carquet_buffer_t* output) {

    if (!builder || !output) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, output);

    thrift_write_struct_begin(&enc);

    /* Field 1: null_pages (list<bool>) */
    thrift_write_field_header(&enc, THRIFT_TYPE_LIST, 1);
    thrift_write_list_begin(&enc, THRIFT_TYPE_TRUE, builder->num_pages);
    for (int32_t i = 0; i < builder->num_pages; i++) {
        thrift_write_bool(&enc, builder->null_pages[i]);
    }

    /* Field 2: min_values (list<binary>) */
    thrift_write_field_header(&enc, THRIFT_TYPE_LIST, 2);
    thrift_write_list_begin(&enc, THRIFT_TYPE_BINARY, builder->num_pages);
    for (int32_t i = 0; i < builder->num_pages; i++) {
        if (builder->min_values[i]) {
            thrift_write_binary(&enc, builder->min_values[i], builder->min_value_lens[i]);
        } else {
            thrift_write_binary(&enc, NULL, 0);
        }
    }

    /* Field 3: max_values (list<binary>) */
    thrift_write_field_header(&enc, THRIFT_TYPE_LIST, 3);
    thrift_write_list_begin(&enc, THRIFT_TYPE_BINARY, builder->num_pages);
    for (int32_t i = 0; i < builder->num_pages; i++) {
        if (builder->max_values[i]) {
            thrift_write_binary(&enc, builder->max_values[i], builder->max_value_lens[i]);
        } else {
            thrift_write_binary(&enc, NULL, 0);
        }
    }

    /* Field 4: boundary_order (i32) */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
    thrift_write_i32(&enc, builder->boundary_order);

    /* Field 5: null_counts (list<i64>) - optional */
    thrift_write_field_header(&enc, THRIFT_TYPE_LIST, 5);
    thrift_write_list_begin(&enc, THRIFT_TYPE_I64, builder->num_pages);
    for (int32_t i = 0; i < builder->num_pages; i++) {
        thrift_write_i64(&enc, builder->null_counts[i]);
    }

    thrift_write_struct_end(&enc);
    return CARQUET_OK;
}

/**
 * Serialize offset index to buffer.
 */
carquet_status_t carquet_offset_index_serialize(
    const carquet_offset_index_builder_t* builder,
    carquet_buffer_t* output) {

    if (!builder || !output) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, output);

    thrift_write_struct_begin(&enc);

    /* Field 1: page_locations (list<PageLocation>) */
    thrift_write_field_header(&enc, THRIFT_TYPE_LIST, 1);
    thrift_write_list_begin(&enc, THRIFT_TYPE_STRUCT, builder->num_pages);

    for (int32_t i = 0; i < builder->num_pages; i++) {
        thrift_write_struct_begin(&enc);

        /* PageLocation field 1: offset */
        thrift_write_field_header(&enc, THRIFT_TYPE_I64, 1);
        thrift_write_i64(&enc, builder->offsets[i]);

        /* PageLocation field 2: compressed_page_size */
        thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
        thrift_write_i32(&enc, builder->compressed_sizes[i]);

        /* PageLocation field 3: first_row_index */
        thrift_write_field_header(&enc, THRIFT_TYPE_I64, 3);
        thrift_write_i64(&enc, builder->first_row_indices[i]);

        thrift_write_struct_end(&enc);
    }

    /* Field 2: uncompressed_page_sizes (list<i32>) - optional */
    if (builder->track_uncompressed && builder->uncompressed_sizes) {
        thrift_write_field_header(&enc, THRIFT_TYPE_LIST, 2);
        thrift_write_list_begin(&enc, THRIFT_TYPE_I32, builder->num_pages);
        for (int32_t i = 0; i < builder->num_pages; i++) {
            thrift_write_i32(&enc, builder->uncompressed_sizes[i]);
        }
    }

    thrift_write_struct_end(&enc);
    return CARQUET_OK;
}

/* ============================================================================
 * Page Filtering Using Column Index
 * ============================================================================
 */

/**
 * Check if a page might contain values in the given range.
 *
 * @param builder Column index builder
 * @param page_idx Page index
 * @param min_value Query min value (NULL for unbounded)
 * @param max_value Query max value (NULL for unbounded)
 * @param value_len Length of value for byte array types
 * @param might_match Output: true if page might contain matching values
 * @return Status code
 */
carquet_status_t carquet_column_index_page_might_match(
    const carquet_column_index_builder_t* builder,
    int32_t page_idx,
    const void* min_value,
    const void* max_value,
    int32_t value_len,
    bool* might_match) {

    if (!builder || !might_match || page_idx < 0 || page_idx >= builder->num_pages) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Null pages never match non-null predicates */
    if (builder->null_pages[page_idx]) {
        *might_match = false;
        return CARQUET_OK;
    }

    *might_match = true;  /* Assume match by default */

    /* If query max < page min, no match */
    if (max_value && builder->min_values[page_idx]) {
        int cmp = memcmp(max_value, builder->min_values[page_idx],
                         value_len < builder->min_value_lens[page_idx] ?
                         value_len : builder->min_value_lens[page_idx]);
        if (cmp < 0 || (cmp == 0 && value_len < builder->min_value_lens[page_idx])) {
            *might_match = false;
            return CARQUET_OK;
        }
    }

    /* If query min > page max, no match */
    if (min_value && builder->max_values[page_idx]) {
        int cmp = memcmp(min_value, builder->max_values[page_idx],
                         value_len < builder->max_value_lens[page_idx] ?
                         value_len : builder->max_value_lens[page_idx]);
        if (cmp > 0 || (cmp == 0 && value_len > builder->max_value_lens[page_idx])) {
            *might_match = false;
            return CARQUET_OK;
        }
    }

    return CARQUET_OK;
}
