/**
 * @file column_writer.c
 * @brief Column chunk writing implementation
 *
 * Manages writing values to a column chunk, handling page breaks,
 * dictionary encoding, and column-level metadata.
 */

#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/buffer.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include <stdlib.h>
#include <string.h>

/* Forward declaration from page_writer.c */
typedef struct carquet_page_writer carquet_page_writer_t;

extern carquet_page_writer_t* carquet_page_writer_create(
    carquet_physical_type_t type,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length);

extern void carquet_page_writer_destroy(carquet_page_writer_t* writer);
extern void carquet_page_writer_reset(carquet_page_writer_t* writer);

extern carquet_status_t carquet_page_writer_add_values(
    carquet_page_writer_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels);

extern carquet_status_t carquet_page_writer_finalize(
    carquet_page_writer_t* writer,
    const uint8_t** page_data,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size);

extern size_t carquet_page_writer_estimated_size(const carquet_page_writer_t* writer);
extern int64_t carquet_page_writer_num_values(const carquet_page_writer_t* writer);

/* ============================================================================
 * Column Writer Structure
 * ============================================================================
 */

typedef struct carquet_column_writer_internal {
    carquet_page_writer_t* page_writer;
    carquet_buffer_t column_buffer;  /* All pages for this column chunk */

    /* Column configuration */
    carquet_physical_type_t type;
    carquet_encoding_t encoding;
    carquet_compression_t compression;
    int32_t type_length;
    int16_t max_def_level;
    int16_t max_rep_level;

    /* Page size limits */
    size_t target_page_size;
    size_t max_page_size;

    /* Statistics */
    int64_t total_values;
    int64_t total_nulls;
    int64_t total_uncompressed_size;
    int64_t total_compressed_size;
    int32_t num_pages;

    /* Min/max tracking */
    bool has_min_max;
    uint8_t min_value[64];
    uint8_t max_value[64];
    size_t min_max_size;

    /* Column path for metadata */
    char** path_in_schema;
    int path_depth;
} carquet_column_writer_internal_t;

/* ============================================================================
 * Column Writer Lifecycle
 * ============================================================================
 */

carquet_column_writer_internal_t* carquet_column_writer_create(
    carquet_physical_type_t type,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    size_t target_page_size) {

    carquet_column_writer_internal_t* writer = calloc(1, sizeof(*writer));
    if (!writer) return NULL;

    writer->page_writer = carquet_page_writer_create(
        type, encoding, compression, max_def_level, max_rep_level, type_length);

    if (!writer->page_writer) {
        free(writer);
        return NULL;
    }

    carquet_buffer_init(&writer->column_buffer);

    writer->type = type;
    writer->encoding = encoding;
    writer->compression = compression;
    writer->type_length = type_length;
    writer->max_def_level = max_def_level;
    writer->max_rep_level = max_rep_level;
    writer->target_page_size = target_page_size > 0 ? target_page_size : (1024 * 1024);
    writer->max_page_size = writer->target_page_size * 2;

    return writer;
}

void carquet_column_writer_destroy(carquet_column_writer_internal_t* writer) {
    if (writer) {
        if (writer->page_writer) {
            carquet_page_writer_destroy(writer->page_writer);
        }
        carquet_buffer_destroy(&writer->column_buffer);

        /* Free path strings */
        if (writer->path_in_schema) {
            for (int i = 0; i < writer->path_depth; i++) {
                free(writer->path_in_schema[i]);
            }
            free(writer->path_in_schema);
        }

        free(writer);
    }
}

/* ============================================================================
 * Page Flushing
 * ============================================================================
 */

static carquet_status_t flush_current_page(carquet_column_writer_internal_t* writer) {
    if (carquet_page_writer_num_values(writer->page_writer) == 0) {
        return CARQUET_OK;
    }

    const uint8_t* page_data;
    size_t page_size;
    int32_t uncompressed_size;
    int32_t compressed_size;

    carquet_status_t status = carquet_page_writer_finalize(
        writer->page_writer, &page_data, &page_size,
        &uncompressed_size, &compressed_size);

    if (status != CARQUET_OK) {
        return status;
    }

    /* Append page to column buffer */
    status = carquet_buffer_append(&writer->column_buffer, page_data, page_size);
    if (status != CARQUET_OK) {
        return status;
    }

    /* Update statistics */
    writer->total_uncompressed_size += uncompressed_size;
    writer->total_compressed_size += compressed_size;
    writer->num_pages++;

    /* Reset page writer for next page */
    carquet_page_writer_reset(writer->page_writer);

    return CARQUET_OK;
}

/* ============================================================================
 * Writing Values
 * ============================================================================
 */

carquet_status_t carquet_column_writer_write_batch(
    carquet_column_writer_internal_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    if (!writer || !values) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_status_t status;

    /* Add values to current page */
    status = carquet_page_writer_add_values(
        writer->page_writer, values, num_values, def_levels, rep_levels);

    if (status != CARQUET_OK) {
        return status;
    }

    writer->total_values += num_values;

    /* Check if we should flush the page */
    size_t current_size = carquet_page_writer_estimated_size(writer->page_writer);
    if (current_size >= writer->target_page_size) {
        status = flush_current_page(writer);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Finalization
 * ============================================================================
 */

carquet_status_t carquet_column_writer_finalize(
    carquet_column_writer_internal_t* writer,
    const uint8_t** data,
    size_t* size,
    int64_t* total_values,
    int64_t* total_compressed_size,
    int64_t* total_uncompressed_size) {

    if (!writer) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Flush any remaining data */
    carquet_status_t status = flush_current_page(writer);
    if (status != CARQUET_OK) {
        return status;
    }

    if (data) *data = writer->column_buffer.data;
    if (size) *size = writer->column_buffer.size;
    if (total_values) *total_values = writer->total_values;
    if (total_compressed_size) *total_compressed_size = writer->total_compressed_size;
    if (total_uncompressed_size) *total_uncompressed_size = writer->total_uncompressed_size;

    return CARQUET_OK;
}

int64_t carquet_column_writer_num_values(const carquet_column_writer_internal_t* writer) {
    return writer ? writer->total_values : 0;
}

int32_t carquet_column_writer_num_pages(const carquet_column_writer_internal_t* writer) {
    return writer ? writer->num_pages : 0;
}
