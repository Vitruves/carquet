/**
 * @file row_group_writer.c
 * @brief Row group writing implementation
 *
 * Manages writing multiple columns to form a row group,
 * tracking row counts and generating row group metadata.
 */

#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/buffer.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include <stdlib.h>
#include <string.h>

/* Forward declaration from column_writer.c */
typedef struct carquet_column_writer_internal carquet_column_writer_internal_t;

extern carquet_column_writer_internal_t* carquet_column_writer_create(
    carquet_physical_type_t type,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    size_t target_page_size);

extern void carquet_column_writer_destroy(carquet_column_writer_internal_t* writer);

extern carquet_status_t carquet_column_writer_write_batch(
    carquet_column_writer_internal_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels);

extern carquet_status_t carquet_column_writer_finalize(
    carquet_column_writer_internal_t* writer,
    const uint8_t** data,
    size_t* size,
    int64_t* total_values,
    int64_t* total_compressed_size,
    int64_t* total_uncompressed_size);

extern int64_t carquet_column_writer_num_values(const carquet_column_writer_internal_t* writer);

/* ============================================================================
 * Column Chunk Metadata
 * ============================================================================
 */

typedef struct column_chunk_info {
    int64_t file_offset;
    int64_t total_compressed_size;
    int64_t total_uncompressed_size;
    int64_t num_values;
    carquet_physical_type_t type;
    carquet_encoding_t encoding;
    carquet_compression_t compression;
    int32_t type_length;
    char* path;
} column_chunk_info_t;

/* ============================================================================
 * Row Group Writer Structure
 * ============================================================================
 */

typedef struct carquet_row_group_writer {
    carquet_column_writer_internal_t** column_writers;
    column_chunk_info_t* column_infos;
    int num_columns;

    carquet_buffer_t row_group_buffer;

    /* Configuration */
    carquet_compression_t compression;
    size_t target_page_size;
    int64_t num_rows;

    /* State */
    int64_t total_byte_size;
    int64_t file_offset;  /* Starting offset in file */
} carquet_row_group_writer_t;

/* ============================================================================
 * Row Group Writer Lifecycle
 * ============================================================================
 */

carquet_row_group_writer_t* carquet_row_group_writer_create(
    const carquet_schema_t* schema,
    carquet_compression_t compression,
    size_t target_page_size,
    int64_t file_offset) {

    (void)schema;  /* Will be used when we have schema traversal */

    carquet_row_group_writer_t* writer = calloc(1, sizeof(*writer));
    if (!writer) return NULL;

    if (carquet_buffer_init(&writer->row_group_buffer) != CARQUET_OK) {
        free(writer);
        return NULL;
    }

    writer->compression = compression;
    writer->target_page_size = target_page_size > 0 ? target_page_size : (1024 * 1024);
    writer->file_offset = file_offset;

    return writer;
}

void carquet_row_group_writer_destroy(carquet_row_group_writer_t* writer) {
    if (writer) {
        if (writer->column_writers) {
            for (int i = 0; i < writer->num_columns; i++) {
                if (writer->column_writers[i]) {
                    carquet_column_writer_destroy(writer->column_writers[i]);
                }
            }
            free(writer->column_writers);
        }

        if (writer->column_infos) {
            for (int i = 0; i < writer->num_columns; i++) {
                free(writer->column_infos[i].path);
            }
            free(writer->column_infos);
        }

        carquet_buffer_destroy(&writer->row_group_buffer);
        free(writer);
    }
}

/* ============================================================================
 * Column Management
 * ============================================================================
 */

carquet_status_t carquet_row_group_writer_add_column(
    carquet_row_group_writer_t* writer,
    const char* name,
    carquet_physical_type_t type,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length) {

    if (!writer || !name) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    int new_count = writer->num_columns + 1;

    /* Expand column writers array */
    carquet_column_writer_internal_t** new_writers = realloc(
        writer->column_writers,
        new_count * sizeof(carquet_column_writer_internal_t*));
    if (!new_writers) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    writer->column_writers = new_writers;

    /* Expand column infos array */
    column_chunk_info_t* new_infos = realloc(
        writer->column_infos,
        new_count * sizeof(column_chunk_info_t));
    if (!new_infos) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    writer->column_infos = new_infos;

    /* Create column writer */
    carquet_column_writer_internal_t* col_writer = carquet_column_writer_create(
        type,
        CARQUET_ENCODING_PLAIN,
        writer->compression,
        max_def_level,
        max_rep_level,
        type_length,
        writer->target_page_size);

    if (!col_writer) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    writer->column_writers[writer->num_columns] = col_writer;

    /* Initialize column info */
    memset(&writer->column_infos[writer->num_columns], 0, sizeof(column_chunk_info_t));
    writer->column_infos[writer->num_columns].type = type;
    writer->column_infos[writer->num_columns].encoding = CARQUET_ENCODING_PLAIN;
    writer->column_infos[writer->num_columns].compression = writer->compression;
    writer->column_infos[writer->num_columns].type_length = type_length;
    writer->column_infos[writer->num_columns].path = strdup(name);

    writer->num_columns = new_count;
    return CARQUET_OK;
}

carquet_status_t carquet_row_group_writer_write_column(
    carquet_row_group_writer_t* writer,
    int column_index,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    if (!writer || column_index < 0 || column_index >= writer->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    return carquet_column_writer_write_batch(
        writer->column_writers[column_index],
        values, num_values, def_levels, rep_levels);
}

/* ============================================================================
 * Finalization
 * ============================================================================
 */

carquet_status_t carquet_row_group_writer_finalize(
    carquet_row_group_writer_t* writer,
    const uint8_t** data,
    size_t* size,
    int64_t num_rows) {

    if (!writer) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    writer->num_rows = num_rows;
    carquet_buffer_clear(&writer->row_group_buffer);

    int64_t current_offset = writer->file_offset;

    /* Finalize each column and append to row group buffer */
    for (int i = 0; i < writer->num_columns; i++) {
        const uint8_t* col_data;
        size_t col_size;
        int64_t total_values;
        int64_t compressed_size;
        int64_t uncompressed_size;

        carquet_status_t status = carquet_column_writer_finalize(
            writer->column_writers[i],
            &col_data, &col_size,
            &total_values, &compressed_size, &uncompressed_size);

        if (status != CARQUET_OK) {
            return status;
        }

        /* Update column info */
        writer->column_infos[i].file_offset = current_offset;
        writer->column_infos[i].total_compressed_size = col_size;
        writer->column_infos[i].total_uncompressed_size = uncompressed_size;
        writer->column_infos[i].num_values = total_values;

        /* Append column data */
        status = carquet_buffer_append(&writer->row_group_buffer, col_data, col_size);
        if (status != CARQUET_OK) {
            return status;
        }

        current_offset += col_size;
        writer->total_byte_size += col_size;
    }

    if (data) *data = writer->row_group_buffer.data;
    if (size) *size = writer->row_group_buffer.size;

    return CARQUET_OK;
}

int carquet_row_group_writer_num_columns(const carquet_row_group_writer_t* writer) {
    return writer ? writer->num_columns : 0;
}

int64_t carquet_row_group_writer_num_rows(const carquet_row_group_writer_t* writer) {
    return writer ? writer->num_rows : 0;
}

int64_t carquet_row_group_writer_total_byte_size(const carquet_row_group_writer_t* writer) {
    return writer ? writer->total_byte_size : 0;
}

const column_chunk_info_t* carquet_row_group_writer_get_column_info(
    const carquet_row_group_writer_t* writer, int index) {
    if (!writer || index < 0 || index >= writer->num_columns) {
        return NULL;
    }
    return &writer->column_infos[index];
}
