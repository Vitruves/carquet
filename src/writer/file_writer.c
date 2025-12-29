/**
 * @file file_writer.c
 * @brief Parquet file writing implementation
 *
 * Manages writing a complete Parquet file including:
 * - File header (PAR1 magic)
 * - Row groups via row_group_writer
 * - File metadata serialization
 * - Footer with metadata size and PAR1 magic
 */

#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/buffer.h"
#include "core/arena.h"
#include "reader/reader_internal.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Parquet magic bytes */
static const uint8_t PARQUET_MAGIC[4] = {'P', 'A', 'R', '1'};

/* Forward declaration from row_group_writer.c */
typedef struct carquet_row_group_writer carquet_row_group_writer_t;

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

extern carquet_row_group_writer_t* carquet_row_group_writer_create(
    const carquet_schema_t* schema,
    carquet_compression_t compression,
    size_t target_page_size,
    int64_t file_offset);

extern void carquet_row_group_writer_destroy(carquet_row_group_writer_t* writer);

extern carquet_status_t carquet_row_group_writer_add_column(
    carquet_row_group_writer_t* writer,
    const char* name,
    carquet_physical_type_t type,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length);

extern carquet_status_t carquet_row_group_writer_write_column(
    carquet_row_group_writer_t* writer,
    int column_index,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels);

extern carquet_status_t carquet_row_group_writer_finalize(
    carquet_row_group_writer_t* writer,
    const uint8_t** data,
    size_t* size,
    int64_t num_rows);

extern int carquet_row_group_writer_num_columns(const carquet_row_group_writer_t* writer);
extern int64_t carquet_row_group_writer_num_rows(const carquet_row_group_writer_t* writer);
extern int64_t carquet_row_group_writer_total_byte_size(const carquet_row_group_writer_t* writer);
extern const column_chunk_info_t* carquet_row_group_writer_get_column_info(
    const carquet_row_group_writer_t* writer, int index);

/* ============================================================================
 * Writer Schema Structure (for building)
 * ============================================================================
 */

typedef struct writer_column_def {
    char* name;
    carquet_physical_type_t physical_type;
    carquet_logical_type_t logical_type;
    carquet_field_repetition_t repetition;
    int32_t type_length;
    int16_t max_def_level;
    int16_t max_rep_level;
} writer_column_def_t;

/* ============================================================================
 * Row Group Metadata Storage
 * ============================================================================
 */

typedef struct row_group_info {
    parquet_row_group_t metadata;
    int64_t file_offset;
} row_group_info_t;

/* ============================================================================
 * Writer Structure
 * ============================================================================
 */

struct carquet_writer {
    FILE* file;
    bool owns_file;
    char* path;

    /* Schema */
    writer_column_def_t* columns;
    int32_t num_columns;
    int32_t column_capacity;

    /* Options */
    carquet_writer_options_t options;

    /* Current row group */
    carquet_row_group_writer_t* current_row_group;
    int64_t current_row_group_rows;
    int64_t* column_values_written;  /* Values written per column in current row group */

    /* Completed row groups */
    row_group_info_t* row_groups;
    int32_t num_row_groups;
    int32_t row_groups_capacity;

    /* File state */
    int64_t file_offset;
    int64_t total_rows;
    bool header_written;

    /* Arena for metadata allocations */
    carquet_arena_t arena;
};

/* ============================================================================
 * Writer Options
 * ============================================================================
 */

void carquet_writer_options_init(carquet_writer_options_t* options) {
    /* options is nonnull per API contract */
    memset(options, 0, sizeof(*options));
    options->compression = CARQUET_COMPRESSION_UNCOMPRESSED;
    options->compression_level = 0;
    options->row_group_size = 128 * 1024 * 1024;  /* 128 MB */
    options->page_size = 1024 * 1024;              /* 1 MB */
    options->write_statistics = true;
    options->write_page_index = false;
    options->write_bloom_filters = false;
    options->dictionary_encoding = CARQUET_ENCODING_PLAIN_DICTIONARY;
    options->dictionary_page_size = 1024 * 1024;   /* 1 MB */
    options->created_by = "Carquet";
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================
 */

static carquet_status_t write_magic(FILE* file) {
    if (fwrite(PARQUET_MAGIC, 1, 4, file) != 4) {
        return CARQUET_ERROR_FILE_WRITE;
    }
    return CARQUET_OK;
}

static carquet_status_t ensure_header_written(carquet_writer_t* writer) {
    if (writer->header_written) {
        return CARQUET_OK;
    }

    carquet_status_t status = write_magic(writer->file);
    if (status != CARQUET_OK) {
        return status;
    }

    writer->file_offset = 4;  /* PAR1 magic */
    writer->header_written = true;
    return CARQUET_OK;
}

static carquet_status_t add_column_internal(
    carquet_writer_t* writer,
    const char* name,
    carquet_physical_type_t physical_type,
    const carquet_logical_type_t* logical_type,
    carquet_field_repetition_t repetition,
    int32_t type_length) {

    /* Expand capacity if needed */
    if (writer->num_columns >= writer->column_capacity) {
        int32_t new_cap = writer->column_capacity == 0 ? 8 : writer->column_capacity * 2;
        writer_column_def_t* new_cols = realloc(writer->columns,
            new_cap * sizeof(writer_column_def_t));
        if (!new_cols) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        writer->columns = new_cols;

        int64_t* new_values = realloc(writer->column_values_written,
            new_cap * sizeof(int64_t));
        if (!new_values) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        writer->column_values_written = new_values;

        writer->column_capacity = new_cap;
    }

    writer_column_def_t* col = &writer->columns[writer->num_columns];
    memset(col, 0, sizeof(*col));

    col->name = strdup(name);
    if (!col->name) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    col->physical_type = physical_type;
    col->repetition = repetition;
    col->type_length = type_length;

    if (logical_type) {
        col->logical_type = *logical_type;
    }

    /* Compute definition level based on repetition */
    col->max_def_level = (repetition == CARQUET_REPETITION_OPTIONAL) ? 1 : 0;
    col->max_rep_level = (repetition == CARQUET_REPETITION_REPEATED) ? 1 : 0;

    writer->column_values_written[writer->num_columns] = 0;
    writer->num_columns++;

    return CARQUET_OK;
}

static carquet_status_t ensure_row_group(carquet_writer_t* writer) {
    if (writer->current_row_group) {
        return CARQUET_OK;
    }

    writer->current_row_group = carquet_row_group_writer_create(
        NULL,  /* Schema not used directly */
        writer->options.compression,
        (size_t)writer->options.page_size,
        writer->file_offset);

    if (!writer->current_row_group) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    /* Add all columns to the row group writer */
    for (int32_t i = 0; i < writer->num_columns; i++) {
        writer_column_def_t* col = &writer->columns[i];
        carquet_status_t status = carquet_row_group_writer_add_column(
            writer->current_row_group,
            col->name,
            col->physical_type,
            col->max_def_level,
            col->max_rep_level,
            col->type_length);

        if (status != CARQUET_OK) {
            carquet_row_group_writer_destroy(writer->current_row_group);
            writer->current_row_group = NULL;
            return status;
        }
    }

    writer->current_row_group_rows = 0;
    for (int32_t i = 0; i < writer->num_columns; i++) {
        writer->column_values_written[i] = 0;
    }

    return CARQUET_OK;
}

static carquet_status_t flush_row_group(carquet_writer_t* writer) {
    if (!writer->current_row_group) {
        return CARQUET_OK;
    }

    /* Finalize the row group */
    const uint8_t* data;
    size_t size;
    carquet_status_t status = carquet_row_group_writer_finalize(
        writer->current_row_group, &data, &size, writer->current_row_group_rows);

    if (status != CARQUET_OK) {
        return status;
    }

    /* Write row group data to file */
    if (size > 0) {
        if (fwrite(data, 1, size, writer->file) != size) {
            return CARQUET_ERROR_FILE_WRITE;
        }
    }

    /* Store row group metadata */
    if (writer->num_row_groups >= writer->row_groups_capacity) {
        int32_t new_cap = writer->row_groups_capacity == 0 ? 4 : writer->row_groups_capacity * 2;
        row_group_info_t* new_rgs = realloc(writer->row_groups,
            new_cap * sizeof(row_group_info_t));
        if (!new_rgs) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        writer->row_groups = new_rgs;
        writer->row_groups_capacity = new_cap;
    }

    row_group_info_t* rg_info = &writer->row_groups[writer->num_row_groups];
    memset(rg_info, 0, sizeof(*rg_info));

    rg_info->file_offset = writer->file_offset;
    rg_info->metadata.num_rows = writer->current_row_group_rows;
    rg_info->metadata.total_byte_size = carquet_row_group_writer_total_byte_size(writer->current_row_group);
    rg_info->metadata.has_file_offset = true;
    rg_info->metadata.file_offset = writer->file_offset;
    rg_info->metadata.has_total_compressed_size = true;
    rg_info->metadata.total_compressed_size = (int64_t)size;
    rg_info->metadata.has_ordinal = true;
    rg_info->metadata.ordinal = (int16_t)writer->num_row_groups;

    /* Build column chunks metadata */
    int num_cols = carquet_row_group_writer_num_columns(writer->current_row_group);
    rg_info->metadata.num_columns = num_cols;
    rg_info->metadata.columns = carquet_arena_calloc(&writer->arena, num_cols,
        sizeof(parquet_column_chunk_t));

    if (!rg_info->metadata.columns) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < num_cols; i++) {
        const column_chunk_info_t* col_info = carquet_row_group_writer_get_column_info(
            writer->current_row_group, i);

        if (!col_info) continue;

        parquet_column_chunk_t* chunk = &rg_info->metadata.columns[i];
        chunk->file_offset = col_info->file_offset;
        chunk->has_metadata = true;

        parquet_column_metadata_t* meta = &chunk->metadata;
        meta->type = col_info->type;
        meta->codec = col_info->compression;
        meta->num_values = col_info->num_values;
        meta->total_compressed_size = col_info->total_compressed_size;
        meta->total_uncompressed_size = col_info->total_uncompressed_size;
        meta->data_page_offset = col_info->file_offset;

        /* Encodings used */
        meta->num_encodings = 2;  /* PLAIN + RLE for levels */
        meta->encodings = carquet_arena_calloc(&writer->arena, 2, sizeof(carquet_encoding_t));
        if (meta->encodings) {
            meta->encodings[0] = CARQUET_ENCODING_PLAIN;
            meta->encodings[1] = CARQUET_ENCODING_RLE;
        }

        /* Path in schema */
        meta->path_len = 1;
        meta->path_in_schema = carquet_arena_calloc(&writer->arena, 1, sizeof(char*));
        if (meta->path_in_schema && col_info->path) {
            meta->path_in_schema[0] = carquet_arena_strdup(&writer->arena, col_info->path);
        }
    }

    writer->num_row_groups++;
    writer->file_offset += (int64_t)size;
    writer->total_rows += writer->current_row_group_rows;

    /* Cleanup current row group */
    carquet_row_group_writer_destroy(writer->current_row_group);
    writer->current_row_group = NULL;
    writer->current_row_group_rows = 0;

    return CARQUET_OK;
}

static carquet_status_t build_file_metadata(
    carquet_writer_t* writer,
    parquet_file_metadata_t* metadata) {

    memset(metadata, 0, sizeof(*metadata));

    metadata->version = 2;  /* Parquet version 2 */
    metadata->num_rows = writer->total_rows;
    metadata->created_by = carquet_arena_strdup(&writer->arena,
        writer->options.created_by ? writer->options.created_by : "Carquet");

    /* Build schema: root group + leaf columns */
    int32_t num_schema_elements = 1 + writer->num_columns;
    metadata->num_schema_elements = num_schema_elements;
    metadata->schema = carquet_arena_calloc(&writer->arena, num_schema_elements,
        sizeof(parquet_schema_element_t));

    if (!metadata->schema) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    /* Root element */
    parquet_schema_element_t* root = &metadata->schema[0];
    root->name = carquet_arena_strdup(&writer->arena, "schema");
    root->num_children = writer->num_columns;
    root->has_repetition = false;

    /* Column elements */
    for (int32_t i = 0; i < writer->num_columns; i++) {
        writer_column_def_t* col = &writer->columns[i];
        parquet_schema_element_t* elem = &metadata->schema[1 + i];

        elem->name = carquet_arena_strdup(&writer->arena, col->name);
        elem->has_type = true;
        elem->type = col->physical_type;
        elem->has_repetition = true;
        elem->repetition_type = col->repetition;
        elem->type_length = col->type_length;

        if (col->logical_type.id != CARQUET_LOGICAL_UNKNOWN) {
            elem->has_logical_type = true;
            elem->logical_type = col->logical_type;
        }
    }

    /* Row groups */
    metadata->num_row_groups = writer->num_row_groups;
    metadata->row_groups = carquet_arena_calloc(&writer->arena, writer->num_row_groups,
        sizeof(parquet_row_group_t));

    if (!metadata->row_groups && writer->num_row_groups > 0) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < writer->num_row_groups; i++) {
        metadata->row_groups[i] = writer->row_groups[i].metadata;
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================
 */

carquet_writer_t* carquet_writer_create(
    const char* path,
    const carquet_schema_t* schema,
    const carquet_writer_options_t* options,
    carquet_error_t* error) {

    /* path and schema are nonnull per API contract */
    carquet_writer_t* writer = calloc(1, sizeof(carquet_writer_t));
    if (!writer) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate writer");
        return NULL;
    }

    /* Initialize arena */
    if (carquet_arena_init_size(&writer->arena, 4096) != CARQUET_OK) {
        free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate arena");
        return NULL;
    }

    /* Open file */
    writer->file = fopen(path, "wb");
    if (!writer->file) {
        carquet_arena_destroy(&writer->arena);
        free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_OPEN, "Failed to open file for writing: %s", path);
        return NULL;
    }
    writer->owns_file = true;

    writer->path = strdup(path);
    if (!writer->path) {
        fclose(writer->file);
        carquet_arena_destroy(&writer->arena);
        free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate path");
        return NULL;
    }

    /* Copy options */
    if (options) {
        writer->options = *options;
    } else {
        carquet_writer_options_init(&writer->options);
    }

    /* Add columns from schema (schema is nonnull per API contract) */
    for (int32_t i = 0; i < schema->num_leaves; i++) {
        int32_t elem_idx = schema->leaf_indices[i];
        parquet_schema_element_t* elem = &schema->elements[elem_idx];

        carquet_logical_type_t* lt = elem->has_logical_type ? &elem->logical_type : NULL;

        carquet_status_t status = add_column_internal(
            writer,
            elem->name,
            elem->type,
            lt,
            elem->repetition_type,
            elem->type_length);

        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to add column from schema");
            return NULL;
        }
    }

    return writer;
}

carquet_writer_t* carquet_writer_create_file(
    FILE* file,
    const carquet_schema_t* schema,
    const carquet_writer_options_t* options,
    carquet_error_t* error) {

    /* file and schema are nonnull per API contract */
    carquet_writer_t* writer = calloc(1, sizeof(carquet_writer_t));
    if (!writer) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate writer");
        return NULL;
    }

    /* Initialize arena */
    if (carquet_arena_init_size(&writer->arena, 4096) != CARQUET_OK) {
        free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate arena");
        return NULL;
    }

    writer->file = file;
    writer->owns_file = false;

    /* Copy options */
    if (options) {
        writer->options = *options;
    } else {
        carquet_writer_options_init(&writer->options);
    }

    /* Add columns from schema (schema is nonnull per API contract) */
    for (int32_t i = 0; i < schema->num_leaves; i++) {
        int32_t elem_idx = schema->leaf_indices[i];
        parquet_schema_element_t* elem = &schema->elements[elem_idx];

        carquet_logical_type_t* lt = elem->has_logical_type ? &elem->logical_type : NULL;

        carquet_status_t status = add_column_internal(
            writer,
            elem->name,
            elem->type,
            lt,
            elem->repetition_type,
            elem->type_length);

        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to add column from schema");
            return NULL;
        }
    }

    return writer;
}

carquet_status_t carquet_writer_write_batch(
    carquet_writer_t* writer,
    int32_t column_index,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    /* writer and values are nonnull per API contract */
    if (column_index < 0 || column_index >= writer->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Ensure header is written */
    carquet_status_t status = ensure_header_written(writer);
    if (status != CARQUET_OK) {
        return status;
    }

    /* Ensure we have a row group */
    status = ensure_row_group(writer);
    if (status != CARQUET_OK) {
        return status;
    }

    /* Write to the row group */
    status = carquet_row_group_writer_write_column(
        writer->current_row_group,
        column_index,
        values,
        num_values,
        def_levels,
        rep_levels);

    if (status != CARQUET_OK) {
        return status;
    }

    writer->column_values_written[column_index] += num_values;

    /* Track rows (use column 0 as reference) */
    if (column_index == 0) {
        writer->current_row_group_rows += num_values;
    }

    return CARQUET_OK;
}

carquet_status_t carquet_writer_new_row_group(carquet_writer_t* writer) {
    /* writer is nonnull per API contract */
    /* Ensure header is written */
    carquet_status_t status = ensure_header_written(writer);
    if (status != CARQUET_OK) {
        return status;
    }

    /* Flush current row group if any */
    return flush_row_group(writer);
}

carquet_status_t carquet_writer_close(carquet_writer_t* writer) {
    /* writer is nonnull per API contract */
    carquet_status_t status = CARQUET_OK;

    /* Ensure header is written */
    status = ensure_header_written(writer);
    if (status != CARQUET_OK) {
        goto cleanup;
    }

    /* Flush any pending row group */
    status = flush_row_group(writer);
    if (status != CARQUET_OK) {
        goto cleanup;
    }

    /* Build file metadata */
    parquet_file_metadata_t metadata;
    status = build_file_metadata(writer, &metadata);
    if (status != CARQUET_OK) {
        goto cleanup;
    }

    /* Serialize metadata to buffer */
    carquet_buffer_t metadata_buffer;
    if (carquet_buffer_init(&metadata_buffer) != CARQUET_OK) {
        status = CARQUET_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    status = parquet_write_file_metadata(&metadata, &metadata_buffer, NULL);
    if (status != CARQUET_OK) {
        carquet_buffer_destroy(&metadata_buffer);
        goto cleanup;
    }

    /* Write metadata */
    if (fwrite(metadata_buffer.data, 1, metadata_buffer.size, writer->file) != metadata_buffer.size) {
        carquet_buffer_destroy(&metadata_buffer);
        status = CARQUET_ERROR_FILE_WRITE;
        goto cleanup;
    }

    /* Write metadata length (4 bytes, little-endian) */
    uint32_t metadata_len = (uint32_t)metadata_buffer.size;
    uint8_t len_bytes[4];
    len_bytes[0] = (uint8_t)(metadata_len & 0xFF);
    len_bytes[1] = (uint8_t)((metadata_len >> 8) & 0xFF);
    len_bytes[2] = (uint8_t)((metadata_len >> 16) & 0xFF);
    len_bytes[3] = (uint8_t)((metadata_len >> 24) & 0xFF);

    if (fwrite(len_bytes, 1, 4, writer->file) != 4) {
        carquet_buffer_destroy(&metadata_buffer);
        status = CARQUET_ERROR_FILE_WRITE;
        goto cleanup;
    }

    carquet_buffer_destroy(&metadata_buffer);

    /* Write footer magic */
    status = write_magic(writer->file);
    if (status != CARQUET_OK) {
        goto cleanup;
    }

    /* Flush and close */
    fflush(writer->file);

cleanup:
    /* Free resources */
    if (writer->current_row_group) {
        carquet_row_group_writer_destroy(writer->current_row_group);
        writer->current_row_group = NULL;
    }

    if (writer->owns_file && writer->file) {
        fclose(writer->file);
        writer->file = NULL;
    }

    /* Free column definitions */
    if (writer->columns) {
        for (int32_t i = 0; i < writer->num_columns; i++) {
            free(writer->columns[i].name);
        }
        free(writer->columns);
    }

    free(writer->column_values_written);
    free(writer->row_groups);
    free(writer->path);
    carquet_arena_destroy(&writer->arena);
    free(writer);

    return status;
}

void carquet_writer_abort(carquet_writer_t* writer) {
    if (!writer) return;

    /* Cleanup row group */
    if (writer->current_row_group) {
        carquet_row_group_writer_destroy(writer->current_row_group);
        writer->current_row_group = NULL;
    }

    /* Close and delete file */
    if (writer->owns_file && writer->file) {
        fclose(writer->file);
        writer->file = NULL;

        if (writer->path) {
            remove(writer->path);
        }
    }

    /* Free column definitions */
    if (writer->columns) {
        for (int32_t i = 0; i < writer->num_columns; i++) {
            free(writer->columns[i].name);
        }
        free(writer->columns);
    }

    free(writer->column_values_written);
    free(writer->row_groups);
    free(writer->path);
    carquet_arena_destroy(&writer->arena);
    free(writer);
}

