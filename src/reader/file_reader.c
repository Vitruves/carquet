/**
 * @file file_reader.c
 * @brief Parquet file reader implementation
 */

#include <carquet/carquet.h>
#include "reader_internal.h"
#include "thrift/parquet_types.h"
#include "core/arena.h"
#include "core/buffer.h"
#include "core/endian.h"
#include "encoding/plain.h"
#include "encoding/rle.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Constants
 * ============================================================================
 */

#define PARQUET_MAGIC "PAR1"
#define PARQUET_MAGIC_LEN 4
#define PARQUET_FOOTER_SIZE_LEN 4

/* ============================================================================
 * Schema Building
 * ============================================================================
 */

static int32_t count_leaves(const parquet_schema_element_t* elements, int32_t count) {
    int32_t leaves = 0;
    for (int32_t i = 0; i < count; i++) {
        if (elements[i].num_children == 0) {
            leaves++;
        }
    }
    return leaves;
}

/**
 * Recursive schema traversal context for computing definition/repetition levels.
 */
typedef struct {
    const parquet_schema_element_t* elements;
    int32_t num_elements;
    int16_t* max_def;
    int16_t* max_rep;
    int32_t* leaf_indices;
    int32_t leaf_idx;
} schema_traverse_ctx_t;

/**
 * Recursively traverse schema tree and compute definition/repetition levels.
 *
 * @param ctx Traversal context
 * @param element_idx Current element index in flat array
 * @param def_level Current definition level from ancestors
 * @param rep_level Current repetition level from ancestors
 * @return Next element index to process (after this subtree)
 */
static int32_t traverse_schema_recursive(
    schema_traverse_ctx_t* ctx,
    int32_t element_idx,
    int16_t def_level,
    int16_t rep_level) {

    if (element_idx >= ctx->num_elements) {
        return element_idx;
    }

    const parquet_schema_element_t* elem = &ctx->elements[element_idx];

    /* Calculate level contribution from this node's repetition type */
    int16_t this_def = def_level;
    int16_t this_rep = rep_level;

    if (elem->has_repetition) {
        switch (elem->repetition_type) {
            case CARQUET_REPETITION_OPTIONAL:
                /* Optional fields add 1 to definition level */
                this_def++;
                break;
            case CARQUET_REPETITION_REPEATED:
                /* Repeated fields add 1 to both definition and repetition levels */
                this_def++;
                this_rep++;
                break;
            case CARQUET_REPETITION_REQUIRED:
            default:
                /* Required fields don't add to levels */
                break;
        }
    }

    if (elem->num_children == 0) {
        /* Leaf node - record the accumulated levels */
        ctx->max_def[ctx->leaf_idx] = this_def;
        ctx->max_rep[ctx->leaf_idx] = this_rep;
        ctx->leaf_indices[ctx->leaf_idx] = element_idx;
        ctx->leaf_idx++;
        return element_idx + 1;
    }

    /* Group node - recursively process children */
    int32_t next_idx = element_idx + 1;
    for (int32_t child = 0; child < elem->num_children; child++) {
        next_idx = traverse_schema_recursive(ctx, next_idx, this_def, this_rep);
    }

    return next_idx;
}

/**
 * Compute definition and repetition levels for all leaf columns.
 *
 * Parquet stores schema as a flat array in depth-first order. This function
 * recursively traverses the schema tree to compute the maximum definition
 * and repetition levels for each leaf column.
 *
 * Definition level: Number of optional/repeated ancestors + 1 if self is optional/repeated
 * Repetition level: Number of repeated ancestors + 1 if self is repeated
 *
 * Example schema:
 *   schema (root, required)
 *   ├── a (optional, int32)        -> def=1, rep=0
 *   ├── b (optional, group)
 *   │   ├── c (required, int32)    -> def=1, rep=0  (from parent b)
 *   │   └── d (optional, int32)    -> def=2, rep=0  (from b + self)
 *   └── e (repeated, group)
 *       ├── f (required, int32)    -> def=1, rep=1  (from parent e)
 *       └── g (optional, int32)    -> def=2, rep=1  (from e + self)
 */
static void compute_levels(
    const parquet_schema_element_t* elements,
    int32_t num_elements,
    int16_t* max_def,
    int16_t* max_rep,
    int32_t* leaf_indices) {

    if (num_elements <= 1) {
        return;  /* Empty or root-only schema */
    }

    schema_traverse_ctx_t ctx = {
        .elements = elements,
        .num_elements = num_elements,
        .max_def = max_def,
        .max_rep = max_rep,
        .leaf_indices = leaf_indices,
        .leaf_idx = 0
    };

    /* Start traversal from root (index 0) with zero levels.
     * Root is required by definition, so it doesn't contribute to levels.
     * We process its children starting at index 1. */
    const parquet_schema_element_t* root = &elements[0];
    int32_t next_idx = 1;
    for (int32_t child = 0; child < root->num_children; child++) {
        next_idx = traverse_schema_recursive(&ctx, next_idx, 0, 0);
    }
}

carquet_schema_t* build_schema(
    carquet_arena_t* arena,
    const parquet_file_metadata_t* metadata,
    carquet_error_t* error) {

    carquet_schema_t* schema = carquet_arena_calloc(arena, 1, sizeof(carquet_schema_t));
    if (!schema) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate schema");
        return NULL;
    }

    schema->elements = metadata->schema;
    schema->num_elements = metadata->num_schema_elements;
    schema->capacity = metadata->num_schema_elements;  /* Fixed size from file */
    schema->num_leaves = count_leaves(metadata->schema, metadata->num_schema_elements);

    schema->leaf_indices = carquet_arena_calloc(arena, schema->num_leaves, sizeof(int32_t));
    schema->max_def_levels = carquet_arena_calloc(arena, schema->num_leaves, sizeof(int16_t));
    schema->max_rep_levels = carquet_arena_calloc(arena, schema->num_leaves, sizeof(int16_t));

    if (!schema->leaf_indices || !schema->max_def_levels || !schema->max_rep_levels) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate schema arrays");
        return NULL;
    }

    compute_levels(schema->elements, schema->num_elements,
                   schema->max_def_levels, schema->max_rep_levels,
                   schema->leaf_indices);

    return schema;
}

/* ============================================================================
 * File Reader Implementation
 * ============================================================================
 */

void carquet_reader_options_init(carquet_reader_options_t* options) {
    /* Parameter is nonnull per API contract */
    memset(options, 0, sizeof(*options));
    options->use_mmap = false;
    options->verify_checksums = true;
    options->buffer_size = 64 * 1024;
    options->num_threads = 0;
}

static carquet_status_t read_footer(carquet_reader_t* reader, carquet_error_t* error) {
    /* Seek to end to get file size */
    if (fseek(reader->file, 0, SEEK_END) != 0) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_SEEK, "Failed to seek to end");
        return CARQUET_ERROR_FILE_SEEK;
    }

    long file_size = ftell(reader->file);
    if (file_size < 0) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to get file size");
        return CARQUET_ERROR_FILE_READ;
    }
    reader->file_size = (size_t)file_size;

    /* Check minimum size */
    if (reader->file_size < PARQUET_MAGIC_LEN * 2 + PARQUET_FOOTER_SIZE_LEN) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_FOOTER, "File too small");
        return CARQUET_ERROR_INVALID_FOOTER;
    }

    /* Read trailing magic and footer size */
    uint8_t footer_tail[8];
    if (fseek(reader->file, -8, SEEK_END) != 0) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_SEEK, "Failed to seek to footer");
        return CARQUET_ERROR_FILE_SEEK;
    }

    if (fread(footer_tail, 1, 8, reader->file) != 8) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to read footer tail");
        return CARQUET_ERROR_FILE_READ;
    }

    /* Verify magic */
    if (memcmp(footer_tail + 4, PARQUET_MAGIC, PARQUET_MAGIC_LEN) != 0) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_MAGIC, "Invalid trailing magic");
        return CARQUET_ERROR_INVALID_MAGIC;
    }

    /* Get footer size */
    uint32_t footer_size = carquet_read_u32_le(footer_tail);
    if (footer_size > reader->file_size - 8) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_FOOTER, "Footer size too large");
        return CARQUET_ERROR_INVALID_FOOTER;
    }

    /* Read footer (Thrift-encoded metadata) */
    uint8_t* footer_data = malloc(footer_size);
    if (!footer_data) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate footer buffer");
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    long footer_offset = (long)(reader->file_size - 8 - footer_size);
    if (fseek(reader->file, footer_offset, SEEK_SET) != 0) {
        free(footer_data);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_SEEK, "Failed to seek to footer data");
        return CARQUET_ERROR_FILE_SEEK;
    }

    if (fread(footer_data, 1, footer_size, reader->file) != footer_size) {
        free(footer_data);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to read footer data");
        return CARQUET_ERROR_FILE_READ;
    }

    /* Parse metadata */
    carquet_status_t status = parquet_parse_file_metadata(
        footer_data, footer_size, &reader->arena, &reader->metadata, error);

    free(footer_data);

    if (status != CARQUET_OK) {
        return status;
    }

    /* Build schema */
    reader->schema = build_schema(&reader->arena, &reader->metadata, error);
    if (!reader->schema) {
        return CARQUET_ERROR_INVALID_SCHEMA;
    }

    return CARQUET_OK;
}

/**
 * Read footer from memory-mapped data.
 */
static carquet_status_t read_footer_mmap(carquet_reader_t* reader, carquet_error_t* error) {
    const uint8_t* data = reader->mmap_data;
    size_t file_size = reader->file_size;

    /* Check minimum size */
    if (file_size < PARQUET_MAGIC_LEN * 2 + PARQUET_FOOTER_SIZE_LEN) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_FOOTER, "File too small");
        return CARQUET_ERROR_INVALID_FOOTER;
    }

    /* Verify magic bytes at start and end */
    if (memcmp(data, PARQUET_MAGIC, PARQUET_MAGIC_LEN) != 0) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_MAGIC, "Invalid header magic");
        return CARQUET_ERROR_INVALID_MAGIC;
    }

    const uint8_t* end = data + file_size;
    if (memcmp(end - 4, PARQUET_MAGIC, PARQUET_MAGIC_LEN) != 0) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_MAGIC, "Invalid trailing magic");
        return CARQUET_ERROR_INVALID_MAGIC;
    }

    /* Get footer size */
    uint32_t footer_size = carquet_read_u32_le(end - 8);
    if (footer_size > file_size - 8) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_FOOTER, "Footer size too large");
        return CARQUET_ERROR_INVALID_FOOTER;
    }

    /* Parse metadata directly from mmap (zero-copy) */
    const uint8_t* footer_data = end - 8 - footer_size;
    carquet_status_t status = parquet_parse_file_metadata(
        footer_data, footer_size, &reader->arena, &reader->metadata, error);

    if (status != CARQUET_OK) {
        return status;
    }

    /* Build schema */
    reader->schema = build_schema(&reader->arena, &reader->metadata, error);
    if (!reader->schema) {
        return CARQUET_ERROR_INVALID_SCHEMA;
    }

    return CARQUET_OK;
}

carquet_reader_t* carquet_reader_open(
    const char* path,
    const carquet_reader_options_t* options,
    carquet_error_t* error) {

    carquet_reader_t* reader = calloc(1, sizeof(carquet_reader_t));
    if (!reader) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate reader");
        return NULL;
    }

    if (options) {
        reader->options = *options;
    } else {
        carquet_reader_options_init(&reader->options);
    }

    /* Initialize arena */
    if (carquet_arena_init(&reader->arena) != CARQUET_OK) {
        free(reader);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to initialize arena");
        return NULL;
    }

    carquet_status_t status;

    /* Try mmap if requested */
    if (reader->options.use_mmap) {
        carquet_mmap_info_t* mmap_info = carquet_mmap_open(path, error);
        if (mmap_info) {
            reader->mmap_info = mmap_info;
            reader->mmap_data = mmap_info->data;
            reader->file_size = mmap_info->size;
            reader->owns_file = false;  /* mmap handles cleanup */

            /* Parse footer from mmap */
            status = read_footer_mmap(reader, error);
            if (status != CARQUET_OK) {
                carquet_mmap_close(reader->mmap_info);
                carquet_arena_destroy(&reader->arena);
                free(reader);
                return NULL;
            }

            reader->is_open = true;
            return reader;
        }
        /* mmap failed, fall through to fread path */
    }

    /* Standard fread path */
    FILE* file = fopen(path, "rb");
    if (!file) {
        carquet_arena_destroy(&reader->arena);
        free(reader);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_OPEN, "Failed to open file: %s", path);
        return NULL;
    }

    reader->file = file;
    reader->owns_file = true;

    /* Read and parse footer */
    status = read_footer(reader, error);
    if (status != CARQUET_OK) {
        carquet_arena_destroy(&reader->arena);
        fclose(file);
        free(reader);
        return NULL;
    }

    reader->is_open = true;
    return reader;
}

void carquet_reader_close(carquet_reader_t* reader) {
    if (!reader) return;

    /* Close mmap if active */
    if (reader->mmap_info) {
        carquet_mmap_close(reader->mmap_info);
        reader->mmap_info = NULL;
        reader->mmap_data = NULL;
    }

    if (reader->owns_file && reader->file) {
        fclose(reader->file);
    }

    carquet_arena_destroy(&reader->arena);
    free(reader);
}

const carquet_schema_t* carquet_reader_schema(const carquet_reader_t* reader) {
    /* reader is nonnull per API contract */
    return reader->schema;
}

int64_t carquet_reader_num_rows(const carquet_reader_t* reader) {
    /* reader is nonnull per API contract */
    return reader->metadata.num_rows;
}

int32_t carquet_reader_num_row_groups(const carquet_reader_t* reader) {
    /* reader is nonnull per API contract */
    return reader->metadata.num_row_groups;
}

int32_t carquet_reader_num_columns(const carquet_reader_t* reader) {
    /* reader is nonnull per API contract */
    return reader->schema->num_leaves;
}

carquet_status_t carquet_reader_row_group_metadata(
    const carquet_reader_t* reader,
    int32_t row_group_index,
    carquet_row_group_metadata_t* metadata) {

    /* reader and metadata are nonnull per API contract */
    if (row_group_index < 0 || row_group_index >= reader->metadata.num_row_groups) {
        return CARQUET_ERROR_ROW_GROUP_NOT_FOUND;
    }

    const parquet_row_group_t* rg = &reader->metadata.row_groups[row_group_index];
    metadata->num_rows = rg->num_rows;
    metadata->total_byte_size = rg->total_byte_size;
    metadata->total_compressed_size = rg->has_total_compressed_size ?
        rg->total_compressed_size : rg->total_byte_size;

    return CARQUET_OK;
}

/* ============================================================================
 * Column Reader Implementation
 * ============================================================================
 */

carquet_column_reader_t* carquet_reader_get_column(
    carquet_reader_t* reader,
    int32_t row_group_index,
    int32_t column_index,
    carquet_error_t* error) {

    /* reader is nonnull per API contract */
    if (row_group_index < 0 || row_group_index >= reader->metadata.num_row_groups) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_ROW_GROUP_NOT_FOUND,
            "Row group %d not found", row_group_index);
        return NULL;
    }

    if (column_index < 0 || column_index >= reader->schema->num_leaves) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_COLUMN_NOT_FOUND,
            "Column %d not found", column_index);
        return NULL;
    }

    const parquet_row_group_t* rg = &reader->metadata.row_groups[row_group_index];

    if (column_index >= rg->num_columns) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_COLUMN_NOT_FOUND,
            "Column %d not in row group", column_index);
        return NULL;
    }

    carquet_column_reader_t* col_reader = calloc(1, sizeof(carquet_column_reader_t));
    if (!col_reader) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY,
            "Failed to allocate column reader");
        return NULL;
    }

    col_reader->file_reader = reader;
    col_reader->row_group_index = row_group_index;
    col_reader->column_index = column_index;
    col_reader->chunk = &rg->columns[column_index];

    if (col_reader->chunk->has_metadata) {
        col_reader->col_meta = &col_reader->chunk->metadata;
    } else {
        /* Metadata might be in separate file - not supported yet */
        free(col_reader);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_NOT_IMPLEMENTED,
            "External column metadata not supported");
        return NULL;
    }

    /* Get schema info */
    int32_t schema_idx = reader->schema->leaf_indices[column_index];
    const parquet_schema_element_t* schema_elem = &reader->schema->elements[schema_idx];

    col_reader->max_def_level = reader->schema->max_def_levels[column_index];
    col_reader->max_rep_level = reader->schema->max_rep_levels[column_index];
    col_reader->type = col_reader->col_meta->type;
    col_reader->type_length = schema_elem->type_length;

    col_reader->values_remaining = col_reader->col_meta->num_values;
    col_reader->data_start_offset = col_reader->col_meta->data_page_offset;

    return col_reader;
}

void carquet_column_reader_free(carquet_column_reader_t* reader) {
    if (!reader) return;

    free(reader->page_buffer);
    free(reader->page_data_for_values);
    free(reader->dictionary_data);
    free(reader->dictionary_offsets);

    /* Only free decoded_values if we own the memory (not a mmap view) */
    if (reader->decoded_ownership == CARQUET_DATA_OWNED) {
        free(reader->decoded_values);
    }

    /* Levels are always owned (decoded from RLE) */
    free(reader->decoded_def_levels);
    free(reader->decoded_rep_levels);
    free(reader->indices_buffer);
    free(reader);
}

bool carquet_column_has_next(const carquet_column_reader_t* reader) {
    /* reader is nonnull per API contract */
    return reader->values_remaining > 0;
}

int64_t carquet_column_remaining(const carquet_column_reader_t* reader) {
    /* reader is nonnull per API contract */
    return reader->values_remaining;
}

/* ============================================================================
 * Memory Mapping API
 * ============================================================================
 */

bool carquet_reader_is_mmap(const carquet_reader_t* reader) {
    /* reader is nonnull per API contract */
    return reader->mmap_info != NULL && reader->mmap_info->is_valid;
}

bool carquet_reader_can_zero_copy(
    const carquet_reader_t* reader,
    int32_t row_group_index,
    int32_t column_index) {

    /* reader is nonnull per API contract */

    /* Must have mmap enabled */
    if (!reader->mmap_info || !reader->mmap_info->is_valid) {
        return false;
    }

    /* Validate indices */
    if (row_group_index < 0 || row_group_index >= reader->metadata.num_row_groups) {
        return false;
    }
    if (column_index < 0 || column_index >= reader->schema->num_leaves) {
        return false;
    }

    const parquet_row_group_t* rg = &reader->metadata.row_groups[row_group_index];
    if (column_index >= rg->num_columns) {
        return false;
    }

    const parquet_column_chunk_t* chunk = &rg->columns[column_index];
    if (!chunk->has_metadata) {
        return false;
    }

    const parquet_column_metadata_t* col_meta = &chunk->metadata;

    /* Must be uncompressed */
    if (col_meta->codec != CARQUET_COMPRESSION_UNCOMPRESSED) {
        return false;
    }

    /* Check if column has definition levels (nullable) */
    int16_t max_def = reader->schema->max_def_levels[column_index];
    if (max_def > 0) {
        return false;  /* Nullable columns need level decoding */
    }

    /* Check physical type - must be fixed-size */
    carquet_physical_type_t type = col_meta->type;
    switch (type) {
        case CARQUET_PHYSICAL_INT32:
        case CARQUET_PHYSICAL_INT64:
        case CARQUET_PHYSICAL_FLOAT:
        case CARQUET_PHYSICAL_DOUBLE:
        case CARQUET_PHYSICAL_INT96:
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            return true;

        case CARQUET_PHYSICAL_BOOLEAN:
        case CARQUET_PHYSICAL_BYTE_ARRAY:
        default:
            return false;
    }
}

/* ============================================================================
 * Library Version
 * ============================================================================
 */

const char* carquet_version(void) {
    return CARQUET_VERSION_STRING;
}

void carquet_version_components(int* major, int* minor, int* patch) {
    if (major) *major = CARQUET_VERSION_MAJOR;
    if (minor) *minor = CARQUET_VERSION_MINOR;
    if (patch) *patch = CARQUET_VERSION_PATCH;
}
