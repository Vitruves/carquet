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

#include "core/allocator.h"
#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/buffer.h"
#include "core/arena.h"
#include "core/compat.h"
#include "reader/reader_internal.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include "writer/arrow_schema.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* Parquet magic bytes */
static const uint8_t PARQUET_MAGIC[4] = {'P', 'A', 'R', '1'};
extern int64_t carquet_dispatch_count_non_nulls(const int16_t* def_levels, int64_t count,
                                                 int16_t max_def_level);

/* Forward declaration from row_group_writer.c */
typedef struct carquet_row_group_writer carquet_row_group_writer_t;

typedef struct column_chunk_info {
    int64_t file_offset;
    int64_t total_compressed_size;
    int64_t total_uncompressed_size;
    int64_t num_values;
    carquet_physical_type_t type;
    carquet_logical_type_t logical_type;
    carquet_encoding_t encoding;
    carquet_compression_t compression;
    int32_t type_length;
    char* path;
    /* Aggregated column statistics (mirrors row_group_writer.c). Min and max
     * are heap-allocated and may have different sizes (BYTE_ARRAY). */
    bool has_min_max;
    uint8_t* min_value;
    size_t min_value_size;
    uint8_t* max_value;
    size_t max_value_size;
    int64_t null_count;
    bool has_null_count;
    /* Dictionary page plumbing (mirrors row_group_writer.c). */
    bool has_dictionary_page;
    int64_t dictionary_page_size;
    /* GeospatialStatistics (mirrors row_group_writer.c). */
    bool has_geo_stats;
    parquet_geospatial_statistics_t geo_stats;
} column_chunk_info_t;

extern carquet_row_group_writer_t* carquet_row_group_writer_create(
    const carquet_schema_t* schema,
    carquet_compression_t compression,
    size_t target_page_size,
    int64_t file_offset);

extern void carquet_row_group_writer_destroy(carquet_row_group_writer_t* writer);
extern void carquet_row_group_writer_reset(
    carquet_row_group_writer_t* writer,
    int64_t file_offset);

extern carquet_status_t carquet_row_group_writer_add_column(
    carquet_row_group_writer_t* writer,
    const char* name,
    carquet_physical_type_t type,
    const carquet_logical_type_t* logical_type,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int32_t compression_level);

extern void carquet_row_group_writer_configure_column_bloom(
    carquet_row_group_writer_t* writer,
    int column_index, bool enabled, int64_t ndv, double fpp);
extern void carquet_row_group_writer_set_column_max_rows_per_page(
    carquet_row_group_writer_t* writer,
    int column_index, int64_t max_rows);
extern void carquet_row_group_writer_set_column_write_batch_size(
    carquet_row_group_writer_t* writer,
    int column_index, int64_t batch_size);
extern void carquet_row_group_writer_set_column_data_page_v2(
    carquet_row_group_writer_t* writer,
    int column_index, bool enabled);

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

extern carquet_status_t carquet_row_group_writer_write_to_file(
    carquet_row_group_writer_t* writer,
    FILE* file,
    size_t* total_size,
    int64_t num_rows);

extern int carquet_row_group_writer_num_columns(const carquet_row_group_writer_t* writer);
extern int64_t carquet_row_group_writer_num_rows(const carquet_row_group_writer_t* writer);
extern int64_t carquet_row_group_writer_total_byte_size(const carquet_row_group_writer_t* writer);
extern const column_chunk_info_t* carquet_row_group_writer_get_column_info(
    const carquet_row_group_writer_t* writer, int index);

extern void carquet_row_group_writer_set_options(
    carquet_row_group_writer_t* writer,
    bool write_bloom_filters, bool write_page_index,
    bool write_statistics,
    bool write_crc,
    int32_t compression_level,
    int64_t dictionary_page_size);

/* Bloom filter and page index accessors */
typedef struct carquet_bloom_filter carquet_bloom_filter_t;
typedef struct carquet_column_index_builder carquet_column_index_builder_t;
typedef struct carquet_offset_index_builder carquet_offset_index_builder_t;

extern carquet_bloom_filter_t* carquet_row_group_writer_get_bloom_filter(
    const carquet_row_group_writer_t* writer, int index);
extern carquet_column_index_builder_t* carquet_row_group_writer_get_column_index(
    const carquet_row_group_writer_t* writer, int index);
extern carquet_offset_index_builder_t* carquet_row_group_writer_get_offset_index(
    const carquet_row_group_writer_t* writer, int index);

extern const uint8_t* carquet_bloom_filter_data(const carquet_bloom_filter_t* filter);
extern size_t carquet_bloom_filter_size(const carquet_bloom_filter_t* filter);
extern carquet_status_t carquet_column_index_serialize(
    const carquet_column_index_builder_t* builder, carquet_buffer_t* output);
extern carquet_status_t carquet_offset_index_serialize(
    const carquet_offset_index_builder_t* builder, carquet_buffer_t* output);

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
    bool statistics_sort_order_defined;
} writer_column_def_t;

/* ============================================================================
 * Row Group Metadata Storage
 * ============================================================================
 */

typedef struct row_group_column_info {
    int64_t file_offset;
    int64_t total_compressed_size;
    int64_t total_uncompressed_size;
    int64_t num_values;
    carquet_physical_type_t type;
    carquet_compression_t codec;
    int64_t data_page_offset;
    bool has_dictionary_page_offset;
    int64_t dictionary_page_offset;
    bool has_dictionary_page;
    /* Per-chunk ColumnMetaData.encodings, derived from whether this chunk
     * actually emitted a dictionary page (post-finalize), not from the static
     * configured-encoding cache. A dictionary chunk advertises
     * {PLAIN, RLE_DICTIONARY, RLE}; a plain/dict-fallback chunk advertises
     * {<data encoding>, RLE}. */
    carquet_encoding_t encodings[3];
    int32_t num_encodings;
    bool has_bloom_filter_offset;
    int64_t bloom_filter_offset;
    bool has_bloom_filter_length;
    int32_t bloom_filter_length;
    bool has_column_index_offset;
    int64_t column_index_offset;
    bool has_column_index_length;
    int32_t column_index_length;
    bool has_offset_index_offset;
    int64_t offset_index_offset;
    bool has_offset_index_length;
    int32_t offset_index_length;
    /* Aggregated column statistics */
    bool has_statistics;
    bool has_min_max;
    bool has_null_count;
    int64_t null_count;
    /* Heap-allocated; freed when the row_group_info_t is torn down. */
    uint8_t* min_value;
    int32_t min_value_size;
    uint8_t* max_value;
    int32_t max_value_size;
    /* GeospatialStatistics (GEOMETRY/GEOGRAPHY) */
    bool has_geo_stats;
    parquet_geospatial_statistics_t geo_stats;
} row_group_column_info_t;

typedef struct row_group_info {
    int64_t file_offset;
    int64_t num_rows;
    int64_t total_byte_size;
    int64_t total_compressed_size;
    int16_t ordinal;
    row_group_column_info_t* columns;
    int32_t num_columns;
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

    /* Full schema elements (including groups) for metadata serialization */
    parquet_schema_element_t* schema_elements;
    int32_t num_schema_elements;
    char*** column_paths;
    int32_t* column_path_lens;
    /* Per-column encodings list for ColumnMetaData.encodings. For dictionary
     * columns this is {PLAIN, RLE_DICTIONARY, RLE}; for plain columns
     * {<data encoding>, RLE}. column_num_encodings holds the live count. */
    carquet_encoding_t (*column_encodings)[3];
    int32_t* column_num_encodings;

    /* Options */
    carquet_writer_options_t options;

    /* Current row group */
    carquet_row_group_writer_t* current_row_group;
    int64_t current_row_group_rows;
    int64_t* column_values_written;  /* Values written per column in current row group */
    int64_t current_row_group_estimated_bytes;

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

    /* Key-value metadata */
    parquet_key_value_t* kv_metadata;
    int32_t num_kv_metadata;
    int32_t kv_metadata_capacity;

    /* Per-column overrides. _set flags distinguish "not overridden" from an
       overriding value of 0 (PLAIN / UNCOMPRESSED are both 0 in their enums). */
    carquet_encoding_t* column_encoding_overrides;
    bool* column_encoding_override_set;
    carquet_compression_t* column_compression_overrides;
    int32_t* column_compression_levels;
    bool* column_compression_override_set;
    bool* column_statistics_overrides;
    bool* column_bloom_filter_overrides;
    /* Per-column bloom NDV/FPP overrides (parallel arrays). _set distinguishes
       "not overridden" from an explicit value. */
    int64_t* column_bloom_ndv_overrides;
    double* column_bloom_fpp_overrides;
    bool* column_bloom_options_set;
    bool column_overrides_allocated;

    /* Sorting columns metadata, applied to every row group */
    parquet_sorting_column_t* sorting_columns;
    int32_t num_sorting_columns;

    /* Buffer writer support */
    bool is_buffer_writer;
    uint8_t* output_buffer;
    size_t output_buffer_size;
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
    options->page_size = 1024 * 1024;               /* 1 MB */
    options->write_statistics = true;
    options->write_crc = true;
    options->write_page_index = false;
    options->write_bloom_filters = false;
    options->dictionary_encoding = CARQUET_ENCODING_RLE_DICTIONARY;
    options->dictionary_page_size = 1024 * 1024;   /* 1 MB */
    options->created_by = "Carquet";
    options->max_rows_per_page = 0;                 /* unlimited */
    options->write_arrow_schema = false;
    options->data_page_version = 1;
    options->coerce_timestamps = false;
    options->coerce_timestamp_unit = CARQUET_TIME_UNIT_MICROS;
    options->allow_timestamp_truncation = false;
    options->write_batch_size = 0;
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

static int64_t saturating_add_i64(int64_t lhs, int64_t rhs) {
    if (rhs <= 0 || lhs >= INT64_MAX - rhs) {
        return INT64_MAX;
    }
    return lhs + rhs;
}

static int64_t estimate_column_batch_bytes(
    const writer_column_def_t* column,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    if (!column || !values || num_values <= 0) {
        return 0;
    }

    int64_t total = 0;

    switch (column->physical_type) {
        case CARQUET_PHYSICAL_BOOLEAN:
            total = num_values * (int64_t)sizeof(uint8_t);
            break;
        case CARQUET_PHYSICAL_INT32:
            total = num_values * (int64_t)sizeof(int32_t);
            break;
        case CARQUET_PHYSICAL_INT64:
            total = num_values * (int64_t)sizeof(int64_t);
            break;
        case CARQUET_PHYSICAL_FLOAT:
            total = num_values * (int64_t)sizeof(float);
            break;
        case CARQUET_PHYSICAL_DOUBLE:
            total = num_values * (int64_t)sizeof(double);
            break;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            if (column->type_length > 0) {
                total = num_values * (int64_t)column->type_length;
            }
            break;
        case CARQUET_PHYSICAL_BYTE_ARRAY: {
            const carquet_byte_array_t* arrays = (const carquet_byte_array_t*)values;
            for (int64_t i = 0; i < num_values; i++) {
                total = saturating_add_i64(total, sizeof(uint32_t));
                total = saturating_add_i64(total, arrays[i].length);
            }
            break;
        }
        default:
            break;
    }

    if (column->max_def_level > 0 && def_levels) {
        total = saturating_add_i64(total, num_values * (int64_t)sizeof(*def_levels));
    }
    if (column->max_rep_level > 0 && rep_levels) {
        total = saturating_add_i64(total, num_values * (int64_t)sizeof(*rep_levels));
    }

    return total;
}

static bool writer_supports_aligned_auto_flush(const carquet_writer_t* writer) {
    if (!writer || writer->options.row_group_size <= 0 || writer->num_columns <= 0) {
        return false;
    }

    for (int32_t i = 0; i < writer->num_columns; i++) {
        if (writer->columns[i].max_rep_level > 0) {
            return false;
        }
    }

    return true;
}

static bool current_row_group_is_aligned(const carquet_writer_t* writer) {
    if (!writer || writer->current_row_group_rows <= 0) {
        return false;
    }

    int64_t expected_rows = writer->current_row_group_rows;
    for (int32_t i = 0; i < writer->num_columns; i++) {
        if (writer->column_values_written[i] != expected_rows) {
            return false;
        }
    }

    return true;
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
    int32_t type_length,
    int16_t max_def_level,
    int16_t max_rep_level,
    bool statistics_sort_order_defined) {

    /* Expand capacity if needed */
    if (writer->num_columns >= writer->column_capacity) {
        int32_t new_cap = writer->column_capacity == 0 ? 8 : writer->column_capacity * 2;
        writer_column_def_t* new_cols = carquet_mem_realloc(writer->columns,
            new_cap * sizeof(writer_column_def_t));
        if (!new_cols) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        writer->columns = new_cols;

        int64_t* new_values = carquet_mem_realloc(writer->column_values_written,
            new_cap * sizeof(int64_t));
        if (!new_values) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        writer->column_values_written = new_values;

        writer->column_capacity = new_cap;
    }

    writer_column_def_t* col = &writer->columns[writer->num_columns];
    memset(col, 0, sizeof(*col));

    col->name = carquet_heap_strdup(name);
    if (!col->name) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    col->physical_type = physical_type;
    col->repetition = repetition;
    col->type_length = type_length;

    if (logical_type) {
        col->logical_type = *logical_type;
    }

    col->max_def_level = max_def_level;
    col->max_rep_level = max_rep_level;
    col->statistics_sort_order_defined = statistics_sort_order_defined;

    writer->column_values_written[writer->num_columns] = 0;
    writer->num_columns++;

    return CARQUET_OK;
}

/* Store the full schema elements (including groups) for metadata serialization */
static carquet_status_t store_schema_elements(
    carquet_writer_t* writer,
    const carquet_schema_t* schema) {

    writer->num_schema_elements = schema->num_elements;
    writer->schema_elements = carquet_mem_calloc(schema->num_elements, sizeof(parquet_schema_element_t));
    if (!writer->schema_elements) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    bool coerce = writer->options.coerce_timestamps;
    carquet_time_unit_t tgt = writer->options.coerce_timestamp_unit;

    for (int32_t i = 0; i < schema->num_elements; i++) {
        writer->schema_elements[i] = schema->elements[i];
        if (schema->elements[i].name) {
            writer->schema_elements[i].name = carquet_heap_strdup(schema->elements[i].name);
            if (!writer->schema_elements[i].name) {
                return CARQUET_ERROR_OUT_OF_MEMORY;
            }
        }

        /* Coerce TIMESTAMP units in the emitted schema so file metadata
         * reflects the target unit (the values are rescaled on write). */
        if (coerce) {
            parquet_schema_element_t* e = &writer->schema_elements[i];
            if (e->has_logical_type &&
                e->logical_type.id == CARQUET_LOGICAL_TIMESTAMP) {
                e->logical_type.params.timestamp.unit = tgt;
                if (e->has_converted_type) {
                    if (tgt == CARQUET_TIME_UNIT_MILLIS) {
                        e->converted_type = CARQUET_CONVERTED_TIMESTAMP_MILLIS;
                    } else if (tgt == CARQUET_TIME_UNIT_MICROS) {
                        e->converted_type = CARQUET_CONVERTED_TIMESTAMP_MICROS;
                    } else {
                        /* NANOS has no legacy ConvertedType. */
                        e->has_converted_type = false;
                    }
                }
            }
        }
    }

    return CARQUET_OK;
}

static carquet_compression_t effective_column_compression(
    const carquet_writer_t* writer, int32_t column_index);
static int32_t effective_column_compression_level(
    const carquet_writer_t* writer, int32_t column_index);
static carquet_encoding_t effective_column_encoding(
    const carquet_writer_t* writer, int32_t column_index,
    carquet_physical_type_t type, carquet_compression_t compression);

static carquet_status_t build_column_metadata_cache(
    carquet_writer_t* writer,
    const carquet_schema_t* schema) {

    writer->column_paths = carquet_mem_calloc((size_t)schema->num_leaves, sizeof(char**));
    writer->column_path_lens = carquet_mem_calloc((size_t)schema->num_leaves, sizeof(int32_t));
    writer->column_encodings = carquet_mem_calloc((size_t)schema->num_leaves,
        sizeof(*writer->column_encodings));
    writer->column_num_encodings = carquet_mem_calloc((size_t)schema->num_leaves,
        sizeof(int32_t));

    if (!writer->column_paths || !writer->column_path_lens ||
        !writer->column_encodings || !writer->column_num_encodings) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < schema->num_leaves; i++) {
        int32_t elem_idx = schema->leaf_indices[i];
        int32_t depth = 0;

        for (int32_t cur = elem_idx; cur > 0; cur = schema->parent_indices[cur]) {
            depth++;
        }

        if (depth <= 0) {
            depth = 1;
        }

        writer->column_paths[i] = carquet_mem_calloc((size_t)depth, sizeof(char*));
        if (!writer->column_paths[i]) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        writer->column_path_lens[i] = depth;
        if (depth == 1) {
            writer->column_paths[i][0] = writer->schema_elements[elem_idx].name;
        } else {
            int32_t cur = elem_idx;
            for (int32_t pi = depth - 1; pi >= 0; pi--) {
                writer->column_paths[i][pi] = writer->schema_elements[cur].name;
                cur = schema->parent_indices[cur];
            }
        }
    }

    return CARQUET_OK;
}

static bool leaf_statistics_sort_order_defined(
    const carquet_schema_t* schema,
    int32_t leaf_elem_idx) {

    const parquet_schema_element_t* leaf = &schema->elements[leaf_elem_idx];
    if (leaf->has_logical_type) {
        switch (leaf->logical_type.id) {
            case CARQUET_LOGICAL_GEOMETRY:
            case CARQUET_LOGICAL_GEOGRAPHY:
            case CARQUET_LOGICAL_VARIANT:
                return false;
            default:
                break;
        }
    }

    for (int32_t cur = schema->parent_indices[leaf_elem_idx]; cur > 0;
         cur = schema->parent_indices[cur]) {
        const parquet_schema_element_t* elem = &schema->elements[cur];
        if (elem->has_logical_type &&
            elem->logical_type.id == CARQUET_LOGICAL_VARIANT) {
            return false;
        }
    }

    return true;
}

static carquet_status_t ensure_column_overrides(carquet_writer_t* writer) {
    if (writer->column_overrides_allocated) return CARQUET_OK;
    int32_t n = writer->num_columns;
    writer->column_encoding_overrides = carquet_mem_calloc(n, sizeof(carquet_encoding_t));
    writer->column_encoding_override_set = carquet_mem_calloc(n, sizeof(bool));
    writer->column_compression_overrides = carquet_mem_calloc(n, sizeof(carquet_compression_t));
    writer->column_compression_levels = carquet_mem_calloc(n, sizeof(int32_t));
    writer->column_compression_override_set = carquet_mem_calloc(n, sizeof(bool));
    writer->column_statistics_overrides = carquet_mem_calloc(n, sizeof(bool));
    writer->column_bloom_filter_overrides = carquet_mem_calloc(n, sizeof(bool));
    writer->column_bloom_ndv_overrides = carquet_mem_calloc(n, sizeof(int64_t));
    writer->column_bloom_fpp_overrides = carquet_mem_calloc(n, sizeof(double));
    writer->column_bloom_options_set = carquet_mem_calloc(n, sizeof(bool));
    if (!writer->column_encoding_overrides || !writer->column_encoding_override_set ||
        !writer->column_compression_overrides || !writer->column_compression_levels ||
        !writer->column_compression_override_set ||
        !writer->column_statistics_overrides || !writer->column_bloom_filter_overrides ||
        !writer->column_bloom_ndv_overrides || !writer->column_bloom_fpp_overrides ||
        !writer->column_bloom_options_set) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    for (int32_t i = 0; i < n; i++) {
        writer->column_statistics_overrides[i] = writer->options.write_statistics;
        writer->column_bloom_filter_overrides[i] = writer->options.write_bloom_filters;
    }
    writer->column_overrides_allocated = true;
    return CARQUET_OK;
}

static void free_column_overrides(carquet_writer_t* writer) {
    carquet_mem_free(writer->column_encoding_overrides);
    carquet_mem_free(writer->column_encoding_override_set);
    carquet_mem_free(writer->column_compression_overrides);
    carquet_mem_free(writer->column_compression_levels);
    carquet_mem_free(writer->column_compression_override_set);
    carquet_mem_free(writer->column_statistics_overrides);
    carquet_mem_free(writer->column_bloom_filter_overrides);
    carquet_mem_free(writer->column_bloom_ndv_overrides);
    carquet_mem_free(writer->column_bloom_fpp_overrides);
    carquet_mem_free(writer->column_bloom_options_set);
    writer->column_encoding_overrides = NULL;
    writer->column_encoding_override_set = NULL;
    writer->column_compression_overrides = NULL;
    writer->column_compression_levels = NULL;
    writer->column_compression_override_set = NULL;
    writer->column_statistics_overrides = NULL;
    writer->column_bloom_filter_overrides = NULL;
    writer->column_bloom_ndv_overrides = NULL;
    writer->column_bloom_fpp_overrides = NULL;
    writer->column_bloom_options_set = NULL;
    writer->column_overrides_allocated = false;
}

static bool any_column_bloom_options_set(const carquet_writer_t* writer) {
    if (!writer->column_overrides_allocated || !writer->column_bloom_options_set) {
        return false;
    }
    for (int32_t i = 0; i < writer->num_columns; i++) {
        if (writer->column_bloom_options_set[i]) return true;
    }
    return false;
}

static carquet_compression_t effective_column_compression(
    const carquet_writer_t* writer, int32_t column_index) {
    carquet_compression_t codec;
    if (writer->column_overrides_allocated &&
        writer->column_compression_override_set[column_index]) {
        codec = writer->column_compression_overrides[column_index];
    } else {
        codec = writer->options.compression;
    }

    /* carquet writes raw LZ4 blocks, which are represented by the LZ4_RAW
     * Parquet codec. The legacy LZ4 enum has a different deprecated framing. */
    return codec == CARQUET_COMPRESSION_LZ4 ? CARQUET_COMPRESSION_LZ4_RAW : codec;
}

static int32_t effective_column_compression_level(
    const carquet_writer_t* writer, int32_t column_index) {
    if (writer->column_overrides_allocated &&
        writer->column_compression_override_set[column_index]) {
        return writer->column_compression_levels[column_index];
    }
    return writer->options.compression_level;
}

static carquet_encoding_t effective_column_encoding(
    const carquet_writer_t* writer,
    int32_t column_index,
    carquet_physical_type_t type,
    carquet_compression_t compression) {
    if (writer->column_overrides_allocated &&
        writer->column_encoding_override_set[column_index]) {
        return writer->column_encoding_overrides[column_index];
    }
    /* Default encoding policy (matches v0.4.4): PLAIN, with automatic
     * BYTE_STREAM_SPLIT for FLOAT/DOUBLE when a compression codec is set
     * (BSS makes the float byte planes far more compressible). Dictionary
     * encoding is a deliberate opt-in via carquet_writer_set_column_encoding()
     * — making it the default regressed read throughput badly (notably a
     * ~270x slowdown on the zero-copy uncompressed read path) for a size win
     * that is zero under zstd. The full dictionary writer remains available. */
    if (compression != CARQUET_COMPRESSION_UNCOMPRESSED &&
        (type == CARQUET_PHYSICAL_FLOAT || type == CARQUET_PHYSICAL_DOUBLE)) {
        return CARQUET_ENCODING_BYTE_STREAM_SPLIT;
    }
    return CARQUET_ENCODING_PLAIN;
}

/* Recompute the per-column ColumnMetaData.encodings cache. Must run after
 * per-column encoding/compression overrides are applied (the cache built at
 * writer-create time predates them), i.e. just before the first row group is
 * created. Dictionary columns advertise {PLAIN, RLE_DICTIONARY, RLE}; other
 * columns {<data encoding>, RLE}. This cache reflects the *configured*
 * encoding only; the actually-emitted ColumnMetaData.encodings is overridden
 * per chunk after finalize using row_group_column_info_t.encodings (see the
 * chunk-assembly loop), so a dict column that fell back to PLAIN advertises
 * only {PLAIN, RLE} and never the extra RLE_DICTIONARY entry. */
static void refresh_column_encodings_cache(carquet_writer_t* writer) {
    for (int32_t i = 0; i < writer->num_columns; i++) {
        carquet_physical_type_t phys = writer->columns[i].physical_type;
        carquet_compression_t col_comp = effective_column_compression(writer, i);
        carquet_encoding_t enc =
            effective_column_encoding(writer, i, phys, col_comp);
        if (enc == CARQUET_ENCODING_RLE_DICTIONARY ||
            enc == CARQUET_ENCODING_PLAIN_DICTIONARY) {
            writer->column_encodings[i][0] = CARQUET_ENCODING_PLAIN;
            writer->column_encodings[i][1] = CARQUET_ENCODING_RLE_DICTIONARY;
            writer->column_encodings[i][2] = CARQUET_ENCODING_RLE;
            writer->column_num_encodings[i] = 3;
        } else {
            writer->column_encodings[i][0] = enc;
            writer->column_encodings[i][1] = CARQUET_ENCODING_RLE;
            writer->column_num_encodings[i] = 2;
        }
    }
}

static bool writer_encoding_supported(
    carquet_encoding_t encoding,
    carquet_physical_type_t type) {

    switch (encoding) {
        case CARQUET_ENCODING_PLAIN:
            return true;
        case CARQUET_ENCODING_BYTE_STREAM_SPLIT:
            /* Parquet spec permits BYTE_STREAM_SPLIT for these physical
             * types (not just FLOAT/DOUBLE). */
            return type == CARQUET_PHYSICAL_FLOAT ||
                   type == CARQUET_PHYSICAL_DOUBLE ||
                   type == CARQUET_PHYSICAL_INT32 ||
                   type == CARQUET_PHYSICAL_INT64 ||
                   type == CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY;
        case CARQUET_ENCODING_DELTA_BINARY_PACKED:
            return type == CARQUET_PHYSICAL_INT32 ||
                   type == CARQUET_PHYSICAL_INT64;
        case CARQUET_ENCODING_DELTA_LENGTH_BYTE_ARRAY:
            return type == CARQUET_PHYSICAL_BYTE_ARRAY;
        case CARQUET_ENCODING_DELTA_BYTE_ARRAY:
            /* Spec permits DELTA_BYTE_ARRAY for BYTE_ARRAY and FLBA. */
            return type == CARQUET_PHYSICAL_BYTE_ARRAY ||
                   type == CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY;
        case CARQUET_ENCODING_RLE_DICTIONARY:
        case CARQUET_ENCODING_PLAIN_DICTIONARY:
            return type == CARQUET_PHYSICAL_INT32 ||
                   type == CARQUET_PHYSICAL_INT64 ||
                   type == CARQUET_PHYSICAL_FLOAT ||
                   type == CARQUET_PHYSICAL_DOUBLE ||
                   type == CARQUET_PHYSICAL_BYTE_ARRAY;
        default:
            return false;
    }
}

static void free_row_groups(carquet_writer_t* writer) {
    if (!writer->row_groups) return;
    for (int32_t i = 0; i < writer->num_row_groups; i++) {
        row_group_info_t* rg = &writer->row_groups[i];
        if (rg->columns) {
            for (int32_t j = 0; j < rg->num_columns; j++) {
                carquet_mem_free(rg->columns[j].min_value);
                carquet_mem_free(rg->columns[j].max_value);
            }
            carquet_mem_free(rg->columns);
        }
    }
    carquet_mem_free(writer->row_groups);
    writer->row_groups = NULL;
    writer->num_row_groups = 0;
}

static void free_kv_metadata(carquet_writer_t* writer) {
    if (writer->kv_metadata) {
        for (int32_t i = 0; i < writer->num_kv_metadata; i++) {
            carquet_mem_free(writer->kv_metadata[i].key);
            carquet_mem_free(writer->kv_metadata[i].value);
        }
        carquet_mem_free(writer->kv_metadata);
        writer->kv_metadata = NULL;
    }
}

static carquet_status_t ensure_row_group(carquet_writer_t* writer) {
    if (writer->current_row_group) {
        return CARQUET_OK;
    }

    /* Overrides may have been set after writer-create; refresh the encodings
     * cache so ColumnMetaData.encodings reflects the resolved encoding. */
    refresh_column_encodings_cache(writer);

    size_t target_page_size = (size_t)writer->options.page_size;

    writer->current_row_group = carquet_row_group_writer_create(
        NULL,  /* Schema not used directly */
        writer->options.compression,
        target_page_size,
        writer->file_offset);

    if (!writer->current_row_group) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    /* Pass optional feature flags */
    carquet_row_group_writer_set_options(
        writer->current_row_group,
        writer->options.write_bloom_filters,
        writer->options.write_page_index,
        writer->options.write_statistics,
        writer->options.write_crc,
        writer->options.compression_level,
        writer->options.dictionary_page_size);

    /* Add all columns to the row group writer, resolving any per-column
       encoding/compression overrides into explicit values */
    for (int32_t i = 0; i < writer->num_columns; i++) {
        writer_column_def_t* col = &writer->columns[i];
        carquet_compression_t col_comp = effective_column_compression(writer, i);
        int32_t col_level = effective_column_compression_level(writer, i);
        carquet_encoding_t col_enc = effective_column_encoding(
            writer, i, col->physical_type, col_comp);
        /* When the leaf has no defined min/max sort order we pass a synthetic
         * logical type so the page writer suppresses min/max. GEOMETRY and
         * GEOGRAPHY are the exception: their real logical type must reach the
         * page writer so it accumulates GeospatialStatistics (min/max is
         * still suppressed for them by stats_order_defined_for_logical). */
        bool is_geo =
            (col->logical_type.id == CARQUET_LOGICAL_GEOMETRY ||
             col->logical_type.id == CARQUET_LOGICAL_GEOGRAPHY);
        carquet_logical_type_t no_stats_logical = { .id = CARQUET_LOGICAL_VARIANT };
        const carquet_logical_type_t* writer_logical_type =
            (col->statistics_sort_order_defined || is_geo)
                ? &col->logical_type : &no_stats_logical;

        carquet_status_t status = carquet_row_group_writer_add_column(
            writer->current_row_group,
            col->name,
            col->physical_type,
            writer_logical_type,
            col->max_def_level,
            col->max_rep_level,
            col->type_length,
            col_enc,
            col_comp,
            col_level);

        if (status != CARQUET_OK) {
            carquet_row_group_writer_destroy(writer->current_row_group);
            writer->current_row_group = NULL;
            return status;
        }

        /* Apply the global max_rows_per_page knob (no-op when 0). */
        if (writer->options.max_rows_per_page > 0) {
            carquet_row_group_writer_set_column_max_rows_per_page(
                writer->current_row_group, i,
                writer->options.max_rows_per_page);
        }

        /* Apply the global write_batch_size knob (no-op when 0). */
        if (writer->options.write_batch_size > 0) {
            carquet_row_group_writer_set_column_write_batch_size(
                writer->current_row_group, i,
                writer->options.write_batch_size);
        }

        /* Opt-in Data Page V2 output. */
        if (writer->options.data_page_version == 2) {
            carquet_row_group_writer_set_column_data_page_v2(
                writer->current_row_group, i, true);
        }

        /* Apply per-column bloom NDV/FPP overrides. Once the new options API
         * has been used for ANY column, take full per-column control so a
         * column the user did not opt in does not silently get the default
         * bloom filter that the (now-on) global flag would create. When the
         * new API was never used this whole block is skipped, keeping the
         * legacy global-flag behavior byte-identical. */
        if (writer->column_overrides_allocated &&
            writer->column_bloom_options_set &&
            any_column_bloom_options_set(writer)) {
            bool col_enabled = writer->column_bloom_options_set[i]
                ? writer->column_bloom_filter_overrides[i]
                : false;
            int64_t col_ndv = writer->column_bloom_options_set[i]
                ? writer->column_bloom_ndv_overrides[i] : 0;
            double col_fpp = writer->column_bloom_options_set[i]
                ? writer->column_bloom_fpp_overrides[i] : 0.0;
            carquet_row_group_writer_configure_column_bloom(
                writer->current_row_group, i,
                col_enabled, col_ndv, col_fpp);
        }
    }

    writer->current_row_group_rows = 0;
    writer->current_row_group_estimated_bytes = 0;
    for (int32_t i = 0; i < writer->num_columns; i++) {
        writer->column_values_written[i] = 0;
    }

    return CARQUET_OK;
}

static carquet_status_t flush_row_group(carquet_writer_t* writer) {
    if (!writer->current_row_group || writer->current_row_group_rows == 0) {
        return CARQUET_OK;
    }

    /* Finalize and write each column directly to file, avoiding
     * an intermediate copy of the entire row group into one buffer */
    size_t size;
    carquet_status_t status = carquet_row_group_writer_write_to_file(
        writer->current_row_group, writer->file, &size,
        writer->current_row_group_rows);

    if (status != CARQUET_OK) {
        return status;
    }

    /* Store row group metadata */
    if (writer->num_row_groups >= writer->row_groups_capacity) {
        int32_t new_cap = writer->row_groups_capacity == 0 ? 4 : writer->row_groups_capacity * 2;
        row_group_info_t* new_rgs = carquet_mem_realloc(writer->row_groups,
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
    rg_info->num_rows = writer->current_row_group_rows;
    rg_info->total_byte_size = carquet_row_group_writer_total_byte_size(writer->current_row_group);
    rg_info->total_compressed_size = (int64_t)size;
    rg_info->ordinal = (int16_t)writer->num_row_groups;

    /* Build column chunks metadata */
    int num_cols = carquet_row_group_writer_num_columns(writer->current_row_group);
    rg_info->num_columns = num_cols;
    rg_info->columns = carquet_mem_calloc((size_t)num_cols, sizeof(*rg_info->columns));
    if (!rg_info->columns) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < num_cols; i++) {
        const column_chunk_info_t* col_info = carquet_row_group_writer_get_column_info(
            writer->current_row_group, i);

        if (!col_info) continue;

        row_group_column_info_t* chunk = &rg_info->columns[i];
        chunk->file_offset = col_info->file_offset;
        chunk->type = col_info->type;
        chunk->codec = col_info->compression;
        chunk->num_values = col_info->num_values;
        chunk->total_compressed_size = col_info->total_compressed_size;
        chunk->total_uncompressed_size = col_info->total_uncompressed_size;
        if (col_info->has_dictionary_page) {
            /* The dictionary page is the first page of the chunk at
             * file_offset; data pages follow it. */
            chunk->has_dictionary_page = true;
            chunk->has_dictionary_page_offset = true;
            chunk->dictionary_page_offset = col_info->file_offset;
            chunk->data_page_offset =
                col_info->file_offset + col_info->dictionary_page_size;
        } else {
            chunk->data_page_offset = col_info->file_offset;
        }

        if (col_info->has_geo_stats) {
            chunk->has_geo_stats = true;
            chunk->geo_stats = col_info->geo_stats;
        }

        /* Derive the per-chunk encodings list from whether this chunk actually
         * emitted a dictionary page. The static refresh_column_encodings_cache
         * is computed from the configured encoding before finalize knows about
         * dictionary fallback, so a dict column that fell back to PLAIN (dict
         * exceeded dictionary_page_size or was all-unique) must NOT advertise
         * RLE_DICTIONARY here. Per the Parquet spec, ColumnMetaData.encodings
         * is the set of encodings actually used in the chunk. */
        if (col_info->has_dictionary_page) {
            chunk->encodings[0] = CARQUET_ENCODING_PLAIN;
            chunk->encodings[1] = CARQUET_ENCODING_RLE_DICTIONARY;
            chunk->encodings[2] = CARQUET_ENCODING_RLE;
            chunk->num_encodings = 3;
        } else {
            /* Plain column, or dictionary column that fell back to PLAIN. For
             * the fallback case the data encoding is PLAIN; for an explicitly
             * non-dictionary column (e.g. BYTE_STREAM_SPLIT) it is the
             * configured encoding. A configured dictionary encoding that
             * produced no dictionary page implies a PLAIN fallback. */
            carquet_encoding_t data_enc = col_info->encoding;
            if (data_enc == CARQUET_ENCODING_RLE_DICTIONARY ||
                data_enc == CARQUET_ENCODING_PLAIN_DICTIONARY) {
                data_enc = CARQUET_ENCODING_PLAIN;
            }
            chunk->encodings[0] = data_enc;
            chunk->encodings[1] = CARQUET_ENCODING_RLE;
            chunk->num_encodings = 2;
        }

        /* Per-column statistics override may suppress emission for one column
         * even when global write_statistics is enabled. */
        bool emit_stats = writer->column_overrides_allocated && i < writer->num_columns
            ? writer->column_statistics_overrides[i]
            : writer->options.write_statistics;
        if (i < writer->num_columns && !writer->columns[i].statistics_sort_order_defined) {
            emit_stats = false;
        }

        if (emit_stats && (col_info->has_min_max || col_info->has_null_count)) {
            chunk->has_statistics = true;
            chunk->has_min_max = col_info->has_min_max;
            chunk->has_null_count = col_info->has_null_count;
            chunk->null_count = col_info->null_count;
            if (col_info->has_min_max &&
                col_info->min_value_size > 0 &&
                col_info->max_value_size > 0) {
                chunk->min_value = carquet_mem_malloc(col_info->min_value_size);
                chunk->max_value = carquet_mem_malloc(col_info->max_value_size);
                if (chunk->min_value && chunk->max_value) {
                    memcpy(chunk->min_value, col_info->min_value,
                           col_info->min_value_size);
                    memcpy(chunk->max_value, col_info->max_value,
                           col_info->max_value_size);
                    chunk->min_value_size = (int32_t)col_info->min_value_size;
                    chunk->max_value_size = (int32_t)col_info->max_value_size;
                } else {
                    carquet_mem_free(chunk->min_value);
                    carquet_mem_free(chunk->max_value);
                    chunk->min_value = NULL;
                    chunk->max_value = NULL;
                    chunk->has_min_max = false;
                }
            }
        }
    }

    writer->num_row_groups++;
    writer->file_offset += (int64_t)size;
    writer->total_rows += writer->current_row_group_rows;

    /* Write bloom filters for each column (after row group data) */
    if (writer->options.write_bloom_filters) {
        for (int i = 0; i < num_cols; i++) {
            carquet_bloom_filter_t* bf = carquet_row_group_writer_get_bloom_filter(
                writer->current_row_group, i);
            if (!bf) continue;

            const uint8_t* bf_data = carquet_bloom_filter_data(bf);
            size_t bf_size = carquet_bloom_filter_size(bf);
            if (!bf_data || bf_size == 0) continue;

            /* Write Bloom Filter Header (Thrift):
             * numBytes: i32, algorithm: MURMUR3_X64_128, hash: XXHASH, compression: UNCOMPRESSED */
            carquet_buffer_t bf_header;
            carquet_buffer_init(&bf_header);
            {
                thrift_encoder_t enc;
                thrift_encoder_init(&enc, &bf_header);
                thrift_write_struct_begin(&enc);
                /* Field 1: numBytes (i32) */
                thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
                thrift_write_i32(&enc, (int32_t)bf_size);
                /* Field 2: algorithm (BloomFilterAlgorithm struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 2);
                thrift_write_struct_begin(&enc);
                /* Field 1: SPLIT_BLOCK_BLOOM_FILTER (empty struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 1);
                thrift_write_struct_begin(&enc);
                thrift_write_struct_end(&enc);
                thrift_write_struct_end(&enc);
                /* Field 3: hash (BloomFilterHash struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 3);
                thrift_write_struct_begin(&enc);
                /* Field 1: XXHASH (empty struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 1);
                thrift_write_struct_begin(&enc);
                thrift_write_struct_end(&enc);
                thrift_write_struct_end(&enc);
                /* Field 4: compression (BloomFilterCompression struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 4);
                thrift_write_struct_begin(&enc);
                /* Field 1: UNCOMPRESSED (empty struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 1);
                thrift_write_struct_begin(&enc);
                thrift_write_struct_end(&enc);
                thrift_write_struct_end(&enc);
                thrift_write_struct_end(&enc);
            }

            /* Record offset in column metadata */
            row_group_column_info_t* chunk = &rg_info->columns[i];
            chunk->has_bloom_filter_offset = true;
            chunk->bloom_filter_offset = writer->file_offset;
            chunk->has_bloom_filter_length = true;
            chunk->bloom_filter_length = (int32_t)(bf_header.size + bf_size);

            /* Write header + data */
            if (fwrite(bf_header.data, 1, bf_header.size, writer->file) != bf_header.size) {
                carquet_buffer_destroy(&bf_header);
                return CARQUET_ERROR_FILE_WRITE;
            }
            writer->file_offset += (int64_t)bf_header.size;
            carquet_buffer_destroy(&bf_header);

            if (fwrite(bf_data, 1, bf_size, writer->file) != bf_size) {
                return CARQUET_ERROR_FILE_WRITE;
            }
            writer->file_offset += (int64_t)bf_size;
        }
    }

    /* Write column indexes and offset indexes (after bloom filters) */
    if (writer->options.write_page_index) {
        for (int i = 0; i < num_cols; i++) {
            row_group_column_info_t* chunk = &rg_info->columns[i];

            /* Column index */
            carquet_column_index_builder_t* ci = carquet_row_group_writer_get_column_index(
                writer->current_row_group, i);
            if (ci) {
                carquet_buffer_t ci_buf;
                carquet_buffer_init(&ci_buf);
                carquet_column_index_serialize(ci, &ci_buf);
                if (ci_buf.size > 0) {
                    chunk->has_column_index_offset = true;
                    chunk->column_index_offset = writer->file_offset;
                    chunk->has_column_index_length = true;
                    chunk->column_index_length = (int32_t)ci_buf.size;
                    if (fwrite(ci_buf.data, 1, ci_buf.size, writer->file) != ci_buf.size) {
                        carquet_buffer_destroy(&ci_buf);
                        return CARQUET_ERROR_FILE_WRITE;
                    }
                    writer->file_offset += (int64_t)ci_buf.size;
                }
                carquet_buffer_destroy(&ci_buf);
            }

            /* Offset index */
            carquet_offset_index_builder_t* oi = carquet_row_group_writer_get_offset_index(
                writer->current_row_group, i);
            if (oi) {
                carquet_buffer_t oi_buf;
                carquet_buffer_init(&oi_buf);
                carquet_offset_index_serialize(oi, &oi_buf);
                if (oi_buf.size > 0) {
                    chunk->has_offset_index_offset = true;
                    chunk->offset_index_offset = writer->file_offset;
                    chunk->has_offset_index_length = true;
                    chunk->offset_index_length = (int32_t)oi_buf.size;
                    if (fwrite(oi_buf.data, 1, oi_buf.size, writer->file) != oi_buf.size) {
                        carquet_buffer_destroy(&oi_buf);
                        return CARQUET_ERROR_FILE_WRITE;
                    }
                    writer->file_offset += (int64_t)oi_buf.size;
                }
                carquet_buffer_destroy(&oi_buf);
            }
        }
    }

    /* Reuse the current row group writer for the next group. */
    carquet_row_group_writer_reset(writer->current_row_group, writer->file_offset);
    writer->current_row_group_rows = 0;
    writer->current_row_group_estimated_bytes = 0;
    for (int32_t i = 0; i < writer->num_columns; i++) {
        writer->column_values_written[i] = 0;
    }

    return CARQUET_OK;
}

static bool deprecated_stats_are_compatible(const writer_column_def_t* column) {
    if (!column) return false;
    if (column->logical_type.id == CARQUET_LOGICAL_INTEGER &&
        !column->logical_type.params.integer.is_signed) {
        return false;
    }
    if (!column->statistics_sort_order_defined) {
        return false;
    }
    return true;
}

static carquet_status_t build_file_metadata(
    carquet_writer_t* writer,
    parquet_file_metadata_t* metadata) {

    memset(metadata, 0, sizeof(*metadata));

    metadata->version = 2;  /* Parquet version 2 */
    metadata->num_rows = writer->total_rows;
    metadata->num_column_orders = writer->num_columns;
    metadata->created_by = carquet_arena_strdup(&writer->arena,
        writer->options.created_by ? writer->options.created_by : "Carquet");

    /* Optional "ARROW:schema" footer metadata. Skipped if the user already
     * supplied that key, if the schema is nested/unsupported, or on OOM. */
    char* arrow_schema_b64 = NULL;
    if (writer->options.write_arrow_schema) {
        bool user_set = false;
        for (int32_t i = 0; i < writer->num_kv_metadata; i++) {
            if (writer->kv_metadata[i].key &&
                strcmp(writer->kv_metadata[i].key, "ARROW:schema") == 0) {
                user_set = true;
                break;
            }
        }
        if (!user_set) {
            arrow_schema_b64 = carquet_build_arrow_schema_b64(
                writer->schema_elements, writer->num_schema_elements);
        }
    }
    int32_t extra_kv = arrow_schema_b64 ? 1 : 0;

    /* Key-value metadata */
    if (writer->num_kv_metadata + extra_kv > 0) {
        metadata->num_key_value = writer->num_kv_metadata + extra_kv;
        metadata->key_value_metadata = carquet_arena_calloc(&writer->arena,
            metadata->num_key_value, sizeof(parquet_key_value_t));
        if (!metadata->key_value_metadata) {
            carquet_mem_free(arrow_schema_b64);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        for (int32_t i = 0; i < writer->num_kv_metadata; i++) {
            metadata->key_value_metadata[i].key = carquet_arena_strdup(
                &writer->arena, writer->kv_metadata[i].key);
            metadata->key_value_metadata[i].value = writer->kv_metadata[i].value ?
                carquet_arena_strdup(&writer->arena, writer->kv_metadata[i].value) : NULL;
        }
        if (arrow_schema_b64) {
            parquet_key_value_t* kv =
                &metadata->key_value_metadata[writer->num_kv_metadata];
            kv->key = carquet_arena_strdup(&writer->arena, "ARROW:schema");
            kv->value = carquet_arena_strdup(&writer->arena, arrow_schema_b64);
            carquet_mem_free(arrow_schema_b64);
        }
    }

    /* Build schema from stored elements (includes groups for nested schemas) */
    metadata->num_schema_elements = writer->num_schema_elements;
    metadata->schema = carquet_arena_calloc(&writer->arena, writer->num_schema_elements,
        sizeof(parquet_schema_element_t));

    if (!metadata->schema) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < writer->num_schema_elements; i++) {
        metadata->schema[i] = writer->schema_elements[i];
        /* Duplicate strings into arena so they outlive the writer */
        if (writer->schema_elements[i].name) {
            metadata->schema[i].name = carquet_arena_strdup(
                &writer->arena, writer->schema_elements[i].name);
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
        const row_group_info_t* src_rg = &writer->row_groups[i];
        parquet_row_group_t* dst_rg = &metadata->row_groups[i];

        dst_rg->num_rows = src_rg->num_rows;
        dst_rg->total_byte_size = src_rg->total_byte_size;
        dst_rg->has_file_offset = true;
        dst_rg->file_offset = src_rg->file_offset;
        dst_rg->has_total_compressed_size = true;
        dst_rg->total_compressed_size = src_rg->total_compressed_size;
        dst_rg->has_ordinal = true;
        dst_rg->ordinal = src_rg->ordinal;
        dst_rg->sorting_columns = writer->sorting_columns;
        dst_rg->num_sorting_columns = writer->num_sorting_columns;
        dst_rg->num_columns = src_rg->num_columns;
        dst_rg->columns = carquet_arena_calloc(&writer->arena, src_rg->num_columns,
            sizeof(parquet_column_chunk_t));
        if (!dst_rg->columns && src_rg->num_columns > 0) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        for (int32_t j = 0; j < src_rg->num_columns; j++) {
            const row_group_column_info_t* src_col = &src_rg->columns[j];
            parquet_column_chunk_t* dst_chunk = &dst_rg->columns[j];
            parquet_column_metadata_t* meta = &dst_chunk->metadata;

            dst_chunk->file_offset = src_col->file_offset;
            dst_chunk->has_metadata = true;

            meta->type = src_col->type;
            meta->codec = src_col->codec;
            meta->num_values = src_col->num_values;
            meta->total_compressed_size = src_col->total_compressed_size;
            meta->total_uncompressed_size = src_col->total_uncompressed_size;
            meta->data_page_offset = src_col->data_page_offset;
            if (src_col->has_dictionary_page_offset) {
                meta->has_dictionary_page_offset = true;
                meta->dictionary_page_offset = src_col->dictionary_page_offset;
            }
            /* Use the per-chunk encodings derived post-finalize from the
             * actual dictionary-page presence rather than the static
             * configured-encoding cache, so a dict-to-PLAIN fallback chunk
             * does not falsely advertise RLE_DICTIONARY. Fall back to the
             * static cache only if the per-chunk list was never populated. */
            if (src_col->num_encodings > 0) {
                meta->num_encodings = src_col->num_encodings;
                meta->encodings = (carquet_encoding_t*)src_col->encodings;
            } else {
                meta->num_encodings = writer->column_num_encodings[j] > 0
                    ? writer->column_num_encodings[j] : 2;
                meta->encodings = writer->column_encodings[j];
            }
            meta->path_len = writer->column_path_lens[j];
            meta->path_in_schema = writer->column_paths[j];

            meta->has_bloom_filter_offset = src_col->has_bloom_filter_offset;
            meta->bloom_filter_offset = src_col->bloom_filter_offset;
            meta->has_bloom_filter_length = src_col->has_bloom_filter_length;
            meta->bloom_filter_length = src_col->bloom_filter_length;

            dst_chunk->has_column_index_offset = src_col->has_column_index_offset;
            dst_chunk->column_index_offset = src_col->column_index_offset;
            dst_chunk->has_column_index_length = src_col->has_column_index_length;
            dst_chunk->column_index_length = src_col->column_index_length;
            dst_chunk->has_offset_index_offset = src_col->has_offset_index_offset;
            dst_chunk->offset_index_offset = src_col->offset_index_offset;
            dst_chunk->has_offset_index_length = src_col->has_offset_index_length;
            dst_chunk->offset_index_length = src_col->offset_index_length;

            if (src_col->has_geo_stats &&
                (src_col->geo_stats.valid || src_col->geo_stats.num_types > 0)) {
                meta->has_geospatial_statistics = true;
                meta->geospatial_statistics = src_col->geo_stats;
            }

            if (src_col->has_statistics) {
                meta->has_statistics = true;
                parquet_statistics_t* stats = &meta->statistics;
                memset(stats, 0, sizeof(*stats));

                if (src_col->has_null_count) {
                    stats->has_null_count = true;
                    stats->null_count = src_col->null_count;
                }

                if (src_col->has_min_max &&
                    src_col->min_value_size > 0 &&
                    src_col->max_value_size > 0) {

                    /* Truncate variable-length min/max per the Parquet spec
                     * recommendation. Numeric / BOOLEAN / FLBA stats are
                     * already at their natural fixed width so pass through. */
                    const size_t TRUNC = 32;
                    bool variable_len =
                        (src_col->type == CARQUET_PHYSICAL_BYTE_ARRAY);

                    int32_t min_n = (int32_t)src_col->min_value_size;
                    int32_t max_n = (int32_t)src_col->max_value_size;
                    bool emit_max = true;
                    bool min_exact = true;
                    bool max_exact = true;

                    uint8_t* min_buf = carquet_arena_alloc(&writer->arena,
                        (size_t)min_n);
                    if (!min_buf) return CARQUET_ERROR_OUT_OF_MEMORY;
                    memcpy(min_buf, src_col->min_value, (size_t)min_n);

                    if (variable_len && (size_t)min_n > TRUNC) {
                        /* Truncated prefix is lex <= original, valid as min. */
                        min_n = (int32_t)TRUNC;
                        min_exact = false;
                    }

                    uint8_t* max_buf = carquet_arena_alloc(&writer->arena,
                        (size_t)max_n);
                    if (!max_buf) return CARQUET_ERROR_OUT_OF_MEMORY;
                    memcpy(max_buf, src_col->max_value, (size_t)max_n);

                    if (variable_len && (size_t)max_n > TRUNC) {
                        /* Truncate then increment to ensure result >= original.
                         * If all bytes in the prefix are 0xFF the increment
                         * wraps and we cannot emit a valid upper bound — drop
                         * max in that case. */
                        max_exact = false;
                        size_t new_len = TRUNC;
                        max_buf[new_len - 1]++;
                        while (new_len > 0 && max_buf[new_len - 1] == 0) {
                            new_len--;
                            if (new_len > 0) max_buf[new_len - 1]++;
                        }
                        if (new_len == 0) {
                            emit_max = false;
                        } else {
                            max_n = (int32_t)new_len;
                        }
                    }

                    stats->min_value = min_buf;
                    stats->min_value_len = min_n;
                    stats->has_is_min_value_exact = true;
                    stats->is_min_value_exact = min_exact;
                    if (emit_max) {
                        stats->max_value = max_buf;
                        stats->max_value_len = max_n;
                        stats->has_is_max_value_exact = true;
                        stats->is_max_value_exact = max_exact;
                    }

                    /* Mirror to deprecated min/max fields for older readers
                     * that ignore min_value/max_value (only for non-truncated
                     * stats to avoid leaking truncation semantics). */
                    bool deprecated_ok = j < writer->num_columns &&
                        deprecated_stats_are_compatible(&writer->columns[j]);
                    if (deprecated_ok &&
                        (!variable_len || (size_t)src_col->min_value_size <= TRUNC)) {
                        stats->min_deprecated = min_buf;
                        stats->min_deprecated_len = min_n;
                    }
                    if (deprecated_ok && emit_max &&
                        (!variable_len || (size_t)src_col->max_value_size <= TRUNC)) {
                        stats->max_deprecated = max_buf;
                        stats->max_deprecated_len = max_n;
                    }
                }
            }
        }
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

    carquet_writer_t* writer = carquet_mem_calloc(1, sizeof(carquet_writer_t));
    if (!writer) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate writer");
        return NULL;
    }

    /* Initialize arena */
    if (carquet_arena_init_size(&writer->arena, 4096) != CARQUET_OK) {
        carquet_mem_free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate arena");
        return NULL;
    }

    /* Open file */
    writer->file = fopen(path, "wb");
    if (!writer->file) {
        carquet_arena_destroy(&writer->arena);
        carquet_mem_free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_OPEN, "Failed to open file for writing: %s", path);
        return NULL;
    }
    writer->owns_file = true;

    writer->path = carquet_heap_strdup(path);
    if (!writer->path) {
        fclose(writer->file);
        carquet_arena_destroy(&writer->arena);
        carquet_mem_free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate path");
        return NULL;
    }

    /* Copy options */
    if (options) {
        writer->options = *options;
    } else {
        carquet_writer_options_init(&writer->options);
    }

    /* Store full schema elements for metadata serialization */
    {
        carquet_status_t status = store_schema_elements(writer, schema);
        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to store schema elements");
            return NULL;
        }

        status = build_column_metadata_cache(writer, schema);
        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to build writer metadata cache");
            return NULL;
        }
    }

    /* Add leaf columns from schema (schema is nonnull per API contract) */
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
            elem->type_length,
            schema->max_def_levels[i],
            schema->max_rep_levels[i],
            leaf_statistics_sort_order_defined(schema, elem_idx));

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
    carquet_writer_t* writer = carquet_mem_calloc(1, sizeof(carquet_writer_t));
    if (!writer) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate writer");
        return NULL;
    }

    /* Initialize arena */
    if (carquet_arena_init_size(&writer->arena, 4096) != CARQUET_OK) {
        carquet_mem_free(writer);
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

    /* Store full schema elements for metadata serialization */
    {
        carquet_status_t status = store_schema_elements(writer, schema);
        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to store schema elements");
            return NULL;
        }

        status = build_column_metadata_cache(writer, schema);
        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to build writer metadata cache");
            return NULL;
        }
    }

    /* Add leaf columns from schema (schema is nonnull per API contract) */
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
            elem->type_length,
            schema->max_def_levels[i],
            schema->max_rep_levels[i],
            leaf_statistics_sort_order_defined(schema, elem_idx));

        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to add column from schema");
            return NULL;
        }
    }

    return writer;
}

/* 1000^|rank diff| between time units (MILLIS=0, MICROS=1, NANOS=2). */
static int64_t timestamp_unit_factor(carquet_time_unit_t a,
                                     carquet_time_unit_t b) {
    int d = (int)a - (int)b;
    if (d < 0) d = -d;
    int64_t f = 1;
    for (int i = 0; i < d; i++) f *= 1000;
    return f;
}

/* Rescale `n` INT64 timestamps from `src` to `dst` unit in place into out.
 * Returns CARQUET_ERROR_INVALID_ARGUMENT on disallowed truncation/overflow. */
static carquet_status_t coerce_timestamp_values(
    const int64_t* in, int64_t* out, int64_t n,
    carquet_time_unit_t src, carquet_time_unit_t dst,
    bool allow_truncation) {

    int64_t factor = timestamp_unit_factor(src, dst);
    if ((int)dst > (int)src) {
        /* finer target: multiply, guard overflow */
        int64_t lim = INT64_MAX / factor;
        for (int64_t i = 0; i < n; i++) {
            int64_t v = in[i];
            if (v > lim || v < -lim) return CARQUET_ERROR_INVALID_ARGUMENT;
            out[i] = v * factor;
        }
    } else {
        /* coarser target: divide (truncates toward zero) */
        for (int64_t i = 0; i < n; i++) {
            int64_t v = in[i];
            if (!allow_truncation && (v % factor) != 0) {
                return CARQUET_ERROR_INVALID_ARGUMENT;
            }
            out[i] = v / factor;
        }
    }
    return CARQUET_OK;
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

    /* Optional TIMESTAMP coercion: rescale the (packed, non-null) INT64
     * values from the schema-declared unit to the target unit. */
    const void* write_values = values;
    int64_t* coerced = NULL;
    const writer_column_def_t* cdef = &writer->columns[column_index];
    if (writer->options.coerce_timestamps &&
        cdef->physical_type == CARQUET_PHYSICAL_INT64 &&
        cdef->logical_type.id == CARQUET_LOGICAL_TIMESTAMP) {
        carquet_time_unit_t src = cdef->logical_type.params.timestamp.unit;
        carquet_time_unit_t dst = writer->options.coerce_timestamp_unit;
        if (src != dst && num_values > 0) {
            int64_t present = num_values;
            if (def_levels && cdef->max_def_level > 0) {
                present = carquet_dispatch_count_non_nulls(
                    def_levels, num_values, cdef->max_def_level);
            }
            coerced = carquet_mem_malloc((size_t)present * sizeof(int64_t));
            if (!coerced) return CARQUET_ERROR_OUT_OF_MEMORY;
            status = coerce_timestamp_values(
                (const int64_t*)values, coerced, present, src, dst,
                writer->options.allow_timestamp_truncation);
            if (status != CARQUET_OK) {
                carquet_mem_free(coerced);
                return status;
            }
            write_values = coerced;
        }
    }

    /* Write to the row group */
    status = carquet_row_group_writer_write_column(
        writer->current_row_group,
        column_index,
        write_values,
        num_values,
        def_levels,
        rep_levels);

    carquet_mem_free(coerced);

    if (status != CARQUET_OK) {
        return status;
    }

    writer->column_values_written[column_index] += num_values;

    /* Track rows (use column 0 as reference).
     * For repeated columns (max_rep_level > 0), the number of logical rows
     * is the count of rep_level == 0 entries (new top-level records).
     * For non-repeated columns, num_values == num_rows. */
    if (column_index == 0) {
        if (rep_levels && writer->columns[0].max_rep_level > 0) {
            int64_t rows = 0;
            for (int64_t i = 0; i < num_values; i++) {
                if (rep_levels[i] == 0) rows++;
            }
            writer->current_row_group_rows += rows;
        } else {
            writer->current_row_group_rows += num_values;
        }
    }

    writer->current_row_group_estimated_bytes = saturating_add_i64(
        writer->current_row_group_estimated_bytes,
        estimate_column_batch_bytes(
            &writer->columns[column_index],
            values,
            num_values,
            def_levels,
            rep_levels));

    if (writer_supports_aligned_auto_flush(writer) &&
        writer->current_row_group_estimated_bytes > writer->options.row_group_size &&
        current_row_group_is_aligned(writer)) {
        status = flush_row_group(writer);
        if (status != CARQUET_OK) {
            return status;
        }
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
    carquet_buffer_init(&metadata_buffer);

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

    /* For buffer writers, read back the entire file into memory.
     * Use 64-bit seek/tell to handle files >2 GB on all platforms. */
    if (writer->is_buffer_writer && writer->file) {
        fflush(writer->file);
        int64_t file_size = -1;
        if (carquet_fseek64(writer->file, 0, SEEK_END) == 0)
            file_size = carquet_ftell64(writer->file);
        if (file_size > 0) {
            writer->output_buffer = carquet_mem_malloc((size_t)file_size);
            if (!writer->output_buffer) {
                status = CARQUET_ERROR_OUT_OF_MEMORY;
                goto cleanup;
            }
            rewind(writer->file);
            size_t nread = fread(writer->output_buffer, 1, (size_t)file_size, writer->file);
            if (nread != (size_t)file_size) {
                carquet_mem_free(writer->output_buffer);
                writer->output_buffer = NULL;
                status = CARQUET_ERROR_FILE_READ;
                goto cleanup;
            }
            writer->output_buffer_size = (size_t)file_size;
        }
    }

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
            carquet_mem_free(writer->columns[i].name);
        }
        carquet_mem_free(writer->columns);
    }

    /* Free schema elements */
    if (writer->schema_elements) {
        for (int32_t i = 0; i < writer->num_schema_elements; i++) {
            carquet_mem_free(writer->schema_elements[i].name);
        }
        carquet_mem_free(writer->schema_elements);
    }
    if (writer->column_paths) {
        for (int32_t i = 0; i < writer->num_columns; i++) {
            carquet_mem_free(writer->column_paths[i]);
        }
        carquet_mem_free(writer->column_paths);
    }
    carquet_mem_free(writer->column_path_lens);
    carquet_mem_free(writer->column_encodings);
    carquet_mem_free(writer->column_num_encodings);

    carquet_mem_free(writer->column_values_written);
    free_row_groups(writer);
    carquet_mem_free(writer->path);

    /* Free key-value metadata */
    free_kv_metadata(writer);

    /* Free per-column overrides */
    free_column_overrides(writer);

    /* For buffer writers, keep the writer alive so get_buffer can be called.
     * The writer struct (and output_buffer) will be freed by get_buffer or
     * by a subsequent abort call. */
    if (writer->is_buffer_writer) {
        return status;
    }

    carquet_mem_free(writer->output_buffer);
    carquet_arena_destroy(&writer->arena);
    carquet_mem_free(writer);

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
            carquet_mem_free(writer->columns[i].name);
        }
        carquet_mem_free(writer->columns);
    }

    /* Free schema elements */
    if (writer->schema_elements) {
        for (int32_t i = 0; i < writer->num_schema_elements; i++) {
            carquet_mem_free(writer->schema_elements[i].name);
        }
        carquet_mem_free(writer->schema_elements);
    }
    if (writer->column_paths) {
        for (int32_t i = 0; i < writer->num_columns; i++) {
            carquet_mem_free(writer->column_paths[i]);
        }
        carquet_mem_free(writer->column_paths);
    }
    carquet_mem_free(writer->column_path_lens);
    carquet_mem_free(writer->column_encodings);
    carquet_mem_free(writer->column_num_encodings);

    carquet_mem_free(writer->column_values_written);
    free_row_groups(writer);
    carquet_mem_free(writer->path);

    /* Free key-value metadata */
    free_kv_metadata(writer);

    /* Free per-column overrides */
    free_column_overrides(writer);

    /* Free buffer writer output */
    carquet_mem_free(writer->output_buffer);

    carquet_arena_destroy(&writer->arena);
    carquet_mem_free(writer);
}

/* ============================================================================
 * Key-Value Metadata API
 * ============================================================================
 */

carquet_status_t carquet_writer_add_metadata(
    carquet_writer_t* writer,
    const char* key,
    const char* value) {

    /* writer and key are nonnull per API contract */
    if (writer->num_kv_metadata >= writer->kv_metadata_capacity) {
        int32_t new_cap = writer->kv_metadata_capacity == 0 ? 8 : writer->kv_metadata_capacity * 2;
        parquet_key_value_t* new_kv = carquet_mem_realloc(writer->kv_metadata,
            new_cap * sizeof(parquet_key_value_t));
        if (!new_kv) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        writer->kv_metadata = new_kv;
        writer->kv_metadata_capacity = new_cap;
    }

    parquet_key_value_t* entry = &writer->kv_metadata[writer->num_kv_metadata];
    entry->key = carquet_heap_strdup(key);
    if (!entry->key) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    entry->value = value ? carquet_heap_strdup(value) : NULL;
    if (value && !entry->value) {
        carquet_mem_free(entry->key);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    writer->num_kv_metadata++;
    return CARQUET_OK;
}

/* ============================================================================
 * Per-Column Writer Options API
 * ============================================================================
 */

carquet_status_t carquet_writer_set_column_encoding(
    carquet_writer_t* writer,
    int32_t column_index,
    carquet_encoding_t encoding) {

    /* writer is nonnull per API contract */
    if (column_index < 0 || column_index >= writer->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (!writer_encoding_supported(
            encoding, writer->columns[column_index].physical_type)) {
        return CARQUET_ERROR_INVALID_ENCODING;
    }

    carquet_status_t status = ensure_column_overrides(writer);
    if (status != CARQUET_OK) return status;

    writer->column_encoding_overrides[column_index] = encoding;
    writer->column_encoding_override_set[column_index] = true;
    return CARQUET_OK;
}

carquet_status_t carquet_writer_set_column_compression(
    carquet_writer_t* writer,
    int32_t column_index,
    carquet_compression_t codec,
    int32_t level) {

    /* writer is nonnull per API contract */
    if (column_index < 0 || column_index >= writer->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_status_t status = ensure_column_overrides(writer);
    if (status != CARQUET_OK) return status;

    writer->column_compression_overrides[column_index] = codec;
    writer->column_compression_levels[column_index] = level;
    writer->column_compression_override_set[column_index] = true;
    return CARQUET_OK;
}

carquet_status_t carquet_writer_set_column_statistics(
    carquet_writer_t* writer,
    int32_t column_index,
    bool enabled) {

    /* writer is nonnull per API contract */
    if (column_index < 0 || column_index >= writer->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_status_t status = ensure_column_overrides(writer);
    if (status != CARQUET_OK) return status;

    writer->column_statistics_overrides[column_index] = enabled;
    return CARQUET_OK;
}

carquet_status_t carquet_writer_set_column_bloom_filter(
    carquet_writer_t* writer,
    int32_t column_index,
    bool enabled) {

    /* writer is nonnull per API contract */
    if (column_index < 0 || column_index >= writer->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_status_t status = ensure_column_overrides(writer);
    if (status != CARQUET_OK) return status;

    writer->column_bloom_filter_overrides[column_index] = enabled;
    return CARQUET_OK;
}

carquet_status_t carquet_writer_set_column_bloom_filter_options(
    carquet_writer_t* writer,
    int32_t column_index,
    bool enabled,
    int64_t ndv,
    double fpp) {

    /* writer is nonnull per API contract */
    if (column_index < 0 || column_index >= writer->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_status_t status = ensure_column_overrides(writer);
    if (status != CARQUET_OK) return status;

    writer->column_bloom_filter_overrides[column_index] = enabled;
    writer->column_bloom_ndv_overrides[column_index] = ndv;
    writer->column_bloom_fpp_overrides[column_index] = fpp;
    writer->column_bloom_options_set[column_index] = true;
    /* Ensure the finalize path actually emits the per-column bloom filter
     * even when the global option was left off. Additive: only flips the
     * flag on; never disables a globally-enabled configuration. */
    if (enabled) {
        writer->options.write_bloom_filters = true;
    }
    return CARQUET_OK;
}

carquet_status_t carquet_writer_set_sorting_columns(
    carquet_writer_t* writer,
    const carquet_sorting_column_t* columns,
    int32_t count) {

    /* writer is nonnull per API contract */
    if (count < 0 || (count > 0 && !columns)) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }
    if (count > writer->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (count == 0) {
        writer->sorting_columns = NULL;
        writer->num_sorting_columns = 0;
        return CARQUET_OK;
    }

    parquet_sorting_column_t* copy = carquet_arena_calloc(&writer->arena,
        count, sizeof(parquet_sorting_column_t));
    if (!copy) return CARQUET_ERROR_OUT_OF_MEMORY;

    for (int32_t i = 0; i < count; i++) {
        if (columns[i].column_index < 0 ||
            columns[i].column_index >= writer->num_columns) {
            return CARQUET_ERROR_INVALID_ARGUMENT;
        }
        copy[i].column_idx = columns[i].column_index;
        copy[i].descending = columns[i].descending;
        copy[i].nulls_first = columns[i].nulls_first;
    }

    writer->sorting_columns = copy;
    writer->num_sorting_columns = count;
    return CARQUET_OK;
}

/* ============================================================================
 * Writer Buffer API
 * ============================================================================
 */

carquet_writer_t* carquet_writer_create_buffer(
    const carquet_schema_t* schema,
    const carquet_writer_options_t* options,
    carquet_error_t* error) {

    /* Create a temporary FILE* for writing.
     * tmpfile() can fail on Windows when the process lacks write access to
     * the root directory.  Fall back to a named temp file in that case. */
    FILE* tmp = tmpfile();
#ifdef _WIN32
    if (!tmp) {
        /* _tempnam uses %TMP%, %TEMP%, or the current directory */
        char* tpath = _tempnam(NULL, "cqt");
        if (tpath) {
            tmp = fopen(tpath, "w+bTD");   /* T=short-lived, D=delete-on-close */
            carquet_mem_free(tpath);
        }
    }
#endif
    if (!tmp) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_OPEN,
            "Failed to create temporary file for buffer writer");
        return NULL;
    }

    /* Create a writer using the FILE* handle */
    carquet_writer_t* writer = carquet_writer_create_file(tmp, schema, options, error);
    if (!writer) {
        fclose(tmp);
        return NULL;
    }

    /* Mark as buffer writer and take ownership of the tmpfile */
    writer->is_buffer_writer = true;
    writer->owns_file = true;

    return writer;
}

carquet_status_t carquet_writer_get_buffer(
    carquet_writer_t* writer,
    void** buffer,
    size_t* size) {

    /* writer, buffer, size are nonnull per API contract */
    if (!writer->is_buffer_writer) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (!writer->output_buffer || writer->output_buffer_size == 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Transfer ownership of the buffer to the caller */
    *buffer = writer->output_buffer;
    *size = writer->output_buffer_size;
    writer->output_buffer = NULL;
    writer->output_buffer_size = 0;

    /* Free the writer struct (close already freed internal resources) */
    carquet_arena_destroy(&writer->arena);
    carquet_mem_free(writer);

    return CARQUET_OK;
}
