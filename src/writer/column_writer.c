/**
 * @file column_writer.c
 * @brief Column chunk writing implementation
 *
 * Manages writing values to a column chunk, handling page breaks,
 * dictionary encoding, and column-level metadata.
 */

#include "core/allocator.h"
#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/buffer.h"
#include "core/float16.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include <stdlib.h>
#include <string.h>

/* Forward declarations for bloom filter and page index */
typedef struct carquet_bloom_filter carquet_bloom_filter_t;
typedef struct carquet_column_index_builder carquet_column_index_builder_t;
typedef struct carquet_offset_index_builder carquet_offset_index_builder_t;

extern carquet_bloom_filter_t* carquet_bloom_filter_create_with_ndv(int64_t ndv, double fpp);
extern void carquet_bloom_filter_destroy(carquet_bloom_filter_t* filter);
extern void carquet_bloom_filter_insert_i32(carquet_bloom_filter_t* filter, int32_t value);
extern void carquet_bloom_filter_insert_i64(carquet_bloom_filter_t* filter, int64_t value);
extern void carquet_bloom_filter_insert_float(carquet_bloom_filter_t* filter, float value);
extern void carquet_bloom_filter_insert_double(carquet_bloom_filter_t* filter, double value);
extern void carquet_bloom_filter_insert_bytes(carquet_bloom_filter_t* filter,
                                               const uint8_t* data, size_t len);
extern const uint8_t* carquet_bloom_filter_data(const carquet_bloom_filter_t* filter);
extern size_t carquet_bloom_filter_size(const carquet_bloom_filter_t* filter);
extern int64_t carquet_dispatch_count_non_nulls(const int16_t* def_levels, int64_t count,
                                                 int16_t max_def_level);
extern void carquet_dispatch_fill_def_levels(int16_t* def_levels, int64_t count, int16_t value);

extern carquet_column_index_builder_t* carquet_column_index_builder_create(
    carquet_physical_type_t type, const carquet_logical_type_t* logical_type,
    int32_t type_length);
extern void carquet_column_index_builder_destroy(carquet_column_index_builder_t* builder);
extern carquet_status_t carquet_column_index_add_page(
    carquet_column_index_builder_t* builder,
    int64_t null_count, const void* min_value, int32_t min_value_len,
    const void* max_value, int32_t max_value_len, bool is_null_page,
    const int64_t* rep_level_hist, int32_t rep_level_hist_len,
    const int64_t* def_level_hist, int32_t def_level_hist_len);
extern carquet_status_t carquet_column_index_serialize(
    const carquet_column_index_builder_t* builder, carquet_buffer_t* output);

extern carquet_offset_index_builder_t* carquet_offset_index_builder_create(bool track_unencoded);
extern void carquet_offset_index_builder_destroy(carquet_offset_index_builder_t* builder);
extern carquet_status_t carquet_offset_index_add_page(
    carquet_offset_index_builder_t* builder,
    int64_t offset, int32_t compressed_size,
    int64_t first_row_index, int64_t unencoded_byte_array_bytes);
extern carquet_status_t carquet_offset_index_serialize(
    const carquet_offset_index_builder_t* builder, carquet_buffer_t* output);
extern void carquet_offset_index_builder_shift_offsets(
    carquet_offset_index_builder_t* builder, int64_t delta);

/* Forward declaration from page_writer.c */
typedef struct carquet_page_writer carquet_page_writer_t;

extern carquet_page_writer_t* carquet_page_writer_create(
    carquet_physical_type_t type,
    const carquet_logical_type_t* logical_type,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    int32_t compression_level);

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
extern carquet_status_t carquet_page_writer_finalize_to_buffer(
    carquet_page_writer_t* writer,
    carquet_buffer_t* output_buffer,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size);

extern size_t carquet_page_writer_estimated_size(const carquet_page_writer_t* writer);
extern int64_t carquet_page_writer_num_values(const carquet_page_writer_t* writer);
extern int64_t carquet_page_writer_byte_array_bytes(const carquet_page_writer_t* writer);
extern void carquet_page_writer_set_byte_array_bytes(
    carquet_page_writer_t* writer, int64_t bytes);
extern const int64_t* carquet_page_writer_def_level_histogram(
    const carquet_page_writer_t* writer, int32_t* len);
extern const int64_t* carquet_page_writer_rep_level_histogram(
    const carquet_page_writer_t* writer, int32_t* len);
extern void carquet_page_writer_set_crc(carquet_page_writer_t* writer, bool enabled);
extern void carquet_page_writer_set_statistics(carquet_page_writer_t* writer, bool enabled);
extern void carquet_page_writer_set_data_page_v2(carquet_page_writer_t* writer, bool enabled);
extern const parquet_geospatial_statistics_t* carquet_page_writer_get_geo_stats(
    const carquet_page_writer_t* writer);

extern carquet_status_t carquet_page_writer_emit_dictionary_page(
    carquet_page_writer_t* writer, carquet_buffer_t* output_buffer,
    const uint8_t* plain_payload, size_t payload_size, int32_t num_entries,
    size_t* page_size, int32_t* uncompressed_size, int32_t* compressed_size);
extern carquet_status_t carquet_page_writer_add_dictionary_indices(
    carquet_page_writer_t* writer, const uint8_t* idx_payload, size_t idx_size,
    const int16_t* def_levels, const int16_t* rep_levels,
    int64_t num_values_total, int64_t num_nulls);
extern void carquet_page_writer_set_encoding(carquet_page_writer_t* writer,
                                             carquet_encoding_t encoding);
extern carquet_status_t carquet_page_writer_set_min_max(
    carquet_page_writer_t* writer,
    const uint8_t* min_value, size_t min_size,
    const uint8_t* max_value, size_t max_size);

/* Dictionary encoders (src/encoding/dictionary.c). Each produces the PLAIN
 * dictionary payload in dict_output and [bit-width][RLE indices] in
 * indices_output, over the full non-null value array in one call. */
extern carquet_status_t carquet_dictionary_encode_int32(
    const int32_t* values, int64_t count,
    carquet_buffer_t* dict_output, carquet_buffer_t* indices_output);
extern carquet_status_t carquet_dictionary_encode_int64(
    const int64_t* values, int64_t count,
    carquet_buffer_t* dict_output, carquet_buffer_t* indices_output);
extern carquet_status_t carquet_dictionary_encode_float(
    const float* values, int64_t count,
    carquet_buffer_t* dict_output, carquet_buffer_t* indices_output);
extern carquet_status_t carquet_dictionary_encode_double(
    const double* values, int64_t count,
    carquet_buffer_t* dict_output, carquet_buffer_t* indices_output);
extern carquet_status_t carquet_dictionary_encode_byte_array(
    const carquet_byte_array_t* values, int64_t count,
    carquet_buffer_t* dict_output, carquet_buffer_t* indices_output);
extern carquet_status_t carquet_dictionary_encode_capped(
    carquet_physical_type_t type, int32_t type_length,
    const void* fixed_values, const carquet_byte_array_t* ba_values,
    int64_t count, size_t max_dict_bytes,
    carquet_buffer_t* dict_output, carquet_buffer_t* indices_output,
    bool* abandoned);

/* ============================================================================
 * Column Writer Structure
 * ============================================================================
 */

typedef struct carquet_column_writer_internal {
    carquet_page_writer_t* page_writer;
    carquet_buffer_t column_buffer;  /* All pages for this column chunk */

    /* Column configuration */
    carquet_physical_type_t type;
    carquet_logical_type_t logical_type;
    carquet_encoding_t encoding;
    carquet_compression_t compression;
    int32_t type_length;
    int16_t max_def_level;
    int16_t max_rep_level;

    /* Page size limits */
    size_t target_page_size;
    size_t max_page_size;
    int64_t max_rows_per_page;  /* 0 = unlimited */
    int64_t write_batch_size;   /* 0 = automatic chunk heuristic */

    /* Statistics */
    int64_t total_values;
    int64_t total_nulls;
    int64_t total_uncompressed_size;
    int64_t total_compressed_size;
    int32_t num_pages;

    /* Min/max tracking. Min and max may have different lengths (BYTE_ARRAY). */
    bool has_min_max;
    uint8_t* min_value;
    size_t min_value_size;
    size_t min_value_capacity;
    uint8_t* max_value;
    size_t max_value_size;
    size_t max_value_capacity;

    /* Column path for metadata */
    char** path_in_schema;
    int path_depth;

    /* Bloom filter (optional) */
    carquet_bloom_filter_t* bloom_filter;
    int64_t bloom_ndv;
    double bloom_fpp;

    /* Page index builders (optional) */
    carquet_column_index_builder_t* column_index;
    carquet_offset_index_builder_t* offset_index;
    bool page_index_enabled;
    int64_t page_row_offset;      /* Row offset for current page (for offset index) */
    int64_t column_file_offset;   /* File offset where this column starts */

    /* Dictionary (chunk-buffered) encoding state. When use_dictionary is set,
     * write_batch accumulates all non-null values plus all def/rep levels for
     * the whole column chunk; finalize then builds the dictionary, decides
     * whether to keep it (fallback heuristic), and emits the dictionary page
     * followed by a single RLE_DICTIONARY data page (or a PLAIN data page on
     * fallback). dictionary_page_size_bytes is reported back so the file
     * writer can compute dictionary_page_offset / data_page_offset. */
    bool use_dictionary;
    size_t dictionary_page_size_limit;  /* options.dictionary_page_size */
    carquet_buffer_t dict_values;       /* Accumulated raw non-null values */
    int64_t dict_value_count;           /* Count of accumulated non-null values */
    carquet_byte_array_t* dict_ba;      /* BYTE_ARRAY: array of {data,len} */
    size_t dict_ba_capacity;
    carquet_buffer_t dict_ba_storage;   /* BYTE_ARRAY: backing byte storage */
    int16_t* dict_def_levels;           /* Accumulated definition levels */
    int64_t dict_def_count;
    size_t dict_def_capacity;
    int16_t* dict_rep_levels;           /* Accumulated repetition levels */
    int64_t dict_rep_count;
    size_t dict_rep_capacity;
    int64_t dict_total_rows;            /* Total logical rows incl. nulls */
    int64_t dict_total_nulls;
    bool has_dictionary_page;           /* Set when a dict page was emitted */
    int64_t dictionary_page_size_bytes; /* Size of the emitted dict page */
    /* Exact distinct non-null value count for this chunk. Set only when the
     * dictionary was built and kept (num_unique); on PLAIN / dict-fallback we
     * cannot count distincts cheaply, so it stays unset and Statistics omits
     * distinct_count. */
    bool has_distinct_count;
    int64_t distinct_count;
    /* Chunk-level SizeStatistics accumulators (Parquet 2.9), summed over pages
     * in flush_current_page. Histograms are sized max_def/rep_level + 1;
     * chunk_unencoded_ba_bytes is meaningful only for BYTE_ARRAY columns. */
    int64_t chunk_unencoded_ba_bytes;
    int64_t* chunk_def_hist;
    int64_t* chunk_rep_hist;

    /* Deferred-encode state. For non-dictionary, compressed, fixed-stride
     * columns in a parallel-capable row group, write_batch stashes the raw
     * input verbatim instead of encoding+compressing eagerly on the caller's
     * thread. The whole stashed batch is replayed through the normal eager
     * encode path inside the OpenMP per-column finalize, so encode AND
     * compression run concurrently across columns. Output is byte-identical
     * to the eager path (same bytes, same code, different thread). */
    bool defer_encode;
    carquet_buffer_t deferred_values;   /* full-width raw value bytes */
    int64_t deferred_count;             /* logical values stashed */
    int16_t* deferred_def_levels;
    size_t deferred_def_capacity;
    int64_t deferred_def_count;
    int16_t* deferred_rep_levels;
    size_t deferred_rep_capacity;
    int64_t deferred_rep_count;
} carquet_column_writer_internal_t;

/* ============================================================================
 * Column Writer Lifecycle
 * ============================================================================
 */

void carquet_column_writer_destroy(carquet_column_writer_internal_t* writer);

carquet_column_writer_internal_t* carquet_column_writer_create(
    carquet_physical_type_t type,
    const carquet_logical_type_t* logical_type,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    size_t target_page_size,
    int32_t compression_level) {

    carquet_column_writer_internal_t* writer = carquet_mem_calloc(1, sizeof(*writer));
    if (!writer) return NULL;

    writer->page_writer = carquet_page_writer_create(
        type, logical_type, encoding, compression, max_def_level, max_rep_level,
        type_length, compression_level);

    if (!writer->page_writer) {
        carquet_mem_free(writer);
        return NULL;
    }

    carquet_buffer_init(&writer->column_buffer);

    writer->type = type;
    if (logical_type) {
        writer->logical_type = *logical_type;
    }
    writer->encoding = encoding;
    writer->compression = compression;
    writer->type_length = type_length;
    writer->max_def_level = max_def_level;
    writer->max_rep_level = max_rep_level;
    writer->target_page_size = target_page_size > 0 ? target_page_size : (1024 * 1024);
    writer->max_page_size = writer->target_page_size * 2;

    /* Dictionary encoding is used when the requested encoding is a dictionary
     * encoding AND the physical type is eligible. INT96 and BOOLEAN have no
     * dictionary encoder, so they keep PLAIN. FIXED_LEN_BYTE_ARRAY is encoded
     * as a fixed-width dictionary (stride == type_length). */
    bool dict_eligible_type =
        type == CARQUET_PHYSICAL_INT32 ||
        type == CARQUET_PHYSICAL_INT64 ||
        type == CARQUET_PHYSICAL_FLOAT ||
        type == CARQUET_PHYSICAL_DOUBLE ||
        type == CARQUET_PHYSICAL_BYTE_ARRAY ||
        (type == CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY && type_length > 0);
    writer->use_dictionary = dict_eligible_type &&
        (encoding == CARQUET_ENCODING_RLE_DICTIONARY ||
         encoding == CARQUET_ENCODING_PLAIN_DICTIONARY);
    writer->dictionary_page_size_limit = 1024 * 1024;
    carquet_buffer_init(&writer->dict_values);
    carquet_buffer_init(&writer->dict_ba_storage);
    carquet_buffer_init(&writer->deferred_values);

    writer->chunk_def_hist =
        carquet_mem_calloc((size_t)max_def_level + 1, sizeof(int64_t));
    writer->chunk_rep_hist =
        carquet_mem_calloc((size_t)max_rep_level + 1, sizeof(int64_t));
    if (!writer->chunk_def_hist || !writer->chunk_rep_hist) {
        carquet_column_writer_destroy(writer);
        return NULL;
    }

    return writer;
}

void carquet_column_writer_destroy(carquet_column_writer_internal_t* writer) {
    if (writer) {
        if (writer->page_writer) {
            carquet_page_writer_destroy(writer->page_writer);
        }
        carquet_mem_free(writer->min_value);
        carquet_mem_free(writer->max_value);
        carquet_buffer_destroy(&writer->column_buffer);
        carquet_buffer_destroy(&writer->dict_values);
        carquet_buffer_destroy(&writer->dict_ba_storage);
        carquet_buffer_destroy(&writer->deferred_values);
        carquet_mem_free(writer->deferred_def_levels);
        carquet_mem_free(writer->deferred_rep_levels);
        carquet_mem_free(writer->dict_ba);
        carquet_mem_free(writer->dict_def_levels);
        carquet_mem_free(writer->dict_rep_levels);
        carquet_mem_free(writer->chunk_def_hist);
        carquet_mem_free(writer->chunk_rep_hist);

        /* Free path strings */
        if (writer->path_in_schema) {
            for (int i = 0; i < writer->path_depth; i++) {
                carquet_mem_free(writer->path_in_schema[i]);
            }
            carquet_mem_free(writer->path_in_schema);
        }

        if (writer->bloom_filter) {
            carquet_bloom_filter_destroy(writer->bloom_filter);
        }
        if (writer->column_index) {
            carquet_column_index_builder_destroy(writer->column_index);
        }
        if (writer->offset_index) {
            carquet_offset_index_builder_destroy(writer->offset_index);
        }

        carquet_mem_free(writer);
    }
}

void carquet_column_writer_set_crc(carquet_column_writer_internal_t* writer, bool enabled) {
    if (writer) {
        carquet_page_writer_set_crc(writer->page_writer, enabled);
    }
}

void carquet_column_writer_set_data_page_v2(
    carquet_column_writer_internal_t* writer, bool enabled) {
    if (writer) {
        carquet_page_writer_set_data_page_v2(writer->page_writer, enabled);
    }
}

static size_t physical_type_stride(carquet_physical_type_t type,
                                   int32_t type_length);

/* True when this column would benefit from deferred encode: a non-dictionary
 * (dictionary already defers to finalize), compressed, fixed-stride column.
 * Uncompressed columns gain nothing (no compression to parallelize) and would
 * only pay an extra stash copy; variable-length BYTE_ARRAY is not stashed. */
bool carquet_column_writer_defer_eligible(
    const carquet_column_writer_internal_t* writer) {
    if (!writer) return false;
    if (writer->use_dictionary) return false;
    if (writer->compression == CARQUET_COMPRESSION_UNCOMPRESSED) return false;
    return physical_type_stride(writer->type, writer->type_length) > 0;
}

void carquet_column_writer_set_defer_encode(
    carquet_column_writer_internal_t* writer, bool enabled) {
    if (writer) {
        writer->defer_encode =
            enabled && carquet_column_writer_defer_eligible(writer);
    }
}

const parquet_geospatial_statistics_t* carquet_column_writer_get_geo_stats(
    const carquet_column_writer_internal_t* writer) {
    if (!writer) return NULL;
    return carquet_page_writer_get_geo_stats(writer->page_writer);
}

void carquet_column_writer_reset(carquet_column_writer_internal_t* writer) {
    if (!writer) return;

    carquet_page_writer_reset(writer->page_writer);
    carquet_buffer_clear(&writer->column_buffer);

    writer->total_values = 0;
    writer->total_nulls = 0;
    writer->total_uncompressed_size = 0;
    writer->total_compressed_size = 0;
    writer->num_pages = 0;
    writer->has_min_max = false;
    writer->min_value_size = 0;
    writer->max_value_size = 0;
    writer->page_row_offset = 0;
    writer->column_file_offset = 0;

    carquet_buffer_clear(&writer->dict_values);
    carquet_buffer_clear(&writer->dict_ba_storage);
    carquet_buffer_clear(&writer->deferred_values);
    writer->deferred_count = 0;
    writer->deferred_def_count = 0;
    writer->deferred_rep_count = 0;
    writer->dict_value_count = 0;
    writer->dict_def_count = 0;
    writer->dict_rep_count = 0;
    writer->dict_total_rows = 0;
    writer->dict_total_nulls = 0;
    writer->has_dictionary_page = false;
    writer->dictionary_page_size_bytes = 0;
    writer->has_distinct_count = false;
    writer->distinct_count = 0;
    writer->chunk_unencoded_ba_bytes = 0;
    if (writer->chunk_def_hist) {
        memset(writer->chunk_def_hist, 0,
               ((size_t)writer->max_def_level + 1) * sizeof(int64_t));
    }
    if (writer->chunk_rep_hist) {
        memset(writer->chunk_rep_hist, 0,
               ((size_t)writer->max_rep_level + 1) * sizeof(int64_t));
    }

    if (writer->bloom_filter) {
        carquet_bloom_filter_destroy(writer->bloom_filter);
        writer->bloom_filter = NULL;
    }
    if (writer->bloom_ndv > 0) {
        double fpp = (writer->bloom_fpp > 0.0 && writer->bloom_fpp < 1.0)
                         ? writer->bloom_fpp : 0.01;
        writer->bloom_filter = carquet_bloom_filter_create_with_ndv(
            writer->bloom_ndv, fpp);
    }

    if (writer->column_index) {
        carquet_column_index_builder_destroy(writer->column_index);
        writer->column_index = NULL;
    }
    if (writer->offset_index) {
        carquet_offset_index_builder_destroy(writer->offset_index);
        writer->offset_index = NULL;
    }
    if (writer->page_index_enabled) {
        writer->column_index = carquet_column_index_builder_create(
            writer->type, &writer->logical_type, writer->type_length);
        writer->offset_index = carquet_offset_index_builder_create(
            writer->type == CARQUET_PHYSICAL_BYTE_ARRAY);
    }
}

/* ============================================================================
 * Page Flushing
 * ============================================================================
 */

/* Forward declarations for page writer statistics */
extern bool carquet_page_writer_get_statistics(
    const carquet_page_writer_t* writer,
    const uint8_t** min_value, size_t* min_size,
    const uint8_t** max_value, size_t* max_size,
    int64_t* null_count);
extern int64_t carquet_page_writer_null_count(const carquet_page_writer_t* writer);

/* Numeric/IEEE-754-aware comparison for fixed-width stat values. For
 * variable-length and FLBA types Parquet uses unsigned lexicographic order. */
static bool logical_integer_is_unsigned(const carquet_logical_type_t* lt) {
    return lt &&
           lt->id == CARQUET_LOGICAL_INTEGER &&
           !lt->params.integer.is_signed;
}

static int compare_stat_values(carquet_physical_type_t type,
                               const carquet_logical_type_t* logical_type,
                               const uint8_t* a, size_t alen,
                               const uint8_t* b, size_t blen) {
    if (alen == blen) {
        switch (type) {
            case CARQUET_PHYSICAL_INT32: {
                if (logical_integer_is_unsigned(logical_type)) {
                    uint32_t av, bv;
                    memcpy(&av, a, sizeof(av));
                    memcpy(&bv, b, sizeof(bv));
                    return (av < bv) ? -1 : (av > bv ? 1 : 0);
                } else {
                    int32_t av, bv;
                    memcpy(&av, a, sizeof(av));
                    memcpy(&bv, b, sizeof(bv));
                    return (av < bv) ? -1 : (av > bv ? 1 : 0);
                }
            }
            case CARQUET_PHYSICAL_INT64: {
                if (logical_integer_is_unsigned(logical_type)) {
                    uint64_t av, bv;
                    memcpy(&av, a, sizeof(av));
                    memcpy(&bv, b, sizeof(bv));
                    return (av < bv) ? -1 : (av > bv ? 1 : 0);
                } else {
                    int64_t av, bv;
                    memcpy(&av, a, sizeof(av));
                    memcpy(&bv, b, sizeof(bv));
                    return (av < bv) ? -1 : (av > bv ? 1 : 0);
                }
            }
            case CARQUET_PHYSICAL_FLOAT: {
                float av, bv;
                memcpy(&av, a, sizeof(av));
                memcpy(&bv, b, sizeof(bv));
                if (av < bv) return -1;
                if (av > bv) return 1;
                return 0;
            }
            case CARQUET_PHYSICAL_DOUBLE: {
                double av, bv;
                memcpy(&av, a, sizeof(av));
                memcpy(&bv, b, sizeof(bv));
                if (av < bv) return -1;
                if (av > bv) return 1;
                return 0;
            }
            case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: {
                /* FLOAT16 is ordered by represented value, not lexicographic. */
                if (logical_type &&
                    logical_type->id == CARQUET_LOGICAL_FLOAT16 && alen == 2) {
                    float av = carquet_half_to_float(
                        (uint16_t)(a[0] | (a[1] << 8)));
                    float bv = carquet_half_to_float(
                        (uint16_t)(b[0] | (b[1] << 8)));
                    if (av < bv) return -1;
                    if (av > bv) return 1;
                    return 0;
                }
                break;
            }
            default:
                break;
        }
    }
    /* BYTE_ARRAY / FLBA / mismatched sizes: lexicographic unsigned compare. */
    size_t n = alen < blen ? alen : blen;
    int c = memcmp(a, b, n);
    if (c != 0) return c;
    if (alen < blen) return -1;
    if (alen > blen) return 1;
    return 0;
}

static carquet_status_t column_stats_grow(uint8_t** buf, size_t* cap, size_t need) {
    if (need <= *cap) return CARQUET_OK;
    size_t new_cap = *cap == 0 ? 64 : *cap;
    while (new_cap < need) new_cap *= 2;
    uint8_t* p = carquet_mem_realloc(*buf, new_cap);
    if (!p) return CARQUET_ERROR_OUT_OF_MEMORY;
    *buf = p;
    *cap = new_cap;
    return CARQUET_OK;
}

static void merge_page_statistics(carquet_column_writer_internal_t* writer,
                                  const uint8_t* page_min, size_t min_size,
                                  const uint8_t* page_max, size_t max_size) {
    if (!page_min || !page_max || min_size == 0 || max_size == 0) {
        return;
    }

    if (!writer->has_min_max) {
        if (column_stats_grow(&writer->min_value, &writer->min_value_capacity,
                              min_size) != CARQUET_OK) return;
        if (column_stats_grow(&writer->max_value, &writer->max_value_capacity,
                              max_size) != CARQUET_OK) return;
        memcpy(writer->min_value, page_min, min_size);
        memcpy(writer->max_value, page_max, max_size);
        writer->min_value_size = min_size;
        writer->max_value_size = max_size;
        writer->has_min_max = true;
        return;
    }

    if (compare_stat_values(writer->type, &writer->logical_type,
                            page_min, min_size,
                            writer->min_value, writer->min_value_size) < 0) {
        if (column_stats_grow(&writer->min_value, &writer->min_value_capacity,
                              min_size) != CARQUET_OK) return;
        memcpy(writer->min_value, page_min, min_size);
        writer->min_value_size = min_size;
    }
    if (compare_stat_values(writer->type, &writer->logical_type,
                            page_max, max_size,
                            writer->max_value, writer->max_value_size) > 0) {
        if (column_stats_grow(&writer->max_value, &writer->max_value_capacity,
                              max_size) != CARQUET_OK) return;
        memcpy(writer->max_value, page_max, max_size);
        writer->max_value_size = max_size;
    }
}

static carquet_status_t flush_current_page(carquet_column_writer_internal_t* writer) {
    if (carquet_page_writer_num_values(writer->page_writer) == 0) {
        return CARQUET_OK;
    }

    size_t page_size;
    int32_t uncompressed_size;
    int32_t compressed_size;

    /* Capture per-page statistics before finalize (used for column-level
     * aggregation and, when enabled, the column/page index). */
    const uint8_t* page_min = NULL;
    const uint8_t* page_max = NULL;
    size_t min_size = 0;
    size_t max_size = 0;
    int64_t page_null_count = 0;
    bool has_stats = carquet_page_writer_get_statistics(
        writer->page_writer, &page_min, &min_size, &page_max, &max_size,
        &page_null_count);

    if (!has_stats) {
        page_null_count = carquet_page_writer_null_count(writer->page_writer);
    }

    /* Accumulate column-level statistics across pages */
    writer->total_nulls += page_null_count;
    if (has_stats) {
        merge_page_statistics(writer, page_min, min_size, page_max, max_size);
    }

    size_t page_start = writer->column_buffer.size;
    carquet_status_t status = carquet_page_writer_finalize_to_buffer(
        writer->page_writer, &writer->column_buffer, &page_size,
        &uncompressed_size, &compressed_size);

    if (status != CARQUET_OK) {
        return status;
    }

    /* Per-page level histograms + unencoded BYTE_ARRAY bytes, still live on the
     * page writer (reset happens below). Fold them into the chunk-level
     * SizeStatistics accumulators regardless of whether a page index is built. */
    int32_t rep_hist_len = 0, def_hist_len = 0;
    const int64_t* rep_hist = carquet_page_writer_rep_level_histogram(
        writer->page_writer, &rep_hist_len);
    const int64_t* def_hist = carquet_page_writer_def_level_histogram(
        writer->page_writer, &def_hist_len);
    if (rep_hist && rep_hist_len == writer->max_rep_level + 1) {
        for (int32_t i = 0; i < rep_hist_len; i++) writer->chunk_rep_hist[i] += rep_hist[i];
    }
    if (def_hist && def_hist_len == writer->max_def_level + 1) {
        for (int32_t i = 0; i < def_hist_len; i++) writer->chunk_def_hist[i] += def_hist[i];
    }
    if (writer->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        writer->chunk_unencoded_ba_bytes +=
            carquet_page_writer_byte_array_bytes(writer->page_writer);
    }

    /* Record page index entries before appending */
    if (writer->column_index) {
        bool is_null_page = !has_stats;
        carquet_column_index_add_page(
            writer->column_index,
            page_null_count,
            has_stats ? page_min : NULL, has_stats ? (int32_t)min_size : 0,
            has_stats ? page_max : NULL, has_stats ? (int32_t)max_size : 0,
            is_null_page,
            rep_hist, rep_hist_len, def_hist, def_hist_len);
    }

    if (writer->offset_index) {
        /* Record offsets relative to the column start; the column's absolute
         * file offset is not yet known when most pages are flushed (eager
         * flushes happen during write_batch, before the row-group writer
         * positions this column). carquet_column_writer_finalize() shifts
         * the accumulated offsets by column_file_offset once that value
         * has been set, producing absolute offsets as the Parquet spec
         * requires for OffsetIndex.PageLocation.offset. */
        int64_t page_offset_relative = (int64_t)page_start;
        carquet_offset_index_add_page(
            writer->offset_index,
            page_offset_relative,
            (int32_t)page_size,
            writer->page_row_offset,
            carquet_page_writer_byte_array_bytes(writer->page_writer));
        writer->page_row_offset += carquet_page_writer_num_values(writer->page_writer);
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

/* Byte size of a value in memory for fixed-size physical types.
 * Returns 0 for variable-length types (BYTE_ARRAY). */
static size_t physical_type_stride(carquet_physical_type_t type, int32_t type_length) {
    switch (type) {
        case CARQUET_PHYSICAL_BOOLEAN:  return 1;
        case CARQUET_PHYSICAL_INT32:    return 4;
        case CARQUET_PHYSICAL_INT64:    return 8;
        case CARQUET_PHYSICAL_FLOAT:    return 4;
        case CARQUET_PHYSICAL_DOUBLE:   return 8;
        case CARQUET_PHYSICAL_INT96:    return 12;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: return (size_t)type_length;
        default: return 0;
    }
}

static void bloom_filter_insert_chunk(
    carquet_column_writer_internal_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels) {

    if (!writer->bloom_filter) return;

    int64_t num_non_null = num_values;
    if (def_levels && writer->max_def_level > 0) {
        num_non_null = carquet_dispatch_count_non_nulls(
            def_levels, num_values, writer->max_def_level);
    }

    switch (writer->type) {
        case CARQUET_PHYSICAL_INT32: {
            const int32_t* v = (const int32_t*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_i32(writer->bloom_filter, v[i]);
            break;
        }
        case CARQUET_PHYSICAL_INT64: {
            const int64_t* v = (const int64_t*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_i64(writer->bloom_filter, v[i]);
            break;
        }
        case CARQUET_PHYSICAL_FLOAT: {
            const float* v = (const float*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_float(writer->bloom_filter, v[i]);
            break;
        }
        case CARQUET_PHYSICAL_DOUBLE: {
            const double* v = (const double*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_double(writer->bloom_filter, v[i]);
            break;
        }
        case CARQUET_PHYSICAL_BYTE_ARRAY: {
            const carquet_byte_array_t* v = (const carquet_byte_array_t*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_bytes(writer->bloom_filter, v[i].data, v[i].length);
            break;
        }
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: {
            const uint8_t* v = (const uint8_t*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_bytes(writer->bloom_filter,
                    v + i * writer->type_length, writer->type_length);
            break;
        }
        default:
            break;
    }
}

/* ============================================================================
 * Dictionary Accumulation (chunk-buffered path)
 * ============================================================================
 */

static carquet_status_t dict_levels_reserve(int16_t** buf, size_t* cap,
                                             int64_t need) {
    if ((size_t)need <= *cap) return CARQUET_OK;
    size_t new_cap = *cap == 0 ? 4096 : *cap;
    while (new_cap < (size_t)need) new_cap *= 2;
    int16_t* p = carquet_mem_realloc(*buf, new_cap * sizeof(int16_t));
    if (!p) return CARQUET_ERROR_OUT_OF_MEMORY;
    *buf = p;
    *cap = new_cap;
    return CARQUET_OK;
}

/* Accumulate one batch into the chunk-wide dictionary buffers. Non-null
 * values are appended packed; all def/rep levels for every logical row are
 * preserved so the eventual data page reproduces them exactly. */
static carquet_status_t dict_accumulate(
    carquet_column_writer_internal_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    int64_t num_non_null = num_values;
    if (def_levels && writer->max_def_level > 0) {
        num_non_null = carquet_dispatch_count_non_nulls(
            def_levels, num_values, writer->max_def_level);
    }
    writer->dict_total_rows += num_values;
    writer->dict_total_nulls += (num_values - num_non_null);

    /* Append non-null values. */
    if (writer->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        const carquet_byte_array_t* arr = (const carquet_byte_array_t*)values;
        if ((size_t)(writer->dict_value_count + num_non_null) >
            writer->dict_ba_capacity) {
            size_t nc = writer->dict_ba_capacity == 0 ? 1024
                                                      : writer->dict_ba_capacity;
            while (nc < (size_t)(writer->dict_value_count + num_non_null))
                nc *= 2;
            carquet_byte_array_t* p = carquet_mem_realloc(writer->dict_ba,
                                              nc * sizeof(*p));
            if (!p) return CARQUET_ERROR_OUT_OF_MEMORY;
            writer->dict_ba = p;
            writer->dict_ba_capacity = nc;
        }
        for (int64_t i = 0; i < num_non_null; i++) {
            /* Store offsets relative to dict_ba_storage; resolve to pointers
             * after all batches accumulated (storage may realloc). */
            carquet_byte_array_t* slot =
                &writer->dict_ba[writer->dict_value_count + i];
            slot->length = arr[i].length;
            slot->data = (uint8_t*)(uintptr_t)writer->dict_ba_storage.size;
            carquet_status_t s = carquet_buffer_append(
                &writer->dict_ba_storage, arr[i].data,
                (size_t)arr[i].length);
            if (s != CARQUET_OK) return s;
        }
    } else {
        size_t stride = physical_type_stride(writer->type,
                                             writer->type_length);
        carquet_status_t s = carquet_buffer_append(
            &writer->dict_values, (const uint8_t*)values,
            (size_t)num_non_null * stride);
        if (s != CARQUET_OK) return s;
    }
    writer->dict_value_count += num_non_null;

    /* Append def/rep levels. */
    if (writer->max_def_level > 0) {
        carquet_status_t s = dict_levels_reserve(
            &writer->dict_def_levels, &writer->dict_def_capacity,
            writer->dict_def_count + num_values);
        if (s != CARQUET_OK) return s;
        if (def_levels) {
            memcpy(writer->dict_def_levels + writer->dict_def_count,
                   def_levels, (size_t)num_values * sizeof(int16_t));
        } else {
            carquet_dispatch_fill_def_levels(
                writer->dict_def_levels + writer->dict_def_count,
                num_values,
                writer->max_def_level);
        }
        writer->dict_def_count += num_values;
    }
    if (writer->max_rep_level > 0 && rep_levels) {
        carquet_status_t s = dict_levels_reserve(
            &writer->dict_rep_levels, &writer->dict_rep_capacity,
            writer->dict_rep_count + num_values);
        if (s != CARQUET_OK) return s;
        memcpy(writer->dict_rep_levels + writer->dict_rep_count,
               rep_levels, (size_t)num_values * sizeof(int16_t));
        writer->dict_rep_count += num_values;
    }

    bloom_filter_insert_chunk(writer, values, num_values, def_levels);
    return CARQUET_OK;
}

/* Encode + (eagerly) compress one batch through the page pipeline. Does NOT
 * track total_values; the caller owns that so the deferred replay path does
 * not double-count. This is the exact, unchanged eager encode path; the
 * deferred path replays a whole row group's stashed input through it from
 * inside the OpenMP per-column finalize, so the output is byte-identical. */
static carquet_status_t encode_batch_eager(
    carquet_column_writer_internal_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    /* For fixed-size types, split large batches into page-sized chunks.
     * This keeps the working set in cache and avoids huge buffer
     * reallocations that would otherwise occur when accumulating
     * hundreds of MB into a single page. */
    size_t stride = physical_type_stride(writer->type, writer->type_length);

    int64_t max_chunk = num_values;
    if (stride > 0) {
        max_chunk = (int64_t)(writer->target_page_size / stride);
        if (max_chunk < 1024) max_chunk = 1024;

        /* Pre-allocate column buffer to avoid repeated realloc+copy as
         * pages accumulate. Each page adds ~target_page_size + header. */
        if (num_values > max_chunk) {
            size_t expected = (size_t)num_values * stride;
            /* Add ~2% overhead for page headers */
            expected += expected / 50;
            carquet_buffer_reserve(&writer->column_buffer, expected);
        }
    }

    /* Honor an explicit write_batch_size cap for all physical types. */
    if (writer->write_batch_size > 0 && writer->write_batch_size < max_chunk) {
        max_chunk = writer->write_batch_size;
    }

    const uint8_t* val_bytes = (const uint8_t*)values;
    int64_t offset = 0;
    /* The caller's `values` array is dense (sparse encoding): for OPTIONAL
     * columns it contains only the non-null entries, packed contiguously,
     * while `def_levels` has one entry per logical row. When we chunk the
     * batch, `chunk_values` must point at the dense offset corresponding
     * to the current logical chunk, not the logical row offset.
     * `values_offset` tracks the cumulative count of non-null entries
     * already consumed; for REQUIRED columns it stays equal to `offset`. */
    int64_t values_offset = 0;

    while (offset < num_values) {
        int64_t chunk = num_values - offset;
        if (chunk > max_chunk) chunk = max_chunk;

        const void* chunk_values = (stride > 0)
            ? (const void*)(val_bytes + values_offset * stride)
            : (const void*)((const carquet_byte_array_t*)values + values_offset);

        carquet_status_t status = carquet_page_writer_add_values(
            writer->page_writer, chunk_values, chunk,
            def_levels ? def_levels + offset : NULL,
            rep_levels ? rep_levels + offset : NULL);

        if (status != CARQUET_OK) return status;

        bloom_filter_insert_chunk(writer, chunk_values, chunk,
                                  def_levels ? def_levels + offset : NULL);

        /* Advance the dense values cursor by the non-null count in this
         * chunk. REQUIRED columns (max_def_level == 0 or def_levels NULL)
         * have all entries non-null. */
        if (def_levels && writer->max_def_level > 0) {
            int64_t non_null = 0;
            int16_t max_def = writer->max_def_level;
            for (int64_t k = 0; k < chunk; k++) {
                if (def_levels[offset + k] == max_def) non_null++;
            }
            values_offset += non_null;
        } else {
            values_offset += chunk;
        }

        offset += chunk;

        /* Flush page when it reaches target size, or (when configured) when
         * it reaches the row-count cap. The row-count check is guarded by a
         * cheap > 0 test first so it costs nothing when the knob is unset. */
        if (carquet_page_writer_estimated_size(writer->page_writer) >= writer->target_page_size ||
            (writer->max_rows_per_page > 0 &&
             carquet_page_writer_num_values(writer->page_writer) >= writer->max_rows_per_page)) {
            status = flush_current_page(writer);
            if (status != CARQUET_OK) return status;
        }
    }

    return CARQUET_OK;
}

/* Stash a full-width batch verbatim for deferred encode. Levels are copied so
 * the caller's buffers need not outlive the call (matching the eager API
 * contract). Only used for fixed-stride types (stride > 0). */
static carquet_status_t stash_deferred_batch(
    carquet_column_writer_internal_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    size_t stride = physical_type_stride(writer->type, writer->type_length);
    carquet_status_t s = carquet_buffer_append(
        &writer->deferred_values, (const uint8_t*)values,
        (size_t)num_values * stride);
    if (s != CARQUET_OK) return s;

    if (writer->max_def_level > 0) {
        s = dict_levels_reserve(&writer->deferred_def_levels,
                                &writer->deferred_def_capacity,
                                writer->deferred_def_count + num_values);
        if (s != CARQUET_OK) return s;
        if (def_levels) {
            memcpy(writer->deferred_def_levels + writer->deferred_def_count,
                   def_levels, (size_t)num_values * sizeof(int16_t));
        } else {
            carquet_dispatch_fill_def_levels(
                writer->deferred_def_levels + writer->deferred_def_count,
                num_values, writer->max_def_level);
        }
        writer->deferred_def_count += num_values;
    }
    if (writer->max_rep_level > 0 && rep_levels) {
        s = dict_levels_reserve(&writer->deferred_rep_levels,
                                &writer->deferred_rep_capacity,
                                writer->deferred_rep_count + num_values);
        if (s != CARQUET_OK) return s;
        memcpy(writer->deferred_rep_levels + writer->deferred_rep_count,
               rep_levels, (size_t)num_values * sizeof(int16_t));
        writer->deferred_rep_count += num_values;
    }
    writer->deferred_count += num_values;
    return CARQUET_OK;
}

/* Replay all stashed input through the eager encode path. Called from
 * carquet_column_writer_finalize, which runs inside the OpenMP per-column
 * parallel region, so encode + compression run concurrently across columns. */
static carquet_status_t drain_deferred(
    carquet_column_writer_internal_t* writer) {
    if (writer->deferred_count == 0) return CARQUET_OK;
    carquet_status_t s = encode_batch_eager(
        writer, writer->deferred_values.data, writer->deferred_count,
        writer->max_def_level > 0 ? writer->deferred_def_levels : NULL,
        writer->max_rep_level > 0 ? writer->deferred_rep_levels : NULL);
    /* Free the stash early; the column buffer now holds the encoded pages. */
    carquet_buffer_clear(&writer->deferred_values);
    writer->deferred_count = 0;
    writer->deferred_def_count = 0;
    writer->deferred_rep_count = 0;
    return s;
}

carquet_status_t carquet_column_writer_write_batch(
    carquet_column_writer_internal_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    if (!writer || !values) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (writer->use_dictionary) {
        carquet_status_t s = dict_accumulate(writer, values, num_values,
                                             def_levels, rep_levels);
        if (s != CARQUET_OK) return s;
        writer->total_values += num_values;
        return CARQUET_OK;
    }

    if (writer->defer_encode) {
        carquet_status_t s = stash_deferred_batch(writer, values, num_values,
                                                  def_levels, rep_levels);
        if (s != CARQUET_OK) return s;
        writer->total_values += num_values;
        return CARQUET_OK;
    }

    carquet_status_t s = encode_batch_eager(writer, values, num_values,
                                            def_levels, rep_levels);
    if (s != CARQUET_OK) return s;
    writer->total_values += num_values;
    return CARQUET_OK;
}

/* ============================================================================
 * Finalization
 * ============================================================================
 */

/* Count dictionary entries from the PLAIN dictionary payload. Fixed types:
 * payload_size / stride. BYTE_ARRAY: walk 4-byte LE length prefixes. */
static int32_t dict_entry_count(carquet_physical_type_t type,
                                int32_t type_length,
                                const uint8_t* payload, size_t size) {
    if (type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        int32_t n = 0;
        size_t off = 0;
        while (off + 4 <= size) {
            uint32_t len = (uint32_t)payload[off] |
                           ((uint32_t)payload[off + 1] << 8) |
                           ((uint32_t)payload[off + 2] << 16) |
                           ((uint32_t)payload[off + 3] << 24);
            off += 4 + len;
            n++;
        }
        return n;
    }
    size_t stride = physical_type_stride(type, type_length);
    return stride ? (int32_t)(size / stride) : 0;
}

/* Compute min/max byte representation across the accumulated non-null values
 * so the RLE_DICTIONARY data page header carries the same stats the PLAIN
 * path would. */
static void dict_compute_min_max(carquet_column_writer_internal_t* writer,
                                 const uint8_t** min_out, size_t* min_len,
                                 const uint8_t** max_out, size_t* max_len) {
    *min_out = NULL; *max_out = NULL; *min_len = 0; *max_len = 0;
    if (writer->dict_value_count == 0) return;

    if (writer->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        const uint8_t* base = writer->dict_ba_storage.data;
        int64_t mn = 0, mx = 0;
        for (int64_t i = 1; i < writer->dict_value_count; i++) {
            const uint8_t* vi = base + (uintptr_t)writer->dict_ba[i].data;
            const uint8_t* vmn = base + (uintptr_t)writer->dict_ba[mn].data;
            const uint8_t* vmx = base + (uintptr_t)writer->dict_ba[mx].data;
            if (compare_stat_values(writer->type, &writer->logical_type,
                    vi, (size_t)writer->dict_ba[i].length,
                    vmn, (size_t)writer->dict_ba[mn].length) < 0) mn = i;
            if (compare_stat_values(writer->type, &writer->logical_type,
                    vi, (size_t)writer->dict_ba[i].length,
                    vmx, (size_t)writer->dict_ba[mx].length) > 0) mx = i;
        }
        *min_out = writer->dict_ba_storage.data +
                   (uintptr_t)writer->dict_ba[mn].data;
        *min_len = (size_t)writer->dict_ba[mn].length;
        *max_out = writer->dict_ba_storage.data +
                   (uintptr_t)writer->dict_ba[mx].data;
        *max_len = (size_t)writer->dict_ba[mx].length;
        return;
    }

    size_t stride = physical_type_stride(writer->type, writer->type_length);
    const uint8_t* base = writer->dict_values.data;
    size_t mn = 0, mx = 0;
    for (int64_t i = 1; i < writer->dict_value_count; i++) {
        if (compare_stat_values(writer->type, &writer->logical_type,
                base + i * stride, stride,
                base + mn * stride, stride) < 0) mn = (size_t)i;
        if (compare_stat_values(writer->type, &writer->logical_type,
                base + i * stride, stride,
                base + mx * stride, stride) > 0) mx = (size_t)i;
    }
    *min_out = base + mn * stride;
    *min_len = stride;
    *max_out = base + mx * stride;
    *max_len = stride;
}

/* Replay the accumulated values+levels through the normal PLAIN page path
 * (used on dictionary fallback). Splits into target-page-sized chunks. */
static carquet_status_t dict_fallback_to_plain(
    carquet_column_writer_internal_t* writer) {

    carquet_page_writer_set_encoding(writer->page_writer,
                                     CARQUET_ENCODING_PLAIN);
    writer->use_dictionary = false;

    int64_t total = writer->dict_total_rows;
    if (total == 0) return CARQUET_OK;

    /* Resolve BYTE_ARRAY offsets to real pointers now that storage is final. */
    if (writer->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        for (int64_t i = 0; i < writer->dict_value_count; i++) {
            writer->dict_ba[i].data = writer->dict_ba_storage.data +
                                      (uintptr_t)writer->dict_ba[i].data;
        }
    }

    size_t stride = physical_type_stride(writer->type, writer->type_length);
    int64_t row_off = 0;       /* logical rows consumed */
    int64_t val_off = 0;       /* non-null values consumed */
    int64_t max_rows = stride > 0
        ? (int64_t)(writer->target_page_size / stride)
        : 8192;
    if (max_rows < 1024) max_rows = 1024;

    while (row_off < total) {
        int64_t rows = total - row_off;
        if (rows > max_rows) rows = max_rows;

        const int16_t* dl = writer->max_def_level > 0
            ? writer->dict_def_levels + row_off : NULL;
        const int16_t* rl = (writer->max_rep_level > 0 && writer->dict_rep_levels)
            ? writer->dict_rep_levels + row_off : NULL;

        /* Count non-null values in this row span. */
        int64_t nn = rows;
        if (dl) {
            nn = carquet_dispatch_count_non_nulls(
                dl, rows, writer->max_def_level);
        }

        /* add_values rejects a NULL values pointer even when every row in
         * the span is null; pass a valid dummy in that case. */
        static const uint8_t dummy_value[16] = {0};
        const void* vals;
        if (writer->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
            vals = (nn > 0 && writer->dict_ba)
                ? (const void*)(writer->dict_ba + val_off)
                : (const void*)dummy_value;
        } else {
            vals = (nn > 0 && writer->dict_values.data)
                ? (const void*)(writer->dict_values.data +
                                (size_t)val_off * stride)
                : (const void*)dummy_value;
        }

        carquet_status_t s = carquet_page_writer_add_values(
            writer->page_writer, vals, rows, dl, rl);
        if (s != CARQUET_OK) return s;

        s = flush_current_page(writer);
        if (s != CARQUET_OK) return s;

        row_off += rows;
        val_off += nn;
    }
    return CARQUET_OK;
}

/* Build the dictionary, decide fallback, and emit the dictionary page
 * followed by a single RLE_DICTIONARY data page. */
static carquet_status_t finalize_dictionary(
    carquet_column_writer_internal_t* writer) {

    /* No data written to this column chunk: emit nothing, exactly like the
     * empty PLAIN path (flush_current_page early-returns on 0 values). */
    if (writer->dict_total_rows == 0) {
        return CARQUET_OK;
    }

    /* All values null: there is nothing to dictionary-encode. Fall back to
     * the PLAIN path which correctly emits a levels-only data page. */
    if (writer->dict_value_count == 0) {
        return dict_fallback_to_plain(writer);
    }

    /* Resolve BYTE_ARRAY offsets to pointers for the encoder (dict_value_count
     * > 0 here; the empty/all-null cases returned above). */
    carquet_byte_array_t* ba_resolved = NULL;
    if (writer->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        ba_resolved = carquet_mem_malloc((size_t)writer->dict_value_count *
                             sizeof(carquet_byte_array_t));
        if (!ba_resolved) return CARQUET_ERROR_OUT_OF_MEMORY;
        for (int64_t i = 0; i < writer->dict_value_count; i++) {
            ba_resolved[i].length = writer->dict_ba[i].length;
            ba_resolved[i].data = writer->dict_ba_storage.data +
                                  (uintptr_t)writer->dict_ba[i].data;
        }
    }

    carquet_buffer_t dict_out, idx_out;
    carquet_buffer_init(&dict_out);
    carquet_buffer_init(&idx_out);

    carquet_status_t status;
    int64_t n = writer->dict_value_count;
    bool dict_abandoned = false;
    const void* fixed_in = (writer->type == CARQUET_PHYSICAL_BYTE_ARRAY)
        ? NULL : (const void*)writer->dict_values.data;
    const carquet_byte_array_t* ba_in =
        (writer->type == CARQUET_PHYSICAL_BYTE_ARRAY) ? ba_resolved : NULL;

    /* Single pass with an early-abort budget: if the PLAIN dictionary would
     * exceed dictionary_page_size, the encoder stops immediately instead of
     * scanning the rest of the chunk and serializing indices we would only
     * throw away on fallback. */
    status = carquet_dictionary_encode_capped(
        writer->type, writer->type_length, fixed_in, ba_in, n,
        writer->dictionary_page_size_limit,
        &dict_out, &idx_out, &dict_abandoned);
    carquet_mem_free(ba_resolved);
    if (status != CARQUET_OK) {
        carquet_buffer_destroy(&dict_out);
        carquet_buffer_destroy(&idx_out);
        return status;
    }
    if (dict_abandoned) {
        carquet_buffer_destroy(&dict_out);
        carquet_buffer_destroy(&idx_out);
        return dict_fallback_to_plain(writer);
    }

    int32_t num_unique = dict_entry_count(writer->type, writer->type_length,
                                          dict_out.data, dict_out.size);

    /* num_unique is the exact number of distinct non-null values in this chunk
     * (it stays correct whether the dictionary is kept below or falls back to
     * PLAIN for being all-unique), so record it for Statistics.distinct_count. */
    writer->has_distinct_count = true;
    writer->distinct_count = num_unique;

    /* The dictionary fit under the size budget, but if it is effectively
     * all-unique it provides no benefit, so fall back to PLAIN. (This path
     * completed a full cheap pass; only the catastrophic high-cardinality
     * case is short-circuited by the early abort above.) */
    if ((int64_t)num_unique >= writer->dict_value_count) {
        carquet_buffer_destroy(&dict_out);
        carquet_buffer_destroy(&idx_out);
        return dict_fallback_to_plain(writer);
    }

    /* Emit the dictionary page first into the column buffer. */
    size_t dp_size = 0;
    int32_t dp_uncomp = 0, dp_comp = 0;
    status = carquet_page_writer_emit_dictionary_page(
        writer->page_writer, &writer->column_buffer,
        dict_out.data, dict_out.size, num_unique,
        &dp_size, &dp_uncomp, &dp_comp);
    carquet_buffer_destroy(&dict_out);
    if (status != CARQUET_OK) {
        carquet_buffer_destroy(&idx_out);
        return status;
    }
    writer->has_dictionary_page = true;
    writer->dictionary_page_size_bytes = (int64_t)dp_size;
    writer->total_uncompressed_size += dp_uncomp;
    writer->total_compressed_size += dp_comp;

    /* Stage the RLE_DICTIONARY data page. */
    const int16_t* dl = writer->max_def_level > 0
        ? writer->dict_def_levels : NULL;
    const int16_t* rl = (writer->max_rep_level > 0 && writer->dict_rep_levels)
        ? writer->dict_rep_levels : NULL;

    status = carquet_page_writer_add_dictionary_indices(
        writer->page_writer, idx_out.data, idx_out.size,
        dl, rl, writer->dict_total_rows, writer->dict_total_nulls);
    carquet_buffer_destroy(&idx_out);
    if (status != CARQUET_OK) return status;

    carquet_page_writer_set_encoding(writer->page_writer,
                                     CARQUET_ENCODING_RLE_DICTIONARY);

    /* Inject column statistics so the data page header matches PLAIN. */
    const uint8_t *mn = NULL, *mx = NULL;
    size_t mn_len = 0, mx_len = 0;
    dict_compute_min_max(writer, &mn, &mn_len, &mx, &mx_len);
    if (mn && mx) {
        status = carquet_page_writer_set_min_max(writer->page_writer,
                                                 mn, mn_len, mx, mx_len);
        if (status != CARQUET_OK) return status;
    }

    /* The RLE_DICTIONARY data page stages indices, not raw values, so the page
     * writer never accumulated the unencoded BYTE_ARRAY byte total. Compute it
     * from the (non-deduplicated) accumulated values and inject it so the
     * OffsetIndex reports the correct unencoded_byte_array_data_bytes. */
    if (writer->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        int64_t ba_bytes = 0;
        for (int64_t i = 0; i < writer->dict_value_count; i++) {
            ba_bytes += writer->dict_ba[i].length;
        }
        carquet_page_writer_set_byte_array_bytes(writer->page_writer, ba_bytes);
    }

    /* flush_current_page reads stats off the page writer, finalizes the data
     * page into column_buffer, and updates column index / offset index. The
     * dictionary page already written is intentionally NOT a data page for
     * offset-index purposes. */
    return flush_current_page(writer);
}

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

    carquet_status_t status;
    if (writer->use_dictionary) {
        status = finalize_dictionary(writer);
        if (status != CARQUET_OK) {
            return status;
        }
    } else {
        /* Replay any deferred input here. This runs inside the OpenMP
         * per-column parallel finalize, so encode + compression of distinct
         * columns proceed concurrently. */
        status = drain_deferred(writer);
        if (status != CARQUET_OK) {
            return status;
        }
        /* Flush any remaining data */
        status = flush_current_page(writer);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    /* Convert relative page offsets (recorded during eager flushes) into
     * absolute file offsets now that this column's start in the file is
     * known. The serial finalize path sets column_file_offset before
     * calling us; the parallel path leaves it 0 and does not build an
     * offset index (write_page_index is disabled in that mode). */
    if (writer->offset_index && writer->column_file_offset != 0) {
        carquet_offset_index_builder_shift_offsets(
            writer->offset_index, writer->column_file_offset);
    }

    if (data) *data = writer->column_buffer.data;
    if (size) *size = writer->column_buffer.size;
    if (total_values) *total_values = writer->total_values;
    if (total_compressed_size) *total_compressed_size = writer->total_compressed_size;
    if (total_uncompressed_size) *total_uncompressed_size = writer->total_uncompressed_size;

    return CARQUET_OK;
}

void carquet_column_writer_set_statistics(
    carquet_column_writer_internal_t* writer,
    bool enabled) {
    if (writer && writer->page_writer) {
        carquet_page_writer_set_statistics(writer->page_writer, enabled);
    }
}

bool carquet_column_writer_get_statistics(
    const carquet_column_writer_internal_t* writer,
    const uint8_t** min_value,
    size_t* min_size,
    const uint8_t** max_value,
    size_t* max_size,
    int64_t* null_count) {
    if (!writer) return false;
    if (null_count) *null_count = writer->total_nulls;
    if (!writer->has_min_max) {
        if (min_value) *min_value = NULL;
        if (max_value) *max_value = NULL;
        if (min_size) *min_size = 0;
        if (max_size) *max_size = 0;
        return false;
    }
    if (min_value) *min_value = writer->min_value;
    if (max_value) *max_value = writer->max_value;
    if (min_size) *min_size = writer->min_value_size;
    if (max_size) *max_size = writer->max_value_size;
    return true;
}

int64_t carquet_column_writer_num_values(const carquet_column_writer_internal_t* writer) {
    return writer ? writer->total_values : 0;
}

/* Exact distinct non-null value count for the finalized chunk, available only
 * when a dictionary was built (see finalize_dictionary). Returns false and
 * leaves *count untouched otherwise. */
bool carquet_column_writer_get_distinct_count(
    const carquet_column_writer_internal_t* writer, int64_t* count) {
    if (!writer || !writer->has_distinct_count) return false;
    if (count) *count = writer->distinct_count;
    return true;
}

/* Chunk-level SizeStatistics (Parquet 2.9). Returns the accumulated per-level
 * histograms (owned by the writer, valid until reset/destroy) and the total
 * unencoded BYTE_ARRAY byte count. Histogram lengths are max_rep/def_level + 1. */
void carquet_column_writer_get_size_statistics(
    const carquet_column_writer_internal_t* writer,
    int64_t* unencoded_byte_array_bytes,
    const int64_t** rep_level_hist, int32_t* rep_len,
    const int64_t** def_level_hist, int32_t* def_len) {
    if (!writer) return;
    if (unencoded_byte_array_bytes) {
        *unencoded_byte_array_bytes =
            writer->type == CARQUET_PHYSICAL_BYTE_ARRAY
                ? writer->chunk_unencoded_ba_bytes : -1;
    }
    if (rep_level_hist) *rep_level_hist = writer->chunk_rep_hist;
    if (rep_len) *rep_len = (int32_t)writer->max_rep_level + 1;
    if (def_level_hist) *def_level_hist = writer->chunk_def_hist;
    if (def_len) *def_len = (int32_t)writer->max_def_level + 1;
}

bool carquet_column_writer_has_dictionary_page(
    const carquet_column_writer_internal_t* writer) {
    return writer && writer->has_dictionary_page;
}

int64_t carquet_column_writer_dictionary_page_size(
    const carquet_column_writer_internal_t* writer) {
    return writer ? writer->dictionary_page_size_bytes : 0;
}

void carquet_column_writer_set_dictionary_page_size_limit(
    carquet_column_writer_internal_t* writer, int64_t limit) {
    if (writer && limit > 0) {
        writer->dictionary_page_size_limit = (size_t)limit;
    }
}

void carquet_column_writer_set_max_rows_per_page(
    carquet_column_writer_internal_t* writer, int64_t max_rows) {
    if (writer && max_rows > 0) {
        writer->max_rows_per_page = max_rows;
    }
}

/* Per-column override for the byte-based page-flush trigger. Safe between page
 * flushes; the new size kicks in for the next page being filled. */
void carquet_column_writer_set_target_page_size(
    carquet_column_writer_internal_t* writer, int64_t bytes) {
    if (writer && bytes > 0) {
        writer->target_page_size = (size_t)bytes;
        writer->max_page_size = writer->target_page_size * 2;
    }
}

void carquet_column_writer_set_write_batch_size(
    carquet_column_writer_internal_t* writer, int64_t batch_size) {
    if (writer && batch_size > 0) {
        writer->write_batch_size = batch_size;
    }
}

int32_t carquet_column_writer_num_pages(const carquet_column_writer_internal_t* writer) {
    return writer ? writer->num_pages : 0;
}

void carquet_column_writer_enable_bloom_filter_fpp(
    carquet_column_writer_internal_t* writer, int64_t ndv, double fpp);

void carquet_column_writer_enable_bloom_filter(
    carquet_column_writer_internal_t* writer, int64_t ndv) {
    carquet_column_writer_enable_bloom_filter_fpp(writer, ndv, 0.01);
}

void carquet_column_writer_enable_bloom_filter_fpp(
    carquet_column_writer_internal_t* writer, int64_t ndv, double fpp) {
    if (!writer || writer->bloom_filter) return;
    writer->bloom_ndv = ndv > 0 ? ndv : 100000;
    writer->bloom_fpp = (fpp > 0.0 && fpp < 1.0) ? fpp : 0.01;
    writer->bloom_filter = carquet_bloom_filter_create_with_ndv(
        writer->bloom_ndv, writer->bloom_fpp);
}

/* Reconfigure (or enable) the bloom filter with explicit ndv/fpp. Safe to
 * call before any data is written; recreates the filter if one already
 * exists so per-column overrides take effect. */
void carquet_column_writer_configure_bloom_filter(
    carquet_column_writer_internal_t* writer,
    bool enabled, int64_t ndv, double fpp) {
    if (!writer) return;
    if (writer->bloom_filter) {
        carquet_bloom_filter_destroy(writer->bloom_filter);
        writer->bloom_filter = NULL;
    }
    if (!enabled) {
        writer->bloom_ndv = 0;
        return;
    }
    writer->bloom_ndv = ndv > 0 ? ndv : 100000;
    writer->bloom_fpp = (fpp > 0.0 && fpp < 1.0) ? fpp : 0.01;
    writer->bloom_filter = carquet_bloom_filter_create_with_ndv(
        writer->bloom_ndv, writer->bloom_fpp);
}

void carquet_column_writer_enable_page_index(
    carquet_column_writer_internal_t* writer) {
    if (!writer) return;
    writer->page_index_enabled = true;
    if (!writer->column_index) {
        writer->column_index = carquet_column_index_builder_create(
            writer->type, &writer->logical_type, writer->type_length);
    }
    if (!writer->offset_index) {
        writer->offset_index = carquet_offset_index_builder_create(
            writer->type == CARQUET_PHYSICAL_BYTE_ARRAY);
    }
}

void carquet_column_writer_set_file_offset(
    carquet_column_writer_internal_t* writer, int64_t offset) {
    if (writer) writer->column_file_offset = offset;
}

carquet_bloom_filter_t* carquet_column_writer_get_bloom_filter(
    const carquet_column_writer_internal_t* writer) {
    return writer ? writer->bloom_filter : NULL;
}

carquet_column_index_builder_t* carquet_column_writer_get_column_index(
    const carquet_column_writer_internal_t* writer) {
    return writer ? writer->column_index : NULL;
}

carquet_offset_index_builder_t* carquet_column_writer_get_offset_index(
    const carquet_column_writer_internal_t* writer) {
    return writer ? writer->offset_index : NULL;
}
