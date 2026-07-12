/**
 * @file page_writer.c
 * @brief Data page and dictionary page creation
 *
 * Handles encoding values into pages with proper headers,
 * definition/repetition levels, and compression.
 */

#include "core/allocator.h"
#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/buffer.h"
#include "core/float16.h"
#include "core/geo_wkb.h"
#include "encoding/plain.h"
#include "encoding/rle.h"
#include "compression/custom.h"
#include "thrift/thrift_decode.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Forward declarations for compression */
extern carquet_status_t carquet_snappy_compress(const uint8_t* src, size_t src_size,
                                                 uint8_t* dst, size_t dst_capacity,
                                                 size_t* dst_size);
extern size_t carquet_snappy_compress_bound(size_t src_size);

/* CRC32 for page integrity verification */
extern uint32_t carquet_crc32(const uint8_t* data, size_t length);
extern uint32_t carquet_crc32_update(uint32_t crc, const uint8_t* data, size_t length);

extern carquet_status_t carquet_lz4_compress(const uint8_t* src, size_t src_size,
                                              uint8_t* dst, size_t dst_capacity,
                                              size_t* dst_size);
extern size_t carquet_lz4_compress_bound(size_t src_size);
extern carquet_status_t carquet_lz4_hadoop_compress(const uint8_t* src, size_t src_size,
                                                    uint8_t* dst, size_t dst_capacity,
                                                    size_t* dst_size);
extern size_t carquet_lz4_hadoop_compress_bound(size_t src_size);

extern int carquet_gzip_compress(const uint8_t* src, size_t src_size,
                                  uint8_t* dst, size_t dst_capacity,
                                  size_t* dst_size, int level);
extern size_t carquet_gzip_compress_bound(size_t src_size);

extern int carquet_zstd_compress(const uint8_t* src, size_t src_size,
                                  uint8_t* dst, size_t dst_capacity,
                                  size_t* dst_size, int level);
extern size_t carquet_zstd_compress_bound(size_t src_size);

extern carquet_status_t carquet_byte_stream_split_encode_float(
    const float* values,
    int64_t count,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written);
extern carquet_status_t carquet_byte_stream_split_encode_double(
    const double* values,
    int64_t count,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written);
extern carquet_status_t carquet_byte_stream_split_encode(
    const uint8_t* values,
    int64_t count,
    int32_t type_length,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written);
extern carquet_status_t carquet_delta_encode_int32(
    const int32_t* values, int32_t num_values,
    uint8_t* data, size_t data_capacity, size_t* bytes_written);
extern carquet_status_t carquet_delta_encode_int64(
    const int64_t* values, int32_t num_values,
    uint8_t* data, size_t data_capacity, size_t* bytes_written);
extern carquet_status_t carquet_delta_length_encode(
    const carquet_byte_array_t* values, int32_t num_values,
    carquet_buffer_t* output);
extern carquet_status_t carquet_delta_strings_encode(
    const carquet_byte_array_t* values, int32_t num_values,
    carquet_buffer_t* output);
extern int64_t carquet_dispatch_count_non_nulls(const int16_t* def_levels, int64_t count,
                                                 int16_t max_def_level);
extern void carquet_dispatch_minmax_i32(const int32_t* values, int64_t count,
                                         int32_t* min_value, int32_t* max_value);
extern void carquet_dispatch_minmax_i64(const int64_t* values, int64_t count,
                                         int64_t* min_value, int64_t* max_value);
extern void carquet_dispatch_minmax_float(const float* values, int64_t count,
                                           float* min_value, float* max_value);
extern void carquet_dispatch_minmax_double(const double* values, int64_t count,
                                            double* min_value, double* max_value);
extern void carquet_dispatch_copy_minmax_i32(const int32_t* values, int64_t count, int32_t* output,
                                              int32_t* min_value, int32_t* max_value);
extern void carquet_dispatch_copy_minmax_i64(const int64_t* values, int64_t count, int64_t* output,
                                              int64_t* min_value, int64_t* max_value);
extern void carquet_dispatch_copy_minmax_float(const float* values, int64_t count, float* output,
                                                float* min_value, float* max_value);
extern void carquet_dispatch_copy_minmax_double(const double* values, int64_t count, double* output,
                                                 double* min_value, double* max_value);

/* ============================================================================
 * Page Writer Structure
 * ============================================================================
 */

typedef struct carquet_page_writer {
    carquet_buffer_t values_buffer;      /* Encoded values */
    carquet_buffer_t def_levels_buffer;  /* Definition levels (RLE) */
    carquet_buffer_t rep_levels_buffer;  /* Repetition levels (RLE) */
    carquet_buffer_t staging_buffer;     /* Reusable page payload staging */
    carquet_buffer_t page_buffer;        /* Final page with header */
    carquet_buffer_t compress_buffer;   /* Reusable compression buffer */

    carquet_physical_type_t type;
    carquet_logical_type_t logical_type;
    carquet_encoding_t encoding;
    carquet_compression_t compression;

    int16_t max_def_level;
    int16_t max_rep_level;
    int32_t type_length;  /* For FIXED_LEN_BYTE_ARRAY */

    int64_t num_values;
    int64_t num_nulls;
    int64_t num_rows;        /* Logical rows (rep_level==0); used by V2 only */
    /* Sum of the lengths of all non-null BYTE_ARRAY values in the current
     * page, exclusive of the length prefixes. This is the Parquet 2.9
     * "unencoded_byte_array_data_bytes" quantity (OffsetIndex field 2 /
     * SizeStatistics). Zero for non-BYTE_ARRAY columns. Reset per page. */
    int64_t byte_array_data_bytes;
    /* Per-page repetition/definition level histograms (Parquet 2.9). Entry i
     * counts values in the current page whose level == i; lengths are
     * max_rep_level+1 and max_def_level+1. Allocated at create, reset per page.
     * Feed both SizeStatistics (chunk sum) and ColumnIndex (per page). */
    int64_t* rep_level_hist;
    int64_t* def_level_hist;

    bool data_page_v2;       /* Emit DATA_PAGE_V2 instead of DATA_PAGE */

    int32_t compression_level;   /* 0 = use codec default */

    /* Options */
    bool write_crc;          /* Compute and write CRC32 for pages */
    bool write_statistics;   /* Write min/max statistics in page header */

    /* Statistics tracking.
     *
     * min_value / max_value are heap-allocated so arbitrary-length BYTE_ARRAY
     * and FIXED_LEN_BYTE_ARRAY values fit. For fixed-size numeric types the
     * bytes are the raw little-endian representation. For BOOLEAN they are
     * 1-byte 0/1. min and max can have different sizes (byte arrays).
     */
    bool has_min_max;
    uint8_t* min_value;
    size_t min_value_size;
    size_t min_value_capacity;
    uint8_t* max_value;
    size_t max_value_size;
    size_t max_value_capacity;

    /* BOOLEAN stats are accumulated as flags and collapsed at the end. */
    bool bool_seen_false;
    bool bool_seen_true;

    /* Compatibility alias: many code paths use a single "size" when the type
     * has fixed-width stats. Numeric paths set both _size fields equal. */
    size_t min_max_size;

    /* GeospatialStatistics accumulator (GEOMETRY/GEOGRAPHY columns). Unlike
     * min/max it is NOT cleared per page — it spans the whole column chunk
     * (one page_writer lifetime), so the column writer reads it once at
     * finalize. */
    bool geo_enabled;
    parquet_geospatial_statistics_t geo_stats;
} carquet_page_writer_t;

static bool stats_order_defined_for_logical(const carquet_logical_type_t* lt) {
    if (!lt) return true;
    switch (lt->id) {
        case CARQUET_LOGICAL_GEOMETRY:
        case CARQUET_LOGICAL_GEOGRAPHY:
        case CARQUET_LOGICAL_VARIANT:
        case CARQUET_LOGICAL_MAP:
        case CARQUET_LOGICAL_LIST:
            return false;
        default:
            return true;
    }
}

static bool logical_integer_is_unsigned(const carquet_logical_type_t* lt) {
    return lt &&
           lt->id == CARQUET_LOGICAL_INTEGER &&
           !lt->params.integer.is_signed;
}

static carquet_status_t stats_grow(uint8_t** buf, size_t* cap, size_t need) {
    if (need <= *cap) return CARQUET_OK;
    size_t new_cap = *cap == 0 ? 64 : *cap;
    while (new_cap < need) new_cap *= 2;
    uint8_t* p = carquet_mem_realloc(*buf, new_cap);
    if (!p) return CARQUET_ERROR_OUT_OF_MEMORY;
    *buf = p;
    *cap = new_cap;
    return CARQUET_OK;
}

static carquet_status_t stats_set_min(carquet_page_writer_t* w,
                                       const void* src, size_t size) {
    carquet_status_t s = stats_grow(&w->min_value, &w->min_value_capacity, size);
    if (s != CARQUET_OK) return s;
    memcpy(w->min_value, src, size);
    w->min_value_size = size;
    return CARQUET_OK;
}

static carquet_status_t stats_set_max(carquet_page_writer_t* w,
                                       const void* src, size_t size) {
    carquet_status_t s = stats_grow(&w->max_value, &w->max_value_capacity, size);
    if (s != CARQUET_OK) return s;
    memcpy(w->max_value, src, size);
    w->max_value_size = size;
    return CARQUET_OK;
}


/* Forward declaration for internal use */
void carquet_page_writer_destroy(carquet_page_writer_t* writer);
carquet_status_t carquet_page_writer_finalize_to_buffer(
    carquet_page_writer_t* writer,
    carquet_buffer_t* output_buffer,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size);

/* ============================================================================
 * Page Writer Lifecycle
 * ============================================================================
 */

carquet_page_writer_t* carquet_page_writer_create(
    carquet_physical_type_t type,
    const carquet_logical_type_t* logical_type,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    int32_t compression_level) {

    carquet_page_writer_t* writer = carquet_mem_calloc(1, sizeof(carquet_page_writer_t));
    if (!writer) return NULL;

    carquet_buffer_init(&writer->values_buffer);
    carquet_buffer_init(&writer->def_levels_buffer);
    carquet_buffer_init(&writer->rep_levels_buffer);
    carquet_buffer_init(&writer->staging_buffer);
    carquet_buffer_init(&writer->page_buffer);
    carquet_buffer_init(&writer->compress_buffer);

    writer->type = type;
    if (logical_type) {
        writer->logical_type = *logical_type;
        writer->geo_enabled =
            (logical_type->id == CARQUET_LOGICAL_GEOMETRY ||
             logical_type->id == CARQUET_LOGICAL_GEOGRAPHY);
    }
    if (writer->geo_enabled) {
        carquet_geo_stats_init(&writer->geo_stats);
    }
    writer->encoding = encoding;
    writer->compression = compression;
    writer->max_def_level = max_def_level;
    writer->max_rep_level = max_rep_level;
    writer->type_length = type_length;
    writer->compression_level = compression_level;
    writer->write_crc = true;         /* Enable CRC by default for integrity */
    writer->write_statistics = stats_order_defined_for_logical(logical_type);

    writer->def_level_hist =
        carquet_mem_calloc((size_t)max_def_level + 1, sizeof(int64_t));
    writer->rep_level_hist =
        carquet_mem_calloc((size_t)max_rep_level + 1, sizeof(int64_t));
    if (!writer->def_level_hist || !writer->rep_level_hist) {
        carquet_page_writer_destroy(writer);
        return NULL;
    }

    return writer;
}

void carquet_page_writer_destroy(carquet_page_writer_t* writer) {
    if (writer) {
        carquet_buffer_destroy(&writer->values_buffer);
        carquet_buffer_destroy(&writer->def_levels_buffer);
        carquet_buffer_destroy(&writer->rep_levels_buffer);
        carquet_buffer_destroy(&writer->staging_buffer);
        carquet_buffer_destroy(&writer->page_buffer);
        carquet_buffer_destroy(&writer->compress_buffer);
        carquet_mem_free(writer->min_value);
        carquet_mem_free(writer->max_value);
        carquet_mem_free(writer->def_level_hist);
        carquet_mem_free(writer->rep_level_hist);
        carquet_mem_free(writer);
    }
}

void carquet_page_writer_reset(carquet_page_writer_t* writer) {
    carquet_buffer_clear(&writer->values_buffer);
    carquet_buffer_clear(&writer->def_levels_buffer);
    carquet_buffer_clear(&writer->rep_levels_buffer);
    carquet_buffer_clear(&writer->staging_buffer);
    carquet_buffer_clear(&writer->page_buffer);
    carquet_buffer_clear(&writer->compress_buffer);
    writer->num_values = 0;
    writer->num_nulls = 0;
    writer->num_rows = 0;
    writer->byte_array_data_bytes = 0;
    if (writer->def_level_hist) {
        memset(writer->def_level_hist, 0,
               ((size_t)writer->max_def_level + 1) * sizeof(int64_t));
    }
    if (writer->rep_level_hist) {
        memset(writer->rep_level_hist, 0,
               ((size_t)writer->max_rep_level + 1) * sizeof(int64_t));
    }
    writer->has_min_max = false;
    writer->min_max_size = 0;
    writer->bool_seen_false = false;
    writer->bool_seen_true = false;
}

/* ============================================================================
 * Level Encoding (RLE/Bit-Packed Hybrid)
 * ============================================================================
 */

static int bit_width_for_max(int16_t max_level) {
    if (max_level == 0) return 0;
    int width = 0;
    int16_t val = max_level;
    while (val > 0) {
        width++;
        val >>= 1;
    }
    return width;
}

/* V1 data pages prefix each level section with a 4-byte little-endian byte
 * length; V2 data pages omit it (the length lives in DataPageHeaderV2), so
 * with_prefix is false for V2. */
static carquet_status_t encode_levels(
    const int16_t* levels,
    int64_t count,
    int16_t max_level,
    carquet_buffer_t* output,
    bool with_prefix) {

    if (max_level == 0) {
        return CARQUET_OK;
    }

    int bit_width = bit_width_for_max(max_level);
    size_t prefix_offset = output->size;
    if (with_prefix) {
        carquet_status_t ps = carquet_buffer_append_u32_le(output, 0);
        if (ps != CARQUET_OK) {
            return ps;
        }
    }

    carquet_status_t status;
    size_t encoded_offset = output->size;
    if (levels) {
        status = carquet_rle_encode_levels(levels, count, bit_width, output);
    } else {
        carquet_rle_encoder_t enc;
        carquet_rle_encoder_init(&enc, output, bit_width);
        status = carquet_rle_encoder_put_repeat(&enc, (uint32_t)max_level, count);
        if (status == CARQUET_OK) {
            status = carquet_rle_encoder_flush(&enc);
        }
    }

    if (status != CARQUET_OK) {
        output->size = prefix_offset;
        return status;
    }

    size_t encoded_size = output->size - encoded_offset;
    if (encoded_size > UINT32_MAX) {
        output->size = prefix_offset;
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    if (with_prefix) {
        output->data[prefix_offset] = (uint8_t)(encoded_size & 0xFF);
        output->data[prefix_offset + 1] = (uint8_t)((encoded_size >> 8) & 0xFF);
        output->data[prefix_offset + 2] = (uint8_t)((encoded_size >> 16) & 0xFF);
        output->data[prefix_offset + 3] = (uint8_t)((encoded_size >> 24) & 0xFF);
    }
    return CARQUET_OK;
}

/* ============================================================================
 * Statistics Tracking
 * ============================================================================
 */

static carquet_status_t stats_set_fixed(carquet_page_writer_t* writer,
                                         const void* min_src, const void* max_src,
                                         size_t size) {
    carquet_status_t s = stats_set_min(writer, min_src, size);
    if (s != CARQUET_OK) return s;
    s = stats_set_max(writer, max_src, size);
    if (s != CARQUET_OK) return s;
    writer->min_max_size = size;
    writer->has_min_max = true;
    return CARQUET_OK;
}

static void update_statistics_i32(carquet_page_writer_t* writer,
                                   const int32_t* values, int64_t count) {
    if (count <= 0) return;
    int32_t chunk_min, chunk_max;
    uint32_t chunk_umin = 0, chunk_umax = 0;
    bool unsigned_order = logical_integer_is_unsigned(&writer->logical_type);
    if (unsigned_order) {
        const uint32_t* u = (const uint32_t*)values;
        chunk_umin = u[0];
        chunk_umax = u[0];
        for (int64_t i = 1; i < count; i++) {
            if (u[i] < chunk_umin) chunk_umin = u[i];
            if (u[i] > chunk_umax) chunk_umax = u[i];
        }
        memcpy(&chunk_min, &chunk_umin, sizeof(chunk_min));
        memcpy(&chunk_max, &chunk_umax, sizeof(chunk_max));
    } else {
        carquet_dispatch_minmax_i32(values, count, &chunk_min, &chunk_max);
    }
    if (!writer->has_min_max) {
        stats_set_fixed(writer, &chunk_min, &chunk_max, sizeof(int32_t));
        return;
    }
    if (unsigned_order) {
        uint32_t min_v, max_v;
        memcpy(&min_v, writer->min_value, sizeof(min_v));
        memcpy(&max_v, writer->max_value, sizeof(max_v));
        if (chunk_umin < min_v) min_v = chunk_umin;
        if (chunk_umax > max_v) max_v = chunk_umax;
        memcpy(writer->min_value, &min_v, sizeof(min_v));
        memcpy(writer->max_value, &max_v, sizeof(max_v));
    } else {
        int32_t min_v, max_v;
        memcpy(&min_v, writer->min_value, sizeof(min_v));
        memcpy(&max_v, writer->max_value, sizeof(max_v));
        if (chunk_min < min_v) min_v = chunk_min;
        if (chunk_max > max_v) max_v = chunk_max;
        memcpy(writer->min_value, &min_v, sizeof(min_v));
        memcpy(writer->max_value, &max_v, sizeof(max_v));
    }
}

static void update_statistics_i64(carquet_page_writer_t* writer,
                                   const int64_t* values, int64_t count) {
    if (count <= 0) return;
    int64_t chunk_min, chunk_max;
    uint64_t chunk_umin = 0, chunk_umax = 0;
    bool unsigned_order = logical_integer_is_unsigned(&writer->logical_type);
    if (unsigned_order) {
        const uint64_t* u = (const uint64_t*)values;
        chunk_umin = u[0];
        chunk_umax = u[0];
        for (int64_t i = 1; i < count; i++) {
            if (u[i] < chunk_umin) chunk_umin = u[i];
            if (u[i] > chunk_umax) chunk_umax = u[i];
        }
        memcpy(&chunk_min, &chunk_umin, sizeof(chunk_min));
        memcpy(&chunk_max, &chunk_umax, sizeof(chunk_max));
    } else {
        carquet_dispatch_minmax_i64(values, count, &chunk_min, &chunk_max);
    }
    if (!writer->has_min_max) {
        stats_set_fixed(writer, &chunk_min, &chunk_max, sizeof(int64_t));
        return;
    }
    if (unsigned_order) {
        uint64_t min_v, max_v;
        memcpy(&min_v, writer->min_value, sizeof(min_v));
        memcpy(&max_v, writer->max_value, sizeof(max_v));
        if (chunk_umin < min_v) min_v = chunk_umin;
        if (chunk_umax > max_v) max_v = chunk_umax;
        memcpy(writer->min_value, &min_v, sizeof(min_v));
        memcpy(writer->max_value, &max_v, sizeof(max_v));
    } else {
        int64_t min_v, max_v;
        memcpy(&min_v, writer->min_value, sizeof(min_v));
        memcpy(&max_v, writer->max_value, sizeof(max_v));
        if (chunk_min < min_v) min_v = chunk_min;
        if (chunk_max > max_v) max_v = chunk_max;
        memcpy(writer->min_value, &min_v, sizeof(min_v));
        memcpy(writer->max_value, &max_v, sizeof(max_v));
    }
}

/* SIMD min/max with NaN-skipping fallback. The dispatched minmax does not
 * filter NaNs, so when its result contains NaN we rescan scalar-wise to skip
 * NaN values per Parquet's float/double statistics semantics. */
static bool float_minmax_nan_safe(const float* values, int64_t count,
                                   float* out_min, float* out_max) {
    float chunk_min, chunk_max;
    carquet_dispatch_minmax_float(values, count, &chunk_min, &chunk_max);
    if (!isnan(chunk_min) && !isnan(chunk_max)) {
        *out_min = chunk_min;
        *out_max = chunk_max;
        return true;
    }
    bool found = false;
    for (int64_t i = 0; i < count; i++) {
        float v = values[i];
        if (isnan(v)) continue;
        if (!found) {
            chunk_min = v;
            chunk_max = v;
            found = true;
        } else {
            if (v < chunk_min) chunk_min = v;
            if (v > chunk_max) chunk_max = v;
        }
    }
    if (!found) return false;
    *out_min = chunk_min;
    *out_max = chunk_max;
    return true;
}

static bool double_minmax_nan_safe(const double* values, int64_t count,
                                    double* out_min, double* out_max) {
    double chunk_min, chunk_max;
    carquet_dispatch_minmax_double(values, count, &chunk_min, &chunk_max);
    if (!isnan(chunk_min) && !isnan(chunk_max)) {
        *out_min = chunk_min;
        *out_max = chunk_max;
        return true;
    }
    bool found = false;
    for (int64_t i = 0; i < count; i++) {
        double v = values[i];
        if (isnan(v)) continue;
        if (!found) {
            chunk_min = v;
            chunk_max = v;
            found = true;
        } else {
            if (v < chunk_min) chunk_min = v;
            if (v > chunk_max) chunk_max = v;
        }
    }
    if (!found) return false;
    *out_min = chunk_min;
    *out_max = chunk_max;
    return true;
}

static void update_statistics_float(carquet_page_writer_t* writer,
                                     const float* values, int64_t count) {
    if (count <= 0) return;
    float chunk_min, chunk_max;
    if (!float_minmax_nan_safe(values, count, &chunk_min, &chunk_max)) return;
    /* Parquet stats: distinguish +0.0 and -0.0 in min/max for correct ordering. */
    if (chunk_min == 0.0f) chunk_min = -0.0f;
    if (chunk_max == 0.0f) chunk_max = 0.0f;
    if (!writer->has_min_max) {
        stats_set_fixed(writer, &chunk_min, &chunk_max, sizeof(float));
        return;
    }
    float min_v, max_v;
    memcpy(&min_v, writer->min_value, sizeof(min_v));
    memcpy(&max_v, writer->max_value, sizeof(max_v));
    if (chunk_min < min_v) min_v = chunk_min;
    if (chunk_max > max_v) max_v = chunk_max;
    if (min_v == 0.0f) min_v = -0.0f;
    if (max_v == 0.0f) max_v = 0.0f;
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
}

static void update_statistics_double(carquet_page_writer_t* writer,
                                      const double* values, int64_t count) {
    if (count <= 0) return;
    double chunk_min, chunk_max;
    if (!double_minmax_nan_safe(values, count, &chunk_min, &chunk_max)) return;
    if (chunk_min == 0.0) chunk_min = -0.0;
    if (chunk_max == 0.0) chunk_max = 0.0;
    if (!writer->has_min_max) {
        stats_set_fixed(writer, &chunk_min, &chunk_max, sizeof(double));
        return;
    }
    double min_v, max_v;
    memcpy(&min_v, writer->min_value, sizeof(min_v));
    memcpy(&max_v, writer->max_value, sizeof(max_v));
    if (chunk_min < min_v) min_v = chunk_min;
    if (chunk_max > max_v) max_v = chunk_max;
    if (min_v == 0.0) min_v = -0.0;
    if (max_v == 0.0) max_v = 0.0;
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
}

static void update_statistics_boolean(carquet_page_writer_t* writer,
                                       const uint8_t* values, int64_t count) {
    if (count <= 0) return;
    for (int64_t i = 0; i < count; i++) {
        if (values[i]) writer->bool_seen_true = true;
        else writer->bool_seen_false = true;
        if (writer->bool_seen_true && writer->bool_seen_false) break;
    }
    if (!writer->bool_seen_true && !writer->bool_seen_false) return;

    uint8_t min_b = writer->bool_seen_false ? 0 : 1;
    uint8_t max_b = writer->bool_seen_true ? 1 : 0;
    stats_set_fixed(writer, &min_b, &max_b, 1);
}

/* Lexicographic compare for unsigned byte sequences. */
static int lex_compare(const uint8_t* a, size_t alen,
                       const uint8_t* b, size_t blen) {
    size_t n = alen < blen ? alen : blen;
    int c = memcmp(a, b, n);
    if (c != 0) return c;
    if (alen < blen) return -1;
    if (alen > blen) return 1;
    return 0;
}

static void update_statistics_byte_array(carquet_page_writer_t* writer,
                                          const carquet_byte_array_t* values,
                                          int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        const uint8_t* v = values[i].data;
        size_t vlen = (size_t)values[i].length;
        if (!v) continue;

        if (!writer->has_min_max) {
            if (stats_set_min(writer, v, vlen) != CARQUET_OK) return;
            if (stats_set_max(writer, v, vlen) != CARQUET_OK) return;
            writer->has_min_max = true;
            continue;
        }

        if (lex_compare(v, vlen,
                        writer->min_value, writer->min_value_size) < 0) {
            if (stats_set_min(writer, v, vlen) != CARQUET_OK) return;
        }
        if (lex_compare(v, vlen,
                        writer->max_value, writer->max_value_size) > 0) {
            if (stats_set_max(writer, v, vlen) != CARQUET_OK) return;
        }
    }
}

/* FLOAT16 min/max: ordered by the represented float value with NaNs skipped;
 * a zero min is stored as -0.0 and a zero max as +0.0 (per the spec). The
 * stored bytes are the original little-endian half representation of the
 * achieving value (preserving subnormals), except the normalized zeros. */
static void update_statistics_float16(carquet_page_writer_t* writer,
                                       const uint8_t* values, int64_t count) {
    static const uint8_t NEG_ZERO[2] = { 0x00, 0x80 };
    static const uint8_t POS_ZERO[2] = { 0x00, 0x00 };
    int have = 0;
    float min_f = 0.0f, max_f = 0.0f;
    uint8_t min_b[2] = {0,0}, max_b[2] = {0,0};

    if (writer->has_min_max && writer->min_value_size == 2 &&
        writer->max_value_size == 2) {
        min_b[0] = writer->min_value[0]; min_b[1] = writer->min_value[1];
        max_b[0] = writer->max_value[0]; max_b[1] = writer->max_value[1];
        min_f = carquet_half_to_float((uint16_t)(min_b[0] | (min_b[1] << 8)));
        max_f = carquet_half_to_float((uint16_t)(max_b[0] | (max_b[1] << 8)));
        have = 1;
    }

    for (int64_t i = 0; i < count; i++) {
        const uint8_t* v = values + i * 2;
        float fv = carquet_half_to_float((uint16_t)(v[0] | (v[1] << 8)));
        if (isnan(fv)) continue;
        if (!have) { min_f = max_f = fv; min_b[0]=max_b[0]=v[0];
                     min_b[1]=max_b[1]=v[1]; have = 1; continue; }
        if (fv < min_f) { min_f = fv; min_b[0]=v[0]; min_b[1]=v[1]; }
        if (fv > max_f) { max_f = fv; max_b[0]=v[0]; max_b[1]=v[1]; }
    }
    if (!have) return;

    if (min_f == 0.0f) { min_b[0]=NEG_ZERO[0]; min_b[1]=NEG_ZERO[1]; }
    if (max_f == 0.0f) { max_b[0]=POS_ZERO[0]; max_b[1]=POS_ZERO[1]; }
    if (stats_set_min(writer, min_b, 2) != CARQUET_OK) return;
    if (stats_set_max(writer, max_b, 2) != CARQUET_OK) return;
    writer->has_min_max = true;
    writer->min_max_size = 2;
}

static void update_statistics_flba(carquet_page_writer_t* writer,
                                    const uint8_t* values,
                                    int64_t count,
                                    int32_t type_length) {
    if (type_length <= 0 || count <= 0) return;
    if (writer->logical_type.id == CARQUET_LOGICAL_FLOAT16 && type_length == 2) {
        update_statistics_float16(writer, values, count);
        return;
    }
    size_t tl = (size_t)type_length;
    for (int64_t i = 0; i < count; i++) {
        const uint8_t* v = values + i * tl;
        if (!writer->has_min_max) {
            if (stats_set_min(writer, v, tl) != CARQUET_OK) return;
            if (stats_set_max(writer, v, tl) != CARQUET_OK) return;
            writer->has_min_max = true;
            writer->min_max_size = tl;
            continue;
        }
        if (memcmp(v, writer->min_value, tl) < 0) {
            memcpy(writer->min_value, v, tl);
        }
        if (memcmp(v, writer->max_value, tl) > 0) {
            memcpy(writer->max_value, v, tl);
        }
    }
}

static carquet_status_t encode_plain_i32_with_stats(
    carquet_page_writer_t* writer,
    const int32_t* values,
    int64_t count) {
#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    size_t bytes_needed = (size_t)count * sizeof(int32_t);
    uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
    if (!dest) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    if (logical_integer_is_unsigned(&writer->logical_type)) {
        /* Unsigned ordering: fused dispatch_copy_minmax uses signed compare. */
        memcpy(dest, values, bytes_needed);
        update_statistics_i32(writer, values, count);
        return CARQUET_OK;
    }
    int32_t min_v, max_v;
    carquet_dispatch_copy_minmax_i32(values, count, (int32_t*)dest, &min_v, &max_v);
    if (writer->has_min_max) {
        int32_t cur_min, cur_max;
        memcpy(&cur_min, writer->min_value, sizeof(cur_min));
        memcpy(&cur_max, writer->max_value, sizeof(cur_max));
        if (cur_min < min_v) min_v = cur_min;
        if (cur_max > max_v) max_v = cur_max;
    }
    return stats_set_fixed(writer, &min_v, &max_v, sizeof(min_v));
#else
    carquet_status_t status = carquet_encode_plain_int32(values, count, &writer->values_buffer);
    if (status == CARQUET_OK) {
        update_statistics_i32(writer, values, count);
    }
    return status;
#endif
}

static carquet_status_t encode_plain_i64_with_stats(
    carquet_page_writer_t* writer,
    const int64_t* values,
    int64_t count) {
#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    size_t bytes_needed = (size_t)count * sizeof(int64_t);
    uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
    if (!dest) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    if (logical_integer_is_unsigned(&writer->logical_type)) {
        memcpy(dest, values, bytes_needed);
        update_statistics_i64(writer, values, count);
        return CARQUET_OK;
    }
    int64_t min_v, max_v;
    carquet_dispatch_copy_minmax_i64(values, count, (int64_t*)dest, &min_v, &max_v);
    if (writer->has_min_max) {
        int64_t cur_min, cur_max;
        memcpy(&cur_min, writer->min_value, sizeof(cur_min));
        memcpy(&cur_max, writer->max_value, sizeof(cur_max));
        if (cur_min < min_v) min_v = cur_min;
        if (cur_max > max_v) max_v = cur_max;
    }
    return stats_set_fixed(writer, &min_v, &max_v, sizeof(min_v));
#else
    carquet_status_t status = carquet_encode_plain_int64(values, count, &writer->values_buffer);
    if (status == CARQUET_OK) {
        update_statistics_i64(writer, values, count);
    }
    return status;
#endif
}

static carquet_status_t encode_plain_float_with_stats(
    carquet_page_writer_t* writer,
    const float* values,
    int64_t count) {
#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    size_t bytes_needed = (size_t)count * sizeof(float);
    uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
    if (!dest) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    float min_v, max_v;
    carquet_dispatch_copy_minmax_float(values, count, (float*)dest, &min_v, &max_v);
    if (isnan(min_v) || isnan(max_v)) {
        /* Dispatch propagates NaN; redo stats with NaN-skipping pass. Data
         * has already been copied into the destination buffer. */
        update_statistics_float(writer, values, count);
        return CARQUET_OK;
    }
    if (min_v == 0.0f) min_v = -0.0f;
    if (max_v == 0.0f) max_v = 0.0f;
    if (writer->has_min_max) {
        float cur_min, cur_max;
        memcpy(&cur_min, writer->min_value, sizeof(cur_min));
        memcpy(&cur_max, writer->max_value, sizeof(cur_max));
        if (cur_min < min_v) min_v = cur_min;
        if (cur_max > max_v) max_v = cur_max;
        if (min_v == 0.0f) min_v = -0.0f;
        if (max_v == 0.0f) max_v = 0.0f;
    }
    return stats_set_fixed(writer, &min_v, &max_v, sizeof(min_v));
#else
    carquet_status_t status = carquet_encode_plain_float(values, count, &writer->values_buffer);
    if (status == CARQUET_OK) {
        update_statistics_float(writer, values, count);
    }
    return status;
#endif
}

static carquet_status_t encode_plain_double_with_stats(
    carquet_page_writer_t* writer,
    const double* values,
    int64_t count) {
#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    size_t bytes_needed = (size_t)count * sizeof(double);
    uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
    if (!dest) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    double min_v, max_v;
    carquet_dispatch_copy_minmax_double(values, count, (double*)dest, &min_v, &max_v);
    if (isnan(min_v) || isnan(max_v)) {
        update_statistics_double(writer, values, count);
        return CARQUET_OK;
    }
    if (min_v == 0.0) min_v = -0.0;
    if (max_v == 0.0) max_v = 0.0;
    if (writer->has_min_max) {
        double cur_min, cur_max;
        memcpy(&cur_min, writer->min_value, sizeof(cur_min));
        memcpy(&cur_max, writer->max_value, sizeof(cur_max));
        if (cur_min < min_v) min_v = cur_min;
        if (cur_max > max_v) max_v = cur_max;
        if (min_v == 0.0) min_v = -0.0;
        if (max_v == 0.0) max_v = 0.0;
    }
    return stats_set_fixed(writer, &min_v, &max_v, sizeof(min_v));
#else
    carquet_status_t status = carquet_encode_plain_double(values, count, &writer->values_buffer);
    if (status == CARQUET_OK) {
        update_statistics_double(writer, values, count);
    }
    return status;
#endif
}

static carquet_status_t encode_float_values(
    carquet_page_writer_t* writer,
    const float* values,
    int64_t count) {

    if (count == 0) {
        return CARQUET_OK;
    }

    if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT) {
        size_t bytes_needed = (size_t)count * sizeof(float);
        size_t offset = writer->values_buffer.size;
        uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
        size_t bytes_written = 0;
        if (!dest) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        carquet_status_t status = carquet_byte_stream_split_encode_float(
            values, count, dest, bytes_needed, &bytes_written);
        if (status != CARQUET_OK || bytes_written != bytes_needed) {
            writer->values_buffer.size = offset;
            return status != CARQUET_OK ? status : CARQUET_ERROR_ENCODE;
        }
        return CARQUET_OK;
    }

    return carquet_encode_plain_float(values, count, &writer->values_buffer);
}

static carquet_status_t encode_double_values(
    carquet_page_writer_t* writer,
    const double* values,
    int64_t count) {

    if (count == 0) {
        return CARQUET_OK;
    }

    if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT) {
        size_t bytes_needed = (size_t)count * sizeof(double);
        size_t offset = writer->values_buffer.size;
        uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
        size_t bytes_written = 0;
        if (!dest) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        carquet_status_t status = carquet_byte_stream_split_encode_double(
            values, count, dest, bytes_needed, &bytes_written);
        if (status != CARQUET_OK || bytes_written != bytes_needed) {
            writer->values_buffer.size = offset;
            return status != CARQUET_OK ? status : CARQUET_ERROR_ENCODE;
        }
        return CARQUET_OK;
    }

    return carquet_encode_plain_double(values, count, &writer->values_buffer);
}

/* Encode INT32 values honoring the non-PLAIN data encodings:
 * DELTA_BINARY_PACKED and BYTE_STREAM_SPLIT. PLAIN is handled on the
 * fast path by the caller. */
static carquet_status_t encode_int32_values(
    carquet_page_writer_t* writer, const int32_t* values, int64_t count) {

    /* DELTA_BINARY_PACKED must still emit its 4-varint header for an
     * all-null (zero value) page, otherwise the decoder hits EOF parsing
     * the header. PLAIN/BYTE_STREAM_SPLIT legitimately produce no bytes. */
    if (count == 0 && writer->encoding != CARQUET_ENCODING_DELTA_BINARY_PACKED) {
        return CARQUET_OK;
    }
    size_t offset = writer->values_buffer.size;

    if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT) {
        size_t need = (size_t)count * sizeof(int32_t);
        uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, need);
        if (!dest) return CARQUET_ERROR_OUT_OF_MEMORY;
        size_t written = 0;
        carquet_status_t s = carquet_byte_stream_split_encode(
            (const uint8_t*)values, count, (int32_t)sizeof(int32_t),
            dest, need, &written);
        if (s != CARQUET_OK || written != need) {
            writer->values_buffer.size = offset;
            return s != CARQUET_OK ? s : CARQUET_ERROR_ENCODE;
        }
        return CARQUET_OK;
    }

    if (writer->encoding == CARQUET_ENCODING_DELTA_BINARY_PACKED) {
        /* Delta output never exceeds plain size by more than block/miniblock
         * headers; this bound is comfortably safe. */
        size_t cap = (size_t)count * sizeof(int32_t) + (size_t)count + 512;
        uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, cap);
        if (!dest) return CARQUET_ERROR_OUT_OF_MEMORY;
        size_t written = 0;
        carquet_status_t s = carquet_delta_encode_int32(
            values, (int32_t)count, dest, cap, &written);
        if (s != CARQUET_OK) { writer->values_buffer.size = offset; return s; }
        writer->values_buffer.size = offset + written;
        return CARQUET_OK;
    }

    return carquet_encode_plain_int32(values, count, &writer->values_buffer);
}

static carquet_status_t encode_int64_values(
    carquet_page_writer_t* writer, const int64_t* values, int64_t count) {

    /* DELTA_BINARY_PACKED must still emit its 4-varint header for an
     * all-null (zero value) page, otherwise the decoder hits EOF parsing
     * the header. PLAIN/BYTE_STREAM_SPLIT legitimately produce no bytes. */
    if (count == 0 && writer->encoding != CARQUET_ENCODING_DELTA_BINARY_PACKED) {
        return CARQUET_OK;
    }
    size_t offset = writer->values_buffer.size;

    if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT) {
        size_t need = (size_t)count * sizeof(int64_t);
        uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, need);
        if (!dest) return CARQUET_ERROR_OUT_OF_MEMORY;
        size_t written = 0;
        carquet_status_t s = carquet_byte_stream_split_encode(
            (const uint8_t*)values, count, (int32_t)sizeof(int64_t),
            dest, need, &written);
        if (s != CARQUET_OK || written != need) {
            writer->values_buffer.size = offset;
            return s != CARQUET_OK ? s : CARQUET_ERROR_ENCODE;
        }
        return CARQUET_OK;
    }

    if (writer->encoding == CARQUET_ENCODING_DELTA_BINARY_PACKED) {
        size_t cap = (size_t)count * sizeof(int64_t) + (size_t)count + 512;
        uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, cap);
        if (!dest) return CARQUET_ERROR_OUT_OF_MEMORY;
        size_t written = 0;
        carquet_status_t s = carquet_delta_encode_int64(
            values, (int32_t)count, dest, cap, &written);
        if (s != CARQUET_OK) { writer->values_buffer.size = offset; return s; }
        writer->values_buffer.size = offset + written;
        return CARQUET_OK;
    }

    return carquet_encode_plain_int64(values, count, &writer->values_buffer);
}

/* Encode BYTE_ARRAY values for the delta string encodings. */
static carquet_status_t encode_byte_array_values(
    carquet_page_writer_t* writer,
    const carquet_byte_array_t* values, int64_t count) {

    /* DELTA_LENGTH_BYTE_ARRAY and DELTA_BYTE_ARRAY must still emit their
     * DELTA header(s) for an all-null (zero value) page, otherwise the
     * decoder hits EOF parsing the header. PLAIN produces no bytes. */
    if (count == 0 &&
        writer->encoding != CARQUET_ENCODING_DELTA_LENGTH_BYTE_ARRAY &&
        writer->encoding != CARQUET_ENCODING_DELTA_BYTE_ARRAY) {
        return CARQUET_OK;
    }

    if (writer->encoding == CARQUET_ENCODING_DELTA_LENGTH_BYTE_ARRAY) {
        return carquet_delta_length_encode(values, (int32_t)count,
                                           &writer->values_buffer);
    }
    if (writer->encoding == CARQUET_ENCODING_DELTA_BYTE_ARRAY) {
        return carquet_delta_strings_encode(values, (int32_t)count,
                                            &writer->values_buffer);
    }
    return carquet_encode_plain_byte_array(values, count,
                                           &writer->values_buffer);
}

/* ============================================================================
 * Value Encoding
 * ============================================================================
 */

carquet_status_t carquet_page_writer_add_values(
    carquet_page_writer_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    if (!writer || !values) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_status_t status = CARQUET_OK;
    size_t values_size_before = writer->values_buffer.size;
    size_t def_size_before = writer->def_levels_buffer.size;
    size_t rep_size_before = writer->rep_levels_buffer.size;
    int64_t num_values_before = writer->num_values;
    int64_t num_nulls_before = writer->num_nulls;
    bool has_min_max_before = writer->has_min_max;
    size_t min_max_size_before = writer->min_max_size;
    size_t min_size_before = writer->min_value_size;
    size_t max_size_before = writer->max_value_size;
    bool bool_seen_false_before = writer->bool_seen_false;
    bool bool_seen_true_before = writer->bool_seen_true;

    /* Stack snapshot of the current min/max bytes for rollback. For values
     * exceeding STAT_SNAPSHOT_SZ (rare for primitive types and most strings)
     * we drop the stats on failure instead of preserving them. */
    enum { STAT_SNAPSHOT_SZ = 256 };
    uint8_t min_snapshot[STAT_SNAPSHOT_SZ];
    uint8_t max_snapshot[STAT_SNAPSHOT_SZ];
    bool snapshot_ok = (min_size_before <= sizeof(min_snapshot) &&
                        max_size_before <= sizeof(max_snapshot));
    if (snapshot_ok && has_min_max_before) {
        memcpy(min_snapshot, writer->min_value, min_size_before);
        memcpy(max_snapshot, writer->max_value, max_size_before);
    }

    /* Count nulls and non-null values */
    int64_t num_non_null = num_values;
    if (def_levels && writer->max_def_level > 0) {
        num_non_null = carquet_dispatch_count_non_nulls(def_levels, num_values,
                                                        writer->max_def_level);
        writer->num_nulls += (num_values - num_non_null);
    }

    /* Encode definition levels.
     * If def_levels is NULL for an OPTIONAL column, generate all-present levels
     * since Parquet requires definition levels for non-REQUIRED columns. */
    if (writer->max_def_level > 0) {
        /* Encode raw RLE with no length prefix. A V1 page accumulates the
         * levels of one or more add_values calls; the single 4-byte length
         * prefix that V1 requires is written once at page assembly time
         * (see build_page_payload / finalize). Concatenated raw RLE runs
         * decode as one stream, so multi-chunk pages stay spec-conformant. */
        status = encode_levels(def_levels, num_values, writer->max_def_level,
                               &writer->def_levels_buffer,
                               false);
        if (status != CARQUET_OK) {
            goto fail;
        }
    }

    /* Encode repetition levels */
    if (writer->max_rep_level > 0 && rep_levels) {
        status = encode_levels(rep_levels, num_values, writer->max_rep_level,
                               &writer->rep_levels_buffer,
                               false);
        if (status != CARQUET_OK) {
            goto fail;
        }
    }

    /* Encode values using PLAIN encoding.
     *
     * The values array uses sparse encoding: it contains only non-null values
     * (packed at the front), with num_non_null entries. The def_levels array
     * has num_values entries (one per logical row) indicating which rows are
     * null vs present.
     */
    switch (writer->type) {
        case CARQUET_PHYSICAL_BOOLEAN: {
            const uint8_t* bools = (const uint8_t*)values;
            if (writer->encoding == CARQUET_ENCODING_RLE) {
                /* RLE value encoding for BOOLEAN: a 4-byte little-endian length
                 * prefix followed by the RLE/bit-packed hybrid at bit width 1
                 * (matches parquet-mr's RunLengthBitPackingHybridValuesWriter). */
                uint32_t* tmp = NULL;
                if (num_non_null > 0) {
                    tmp = carquet_mem_malloc((size_t)num_non_null * sizeof(uint32_t));
                    if (!tmp) { status = CARQUET_ERROR_OUT_OF_MEMORY; break; }
                    for (int64_t i = 0; i < num_non_null; i++)
                        tmp[i] = bools[i] ? 1u : 0u;
                }
                carquet_buffer_t rle;
                carquet_buffer_init(&rle);
                status = num_non_null > 0
                    ? carquet_rle_encode_all(tmp, num_non_null, 1, &rle)
                    : CARQUET_OK;
                carquet_mem_free(tmp);
                if (status == CARQUET_OK) {
                    uint32_t rlen = (uint32_t)rle.size;
                    uint8_t len_le[4] = {
                        (uint8_t)rlen, (uint8_t)(rlen >> 8),
                        (uint8_t)(rlen >> 16), (uint8_t)(rlen >> 24) };
                    status = carquet_buffer_append(&writer->values_buffer, len_le, 4);
                    if (status == CARQUET_OK && rle.size > 0) {
                        status = carquet_buffer_append(&writer->values_buffer,
                                                       rle.data, rle.size);
                    }
                }
                carquet_buffer_destroy(&rle);
            } else {
                status = carquet_encode_plain_boolean(bools, num_non_null,
                                                       &writer->values_buffer);
            }
            if (status == CARQUET_OK && writer->write_statistics) {
                update_statistics_boolean(writer, bools, num_non_null);
            }
            break;
        }

        case CARQUET_PHYSICAL_INT32: {
            const int32_t* ints = (const int32_t*)values;
            if (writer->encoding != CARQUET_ENCODING_PLAIN) {
                status = encode_int32_values(writer, ints, num_non_null);
                if (status == CARQUET_OK && writer->write_statistics) {
                    update_statistics_i32(writer, ints, num_non_null);
                }
            } else {
                status = writer->write_statistics
                    ? encode_plain_i32_with_stats(writer, ints, num_non_null)
                    : carquet_encode_plain_int32(ints, num_non_null, &writer->values_buffer);
            }
            break;
        }

        case CARQUET_PHYSICAL_INT64: {
            const int64_t* ints = (const int64_t*)values;
            if (writer->encoding != CARQUET_ENCODING_PLAIN) {
                status = encode_int64_values(writer, ints, num_non_null);
                if (status == CARQUET_OK && writer->write_statistics) {
                    update_statistics_i64(writer, ints, num_non_null);
                }
            } else {
                status = writer->write_statistics
                    ? encode_plain_i64_with_stats(writer, ints, num_non_null)
                    : carquet_encode_plain_int64(ints, num_non_null, &writer->values_buffer);
            }
            break;
        }

        case CARQUET_PHYSICAL_FLOAT: {
            const float* floats = (const float*)values;
            if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT || !writer->write_statistics) {
                status = encode_float_values(writer, floats, num_non_null);
                if (status == CARQUET_OK && writer->write_statistics) {
                    update_statistics_float(writer, floats, num_non_null);
                }
            } else {
                status = encode_plain_float_with_stats(writer, floats, num_non_null);
            }
            break;
        }

        case CARQUET_PHYSICAL_DOUBLE: {
            const double* doubles = (const double*)values;
            if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT || !writer->write_statistics) {
                status = encode_double_values(writer, doubles, num_non_null);
                if (status == CARQUET_OK && writer->write_statistics) {
                    update_statistics_double(writer, doubles, num_non_null);
                }
            } else {
                status = encode_plain_double_with_stats(writer, doubles, num_non_null);
            }
            break;
        }

        case CARQUET_PHYSICAL_BYTE_ARRAY: {
            const carquet_byte_array_t* arrays = (const carquet_byte_array_t*)values;
            status = encode_byte_array_values(writer, arrays, num_non_null);
            if (status == CARQUET_OK) {
                /* Accumulate unencoded (length-prefix-exclusive) value bytes for
                 * OffsetIndex field 2 / SizeStatistics, independent of the
                 * write_statistics flag which only gates min/max. */
                for (int64_t bi = 0; bi < num_non_null; bi++) {
                    writer->byte_array_data_bytes += arrays[bi].length;
                }
            }
            if (status == CARQUET_OK && writer->write_statistics) {
                update_statistics_byte_array(writer, arrays, num_non_null);
            }
            if (status == CARQUET_OK && writer->geo_enabled) {
                /* GEOMETRY/GEOGRAPHY: fold WKB into GeospatialStatistics
                 * (min/max stats are suppressed for these logical types). */
                for (int64_t gi = 0; gi < num_non_null; gi++) {
                    carquet_geo_stats_add_wkb(&writer->geo_stats,
                        arrays[gi].data, (size_t)arrays[gi].length);
                }
            }
            break;
        }

        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: {
            const uint8_t* fixed = (const uint8_t*)values;
            if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT) {
                size_t need = (size_t)num_non_null * (size_t)writer->type_length;
                size_t off = writer->values_buffer.size;
                uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, need);
                if (!dest) {
                    status = CARQUET_ERROR_OUT_OF_MEMORY;
                } else {
                    size_t written = 0;
                    status = carquet_byte_stream_split_encode(
                        fixed, num_non_null, writer->type_length,
                        dest, need, &written);
                    if (status != CARQUET_OK || written != need) {
                        writer->values_buffer.size = off;
                        if (status == CARQUET_OK) status = CARQUET_ERROR_ENCODE;
                    }
                }
            } else if (writer->encoding == CARQUET_ENCODING_DELTA_BYTE_ARRAY) {
                /* Spec allows DELTA_BYTE_ARRAY for FLBA: present each
                 * fixed-width value as a byte array of length type_length. */
                if (num_non_null > 0) {
                    carquet_byte_array_t* tmp = carquet_mem_malloc(
                        (size_t)num_non_null * sizeof(carquet_byte_array_t));
                    if (!tmp) {
                        status = CARQUET_ERROR_OUT_OF_MEMORY;
                    } else {
                        for (int64_t i = 0; i < num_non_null; i++) {
                            tmp[i].data = (uint8_t*)(fixed + i * writer->type_length);
                            tmp[i].length = writer->type_length;
                        }
                        status = carquet_delta_strings_encode(
                            tmp, (int32_t)num_non_null, &writer->values_buffer);
                        carquet_mem_free(tmp);
                    }
                }
            } else {
                status = carquet_encode_plain_fixed_byte_array(fixed, num_non_null,
                                                                writer->type_length,
                                                                &writer->values_buffer);
            }
            if (status == CARQUET_OK && writer->write_statistics) {
                update_statistics_flba(writer, fixed, num_non_null, writer->type_length);
            }
            break;
        }

        case CARQUET_PHYSICAL_INT96: {
            /* INT96 is deprecated and has undefined sort order, so no
             * min/max statistics are produced (matching parquet-cpp). PLAIN
             * is the only valid encoding. */
            const carquet_int96_t* v96 = (const carquet_int96_t*)values;
            status = carquet_encode_plain_int96(v96, num_non_null,
                                                &writer->values_buffer);
            break;
        }

        default:
            status = CARQUET_ERROR_NOT_IMPLEMENTED;
    }

    if (status != CARQUET_OK) {
        goto fail;
    }

    writer->num_values += num_values;
    if (writer->data_page_v2) {
        if (writer->max_rep_level > 0 && rep_levels) {
            for (int64_t i = 0; i < num_values; i++) {
                if (rep_levels[i] == 0) writer->num_rows++;
            }
        } else {
            writer->num_rows += num_values;
        }
    }

    /* Accumulate per-page level histograms (Parquet 2.9). A NULL def_levels on
     * an OPTIONAL column means every value is present (level == max_def_level);
     * a NULL rep_levels means level 0. When max_def/rep_level == 0 the single
     * bucket is index 0, which is exactly what these fall-through paths hit. */
    if (writer->max_def_level > 0 && def_levels) {
        for (int64_t i = 0; i < num_values; i++) {
            int16_t d = def_levels[i];
            if (d >= 0 && d <= writer->max_def_level) writer->def_level_hist[d]++;
        }
    } else {
        writer->def_level_hist[writer->max_def_level] += num_values;
    }
    if (writer->max_rep_level > 0 && rep_levels) {
        for (int64_t i = 0; i < num_values; i++) {
            int16_t r = rep_levels[i];
            if (r >= 0 && r <= writer->max_rep_level) writer->rep_level_hist[r]++;
        }
    } else {
        writer->rep_level_hist[0] += num_values;
    }
    return status;

fail:
    writer->values_buffer.size = values_size_before;
    writer->def_levels_buffer.size = def_size_before;
    writer->rep_levels_buffer.size = rep_size_before;
    writer->num_values = num_values_before;
    writer->num_nulls = num_nulls_before;
    writer->min_max_size = min_max_size_before;
    writer->bool_seen_false = bool_seen_false_before;
    writer->bool_seen_true = bool_seen_true_before;
    if (snapshot_ok) {
        writer->has_min_max = has_min_max_before;
        writer->min_value_size = min_size_before;
        writer->max_value_size = max_size_before;
        if (has_min_max_before) {
            memcpy(writer->min_value, min_snapshot, min_size_before);
            memcpy(writer->max_value, max_snapshot, max_size_before);
        }
    } else {
        writer->has_min_max = false;
        writer->min_value_size = 0;
        writer->max_value_size = 0;
    }
    return status;
}

/* ============================================================================
 * Compression
 * ============================================================================
 */

static carquet_status_t compress_data(
    carquet_compression_t codec,
    const uint8_t* input,
    size_t input_size,
    carquet_buffer_t* temp_buffer,
    const uint8_t** compressed_data,
    size_t* compressed_size,
    int32_t compression_level) {

    if (!compressed_data || !compressed_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (codec == CARQUET_COMPRESSION_UNCOMPRESSED) {
        *compressed_data = input;
        *compressed_size = input_size;
        return CARQUET_OK;
    }

    /* User-registered codec (if any) wins over the built-in. */
    carquet_custom_codec_t custom;
    bool have_custom = carquet_custom_codec_lookup(codec, &custom);

    size_t bound = 0;
    if (have_custom) {
        bound = custom.compress_bound(input_size, custom.user_data);
    } else {
        switch (codec) {
            case CARQUET_COMPRESSION_SNAPPY:
                bound = carquet_snappy_compress_bound(input_size);
                break;
            case CARQUET_COMPRESSION_LZ4:
                bound = carquet_lz4_hadoop_compress_bound(input_size);
                break;
            case CARQUET_COMPRESSION_LZ4_RAW:
                bound = carquet_lz4_compress_bound(input_size);
                break;
            case CARQUET_COMPRESSION_GZIP:
                bound = carquet_gzip_compress_bound(input_size);
                break;
            case CARQUET_COMPRESSION_ZSTD:
                bound = carquet_zstd_compress_bound(input_size);
                break;
            default:
                return CARQUET_ERROR_UNSUPPORTED_CODEC;
        }
    }

    /* Ensure temp buffer is large enough */
    if (temp_buffer->capacity < bound) {
        carquet_status_t reserve_status = carquet_buffer_reserve(temp_buffer, bound);
        if (reserve_status != CARQUET_OK) {
            return reserve_status;
        }
    }
    uint8_t* compressed = temp_buffer->data;

    size_t local_compressed_size = 0;
    carquet_status_t status;

    if (have_custom) {
        status = custom.compress(input, input_size, compressed, bound,
                                 &local_compressed_size, compression_level,
                                 custom.user_data);
    } else {
        switch (codec) {
            case CARQUET_COMPRESSION_SNAPPY:
                status = carquet_snappy_compress(input, input_size,
                                                  compressed, bound, &local_compressed_size);
                break;
            case CARQUET_COMPRESSION_LZ4:
                status = carquet_lz4_hadoop_compress(input, input_size,
                                               compressed, bound, &local_compressed_size);
                break;
            case CARQUET_COMPRESSION_LZ4_RAW:
                status = carquet_lz4_compress(input, input_size,
                                               compressed, bound, &local_compressed_size);
                break;
            case CARQUET_COMPRESSION_GZIP:
                status = carquet_gzip_compress(input, input_size,
                                                compressed, bound, &local_compressed_size,
                                                compression_level > 0 ? compression_level : 6);
                break;
            case CARQUET_COMPRESSION_ZSTD:
                status = carquet_zstd_compress(input, input_size,
                                                compressed, bound, &local_compressed_size,
                                                compression_level > 0 ? compression_level : 3);
                break;
            default:
                status = CARQUET_ERROR_UNSUPPORTED_CODEC;
        }
    }

    if (status != CARQUET_OK) {
        return status;
    }

    temp_buffer->size = local_compressed_size;
    *compressed_data = compressed;
    *compressed_size = local_compressed_size;
    return CARQUET_OK;
}

/* Append a V1 level section to a growable buffer: a 4-byte little-endian
 * length prefix followed by the raw RLE bytes. The page writer stores levels
 * without a prefix (so multiple add_values calls concatenate into one valid
 * RLE stream); this writes the single per-section prefix the V1 page format
 * requires, exactly once, at assembly time. */
static carquet_status_t append_v1_level_section(
    carquet_buffer_t* out, const carquet_buffer_t* lvl) {
    carquet_status_t s = carquet_buffer_append_u32_le(out, (uint32_t)lvl->size);
    if (s != CARQUET_OK) return s;
    return carquet_buffer_append(out, lvl->data, lvl->size);
}

static carquet_status_t build_page_payload(
    carquet_page_writer_t* writer,
    const uint8_t** payload_data,
    size_t* payload_size) {

    /* Each present level section gains a 4-byte length prefix. */
    size_t total_size = writer->rep_levels_buffer.size +
                        writer->def_levels_buffer.size +
                        writer->values_buffer.size +
                        (writer->rep_levels_buffer.size > 0 ? 4 : 0) +
                        (writer->def_levels_buffer.size > 0 ? 4 : 0);

    if (writer->rep_levels_buffer.size == 0 && writer->def_levels_buffer.size == 0) {
        *payload_data = writer->values_buffer.data;
        *payload_size = writer->values_buffer.size;
        return CARQUET_OK;
    }

    carquet_buffer_clear(&writer->staging_buffer);
    carquet_status_t status = carquet_buffer_reserve(&writer->staging_buffer, total_size);
    if (status != CARQUET_OK) {
        return status;
    }

    if (writer->rep_levels_buffer.size > 0) {
        status = append_v1_level_section(&writer->staging_buffer,
                                         &writer->rep_levels_buffer);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    if (writer->def_levels_buffer.size > 0) {
        status = append_v1_level_section(&writer->staging_buffer,
                                         &writer->def_levels_buffer);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    if (writer->values_buffer.size > 0) {
        status = carquet_buffer_append(&writer->staging_buffer,
                                       writer->values_buffer.data,
                                       writer->values_buffer.size);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    *payload_data = writer->staging_buffer.data;
    *payload_size = writer->staging_buffer.size;
    return CARQUET_OK;
}

static uint32_t compute_page_crc(
    const carquet_page_writer_t* writer,
    const uint8_t* payload_data,
    size_t payload_size) {

    if (!writer->write_crc) {
        return 0;
    }

    if (payload_data) {
        return carquet_crc32(payload_data, payload_size);
    }

    uint32_t crc = 0;
    /* Mirror the on-disk layout: each level section is preceded by a 4-byte
     * little-endian length prefix (see append_v1_level_section). */
    if (writer->rep_levels_buffer.size > 0) {
        uint32_t n = (uint32_t)writer->rep_levels_buffer.size;
        uint8_t pfx[4] = { (uint8_t)(n & 0xFF), (uint8_t)((n >> 8) & 0xFF),
                           (uint8_t)((n >> 16) & 0xFF), (uint8_t)((n >> 24) & 0xFF) };
        crc = carquet_crc32_update(crc, pfx, 4);
        crc = carquet_crc32_update(crc,
                                   writer->rep_levels_buffer.data,
                                   writer->rep_levels_buffer.size);
    }
    if (writer->def_levels_buffer.size > 0) {
        uint32_t n = (uint32_t)writer->def_levels_buffer.size;
        uint8_t pfx[4] = { (uint8_t)(n & 0xFF), (uint8_t)((n >> 8) & 0xFF),
                           (uint8_t)((n >> 16) & 0xFF), (uint8_t)((n >> 24) & 0xFF) };
        crc = carquet_crc32_update(crc, pfx, 4);
        crc = carquet_crc32_update(crc,
                                   writer->def_levels_buffer.data,
                                   writer->def_levels_buffer.size);
    }
    if (writer->values_buffer.size > 0) {
        crc = carquet_crc32_update(crc,
                                   writer->values_buffer.data,
                                   writer->values_buffer.size);
    }
    return crc;
}

static carquet_status_t append_data_page_header(
    carquet_buffer_t* output_buffer,
    const carquet_page_writer_t* writer,
    int32_t uncompressed_size,
    int32_t compressed_size,
    uint32_t page_crc) {

    carquet_status_t status = carquet_buffer_reserve(
        output_buffer, output_buffer->size + 128);
    if (status != CARQUET_OK) {
        return status;
    }

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, output_buffer);

    thrift_write_struct_begin(&enc);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, CARQUET_PAGE_DATA);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, uncompressed_size);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
    thrift_write_i32(&enc, compressed_size);

    if (writer->write_crc) {
        thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
        thrift_write_i32(&enc, (int32_t)page_crc);
    }

    thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 5);
    thrift_write_struct_begin(&enc);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, (int32_t)writer->num_values);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, (int32_t)writer->encoding);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
    thrift_write_i32(&enc, CARQUET_ENCODING_RLE);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
    thrift_write_i32(&enc, CARQUET_ENCODING_RLE);

    if (writer->write_statistics && writer->has_min_max) {
        thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 5);
        thrift_write_struct_begin(&enc);
        thrift_write_field_header(&enc, THRIFT_TYPE_I64, 3);
        thrift_write_i64(&enc, writer->num_nulls);
        thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 5);
        thrift_write_binary(&enc, writer->max_value, (int32_t)writer->max_value_size);
        thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 6);
        thrift_write_binary(&enc, writer->min_value, (int32_t)writer->min_value_size);
        thrift_write_struct_end(&enc);
    }

    thrift_write_struct_end(&enc);
    thrift_write_struct_end(&enc);
    return enc.status;
}

/* PageHeader carrying a DataPageHeaderV2 (PageType=DATA_PAGE_V2, field 8).
 * Levels are stored uncompressed ahead of the (optionally compressed) value
 * region; their byte lengths are carried in the header instead of an inline
 * 4-byte prefix. */
static carquet_status_t append_data_page_header_v2(
    carquet_buffer_t* output_buffer,
    const carquet_page_writer_t* writer,
    int32_t uncompressed_size,
    int32_t compressed_size,
    uint32_t page_crc,
    int32_t def_levels_len,
    int32_t rep_levels_len,
    bool is_compressed) {

    carquet_status_t status = carquet_buffer_reserve(
        output_buffer, output_buffer->size + 128);
    if (status != CARQUET_OK) {
        return status;
    }

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, output_buffer);

    thrift_write_struct_begin(&enc);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, CARQUET_PAGE_DATA_V2);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, uncompressed_size);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
    thrift_write_i32(&enc, compressed_size);

    if (writer->write_crc) {
        thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
        thrift_write_i32(&enc, (int32_t)page_crc);
    }

    /* Field 8: DataPageHeaderV2 */
    thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 8);
    thrift_write_struct_begin(&enc);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, (int32_t)writer->num_values);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, (int32_t)writer->num_nulls);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
    thrift_write_i32(&enc, (int32_t)writer->num_rows);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
    thrift_write_i32(&enc, (int32_t)writer->encoding);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 5);
    thrift_write_i32(&enc, def_levels_len);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 6);
    thrift_write_i32(&enc, rep_levels_len);
    /* Field 7: is_compressed (bool encoded in the field-header type). */
    thrift_write_field_header(&enc,
        is_compressed ? THRIFT_TYPE_TRUE : THRIFT_TYPE_FALSE, 7);

    if (writer->write_statistics && writer->has_min_max) {
        thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 8);
        thrift_write_struct_begin(&enc);
        thrift_write_field_header(&enc, THRIFT_TYPE_I64, 3);
        thrift_write_i64(&enc, writer->num_nulls);
        thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 5);
        thrift_write_binary(&enc, writer->max_value, (int32_t)writer->max_value_size);
        thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 6);
        thrift_write_binary(&enc, writer->min_value, (int32_t)writer->min_value_size);
        thrift_write_struct_end(&enc);
    }

    thrift_write_struct_end(&enc);
    thrift_write_struct_end(&enc);
    return enc.status;
}

/* Finalize a DATA_PAGE_V2: [rep levels][def levels][maybe-compressed values],
 * levels always uncompressed. */
static carquet_status_t finalize_v2_to_buffer(
    carquet_page_writer_t* writer,
    carquet_buffer_t* output_buffer,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size) {

    size_t rep_len = writer->rep_levels_buffer.size;
    size_t def_len = writer->def_levels_buffer.size;
    size_t levels_len = rep_len + def_len;
    size_t page_start = output_buffer->size;
    carquet_status_t status;

    const uint8_t* value_data = writer->values_buffer.data;
    size_t value_size = writer->values_buffer.size;
    bool is_compressed = (writer->compression != CARQUET_COMPRESSION_UNCOMPRESSED);

    const uint8_t* out_values = value_data;
    size_t out_values_size = value_size;
    if (is_compressed && value_size > 0) {
        status = compress_data(writer->compression, value_data, value_size,
                               &writer->compress_buffer, &out_values,
                               &out_values_size, writer->compression_level);
        if (status != CARQUET_OK) {
            return status;
        }
    } else {
        is_compressed = false;
    }

    *uncompressed_size = (int32_t)(levels_len + value_size);
    *compressed_size = (int32_t)(levels_len + out_values_size);

    uint32_t crc = 0;
    if (writer->write_crc) {
        crc = carquet_crc32_update(crc, writer->rep_levels_buffer.data, rep_len);
        crc = carquet_crc32_update(crc, writer->def_levels_buffer.data, def_len);
        crc = carquet_crc32_update(crc, out_values, out_values_size);
    }

    status = carquet_buffer_reserve(output_buffer,
        output_buffer->size + 128 + levels_len + out_values_size);
    if (status != CARQUET_OK) { output_buffer->size = page_start; return status; }

    status = append_data_page_header_v2(output_buffer, writer,
        *uncompressed_size, *compressed_size, crc,
        (int32_t)def_len, (int32_t)rep_len, is_compressed);
    if (status != CARQUET_OK) { output_buffer->size = page_start; return status; }

    if (rep_len > 0) {
        status = carquet_buffer_append(output_buffer,
            writer->rep_levels_buffer.data, rep_len);
        if (status != CARQUET_OK) { output_buffer->size = page_start; return status; }
    }
    if (def_len > 0) {
        status = carquet_buffer_append(output_buffer,
            writer->def_levels_buffer.data, def_len);
        if (status != CARQUET_OK) { output_buffer->size = page_start; return status; }
    }
    if (out_values_size > 0) {
        status = carquet_buffer_append(output_buffer, out_values, out_values_size);
        if (status != CARQUET_OK) { output_buffer->size = page_start; return status; }
    }

    *page_size = output_buffer->size - page_start;
    return CARQUET_OK;
}

/* ============================================================================
 * Page Finalization
 * ============================================================================
 */

carquet_status_t carquet_page_writer_finalize(
    carquet_page_writer_t* writer,
    const uint8_t** page_data,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size) {

    if (!writer || !page_data || !page_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_buffer_clear(&writer->page_buffer);
    carquet_status_t status = carquet_page_writer_finalize_to_buffer(
        writer, &writer->page_buffer, page_size, uncompressed_size, compressed_size);
    if (status != CARQUET_OK) {
        return status;
    }

    *page_data = writer->page_buffer.data;
    return CARQUET_OK;
}

carquet_status_t carquet_page_writer_finalize_to_buffer(
    carquet_page_writer_t* writer,
    carquet_buffer_t* output_buffer,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size) {

    if (!writer || !output_buffer || !page_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (writer->data_page_v2) {
        return finalize_v2_to_buffer(writer, output_buffer, page_size,
                                     uncompressed_size, compressed_size);
    }

    bool has_levels = (writer->rep_levels_buffer.size > 0 ||
                       writer->def_levels_buffer.size > 0);
    const uint8_t* payload_data = NULL;
    size_t payload_size = 0;
    const uint8_t* compressed_data = NULL;
    size_t compressed_data_size = 0;
    carquet_status_t status;

    size_t page_start = output_buffer->size;

    if (writer->compression == CARQUET_COMPRESSION_UNCOMPRESSED) {
        *uncompressed_size = (int32_t)(writer->rep_levels_buffer.size +
                                       writer->def_levels_buffer.size +
                                       writer->values_buffer.size +
                                       (writer->rep_levels_buffer.size > 0 ? 4 : 0) +
                                       (writer->def_levels_buffer.size > 0 ? 4 : 0));
        *compressed_size = *uncompressed_size;

        uint32_t page_crc = compute_page_crc(writer, NULL, 0);
        status = carquet_buffer_reserve(output_buffer, output_buffer->size + 128 +
                                        (size_t)*compressed_size);
        if (status != CARQUET_OK) {
            output_buffer->size = page_start;
            return status;
        }

        status = append_data_page_header(output_buffer, writer,
                                         *uncompressed_size, *compressed_size, page_crc);
        if (status != CARQUET_OK) {
            output_buffer->size = page_start;
            return status;
        }

        if (has_levels) {
            if (writer->rep_levels_buffer.size > 0) {
                status = append_v1_level_section(output_buffer,
                                                 &writer->rep_levels_buffer);
                if (status != CARQUET_OK) {
                    output_buffer->size = page_start;
                    return status;
                }
            }
            if (writer->def_levels_buffer.size > 0) {
                status = append_v1_level_section(output_buffer,
                                                 &writer->def_levels_buffer);
                if (status != CARQUET_OK) {
                    output_buffer->size = page_start;
                    return status;
                }
            }
        }

        status = carquet_buffer_append(output_buffer,
                                       writer->values_buffer.data,
                                       writer->values_buffer.size);
        if (status != CARQUET_OK) {
            output_buffer->size = page_start;
            return status;
        }

        *page_size = output_buffer->size - page_start;
        return CARQUET_OK;
    }

    status = build_page_payload(writer, &payload_data, &payload_size);
    if (status != CARQUET_OK) {
        return status;
    }

    *uncompressed_size = (int32_t)payload_size;
    status = compress_data(writer->compression,
                           payload_data, payload_size,
                           &writer->compress_buffer,
                           &compressed_data,
                           &compressed_data_size,
                           writer->compression_level);
    if (status != CARQUET_OK) {
        return status;
    }

    *compressed_size = (int32_t)compressed_data_size;
    status = carquet_buffer_reserve(output_buffer, output_buffer->size + 128 +
                                    compressed_data_size);
    if (status != CARQUET_OK) {
        output_buffer->size = page_start;
        return status;
    }

    status = append_data_page_header(output_buffer, writer,
                                     *uncompressed_size, *compressed_size,
                                     compute_page_crc(writer, compressed_data, compressed_data_size));
    if (status != CARQUET_OK) {
        output_buffer->size = page_start;
        return status;
    }

    status = carquet_buffer_append(output_buffer, compressed_data, compressed_data_size);
    if (status != CARQUET_OK) {
        output_buffer->size = page_start;
        return status;
    }

    *page_size = output_buffer->size - page_start;
    return CARQUET_OK;
}

/* ============================================================================
 * Dictionary Page Emission
 * ============================================================================
 *
 * Builds a spec-conformant DICTIONARY_PAGE (PageType=2). The payload is the
 * already-PLAIN-encoded dictionary entries; it is compressed with the column
 * codec exactly like a data page. The PageHeader carries DictionaryPageHeader
 * at field 7 (1: num_values, 2: encoding=PLAIN, 3: is_sorted=false).
 */
carquet_status_t carquet_page_writer_emit_dictionary_page(
    carquet_page_writer_t* writer,
    carquet_buffer_t* output_buffer,
    const uint8_t* plain_payload,
    size_t payload_size,
    int32_t num_entries,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size) {

    if (!writer || !output_buffer || !plain_payload || !page_size ||
        !uncompressed_size || !compressed_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    const uint8_t* compressed_data = NULL;
    size_t compressed_data_size = 0;
    carquet_status_t status = compress_data(writer->compression,
                                            plain_payload, payload_size,
                                            &writer->compress_buffer,
                                            &compressed_data,
                                            &compressed_data_size,
                                            writer->compression_level);
    if (status != CARQUET_OK) {
        return status;
    }

    *uncompressed_size = (int32_t)payload_size;
    *compressed_size = (int32_t)compressed_data_size;

    size_t page_start = output_buffer->size;
    status = carquet_buffer_reserve(output_buffer,
                                    output_buffer->size + 128 + compressed_data_size);
    if (status != CARQUET_OK) {
        return status;
    }

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, output_buffer);

    thrift_write_struct_begin(&enc);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, CARQUET_PAGE_DICTIONARY);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, *uncompressed_size);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
    thrift_write_i32(&enc, *compressed_size);

    if (writer->write_crc) {
        thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
        thrift_write_i32(&enc, (int32_t)carquet_crc32(compressed_data,
                                                      compressed_data_size));
    }

    /* Field 7: DictionaryPageHeader */
    thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 7);
    thrift_write_struct_begin(&enc);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, num_entries);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, CARQUET_ENCODING_PLAIN);
    /* is_sorted=false: thrift compact encodes booleans in the field-header
     * type itself (TRUE=1, FALSE=2) with no separate value. */
    thrift_write_field_header(&enc, THRIFT_TYPE_FALSE, 3);
    thrift_write_struct_end(&enc);

    thrift_write_struct_end(&enc);

    if (enc.status != CARQUET_OK) {
        output_buffer->size = page_start;
        return enc.status;
    }

    status = carquet_buffer_append(output_buffer, compressed_data, compressed_data_size);
    if (status != CARQUET_OK) {
        output_buffer->size = page_start;
        return status;
    }

    *page_size = output_buffer->size - page_start;
    return CARQUET_OK;
}

/* Stage a RLE_DICTIONARY data page: encode the def/rep levels exactly as the
 * PLAIN path does, then set the values buffer verbatim to the pre-built
 * [bit-width][RLE indices] payload. The caller subsequently sets the page
 * encoding (RLE_DICTIONARY) and any min/max stats, then calls
 * carquet_page_writer_finalize_to_buffer. */
carquet_status_t carquet_page_writer_add_dictionary_indices(
    carquet_page_writer_t* writer,
    const uint8_t* idx_payload,
    size_t idx_size,
    const int16_t* def_levels,
    const int16_t* rep_levels,
    int64_t num_values_total,
    int64_t num_nulls) {

    if (!writer || !idx_payload) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_status_t status = CARQUET_OK;
    if (writer->max_def_level > 0) {
        /* Raw RLE; the V1 length prefix is applied once at page assembly. */
        status = encode_levels(def_levels, num_values_total, writer->max_def_level,
                               &writer->def_levels_buffer,
                               false);
        if (status != CARQUET_OK) return status;
    }
    if (writer->max_rep_level > 0 && rep_levels) {
        status = encode_levels(rep_levels, num_values_total, writer->max_rep_level,
                               &writer->rep_levels_buffer,
                               false);
        if (status != CARQUET_OK) return status;
    }

    status = carquet_buffer_append(&writer->values_buffer, idx_payload, idx_size);
    if (status != CARQUET_OK) return status;

    writer->num_values = num_values_total;
    writer->num_nulls = num_nulls;
    if (writer->data_page_v2) {
        if (writer->max_rep_level > 0 && rep_levels) {
            for (int64_t i = 0; i < num_values_total; i++) {
                if (rep_levels[i] == 0) writer->num_rows++;
            }
        } else {
            writer->num_rows += num_values_total;
        }
    }

    /* Per-page level histograms for the single RLE_DICTIONARY data page
     * (see the matching logic in carquet_page_writer_add_values). */
    if (writer->max_def_level > 0 && def_levels) {
        for (int64_t i = 0; i < num_values_total; i++) {
            int16_t d = def_levels[i];
            if (d >= 0 && d <= writer->max_def_level) writer->def_level_hist[d]++;
        }
    } else {
        writer->def_level_hist[writer->max_def_level] += num_values_total;
    }
    if (writer->max_rep_level > 0 && rep_levels) {
        for (int64_t i = 0; i < num_values_total; i++) {
            int16_t r = rep_levels[i];
            if (r >= 0 && r <= writer->max_rep_level) writer->rep_level_hist[r]++;
        }
    } else {
        writer->rep_level_hist[0] += num_values_total;
    }
    return CARQUET_OK;
}

void carquet_page_writer_set_encoding(carquet_page_writer_t* writer,
                                      carquet_encoding_t encoding) {
    if (writer) writer->encoding = encoding;
}

/* Inject column-level min/max stats into the page writer so the
 * RLE_DICTIONARY data page header carries the same statistics the PLAIN
 * path would have produced from the raw values. */
carquet_status_t carquet_page_writer_set_min_max(
    carquet_page_writer_t* writer,
    const uint8_t* min_value, size_t min_size,
    const uint8_t* max_value, size_t max_size) {
    if (!writer) return CARQUET_ERROR_INVALID_ARGUMENT;
    if (!writer->write_statistics || !min_value || !max_value ||
        min_size == 0 || max_size == 0) {
        return CARQUET_OK;
    }
    carquet_status_t s = stats_set_min(writer, min_value, min_size);
    if (s != CARQUET_OK) return s;
    s = stats_set_max(writer, max_value, max_size);
    if (s != CARQUET_OK) return s;
    writer->min_max_size = min_size;
    writer->has_min_max = true;
    return CARQUET_OK;
}

void carquet_page_writer_set_data_page_v2(carquet_page_writer_t* writer,
                                          bool enabled) {
    if (writer) writer->data_page_v2 = enabled;
}

const parquet_geospatial_statistics_t* carquet_page_writer_get_geo_stats(
    const carquet_page_writer_t* writer) {
    if (!writer || !writer->geo_enabled) return NULL;
    return &writer->geo_stats;
}

size_t carquet_page_writer_estimated_size(const carquet_page_writer_t* writer) {
    if (!writer) return 0;
    return writer->values_buffer.size +
           writer->def_levels_buffer.size +
           writer->rep_levels_buffer.size + 64;  /* Header overhead */
}

int64_t carquet_page_writer_num_values(const carquet_page_writer_t* writer) {
    return writer ? writer->num_values : 0;
}

/* Unencoded BYTE_ARRAY value bytes accumulated for the current (not-yet-flushed)
 * page. Meaningful only for BYTE_ARRAY columns; 0 otherwise. */
int64_t carquet_page_writer_byte_array_bytes(const carquet_page_writer_t* writer) {
    return writer ? writer->byte_array_data_bytes : 0;
}

/* Per-page level histograms for the current (not-yet-flushed) page. The
 * returned pointers are owned by the page writer and valid until the next
 * reset; lengths are max_def_level+1 / max_rep_level+1. */
const int64_t* carquet_page_writer_def_level_histogram(
    const carquet_page_writer_t* writer, int32_t* len) {
    if (!writer) { if (len) *len = 0; return NULL; }
    if (len) *len = (int32_t)writer->max_def_level + 1;
    return writer->def_level_hist;
}

const int64_t* carquet_page_writer_rep_level_histogram(
    const carquet_page_writer_t* writer, int32_t* len) {
    if (!writer) { if (len) *len = 0; return NULL; }
    if (len) *len = (int32_t)writer->max_rep_level + 1;
    return writer->rep_level_hist;
}

/* Override the accumulated BYTE_ARRAY byte count. The dictionary path stages a
 * data page from RLE indices rather than raw values (so the accumulator never
 * sees the byte arrays), and injects the chunk total computed from the unique
 * dictionary values here before the page is flushed. */
void carquet_page_writer_set_byte_array_bytes(carquet_page_writer_t* writer,
                                              int64_t bytes) {
    if (writer) writer->byte_array_data_bytes = bytes;
}

/* ============================================================================
 * Options Configuration
 * ============================================================================
 */

void carquet_page_writer_set_crc(carquet_page_writer_t* writer, bool enabled) {
    if (writer) {
        writer->write_crc = enabled;
    }
}

void carquet_page_writer_set_statistics(carquet_page_writer_t* writer, bool enabled) {
    if (writer) {
        writer->write_statistics = enabled &&
            stats_order_defined_for_logical(&writer->logical_type);
    }
}

/* ============================================================================
 * Statistics Retrieval (for column-level aggregation)
 * ============================================================================
 */

bool carquet_page_writer_get_statistics(
    const carquet_page_writer_t* writer,
    const uint8_t** min_value, size_t* min_size,
    const uint8_t** max_value, size_t* max_size,
    int64_t* null_count) {

    if (!writer || !writer->has_min_max) {
        return false;
    }
    if (min_value) *min_value = writer->min_value;
    if (max_value) *max_value = writer->max_value;
    if (min_size) *min_size = writer->min_value_size;
    if (max_size) *max_size = writer->max_value_size;
    if (null_count) *null_count = writer->num_nulls;
    return true;
}

int64_t carquet_page_writer_null_count(const carquet_page_writer_t* writer) {
    return writer ? writer->num_nulls : 0;
}
