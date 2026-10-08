/**
 * @file row_group_writer.c
 * @brief Row group writing implementation
 *
 * Manages writing multiple columns to form a row group,
 * tracking row counts and generating row group metadata.
 */

#include "core/allocator.h"
#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/buffer.h"
#include "core/compat.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include <stdlib.h>
#include <string.h>
#include "page_tasks.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* Forward declaration from column_writer.c */
typedef struct carquet_column_writer_internal carquet_column_writer_internal_t;

extern carquet_column_writer_internal_t* carquet_column_writer_create(
    carquet_physical_type_t type,
    const carquet_logical_type_t* logical_type,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    size_t target_page_size,
    int32_t compression_level);

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
extern bool carquet_column_writer_has_dictionary_page(
    const carquet_column_writer_internal_t* writer);
extern int64_t carquet_column_writer_dictionary_page_size(
    const carquet_column_writer_internal_t* writer);
extern void carquet_column_writer_set_dictionary_page_size_limit(
    carquet_column_writer_internal_t* writer, int64_t limit);

extern void carquet_column_writer_enable_bloom_filter(
    carquet_column_writer_internal_t* writer, int64_t ndv);
extern void carquet_column_writer_configure_bloom_filter(
    carquet_column_writer_internal_t* writer,
    bool enabled, int64_t ndv, double fpp);
extern void carquet_column_writer_set_max_rows_per_page(
    carquet_column_writer_internal_t* writer, int64_t max_rows);
extern void carquet_column_writer_set_write_batch_size(
    carquet_column_writer_internal_t* writer, int64_t batch_size);
extern void carquet_column_writer_set_target_page_size(
    carquet_column_writer_internal_t* writer, int64_t bytes);
extern void carquet_column_writer_enable_page_index(
    carquet_column_writer_internal_t* writer);
extern void carquet_column_writer_set_file_offset(
    carquet_column_writer_internal_t* writer, int64_t offset);
extern void carquet_column_writer_set_statistics(
    carquet_column_writer_internal_t* writer, bool enabled);
extern const parquet_geospatial_statistics_t* carquet_column_writer_get_geo_stats(
    const carquet_column_writer_internal_t* writer);
extern bool carquet_column_writer_get_statistics(
    const carquet_column_writer_internal_t* writer,
    const uint8_t** min_value, size_t* min_size,
    const uint8_t** max_value, size_t* max_size,
    int64_t* null_count);
extern bool carquet_column_writer_get_distinct_count(
    const carquet_column_writer_internal_t* writer, int64_t* count);
extern void carquet_column_writer_get_size_statistics(
    const carquet_column_writer_internal_t* writer,
    int64_t* unencoded_byte_array_bytes,
    const int64_t** rep_level_hist, int32_t* rep_len,
    const int64_t** def_level_hist, int32_t* def_len);
extern void carquet_column_writer_set_crc(
    carquet_column_writer_internal_t* writer, bool enabled);
extern void carquet_column_writer_set_data_page_v2(
    carquet_column_writer_internal_t* writer, bool enabled);
extern void carquet_column_writer_set_defer_encode(
    carquet_column_writer_internal_t* writer, bool enabled);
extern void carquet_column_writer_reset(
    carquet_column_writer_internal_t* writer);

/* Bloom filter and page index accessors */
typedef struct carquet_bloom_filter carquet_bloom_filter_t;
typedef struct carquet_column_index_builder carquet_column_index_builder_t;
typedef struct carquet_offset_index_builder carquet_offset_index_builder_t;

extern void carquet_page_writer_destroy(carquet_page_writer_t* writer);
extern void carquet_column_writer_detach_buffer(
    carquet_column_writer_internal_t* writer,
    carquet_buffer_t* out,
    carquet_buffer_t* spare);
extern carquet_bloom_filter_t* carquet_column_writer_get_bloom_filter(
    const carquet_column_writer_internal_t* writer);
extern carquet_column_index_builder_t* carquet_column_writer_get_column_index(
    const carquet_column_writer_internal_t* writer);
extern carquet_offset_index_builder_t* carquet_column_writer_get_offset_index(
    const carquet_column_writer_internal_t* writer);

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
    carquet_logical_type_t logical_type;
    carquet_encoding_t encoding;
    carquet_compression_t compression;
    int32_t type_length;
    char* path;
    /* Aggregated column statistics (populated on finalize when stats enabled).
     * Min and max are owned (malloc'd) and may have different sizes for
     * variable-length BYTE_ARRAY columns. */
    bool has_min_max;
    uint8_t* min_value;
    size_t min_value_size;
    uint8_t* max_value;
    size_t max_value_size;
    int64_t null_count;
    bool has_null_count;
    /* Dictionary page plumbing. When has_dictionary_page is set the chunk
     * starts with a DICTIONARY_PAGE of dictionary_page_size bytes at
     * file_offset; the first data page follows it. */
    bool has_dictionary_page;
    int64_t dictionary_page_size;
    /* GeospatialStatistics (GEOMETRY/GEOGRAPHY); cumulative over the chunk. */
    bool has_geo_stats;
    parquet_geospatial_statistics_t geo_stats;
    /* Exact distinct non-null count (dictionary-encoded chunks only). */
    bool has_distinct_count;
    int64_t distinct_count;
    /* SizeStatistics (Parquet 2.9). Histogram pointers alias the column
     * writer's buffers and stay valid until it is reset/destroyed; the file
     * writer copies them out immediately after finalize. unencoded_ba_bytes is
     * -1 for non-BYTE_ARRAY columns. */
    int64_t unencoded_ba_bytes;
    const int64_t* rep_level_hist;
    int32_t rep_hist_len;
    const int64_t* def_level_hist;
    int32_t def_hist_len;
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

    /* Optional features */
    bool write_bloom_filters;
    bool write_page_index;
    bool write_statistics;
    bool write_crc;
    int32_t compression_level;
    int64_t dictionary_page_size;  /* 0 = column default (1MB) */

    /* Page-granular parallel finalize scratch (see page_tasks.h). Grown on
     * demand and kept across row groups so the per-page buffers stay warm. */
    carquet_page_writer_t** scratch_writers;  /* one per OpenMP thread */
    int32_t num_scratch;
    carquet_page_task_t* page_tasks;
    carquet_buffer_t* task_bufs;
    int32_t page_task_capacity;
    int32_t* col_task_counts;    /* [num_columns] pages per column, 0 = serial */
    int32_t* col_task_offsets;   /* [num_columns] first task index */
} carquet_row_group_writer_t;

typedef struct finalized_column_chunk {
    const uint8_t* data;
    size_t size;
    int64_t total_values;
    int64_t compressed_size;
    int64_t uncompressed_size;
    carquet_status_t status;
} finalized_column_chunk_t;

static void capture_column_statistics(carquet_row_group_writer_t* writer, int i) {
    column_chunk_info_t* info = &writer->column_infos[i];

    /* GeospatialStatistics are independent of min/max statistics and of the
     * write_statistics flag. The page writer accumulates them cumulatively
     * over the whole chunk, so a plain copy of the latest snapshot is the
     * complete chunk-level value. */
    const parquet_geospatial_statistics_t* g =
        carquet_column_writer_get_geo_stats(writer->column_writers[i]);
    if (g) {
        info->geo_stats = *g;
        info->has_geo_stats = true;
    }

    if (!writer->write_statistics) return;

    const uint8_t* min_v = NULL;
    const uint8_t* max_v = NULL;
    size_t min_size = 0;
    size_t max_size = 0;
    int64_t null_count = 0;

    bool has_min_max = carquet_column_writer_get_statistics(
        writer->column_writers[i], &min_v, &min_size, &max_v, &max_size,
        &null_count);

    info->has_null_count = true;
    info->null_count = null_count;

    /* Free any stats left over from a previous row group on this writer. */
    carquet_mem_free(info->min_value);
    carquet_mem_free(info->max_value);
    info->min_value = NULL;
    info->max_value = NULL;
    info->min_value_size = 0;
    info->max_value_size = 0;
    info->has_min_max = false;

    if (has_min_max && min_size > 0 && max_size > 0) {
        info->min_value = carquet_mem_malloc(min_size);
        info->max_value = carquet_mem_malloc(max_size);
        if (info->min_value && info->max_value) {
            memcpy(info->min_value, min_v, min_size);
            memcpy(info->max_value, max_v, max_size);
            info->min_value_size = min_size;
            info->max_value_size = max_size;
            info->has_min_max = true;
        } else {
            carquet_mem_free(info->min_value);
            carquet_mem_free(info->max_value);
            info->min_value = NULL;
            info->max_value = NULL;
        }
    }
}

static void capture_dictionary_info(carquet_row_group_writer_t* writer, int i) {
    column_chunk_info_t* info = &writer->column_infos[i];
    info->has_dictionary_page =
        carquet_column_writer_has_dictionary_page(writer->column_writers[i]);
    info->dictionary_page_size =
        carquet_column_writer_dictionary_page_size(writer->column_writers[i]);
    info->has_distinct_count = carquet_column_writer_get_distinct_count(
        writer->column_writers[i], &info->distinct_count);
    carquet_column_writer_get_size_statistics(
        writer->column_writers[i], &info->unencoded_ba_bytes,
        &info->rep_level_hist, &info->rep_hist_len,
        &info->def_level_hist, &info->def_hist_len);
}

static bool can_parallel_finalize(const carquet_row_group_writer_t* writer) {
#ifdef _OPENMP
    return writer && writer->num_columns > 1 && !writer->write_page_index;
#else
    (void)writer;
    return false;
#endif
}

static void release_parallel_scratch(carquet_row_group_writer_t* writer) {
    if (writer->scratch_writers) {
        for (int32_t i = 0; i < writer->num_scratch; i++) {
            carquet_page_writer_destroy(writer->scratch_writers[i]);
        }
        carquet_mem_free(writer->scratch_writers);
        writer->scratch_writers = NULL;
        writer->num_scratch = 0;
    }
    if (writer->page_tasks) {
        for (int32_t i = 0; i < writer->page_task_capacity; i++) {
            carquet_page_task_release(&writer->page_tasks[i]);
        }
        carquet_mem_free(writer->page_tasks);
        writer->page_tasks = NULL;
    }
    if (writer->task_bufs) {
        for (int32_t i = 0; i < writer->page_task_capacity; i++) {
            carquet_buffer_destroy(&writer->task_bufs[i]);
        }
        carquet_mem_free(writer->task_bufs);
        writer->task_bufs = NULL;
    }
    writer->page_task_capacity = 0;
    carquet_mem_free(writer->col_task_counts);
    carquet_mem_free(writer->col_task_offsets);
    writer->col_task_counts = NULL;
    writer->col_task_offsets = NULL;
}

#ifdef _OPENMP
/* Make sure there are at least `count` page tasks and output buffers. */
static carquet_status_t ensure_page_task_capacity(
    carquet_row_group_writer_t* writer, int32_t count) {
    if (count <= writer->page_task_capacity) return CARQUET_OK;
    int32_t new_cap = writer->page_task_capacity ? writer->page_task_capacity : 16;
    while (new_cap < count) {
        if (new_cap > INT32_MAX / 2) { new_cap = count; break; }
        new_cap *= 2;
    }
    carquet_page_task_t* tasks = carquet_mem_realloc(
        writer->page_tasks, (size_t)new_cap * sizeof(*tasks));
    if (!tasks) return CARQUET_ERROR_OUT_OF_MEMORY;
    writer->page_tasks = tasks;
    carquet_buffer_t* bufs = carquet_mem_realloc(
        writer->task_bufs, (size_t)new_cap * sizeof(*bufs));
    if (!bufs) return CARQUET_ERROR_OUT_OF_MEMORY;
    writer->task_bufs = bufs;
    for (int32_t i = writer->page_task_capacity; i < new_cap; i++) {
        memset(&writer->page_tasks[i], 0, sizeof(writer->page_tasks[i]));
        carquet_buffer_init(&writer->task_bufs[i]);
    }
    writer->page_task_capacity = new_cap;
    return CARQUET_OK;
}

static carquet_status_t ensure_scratch_writers(
    carquet_row_group_writer_t* writer, int32_t count) {
    if (count <= writer->num_scratch) return CARQUET_OK;
    carquet_page_writer_t** arr = carquet_mem_realloc(
        writer->scratch_writers, (size_t)count * sizeof(*arr));
    if (!arr) return CARQUET_ERROR_OUT_OF_MEMORY;
    writer->scratch_writers = arr;
    for (int32_t i = writer->num_scratch; i < count; i++) {
        arr[i] = carquet_page_writer_create_scratch();
        if (!arr[i]) return CARQUET_ERROR_OUT_OF_MEMORY;
        writer->num_scratch = i + 1;
    }
    return CARQUET_OK;
}

static void finalize_whole_column(carquet_row_group_writer_t* writer, int i,
                                  finalized_column_chunk_t* chunk) {
    chunk->status = carquet_column_writer_finalize(
        writer->column_writers[i],
        &chunk->data, &chunk->size,
        &chunk->total_values,
        &chunk->compressed_size,
        &chunk->uncompressed_size);
}
#endif

/*
 * Finalize every column of the row group concurrently.
 *
 * Work is scheduled at page granularity wherever a column allows it (flat,
 * fixed-stride, deferred input; see carquet_column_writer_plan_page_tasks):
 * one task per page, dynamically balanced over all OpenMP threads, so a row
 * group of three numeric columns keeps eight cores busy instead of three and
 * a wide column no longer bounds the wall time on its own. Columns that do
 * not qualify (dictionary, nullable, BYTE_ARRAY, bloom filter, ...) run as one
 * task each, scheduled first since they are the longest. Each qualifying
 * column then stitches its pages together in order; that pass is a memcpy
 * per column and runs column-parallel.
 */
static carquet_status_t finalize_columns_parallel(
    carquet_row_group_writer_t* writer,
    finalized_column_chunk_t* chunks) {
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;
    int nc = writer->num_columns;

    if (!writer->col_task_counts || !writer->col_task_offsets) {
        carquet_mem_free(writer->col_task_counts);
        carquet_mem_free(writer->col_task_offsets);
        writer->col_task_counts = carquet_mem_calloc((size_t)nc, sizeof(int32_t));
        writer->col_task_offsets = carquet_mem_calloc((size_t)nc, sizeof(int32_t));
        if (!writer->col_task_counts || !writer->col_task_offsets) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
    }

    /* Plan: count page tasks per column (0 = finalize serially as one task). */
    int32_t total_tasks = 0;
    int32_t whole_columns = 0;
    for (int i = 0; i < nc; i++) {
        int32_t n = (num_threads > 1)
            ? carquet_column_writer_plan_page_tasks(writer->column_writers[i], NULL, 0)
            : 0;
        if (n > 0 && n > INT32_MAX - total_tasks) n = 0;  /* absurd; go serial */
        writer->col_task_counts[i] = n;
        writer->col_task_offsets[i] = total_tasks;
        total_tasks += n;
        if (n == 0) whole_columns++;
    }

    if (total_tasks > 0) {
        carquet_status_t s = ensure_page_task_capacity(writer, total_tasks);
        if (s != CARQUET_OK) return s;
        s = ensure_scratch_writers(writer, num_threads);
        if (s != CARQUET_OK) return s;
        for (int i = 0; i < nc; i++) {
            int32_t n = writer->col_task_counts[i];
            if (n == 0) continue;
            carquet_page_task_t* t = &writer->page_tasks[writer->col_task_offsets[i]];
            int32_t got = carquet_column_writer_plan_page_tasks(
                writer->column_writers[i], t, n);
            if (got != n) return CARQUET_ERROR_INTERNAL;
            for (int32_t k = 0; k < n; k++) {
                t[k].out = &writer->task_bufs[writer->col_task_offsets[i] + k];
                t[k].status = CARQUET_OK;
            }
        }
    }

    /* Encode. Work item w < whole_columns is the w-th serial column; the rest
     * are page tasks in column order. */
    int32_t work_items = whole_columns + total_tasks;
    int team = num_threads < work_items ? num_threads : work_items;
    if (team < 1) team = 1;
    int32_t w;
    #pragma omp parallel for num_threads(team) schedule(dynamic, 1)
    for (w = 0; w < work_items; w++) {
        if (w < whole_columns) {
            int32_t seen = 0;
            for (int i = 0; i < nc; i++) {
                if (writer->col_task_counts[i] != 0) continue;
                if (seen++ == w) {
                    finalize_whole_column(writer, i, &chunks[i]);
                    break;
                }
            }
        } else {
            int tid = omp_get_thread_num();
            if (tid < 0 || tid >= writer->num_scratch) tid = 0;
            carquet_column_writer_run_page_task(
                &writer->page_tasks[w - whole_columns],
                writer->scratch_writers[tid]);
        }
    }

    /* Stitch each page-parallel column back together, in page order. */
    if (total_tasks > 0) {
        int cteam = num_threads < nc ? num_threads : nc;
        int i;
        #pragma omp parallel for num_threads(cteam) schedule(dynamic, 1)
        for (i = 0; i < nc; i++) {
            if (writer->col_task_counts[i] == 0) continue;
            chunks[i].status = carquet_column_writer_assemble_page_tasks(
                writer->column_writers[i],
                &writer->page_tasks[writer->col_task_offsets[i]],
                writer->col_task_counts[i]);
            if (chunks[i].status == CARQUET_OK) {
                finalize_whole_column(writer, i, &chunks[i]);
            }
        }
    }

    for (int i = 0; i < nc; i++) {
        if (chunks[i].status != CARQUET_OK) {
            return chunks[i].status;
        }
    }
#else
    (void)writer;
    (void)chunks;
#endif
    return CARQUET_OK;
}

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

    carquet_row_group_writer_t* writer = carquet_mem_calloc(1, sizeof(*writer));
    if (!writer) return NULL;

    carquet_buffer_init(&writer->row_group_buffer);

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
            carquet_mem_free(writer->column_writers);
        }

        if (writer->column_infos) {
            for (int i = 0; i < writer->num_columns; i++) {
                carquet_mem_free(writer->column_infos[i].path);
                carquet_mem_free(writer->column_infos[i].min_value);
                carquet_mem_free(writer->column_infos[i].max_value);
            }
            carquet_mem_free(writer->column_infos);
        }

        carquet_buffer_destroy(&writer->row_group_buffer);
        release_parallel_scratch(writer);
        carquet_mem_free(writer);
    }
}

void carquet_row_group_writer_reset(carquet_row_group_writer_t* writer, int64_t file_offset) {
    if (!writer) return;

    writer->num_rows = 0;
    writer->total_byte_size = 0;
    writer->file_offset = file_offset;
    carquet_buffer_clear(&writer->row_group_buffer);

    for (int i = 0; i < writer->num_columns; i++) {
        carquet_column_writer_reset(writer->column_writers[i]);
        writer->column_infos[i].file_offset = 0;
        writer->column_infos[i].total_compressed_size = 0;
        writer->column_infos[i].total_uncompressed_size = 0;
        writer->column_infos[i].num_values = 0;
        writer->column_infos[i].has_dictionary_page = false;
        writer->column_infos[i].dictionary_page_size = 0;
        carquet_mem_free(writer->column_infos[i].min_value);
        carquet_mem_free(writer->column_infos[i].max_value);
        writer->column_infos[i].min_value = NULL;
        writer->column_infos[i].max_value = NULL;
        writer->column_infos[i].min_value_size = 0;
        writer->column_infos[i].max_value_size = 0;
        writer->column_infos[i].has_min_max = false;
        writer->column_infos[i].has_null_count = false;
        writer->column_infos[i].null_count = 0;
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
    const carquet_logical_type_t* logical_type,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int32_t compression_level) {

    if (!writer || !name) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    int new_count = writer->num_columns + 1;

    /* Expand column writers array */
    carquet_column_writer_internal_t** new_writers = carquet_mem_realloc(
        writer->column_writers,
        new_count * sizeof(carquet_column_writer_internal_t*));
    if (!new_writers) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    writer->column_writers = new_writers;

    /* Expand column infos array */
    column_chunk_info_t* new_infos = carquet_mem_realloc(
        writer->column_infos,
        new_count * sizeof(column_chunk_info_t));
    if (!new_infos) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    writer->column_infos = new_infos;

    /* Create column writer with caller-resolved encoding/compression */
    carquet_column_writer_internal_t* col_writer = carquet_column_writer_create(
        type,
        logical_type,
        encoding,
        compression,
        max_def_level,
        max_rep_level,
        type_length,
        writer->target_page_size,
        compression_level);

    if (!col_writer) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    /* Enable optional features */
    if (writer->write_bloom_filters) {
        carquet_column_writer_enable_bloom_filter(col_writer, 100000);
    }
    if (writer->write_page_index) {
        carquet_column_writer_enable_page_index(col_writer);
    }
    carquet_column_writer_set_statistics(col_writer, writer->write_statistics);
    carquet_column_writer_set_crc(col_writer, writer->write_crc);
    if (writer->dictionary_page_size > 0) {
        carquet_column_writer_set_dictionary_page_size_limit(
            col_writer, writer->dictionary_page_size);
    }

    writer->column_writers[writer->num_columns] = col_writer;

    /* Initialize column info */
    memset(&writer->column_infos[writer->num_columns], 0, sizeof(column_chunk_info_t));
    writer->column_infos[writer->num_columns].type = type;
    if (logical_type) {
        writer->column_infos[writer->num_columns].logical_type = *logical_type;
    }
    writer->column_infos[writer->num_columns].encoding = encoding;
    writer->column_infos[writer->num_columns].compression = compression;
    writer->column_infos[writer->num_columns].type_length = type_length;
    writer->column_infos[writer->num_columns].path = carquet_heap_strdup(name);
    if (!writer->column_infos[writer->num_columns].path) {
        carquet_column_writer_destroy(col_writer);
        writer->column_writers[writer->num_columns] = NULL;
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

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

    /* Defer encode+compress to the parallel per-column finalize only when that
     * finalize will actually run in parallel. num_columns and write_page_index
     * are fixed before any write, so this is invariant across a row group's
     * batches; set_defer_encode also self-gates on column eligibility. */
    carquet_column_writer_set_defer_encode(
        writer->column_writers[column_index], can_parallel_finalize(writer));

    return carquet_column_writer_write_batch(
        writer->column_writers[column_index],
        values, num_values, def_levels, rep_levels);
}

/* ============================================================================
 * Finalization
 * ============================================================================
 */

/* Encode and compress every column of the row group, fill in the per-column
 * chunk info (file offsets, sizes, statistics) and report the column bytes
 * in chunks[]. No I/O: each column's bytes stay in its chunk buffer. */
static carquet_status_t finalize_all_columns(
    carquet_row_group_writer_t* writer,
    int64_t num_rows,
    finalized_column_chunk_t* chunks) {

    writer->num_rows = num_rows;
    writer->total_byte_size = 0;
    int64_t current_offset = writer->file_offset;

    if (can_parallel_finalize(writer)) {
        carquet_status_t status = finalize_columns_parallel(writer, chunks);
        if (status != CARQUET_OK) {
            return status;
        }
        for (int i = 0; i < writer->num_columns; i++) {
            writer->column_infos[i].file_offset = current_offset;
            writer->column_infos[i].total_compressed_size = chunks[i].size;
            writer->column_infos[i].total_uncompressed_size = chunks[i].uncompressed_size;
            writer->column_infos[i].num_values = chunks[i].total_values;
            capture_column_statistics(writer, i);
            capture_dictionary_info(writer, i);
            current_offset += chunks[i].size;
            writer->total_byte_size += chunks[i].size;
        }
        return CARQUET_OK;
    }

    for (int i = 0; i < writer->num_columns; i++) {
        /* Set file offset before finalize so page index has correct offsets */
        carquet_column_writer_set_file_offset(writer->column_writers[i], current_offset);

        chunks[i].status = carquet_column_writer_finalize(
            writer->column_writers[i],
            &chunks[i].data, &chunks[i].size,
            &chunks[i].total_values,
            &chunks[i].compressed_size,
            &chunks[i].uncompressed_size);
        if (chunks[i].status != CARQUET_OK) {
            return chunks[i].status;
        }

        writer->column_infos[i].file_offset = current_offset;
        writer->column_infos[i].total_compressed_size = chunks[i].size;
        writer->column_infos[i].total_uncompressed_size = chunks[i].uncompressed_size;
        writer->column_infos[i].num_values = chunks[i].total_values;
        capture_column_statistics(writer, i);
        capture_dictionary_info(writer, i);

        current_offset += chunks[i].size;
        writer->total_byte_size += chunks[i].size;
    }
    return CARQUET_OK;
}

carquet_status_t carquet_row_group_writer_finalize(
    carquet_row_group_writer_t* writer,
    const uint8_t** data,
    size_t* size,
    int64_t num_rows) {

    if (!writer) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_buffer_clear(&writer->row_group_buffer);

    finalized_column_chunk_t* chunks = carquet_mem_calloc((size_t)writer->num_columns, sizeof(*chunks));
    if (!chunks) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    carquet_status_t status = finalize_all_columns(writer, num_rows, chunks);
    for (int i = 0; status == CARQUET_OK && i < writer->num_columns; i++) {
        status = carquet_buffer_append(&writer->row_group_buffer, chunks[i].data, chunks[i].size);
    }
    carquet_mem_free(chunks);
    if (status != CARQUET_OK) {
        return status;
    }

    if (data) *data = writer->row_group_buffer.data;
    if (size) *size = writer->row_group_buffer.size;
    return CARQUET_OK;
}

carquet_status_t carquet_row_group_writer_write_to_file(
    carquet_row_group_writer_t* writer,
    FILE* file,
    size_t* total_size,
    int64_t num_rows) {

    if (!writer || !file) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    finalized_column_chunk_t* chunks = carquet_mem_calloc((size_t)writer->num_columns, sizeof(*chunks));
    if (!chunks) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    carquet_status_t status = finalize_all_columns(writer, num_rows, chunks);
    size_t written = 0;
    for (int i = 0; status == CARQUET_OK && i < writer->num_columns; i++) {
        if (chunks[i].size > 0 &&
            fwrite(chunks[i].data, 1, chunks[i].size, file) != chunks[i].size) {
            status = CARQUET_ERROR_FILE_WRITE;
            break;
        }
        written += chunks[i].size;
    }
    carquet_mem_free(chunks);
    if (status != CARQUET_OK) {
        return status;
    }

    if (total_size) *total_size = written;
    return CARQUET_OK;
}

carquet_status_t carquet_row_group_writer_finalize_columns(
    carquet_row_group_writer_t* writer,
    size_t* total_size,
    int64_t num_rows) {

    if (!writer) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    finalized_column_chunk_t* chunks = carquet_mem_calloc((size_t)writer->num_columns, sizeof(*chunks));
    if (!chunks) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    carquet_status_t status = finalize_all_columns(writer, num_rows, chunks);
    size_t total = 0;
    for (int i = 0; i < writer->num_columns; i++) {
        total += chunks[i].size;
    }
    carquet_mem_free(chunks);
    if (status != CARQUET_OK) {
        return status;
    }
    if (total_size) *total_size = total;
    return CARQUET_OK;
}

void carquet_row_group_writer_detach_column(
    carquet_row_group_writer_t* writer,
    int column_index,
    carquet_buffer_t* out,
    carquet_buffer_t* spare) {

    if (!writer || column_index < 0 || column_index >= writer->num_columns || !out) {
        if (out) carquet_buffer_init(out);
        return;
    }
    carquet_column_writer_detach_buffer(writer->column_writers[column_index], out, spare);
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

void carquet_row_group_writer_set_options(
    carquet_row_group_writer_t* writer,
    bool write_bloom_filters,
    bool write_page_index,
    bool write_statistics,
    bool write_crc,
    int32_t compression_level,
    int64_t dictionary_page_size) {
    if (writer) {
        writer->write_bloom_filters = write_bloom_filters;
        writer->write_page_index = write_page_index;
        writer->write_statistics = write_statistics;
        writer->write_crc = write_crc;
        writer->compression_level = compression_level;
        writer->dictionary_page_size = dictionary_page_size;
        if (dictionary_page_size > 0) {
            for (int i = 0; i < writer->num_columns; i++) {
                carquet_column_writer_set_dictionary_page_size_limit(
                    writer->column_writers[i], dictionary_page_size);
            }
        }
    }
}

void carquet_row_group_writer_configure_column_bloom(
    carquet_row_group_writer_t* writer,
    int column_index, bool enabled, int64_t ndv, double fpp) {
    if (!writer || column_index < 0 || column_index >= writer->num_columns) {
        return;
    }
    carquet_column_writer_configure_bloom_filter(
        writer->column_writers[column_index], enabled, ndv, fpp);
}

void carquet_row_group_writer_set_column_max_rows_per_page(
    carquet_row_group_writer_t* writer,
    int column_index, int64_t max_rows) {
    if (!writer || column_index < 0 || column_index >= writer->num_columns) {
        return;
    }
    carquet_column_writer_set_max_rows_per_page(
        writer->column_writers[column_index], max_rows);
}

void carquet_row_group_writer_set_column_write_batch_size(
    carquet_row_group_writer_t* writer,
    int column_index, int64_t batch_size) {
    if (!writer || column_index < 0 || column_index >= writer->num_columns) {
        return;
    }
    carquet_column_writer_set_write_batch_size(
        writer->column_writers[column_index], batch_size);
}

void carquet_row_group_writer_set_column_page_size(
    carquet_row_group_writer_t* writer,
    int column_index, int64_t bytes) {
    if (!writer || column_index < 0 || column_index >= writer->num_columns) {
        return;
    }
    carquet_column_writer_set_target_page_size(
        writer->column_writers[column_index], bytes);
}

void carquet_row_group_writer_set_column_data_page_v2(
    carquet_row_group_writer_t* writer,
    int column_index, bool enabled) {
    if (!writer || column_index < 0 || column_index >= writer->num_columns) {
        return;
    }
    carquet_column_writer_set_data_page_v2(
        writer->column_writers[column_index], enabled);
}

carquet_bloom_filter_t* carquet_row_group_writer_get_bloom_filter(
    const carquet_row_group_writer_t* writer, int index) {
    if (!writer || index < 0 || index >= writer->num_columns) return NULL;
    return carquet_column_writer_get_bloom_filter(writer->column_writers[index]);
}

carquet_column_index_builder_t* carquet_row_group_writer_get_column_index(
    const carquet_row_group_writer_t* writer, int index) {
    if (!writer || index < 0 || index >= writer->num_columns) return NULL;
    return carquet_column_writer_get_column_index(writer->column_writers[index]);
}

carquet_offset_index_builder_t* carquet_row_group_writer_get_offset_index(
    const carquet_row_group_writer_t* writer, int index) {
    if (!writer || index < 0 || index >= writer->num_columns) return NULL;
    return carquet_column_writer_get_offset_index(writer->column_writers[index]);
}
