/**
 * @file batch_reader.c
 * @brief High-level batch reader with column projection and parallel I/O
 *
 * This provides a production-ready API for efficiently reading Parquet files
 * with support for:
 * - Column projection (only read needed columns)
 * - Parallel column reading
 * - Memory-mapped I/O
 * - Batched output
 * - Buffer pooling to minimize allocations
 */

#include "core/allocator.h"
#include <carquet/carquet.h>
#include "reader_internal.h"
#include "worker_pool.h"
#include "page_filter.h"
#include "core/arena.h"
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>   /* sysconf(_SC_PAGESIZE) */
#endif

/* SIMD dispatch function for null bitmap construction */
extern void carquet_dispatch_build_null_bitmap(const int16_t* def_levels, int64_t count,
                                                int16_t max_def_level, uint8_t* null_bitmap);

#define CARQUET_MAX_PAGE_PAYLOAD_SIZE (256ULL * 1024 * 1024)

/* Upper bound for the pipeline's per-slot, per-column buffer pre-allocation.
 * Sized from attacker-controlled row_group.num_rows; above this we fall back
 * to lazy allocation in pipeline_fill instead of eagerly malloc'ing. */
#define CARQUET_MAX_PREALLOC_BYTES (1024ULL * 1024 * 1024)

/* ============================================================================
 * Internal Structures
 * ============================================================================
 */

typedef struct carquet_column_data {
    void* data;                 /* Column values (or uint32_t* indices if dict preserved) */
    uint8_t* null_bitmap;       /* Null bitmap (1 bit per value), NULL for REQUIRED */
    int64_t num_values;         /* Number of values */
    size_t data_capacity;       /* Allocated capacity for data */
    carquet_physical_type_t type;
    int32_t type_length;        /* For fixed-length types */
    carquet_data_ownership_t ownership;  /* OWNED or VIEW (for future zero-copy) */

    /* Dictionary preservation (when config.preserve_dictionaries == true) */
    bool is_dictionary;             /* True if this column has preserved dictionary */
    const uint8_t* dictionary_data; /* Pointer to dictionary bytes (view, not owned) */
    int32_t dictionary_count;       /* Number of dictionary entries */
    const uint32_t* dictionary_offsets; /* Offset table for BYTE_ARRAY (view) */

    /* Nested (single-level LIST/MAP-leaf) reconstruction. When list_offsets is
     * non-NULL this column is a list: `data`/`null_bitmap`/`num_values` describe
     * the flattened child (element) array (Arrow child layout — values with a
     * validity bitmap), and list_offsets[i]..list_offsets[i+1] delimit list i in
     * that child array. list_validity (may be NULL) is the list-level null
     * bitmap. num_lists is the logical row count. */
    int32_t* list_offsets;      /* [num_lists + 1] Arrow list offsets, or NULL */
    uint8_t* list_validity;     /* list-level validity bitmap (LSB, present=1), or NULL */
    int64_t  num_lists;         /* number of logical rows (lists) */
    int16_t  max_rep_level;     /* > 0 marks a repeated (list) column */
} carquet_column_data_t;

/* Pre-allocated column buffer pool for reuse across batches */
typedef struct carquet_column_pool {
    void* data;                 /* Pre-allocated data buffer */
    size_t data_capacity;       /* Capacity in bytes */
    uint8_t* null_bitmap;       /* Pre-allocated null bitmap */
    size_t bitmap_capacity;     /* Capacity in bytes */
    int16_t* def_levels;        /* Pre-allocated def levels buffer */
    size_t def_levels_capacity; /* Capacity in elements */
    /* Nested (list) reconstruction scratch/output buffers */
    int16_t* rep_levels;        /* Pre-allocated rep levels buffer */
    size_t rep_levels_capacity; /* Capacity in elements */
    int32_t* list_offsets;      /* Pre-allocated list offsets buffer */
    size_t list_offsets_capacity; /* Capacity in elements */
    uint8_t* list_validity;     /* Pre-allocated list-level validity bitmap */
    size_t list_validity_capacity; /* Capacity in bytes */
} carquet_column_pool_t;

struct carquet_row_batch {
    carquet_column_data_t* columns;
    int32_t num_columns;
    int64_t num_rows;
    carquet_arena_t arena;
    bool pooled;                /* If true, data buffers are from batch_reader pool */
};

/* Pipeline ring buffer slot: holds pre-read column data for one RG */
typedef struct {
    carquet_column_reader_t** col_readers;  /* [num_projected] readers, used for bulk read */
    int32_t rg_index;                       /* row group index, -1 = empty */
    bool ready;                             /* all columns fully read */

    /* Pre-read value buffers (entire column chunk per column) */
    void** col_values;       /* [num_projected] value buffers */
    size_t* col_buf_sizes;   /* [num_projected] buffer capacities in bytes */
    int64_t* col_num_values; /* [num_projected] values actually read */
    int64_t total_rows;      /* total rows in this slot (= range total when filter active) */
    int64_t rows_consumed;   /* rows already served to batch_reader_next */

    /* Page-filter row ranges for this slot (only populated when a page
     * filter is active). Per-projected-column offset indexes are cached
     * here so worker tasks can seek to matching pages without going
     * through the file reader again. */
    carquet_row_range_list_t ranges;
    bool filter_ranges_valid;
    carquet_offset_index_t** col_offset_indexes;  /* [num_projected], may be NULL entries */

    /* Per-slot independent mmap for this row group's byte range.
     * Avoids page table lock contention when 12+ threads fault pages
     * from the same shared mmap simultaneously. */
#if !defined(_WIN32)
    uint8_t* slot_mmap;         /* independent mmap for this RG, or NULL */
    size_t   slot_mmap_size;    /* mmap length */
    int64_t  slot_mmap_offset;  /* file offset corresponding to slot_mmap[0] */
#endif
} rg_slot_t;

/* Forward decl shared with filtered bulk-read task (defined later). */
static int32_t find_page_for_row(
    const carquet_offset_index_t* oi,
    int64_t row_group_num_rows,
    int64_t target_row,
    int64_t* page_first_row_out);

/* Forward declarations for coalesced read fast path.
 * data_base: pointer to file data (per-slot mmap or shared mmap).
 *            Byte at file offset N is at data_base[N]. */
static bool can_coalesce_column(const carquet_column_reader_t* cr);
static void coalesced_read_column_range(const carquet_column_reader_t* cr,
    const uint8_t* data_base, void* dest, int64_t max_values,
    int64_t start_offset, int64_t end_offset, int64_t* out_values_read);
static void coalesced_read_column(const carquet_column_reader_t* cr,
    void* dest, int64_t max_values, int64_t* out_values_read);
static int32_t plan_coalesced_column_splits(const carquet_column_reader_t* cr,
    const uint8_t* data_base, int64_t max_values, int32_t max_splits,
    int64_t* split_offsets, int64_t* split_values);

extern carquet_status_t carquet_byte_stream_split_decode_float(
    const uint8_t* data, size_t data_size, float* values, int64_t count);
extern carquet_status_t carquet_byte_stream_split_decode_double(
    const uint8_t* data, size_t data_size, double* values, int64_t count);

/* Task argument for parallel bulk column reading */
typedef struct {
    carquet_column_reader_t* col_reader;
    const uint8_t* data_base;  /* file data pointer (per-slot or shared mmap) */
    void* dest;
    int64_t max_values;
    int64_t* out_values_read;
    int64_t start_offset;      /* 0 = full chunk */
    int64_t end_offset;        /* 0 = full chunk */
    int64_t local_values_read; /* scratch for split tasks */

    /* Page-filter mode: when ranges is non-NULL the task reads only the
     * matching pages, writing rows contiguously into dest. */
    const carquet_row_range_list_t* ranges;
    const carquet_offset_index_t* offset_index;
    int64_t rg_num_rows;
    size_t value_size;
} bulk_read_arg_t;

struct carquet_batch_reader {
    carquet_reader_t* reader;
    carquet_batch_reader_config_t config;

    /* Column projection */
    int32_t* projected_columns;  /* File column indices to read */
    int32_t num_projected;       /* Number of projected columns */
    carquet_physical_type_t* projected_types;
    int32_t* projected_type_lengths;
    int16_t* projected_max_defs;
    int16_t* projected_max_reps;
    size_t* projected_value_sizes;
    bool has_repeated;           /* true if any projected column has max_rep > 0 */

    /* Reading state */
    int32_t current_row_group;
    int64_t rows_read_in_group;
    int64_t total_rows_read;

    /* Column readers for current row group */
    carquet_column_reader_t** col_readers;

    /* Memory-mapped data */
    uint8_t* mmap_data;
    size_t mmap_size;

    /* Buffer pool for reuse across batches (one per projected column) */
    carquet_column_pool_t* col_pools;

    /* Cached batch struct to avoid repeated alloc/free */
    carquet_row_batch_t* cached_batch;

    /* Persistent worker pool for cross-RG parallel decompression */
    carquet_worker_pool_t* pool;
    bool pool_is_borrowed;  /* true when pool comes from config.thread_pool */

    /* Pipeline ring buffer for multi-RG parallel decompression.
     * Pre-decompresses pages for upcoming row groups so that by the time
     * batch_reader_next() needs data, it's already decompressed. */
    rg_slot_t* pipeline;          /* [pipeline_depth] ring buffer */
    int32_t pipeline_depth;       /* window size */
    int32_t pipeline_head;        /* next slot to consume */
    int32_t pipeline_count;       /* slots in use */
    int32_t* rg_order;            /* pre-filtered list of RG indices */
    int32_t rg_order_len;         /* total filtered RGs */
    int32_t rg_order_next;        /* next RG to submit */
    bool pipeline_active;         /* multi-RG pipeline enabled */

    /* Per-reader task args (replaces static global array) */
    bulk_read_arg_t* task_args;
    int32_t task_args_capacity;

    /* ====================================================================
     * Page filter state
     * ==================================================================== */
    /* Active filter clauses (caller-owned; not copied). NULL = no filter. */
    const carquet_filter_clause_t* filter_clauses;
    int32_t filter_clause_count;

    /* Row ranges that survive the conjunction for current_row_group.
     * Valid only when filter_rg_state_valid is true. */
    carquet_row_range_list_t current_rg_ranges;
    bool filter_rg_state_valid;
    int32_t current_range_index;
    int64_t current_range_rows_emitted;
    bool range_positioned;          /* Column readers seeked to current range start? */
    int64_t rows_skipped;           /* Diagnostic accumulator */

    /* Per-projected-column offset index cache for the current row group.
     * Loaded lazily on first range positioning, freed on RG transition. */
    carquet_offset_index_t** projected_offset_indexes;
    int32_t projected_oi_rg;        /* -1 when cache is empty */
};

/* ============================================================================
 * Configuration
 * ============================================================================
 */

void carquet_batch_reader_config_init(carquet_batch_reader_config_t* config) {
    /* config is nonnull per API contract */
    memset(config, 0, sizeof(*config));
    config->batch_size = 65536;  /* 64K rows per batch */
    config->num_threads = 0;     /* Auto-detect */
    config->use_mmap = false;
}

/* ============================================================================
 * Helper Functions
 * ============================================================================
 */

/* Maximum reasonable type_length for FIXED_LEN_BYTE_ARRAY (16 MB) */
#define CARQUET_MAX_TYPE_LENGTH (16 * 1024 * 1024)

static size_t get_type_size(carquet_physical_type_t type, int32_t type_length) {
    switch (type) {
        case CARQUET_PHYSICAL_BOOLEAN: return 1;
        case CARQUET_PHYSICAL_INT32: return 4;
        case CARQUET_PHYSICAL_INT64: return 8;
        case CARQUET_PHYSICAL_INT96: return 12;
        case CARQUET_PHYSICAL_FLOAT: return 4;
        case CARQUET_PHYSICAL_DOUBLE: return 8;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            /* Validate type_length to prevent overflow attacks */
            if (type_length <= 0 || type_length > CARQUET_MAX_TYPE_LENGTH) {
                return 0;  /* Invalid - will cause allocation to fail safely */
            }
            return (size_t)type_length;
        case CARQUET_PHYSICAL_BYTE_ARRAY: return sizeof(carquet_byte_array_t);
        default: return 0;
    }
}

static int resolve_column_name(const carquet_reader_t* reader, const char* name) {
    const carquet_schema_t* schema = carquet_reader_schema(reader);
    if (!schema) return -1;

    return carquet_schema_find_column(schema, name);
}

/* Ensure a pool buffer is at least 'needed' bytes, growing if necessary */
static void* pool_ensure_data(carquet_column_pool_t* pool, size_t needed) {
    if (needed <= pool->data_capacity) {
        return pool->data;
    }
    carquet_mem_free(pool->data);
    pool->data = carquet_mem_malloc(needed);
    pool->data_capacity = pool->data ? needed : 0;
    return pool->data;
}

static uint8_t* pool_ensure_bitmap(carquet_column_pool_t* pool, size_t needed) {
    if (needed <= pool->bitmap_capacity) {
        memset(pool->null_bitmap, 0, needed);
        return pool->null_bitmap;
    }
    carquet_mem_free(pool->null_bitmap);
    pool->null_bitmap = carquet_mem_calloc(1, needed);
    pool->bitmap_capacity = pool->null_bitmap ? needed : 0;
    return pool->null_bitmap;
}

static int16_t* pool_ensure_def_levels(carquet_column_pool_t* pool, size_t count) {
    if (count <= pool->def_levels_capacity) {
        return pool->def_levels;
    }
    carquet_mem_free(pool->def_levels);
    pool->def_levels = carquet_mem_malloc(sizeof(int16_t) * count);
    pool->def_levels_capacity = pool->def_levels ? count : 0;
    return pool->def_levels;
}

static int16_t* pool_ensure_rep_levels(carquet_column_pool_t* pool, size_t count) {
    if (count <= pool->rep_levels_capacity) {
        return pool->rep_levels;
    }
    carquet_mem_free(pool->rep_levels);
    pool->rep_levels = carquet_mem_malloc(sizeof(int16_t) * count);
    pool->rep_levels_capacity = pool->rep_levels ? count : 0;
    return pool->rep_levels;
}

static int32_t* pool_ensure_list_offsets(carquet_column_pool_t* pool, size_t count) {
    if (count <= pool->list_offsets_capacity) {
        return pool->list_offsets;
    }
    carquet_mem_free(pool->list_offsets);
    pool->list_offsets = carquet_mem_malloc(sizeof(int32_t) * count);
    pool->list_offsets_capacity = pool->list_offsets ? count : 0;
    return pool->list_offsets;
}

static uint8_t* pool_ensure_list_validity(carquet_column_pool_t* pool, size_t bytes) {
    if (bytes <= pool->list_validity_capacity) {
        memset(pool->list_validity, 0, bytes);
        return pool->list_validity;
    }
    carquet_mem_free(pool->list_validity);
    pool->list_validity = carquet_mem_calloc(1, bytes);
    pool->list_validity_capacity = pool->list_validity ? bytes : 0;
    return pool->list_validity;
}

static bool column_can_zero_copy_batch(
    const carquet_column_reader_t* col_reader,
    carquet_physical_type_t type,
    int16_t max_def,
    int64_t rows_to_read) {

    if (!col_reader || !col_reader->page_loaded ||
        col_reader->decoded_ownership != CARQUET_DATA_VIEW ||
        max_def != 0 || col_reader->max_rep_level > 0 ||
        type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        return false;
    }

    int32_t page_available = col_reader->page_num_values - col_reader->page_values_read;
    return page_available > 0 && page_available >= (int32_t)rows_to_read;
}

static int64_t column_zero_copy_rows_available(
    const carquet_column_reader_t* col_reader,
    carquet_physical_type_t type,
    int16_t max_def) {

    if (!col_reader || !col_reader->page_loaded ||
        col_reader->decoded_ownership != CARQUET_DATA_VIEW ||
        max_def != 0 || col_reader->max_rep_level > 0 ||
        type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        return 0;
    }

    int32_t page_available = col_reader->page_num_values - col_reader->page_values_read;
    return page_available > 0 ? page_available : 0;
}

static int64_t clamp_rows_to_zero_copy_window(
    const carquet_batch_reader_t* batch_reader,
    int64_t rows_to_read) {

    int64_t zero_copy_rows = rows_to_read;

    for (int32_t i = 0; i < batch_reader->num_projected; i++) {
        int64_t page_rows = column_zero_copy_rows_available(
            batch_reader->col_readers[i],
            batch_reader->projected_types[i],
            batch_reader->projected_max_defs[i]);

        if (page_rows <= 0) {
            return rows_to_read;
        }
        if (page_rows < zero_copy_rows) {
            zero_copy_rows = page_rows;
        }
    }

    return zero_copy_rows;
}

static bool column_is_zero_copy_candidate(
    const carquet_column_reader_t* col_reader,
    carquet_physical_type_t type,
    int16_t max_def) {

    if (!col_reader || !col_reader->file_reader ||
        col_reader->file_reader->mmap_data == NULL ||
        !col_reader->col_meta ||
        max_def != 0 || col_reader->max_rep_level > 0 ||
        col_reader->col_meta->codec != CARQUET_COMPRESSION_UNCOMPRESSED) {
        return false;
    }

    switch (type) {
        case CARQUET_PHYSICAL_INT32:
        case CARQUET_PHYSICAL_INT64:
        case CARQUET_PHYSICAL_INT96:
        case CARQUET_PHYSICAL_FLOAT:
        case CARQUET_PHYSICAL_DOUBLE:
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            return true;
        case CARQUET_PHYSICAL_BOOLEAN:
        case CARQUET_PHYSICAL_BYTE_ARRAY:
        default:
            return false;
    }
}

static void read_projected_column(
    carquet_batch_reader_t* batch_reader,
    carquet_row_batch_t* new_batch,
    int32_t col_i,
    int64_t rows_to_read,
    bool allow_zero_copy,
    bool* read_error) {

    if (*read_error) {
        return;
    }

    carquet_column_reader_t* col_reader = batch_reader->col_readers[col_i];
    carquet_column_data_t* col_data = &new_batch->columns[col_i];
    carquet_column_pool_t* pool = &batch_reader->col_pools[col_i];
    size_t value_size = batch_reader->projected_value_sizes[col_i];
    int16_t max_def = batch_reader->projected_max_defs[col_i];

    col_data->type = batch_reader->projected_types[col_i];
    col_data->type_length = batch_reader->projected_type_lengths[col_i];
    col_data->is_dictionary = false;
    col_data->dictionary_data = NULL;
    col_data->dictionary_count = 0;
    col_data->dictionary_offsets = NULL;

    /* Dictionary preservation was decided in reset_column_reader_for_row_group
     * (before any page load), so the decode/copy width is consistent. Just
     * read the resolved flag here. */
    bool use_dict_preserve = col_reader->preserve_dictionary;

    /* When preserving dictionaries, value_size is sizeof(uint32_t) for indices */
    size_t effective_value_size = use_dict_preserve ? sizeof(uint32_t) : value_size;

    /* Check if direct page handoff is possible:
     * - Column is REQUIRED (no nulls, no definition levels)
     * - Page loader can expose a stable view (mmap or reusable page buffer)
     * - Entire page slice fits in this batch
     * - Not in dictionary-preserve mode (indices layout differs)
     */
    bool try_zero_copy = allow_zero_copy &&
                         (max_def == 0) &&
                         (!col_reader->page_loaded) &&
                         !use_dict_preserve;

    if (try_zero_copy) {
        /* Trigger page load to check if it's a zero-copy page */
        int64_t dummy_read = carquet_column_read_batch(
            col_reader, NULL, 0, NULL, NULL);
        (void)dummy_read;
    }

    bool use_zero_copy = allow_zero_copy && !use_dict_preserve &&
                         column_can_zero_copy_batch(
                             col_reader, col_data->type, max_def, rows_to_read);

    if (use_zero_copy) {
        /* ====== ZERO-COPY PATH ====== */
        /* Point directly to the currently loaded page slice. */
        size_t byte_offset = (size_t)col_reader->page_values_read * value_size;
        col_data->data = (uint8_t*)col_reader->decoded_values + byte_offset;
        col_data->data_capacity = 0;  /* Not our allocation */
        col_data->ownership = CARQUET_DATA_VIEW;
        col_data->num_values = rows_to_read;

        /* No nulls in REQUIRED columns - return NULL bitmap */
        col_data->null_bitmap = NULL;

        /* Mark page as consumed */
        col_reader->page_values_read += (int32_t)rows_to_read;
        col_reader->values_remaining -= rows_to_read;
        return;
    }

    /* ====== STANDARD PATH (with copy, using pooled buffers) ====== */

    /* Validate value_size and check for overflow */
    if (effective_value_size == 0 || rows_to_read <= 0) {
        *read_error = true;
        return;
    }

    /* Check for multiplication overflow (max 1GB allocation) */
    #define CARQUET_MAX_BATCH_ALLOC (1024ULL * 1024 * 1024)
    if (effective_value_size > CARQUET_MAX_BATCH_ALLOC / (size_t)rows_to_read) {
        *read_error = true;
        return;
    }

    size_t data_size = effective_value_size * (size_t)rows_to_read;

    /* Use pooled data buffer (grows as needed, never shrinks) */
    col_data->data = pool_ensure_data(pool, data_size);
    if (!col_data->data) {
        *read_error = true;
        return;
    }
    col_data->data_capacity = data_size;
    col_data->ownership = CARQUET_DATA_VIEW;  /* Pool owns the buffer */

    /* Only allocate null bitmap for OPTIONAL columns */
    if (max_def > 0) {
        size_t bitmap_size = ((size_t)rows_to_read + 7) / 8;
        col_data->null_bitmap = pool_ensure_bitmap(pool, bitmap_size);
    } else {
        col_data->null_bitmap = NULL;  /* REQUIRED columns have no nulls */
    }

    /* Read values (reuse pooled def_levels buffer) */
    int16_t* def_levels = NULL;
    if (max_def > 0) {
        def_levels = pool_ensure_def_levels(pool, (size_t)rows_to_read);
    }

    int64_t values_read = carquet_column_read_batch(
        col_reader, col_data->data, rows_to_read, def_levels, NULL);

    if (values_read < 0) {
        *read_error = true;
        return;
    }

    col_data->num_values = values_read;

    /* Attach dictionary metadata for preserved dictionary columns */
    if (use_dict_preserve) {
        col_data->is_dictionary = true;
        col_data->dictionary_data = col_reader->dictionary_data;
        col_data->dictionary_count = col_reader->dictionary_count;
        col_data->dictionary_offsets = col_reader->dictionary_offsets;
    }

    /* The column reader returns dense non-null values (Parquet convention).
     * The batch reader's contract is row-aligned: value[i] corresponds to
     * logical row i, with null slots zeroed.  Expand in-place (back-to-front)
     * so the data and null bitmap are consistent. */
    if (def_levels && max_def > 0 && values_read > 0) {
        int64_t non_null = 0;
        for (int64_t k = 0; k < values_read; k++) {
            if (def_levels[k] == max_def) non_null++;
        }

        if (non_null < values_read) {
            uint8_t* data = (uint8_t*)col_data->data;
            int64_t src = non_null - 1;
            for (int64_t dst = values_read - 1; dst >= 0; dst--) {
                uint8_t* dp = data + (size_t)dst * effective_value_size;
                if (def_levels[dst] == max_def) {
                    uint8_t* sp = data + (size_t)src * effective_value_size;
                    if (sp != dp) memmove(dp, sp, effective_value_size);
                    src--;
                } else {
                    memset(dp, 0, effective_value_size);
                }
            }
        }
    }

    /* Build null bitmap from definition levels (uses SIMD when available) */
    if (def_levels && col_data->null_bitmap) {
        carquet_dispatch_build_null_bitmap(def_levels, values_read,
                                           max_def, col_data->null_bitmap);
    }
}

/* ============================================================================
 * Nested (single-level LIST) reconstruction
 * ============================================================================
 * Reconstructs an Arrow List<T> layout for a repeated column (max_rep == 1) by
 * reading the entire column chunk's leaf slots (values + def/rep levels) and
 * folding them into: a flattened child (element) array, its validity bitmap,
 * a list-offsets buffer, and a list-level validity bitmap.
 *
 * This handles the shapes produced by carquet_schema_add_list (and each MAP
 * leaf), i.e. a single repeated ancestor. Deeper nesting (max_rep > 1) is not
 * supported and sets *read_error.
 *
 * Definition-level bands for a single-level list (max_def = D):
 *   - a slot is an *element* of its list when def >= elem_exists (D-1 if the
 *     element leaf is OPTIONAL, else D);
 *   - the element's value is *present* when def == D (else it is a null element);
 *   - a rep==0 slot with def == 0 is a *null list* (only reachable when an
 *     optional ancestor sits above the repeated group).
 */
static void read_nested_list_column(
    carquet_batch_reader_t* batch_reader,
    carquet_row_batch_t* new_batch,
    int32_t col_i,
    int64_t expected_rows,
    bool* read_error) {

    if (*read_error) {
        return;
    }

    carquet_column_reader_t* col_reader = batch_reader->col_readers[col_i];
    carquet_column_data_t* col_data = &new_batch->columns[col_i];
    carquet_column_pool_t* pool = &batch_reader->col_pools[col_i];
    size_t value_size = batch_reader->projected_value_sizes[col_i];
    int16_t max_def = batch_reader->projected_max_defs[col_i];
    int16_t max_rep = batch_reader->projected_max_reps[col_i];

    col_data->type = batch_reader->projected_types[col_i];
    col_data->type_length = batch_reader->projected_type_lengths[col_i];
    col_data->max_rep_level = max_rep;

    /* Only single-level lists are supported in this release. */
    if (max_rep != 1 || value_size == 0 || max_def < 1) {
        *read_error = true;
        return;
    }

    /* Element-optional flag from the leaf node's own repetition. */
    const carquet_schema_t* schema = batch_reader->reader->schema;
    int32_t file_col = batch_reader->projected_columns[col_i];
    const parquet_schema_element_t* leaf = &schema->elements[schema->leaf_indices[file_col]];
    bool elem_optional = (leaf->repetition_type == CARQUET_REPETITION_OPTIONAL);
    int16_t elem_exists = elem_optional ? (int16_t)(max_def - 1) : max_def;

    /* Total leaf slots in this chunk = number of (def, rep) entries. */
    int64_t total_slots = carquet_column_remaining(col_reader);
    if (total_slots < 0) { *read_error = true; return; }

    /* Bound allocations. */
    if (total_slots > 0 &&
        value_size > CARQUET_MAX_BATCH_ALLOC / (size_t)total_slots) {
        *read_error = true;
        return;
    }

    size_t slots_alloc = total_slots > 0 ? (size_t)total_slots : 1;
    void* data = pool_ensure_data(pool, value_size * slots_alloc);
    int16_t* def_levels = pool_ensure_def_levels(pool, slots_alloc);
    int16_t* rep_levels = pool_ensure_rep_levels(pool, slots_alloc);
    if (!data || !def_levels || !rep_levels) { *read_error = true; return; }

    int64_t slots = carquet_column_read_batch(
        col_reader, data, total_slots, def_levels, rep_levels);
    if (slots < 0) { *read_error = true; return; }

    /* Pass 1: count lists (rep==0), child elements (def >= elem_exists). */
    int64_t num_lists = 0, child_count = 0;
    for (int64_t j = 0; j < slots; j++) {
        if (rep_levels[j] == 0) num_lists++;
        if (def_levels[j] >= elem_exists) child_count++;
    }
    if (num_lists > INT32_MAX || child_count > INT32_MAX) {
        *read_error = true;
        return;
    }
    (void)expected_rows;  /* num_lists is authoritative; equals the RG row count */

    col_data->data = data;
    col_data->data_capacity = value_size * slots_alloc;
    col_data->ownership = CARQUET_DATA_VIEW;
    col_data->num_values = child_count;
    col_data->num_lists = num_lists;

    /* Offsets buffer (num_lists + 1). */
    int32_t* offsets = pool_ensure_list_offsets(pool, (size_t)num_lists + 1);
    if (!offsets) { *read_error = true; return; }
    col_data->list_offsets = offsets;

    /* List-level validity: only materialized if some list is null. */
    uint8_t* list_valid = pool_ensure_list_validity(pool, ((size_t)num_lists + 7) / 8 + 1);
    if (!list_valid) { *read_error = true; return; }

    /* Child-element validity: only needed when the element can be null. */
    uint8_t* child_valid = NULL;
    if (elem_optional) {
        child_valid = pool_ensure_bitmap(pool, ((size_t)child_count + 7) / 8 + 1);
        if (!child_valid) { *read_error = true; return; }
    }
    col_data->null_bitmap = child_valid;

    /* Pass 2: build offsets + list validity. */
    int64_t li = -1, cc = 0;
    bool any_list_null = false;
    for (int64_t j = 0; j < slots; j++) {
        if (rep_levels[j] == 0) {
            li++;
            offsets[li] = (int32_t)cc;
            if (def_levels[j] > 0) {
                list_valid[li >> 3] |= (uint8_t)(1u << (li & 7));
            } else {
                any_list_null = true;
            }
        }
        if (def_levels[j] >= elem_exists) cc++;
    }
    offsets[num_lists] = (int32_t)cc;
    col_data->list_validity = any_list_null ? list_valid : NULL;

    /* Pass 3: expand dense (present-only) values into child-slot positions,
     * back-to-front so it can run in place, and build child validity. The
     * reader wrote `child_present` dense values at the front of `data`. */
    if (child_count > 0) {
        int64_t child_present = 0;
        for (int64_t j = 0; j < slots; j++) {
            if (def_levels[j] == max_def) child_present++;
        }
        uint8_t* bytes = (uint8_t*)data;
        /* child index for each element slot, walked back-to-front */
        int64_t ci = child_count - 1;
        int64_t src = child_present - 1;
        for (int64_t j = slots - 1; j >= 0; j--) {
            if (def_levels[j] < elem_exists) continue;  /* not an element */
            bool present = (def_levels[j] == max_def);
            uint8_t* dp = bytes + (size_t)ci * value_size;
            if (present) {
                uint8_t* sp = bytes + (size_t)src * value_size;
                if (sp != dp) memmove(dp, sp, value_size);
                src--;
                if (child_valid) {
                    child_valid[ci >> 3] |= (uint8_t)(1u << (ci & 7));
                }
            } else {
                memset(dp, 0, value_size);
            }
            ci--;
        }
    }
}

/* ============================================================================
 * Column Reader Reset (reuse across row groups)
 * ============================================================================
 */

static void reset_column_reader_for_row_group(
    carquet_column_reader_t* col_reader,
    carquet_reader_t* file_reader,
    int32_t row_group_index,
    int32_t column_index,
    bool preserve_dictionaries) {

    const parquet_row_group_t* rg = &file_reader->metadata.row_groups[row_group_index];

    col_reader->row_group_index = row_group_index;
    col_reader->column_index = column_index;
    col_reader->preserve_dictionary = false;

    if (!rg->columns || column_index >= rg->num_columns) {
        col_reader->chunk = NULL;
        col_reader->col_meta = NULL;
        col_reader->values_remaining = 0;
        return;
    }

    col_reader->chunk = &rg->columns[column_index];
    if (!col_reader->chunk->has_metadata) {
        col_reader->col_meta = NULL;
        col_reader->values_remaining = 0;
        return;
    }
    col_reader->col_meta = &col_reader->chunk->metadata;

    /* Decide dictionary preservation up front, before any page is loaded.
     * Pages are pre-loaded (and decoded/sized) ahead of the read phase, so the
     * decode width (physical value vs. 4-byte index) must be fixed now; keying
     * off has_dictionary instead would only flip after the first load and
     * desync the buffer width from the copy width. */
    col_reader->preserve_dictionary =
        preserve_dictionaries && col_reader->col_meta->has_dictionary_page_offset;

    /* Reset reading state */
    col_reader->values_remaining = col_reader->col_meta->num_values;
    col_reader->data_start_offset = col_reader->col_meta->data_page_offset;
    col_reader->current_page = 0;
    col_reader->page_loaded = false;
    col_reader->page_num_values = 0;
    col_reader->page_values_read = 0;
    col_reader->page_header_size = 0;
    col_reader->page_compressed_size = 0;

    /* Dictionary may differ between row groups - must reload */
    if (col_reader->has_dictionary) {
        if (col_reader->dictionary_ownership == CARQUET_DATA_OWNED) {
            carquet_mem_free(col_reader->dictionary_data);
        }
        carquet_mem_free(col_reader->dictionary_offsets);
        col_reader->dictionary_data = NULL;
        col_reader->dictionary_offsets = NULL;
        col_reader->dictionary_size = 0;
        col_reader->dictionary_count = 0;
        col_reader->dictionary_ownership = CARQUET_DATA_OWNED;
        col_reader->has_dictionary = false;
    }

    /* If decoded_values is a VIEW (mmap pointer), don't free - just clear */
    if (col_reader->decoded_ownership == CARQUET_DATA_VIEW) {
        col_reader->decoded_values = NULL;
        col_reader->decoded_capacity = 0;
    }
    col_reader->decoded_ownership = CARQUET_DATA_OWNED;

    /* Free BYTE_ARRAY page data retention list */
    carquet_column_clear_retained_pages(col_reader);

    /* Keep reusable buffers: decoded_values, decoded_def_levels,
     * decoded_rep_levels, indices_buffer, decompress_buffer.
     * These will be reused on the next page load. */
}

/* ============================================================================
 * Batch Reader Implementation
 * ============================================================================
 */

carquet_batch_reader_t* carquet_batch_reader_create(
    carquet_reader_t* reader,
    const carquet_batch_reader_config_t* config,
    carquet_error_t* error) {

    /* reader is nonnull per API contract */
    carquet_batch_reader_t* batch_reader = carquet_mem_calloc(1, sizeof(carquet_batch_reader_t));
    if (!batch_reader) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate batch reader");
        return NULL;
    }

    batch_reader->reader = reader;

    /* Copy config or use defaults */
    if (config) {
        batch_reader->config = *config;
    } else {
        carquet_batch_reader_config_init(&batch_reader->config);
    }

    /* Resolve column projection */
    int32_t total_columns = carquet_reader_num_columns(reader);

    if (batch_reader->config.column_indices && batch_reader->config.num_columns > 0) {
        /* Use provided column indices */
        batch_reader->num_projected = batch_reader->config.num_columns;
        batch_reader->projected_columns = carquet_mem_malloc(sizeof(int32_t) * batch_reader->num_projected);
        if (!batch_reader->projected_columns) {
            carquet_mem_free(batch_reader);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection");
            return NULL;
        }
        memcpy(batch_reader->projected_columns, batch_reader->config.column_indices,
               sizeof(int32_t) * batch_reader->num_projected);
    } else if (batch_reader->config.column_names && batch_reader->config.num_column_names > 0) {
        /* Resolve column names to indices */
        batch_reader->num_projected = batch_reader->config.num_column_names;
        batch_reader->projected_columns = carquet_mem_malloc(sizeof(int32_t) * batch_reader->num_projected);
        if (!batch_reader->projected_columns) {
            carquet_mem_free(batch_reader);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection");
            return NULL;
        }

        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            const char* col_name = batch_reader->config.column_names[i];
            int32_t idx = resolve_column_name(reader, col_name);
            if (idx < 0) {
                carquet_mem_free(batch_reader->projected_columns);
                carquet_mem_free(batch_reader);
                CARQUET_SET_ERROR(error, CARQUET_ERROR_COLUMN_NOT_FOUND,
                    "Column not found: %s", col_name);
                return NULL;
            }
            batch_reader->projected_columns[i] = idx;
        }
    } else {
        /* Read all columns */
        batch_reader->num_projected = total_columns;
        batch_reader->projected_columns = carquet_mem_malloc(sizeof(int32_t) * total_columns);
        if (!batch_reader->projected_columns) {
            carquet_mem_free(batch_reader);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection");
            return NULL;
        }
        for (int32_t i = 0; i < total_columns; i++) {
            batch_reader->projected_columns[i] = i;
        }
    }

    /* Allocate column reader array */
    batch_reader->col_readers = carquet_mem_calloc(batch_reader->num_projected,
                                        sizeof(carquet_column_reader_t*));
    if (!batch_reader->col_readers) {
        carquet_mem_free(batch_reader->projected_columns);
        carquet_mem_free(batch_reader);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate column readers");
        return NULL;
    }

    batch_reader->projected_types = carquet_mem_malloc(sizeof(carquet_physical_type_t) *
                                           (size_t)batch_reader->num_projected);
    batch_reader->projected_type_lengths = carquet_mem_malloc(sizeof(int32_t) *
                                                  (size_t)batch_reader->num_projected);
    batch_reader->projected_max_defs = carquet_mem_malloc(sizeof(int16_t) *
                                              (size_t)batch_reader->num_projected);
    batch_reader->projected_max_reps = carquet_mem_malloc(sizeof(int16_t) *
                                              (size_t)batch_reader->num_projected);
    batch_reader->projected_value_sizes = carquet_mem_malloc(sizeof(size_t) *
                                                 (size_t)batch_reader->num_projected);
    if (!batch_reader->projected_types || !batch_reader->projected_type_lengths ||
        !batch_reader->projected_max_defs || !batch_reader->projected_max_reps ||
        !batch_reader->projected_value_sizes) {
        carquet_mem_free(batch_reader->projected_value_sizes);
        carquet_mem_free(batch_reader->projected_max_reps);
        carquet_mem_free(batch_reader->projected_max_defs);
        carquet_mem_free(batch_reader->projected_type_lengths);
        carquet_mem_free(batch_reader->projected_types);
        carquet_mem_free(batch_reader->col_readers);
        carquet_mem_free(batch_reader->projected_columns);
        carquet_mem_free(batch_reader);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection metadata");
        return NULL;
    }

    {
        const carquet_schema_t* schema = carquet_reader_schema(reader);
        batch_reader->has_repeated = false;
        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            int32_t file_col_idx = batch_reader->projected_columns[i];
            int32_t schema_idx = schema->leaf_indices[file_col_idx];
            const parquet_schema_element_t* elem = &schema->elements[schema_idx];

            batch_reader->projected_types[i] =
                elem->has_type ? elem->type : CARQUET_PHYSICAL_BYTE_ARRAY;
            batch_reader->projected_type_lengths[i] = elem->type_length;
            batch_reader->projected_max_defs[i] = schema->max_def_levels[file_col_idx];
            batch_reader->projected_max_reps[i] = schema->max_rep_levels[file_col_idx];
            if (batch_reader->projected_max_reps[i] > 0) {
                batch_reader->has_repeated = true;
            }
            batch_reader->projected_value_sizes[i] = get_type_size(
                batch_reader->projected_types[i],
                batch_reader->projected_type_lengths[i]);
        }
    }

    /* Allocate buffer pool (one per projected column) */
    batch_reader->col_pools = carquet_mem_calloc(batch_reader->num_projected,
                                      sizeof(carquet_column_pool_t));
    if (!batch_reader->col_pools) {
        carquet_mem_free(batch_reader->projected_value_sizes);
        carquet_mem_free(batch_reader->projected_max_reps);
        carquet_mem_free(batch_reader->projected_max_defs);
        carquet_mem_free(batch_reader->projected_type_lengths);
        carquet_mem_free(batch_reader->projected_types);
        carquet_mem_free(batch_reader->col_readers);
        carquet_mem_free(batch_reader->projected_columns);
        carquet_mem_free(batch_reader);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate buffer pool");
        return NULL;
    }

    batch_reader->current_row_group = -1;

    /* Filter state init */
    carquet_row_range_list_init(&batch_reader->current_rg_ranges);
    batch_reader->filter_rg_state_valid = false;
    batch_reader->current_range_index = 0;
    batch_reader->current_range_rows_emitted = 0;
    batch_reader->range_positioned = false;
    batch_reader->rows_skipped = 0;
    batch_reader->projected_offset_indexes = NULL;
    batch_reader->projected_oi_rg = -1;

    /* ====================================================================
     * Pre-compute filtered row group order
     * ==================================================================== */
    int32_t num_row_groups = carquet_reader_num_row_groups(reader);
    batch_reader->rg_order = carquet_mem_malloc(sizeof(int32_t) * (size_t)num_row_groups);
    if (!batch_reader->rg_order) {
        batch_reader->rg_order_len = 0;
    } else {
        int32_t count = 0;
        for (int32_t rg = 0; rg < num_row_groups; rg++) {
            if (batch_reader->config.row_group_filter) {
                bool should_read = batch_reader->config.row_group_filter(
                    reader, rg, batch_reader->config.row_group_filter_ctx);
                if (!should_read) continue;
            }
            batch_reader->rg_order[count++] = rg;
        }
        batch_reader->rg_order_len = count;
    }
    batch_reader->rg_order_next = 0;

    /* ====================================================================
     * Create worker pool + pipeline for compressed mmap multi-RG files
     * ==================================================================== */
    /* Enable the pipeline for multi-RG files, or single-RG files that are
     * large enough for the pipeline overhead to be amortized.  Small files
     * (< 500K rows) are faster with the simpler per-batch OMP path. */
    /* row_groups[].num_rows is parsed straight from attacker-controlled
     * metadata, so a crafted file can supply negative or absurd values.
     * Saturate the accumulation (this total only gates a >= 500000 check)
     * and ignore non-positive counts rather than overflowing int64_t. */
    int64_t total_pipeline_rows = 0;
    for (int32_t r = 0; r < batch_reader->rg_order_len; r++) {
        int64_t rg_rows = reader->metadata.row_groups[batch_reader->rg_order[r]].num_rows;
        if (rg_rows <= 0) continue;
        if (rg_rows > INT64_MAX - total_pipeline_rows) {
            total_pipeline_rows = INT64_MAX;
            break;
        }
        total_pipeline_rows += rg_rows;
    }
    bool pipeline_safe = true;
    for (int32_t ci = 0; ci < batch_reader->num_projected; ci++) {
        int32_t file_col = batch_reader->projected_columns[ci];
        if (reader->schema->max_def_levels[file_col] > 0 ||
            reader->schema->max_rep_levels[file_col] > 0) {
            pipeline_safe = false;
            break;
        }
    }

    if (pipeline_safe &&
        reader->mmap_data != NULL &&
        (batch_reader->rg_order_len > 1 || total_pipeline_rows >= 500000)) {
        bool has_compression = false;
        const parquet_file_metadata_t* meta = &reader->metadata;
        if (meta->num_row_groups > 0 && meta->row_groups[0].columns) {
            for (int32_t ci = 0; ci < batch_reader->num_projected; ci++) {
                int32_t file_col = batch_reader->projected_columns[ci];
                if (file_col < meta->row_groups[0].num_columns) {
                    const parquet_column_chunk_t* chunk = &meta->row_groups[0].columns[file_col];
                    if (chunk->has_metadata &&
                        chunk->metadata.codec != CARQUET_COMPRESSION_UNCOMPRESSED) {
                        has_compression = true;
                        break;
                    }
                }
            }
        }

        if (has_compression) {
            /* Optimization 6: borrow external pool if provided */
            if (batch_reader->config.thread_pool) {
                batch_reader->pool = (carquet_worker_pool_t*)batch_reader->config.thread_pool;
                batch_reader->pool_is_borrowed = true;
            } else {
                int32_t pt = batch_reader->config.num_threads;
#ifdef _OPENMP
                if (pt <= 0) pt = omp_get_max_threads();
#else
                if (pt <= 0) pt = 4;
#endif
                if (pt < 2) pt = 2;

                batch_reader->pool = carquet_worker_pool_create(pt);
                batch_reader->pool_is_borrowed = false;
            }

            if (batch_reader->pool) {
                int32_t pt = batch_reader->pool->num_threads;
                int32_t depth = batch_reader->rg_order_len;
                if (depth > pt * 2) depth = pt * 2;
                if (depth < 1) depth = 1;

                batch_reader->pipeline = carquet_mem_calloc(depth, sizeof(rg_slot_t));
                if (batch_reader->pipeline) {
                    batch_reader->pipeline_depth = depth;
                    batch_reader->pipeline_head = 0;
                    batch_reader->pipeline_count = 0;
                    batch_reader->pipeline_active = true;

                    /* Allocate per-reader task args.  The pipeline currently
                     * submits one task per projected column.  Intra-column
                     * splitting needs per-task scratch buffers before it can
                     * safely share column readers. */
                    int32_t np = batch_reader->num_projected;
                    int32_t max_splits = 1;
                    batch_reader->task_args_capacity = depth * np * (max_splits + 1);
                    batch_reader->task_args = carquet_mem_calloc(batch_reader->task_args_capacity,
                                                     sizeof(bulk_read_arg_t));
                    if (!batch_reader->task_args) {
                        carquet_mem_free(batch_reader->pipeline);
                        batch_reader->pipeline = NULL;
                        batch_reader->pipeline_active = false;
                        batch_reader->task_args_capacity = 0;
                    }

                    bool alloc_ok = batch_reader->pipeline_active;
                    for (int32_t s = 0; s < depth && alloc_ok; s++) {
                        batch_reader->pipeline[s].col_readers = carquet_mem_calloc(np, sizeof(carquet_column_reader_t*));
                        batch_reader->pipeline[s].col_values = carquet_mem_calloc(np, sizeof(void*));
                        batch_reader->pipeline[s].col_buf_sizes = carquet_mem_calloc(np, sizeof(size_t));
                        batch_reader->pipeline[s].col_num_values = carquet_mem_calloc(np, sizeof(int64_t));
                        batch_reader->pipeline[s].rg_index = -1;
                        if (!batch_reader->pipeline[s].col_readers ||
                            !batch_reader->pipeline[s].col_values ||
                            !batch_reader->pipeline[s].col_buf_sizes ||
                            !batch_reader->pipeline[s].col_num_values) {
                            /* Cleanup all slots on failure */
                            for (int32_t j = 0; j <= s; j++) {
                                carquet_mem_free(batch_reader->pipeline[j].col_readers);
                                carquet_mem_free(batch_reader->pipeline[j].col_values);
                                carquet_mem_free(batch_reader->pipeline[j].col_buf_sizes);
                                carquet_mem_free(batch_reader->pipeline[j].col_num_values);
                            }
                            carquet_mem_free(batch_reader->pipeline);
                            batch_reader->pipeline = NULL;
                            batch_reader->pipeline_active = false;
                            alloc_ok = false;
                        }
                    }

                    /* Optimization 3: pre-allocate value buffers based on
                     * max row group size so pipeline_fill avoids malloc
                     * storms on the hot path. */
                    if (alloc_ok) {
                        int64_t max_rg_rows = 0;
                        for (int32_t r = 0; r < batch_reader->rg_order_len; r++) {
                            int64_t rr = meta->row_groups[batch_reader->rg_order[r]].num_rows;
                            if (rr > max_rg_rows) max_rg_rows = rr;
                        }
                        /* This is a hot-path optimization only: a NULL/zero
                         * slot buffer is lazily (re)allocated by pipeline_fill.
                         * num_rows is attacker-controlled metadata, so skip the
                         * pre-allocation entirely when the size is non-positive,
                         * overflows size_t, or is implausibly large rather than
                         * trusting it into carquet_mem_malloc(). */
                        if (max_rg_rows > 0) {
                            for (int32_t s = 0; s < depth; s++) {
                                rg_slot_t* slot = &batch_reader->pipeline[s];
                                for (int32_t i = 0; i < np; i++) {
                                    size_t vsize = batch_reader->projected_value_sizes[i];
                                    if (vsize == 0 ||
                                        (uint64_t)max_rg_rows > SIZE_MAX / vsize) {
                                        continue;
                                    }
                                    size_t needed = (size_t)max_rg_rows * vsize;
                                    if (needed == 0 ||
                                        needed > CARQUET_MAX_PREALLOC_BYTES) {
                                        continue;
                                    }
                                    slot->col_values[i] = carquet_mem_malloc(needed);
                                    slot->col_buf_sizes[i] =
                                        slot->col_values[i] ? needed : 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return batch_reader;
}

static carquet_status_t open_row_group_readers(
    carquet_batch_reader_t* batch_reader,
    int32_t row_group_index,
    carquet_error_t* error) {

    /* Reuse existing readers if possible, otherwise create new ones */
    for (int32_t i = 0; i < batch_reader->num_projected; i++) {
        int32_t file_col_idx = batch_reader->projected_columns[i];

        if (batch_reader->col_readers[i]) {
            /* Reuse: reset state but keep allocated buffers */
            reset_column_reader_for_row_group(
                batch_reader->col_readers[i],
                batch_reader->reader,
                row_group_index, file_col_idx,
                batch_reader->config.preserve_dictionaries);
        } else {
            /* First time: create new reader */
            batch_reader->col_readers[i] = carquet_reader_get_column(
                batch_reader->reader, row_group_index, file_col_idx, error);

            if (!batch_reader->col_readers[i]) {
                /* Close already opened readers */
                for (int32_t j = 0; j < i; j++) {
                    carquet_column_reader_free(batch_reader->col_readers[j]);
                    batch_reader->col_readers[j] = NULL;
                }
                return error ? error->code : CARQUET_ERROR_COLUMN_NOT_FOUND;
            }
        }

        /* Decide dictionary preservation up front for BOTH the reset and the
         * freshly-created reader, before any page is pre-loaded. The decode
         * buffer width depends on this (physical value vs. 4-byte index), and
         * pages are pre-loaded ahead of the read phase, so it must be fixed
         * now rather than at read time. */
        carquet_column_reader_t* cr = batch_reader->col_readers[i];
        cr->preserve_dictionary = batch_reader->config.preserve_dictionaries &&
            cr->col_meta && cr->col_meta->has_dictionary_page_offset;
    }

    batch_reader->current_row_group = row_group_index;
    batch_reader->rows_read_in_group = 0;

    return CARQUET_OK;
}

/* ============================================================================
 * Pipeline Ring Buffer
 * ============================================================================
 *
 * Pre-decompresses pages for multiple row groups in parallel using the worker
 * pool. On the first call to batch_reader_next(), decompression tasks are
 * submitted for up to pipeline_depth row groups. As the user consumes
 * batches and exhausts a row group, that slot is retired and the next
 * uncovered RG is submitted. With pipeline_depth >= total_RGs (the common
 * benchmark case), ALL decompression happens upfront in parallel.
 */

/**
 * Bulk-read task: reads ALL values from a column reader into a pre-allocated
 * buffer. This forces decompression of ALL pages in the column chunk.
 */
static void bulk_read_task(void* arg) {
    bulk_read_arg_t* t = (bulk_read_arg_t*)arg;
    if (!t->col_reader || !t->dest || t->max_values <= 0) {
        *t->out_values_read = 0;
        return;
    }

    /* ------------------------------------------------------------------
     * Filtered branch: read only matching pages for this column, writing
     * each range's rows contiguously into the slot buffer.
     *
     * When this column has an offset index we seek by file offset to the
     * page covering each range. When it doesn't (e.g. an externally
     * written file that supplied a page index for the predicate column
     * but not for this one), we degrade to monotonic read-and-discard:
     * skip the gap between the previous range end and the next range
     * start, then read the range's row count. This is the same fallback
     * the sequential filtered path uses (§6.5 of the design doc).
     * ------------------------------------------------------------------ */
    if (t->ranges && t->ranges->count > 0 && t->value_size > 0) {
        carquet_error_t err = CARQUET_ERROR_INIT;
        size_t dest_offset_bytes = 0;
        int64_t total_read = 0;
        int64_t cursor_row = 0;   /* logical row position used by the
                                   * no-offset-index fallback. */
        for (int32_t r = 0; r < t->ranges->count; r++) {
            int64_t first = t->ranges->ranges[r].first_row;
            int64_t num = t->ranges->ranges[r].num_rows;
            if (num <= 0) continue;

            if (t->offset_index) {
                int64_t page_first_row = 0;
                int32_t page_idx = find_page_for_row(
                    t->offset_index, t->rg_num_rows, first, &page_first_row);
                if (page_idx < 0) break;

                carquet_page_location_t loc;
                if (carquet_offset_index_get_page_location(
                        t->offset_index, page_idx, &loc) != CARQUET_OK) {
                    break;
                }

                if (carquet_column_reader_seek_to_data_page(
                        t->col_reader, loc.offset, 0, &err) != CARQUET_OK) {
                    break;
                }

                int64_t intra_skip = first - page_first_row;
                if (intra_skip > 0) {
                    int64_t skipped = carquet_column_skip(
                        t->col_reader, intra_skip);
                    if (skipped != intra_skip) break;
                }
            } else {
                /* Forward read-and-discard from cursor to range start. */
                int64_t gap = first - cursor_row;
                if (gap > 0) {
                    int64_t skipped = carquet_column_skip(t->col_reader, gap);
                    if (skipped != gap) break;
                }
            }

            uint8_t* dest_ptr = (uint8_t*)t->dest + dest_offset_bytes;
            int64_t got = carquet_column_read_batch(
                t->col_reader, dest_ptr, num, NULL, NULL);
            if (got != num) break;
            dest_offset_bytes += (size_t)num * t->value_size;
            total_read += num;
            cursor_row = first + num;
        }
        *t->out_values_read = total_read;
        return;
    }

    /* ------------------------------------------------------------------
     * Unfiltered branch: existing fast paths.
     * ------------------------------------------------------------------ */
    if (can_coalesce_column(t->col_reader)) {
        if (t->start_offset > 0 && t->end_offset > t->start_offset) {
            coalesced_read_column_range(t->col_reader, t->data_base,
                                        t->dest, t->max_values,
                                        t->start_offset, t->end_offset,
                                        t->out_values_read);
        } else {
            coalesced_read_column(t->col_reader, t->dest, t->max_values,
                                  t->out_values_read);
        }
    } else {
        *t->out_values_read = carquet_column_read_batch(
            t->col_reader, t->dest, t->max_values, NULL, NULL);
    }
}

/* ============================================================================
 * Coalesced Column Chunk Read — Fast Path for Pipeline Mode
 * ============================================================================
 * Instead of the page-by-page carquet_column_read_batch path, this scans all
 * page headers upfront and decompresses in a tight loop, writing directly to
 * the output buffer. Eliminates intermediate buffers and per-page state machine
 * overhead.
 *
 * Eligible: REQUIRED columns, fixed-width, PLAIN or BYTE_STREAM_SPLIT encoding,
 * no dictionary, compressed, mmap available.
 */

static bool can_coalesce_column(const carquet_column_reader_t* cr) {
    if (!cr || !cr->col_meta || !cr->file_reader || !cr->file_reader->mmap_data) return false;
    if (cr->max_def_level > 0 || cr->max_rep_level > 0) return false;
    if (cr->type == CARQUET_PHYSICAL_BOOLEAN || cr->type == CARQUET_PHYSICAL_BYTE_ARRAY) return false;
    if (cr->col_meta->has_dictionary_page_offset) return false;
    if (cr->col_meta->codec == CARQUET_COMPRESSION_UNCOMPRESSED) return false;
    return true;
}

static void coalesced_read_column_range(
    const carquet_column_reader_t* cr,
    const uint8_t* data_base,
    void* dest,
    int64_t max_values,
    int64_t start_offset,
    int64_t end_offset,
    int64_t* out_values_read) {

    const carquet_reader_t* fr = cr->file_reader;
    carquet_column_reader_t* reader = (carquet_column_reader_t*)cr;
    const parquet_column_metadata_t* meta = cr->col_meta;
    size_t file_size = fr->file_size;

    size_t value_size = 0;
    switch (cr->type) {
        case CARQUET_PHYSICAL_INT32: case CARQUET_PHYSICAL_FLOAT: value_size = 4; break;
        case CARQUET_PHYSICAL_INT64: case CARQUET_PHYSICAL_DOUBLE: value_size = 8; break;
        case CARQUET_PHYSICAL_INT96: value_size = 12; break;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: value_size = (size_t)cr->type_length; break;
        default: goto fallback;
    }

    int64_t offset = start_offset;
    int64_t chunk_end = end_offset;
    if (offset <= 0) offset = meta->data_page_offset;
    if (chunk_end <= 0) chunk_end = meta->data_page_offset + meta->total_compressed_size;
    if (offset < meta->data_page_offset || chunk_end > meta->data_page_offset + meta->total_compressed_size) {
        goto fallback;
    }
    if (offset < 0 || chunk_end > (int64_t)file_size) goto fallback;

    uint8_t* out = (uint8_t*)dest;
    int64_t total_values = 0;

    while (offset < chunk_end && total_values < max_values) {
        /* Parse page header */
        const uint8_t* ptr = data_base + offset;
        size_t remaining = (size_t)(chunk_end - offset);
        size_t max_hdr = remaining < 512 ? remaining : 512;

        parquet_page_header_t hdr;
        size_t hdr_size;
        if (parquet_parse_page_header(ptr, max_hdr, &hdr, &hdr_size, NULL) != CARQUET_OK)
            goto fallback;

        if (hdr.type != CARQUET_PAGE_DATA)
            goto fallback;

        int32_t num_values = hdr.data_page_header.num_values;
        carquet_encoding_t encoding = hdr.data_page_header.encoding;

        if (num_values <= 0 || total_values + num_values > max_values) goto fallback;

        const uint8_t* compressed = ptr + hdr_size;
        size_t comp_size = (size_t)hdr.compressed_page_size;
        size_t uncomp_size = (size_t)hdr.uncompressed_page_size;
        if (hdr.compressed_page_size <= 0 ||
            hdr.uncompressed_page_size < 0 ||
            comp_size > CARQUET_MAX_PAGE_PAYLOAD_SIZE ||
            uncomp_size > CARQUET_MAX_PAGE_PAYLOAD_SIZE ||
            hdr_size > remaining ||
            comp_size > remaining - hdr_size) {
            goto fallback;
        }

        if (encoding == CARQUET_ENCODING_PLAIN) {
            /* Decompress directly to output — data IS the final values */
            size_t actual;
            if (carquet_decompress_page(meta->codec, compressed, comp_size,
                                         out, uncomp_size, &actual) != CARQUET_OK)
                break;
        } else if (encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT) {
            /* Decompress to temp, then cache-tiled transpose to output */
            if (uncomp_size > reader->decompress_capacity) {
                uint8_t* new_buf = carquet_mem_realloc(reader->decompress_buffer, uncomp_size);
                if (!new_buf) {
                    break;
                }
                reader->decompress_buffer = new_buf;
                reader->decompress_capacity = uncomp_size;
            }
            size_t actual;
            if (carquet_decompress_page(meta->codec, compressed, comp_size,
                                         reader->decompress_buffer, uncomp_size, &actual) != CARQUET_OK)
                break;
            if (value_size == 4) {
                if (carquet_byte_stream_split_decode_float(
                        reader->decompress_buffer, actual, (float*)out, num_values) != CARQUET_OK) {
                    break;
                }
            } else {
                if (carquet_byte_stream_split_decode_double(
                        reader->decompress_buffer, actual, (double*)out, num_values) != CARQUET_OK) {
                    break;
                }
            }
        } else {
            /* Unsupported encoding — bail to fallback */
            goto fallback;
        }

        out += (size_t)num_values * value_size;
        total_values += num_values;
        offset += (int64_t)hdr_size + (int64_t)comp_size;
    }

    *out_values_read = total_values;
    return;

fallback:
    /* Fall back to standard page-by-page reader */
    *out_values_read = carquet_column_read_batch(
        (carquet_column_reader_t*)cr, dest, max_values, NULL, NULL);
}

static void coalesced_read_column(
    const carquet_column_reader_t* cr,
    void* dest,
    int64_t max_values,
    int64_t* out_values_read) {
    coalesced_read_column_range(cr, cr->file_reader->mmap_data,
                                dest, max_values, 0, 0, out_values_read);
}

/**
 * Plan N-way split of a column chunk for parallel decompression.
 * Walks page headers to find page-aligned boundaries that divide the chunk
 * into roughly equal pieces (by value count).
 *
 * @param cr          Column reader (must pass can_coalesce_column)
 * @param max_values  Total values in the column chunk
 * @param max_splits  Maximum number of splits to produce (>= 2)
 * @param split_offsets  Output: [max_splits+1] byte offsets (start of each segment + end)
 * @param split_values   Output: [max_splits+1] cumulative value counts at each boundary
 * @return Number of segments (>= 2 on success, 0 if split not possible)
 */
static int32_t plan_coalesced_column_splits(
    const carquet_column_reader_t* cr,
    const uint8_t* data_base,
    int64_t max_values,
    int32_t max_splits,
    int64_t* split_offsets,
    int64_t* split_values) {

    if (!split_offsets || !split_values || !can_coalesce_column(cr)) return 0;
    if (max_splits < 2) return 0;
    if (cr->col_meta->codec != CARQUET_COMPRESSION_ZSTD &&
        cr->col_meta->codec != CARQUET_COMPRESSION_LZ4 &&
        cr->col_meta->codec != CARQUET_COMPRESSION_LZ4_RAW) {
        return 0;
    }
    if (max_values < 100000) return 0;

    const carquet_reader_t* fr = cr->file_reader;
    const parquet_column_metadata_t* meta = cr->col_meta;
    int64_t offset = meta->data_page_offset;
    int64_t chunk_end = offset + meta->total_compressed_size;
    if (offset < 0 || chunk_end > (int64_t)fr->file_size) return 0;

    /* First segment starts at the beginning */
    split_offsets[0] = offset;
    split_values[0] = 0;
    int32_t num_segments = 1;

    int64_t target_per_split = max_values / max_splits;
    /* Need at least 50k values per segment to be worthwhile */
    if (target_per_split < 50000) {
        max_splits = (int32_t)(max_values / 50000);
        if (max_splits < 2) return 0;
        target_per_split = max_values / max_splits;
    }
    int64_t next_target = target_per_split;
    int64_t values_so_far = 0;

    while (offset < chunk_end) {
        const uint8_t* ptr = data_base + offset;
        size_t remaining = (size_t)(chunk_end - offset);
        size_t max_hdr = remaining < 512 ? remaining : 512;

        parquet_page_header_t hdr;
        size_t hdr_size;
        if (parquet_parse_page_header(ptr, max_hdr, &hdr, &hdr_size, NULL) != CARQUET_OK)
            return 0;

        if (hdr.type != CARQUET_PAGE_DATA)
            return 0;

        if (hdr.compressed_page_size <= 0 ||
            hdr.uncompressed_page_size < 0 ||
            (size_t)hdr.compressed_page_size > CARQUET_MAX_PAGE_PAYLOAD_SIZE ||
            (size_t)hdr.uncompressed_page_size > CARQUET_MAX_PAGE_PAYLOAD_SIZE ||
            hdr_size > remaining ||
            (size_t)hdr.compressed_page_size > remaining - hdr_size) {
            return 0;
        }

        int32_t num_values = hdr.data_page_header.num_values;
        if (num_values <= 0) return 0;

        values_so_far += num_values;
        offset += (int64_t)hdr_size + (int64_t)hdr.compressed_page_size;

        /* Place a split boundary when we've accumulated enough values,
         * but only if there's still data remaining for the next segment. */
        if (values_so_far >= next_target && offset < chunk_end &&
            num_segments < max_splits) {
            split_offsets[num_segments] = offset;
            split_values[num_segments] = values_so_far;
            num_segments++;
            next_target = values_so_far + target_per_split;
        }
    }

    /* Close the final segment */
    split_offsets[num_segments] = chunk_end;
    split_values[num_segments] = values_so_far;

    return (num_segments >= 2) ? num_segments : 0;
}

static void slot_release_filter_state(rg_slot_t* slot, int32_t num_projected) {
    if (slot->col_offset_indexes) {
        for (int32_t i = 0; i < num_projected; i++) {
            if (slot->col_offset_indexes[i]) {
                carquet_offset_index_free(slot->col_offset_indexes[i]);
                slot->col_offset_indexes[i] = NULL;
            }
        }
    }
    if (slot->filter_ranges_valid) {
        carquet_row_range_list_destroy(&slot->ranges);
        slot->filter_ranges_valid = false;
    }
}

/**
 * Fill pipeline slots by reading entire column chunks in parallel.
 * Each task reads ALL values from one column in one row group.
 *
 * When a page filter is active, each slot's row-range list is computed
 * up front; the worker tasks read only matching pages, sized to the
 * range total rather than the whole row group. Row groups that match no
 * rows are skipped without consuming a pipeline slot (their rows are
 * still credited to rows_skipped).
 */
static void pipeline_fill(carquet_batch_reader_t* br) {
    if (!br->pipeline_active || !br->pool) return;

    bool filter_active =
        br->filter_clauses != NULL && br->filter_clause_count > 0;

    while (br->pipeline_count < br->pipeline_depth &&
           br->rg_order_next < br->rg_order_len) {

        int32_t slot_idx = (br->pipeline_head + br->pipeline_count) % br->pipeline_depth;
        rg_slot_t* slot = &br->pipeline[slot_idx];
        int32_t target_rg = br->rg_order[br->rg_order_next];

        /* Get row count for this RG */
        const parquet_row_group_t* rg = &br->reader->metadata.row_groups[target_rg];
        int64_t rg_rows = rg->num_rows;

        /* Page filter evaluation: skip whole row groups that match nothing,
         * and clip per-slot allocations to the matching row count. */
        slot_release_filter_state(slot, br->num_projected);
        int64_t slot_rows = rg_rows;
        if (filter_active) {
            carquet_error_t feval_err = CARQUET_ERROR_INIT;
            carquet_row_range_list_init(&slot->ranges);
            carquet_status_t fst = carquet_page_filter_eval_row_group(
                br->reader, target_rg,
                br->filter_clauses, br->filter_clause_count,
                &slot->ranges, &feval_err);
            if (fst != CARQUET_OK) {
                /* Surface the error on the next batch_reader_next() by
                 * leaving the slot empty and advancing past the row group. */
                carquet_row_range_list_destroy(&slot->ranges);
                br->rg_order_next++;
                continue;
            }
            slot->filter_ranges_valid = true;

            br->rows_skipped += rg_rows - slot->ranges.total_rows;
            if (slot->ranges.count == 0) {
                /* No rows from this RG; do not occupy a pipeline slot. */
                carquet_row_range_list_destroy(&slot->ranges);
                slot->filter_ranges_valid = false;
                br->rg_order_next++;
                continue;
            }
            slot_rows = slot->ranges.total_rows;

            /* Load per-column offset indexes so worker tasks can seek
             * directly to matching pages. A NULL entry is allowed: that
             * column simply falls back to read-and-discard skip in the
             * worker (this can happen with externally-written files that
             * supplied a page index for the predicate column but not for
             * every projected column). */
            if (!slot->col_offset_indexes) {
                slot->col_offset_indexes = carquet_mem_calloc(
                    (size_t)br->num_projected,
                    sizeof(carquet_offset_index_t*));
                if (!slot->col_offset_indexes) return;
            }
            for (int32_t i = 0; i < br->num_projected; i++) {
                carquet_error_t oi_err = CARQUET_ERROR_INIT;
                slot->col_offset_indexes[i] = carquet_reader_get_offset_index(
                    br->reader, target_rg, br->projected_columns[i], &oi_err);
                /* NULL is fine — handled by the worker fallback. */
            }
        }

        /* A row group physically cannot contain more rows than the file has
         * bits (the densest encoding is 1 bit/row), so a num_rows beyond
         * file_size*8 is malformed. Reject it before sizing per-column buffers
         * so a tiny crafted file can't claim billions of rows and drive a
         * multi-hundred-GB allocation (memory-exhaustion DoS). */
        if (br->reader->file_size > 0 &&
            (uint64_t)slot_rows > (uint64_t)br->reader->file_size * 8u) {
            return;
        }

        /* Ensure column readers exist and are reset for this slot */
        carquet_error_t err = CARQUET_ERROR_INIT;
        for (int32_t i = 0; i < br->num_projected; i++) {
            int32_t file_col_idx = br->projected_columns[i];
            if (slot->col_readers[i]) {
                reset_column_reader_for_row_group(
                    slot->col_readers[i], br->reader,
                    target_rg, file_col_idx,
                    br->config.preserve_dictionaries);
            } else {
                slot->col_readers[i] = carquet_reader_get_column(
                    br->reader, target_rg, file_col_idx, &err);
                if (!slot->col_readers[i]) {
                    return;
                }
            }

            /* Ensure value buffer is large enough (grow-only via realloc).
             * slot_rows derives from row-group metadata (num_rows), which is
             * attacker-controlled: guard against a negative count and against
             * size_t overflow in the multiply so a malformed file cannot drive
             * a wrapped-around (or absurd) allocation. */
            size_t vsz = br->projected_value_sizes[i];
            if (slot_rows < 0 ||
                (vsz != 0 && (uint64_t)slot_rows > (uint64_t)(SIZE_MAX / vsz))) {
                return;
            }
            size_t needed = (size_t)slot_rows * vsz;
            if (needed > slot->col_buf_sizes[i]) {
                void* new_buf = carquet_mem_realloc(slot->col_values[i], needed);
                if (!new_buf) return;
                slot->col_values[i] = new_buf;
                slot->col_buf_sizes[i] = needed;
            }
        }

        slot->rg_index = target_rg;
        slot->ready = false;
        slot->total_rows = slot_rows;
        slot->rows_consumed = 0;

        /* Create an independent mmap for this row group's byte range.
         * Each slot gets its own virtual mapping, so worker threads fault
         * pages into independent page tables without contending on the
         * shared mmap's page table lock.  Falls back to the shared mmap
         * if the per-slot mmap fails. */
        const uint8_t* slot_data = br->reader->mmap_data;  /* fallback */
#if !defined(_WIN32)
        if (br->reader->mmap_info && br->reader->mmap_info->fd >= 0) {
            /* Find byte range spanning all projected column chunks */
            int64_t range_lo = INT64_MAX, range_hi = 0;
            for (int32_t i = 0; i < br->num_projected; i++) {
                const parquet_column_metadata_t* cmeta = slot->col_readers[i]->col_meta;
                if (cmeta && cmeta->total_compressed_size > 0 && cmeta->data_page_offset >= 0) {
                    int64_t lo = cmeta->data_page_offset;
                    int64_t hi = lo + cmeta->total_compressed_size;
                    if (lo < range_lo) range_lo = lo;
                    if (hi > range_hi) range_hi = hi;
                }
            }
            if (range_lo < range_hi) {
                /* Page-align the offset for mmap. mmap requires the offset to be
                 * a multiple of the system page size, which is 16K on Apple
                 * Silicon and up to 64K on some Linux arm64/ppc64 configs — a
                 * hardcoded 4K mask would EINVAL there. Query it at runtime. */
                long ps = sysconf(_SC_PAGESIZE);
                int64_t page_mask = (ps > 0) ? (int64_t)ps - 1 : (int64_t)4095;
                int64_t page_lo = range_lo & ~page_mask;
                size_t mmap_len = (size_t)(range_hi - page_lo);
                uint8_t* m = (uint8_t*)mmap(NULL, mmap_len, PROT_READ, MAP_PRIVATE,
                                             br->reader->mmap_info->fd, (off_t)page_lo);
                if (m != MAP_FAILED) {
                    /* Unmap previous slot mmap if it exists (reuse across fills) */
                    if (slot->slot_mmap && slot->slot_mmap_size > 0)
                        munmap(slot->slot_mmap, slot->slot_mmap_size);
                    slot->slot_mmap = m;
                    slot->slot_mmap_size = mmap_len;
                    slot->slot_mmap_offset = page_lo;
                    /* data_base[file_offset] = slot_mmap[file_offset - page_lo]
                     * so data_base = slot_mmap - page_lo */
                    slot_data = m - page_lo;
                }
            }
        }
#endif

        /* Submit one task per compressed column.  Sharing a column reader
         * across split tasks would race on its reusable decompression buffer. */
        int32_t max_splits_per_col = 1;

        /* Bound by task_args space available for this pipeline slot */
        int32_t tasks_per_slot = br->task_args_capacity / (br->pipeline_depth > 0 ? br->pipeline_depth : 1);
        int32_t base = br->pipeline_count * tasks_per_slot;
        int32_t task_offset = 0;

        int64_t split_offsets[513];
        int64_t split_values_arr[513];

        for (int32_t i = 0; i < br->num_projected; i++) {
            /* Filtered branch: one task per column, reads matching pages
             * only via the cached offset index. Splitting is not used —
             * range-skipping already constrains the work. */
            if (filter_active) {
                int32_t tidx = base + task_offset++;
                if (tidx >= br->task_args_capacity) break;
                br->task_args[tidx].col_reader = slot->col_readers[i];
                br->task_args[tidx].data_base = slot_data;
                br->task_args[tidx].dest = slot->col_values[i];
                br->task_args[tidx].max_values = slot_rows;
                br->task_args[tidx].out_values_read = &slot->col_num_values[i];
                br->task_args[tidx].start_offset = 0;
                br->task_args[tidx].end_offset = 0;
                br->task_args[tidx].local_values_read = 0;
                br->task_args[tidx].ranges = &slot->ranges;
                br->task_args[tidx].offset_index = slot->col_offset_indexes[i];
                br->task_args[tidx].rg_num_rows = rg_rows;
                br->task_args[tidx].value_size = br->projected_value_sizes[i];
                slot->col_num_values[i] = slot_rows;
                carquet_worker_pool_submit(br->pool, bulk_read_task,
                                           &br->task_args[tidx]);
                continue;
            }

            int32_t nseg = plan_coalesced_column_splits(
                slot->col_readers[i], slot_data,
                rg_rows, max_splits_per_col,
                split_offsets, split_values_arr);

            if (nseg >= 2 && base + task_offset + nseg <= br->task_args_capacity) {
                size_t value_size = br->projected_value_sizes[i];
                slot->col_num_values[i] = rg_rows;

                for (int32_t s = 0; s < nseg; s++) {
                    int32_t tidx = base + task_offset++;
                    int64_t seg_start_val = split_values_arr[s];
                    int64_t seg_end_val = split_values_arr[s + 1];

                    br->task_args[tidx].col_reader = slot->col_readers[i];
                    br->task_args[tidx].data_base = slot_data;
                    br->task_args[tidx].dest = (uint8_t*)slot->col_values[i] +
                                               (size_t)seg_start_val * value_size;
                    br->task_args[tidx].max_values = seg_end_val - seg_start_val;
                    br->task_args[tidx].out_values_read = &br->task_args[tidx].local_values_read;
                    br->task_args[tidx].start_offset = split_offsets[s];
                    br->task_args[tidx].end_offset = split_offsets[s + 1];
                    br->task_args[tidx].local_values_read = 0;
                    br->task_args[tidx].ranges = NULL;
                    br->task_args[tidx].offset_index = NULL;
                    carquet_worker_pool_submit(br->pool, bulk_read_task,
                                               &br->task_args[tidx]);
                }
            } else {
                int32_t tidx = base + task_offset++;
                if (tidx >= br->task_args_capacity) break;
                br->task_args[tidx].col_reader = slot->col_readers[i];
                br->task_args[tidx].data_base = slot_data;
                br->task_args[tidx].dest = slot->col_values[i];
                br->task_args[tidx].max_values = rg_rows;
                br->task_args[tidx].out_values_read = &slot->col_num_values[i];
                br->task_args[tidx].start_offset = 0;
                br->task_args[tidx].end_offset = 0;
                br->task_args[tidx].local_values_read = 0;
                br->task_args[tidx].ranges = NULL;
                br->task_args[tidx].offset_index = NULL;
                carquet_worker_pool_submit(br->pool, bulk_read_task,
                                           &br->task_args[tidx]);
            }
        }

        br->pipeline_count++;
        br->rg_order_next++;
    }
}

/* ============================================================================
 * Page Filter — internal helpers
 * ============================================================================ */

static void filter_release_offset_indexes(carquet_batch_reader_t* br) {
    if (!br->projected_offset_indexes) return;
    for (int32_t i = 0; i < br->num_projected; i++) {
        if (br->projected_offset_indexes[i]) {
            carquet_offset_index_free(br->projected_offset_indexes[i]);
            br->projected_offset_indexes[i] = NULL;
        }
    }
    br->projected_oi_rg = -1;
}

static carquet_status_t filter_load_offset_indexes(
    carquet_batch_reader_t* br, int32_t row_group_index,
    carquet_error_t* error) {

    if (br->projected_oi_rg == row_group_index &&
        br->projected_offset_indexes != NULL) {
        return CARQUET_OK;
    }
    filter_release_offset_indexes(br);
    if (!br->projected_offset_indexes) {
        br->projected_offset_indexes = carquet_mem_calloc(
            (size_t)br->num_projected, sizeof(carquet_offset_index_t*));
        if (!br->projected_offset_indexes) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY,
                "Failed to allocate offset index cache");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
    }
    for (int32_t i = 0; i < br->num_projected; i++) {
        carquet_error_t local = CARQUET_ERROR_INIT;
        br->projected_offset_indexes[i] = carquet_reader_get_offset_index(
            br->reader, row_group_index, br->projected_columns[i], &local);
        /* NULL is allowed (no offset index ⇒ fallback read-and-discard); the
         * skip path is only taken for that column. */
    }
    br->projected_oi_rg = row_group_index;
    return CARQUET_OK;
}

/* Returns the page index in oi that contains logical row `target_row`, or
 * -1 if not found. Sets *page_first_row to that page's first_row_index. */
static int32_t find_page_for_row(
    const carquet_offset_index_t* oi,
    int64_t row_group_num_rows,
    int64_t target_row,
    int64_t* page_first_row_out) {

    int32_t n = carquet_offset_index_num_pages(oi);
    /* Binary search: pages are sorted by first_row_index. */
    int32_t lo = 0, hi = n - 1;
    while (lo <= hi) {
        int32_t mid = lo + (hi - lo) / 2;
        carquet_page_location_t loc;
        if (carquet_offset_index_get_page_location(oi, mid, &loc) != CARQUET_OK) {
            return -1;
        }
        int64_t end_row;
        if (mid + 1 < n) {
            carquet_page_location_t nxt;
            if (carquet_offset_index_get_page_location(oi, mid + 1, &nxt) !=
                CARQUET_OK) {
                return -1;
            }
            end_row = nxt.first_row_index;
        } else {
            end_row = row_group_num_rows;
        }
        if (target_row < loc.first_row_index) {
            hi = mid - 1;
        } else if (target_row >= end_row) {
            lo = mid + 1;
        } else {
            *page_first_row_out = loc.first_row_index;
            return mid;
        }
    }
    return -1;
}

static carquet_status_t position_projected_column(
    carquet_batch_reader_t* br,
    int32_t pi,
    int64_t target_row,
    carquet_error_t* error) {

    carquet_column_reader_t* cr = br->col_readers[pi];
    int32_t file_col = br->projected_columns[pi];
    int64_t rg_num_rows = br->reader->metadata.row_groups[
        br->current_row_group].num_rows;

    carquet_offset_index_t* oi = br->projected_offset_indexes
        ? br->projected_offset_indexes[pi] : NULL;

    if (oi) {
        int64_t page_first_row = 0;
        int32_t page_idx = find_page_for_row(oi, rg_num_rows, target_row,
                                             &page_first_row);
        if (page_idx < 0) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_INTERNAL,
                "Could not locate page for row %lld in column %d",
                (long long)target_row, file_col);
            return CARQUET_ERROR_INTERNAL;
        }
        carquet_page_location_t loc;
        (void)carquet_offset_index_get_page_location(oi, page_idx, &loc);

        carquet_status_t st = carquet_column_reader_seek_to_data_page(
            cr, loc.offset, 0, error);
        if (st != CARQUET_OK) return st;

        int64_t intra_skip = target_row - page_first_row;
        if (intra_skip > 0) {
            int64_t skipped = carquet_column_skip(cr, intra_skip);
            if (skipped != intra_skip) {
                CARQUET_SET_ERROR(error, CARQUET_ERROR_INTERNAL,
                    "Intra-page skip short (column %d): asked %lld, got %lld",
                    file_col, (long long)intra_skip, (long long)skipped);
                return CARQUET_ERROR_INTERNAL;
            }
        }
        return CARQUET_OK;
    }

    /* No offset index for this column: reset to chunk start and read-and-
     * discard up to target_row. Forward-only — backward seeks are handled
     * by the reset. */
    reset_column_reader_for_row_group(cr, br->reader,
        br->current_row_group, file_col,
        br->config.preserve_dictionaries);
    if (target_row > 0) {
        int64_t skipped = carquet_column_skip(cr, target_row);
        if (skipped != target_row) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_INTERNAL,
                "Fallback skip short (column %d): asked %lld, got %lld",
                file_col, (long long)target_row, (long long)skipped);
            return CARQUET_ERROR_INTERNAL;
        }
    }
    return CARQUET_OK;
}

/**
 * Advance to a row group that survives both the user's row-group filter
 * and the active page filter. On success, the batch reader is positioned
 * at the first non-empty range of that row group, or returns
 * CARQUET_ERROR_END_OF_DATA when no more matching rows exist.
 */
static carquet_status_t filter_advance_to_active_row_group(
    carquet_batch_reader_t* br, carquet_error_t* error) {

    int32_t num_row_groups = carquet_reader_num_row_groups(br->reader);
    for (;;) {
        if (br->current_row_group < 0) {
            br->current_row_group = 0;
        } else if (!br->filter_rg_state_valid ||
                   br->current_range_index >= br->current_rg_ranges.count) {
            br->current_row_group++;
            br->filter_rg_state_valid = false;
        }
        if (br->current_row_group >= num_row_groups) {
            return CARQUET_ERROR_END_OF_DATA;
        }

        if (br->config.row_group_filter) {
            bool keep = br->config.row_group_filter(br->reader,
                br->current_row_group, br->config.row_group_filter_ctx);
            if (!keep) {
                /* Move on without counting rows toward rows_skipped (user-
                 * level RG filter, not page filter). */
                br->filter_rg_state_valid = false;
                br->current_range_index = br->current_rg_ranges.count;
                continue;
            }
        }

        if (!br->filter_rg_state_valid) {
            carquet_status_t st = carquet_page_filter_eval_row_group(
                br->reader, br->current_row_group,
                br->filter_clauses, br->filter_clause_count,
                &br->current_rg_ranges, error);
            if (st != CARQUET_OK) return st;
            br->filter_rg_state_valid = true;
            br->current_range_index = 0;
            br->current_range_rows_emitted = 0;
            br->range_positioned = false;

            int64_t rg_rows = br->reader->metadata.row_groups[
                br->current_row_group].num_rows;
            br->rows_skipped += rg_rows - br->current_rg_ranges.total_rows;

            /* Open the row group's column readers if we'll need them. */
            if (br->current_rg_ranges.count > 0) {
                carquet_status_t open_st = open_row_group_readers(
                    br, br->current_row_group, error);
                if (open_st != CARQUET_OK) return open_st;
                open_st = filter_load_offset_indexes(br,
                    br->current_row_group, error);
                if (open_st != CARQUET_OK) return open_st;
            } else {
                filter_release_offset_indexes(br);
            }
        }

        if (br->current_range_index < br->current_rg_ranges.count) {
            return CARQUET_OK;
        }
        /* Empty row group ⇒ try the next. */
    }
}

/**
 * Sequential next() path with an active page filter. Reads one range
 * (clipped to batch_size) per call, advancing range/row-group state.
 */
static carquet_status_t batch_reader_next_filtered(
    carquet_batch_reader_t* br, carquet_row_batch_t** batch) {

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_status_t st = filter_advance_to_active_row_group(br, &err);
    if (st != CARQUET_OK) {
        *batch = NULL;
        return st;
    }

    const carquet_row_range_t* range =
        &br->current_rg_ranges.ranges[br->current_range_index];
    int64_t range_remaining = range->num_rows - br->current_range_rows_emitted;
    int64_t batch_size = br->config.batch_size;
    if (batch_size <= 0) batch_size = 65536;
    int64_t rows_to_read = range_remaining < batch_size
        ? range_remaining : batch_size;

    if (!br->range_positioned) {
        int64_t target_row = range->first_row + br->current_range_rows_emitted;
        for (int32_t i = 0; i < br->num_projected; i++) {
            st = position_projected_column(br, i, target_row, &err);
            if (st != CARQUET_OK) return st;
        }
        br->range_positioned = true;
    }

    /* Reuse or allocate batch struct. */
    carquet_row_batch_t* new_batch = br->cached_batch;
    if (!new_batch) {
        new_batch = carquet_mem_calloc(1, sizeof(carquet_row_batch_t));
        if (!new_batch) return CARQUET_ERROR_OUT_OF_MEMORY;
        if (carquet_arena_init(&new_batch->arena) != CARQUET_OK) {
            carquet_mem_free(new_batch);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        new_batch->columns = carquet_arena_calloc(&new_batch->arena,
            br->num_projected, sizeof(carquet_column_data_t));
        if (!new_batch->columns) {
            carquet_arena_destroy(&new_batch->arena);
            carquet_mem_free(new_batch);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        new_batch->pooled = true;
        br->cached_batch = new_batch;
    }
    memset(new_batch->columns, 0,
           sizeof(carquet_column_data_t) * br->num_projected);
    new_batch->num_columns = br->num_projected;

    bool read_error = false;
    for (int32_t i = 0; i < br->num_projected; i++) {
        read_projected_column(br, new_batch, i, rows_to_read, true, &read_error);
    }
    if (read_error) {
        return CARQUET_ERROR_DECODE;
    }

    new_batch->num_rows = new_batch->columns[0].num_values;
    br->total_rows_read += new_batch->num_rows;

    br->current_range_rows_emitted += new_batch->num_rows;
    if (br->current_range_rows_emitted >= range->num_rows) {
        br->current_range_index++;
        br->current_range_rows_emitted = 0;
        br->range_positioned = false;
    }

    *batch = new_batch;
    return CARQUET_OK;
}

/* ============================================================================
 * Page Filter — public API
 * ============================================================================ */

carquet_status_t carquet_batch_reader_set_page_filter(
    carquet_batch_reader_t* reader,
    const carquet_filter_clause_t* clauses,
    int32_t count) {

    /* reader is nonnull per API contract. */

    if (clauses == NULL || count <= 0) {
        reader->filter_clauses = NULL;
        reader->filter_clause_count = 0;
        carquet_row_range_list_clear(&reader->current_rg_ranges);
        reader->filter_rg_state_valid = false;
        reader->current_range_index = 0;
        reader->current_range_rows_emitted = 0;
        reader->range_positioned = false;
        filter_release_offset_indexes(reader);
        return CARQUET_OK;
    }

    /* Validate every clause up front. */
    carquet_error_t err = CARQUET_ERROR_INIT;
    for (int32_t i = 0; i < count; i++) {
        carquet_status_t st = carquet_page_filter_validate_clause(
            reader->reader, &clauses[i], &err);
        if (st != CARQUET_OK) return st;
    }

    /* If a pipeline is currently active, drain in-flight tasks and drop
     * any pre-read slots whose contents predate the new filter state. */
    if (reader->pipeline_active && reader->pool) {
        carquet_worker_pool_wait(reader->pool);
        for (int32_t s = 0; s < reader->pipeline_depth; s++) {
            rg_slot_t* slot = &reader->pipeline[s];
            if (slot->rg_index >= 0) {
                slot_release_filter_state(slot, reader->num_projected);
                slot->rg_index = -1;
            }
        }
        reader->pipeline_head = 0;
        reader->pipeline_count = 0;
        reader->rg_order_next = 0;
    }

    reader->filter_clauses = clauses;
    reader->filter_clause_count = count;
    reader->filter_rg_state_valid = false;
    reader->current_range_index = 0;
    reader->current_range_rows_emitted = 0;
    reader->range_positioned = false;
    filter_release_offset_indexes(reader);

    /* A new filter restarts iteration from the beginning of the file:
     * the predicate may match row groups the previous filter (or
     * unfiltered read) already advanced past. */
    reader->current_row_group = -1;
    reader->rows_read_in_group = 0;
    return CARQUET_OK;
}

int64_t carquet_batch_reader_rows_skipped(
    const carquet_batch_reader_t* reader) {
    /* reader is nonnull per API contract. */
    return reader->rows_skipped;
}

/* ============================================================================
 * Nested batch driver
 * ============================================================================
 * Used when any projected column is repeated (max_rep > 0). Reads a whole row
 * group per batch (the natural granularity that avoids splitting a logical row
 * across batches) and reconstructs list columns via read_nested_list_column().
 * Flat columns in the same projection are read normally. Page filters are not
 * combined with nested reads in this release.
 */
static carquet_status_t batch_reader_next_nested(
    carquet_batch_reader_t* batch_reader,
    carquet_row_batch_t** batch) {

    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Advance to the next row group when the current one is exhausted.
     *
     * Repeated columns are read a whole row group at a time, so a page filter
     * is composed at ROW-GROUP granularity: a row group whose statistics prove
     * no row can match is skipped entirely; a row group with any match is read
     * in full (sub-row-group page ranges are not applied to repeated leaves,
     * whose slot counts do not align with logical row ranges). The user-level
     * row_group_filter callback is honoured the same way. */
    bool have_page_filter = batch_reader->filter_clauses &&
                            batch_reader->filter_clause_count > 0;
    if (batch_reader->current_row_group < 0 ||
        !carquet_column_has_next(batch_reader->col_readers[0])) {

        int32_t num_row_groups = carquet_reader_num_row_groups(batch_reader->reader);
        for (;;) {
            batch_reader->current_row_group++;
            if (batch_reader->current_row_group >= num_row_groups) {
                *batch = NULL;
                return CARQUET_ERROR_END_OF_DATA;
            }
            if (batch_reader->config.row_group_filter &&
                !batch_reader->config.row_group_filter(
                    batch_reader->reader, batch_reader->current_row_group,
                    batch_reader->config.row_group_filter_ctx)) {
                continue;
            }
            if (have_page_filter) {
                carquet_row_range_list_t ranges;
                carquet_row_range_list_init(&ranges);
                carquet_status_t fst = carquet_page_filter_eval_row_group(
                    batch_reader->reader, batch_reader->current_row_group,
                    batch_reader->filter_clauses, batch_reader->filter_clause_count,
                    &ranges, &err);
                if (fst != CARQUET_OK) {
                    carquet_row_range_list_destroy(&ranges);
                    return fst;
                }
                int64_t matched = ranges.total_rows;
                carquet_row_range_list_destroy(&ranges);
                if (matched == 0) {
                    continue;  /* statistics prove the row group has no match */
                }
            }
            break;
        }
        carquet_status_t status = open_row_group_readers(
            batch_reader, batch_reader->current_row_group, &err);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    /* Reuse or allocate batch struct. */
    carquet_row_batch_t* new_batch = batch_reader->cached_batch;
    if (!new_batch) {
        new_batch = carquet_mem_calloc(1, sizeof(carquet_row_batch_t));
        if (!new_batch) return CARQUET_ERROR_OUT_OF_MEMORY;
        if (carquet_arena_init(&new_batch->arena) != CARQUET_OK) {
            carquet_mem_free(new_batch);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        new_batch->columns = carquet_arena_calloc(&new_batch->arena,
            batch_reader->num_projected, sizeof(carquet_column_data_t));
        if (!new_batch->columns) {
            carquet_arena_destroy(&new_batch->arena);
            carquet_mem_free(new_batch);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        new_batch->pooled = true;
        batch_reader->cached_batch = new_batch;
    }

    memset(new_batch->columns, 0,
           sizeof(carquet_column_data_t) * batch_reader->num_projected);
    new_batch->num_columns = batch_reader->num_projected;

    int64_t rg_rows =
        batch_reader->reader->metadata.row_groups[batch_reader->current_row_group].num_rows;
    if (rg_rows <= 0) {
        new_batch->num_rows = 0;
        *batch = new_batch;
        return CARQUET_OK;
    }

    bool read_error = false;
    for (int32_t i = 0; i < batch_reader->num_projected; i++) {
        if (batch_reader->projected_max_reps[i] == 0) {
            read_projected_column(batch_reader, new_batch, i, rg_rows, false, &read_error);
        } else if (batch_reader->projected_max_reps[i] == 1) {
            read_nested_list_column(batch_reader, new_batch, i, rg_rows, &read_error);
        } else {
            CARQUET_SET_ERROR(&err, CARQUET_ERROR_NOT_IMPLEMENTED,
                "Batch reader: nested column depth > 1 (max_rep=%d) not supported",
                batch_reader->projected_max_reps[i]);
            return CARQUET_ERROR_NOT_IMPLEMENTED;
        }
        if (read_error) {
            CARQUET_SET_ERROR(&err, CARQUET_ERROR_INTERNAL,
                "Batch reader: failed to read nested column %d", i);
            return CARQUET_ERROR_INTERNAL;
        }
    }

    new_batch->num_rows = rg_rows;
    batch_reader->total_rows_read += rg_rows;
    *batch = new_batch;
    return CARQUET_OK;
}

carquet_status_t carquet_batch_reader_next(
    carquet_batch_reader_t* batch_reader,
    carquet_row_batch_t** batch) {

    /* batch_reader and batch are nonnull per API contract */
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Repeated (LIST/MAP-leaf) columns use the dedicated nested driver, which
     * reconstructs Arrow list layout a whole row group at a time. */
    if (batch_reader->has_repeated) {
        return batch_reader_next_nested(batch_reader, batch);
    }

    /* ====================================================================
     * FILTERED + SEQUENTIAL PATH (page filter active, pipeline disabled)
     *
     * The pipeline path's filtered variant (pipeline_fill below) handles
     * the case where pipeline_active is true: it pre-reads only matching
     * pages into the slot buffers and is then served by the pipeline
     * fast path further down. When the pipeline is not active (e.g.
     * single-row-group, uncompressed, OPTIONAL columns), we drive the
     * sequential range-iterator instead.
     * ==================================================================== */
    if (batch_reader->filter_clauses &&
        batch_reader->filter_clause_count > 0 &&
        !batch_reader->pipeline_active) {
        return batch_reader_next_filtered(batch_reader, batch);
    }

    /* ====================================================================
     * PIPELINE FAST PATH: serve pre-read data directly from ring buffer
     * ====================================================================
     * When pipeline is active, ALL column data has been bulk-read into
     * contiguous buffers by worker pool threads. We just memcpy batches
     * from those buffers. No column readers, no per-page overhead. */
    if (batch_reader->pipeline_active) {
        /* Check if we need to advance to the next pipeline slot */
        rg_slot_t* slot = NULL;
        if (batch_reader->pipeline_count > 0) {
            slot = &batch_reader->pipeline[batch_reader->pipeline_head];
            if (slot->rows_consumed >= slot->total_rows) {
                /* Current slot exhausted — retire it and advance */
                slot->rg_index = -1;
                slot_release_filter_state(slot, batch_reader->num_projected);
                batch_reader->pipeline_head = (batch_reader->pipeline_head + 1) % batch_reader->pipeline_depth;
                batch_reader->pipeline_count--;
                slot = NULL;
            }
        }

        if (!slot) {
            /* Fill and wait for new pipeline slots */
            pipeline_fill(batch_reader);
            if (batch_reader->pipeline_count == 0) {
                *batch = NULL;
                return CARQUET_ERROR_END_OF_DATA;
            }
            carquet_worker_pool_wait(batch_reader->pool);
            slot = &batch_reader->pipeline[batch_reader->pipeline_head];

            /* Refill freed slots for next round */
            pipeline_fill(batch_reader);
        }

        /* Reuse or allocate batch struct */
        carquet_row_batch_t* new_batch = batch_reader->cached_batch;
        if (!new_batch) {
            new_batch = carquet_mem_calloc(1, sizeof(carquet_row_batch_t));
            if (!new_batch) return CARQUET_ERROR_OUT_OF_MEMORY;
            if (carquet_arena_init(&new_batch->arena) != CARQUET_OK) {
                carquet_mem_free(new_batch);
                return CARQUET_ERROR_OUT_OF_MEMORY;
            }
            new_batch->columns = carquet_arena_calloc(&new_batch->arena,
                batch_reader->num_projected, sizeof(carquet_column_data_t));
            if (!new_batch->columns) {
                carquet_arena_destroy(&new_batch->arena);
                carquet_mem_free(new_batch);
                return CARQUET_ERROR_OUT_OF_MEMORY;
            }
            new_batch->pooled = true;
            batch_reader->cached_batch = new_batch;
        }

        memset(new_batch->columns, 0, sizeof(carquet_column_data_t) * batch_reader->num_projected);
        new_batch->num_columns = batch_reader->num_projected;

        int64_t remaining = slot->total_rows - slot->rows_consumed;
        int64_t rows_to_read = remaining > batch_reader->config.batch_size
                             ? batch_reader->config.batch_size : remaining;

        /* Copy data from pre-read buffers into batch (zero-copy view) */
        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            carquet_column_data_t* col = &new_batch->columns[i];
            size_t vs = batch_reader->projected_value_sizes[i];
            size_t offset = (size_t)slot->rows_consumed * vs;

            col->data = (uint8_t*)slot->col_values[i] + offset;
            col->num_values = rows_to_read;
            col->type = batch_reader->projected_types[i];
            col->type_length = batch_reader->projected_type_lengths[i];
            col->ownership = CARQUET_DATA_VIEW;
            col->null_bitmap = NULL;  /* REQUIRED columns — no nulls */
        }

        slot->rows_consumed += rows_to_read;
        new_batch->num_rows = rows_to_read;
        batch_reader->total_rows_read += rows_to_read;
        *batch = new_batch;
        return CARQUET_OK;
    }

    /* ====================================================================
     * SEQUENTIAL PATH (non-mmap, uncompressed, or single RG)
     * ==================================================================== */

    /* Check if we need to move to next row group */
    if (batch_reader->current_row_group < 0 ||
        !carquet_column_has_next(batch_reader->col_readers[0])) {

        int32_t num_row_groups = carquet_reader_num_row_groups(batch_reader->reader);
        batch_reader->current_row_group++;
        if (batch_reader->current_row_group >= num_row_groups) {
            *batch = NULL;
            return CARQUET_ERROR_END_OF_DATA;
        }

        /* Apply row group filter */
        while (batch_reader->config.row_group_filter) {
            bool should_read = batch_reader->config.row_group_filter(
                batch_reader->reader, batch_reader->current_row_group,
                batch_reader->config.row_group_filter_ctx);
            if (should_read) break;
            batch_reader->current_row_group++;
            if (batch_reader->current_row_group >= num_row_groups) {
                *batch = NULL;
                return CARQUET_ERROR_END_OF_DATA;
            }
        }

        carquet_status_t status = open_row_group_readers(
            batch_reader, batch_reader->current_row_group, &err);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    /* Reuse or allocate batch struct */
    carquet_row_batch_t* new_batch = batch_reader->cached_batch;
    if (!new_batch) {
        new_batch = carquet_mem_calloc(1, sizeof(carquet_row_batch_t));
        if (!new_batch) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        if (carquet_arena_init(&new_batch->arena) != CARQUET_OK) {
            carquet_mem_free(new_batch);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        new_batch->columns = carquet_arena_calloc(&new_batch->arena,
            batch_reader->num_projected, sizeof(carquet_column_data_t));
        if (!new_batch->columns) {
            carquet_arena_destroy(&new_batch->arena);
            carquet_mem_free(new_batch);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        new_batch->pooled = true;
        batch_reader->cached_batch = new_batch;
    }

    /* Reset column data for this batch */
    memset(new_batch->columns, 0, sizeof(carquet_column_data_t) * batch_reader->num_projected);
    new_batch->num_columns = batch_reader->num_projected;

    int64_t batch_size = batch_reader->config.batch_size;
    int64_t rows_to_read = carquet_column_remaining(batch_reader->col_readers[0]);
    if (rows_to_read > batch_size) {
        rows_to_read = batch_size;
    }

    /* Handle empty row group - return empty batch, not an error */
    if (rows_to_read == 0) {
        new_batch->num_rows = 0;
        *batch = new_batch;
        return CARQUET_OK;
    }

    /* Read each column - potentially in parallel */
    bool read_error = false;

    /* Uncompressed fixed-width mmap columns can often be served entirely as
     * direct page views. Pre-load those pages serially before the parallel
     * decision so we can clamp the batch to the current page window and avoid
     * both the zero-byte peek copy path and the column-parallel barrier cost. */
    {
        bool zero_copy_candidates = true;
        for (int32_t zi = 0; zi < batch_reader->num_projected; zi++) {
            if (!column_is_zero_copy_candidate(
                    batch_reader->col_readers[zi],
                    batch_reader->projected_types[zi],
                    batch_reader->projected_max_defs[zi])) {
                zero_copy_candidates = false;
                break;
            }
        }

        if (zero_copy_candidates) {
            for (int32_t zi = 0; zi < batch_reader->num_projected; zi++) {
                carquet_column_reader_t* col_reader = batch_reader->col_readers[zi];
                if (col_reader && col_reader->values_remaining > 0) {
                    carquet_status_t status = carquet_column_ensure_page_loaded(col_reader, &err);
                    if (status != CARQUET_OK) {
                        return status;
                    }
                }
            }
        }
    }

    /* ========================================================================
     * PARALLEL PAGE PREFETCH PHASE
     * ========================================================================
     * Pre-load pages for ALL columns in parallel BEFORE reading.
     * Uses persistent worker pool (no per-batch fork/join overhead) when
     * available, falls back to OpenMP, then serial.
     *
     * Only parallelize for mmap (fread is not thread-safe) and when there
     * are columns needing decompression (uncompressed pages are trivial). */
    bool is_mmap = (batch_reader->reader->mmap_data != NULL);
    bool needs_decompression = false;
    for (int32_t pi = 0; pi < batch_reader->num_projected; pi++) {
        carquet_column_reader_t* cr = batch_reader->col_readers[pi];
        if (cr && cr->col_meta &&
            cr->col_meta->codec != CARQUET_COMPRESSION_UNCOMPRESSED) {
            needs_decompression = true;
            break;
        }
    }

#ifdef _OPENMP
    {
        int num_threads = batch_reader->config.num_threads;
        if (num_threads <= 0) num_threads = omp_get_max_threads();
        if (num_threads > batch_reader->num_projected) num_threads = batch_reader->num_projected;
        if (num_threads < 1) num_threads = 1;

        int32_t omp_i;
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1) if(is_mmap && needs_decompression && num_threads > 1)
        for (omp_i = 0; omp_i < batch_reader->num_projected; omp_i++) {
            carquet_column_reader_t* col_reader = batch_reader->col_readers[omp_i];
            if (col_reader && !col_reader->page_loaded && col_reader->values_remaining > 0) {
                (void)carquet_column_read_batch(col_reader, NULL, 0, NULL, NULL);
            }
        }
    }
#else
    for (int32_t pi = 0; pi < batch_reader->num_projected; pi++) {
        carquet_column_reader_t* col_reader = batch_reader->col_readers[pi];
        if (col_reader && !col_reader->page_loaded && col_reader->values_remaining > 0) {
            (void)carquet_column_read_batch(col_reader, NULL, 0, NULL, NULL);
        }
    }
#endif

    /* If every projected column is backed by a direct page view, trim the
     * batch to the smallest currently available page slice. This avoids
     * copying across page boundaries and lets the main read phase stay on
     * the zero-copy path even when the requested batch size is larger than
     * an individual page. */
    {
        int64_t zero_copy_rows = clamp_rows_to_zero_copy_window(batch_reader, rows_to_read);
        if (zero_copy_rows > 0 && zero_copy_rows < rows_to_read) {
            rows_to_read = zero_copy_rows;
        }
    }

    /* ========================================================================
     * MAIN COLUMN READING PHASE
     * ========================================================================
     * Read from pre-loaded pages. Since pages are already decompressed,
     * this phase is mostly memory copies / zero-copy pointer setup.
     * Uses pooled buffers to avoid per-batch malloc/free.
     *
     * Worker pool is used for parallel column reading when pool is available
     * and columns need non-trivial work. Otherwise serial (which is often
     * optimal for zero-copy columns where read_projected_column is ~free).
     */
    bool all_zero_copy_ready = true;
    for (int32_t zi = 0; zi < batch_reader->num_projected; zi++) {
        if (!column_can_zero_copy_batch(
                batch_reader->col_readers[zi],
                batch_reader->projected_types[zi],
                batch_reader->projected_max_defs[zi],
                rows_to_read)) {
            all_zero_copy_ready = false;
            break;
        }
    }

    int32_t col_i;
#ifdef _OPENMP
    {
        int num_threads_read = batch_reader->config.num_threads;
        if (num_threads_read <= 0) num_threads_read = omp_get_max_threads();
        if (num_threads_read > batch_reader->num_projected) num_threads_read = batch_reader->num_projected;
        if (num_threads_read < 1) num_threads_read = 1;

        bool can_par = is_mmap && (num_threads_read > 1) &&
                       (batch_reader->num_projected > 1) && !all_zero_copy_ready;
        if (can_par) {
            /* Per-column error slots: each thread writes only its own slot, so
             * the failure flag is never a shared write across threads (no data
             * race). Reduce into read_error after the region. */
            bool* col_err = carquet_mem_calloc(
                (size_t)batch_reader->num_projected, sizeof(bool));
            if (!col_err) {
                return CARQUET_ERROR_OUT_OF_MEMORY;
            }
            #pragma omp parallel for num_threads(num_threads_read) schedule(dynamic, 1)
            for (col_i = 0; col_i < batch_reader->num_projected; col_i++) {
                read_projected_column(batch_reader, new_batch, col_i, rows_to_read, true, &col_err[col_i]);
            }
            for (col_i = 0; col_i < batch_reader->num_projected; col_i++) {
                if (col_err[col_i]) read_error = true;
            }
            carquet_mem_free(col_err);
        } else {
            for (col_i = 0; col_i < batch_reader->num_projected; col_i++) {
                read_projected_column(batch_reader, new_batch, col_i, rows_to_read, true, &read_error);
            }
        }
    }
#else
    for (col_i = 0; col_i < batch_reader->num_projected; col_i++) {
        read_projected_column(batch_reader, new_batch, col_i, rows_to_read, true, &read_error);
    }
#endif

    if (read_error) {
        /* Don't free cached_batch, just return error */
        return CARQUET_ERROR_DECODE;
    }

    new_batch->num_rows = new_batch->columns[0].num_values;
    batch_reader->total_rows_read += new_batch->num_rows;

    *batch = new_batch;
    return CARQUET_OK;
}

void carquet_batch_reader_free(carquet_batch_reader_t* batch_reader) {
    if (!batch_reader) return;

    /* Drain any in-flight pipeline tasks before freeing */
    if (batch_reader->pool) {
        carquet_worker_pool_wait(batch_reader->pool);
    }

    /* Free pipeline slots */
    if (batch_reader->pipeline) {
        for (int32_t s = 0; s < batch_reader->pipeline_depth; s++) {
            rg_slot_t* slot = &batch_reader->pipeline[s];
            slot_release_filter_state(slot, batch_reader->num_projected);
            if (slot->col_offset_indexes) {
                carquet_mem_free(slot->col_offset_indexes);
                slot->col_offset_indexes = NULL;
            }
            if (slot->col_readers) {
                for (int32_t i = 0; i < batch_reader->num_projected; i++) {
                    if (slot->col_readers[i]) {
                        carquet_column_reader_free(slot->col_readers[i]);
                    }
                }
                carquet_mem_free(slot->col_readers);
            }
            if (slot->col_values) {
                for (int32_t i = 0; i < batch_reader->num_projected; i++) {
                    carquet_mem_free(slot->col_values[i]);
                }
                carquet_mem_free(slot->col_values);
            }
            carquet_mem_free(slot->col_buf_sizes);
            carquet_mem_free(slot->col_num_values);
#if !defined(_WIN32)
            if (slot->slot_mmap && slot->slot_mmap_size > 0)
                munmap(slot->slot_mmap, slot->slot_mmap_size);
#endif
        }
        carquet_mem_free(batch_reader->pipeline);
    }

    /* Free per-reader task args */
    carquet_mem_free(batch_reader->task_args);

    /* Destroy worker pool (only if we created it) */
    if (!batch_reader->pool_is_borrowed) {
        carquet_worker_pool_destroy(batch_reader->pool);
    }

    /* Free column readers */
    if (batch_reader->col_readers) {
        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            if (batch_reader->col_readers[i]) {
                carquet_column_reader_free(batch_reader->col_readers[i]);
            }
        }
        carquet_mem_free(batch_reader->col_readers);
    }

    /* Free buffer pools */
    if (batch_reader->col_pools) {
        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            carquet_mem_free(batch_reader->col_pools[i].data);
            carquet_mem_free(batch_reader->col_pools[i].null_bitmap);
            carquet_mem_free(batch_reader->col_pools[i].def_levels);
            carquet_mem_free(batch_reader->col_pools[i].rep_levels);
            carquet_mem_free(batch_reader->col_pools[i].list_offsets);
            carquet_mem_free(batch_reader->col_pools[i].list_validity);
        }
        carquet_mem_free(batch_reader->col_pools);
    }

    /* Free cached batch struct (but NOT pool buffers - those are freed above) */
    if (batch_reader->cached_batch) {
        carquet_arena_destroy(&batch_reader->cached_batch->arena);
        carquet_mem_free(batch_reader->cached_batch);
    }

    /* Filter state cleanup */
    filter_release_offset_indexes(batch_reader);
    carquet_mem_free(batch_reader->projected_offset_indexes);
    carquet_row_range_list_destroy(&batch_reader->current_rg_ranges);

    carquet_mem_free(batch_reader->rg_order);
    carquet_mem_free(batch_reader->projected_value_sizes);
    carquet_mem_free(batch_reader->projected_max_reps);
    carquet_mem_free(batch_reader->projected_max_defs);
    carquet_mem_free(batch_reader->projected_type_lengths);
    carquet_mem_free(batch_reader->projected_types);
    carquet_mem_free(batch_reader->projected_columns);
    carquet_mem_free(batch_reader);
}

/* ============================================================================
 * Public Thread Pool API
 * ============================================================================
 */

carquet_thread_pool_t* carquet_thread_pool_create(int32_t num_threads) {
    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 4;
#endif
    }
    if (num_threads < 2) num_threads = 2;
    return (carquet_thread_pool_t*)carquet_worker_pool_create(num_threads);
}

void carquet_thread_pool_destroy(carquet_thread_pool_t* pool) {
    carquet_worker_pool_destroy((carquet_worker_pool_t*)pool);
}

/* ============================================================================
 * Row Batch Implementation
 * ============================================================================
 */

int64_t carquet_row_batch_num_rows(const carquet_row_batch_t* batch) {
    /* batch is nonnull per API contract */
    return batch->num_rows;
}

int32_t carquet_row_batch_num_columns(const carquet_row_batch_t* batch) {
    /* batch is nonnull per API contract */
    return batch->num_columns;
}

carquet_status_t carquet_row_batch_column(
    const carquet_row_batch_t* batch,
    int32_t column_index,
    const void** data,
    const uint8_t** null_bitmap,
    int64_t* num_values) {

    /* batch, data, null_bitmap, num_values are nonnull per API contract */
    if (column_index < 0 || column_index >= batch->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    const carquet_column_data_t* col = &batch->columns[column_index];

    /* When preserve_dictionaries is enabled, col->data holds uint32_t indices,
     * not materialized values. Returning it through the value accessor would
     * hand the caller indices silently mis-cast as the column's physical type.
     * Force the caller to use carquet_row_batch_column_dictionary() instead. */
    if (col->is_dictionary) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* List (repeated) columns must be read via carquet_row_batch_column_list();
     * returning the flattened child through the flat accessor would silently
     * drop the list structure. */
    if (col->list_offsets) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    *data = col->data;
    *null_bitmap = col->null_bitmap;
    *num_values = col->num_values;

    return CARQUET_OK;
}

carquet_status_t carquet_row_batch_column_list(
    const carquet_row_batch_t* batch,
    int32_t column_index,
    const int32_t** offsets,
    int64_t* num_lists,
    const void** values,
    const uint8_t** value_validity,
    int64_t* num_values,
    const uint8_t** list_validity) {

    if (column_index < 0 || column_index >= batch->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }
    const carquet_column_data_t* col = &batch->columns[column_index];
    if (!col->list_offsets) {
        return CARQUET_ERROR_INVALID_ARGUMENT;  /* not a list column */
    }

    *offsets = col->list_offsets;
    *num_lists = col->num_lists;
    *values = col->data;
    *value_validity = col->null_bitmap;
    *num_values = col->num_values;
    if (list_validity) {
        *list_validity = col->list_validity;
    }
    return CARQUET_OK;
}

carquet_status_t carquet_row_batch_column_dictionary(
    const carquet_row_batch_t* batch,
    int32_t column_index,
    const uint32_t** indices,
    const uint8_t** null_bitmap,
    int64_t* num_values,
    const uint8_t** dictionary_data,
    int32_t* dictionary_count,
    const uint32_t** dictionary_offsets) {

    if (column_index < 0 || column_index >= batch->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    const carquet_column_data_t* col = &batch->columns[column_index];

    if (!col->is_dictionary) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    *indices = (const uint32_t*)col->data;
    *null_bitmap = col->null_bitmap;
    *num_values = col->num_values;
    *dictionary_data = col->dictionary_data;
    *dictionary_count = col->dictionary_count;
    if (dictionary_offsets) {
        *dictionary_offsets = col->dictionary_offsets;
    }

    return CARQUET_OK;
}

void carquet_row_batch_free(carquet_row_batch_t* batch) {
    if (!batch) return;

    /* Pooled batches are owned by the batch_reader - don't free data */
    if (batch->pooled) {
        /* Data buffers belong to the batch_reader's pool.
         * The batch struct itself is cached and reused.
         * This is a no-op - the caller should just drop the pointer. */
        return;
    }

    /* Non-pooled batch: free column data (only if owned, not views into mmap) */
    for (int32_t i = 0; i < batch->num_columns; i++) {
        if (batch->columns[i].ownership == CARQUET_DATA_OWNED) {
            carquet_mem_free(batch->columns[i].data);
        }
        carquet_mem_free(batch->columns[i].null_bitmap);
    }

    carquet_arena_destroy(&batch->arena);
    carquet_mem_free(batch);
}
