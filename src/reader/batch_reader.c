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
 */

#include <carquet/carquet.h>
#include "reader_internal.h"
#include "core/arena.h"
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * Internal Structures
 * ============================================================================
 */

typedef struct carquet_column_data {
    void* data;                 /* Column values */
    uint8_t* null_bitmap;       /* Null bitmap (1 bit per value) */
    int64_t num_values;         /* Number of values */
    size_t data_capacity;       /* Allocated capacity for data */
    carquet_physical_type_t type;
    int32_t type_length;        /* For fixed-length types */
} carquet_column_data_t;

struct carquet_row_batch {
    carquet_column_data_t* columns;
    int32_t num_columns;
    int64_t num_rows;
    carquet_arena_t arena;
};

struct carquet_batch_reader {
    carquet_reader_t* reader;
    carquet_batch_reader_config_t config;

    /* Column projection */
    int32_t* projected_columns;  /* File column indices to read */
    int32_t num_projected;       /* Number of projected columns */

    /* Reading state */
    int32_t current_row_group;
    int64_t rows_read_in_group;
    int64_t total_rows_read;

    /* Column readers for current row group */
    carquet_column_reader_t** col_readers;

    /* Memory-mapped data */
    uint8_t* mmap_data;
    size_t mmap_size;
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

static size_t get_type_size(carquet_physical_type_t type, int32_t type_length) {
    switch (type) {
        case CARQUET_PHYSICAL_BOOLEAN: return 1;
        case CARQUET_PHYSICAL_INT32: return 4;
        case CARQUET_PHYSICAL_INT64: return 8;
        case CARQUET_PHYSICAL_INT96: return 12;
        case CARQUET_PHYSICAL_FLOAT: return 4;
        case CARQUET_PHYSICAL_DOUBLE: return 8;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: return (size_t)type_length;
        case CARQUET_PHYSICAL_BYTE_ARRAY: return sizeof(carquet_byte_array_t);
        default: return 0;
    }
}

static int resolve_column_name(const carquet_reader_t* reader, const char* name) {
    const carquet_schema_t* schema = carquet_reader_schema(reader);
    if (!schema) return -1;

    return carquet_schema_find_column(schema, name);
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
    carquet_batch_reader_t* batch_reader = calloc(1, sizeof(carquet_batch_reader_t));
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
        batch_reader->projected_columns = malloc(sizeof(int32_t) * batch_reader->num_projected);
        if (!batch_reader->projected_columns) {
            free(batch_reader);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection");
            return NULL;
        }
        memcpy(batch_reader->projected_columns, batch_reader->config.column_indices,
               sizeof(int32_t) * batch_reader->num_projected);
    } else if (batch_reader->config.column_names && batch_reader->config.num_column_names > 0) {
        /* Resolve column names to indices */
        batch_reader->num_projected = batch_reader->config.num_column_names;
        batch_reader->projected_columns = malloc(sizeof(int32_t) * batch_reader->num_projected);
        if (!batch_reader->projected_columns) {
            free(batch_reader);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection");
            return NULL;
        }

        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            const char* col_name = batch_reader->config.column_names[i];
            int32_t idx = resolve_column_name(reader, col_name);
            if (idx < 0) {
                free(batch_reader->projected_columns);
                free(batch_reader);
                CARQUET_SET_ERROR(error, CARQUET_ERROR_COLUMN_NOT_FOUND,
                    "Column not found: %s", col_name);
                return NULL;
            }
            batch_reader->projected_columns[i] = idx;
        }
    } else {
        /* Read all columns */
        batch_reader->num_projected = total_columns;
        batch_reader->projected_columns = malloc(sizeof(int32_t) * total_columns);
        if (!batch_reader->projected_columns) {
            free(batch_reader);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection");
            return NULL;
        }
        for (int32_t i = 0; i < total_columns; i++) {
            batch_reader->projected_columns[i] = i;
        }
    }

    /* Allocate column reader array */
    batch_reader->col_readers = calloc(batch_reader->num_projected,
                                        sizeof(carquet_column_reader_t*));
    if (!batch_reader->col_readers) {
        free(batch_reader->projected_columns);
        free(batch_reader);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate column readers");
        return NULL;
    }

    batch_reader->current_row_group = -1;

    return batch_reader;
}

static carquet_status_t open_row_group_readers(
    carquet_batch_reader_t* batch_reader,
    int32_t row_group_index,
    carquet_error_t* error) {

    /* Close existing readers */
    for (int32_t i = 0; i < batch_reader->num_projected; i++) {
        if (batch_reader->col_readers[i]) {
            carquet_column_reader_free(batch_reader->col_readers[i]);
            batch_reader->col_readers[i] = NULL;
        }
    }

    /* Open new readers for each projected column */
    for (int32_t i = 0; i < batch_reader->num_projected; i++) {
        int32_t file_col_idx = batch_reader->projected_columns[i];
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

    batch_reader->current_row_group = row_group_index;
    batch_reader->rows_read_in_group = 0;

    return CARQUET_OK;
}

carquet_status_t carquet_batch_reader_next(
    carquet_batch_reader_t* batch_reader,
    carquet_row_batch_t** batch) {

    /* batch_reader and batch are nonnull per API contract */
    carquet_error_t err = CARQUET_ERROR_INIT;
    int32_t num_row_groups = carquet_reader_num_row_groups(batch_reader->reader);

    /* Check if we need to move to next row group */
    if (batch_reader->current_row_group < 0 ||
        !carquet_column_has_next(batch_reader->col_readers[0])) {

        batch_reader->current_row_group++;
        if (batch_reader->current_row_group >= num_row_groups) {
            *batch = NULL;
            return CARQUET_ERROR_END_OF_DATA;
        }

        carquet_status_t status = open_row_group_readers(
            batch_reader, batch_reader->current_row_group, &err);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    /* Allocate batch */
    carquet_row_batch_t* new_batch = calloc(1, sizeof(carquet_row_batch_t));
    if (!new_batch) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    if (carquet_arena_init(&new_batch->arena) != CARQUET_OK) {
        free(new_batch);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    new_batch->num_columns = batch_reader->num_projected;
    new_batch->columns = carquet_arena_calloc(&new_batch->arena,
        batch_reader->num_projected, sizeof(carquet_column_data_t));
    if (!new_batch->columns) {
        carquet_arena_destroy(&new_batch->arena);
        free(new_batch);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    int64_t batch_size = batch_reader->config.batch_size;
    int64_t rows_to_read = carquet_column_remaining(batch_reader->col_readers[0]);
    if (rows_to_read > batch_size) {
        rows_to_read = batch_size;
    }

    /* Read each column - potentially in parallel */
    bool read_error = false;

#ifdef _OPENMP
    int num_threads = batch_reader->config.num_threads;
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
#endif
    for (int32_t i = 0; i < batch_reader->num_projected; i++) {
        if (read_error) continue;

        carquet_column_reader_t* col_reader = batch_reader->col_readers[i];
        carquet_column_data_t* col_data = &new_batch->columns[i];

        /* Get column type info */
        int32_t file_col_idx = batch_reader->projected_columns[i];
        const carquet_schema_t* schema = carquet_reader_schema(batch_reader->reader);
        int32_t schema_idx = schema->leaf_indices[file_col_idx];
        const parquet_schema_element_t* elem = &schema->elements[schema_idx];

        col_data->type = elem->has_type ? elem->type : CARQUET_PHYSICAL_BYTE_ARRAY;
        col_data->type_length = elem->type_length;

        size_t value_size = get_type_size(col_data->type, col_data->type_length);
        size_t data_size = value_size * (size_t)rows_to_read;

        /* Allocate column data buffer */
        col_data->data = malloc(data_size);
        if (!col_data->data) {
            read_error = true;
            continue;
        }
        col_data->data_capacity = data_size;

        /* Allocate null bitmap */
        size_t bitmap_size = ((size_t)rows_to_read + 7) / 8;
        col_data->null_bitmap = calloc(1, bitmap_size);

        /* Read values */
        int16_t* def_levels = NULL;
        if (schema->max_def_levels[file_col_idx] > 0) {
            def_levels = malloc(sizeof(int16_t) * (size_t)rows_to_read);
        }

        int64_t values_read = carquet_column_read_batch(
            col_reader, col_data->data, rows_to_read, def_levels, NULL);

        if (values_read < 0) {
            read_error = true;
            free(def_levels);
            continue;
        }

        col_data->num_values = values_read;

        /* Build null bitmap from definition levels */
        if (def_levels && col_data->null_bitmap) {
            int16_t max_def = schema->max_def_levels[file_col_idx];
            for (int64_t j = 0; j < values_read; j++) {
                if (def_levels[j] < max_def) {
                    /* Value is null - set bit */
                    col_data->null_bitmap[j / 8] |= (1 << (j % 8));
                }
            }
        }

        free(def_levels);
    }

    if (read_error) {
        carquet_row_batch_free(new_batch);
        return CARQUET_ERROR_DECODE;
    }

    new_batch->num_rows = new_batch->columns[0].num_values;
    batch_reader->total_rows_read += new_batch->num_rows;

    *batch = new_batch;
    return CARQUET_OK;
}

void carquet_batch_reader_free(carquet_batch_reader_t* batch_reader) {
    if (!batch_reader) return;

    /* Free column readers */
    if (batch_reader->col_readers) {
        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            if (batch_reader->col_readers[i]) {
                carquet_column_reader_free(batch_reader->col_readers[i]);
            }
        }
        free(batch_reader->col_readers);
    }

    free(batch_reader->projected_columns);
    free(batch_reader);
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

    *data = col->data;
    *null_bitmap = col->null_bitmap;
    *num_values = col->num_values;

    return CARQUET_OK;
}

void carquet_row_batch_free(carquet_row_batch_t* batch) {
    if (!batch) return;

    /* Free column data */
    for (int32_t i = 0; i < batch->num_columns; i++) {
        free(batch->columns[i].data);
        free(batch->columns[i].null_bitmap);
    }

    carquet_arena_destroy(&batch->arena);
    free(batch);
}
