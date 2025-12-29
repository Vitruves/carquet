/**
 * @file statistics.c
 * @brief Column statistics computation and comparison
 *
 * Implements min/max tracking and comparison for all Parquet physical types.
 * Statistics are used for predicate pushdown and query optimization.
 */

#include <carquet/carquet.h>
#include <carquet/error.h>
#include "thrift/parquet_types.h"
#include "core/arena.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Statistics Builder Structure
 * ============================================================================
 */

typedef struct carquet_statistics_builder {
    carquet_physical_type_t type;
    int32_t type_length;  /* For FIXED_LEN_BYTE_ARRAY */

    bool has_min;
    bool has_max;
    int64_t null_count;
    int64_t distinct_count;
    int64_t num_values;

    /* Min/max storage (large enough for any type) */
    uint8_t min_value[256];
    uint8_t max_value[256];
    size_t min_len;
    size_t max_len;

    /* For computing distinct count (simple approximation) */
    /* Full HyperLogLog would be better but more complex */
} carquet_statistics_builder_t;

/* ============================================================================
 * Type-Specific Comparison Functions
 * ============================================================================
 */

static int compare_boolean(const void* a, const void* b) {
    uint8_t va = *(const uint8_t*)a;
    uint8_t vb = *(const uint8_t*)b;
    return (va > vb) - (va < vb);
}

static int compare_int32(const void* a, const void* b) {
    int32_t va = *(const int32_t*)a;
    int32_t vb = *(const int32_t*)b;
    return (va > vb) - (va < vb);
}

static int compare_int64(const void* a, const void* b) {
    int64_t va = *(const int64_t*)a;
    int64_t vb = *(const int64_t*)b;
    return (va > vb) - (va < vb);
}

static int compare_float(const void* a, const void* b) {
    float va = *(const float*)a;
    float vb = *(const float*)b;

    /* Handle NaN - NaN is neither < nor > any value */
    if (isnan(va) && isnan(vb)) return 0;
    if (isnan(va)) return 1;  /* NaN sorts after everything */
    if (isnan(vb)) return -1;

    return (va > vb) - (va < vb);
}

static int compare_double(const void* a, const void* b) {
    double va = *(const double*)a;
    double vb = *(const double*)b;

    /* Handle NaN */
    if (isnan(va) && isnan(vb)) return 0;
    if (isnan(va)) return 1;
    if (isnan(vb)) return -1;

    return (va > vb) - (va < vb);
}

static int compare_int96(const void* a, const void* b) {
    /* INT96: compare as 3 uint32s in big-endian order */
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;

    /* Compare from high to low */
    for (int i = 2; i >= 0; i--) {
        if (va[i] != vb[i]) {
            return (va[i] > vb[i]) - (va[i] < vb[i]);
        }
    }
    return 0;
}

static int compare_byte_array(const void* a, size_t a_len,
                               const void* b, size_t b_len) {
    size_t min_len = a_len < b_len ? a_len : b_len;
    int cmp = memcmp(a, b, min_len);
    if (cmp != 0) return cmp;
    return (a_len > b_len) - (a_len < b_len);
}

/* ============================================================================
 * Statistics Builder API
 * ============================================================================
 */

/**
 * Create a statistics builder for a column.
 */
carquet_statistics_builder_t* carquet_statistics_builder_create(
    carquet_physical_type_t type,
    int32_t type_length) {

    carquet_statistics_builder_t* builder = calloc(1, sizeof(*builder));
    if (!builder) return NULL;

    builder->type = type;
    builder->type_length = type_length;

    return builder;
}

/**
 * Destroy a statistics builder.
 */
void carquet_statistics_builder_destroy(carquet_statistics_builder_t* builder) {
    free(builder);
}

/**
 * Reset a statistics builder for reuse.
 */
void carquet_statistics_builder_reset(carquet_statistics_builder_t* builder) {
    if (!builder) return;

    builder->has_min = false;
    builder->has_max = false;
    builder->null_count = 0;
    builder->distinct_count = 0;
    builder->num_values = 0;
    builder->min_len = 0;
    builder->max_len = 0;
}

/* ============================================================================
 * Value Size for Type
 * ============================================================================
 */

static size_t get_value_size(carquet_physical_type_t type, int32_t type_length) {
    switch (type) {
        case CARQUET_PHYSICAL_BOOLEAN: return 1;
        case CARQUET_PHYSICAL_INT32: return 4;
        case CARQUET_PHYSICAL_INT64: return 8;
        case CARQUET_PHYSICAL_INT96: return 12;
        case CARQUET_PHYSICAL_FLOAT: return 4;
        case CARQUET_PHYSICAL_DOUBLE: return 8;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: return (size_t)type_length;
        default: return 0;  /* Variable length */
    }
}

/* ============================================================================
 * Add Values to Statistics
 * ============================================================================
 */

/**
 * Add null count to statistics.
 */
void carquet_statistics_add_nulls(
    carquet_statistics_builder_t* builder,
    int64_t count) {

    if (builder) {
        builder->null_count += count;
    }
}

/**
 * Add values to statistics for fixed-size types.
 */
carquet_status_t carquet_statistics_add_values(
    carquet_statistics_builder_t* builder,
    const void* values,
    int64_t num_values) {

    if (!builder || !values || num_values <= 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t value_size = get_value_size(builder->type, builder->type_length);
    if (value_size == 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;  /* Use byte array API */
    }

    const uint8_t* data = (const uint8_t*)values;

    for (int64_t i = 0; i < num_values; i++) {
        const void* val = data + (i * value_size);
        int cmp_min = 0, cmp_max = 0;

        if (builder->has_min) {
            switch (builder->type) {
                case CARQUET_PHYSICAL_BOOLEAN:
                    cmp_min = compare_boolean(val, builder->min_value);
                    break;
                case CARQUET_PHYSICAL_INT32:
                    cmp_min = compare_int32(val, builder->min_value);
                    break;
                case CARQUET_PHYSICAL_INT64:
                    cmp_min = compare_int64(val, builder->min_value);
                    break;
                case CARQUET_PHYSICAL_INT96:
                    cmp_min = compare_int96(val, builder->min_value);
                    break;
                case CARQUET_PHYSICAL_FLOAT:
                    cmp_min = compare_float(val, builder->min_value);
                    break;
                case CARQUET_PHYSICAL_DOUBLE:
                    cmp_min = compare_double(val, builder->min_value);
                    break;
                case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
                    cmp_min = compare_byte_array(val, value_size,
                                                  builder->min_value, value_size);
                    break;
                default:
                    break;
            }
        }

        if (builder->has_max) {
            switch (builder->type) {
                case CARQUET_PHYSICAL_BOOLEAN:
                    cmp_max = compare_boolean(val, builder->max_value);
                    break;
                case CARQUET_PHYSICAL_INT32:
                    cmp_max = compare_int32(val, builder->max_value);
                    break;
                case CARQUET_PHYSICAL_INT64:
                    cmp_max = compare_int64(val, builder->max_value);
                    break;
                case CARQUET_PHYSICAL_INT96:
                    cmp_max = compare_int96(val, builder->max_value);
                    break;
                case CARQUET_PHYSICAL_FLOAT:
                    cmp_max = compare_float(val, builder->max_value);
                    break;
                case CARQUET_PHYSICAL_DOUBLE:
                    cmp_max = compare_double(val, builder->max_value);
                    break;
                case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
                    cmp_max = compare_byte_array(val, value_size,
                                                  builder->max_value, value_size);
                    break;
                default:
                    break;
            }
        }

        /* Update min */
        if (!builder->has_min || cmp_min < 0) {
            memcpy(builder->min_value, val, value_size);
            builder->min_len = value_size;
            builder->has_min = true;
        }

        /* Update max */
        if (!builder->has_max || cmp_max > 0) {
            memcpy(builder->max_value, val, value_size);
            builder->max_len = value_size;
            builder->has_max = true;
        }
    }

    builder->num_values += num_values;
    return CARQUET_OK;
}

/**
 * Add byte array values to statistics.
 */
carquet_status_t carquet_statistics_add_byte_arrays(
    carquet_statistics_builder_t* builder,
    const carquet_byte_array_t* values,
    int64_t num_values) {

    if (!builder || !values || num_values <= 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (builder->type != CARQUET_PHYSICAL_BYTE_ARRAY) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    for (int64_t i = 0; i < num_values; i++) {
        const uint8_t* val = values[i].data;
        size_t val_len = (size_t)values[i].length;

        /* Skip if too large */
        if (val_len > sizeof(builder->min_value)) {
            continue;
        }

        int cmp_min = 0, cmp_max = 0;

        if (builder->has_min) {
            cmp_min = compare_byte_array(val, val_len,
                                          builder->min_value, builder->min_len);
        }

        if (builder->has_max) {
            cmp_max = compare_byte_array(val, val_len,
                                          builder->max_value, builder->max_len);
        }

        /* Update min */
        if (!builder->has_min || cmp_min < 0) {
            memcpy(builder->min_value, val, val_len);
            builder->min_len = val_len;
            builder->has_min = true;
        }

        /* Update max */
        if (!builder->has_max || cmp_max > 0) {
            memcpy(builder->max_value, val, val_len);
            builder->max_len = val_len;
            builder->has_max = true;
        }
    }

    builder->num_values += num_values;
    return CARQUET_OK;
}

/* ============================================================================
 * Build Statistics
 * ============================================================================
 */

/**
 * Build parquet statistics from builder.
 *
 * @param builder The statistics builder
 * @param arena Arena for allocating memory
 * @param stats Output statistics structure
 * @return Status code
 */
carquet_status_t carquet_statistics_build(
    const carquet_statistics_builder_t* builder,
    carquet_arena_t* arena,
    parquet_statistics_t* stats) {

    if (!builder || !stats) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    memset(stats, 0, sizeof(*stats));

    /* Null count */
    stats->has_null_count = true;
    stats->null_count = builder->null_count;

    /* Distinct count (if tracked) */
    if (builder->distinct_count > 0) {
        stats->has_distinct_count = true;
        stats->distinct_count = builder->distinct_count;
    }

    /* Min value */
    if (builder->has_min && builder->min_len > 0) {
        if (arena) {
            stats->min_value = carquet_arena_memdup(arena,
                builder->min_value, builder->min_len);
        } else {
            stats->min_value = malloc(builder->min_len);
            if (stats->min_value) {
                memcpy(stats->min_value, builder->min_value, builder->min_len);
            }
        }
        if (stats->min_value) {
            stats->min_value_len = (int32_t)builder->min_len;
            stats->has_is_min_value_exact = true;
            stats->is_min_value_exact = true;
        }
    }

    /* Max value */
    if (builder->has_max && builder->max_len > 0) {
        if (arena) {
            stats->max_value = carquet_arena_memdup(arena,
                builder->max_value, builder->max_len);
        } else {
            stats->max_value = malloc(builder->max_len);
            if (stats->max_value) {
                memcpy(stats->max_value, builder->max_value, builder->max_len);
            }
        }
        if (stats->max_value) {
            stats->max_value_len = (int32_t)builder->max_len;
            stats->has_is_max_value_exact = true;
            stats->is_max_value_exact = true;
        }
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Statistics Comparison
 * ============================================================================
 */

/**
 * Compare a value against statistics min/max.
 *
 * @param stats The statistics
 * @param type Physical type
 * @param value The value to compare
 * @param value_len Length for byte array types
 * @param result Output: -1 if value < min, 0 if in range, 1 if value > max
 * @return Status code
 */
carquet_status_t carquet_statistics_compare(
    const parquet_statistics_t* stats,
    carquet_physical_type_t type,
    const void* value,
    size_t value_len,
    int* result) {

    if (!stats || !value || !result) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    *result = 0;  /* Assume in range by default */

    /* Compare against min */
    if (stats->min_value && stats->min_value_len > 0) {
        int cmp;
        switch (type) {
            case CARQUET_PHYSICAL_BOOLEAN:
                cmp = compare_boolean(value, stats->min_value);
                break;
            case CARQUET_PHYSICAL_INT32:
                cmp = compare_int32(value, stats->min_value);
                break;
            case CARQUET_PHYSICAL_INT64:
                cmp = compare_int64(value, stats->min_value);
                break;
            case CARQUET_PHYSICAL_INT96:
                cmp = compare_int96(value, stats->min_value);
                break;
            case CARQUET_PHYSICAL_FLOAT:
                cmp = compare_float(value, stats->min_value);
                break;
            case CARQUET_PHYSICAL_DOUBLE:
                cmp = compare_double(value, stats->min_value);
                break;
            case CARQUET_PHYSICAL_BYTE_ARRAY:
            case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
                cmp = compare_byte_array(value, value_len,
                    stats->min_value, (size_t)stats->min_value_len);
                break;
            default:
                return CARQUET_ERROR_INVALID_ARGUMENT;
        }
        if (cmp < 0) {
            *result = -1;
            return CARQUET_OK;
        }
    }

    /* Compare against max */
    if (stats->max_value && stats->max_value_len > 0) {
        int cmp;
        switch (type) {
            case CARQUET_PHYSICAL_BOOLEAN:
                cmp = compare_boolean(value, stats->max_value);
                break;
            case CARQUET_PHYSICAL_INT32:
                cmp = compare_int32(value, stats->max_value);
                break;
            case CARQUET_PHYSICAL_INT64:
                cmp = compare_int64(value, stats->max_value);
                break;
            case CARQUET_PHYSICAL_INT96:
                cmp = compare_int96(value, stats->max_value);
                break;
            case CARQUET_PHYSICAL_FLOAT:
                cmp = compare_float(value, stats->max_value);
                break;
            case CARQUET_PHYSICAL_DOUBLE:
                cmp = compare_double(value, stats->max_value);
                break;
            case CARQUET_PHYSICAL_BYTE_ARRAY:
            case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
                cmp = compare_byte_array(value, value_len,
                    stats->max_value, (size_t)stats->max_value_len);
                break;
            default:
                return CARQUET_ERROR_INVALID_ARGUMENT;
        }
        if (cmp > 0) {
            *result = 1;
            return CARQUET_OK;
        }
    }

    return CARQUET_OK;
}

/**
 * Check if a value range overlaps with statistics.
 *
 * @param stats The statistics
 * @param type Physical type
 * @param min_value Range minimum (NULL for unbounded)
 * @param max_value Range maximum (NULL for unbounded)
 * @param value_len Length for byte array types
 * @param overlaps Output: true if ranges overlap
 * @return Status code
 */
carquet_status_t carquet_statistics_range_overlaps(
    const parquet_statistics_t* stats,
    carquet_physical_type_t type,
    const void* min_value,
    const void* max_value,
    size_t value_len,
    bool* overlaps) {

    if (!stats || !overlaps) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    *overlaps = true;  /* Assume overlap by default */

    /* Check if query max < stats min (no overlap) */
    if (max_value && stats->min_value && stats->min_value_len > 0) {
        int cmp;
        switch (type) {
            case CARQUET_PHYSICAL_INT32:
                cmp = compare_int32(max_value, stats->min_value);
                break;
            case CARQUET_PHYSICAL_INT64:
                cmp = compare_int64(max_value, stats->min_value);
                break;
            case CARQUET_PHYSICAL_FLOAT:
                cmp = compare_float(max_value, stats->min_value);
                break;
            case CARQUET_PHYSICAL_DOUBLE:
                cmp = compare_double(max_value, stats->min_value);
                break;
            default:
                cmp = compare_byte_array(max_value, value_len,
                    stats->min_value, (size_t)stats->min_value_len);
                break;
        }
        if (cmp < 0) {
            *overlaps = false;
            return CARQUET_OK;
        }
    }

    /* Check if query min > stats max (no overlap) */
    if (min_value && stats->max_value && stats->max_value_len > 0) {
        int cmp;
        switch (type) {
            case CARQUET_PHYSICAL_INT32:
                cmp = compare_int32(min_value, stats->max_value);
                break;
            case CARQUET_PHYSICAL_INT64:
                cmp = compare_int64(min_value, stats->max_value);
                break;
            case CARQUET_PHYSICAL_FLOAT:
                cmp = compare_float(min_value, stats->max_value);
                break;
            case CARQUET_PHYSICAL_DOUBLE:
                cmp = compare_double(min_value, stats->max_value);
                break;
            default:
                cmp = compare_byte_array(min_value, value_len,
                    stats->max_value, (size_t)stats->max_value_len);
                break;
        }
        if (cmp > 0) {
            *overlaps = false;
            return CARQUET_OK;
        }
    }

    return CARQUET_OK;
}
