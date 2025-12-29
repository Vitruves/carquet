/**
 * @file statistics.c
 * @brief Row group statistics access and predicate pushdown
 *
 * Provides access to column statistics for intelligent row group filtering.
 * This enables predicate pushdown, allowing queries to skip entire row groups
 * that cannot contain matching data.
 */

#include <carquet/carquet.h>
#include "reader_internal.h"
#include "thrift/parquet_types.h"
#include <string.h>

/* ============================================================================
 * Type-specific comparison
 * ============================================================================
 */

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
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

static int compare_double(const void* a, const void* b) {
    double va = *(const double*)a;
    double vb = *(const double*)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

static int compare_bytes(const void* a, size_t a_len, const void* b, size_t b_len) {
    size_t min_len = a_len < b_len ? a_len : b_len;
    int cmp = memcmp(a, b, min_len);
    if (cmp != 0) return cmp;
    return (a_len > b_len) - (a_len < b_len);
}

typedef int (*compare_fn_t)(const void*, const void*);

static compare_fn_t get_compare_fn(carquet_physical_type_t type) {
    switch (type) {
        case CARQUET_PHYSICAL_INT32:
        case CARQUET_PHYSICAL_BOOLEAN:
            return compare_int32;
        case CARQUET_PHYSICAL_INT64:
            return compare_int64;
        case CARQUET_PHYSICAL_FLOAT:
            return compare_float;
        case CARQUET_PHYSICAL_DOUBLE:
            return compare_double;
        default:
            return NULL;  /* Use byte comparison */
    }
}

/* ============================================================================
 * Statistics Access
 * ============================================================================
 */

carquet_status_t carquet_reader_column_statistics(
    const carquet_reader_t* reader,
    int32_t row_group_index,
    int32_t column_index,
    carquet_column_statistics_t* stats) {

    /* reader and stats are nonnull per API contract */
    if (row_group_index < 0 || row_group_index >= reader->metadata.num_row_groups) {
        return CARQUET_ERROR_ROW_GROUP_NOT_FOUND;
    }

    if (column_index < 0 || column_index >= reader->schema->num_leaves) {
        return CARQUET_ERROR_COLUMN_NOT_FOUND;
    }

    memset(stats, 0, sizeof(*stats));

    const parquet_row_group_t* rg = &reader->metadata.row_groups[row_group_index];
    if (column_index >= rg->num_columns) {
        return CARQUET_ERROR_COLUMN_NOT_FOUND;
    }

    const parquet_column_chunk_t* chunk = &rg->columns[column_index];
    if (!chunk->has_metadata) {
        return CARQUET_OK;  /* No statistics available */
    }

    const parquet_column_metadata_t* meta = &chunk->metadata;
    stats->num_values = meta->num_values;

    if (!meta->has_statistics) {
        return CARQUET_OK;
    }

    const parquet_statistics_t* pstats = &meta->statistics;

    /* Null count */
    if (pstats->has_null_count) {
        stats->has_null_count = true;
        stats->null_count = pstats->null_count;
    }

    /* Distinct count */
    if (pstats->has_distinct_count) {
        stats->has_distinct_count = true;
        stats->distinct_count = pstats->distinct_count;
    }

    /* Min/max values - prefer new format, fall back to deprecated */
    if (pstats->min_value && pstats->min_value_len > 0 &&
        pstats->max_value && pstats->max_value_len > 0) {
        stats->has_min_max = true;
        stats->min_value = pstats->min_value;
        stats->min_value_size = pstats->min_value_len;
        stats->max_value = pstats->max_value;
        stats->max_value_size = pstats->max_value_len;
    } else if (pstats->min_deprecated && pstats->min_deprecated_len > 0 &&
               pstats->max_deprecated && pstats->max_deprecated_len > 0) {
        stats->has_min_max = true;
        stats->min_value = pstats->min_deprecated;
        stats->min_value_size = pstats->min_deprecated_len;
        stats->max_value = pstats->max_deprecated;
        stats->max_value_size = pstats->max_deprecated_len;
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Predicate Pushdown
 * ============================================================================
 */

carquet_status_t carquet_reader_row_group_matches(
    const carquet_reader_t* reader,
    int32_t row_group_index,
    int32_t column_index,
    carquet_compare_op_t op,
    const void* value,
    int32_t value_size,
    bool* might_match) {

    /* reader, value, might_match are nonnull per API contract */
    /* Default: might match (conservative) */
    *might_match = true;

    /* Get column statistics */
    carquet_column_statistics_t stats;
    carquet_status_t status = carquet_reader_column_statistics(
        reader, row_group_index, column_index, &stats);

    if (status != CARQUET_OK) {
        return status;
    }

    /* If no min/max stats, we can't filter */
    if (!stats.has_min_max) {
        return CARQUET_OK;
    }

    /* Get column type */
    int32_t schema_idx = reader->schema->leaf_indices[column_index];
    const parquet_schema_element_t* elem = &reader->schema->elements[schema_idx];
    carquet_physical_type_t type = elem->has_type ? elem->type : CARQUET_PHYSICAL_BYTE_ARRAY;

    compare_fn_t cmp_fn = get_compare_fn(type);

    int cmp_min, cmp_max;

    if (cmp_fn) {
        cmp_min = cmp_fn(value, stats.min_value);
        cmp_max = cmp_fn(value, stats.max_value);
    } else {
        /* Byte comparison for variable-length types */
        cmp_min = compare_bytes(value, (size_t)value_size,
                                stats.min_value, (size_t)stats.min_value_size);
        cmp_max = compare_bytes(value, (size_t)value_size,
                                stats.max_value, (size_t)stats.max_value_size);
    }

    /*
     * Determine if row group can be skipped based on comparison:
     *
     * For value comparison against [min, max] range:
     * - EQ: skip if value < min OR value > max
     * - NE: skip if min == max == value (all values are the same)
     * - LT: skip if min >= value (all values >= value)
     * - LE: skip if min > value
     * - GT: skip if max <= value
     * - GE: skip if max < value
     */

    switch (op) {
        case CARQUET_COMPARE_EQ:
            /* value == x: skip if value not in [min, max] */
            if (cmp_min < 0 || cmp_max > 0) {
                *might_match = false;
            }
            break;

        case CARQUET_COMPARE_NE:
            /* value != x: skip only if all values equal x */
            if (cmp_min == 0 && cmp_max == 0) {
                /* min == max == value, all values equal the search value */
                *might_match = false;
            }
            break;

        case CARQUET_COMPARE_LT:
            /* x < value: skip if min >= value */
            if (cmp_min <= 0) {
                *might_match = false;
            }
            break;

        case CARQUET_COMPARE_LE:
            /* x <= value: skip if min > value */
            if (cmp_min < 0) {
                *might_match = false;
            }
            break;

        case CARQUET_COMPARE_GT:
            /* x > value: skip if max <= value */
            if (cmp_max >= 0) {
                *might_match = false;
            }
            break;

        case CARQUET_COMPARE_GE:
            /* x >= value: skip if max < value */
            if (cmp_max > 0) {
                *might_match = false;
            }
            break;
    }

    return CARQUET_OK;
}

int32_t carquet_reader_filter_row_groups(
    const carquet_reader_t* reader,
    int32_t column_index,
    carquet_compare_op_t op,
    const void* value,
    int32_t value_size,
    int32_t* matching_indices,
    int32_t max_indices) {

    /* reader, value, matching_indices are nonnull per API contract */
    if (max_indices <= 0) {
        return -1;
    }

    int32_t num_row_groups = carquet_reader_num_row_groups(reader);
    int32_t num_matching = 0;

    for (int32_t i = 0; i < num_row_groups && num_matching < max_indices; i++) {
        bool might_match = true;

        carquet_status_t status = carquet_reader_row_group_matches(
            reader, i, column_index, op, value, value_size, &might_match);

        if (status != CARQUET_OK) {
            /* On error, include row group (conservative) */
            might_match = true;
        }

        if (might_match) {
            matching_indices[num_matching++] = i;
        }
    }

    return num_matching;
}
