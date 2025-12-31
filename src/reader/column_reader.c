/**
 * @file column_reader.c
 * @brief Column reading implementation
 */

#include <carquet/carquet.h>
#include "reader_internal.h"
#include "thrift/parquet_types.h"
#include "encoding/plain.h"
#include "encoding/rle.h"
#include "core/endian.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Forward declaration */
extern carquet_status_t carquet_read_next_page(
    carquet_column_reader_t* reader,
    void* values,
    int64_t max_values,
    int16_t* def_levels,
    int16_t* rep_levels,
    int64_t* values_read,
    carquet_error_t* error);

/* ============================================================================
 * Batch Reading
 * ============================================================================
 */

int64_t carquet_column_read_batch(
    carquet_column_reader_t* reader,
    void* values,
    int64_t max_values,
    int16_t* def_levels,
    int16_t* rep_levels) {

    /* max_values < 0 is invalid; max_values = 0 is a "peek" to trigger page loading */
    if (max_values < 0) {
        return -1;
    }
    if (max_values == 0) {
        /* Load page if needed, but don't read any values */
        if (reader->values_remaining > 0 && !reader->page_loaded) {
            carquet_error_t error = CARQUET_ERROR_INIT;
            int64_t values_read = 0;
            uint8_t dummy[16];
            carquet_status_t status = carquet_read_next_page(
                reader, dummy, 0, NULL, NULL, &values_read, &error);
            (void)status;
        }
        return 0;
    }

    if (reader->values_remaining <= 0) {
        return 0;
    }

    carquet_error_t error = CARQUET_ERROR_INIT;
    int64_t total_read = 0;
    size_t value_size = 0;

    /* Determine value size for pointer arithmetic */
    switch (reader->type) {
        case CARQUET_PHYSICAL_BOOLEAN:
            value_size = 1;
            break;
        case CARQUET_PHYSICAL_INT32:
        case CARQUET_PHYSICAL_FLOAT:
            value_size = 4;
            break;
        case CARQUET_PHYSICAL_INT64:
        case CARQUET_PHYSICAL_DOUBLE:
            value_size = 8;
            break;
        case CARQUET_PHYSICAL_INT96:
            value_size = 12;
            break;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            value_size = reader->type_length;
            break;
        case CARQUET_PHYSICAL_BYTE_ARRAY:
            /* Variable length - handled differently */
            value_size = sizeof(carquet_byte_array_t);
            break;
        default:
            return -1;
    }

    /* Read pages until we have enough values or run out */
    while (total_read < max_values && reader->values_remaining > 0) {
        int64_t values_read = 0;
        int64_t to_read = max_values - total_read;

        uint8_t* value_ptr = (uint8_t*)values + total_read * value_size;
        int16_t* def_ptr = def_levels ? def_levels + total_read : NULL;
        int16_t* rep_ptr = rep_levels ? rep_levels + total_read : NULL;

        carquet_status_t status = carquet_read_next_page(
            reader, value_ptr, to_read, def_ptr, rep_ptr, &values_read, &error);

        if (status != CARQUET_OK) {
            if (total_read > 0) {
                /* Return what we have so far */
                break;
            }
            return -1;
        }

        if (values_read == 0) {
            break;
        }

        total_read += values_read;
    }

    return total_read;
}

/* ============================================================================
 * Skip Values
 * ============================================================================
 */

int64_t carquet_column_skip(
    carquet_column_reader_t* reader,
    int64_t num_values) {

    /* reader is nonnull per API contract */
    if (num_values <= 0 || reader->values_remaining <= 0) {
        return 0;
    }

    /* Allocate temporary buffer for skipping */
    size_t value_size = 0;
    switch (reader->type) {
        case CARQUET_PHYSICAL_BOOLEAN:
            value_size = 1;
            break;
        case CARQUET_PHYSICAL_INT32:
        case CARQUET_PHYSICAL_FLOAT:
            value_size = 4;
            break;
        case CARQUET_PHYSICAL_INT64:
        case CARQUET_PHYSICAL_DOUBLE:
            value_size = 8;
            break;
        case CARQUET_PHYSICAL_INT96:
            value_size = 12;
            break;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            value_size = reader->type_length;
            break;
        case CARQUET_PHYSICAL_BYTE_ARRAY:
            value_size = sizeof(carquet_byte_array_t);
            break;
        default:
            return 0;
    }

    /* Read and discard values in chunks */
    int64_t total_skipped = 0;
    int64_t chunk_size = 1024;

    void* temp = malloc(chunk_size * value_size);
    if (!temp) {
        return 0;
    }

    while (total_skipped < num_values && reader->values_remaining > 0) {
        int64_t to_skip = num_values - total_skipped;
        if (to_skip > chunk_size) {
            to_skip = chunk_size;
        }

        int64_t skipped = carquet_column_read_batch(reader, temp, to_skip, NULL, NULL);
        if (skipped <= 0) {
            break;
        }
        total_skipped += skipped;
    }

    free(temp);
    return total_skipped;
}
