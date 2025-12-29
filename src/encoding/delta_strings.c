/**
 * @file delta_strings.c
 * @brief DELTA_BYTE_ARRAY encoding implementation
 *
 * This encoding uses incremental (prefix sharing) encoding for strings.
 * It stores:
 * 1. Prefix lengths (common prefix with previous string) using DELTA_BINARY_PACKED
 * 2. Suffix lengths using DELTA_BINARY_PACKED
 * 3. All suffix data concatenated
 *
 * This is particularly efficient for sorted string columns where
 * adjacent strings often share common prefixes.
 *
 * Reference: https://parquet.apache.org/docs/file-format/data-pages/encodings/
 */

#include <carquet/error.h>
#include <carquet/types.h>
#include "core/buffer.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Forward declaration from delta.c */
extern carquet_status_t carquet_delta_decode_int32(
    const uint8_t* data,
    size_t data_size,
    int32_t* values,
    int32_t num_values,
    size_t* bytes_consumed);

extern carquet_status_t carquet_delta_encode_int32(
    const int32_t* values,
    int32_t num_values,
    uint8_t* data,
    size_t data_capacity,
    size_t* bytes_written);

/* ============================================================================
 * Helper Functions
 * ============================================================================
 */

/**
 * Find the length of common prefix between two byte arrays.
 */
static int32_t common_prefix_length(
    const uint8_t* a, uint32_t a_len,
    const uint8_t* b, uint32_t b_len) {

    uint32_t min_len = a_len < b_len ? a_len : b_len;
    int32_t prefix_len = 0;

    for (uint32_t i = 0; i < min_len; i++) {
        if (a[i] != b[i]) break;
        prefix_len++;
    }

    return prefix_len;
}

/* ============================================================================
 * DELTA_BYTE_ARRAY Decoder
 * ============================================================================
 */

/**
 * Decode DELTA_BYTE_ARRAY encoded data.
 *
 * @param data Input buffer containing encoded data
 * @param data_size Size of input buffer
 * @param values Output array of byte arrays (must be pre-allocated)
 * @param num_values Number of values to decode
 * @param work_buffer Work buffer for reconstructing strings
 * @param work_buffer_size Size of work buffer
 * @param bytes_consumed Output: number of input bytes consumed
 * @return Status code
 */
carquet_status_t carquet_delta_strings_decode(
    const uint8_t* data,
    size_t data_size,
    carquet_byte_array_t* values,
    int32_t num_values,
    uint8_t* work_buffer,
    size_t work_buffer_size,
    size_t* bytes_consumed) {

    if (!data || !values || num_values <= 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Allocate arrays for prefix and suffix lengths */
    int32_t* prefix_lengths = malloc(num_values * sizeof(int32_t));
    int32_t* suffix_lengths = malloc(num_values * sizeof(int32_t));

    if (!prefix_lengths || !suffix_lengths) {
        free(prefix_lengths);
        free(suffix_lengths);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    size_t pos = 0;

    /* Decode prefix lengths */
    size_t consumed = 0;
    carquet_status_t status = carquet_delta_decode_int32(
        data + pos, data_size - pos, prefix_lengths, num_values, &consumed);

    if (status != CARQUET_OK) {
        free(prefix_lengths);
        free(suffix_lengths);
        return status;
    }
    pos += consumed;

    /* Decode suffix lengths */
    status = carquet_delta_decode_int32(
        data + pos, data_size - pos, suffix_lengths, num_values, &consumed);

    if (status != CARQUET_OK) {
        free(prefix_lengths);
        free(suffix_lengths);
        return status;
    }
    pos += consumed;

    /* Calculate total suffix data size */
    size_t total_suffix_size = 0;
    for (int32_t i = 0; i < num_values; i++) {
        if (suffix_lengths[i] < 0 || prefix_lengths[i] < 0) {
            free(prefix_lengths);
            free(suffix_lengths);
            return CARQUET_ERROR_DECODE;
        }
        total_suffix_size += (size_t)suffix_lengths[i];
    }

    /* Check bounds */
    if (pos + total_suffix_size > data_size) {
        free(prefix_lengths);
        free(suffix_lengths);
        return CARQUET_ERROR_DECODE;
    }

    /* Reconstruct strings */
    const uint8_t* suffix_data = data + pos;
    size_t suffix_offset = 0;
    size_t work_offset = 0;
    uint8_t* prev_string = NULL;
    uint32_t prev_len = 0;

    for (int32_t i = 0; i < num_values; i++) {
        int32_t prefix_len = prefix_lengths[i];
        int32_t suffix_len = suffix_lengths[i];
        uint32_t total_len = (uint32_t)(prefix_len + suffix_len);

        /* Check work buffer space */
        if (work_offset + total_len > work_buffer_size) {
            free(prefix_lengths);
            free(suffix_lengths);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        uint8_t* dest = work_buffer + work_offset;

        /* Copy prefix from previous string */
        if (prefix_len > 0) {
            if (!prev_string || prefix_len > (int32_t)prev_len) {
                free(prefix_lengths);
                free(suffix_lengths);
                return CARQUET_ERROR_DECODE;
            }
            memcpy(dest, prev_string, prefix_len);
        }

        /* Copy suffix from encoded data */
        if (suffix_len > 0) {
            memcpy(dest + prefix_len, suffix_data + suffix_offset, suffix_len);
            suffix_offset += suffix_len;
        }

        values[i].data = dest;
        values[i].length = total_len;

        prev_string = dest;
        prev_len = total_len;
        work_offset += total_len;
    }

    free(prefix_lengths);
    free(suffix_lengths);

    if (bytes_consumed) {
        *bytes_consumed = pos + total_suffix_size;
    }

    return CARQUET_OK;
}

/* ============================================================================
 * DELTA_BYTE_ARRAY Encoder
 * ============================================================================
 */

/**
 * Encode byte arrays using DELTA_BYTE_ARRAY (incremental) encoding.
 *
 * @param values Input byte arrays to encode
 * @param num_values Number of values to encode
 * @param output Output buffer for encoded data
 * @return Status code
 */
carquet_status_t carquet_delta_strings_encode(
    const carquet_byte_array_t* values,
    int32_t num_values,
    carquet_buffer_t* output) {

    if (!values || !output || num_values <= 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Allocate arrays for prefix and suffix lengths */
    int32_t* prefix_lengths = malloc(num_values * sizeof(int32_t));
    int32_t* suffix_lengths = malloc(num_values * sizeof(int32_t));

    if (!prefix_lengths || !suffix_lengths) {
        free(prefix_lengths);
        free(suffix_lengths);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    /* Calculate prefix and suffix lengths */
    const uint8_t* prev_data = NULL;
    uint32_t prev_len = 0;

    for (int32_t i = 0; i < num_values; i++) {
        if (i == 0) {
            prefix_lengths[i] = 0;
            suffix_lengths[i] = (int32_t)values[i].length;
        } else {
            int32_t prefix_len = common_prefix_length(
                prev_data, prev_len,
                values[i].data, values[i].length);
            prefix_lengths[i] = prefix_len;
            suffix_lengths[i] = (int32_t)(values[i].length - prefix_len);
        }

        prev_data = values[i].data;
        prev_len = values[i].length;
    }

    /* Encode prefix lengths */
    size_t delta_capacity = (size_t)num_values * 10 + 100;
    uint8_t* delta_buffer = malloc(delta_capacity);
    if (!delta_buffer) {
        free(prefix_lengths);
        free(suffix_lengths);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    size_t bytes_written = 0;
    carquet_status_t status = carquet_delta_encode_int32(
        prefix_lengths, num_values, delta_buffer, delta_capacity, &bytes_written);

    if (status != CARQUET_OK) {
        free(prefix_lengths);
        free(suffix_lengths);
        free(delta_buffer);
        return status;
    }

    status = carquet_buffer_append(output, delta_buffer, bytes_written);
    if (status != CARQUET_OK) {
        free(prefix_lengths);
        free(suffix_lengths);
        free(delta_buffer);
        return status;
    }

    /* Encode suffix lengths */
    status = carquet_delta_encode_int32(
        suffix_lengths, num_values, delta_buffer, delta_capacity, &bytes_written);

    free(prefix_lengths);

    if (status != CARQUET_OK) {
        free(suffix_lengths);
        free(delta_buffer);
        return status;
    }

    status = carquet_buffer_append(output, delta_buffer, bytes_written);
    free(delta_buffer);

    if (status != CARQUET_OK) {
        free(suffix_lengths);
        return status;
    }

    /* Write suffix data */
    prev_len = 0;
    for (int32_t i = 0; i < num_values; i++) {
        int32_t prefix_len = (i == 0) ? 0 : (int32_t)common_prefix_length(
            values[i-1].data, values[i-1].length,
            values[i].data, values[i].length);
        int32_t suffix_len = suffix_lengths[i];

        if (suffix_len > 0 && values[i].data) {
            status = carquet_buffer_append(output,
                values[i].data + prefix_len, suffix_len);
            if (status != CARQUET_OK) {
                free(suffix_lengths);
                return status;
            }
        }
    }

    free(suffix_lengths);
    return CARQUET_OK;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================
 */

/**
 * Estimate work buffer size needed for decoding.
 *
 * @param values Array of byte arrays (with only length information needed)
 * @param num_values Number of values
 * @return Required work buffer size
 */
size_t carquet_delta_strings_work_buffer_size(
    const carquet_byte_array_t* values,
    int32_t num_values) {

    if (!values || num_values <= 0) {
        return 0;
    }

    size_t total = 0;
    for (int32_t i = 0; i < num_values; i++) {
        total += values[i].length;
    }

    return total;
}

/**
 * Estimate maximum encoded size for DELTA_BYTE_ARRAY.
 *
 * @param values Input byte arrays
 * @param num_values Number of values
 * @return Estimated maximum encoded size
 */
size_t carquet_delta_strings_max_encoded_size(
    const carquet_byte_array_t* values,
    int32_t num_values) {

    if (!values || num_values <= 0) {
        return 0;
    }

    /* Sum of all string lengths */
    size_t total_size = 0;
    for (int32_t i = 0; i < num_values; i++) {
        total_size += values[i].length;
    }

    /* Overhead for two delta-encoded integer arrays (prefix and suffix lengths) */
    size_t overhead = (size_t)num_values * 10 + 200;

    return total_size + overhead;
}
