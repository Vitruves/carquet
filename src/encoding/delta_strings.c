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

#include "core/allocator.h"
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

    if (!data || num_values < 0 || (num_values > 0 && !values)) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Empty (all-null) page: consume both DELTA headers (prefix + suffix
     * length sub-streams) for correct byte accounting, then yield zero
     * values. */
    if (num_values == 0) {
        size_t c1 = 0, c2 = 0;
        carquet_status_t s = carquet_delta_decode_int32(
            data, data_size, NULL, 0, &c1);
        if (s != CARQUET_OK) return s;
        s = carquet_delta_decode_int32(
            data + c1, data_size - c1, NULL, 0, &c2);
        if (s != CARQUET_OK) return s;
        if (bytes_consumed) *bytes_consumed = c1 + c2;
        return CARQUET_OK;
    }

    /* Allocate arrays for prefix and suffix lengths. num_values comes from the
     * (untrusted) page header; guard the multiply so it cannot overflow size_t
     * and yield an undersized buffer (only reachable where size_t is 32-bit). */
    if ((size_t)num_values > SIZE_MAX / sizeof(int32_t)) {
        return CARQUET_ERROR_DECODE;
    }
    int32_t* prefix_lengths = carquet_mem_malloc((size_t)num_values * sizeof(int32_t));
    int32_t* suffix_lengths = carquet_mem_malloc((size_t)num_values * sizeof(int32_t));

    if (!prefix_lengths || !suffix_lengths) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    size_t pos = 0;

    /* Decode prefix lengths */
    size_t consumed = 0;
    carquet_status_t status = carquet_delta_decode_int32(
        data + pos, data_size - pos, prefix_lengths, num_values, &consumed);

    if (status != CARQUET_OK) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
        return status;
    }
    pos += consumed;

    /* Decode suffix lengths */
    status = carquet_delta_decode_int32(
        data + pos, data_size - pos, suffix_lengths, num_values, &consumed);

    if (status != CARQUET_OK) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
        return status;
    }
    pos += consumed;

    /* Calculate total suffix data size */
    size_t total_suffix_size = 0;
    for (int32_t i = 0; i < num_values; i++) {
        if (suffix_lengths[i] < 0 || prefix_lengths[i] < 0) {
            carquet_mem_free(prefix_lengths);
            carquet_mem_free(suffix_lengths);
            return CARQUET_ERROR_DECODE;
        }
        total_suffix_size += (size_t)suffix_lengths[i];
    }

    /* Check bounds */
    if (pos + total_suffix_size > data_size) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
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
            carquet_mem_free(prefix_lengths);
            carquet_mem_free(suffix_lengths);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        uint8_t* dest = work_buffer + work_offset;

        /* Copy prefix from previous string */
        if (prefix_len > 0) {
            if (!prev_string || prefix_len > (int32_t)prev_len) {
                carquet_mem_free(prefix_lengths);
                carquet_mem_free(suffix_lengths);
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

    carquet_mem_free(prefix_lengths);
    carquet_mem_free(suffix_lengths);

    if (bytes_consumed) {
        *bytes_consumed = pos + total_suffix_size;
    }

    return CARQUET_OK;
}

/**
 * Compute the exact work buffer size required to decode a DELTA_BYTE_ARRAY
 * page, without reconstructing the strings.
 *
 * It decodes only the prefix/suffix length headers and sums (prefix+suffix)
 * over all values, which is exactly the number of bytes the reconstruction
 * step writes into the work buffer. This gives a precise, safe size so the
 * caller never under-allocates (avoids OUT_OF_MEMORY) nor wildly over-allocates.
 *
 * @param data Input buffer containing encoded data
 * @param data_size Size of input buffer
 * @param num_values Number of values to decode
 * @param required_size Output: exact work buffer size in bytes
 * @return Status code
 */
carquet_status_t carquet_delta_strings_decoded_size(
    const uint8_t* data,
    size_t data_size,
    int32_t num_values,
    size_t* required_size) {

    if (!data || !required_size || num_values <= 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    int32_t* prefix_lengths = carquet_mem_malloc((size_t)num_values * sizeof(int32_t));
    int32_t* suffix_lengths = carquet_mem_malloc((size_t)num_values * sizeof(int32_t));
    if (!prefix_lengths || !suffix_lengths) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    size_t pos = 0;
    size_t consumed = 0;
    carquet_status_t status = carquet_delta_decode_int32(
        data + pos, data_size - pos, prefix_lengths, num_values, &consumed);
    if (status != CARQUET_OK) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
        return status;
    }
    pos += consumed;

    status = carquet_delta_decode_int32(
        data + pos, data_size - pos, suffix_lengths, num_values, &consumed);
    if (status != CARQUET_OK) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
        return status;
    }

    size_t total = 0;
    for (int32_t i = 0; i < num_values; i++) {
        if (prefix_lengths[i] < 0 || suffix_lengths[i] < 0) {
            carquet_mem_free(prefix_lengths);
            carquet_mem_free(suffix_lengths);
            return CARQUET_ERROR_DECODE;
        }
        total += (size_t)prefix_lengths[i] + (size_t)suffix_lengths[i];
    }

    carquet_mem_free(prefix_lengths);
    carquet_mem_free(suffix_lengths);

    *required_size = total;
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

    if (!output || num_values < 0 || (num_values > 0 && !values)) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* An all-null (zero value) page still needs both DELTA headers
     * (prefix-length and suffix-length sub-streams) so the decoder doesn't
     * hit EOF; emit them with no trailing suffix bytes. */
    if (num_values == 0) {
        uint8_t header[64];
        size_t written = 0;
        carquet_status_t s = carquet_delta_encode_int32(
            NULL, 0, header, sizeof(header), &written);
        if (s != CARQUET_OK) return s;
        s = carquet_buffer_append(output, header, written);   /* prefix */
        if (s != CARQUET_OK) return s;
        return carquet_buffer_append(output, header, written); /* suffix */
    }

    /* Allocate arrays for prefix and suffix lengths */
    int32_t* prefix_lengths = carquet_mem_malloc(num_values * sizeof(int32_t));
    int32_t* suffix_lengths = carquet_mem_malloc(num_values * sizeof(int32_t));

    if (!prefix_lengths || !suffix_lengths) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
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
    uint8_t* delta_buffer = carquet_mem_malloc(delta_capacity);
    if (!delta_buffer) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    size_t bytes_written = 0;
    carquet_status_t status = carquet_delta_encode_int32(
        prefix_lengths, num_values, delta_buffer, delta_capacity, &bytes_written);

    if (status != CARQUET_OK) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
        carquet_mem_free(delta_buffer);
        return status;
    }

    status = carquet_buffer_append(output, delta_buffer, bytes_written);
    if (status != CARQUET_OK) {
        carquet_mem_free(prefix_lengths);
        carquet_mem_free(suffix_lengths);
        carquet_mem_free(delta_buffer);
        return status;
    }

    /* Encode suffix lengths */
    status = carquet_delta_encode_int32(
        suffix_lengths, num_values, delta_buffer, delta_capacity, &bytes_written);

    carquet_mem_free(prefix_lengths);

    if (status != CARQUET_OK) {
        carquet_mem_free(suffix_lengths);
        carquet_mem_free(delta_buffer);
        return status;
    }

    status = carquet_buffer_append(output, delta_buffer, bytes_written);
    carquet_mem_free(delta_buffer);

    if (status != CARQUET_OK) {
        carquet_mem_free(suffix_lengths);
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
                carquet_mem_free(suffix_lengths);
                return status;
            }
        }
    }

    carquet_mem_free(suffix_lengths);
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
