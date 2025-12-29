/**
 * @file plain.c
 * @brief PLAIN encoding implementation
 */

#include "plain.h"
#include "core/endian.h"
#include <string.h>

/* ============================================================================
 * PLAIN Decoding
 * ============================================================================
 */

int64_t carquet_decode_plain_boolean(
    const uint8_t* input,
    size_t input_size,
    uint8_t* output,
    int64_t count) {

    if (!input || !output || count < 0) {
        return -1;
    }

    /* Booleans are packed 8 per byte */
    size_t bytes_needed = ((size_t)count + 7) / 8;
    if (input_size < bytes_needed) {
        return -1;
    }

    int64_t i = 0;
    size_t byte_idx = 0;

    /* Process full bytes */
    while (i + 8 <= count) {
        uint8_t byte = input[byte_idx++];
        output[i++] = (byte >> 0) & 1;
        output[i++] = (byte >> 1) & 1;
        output[i++] = (byte >> 2) & 1;
        output[i++] = (byte >> 3) & 1;
        output[i++] = (byte >> 4) & 1;
        output[i++] = (byte >> 5) & 1;
        output[i++] = (byte >> 6) & 1;
        output[i++] = (byte >> 7) & 1;
    }

    /* Process remaining bits */
    if (i < count) {
        uint8_t byte = input[byte_idx++];
        int bit = 0;
        while (i < count) {
            output[i++] = (byte >> bit) & 1;
            bit++;
        }
    }

    return (int64_t)bytes_needed;
}

int64_t carquet_decode_plain_int32(
    const uint8_t* input,
    size_t input_size,
    int32_t* output,
    int64_t count) {

    if (!input || !output || count < 0) {
        return -1;
    }

    size_t bytes_needed = (size_t)count * 4;
    if (input_size < bytes_needed) {
        return -1;
    }

#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    /* Fast path: direct memory copy on little-endian systems */
    memcpy(output, input, bytes_needed);
#else
    for (int64_t i = 0; i < count; i++) {
        output[i] = carquet_read_i32_le(input + i * 4);
    }
#endif

    return (int64_t)bytes_needed;
}

int64_t carquet_decode_plain_int64(
    const uint8_t* input,
    size_t input_size,
    int64_t* output,
    int64_t count) {

    if (!input || !output || count < 0) {
        return -1;
    }

    size_t bytes_needed = (size_t)count * 8;
    if (input_size < bytes_needed) {
        return -1;
    }

#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    memcpy(output, input, bytes_needed);
#else
    for (int64_t i = 0; i < count; i++) {
        output[i] = carquet_read_i64_le(input + i * 8);
    }
#endif

    return (int64_t)bytes_needed;
}

int64_t carquet_decode_plain_int96(
    const uint8_t* input,
    size_t input_size,
    carquet_int96_t* output,
    int64_t count) {

    if (!input || !output || count < 0) {
        return -1;
    }

    size_t bytes_needed = (size_t)count * 12;
    if (input_size < bytes_needed) {
        return -1;
    }

    for (int64_t i = 0; i < count; i++) {
        const uint8_t* p = input + i * 12;
        output[i].value[0] = carquet_read_u32_le(p);
        output[i].value[1] = carquet_read_u32_le(p + 4);
        output[i].value[2] = carquet_read_u32_le(p + 8);
    }

    return (int64_t)bytes_needed;
}

int64_t carquet_decode_plain_float(
    const uint8_t* input,
    size_t input_size,
    float* output,
    int64_t count) {

    if (!input || !output || count < 0) {
        return -1;
    }

    size_t bytes_needed = (size_t)count * 4;
    if (input_size < bytes_needed) {
        return -1;
    }

#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    memcpy(output, input, bytes_needed);
#else
    for (int64_t i = 0; i < count; i++) {
        output[i] = carquet_read_f32_le(input + i * 4);
    }
#endif

    return (int64_t)bytes_needed;
}

int64_t carquet_decode_plain_double(
    const uint8_t* input,
    size_t input_size,
    double* output,
    int64_t count) {

    if (!input || !output || count < 0) {
        return -1;
    }

    size_t bytes_needed = (size_t)count * 8;
    if (input_size < bytes_needed) {
        return -1;
    }

#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    memcpy(output, input, bytes_needed);
#else
    for (int64_t i = 0; i < count; i++) {
        output[i] = carquet_read_f64_le(input + i * 8);
    }
#endif

    return (int64_t)bytes_needed;
}

int64_t carquet_decode_plain_byte_array(
    const uint8_t* input,
    size_t input_size,
    carquet_byte_array_t* output,
    int64_t count) {

    if (!input || !output || count < 0) {
        return -1;
    }

    size_t pos = 0;

    for (int64_t i = 0; i < count; i++) {
        /* Read 4-byte length prefix */
        if (pos + 4 > input_size) {
            return -1;
        }

        int32_t len = carquet_read_i32_le(input + pos);
        pos += 4;

        if (len < 0 || pos + (size_t)len > input_size) {
            return -1;
        }

        output[i].data = (uint8_t*)(input + pos);
        output[i].length = len;
        pos += (size_t)len;
    }

    return (int64_t)pos;
}

int64_t carquet_decode_plain_fixed_byte_array(
    const uint8_t* input,
    size_t input_size,
    uint8_t* output,
    int64_t count,
    int32_t fixed_len) {

    if (!input || !output || count < 0 || fixed_len <= 0) {
        return -1;
    }

    size_t bytes_needed = (size_t)count * (size_t)fixed_len;
    if (input_size < bytes_needed) {
        return -1;
    }

    memcpy(output, input, bytes_needed);
    return (int64_t)bytes_needed;
}

/* ============================================================================
 * PLAIN Encoding
 * ============================================================================
 */

carquet_status_t carquet_encode_plain_boolean(
    const uint8_t* input,
    int64_t count,
    carquet_buffer_t* output) {

    if (!input || !output || count < 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t bytes_needed = ((size_t)count + 7) / 8;
    uint8_t* dest = carquet_buffer_advance(output, bytes_needed);
    if (!dest) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    memset(dest, 0, bytes_needed);

    for (int64_t i = 0; i < count; i++) {
        if (input[i]) {
            dest[i / 8] |= (1 << (i % 8));
        }
    }

    return CARQUET_OK;
}

carquet_status_t carquet_encode_plain_int32(
    const int32_t* input,
    int64_t count,
    carquet_buffer_t* output) {

    if (!input || !output || count < 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t bytes_needed = (size_t)count * 4;

#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    return carquet_buffer_append(output, input, bytes_needed);
#else
    for (int64_t i = 0; i < count; i++) {
        carquet_status_t status = carquet_buffer_append_u32_le(output, (uint32_t)input[i]);
        if (status != CARQUET_OK) return status;
    }
    return CARQUET_OK;
#endif
}

carquet_status_t carquet_encode_plain_int64(
    const int64_t* input,
    int64_t count,
    carquet_buffer_t* output) {

    if (!input || !output || count < 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    return carquet_buffer_append(output, input, (size_t)count * 8);
#else
    for (int64_t i = 0; i < count; i++) {
        carquet_status_t status = carquet_buffer_append_u64_le(output, (uint64_t)input[i]);
        if (status != CARQUET_OK) return status;
    }
    return CARQUET_OK;
#endif
}

carquet_status_t carquet_encode_plain_int96(
    const carquet_int96_t* input,
    int64_t count,
    carquet_buffer_t* output) {

    if (!input || !output || count < 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    for (int64_t i = 0; i < count; i++) {
        carquet_status_t status;
        status = carquet_buffer_append_u32_le(output, input[i].value[0]);
        if (status != CARQUET_OK) return status;
        status = carquet_buffer_append_u32_le(output, input[i].value[1]);
        if (status != CARQUET_OK) return status;
        status = carquet_buffer_append_u32_le(output, input[i].value[2]);
        if (status != CARQUET_OK) return status;
    }

    return CARQUET_OK;
}

carquet_status_t carquet_encode_plain_float(
    const float* input,
    int64_t count,
    carquet_buffer_t* output) {

    if (!input || !output || count < 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    return carquet_buffer_append(output, input, (size_t)count * 4);
#else
    for (int64_t i = 0; i < count; i++) {
        carquet_status_t status = carquet_buffer_append_f32_le(output, input[i]);
        if (status != CARQUET_OK) return status;
    }
    return CARQUET_OK;
#endif
}

carquet_status_t carquet_encode_plain_double(
    const double* input,
    int64_t count,
    carquet_buffer_t* output) {

    if (!input || !output || count < 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    return carquet_buffer_append(output, input, (size_t)count * 8);
#else
    for (int64_t i = 0; i < count; i++) {
        carquet_status_t status = carquet_buffer_append_f64_le(output, input[i]);
        if (status != CARQUET_OK) return status;
    }
    return CARQUET_OK;
#endif
}

carquet_status_t carquet_encode_plain_byte_array(
    const carquet_byte_array_t* input,
    int64_t count,
    carquet_buffer_t* output) {

    if (!input || !output || count < 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    for (int64_t i = 0; i < count; i++) {
        carquet_status_t status = carquet_buffer_append_u32_le(output, (uint32_t)input[i].length);
        if (status != CARQUET_OK) return status;

        if (input[i].length > 0 && input[i].data) {
            status = carquet_buffer_append(output, input[i].data, (size_t)input[i].length);
            if (status != CARQUET_OK) return status;
        }
    }

    return CARQUET_OK;
}

carquet_status_t carquet_encode_plain_fixed_byte_array(
    const uint8_t* input,
    int64_t count,
    int32_t fixed_len,
    carquet_buffer_t* output) {

    if (!input || !output || count < 0 || fixed_len <= 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    return carquet_buffer_append(output, input, (size_t)count * (size_t)fixed_len);
}

/* ============================================================================
 * Generic PLAIN Function
 * ============================================================================
 */

int64_t carquet_decode_plain(
    const uint8_t* input,
    size_t input_size,
    carquet_physical_type_t type,
    int32_t type_length,
    void* output,
    int64_t count) {

    switch (type) {
        case CARQUET_PHYSICAL_BOOLEAN:
            return carquet_decode_plain_boolean(input, input_size,
                (uint8_t*)output, count);

        case CARQUET_PHYSICAL_INT32:
            return carquet_decode_plain_int32(input, input_size,
                (int32_t*)output, count);

        case CARQUET_PHYSICAL_INT64:
            return carquet_decode_plain_int64(input, input_size,
                (int64_t*)output, count);

        case CARQUET_PHYSICAL_INT96:
            return carquet_decode_plain_int96(input, input_size,
                (carquet_int96_t*)output, count);

        case CARQUET_PHYSICAL_FLOAT:
            return carquet_decode_plain_float(input, input_size,
                (float*)output, count);

        case CARQUET_PHYSICAL_DOUBLE:
            return carquet_decode_plain_double(input, input_size,
                (double*)output, count);

        case CARQUET_PHYSICAL_BYTE_ARRAY:
            return carquet_decode_plain_byte_array(input, input_size,
                (carquet_byte_array_t*)output, count);

        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            return carquet_decode_plain_fixed_byte_array(input, input_size,
                (uint8_t*)output, count, type_length);

        default:
            return -1;
    }
}
