/**
 * @file byte_stream_split.c
 * @brief BYTE_STREAM_SPLIT encoding implementation
 *
 * This encoding transposes byte streams for better compression of floating-point data.
 * For N values of size S bytes each, the encoding interleaves bytes:
 * - All first bytes of each value, then all second bytes, etc.
 *
 * Example with 3 floats (A1A2A3A4, B1B2B3B4, C1C2C3C4):
 * Encoded: A1B1C1 A2B2C2 A3B3C3 A4B4C4
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* SIMD dispatch functions */
extern void carquet_dispatch_byte_split_encode_float(const float* values, int64_t count, uint8_t* output);
extern void carquet_dispatch_byte_split_decode_float(const uint8_t* data, int64_t count, float* values);
extern void carquet_dispatch_byte_split_encode_double(const double* values, int64_t count, uint8_t* output);
extern void carquet_dispatch_byte_split_decode_double(const uint8_t* data, int64_t count, double* values);

/* ============================================================================
 * Float Encoding (32-bit, 4 bytes)
 * ============================================================================
 */

carquet_status_t carquet_byte_stream_split_encode_float(
    const float* values,
    int64_t count,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written) {

    if (!values || !output || !bytes_written) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t required_size = (size_t)count * sizeof(float);
    if (output_capacity < required_size) {
        return CARQUET_ERROR_ENCODE;
    }

    /* Use SIMD-optimized transpose */
    carquet_dispatch_byte_split_encode_float(values, count, output);

    *bytes_written = required_size;
    return CARQUET_OK;
}

carquet_status_t carquet_byte_stream_split_decode_float(
    const uint8_t* data,
    size_t data_size,
    float* values,
    int64_t count) {

    if (!data || !values) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t required_size = (size_t)count * sizeof(float);
    if (data_size < required_size) {
        return CARQUET_ERROR_DECODE;
    }

    /* Use SIMD-optimized un-transpose */
    carquet_dispatch_byte_split_decode_float(data, count, values);

    return CARQUET_OK;
}

/* ============================================================================
 * Double Encoding (64-bit, 8 bytes)
 * ============================================================================
 */

carquet_status_t carquet_byte_stream_split_encode_double(
    const double* values,
    int64_t count,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written) {

    if (!values || !output || !bytes_written) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t required_size = (size_t)count * sizeof(double);
    if (output_capacity < required_size) {
        return CARQUET_ERROR_ENCODE;
    }

    /* Use SIMD-optimized transpose */
    carquet_dispatch_byte_split_encode_double(values, count, output);

    *bytes_written = required_size;
    return CARQUET_OK;
}

carquet_status_t carquet_byte_stream_split_decode_double(
    const uint8_t* data,
    size_t data_size,
    double* values,
    int64_t count) {

    if (!data || !values) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t required_size = (size_t)count * sizeof(double);
    if (data_size < required_size) {
        return CARQUET_ERROR_DECODE;
    }

    /* Use SIMD-optimized un-transpose */
    carquet_dispatch_byte_split_decode_double(data, count, values);

    return CARQUET_OK;
}

/* ============================================================================
 * Fixed Length Byte Array Encoding (generic)
 * ============================================================================
 */

carquet_status_t carquet_byte_stream_split_encode(
    const uint8_t* values,
    int64_t count,
    int32_t type_length,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written) {

    if (!values || !output || !bytes_written || type_length <= 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t required_size = (size_t)count * (size_t)type_length;
    if (output_capacity < required_size) {
        return CARQUET_ERROR_ENCODE;
    }

    /* Transpose: put byte 0 of all values, then byte 1, etc. */
    for (int b = 0; b < type_length; b++) {
        for (int64_t i = 0; i < count; i++) {
            output[b * count + i] = values[i * type_length + b];
        }
    }

    *bytes_written = required_size;
    return CARQUET_OK;
}

carquet_status_t carquet_byte_stream_split_decode(
    const uint8_t* data,
    size_t data_size,
    int32_t type_length,
    uint8_t* values,
    int64_t count) {

    if (!data || !values || type_length <= 0) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t required_size = (size_t)count * (size_t)type_length;
    if (data_size < required_size) {
        return CARQUET_ERROR_DECODE;
    }

    /* Un-transpose: gather byte streams back into values */
    for (int64_t i = 0; i < count; i++) {
        for (int b = 0; b < type_length; b++) {
            values[i * type_length + b] = data[b * count + i];
        }
    }

    return CARQUET_OK;
}
