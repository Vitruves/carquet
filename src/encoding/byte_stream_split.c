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
#include <stdlib.h>
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

    if (count < 0 || (uint64_t)count > SIZE_MAX / sizeof(float)) {
        return CARQUET_ERROR_DECODE;
    }
    size_t required_size = (size_t)count * sizeof(float);
    if (data_size < required_size) {
        return CARQUET_ERROR_DECODE;
    }

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

    if (count < 0 || (uint64_t)count > SIZE_MAX / sizeof(double)) {
        return CARQUET_ERROR_DECODE;
    }
    size_t required_size = (size_t)count * sizeof(double);
    if (data_size < required_size) {
        return CARQUET_ERROR_DECODE;
    }

    carquet_dispatch_byte_split_decode_double(data, count, values);

    return CARQUET_OK;
}

/* Decoding reads the S streams of a page directly. A cache-tiled variant
 * (gather a tile of every stream into a contiguous buffer, then transpose)
 * was slower at every page size measured: 1M doubles in 1.56 ms direct vs
 * 2.06 ms tiled on a Xeon D-1531 (256 KB L2), 1M floats 0.45 vs 0.73 ms, and
 * 19 vs 11 GB/s on an Apple M3: the hardware prefetchers follow the S
 * streams, and the gather was one more pass over the page. */

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

    /* The float/double fast paths reinterpret `values` as float* and double*. That
     * is only sound when the buffer is naturally aligned. INT32/INT64 callers
     * pass aligned buffers, but the FIXED_LEN_BYTE_ARRAY(4/8) caller passes a
     * raw byte buffer with no such guarantee -> UB / SIGBUS on strict-alignment
     * targets. Gate the fast path on alignment and fall through to the generic
     * byte transpose otherwise. */
    if (type_length == 4 && ((uintptr_t)values & 3u) == 0) {
        carquet_dispatch_byte_split_encode_float((const float*)values, count, output);
        *bytes_written = required_size;
        return CARQUET_OK;
    }

    if (type_length == 8 && ((uintptr_t)values & 7u) == 0) {
        carquet_dispatch_byte_split_encode_double((const double*)values, count, output);
        *bytes_written = required_size;
        return CARQUET_OK;
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

    if (count < 0 || (uint64_t)count > SIZE_MAX / (size_t)type_length) {
        return CARQUET_ERROR_DECODE;
    }
    size_t required_size = (size_t)count * (size_t)type_length;
    if (data_size < required_size) {
        return CARQUET_ERROR_DECODE;
    }

    /* See the encode path: the float/double fast paths reinterpret `values` as
     * float* and double* and are only sound on a naturally aligned buffer. The
     * FIXED_LEN_BYTE_ARRAY(4/8) caller passes a raw, possibly-unaligned byte
     * buffer, so gate on alignment and fall through to the generic transpose. */
    if (type_length == 4 && ((uintptr_t)values & 3u) == 0) {
        carquet_dispatch_byte_split_decode_float(data, count, (float*)values);
        return CARQUET_OK;
    }

    if (type_length == 8 && ((uintptr_t)values & 7u) == 0) {
        carquet_dispatch_byte_split_decode_double(data, count, (double*)values);
        return CARQUET_OK;
    }

    /* Un-transpose: gather byte streams back into values */
    for (int64_t i = 0; i < count; i++) {
        for (int b = 0; b < type_length; b++) {
            values[i * type_length + b] = data[b * count + i];
        }
    }

    return CARQUET_OK;
}
