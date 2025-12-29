/**
 * @file test_encodings_extended.c
 * @brief Extended tests for Parquet encodings
 *
 * Tests for:
 * - DELTA_BINARY_PACKED encoding
 * - Dictionary encoding
 * - BYTE_STREAM_SPLIT encoding
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <carquet/error.h>
#include <carquet/types.h>
#include "core/buffer.h"

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* ============================================================================
 * Delta Encoding Function Declarations
 * ============================================================================
 */

carquet_status_t carquet_delta_encode_int32(
    const int32_t* values,
    int32_t num_values,
    uint8_t* data,
    size_t data_capacity,
    size_t* bytes_written);

carquet_status_t carquet_delta_decode_int32(
    const uint8_t* data,
    size_t data_size,
    int32_t* values,
    int32_t num_values,
    size_t* bytes_consumed);

carquet_status_t carquet_delta_encode_int64(
    const int64_t* values,
    int32_t num_values,
    uint8_t* data,
    size_t data_capacity,
    size_t* bytes_written);

carquet_status_t carquet_delta_decode_int64(
    const uint8_t* data,
    size_t data_size,
    int64_t* values,
    int32_t num_values,
    size_t* bytes_consumed);

/* ============================================================================
 * Dictionary Encoding Function Declarations
 * ============================================================================
 */

carquet_status_t carquet_dictionary_encode_int32(
    const int32_t* values,
    int64_t count,
    carquet_buffer_t* dict_output,
    carquet_buffer_t* indices_output);

carquet_status_t carquet_dictionary_decode_int32(
    const uint8_t* dict_data,
    size_t dict_size,
    int32_t dict_count,
    const uint8_t* indices_data,
    size_t indices_size,
    int32_t* output,
    int64_t output_count);

carquet_status_t carquet_dictionary_encode_int64(
    const int64_t* values,
    int64_t count,
    carquet_buffer_t* dict_output,
    carquet_buffer_t* indices_output);

carquet_status_t carquet_dictionary_decode_int64(
    const uint8_t* dict_data,
    size_t dict_size,
    int32_t dict_count,
    const uint8_t* indices_data,
    size_t indices_size,
    int64_t* output,
    int64_t output_count);

carquet_status_t carquet_dictionary_encode_float(
    const float* values,
    int64_t count,
    carquet_buffer_t* dict_output,
    carquet_buffer_t* indices_output);

carquet_status_t carquet_dictionary_decode_float(
    const uint8_t* dict_data,
    size_t dict_size,
    int32_t dict_count,
    const uint8_t* indices_data,
    size_t indices_size,
    float* output,
    int64_t output_count);

carquet_status_t carquet_dictionary_encode_double(
    const double* values,
    int64_t count,
    carquet_buffer_t* dict_output,
    carquet_buffer_t* indices_output);

carquet_status_t carquet_dictionary_decode_double(
    const uint8_t* dict_data,
    size_t dict_size,
    int32_t dict_count,
    const uint8_t* indices_data,
    size_t indices_size,
    double* output,
    int64_t output_count);

/* ============================================================================
 * Byte Stream Split Function Declarations
 * ============================================================================
 */

carquet_status_t carquet_byte_stream_split_encode_float(
    const float* values,
    int64_t count,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written);

carquet_status_t carquet_byte_stream_split_decode_float(
    const uint8_t* data,
    size_t data_size,
    float* values,
    int64_t count);

carquet_status_t carquet_byte_stream_split_encode_double(
    const double* values,
    int64_t count,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written);

carquet_status_t carquet_byte_stream_split_decode_double(
    const uint8_t* data,
    size_t data_size,
    double* values,
    int64_t count);

carquet_status_t carquet_byte_stream_split_encode(
    const uint8_t* values,
    int64_t count,
    int32_t type_length,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written);

carquet_status_t carquet_byte_stream_split_decode(
    const uint8_t* data,
    size_t data_size,
    int32_t type_length,
    uint8_t* values,
    int64_t count);

/* ============================================================================
 * Delta Encoding Tests
 * ============================================================================
 */

static int test_delta_int32_sequential(void) {
    int32_t input[100];
    for (int i = 0; i < 100; i++) {
        input[i] = i;
    }

    uint8_t encoded[4096];
    size_t bytes_written;

    carquet_status_t status = carquet_delta_encode_int32(
        input, 100, encoded, sizeof(encoded), &bytes_written);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_int32_sequential", "encode failed");
    }

    /* Sequential data should compress well */
    printf("  [DEBUG] Sequential: 100 int32s -> %zu bytes\n", bytes_written);

    int32_t output[100];
    size_t bytes_consumed;
    status = carquet_delta_decode_int32(
        encoded, bytes_written, output, 100, &bytes_consumed);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_int32_sequential", "decode failed");
    }

    for (int i = 0; i < 100; i++) {
        if (output[i] != input[i]) {
            printf("  [DEBUG] Mismatch at %d: expected %d, got %d\n", i, input[i], output[i]);
            TEST_FAIL("delta_int32_sequential", "value mismatch");
        }
    }

    TEST_PASS("delta_int32_sequential");
    return 0;
}

static int test_delta_int32_negative(void) {
    int32_t input[] = {-100, -50, 0, 50, 100, -200, -150, -100};
    int count = sizeof(input) / sizeof(input[0]);

    uint8_t encoded[4096];
    size_t bytes_written;

    carquet_status_t status = carquet_delta_encode_int32(
        input, count, encoded, sizeof(encoded), &bytes_written);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_int32_negative", "encode failed");
    }

    int32_t output[8];
    size_t bytes_consumed;
    status = carquet_delta_decode_int32(
        encoded, bytes_written, output, count, &bytes_consumed);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_int32_negative", "decode failed");
    }

    for (int i = 0; i < count; i++) {
        if (output[i] != input[i]) {
            printf("  [DEBUG] Mismatch at %d: expected %d, got %d\n", i, input[i], output[i]);
            TEST_FAIL("delta_int32_negative", "value mismatch");
        }
    }

    TEST_PASS("delta_int32_negative");
    return 0;
}

static int test_delta_int32_large_jumps(void) {
    int32_t input[] = {0, 1000000, 2000000, 1500000, 0, -1000000};
    int count = sizeof(input) / sizeof(input[0]);

    uint8_t encoded[4096];
    size_t bytes_written;

    carquet_status_t status = carquet_delta_encode_int32(
        input, count, encoded, sizeof(encoded), &bytes_written);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_int32_large_jumps", "encode failed");
    }

    int32_t output[6];
    size_t bytes_consumed;
    status = carquet_delta_decode_int32(
        encoded, bytes_written, output, count, &bytes_consumed);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_int32_large_jumps", "decode failed");
    }

    for (int i = 0; i < count; i++) {
        if (output[i] != input[i]) {
            TEST_FAIL("delta_int32_large_jumps", "value mismatch");
        }
    }

    TEST_PASS("delta_int32_large_jumps");
    return 0;
}

static int test_delta_int64_sequential(void) {
    int64_t input[100];
    for (int i = 0; i < 100; i++) {
        input[i] = (int64_t)i * 1000000000LL;
    }

    uint8_t encoded[8192];
    size_t bytes_written;

    carquet_status_t status = carquet_delta_encode_int64(
        input, 100, encoded, sizeof(encoded), &bytes_written);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_int64_sequential", "encode failed");
    }

    printf("  [DEBUG] Sequential int64: 100 values -> %zu bytes\n", bytes_written);

    int64_t output[100];
    size_t bytes_consumed;
    status = carquet_delta_decode_int64(
        encoded, bytes_written, output, 100, &bytes_consumed);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_int64_sequential", "decode failed");
    }

    for (int i = 0; i < 100; i++) {
        if (output[i] != input[i]) {
            printf("  [DEBUG] Mismatch at %d: expected %lld, got %lld\n",
                   i, (long long)input[i], (long long)output[i]);
            TEST_FAIL("delta_int64_sequential", "value mismatch");
        }
    }

    TEST_PASS("delta_int64_sequential");
    return 0;
}

static int test_delta_int64_timestamps(void) {
    /* Simulate millisecond timestamps with small variations */
    int64_t base = 1704067200000LL; /* 2024-01-01 00:00:00 UTC */
    int64_t input[50];
    for (int i = 0; i < 50; i++) {
        input[i] = base + i * 1000 + (i % 3) * 10;
    }

    uint8_t encoded[8192];
    size_t bytes_written;

    carquet_status_t status = carquet_delta_encode_int64(
        input, 50, encoded, sizeof(encoded), &bytes_written);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_int64_timestamps", "encode failed");
    }

    printf("  [DEBUG] Timestamps: 50 values -> %zu bytes\n", bytes_written);

    int64_t output[50];
    size_t bytes_consumed;
    status = carquet_delta_decode_int64(
        encoded, bytes_written, output, 50, &bytes_consumed);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_int64_timestamps", "decode failed");
    }

    for (int i = 0; i < 50; i++) {
        if (output[i] != input[i]) {
            TEST_FAIL("delta_int64_timestamps", "value mismatch");
        }
    }

    TEST_PASS("delta_int64_timestamps");
    return 0;
}

static int test_delta_single_value(void) {
    int32_t input[] = {42};

    uint8_t encoded[256];
    size_t bytes_written;

    carquet_status_t status = carquet_delta_encode_int32(
        input, 1, encoded, sizeof(encoded), &bytes_written);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_single_value", "encode failed");
    }

    int32_t output[1];
    size_t bytes_consumed;
    status = carquet_delta_decode_int32(
        encoded, bytes_written, output, 1, &bytes_consumed);
    if (status != CARQUET_OK) {
        TEST_FAIL("delta_single_value", "decode failed");
    }

    if (output[0] != input[0]) {
        TEST_FAIL("delta_single_value", "value mismatch");
    }

    TEST_PASS("delta_single_value");
    return 0;
}

/* ============================================================================
 * Dictionary Encoding Tests
 * ============================================================================
 */

static int test_dictionary_int32_unique(void) {
    int32_t input[] = {100, 200, 300, 400, 500};
    int count = sizeof(input) / sizeof(input[0]);

    carquet_buffer_t dict_buf, indices_buf;
    carquet_buffer_init(&dict_buf);
    carquet_buffer_init(&indices_buf);

    carquet_status_t status = carquet_dictionary_encode_int32(
        input, count, &dict_buf, &indices_buf);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_int32_unique", "encode failed");
    }

    printf("  [DEBUG] Dict size: %zu, indices size: %zu\n",
           carquet_buffer_size(&dict_buf), carquet_buffer_size(&indices_buf));

    int32_t output[5];
    status = carquet_dictionary_decode_int32(
        carquet_buffer_data_const(&dict_buf), carquet_buffer_size(&dict_buf),
        count,
        carquet_buffer_data_const(&indices_buf), carquet_buffer_size(&indices_buf),
        output, count);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_int32_unique", "decode failed");
    }

    for (int i = 0; i < count; i++) {
        if (output[i] != input[i]) {
            TEST_FAIL("dictionary_int32_unique", "value mismatch");
        }
    }

    carquet_buffer_destroy(&dict_buf);
    carquet_buffer_destroy(&indices_buf);
    TEST_PASS("dictionary_int32_unique");
    return 0;
}

static int test_dictionary_int32_repeated(void) {
    int32_t input[] = {1, 2, 1, 2, 1, 2, 3, 3, 3, 1};
    int count = sizeof(input) / sizeof(input[0]);

    carquet_buffer_t dict_buf, indices_buf;
    carquet_buffer_init(&dict_buf);
    carquet_buffer_init(&indices_buf);

    carquet_status_t status = carquet_dictionary_encode_int32(
        input, count, &dict_buf, &indices_buf);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_int32_repeated", "encode failed");
    }

    /* Should have only 3 unique values in dictionary */
    size_t expected_dict_size = 3 * sizeof(int32_t);
    if (carquet_buffer_size(&dict_buf) != expected_dict_size) {
        printf("  [DEBUG] Expected dict size %zu, got %zu\n",
               expected_dict_size, carquet_buffer_size(&dict_buf));
        TEST_FAIL("dictionary_int32_repeated", "dictionary size unexpected");
    }

    int32_t output[10];
    status = carquet_dictionary_decode_int32(
        carquet_buffer_data_const(&dict_buf), carquet_buffer_size(&dict_buf),
        3, /* dict_count */
        carquet_buffer_data_const(&indices_buf), carquet_buffer_size(&indices_buf),
        output, count);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_int32_repeated", "decode failed");
    }

    for (int i = 0; i < count; i++) {
        if (output[i] != input[i]) {
            printf("  [DEBUG] Mismatch at %d: expected %d, got %d\n", i, input[i], output[i]);
            TEST_FAIL("dictionary_int32_repeated", "value mismatch");
        }
    }

    carquet_buffer_destroy(&dict_buf);
    carquet_buffer_destroy(&indices_buf);
    TEST_PASS("dictionary_int32_repeated");
    return 0;
}

static int test_dictionary_int64(void) {
    int64_t input[] = {1000000000000LL, 2000000000000LL, 1000000000000LL, 3000000000000LL};
    int count = sizeof(input) / sizeof(input[0]);

    carquet_buffer_t dict_buf, indices_buf;
    carquet_buffer_init(&dict_buf);
    carquet_buffer_init(&indices_buf);

    carquet_status_t status = carquet_dictionary_encode_int64(
        input, count, &dict_buf, &indices_buf);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_int64", "encode failed");
    }

    int64_t output[4];
    status = carquet_dictionary_decode_int64(
        carquet_buffer_data_const(&dict_buf), carquet_buffer_size(&dict_buf),
        3, /* 3 unique values */
        carquet_buffer_data_const(&indices_buf), carquet_buffer_size(&indices_buf),
        output, count);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_int64", "decode failed");
    }

    for (int i = 0; i < count; i++) {
        if (output[i] != input[i]) {
            TEST_FAIL("dictionary_int64", "value mismatch");
        }
    }

    carquet_buffer_destroy(&dict_buf);
    carquet_buffer_destroy(&indices_buf);
    TEST_PASS("dictionary_int64");
    return 0;
}

static int test_dictionary_float(void) {
    float input[] = {1.0f, 2.0f, 1.0f, 3.0f, 2.0f, 1.0f};
    int count = sizeof(input) / sizeof(input[0]);

    carquet_buffer_t dict_buf, indices_buf;
    carquet_buffer_init(&dict_buf);
    carquet_buffer_init(&indices_buf);

    carquet_status_t status = carquet_dictionary_encode_float(
        input, count, &dict_buf, &indices_buf);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_float", "encode failed");
    }

    float output[6];
    status = carquet_dictionary_decode_float(
        carquet_buffer_data_const(&dict_buf), carquet_buffer_size(&dict_buf),
        3, /* 3 unique values */
        carquet_buffer_data_const(&indices_buf), carquet_buffer_size(&indices_buf),
        output, count);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_float", "decode failed");
    }

    for (int i = 0; i < count; i++) {
        if (output[i] != input[i]) {
            TEST_FAIL("dictionary_float", "value mismatch");
        }
    }

    carquet_buffer_destroy(&dict_buf);
    carquet_buffer_destroy(&indices_buf);
    TEST_PASS("dictionary_float");
    return 0;
}

static int test_dictionary_double(void) {
    double input[] = {3.14159, 2.71828, 3.14159, 1.41421, 2.71828};
    int count = sizeof(input) / sizeof(input[0]);

    carquet_buffer_t dict_buf, indices_buf;
    carquet_buffer_init(&dict_buf);
    carquet_buffer_init(&indices_buf);

    carquet_status_t status = carquet_dictionary_encode_double(
        input, count, &dict_buf, &indices_buf);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_double", "encode failed");
    }

    double output[5];
    status = carquet_dictionary_decode_double(
        carquet_buffer_data_const(&dict_buf), carquet_buffer_size(&dict_buf),
        3, /* 3 unique values */
        carquet_buffer_data_const(&indices_buf), carquet_buffer_size(&indices_buf),
        output, count);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_double", "decode failed");
    }

    for (int i = 0; i < count; i++) {
        if (output[i] != input[i]) {
            TEST_FAIL("dictionary_double", "value mismatch");
        }
    }

    carquet_buffer_destroy(&dict_buf);
    carquet_buffer_destroy(&indices_buf);
    TEST_PASS("dictionary_double");
    return 0;
}

static int test_dictionary_many_values(void) {
    /* Test with 1000 values but only 10 unique */
    int32_t input[1000];
    for (int i = 0; i < 1000; i++) {
        input[i] = (i * 17) % 10;
    }

    carquet_buffer_t dict_buf, indices_buf;
    carquet_buffer_init(&dict_buf);
    carquet_buffer_init(&indices_buf);

    carquet_status_t status = carquet_dictionary_encode_int32(
        input, 1000, &dict_buf, &indices_buf);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_many_values", "encode failed");
    }

    printf("  [DEBUG] 1000 values, 10 unique: dict=%zu bytes, indices=%zu bytes\n",
           carquet_buffer_size(&dict_buf), carquet_buffer_size(&indices_buf));

    int32_t output[1000];
    status = carquet_dictionary_decode_int32(
        carquet_buffer_data_const(&dict_buf), carquet_buffer_size(&dict_buf),
        10, /* 10 unique values */
        carquet_buffer_data_const(&indices_buf), carquet_buffer_size(&indices_buf),
        output, 1000);
    if (status != CARQUET_OK) {
        TEST_FAIL("dictionary_many_values", "decode failed");
    }

    for (int i = 0; i < 1000; i++) {
        if (output[i] != input[i]) {
            TEST_FAIL("dictionary_many_values", "value mismatch");
        }
    }

    carquet_buffer_destroy(&dict_buf);
    carquet_buffer_destroy(&indices_buf);
    TEST_PASS("dictionary_many_values");
    return 0;
}

/* ============================================================================
 * Byte Stream Split Tests
 * ============================================================================
 */

static int test_byte_stream_split_float(void) {
    float input[] = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    int count = sizeof(input) / sizeof(input[0]);

    uint8_t encoded[256];
    size_t bytes_written;

    carquet_status_t status = carquet_byte_stream_split_encode_float(
        input, count, encoded, sizeof(encoded), &bytes_written);
    if (status != CARQUET_OK) {
        TEST_FAIL("byte_stream_split_float", "encode failed");
    }

    if (bytes_written != count * sizeof(float)) {
        TEST_FAIL("byte_stream_split_float", "unexpected output size");
    }

    float output[5];
    status = carquet_byte_stream_split_decode_float(
        encoded, bytes_written, output, count);
    if (status != CARQUET_OK) {
        TEST_FAIL("byte_stream_split_float", "decode failed");
    }

    for (int i = 0; i < count; i++) {
        if (output[i] != input[i]) {
            printf("  [DEBUG] Mismatch at %d: expected %f, got %f\n", i, input[i], output[i]);
            TEST_FAIL("byte_stream_split_float", "value mismatch");
        }
    }

    TEST_PASS("byte_stream_split_float");
    return 0;
}

static int test_byte_stream_split_double(void) {
    double input[] = {1.123456789, 2.234567890, 3.345678901, 4.456789012};
    int count = sizeof(input) / sizeof(input[0]);

    uint8_t encoded[256];
    size_t bytes_written;

    carquet_status_t status = carquet_byte_stream_split_encode_double(
        input, count, encoded, sizeof(encoded), &bytes_written);
    if (status != CARQUET_OK) {
        TEST_FAIL("byte_stream_split_double", "encode failed");
    }

    if (bytes_written != count * sizeof(double)) {
        TEST_FAIL("byte_stream_split_double", "unexpected output size");
    }

    double output[4];
    status = carquet_byte_stream_split_decode_double(
        encoded, bytes_written, output, count);
    if (status != CARQUET_OK) {
        TEST_FAIL("byte_stream_split_double", "decode failed");
    }

    for (int i = 0; i < count; i++) {
        if (output[i] != input[i]) {
            TEST_FAIL("byte_stream_split_double", "value mismatch");
        }
    }

    TEST_PASS("byte_stream_split_double");
    return 0;
}

static int test_byte_stream_split_generic(void) {
    /* Test with 16-bit fixed length values */
    uint8_t input[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    int count = 4; /* 4 values of 2 bytes each */

    uint8_t encoded[256];
    size_t bytes_written;

    carquet_status_t status = carquet_byte_stream_split_encode(
        input, count, 2, encoded, sizeof(encoded), &bytes_written);
    if (status != CARQUET_OK) {
        TEST_FAIL("byte_stream_split_generic", "encode failed");
    }

    /* Expected: all first bytes, then all second bytes */
    /* Input: [0x01,0x02], [0x03,0x04], [0x05,0x06], [0x07,0x08] */
    /* Expected: [0x01,0x03,0x05,0x07], [0x02,0x04,0x06,0x08] */
    uint8_t expected[] = {0x01, 0x03, 0x05, 0x07, 0x02, 0x04, 0x06, 0x08};
    if (memcmp(encoded, expected, 8) != 0) {
        printf("  [DEBUG] Encoded data mismatch\n");
        for (int i = 0; i < 8; i++) {
            printf("    [%d] expected 0x%02x, got 0x%02x\n", i, expected[i], encoded[i]);
        }
        TEST_FAIL("byte_stream_split_generic", "encoded data mismatch");
    }

    uint8_t output[8];
    status = carquet_byte_stream_split_decode(
        encoded, bytes_written, 2, output, count);
    if (status != CARQUET_OK) {
        TEST_FAIL("byte_stream_split_generic", "decode failed");
    }

    if (memcmp(output, input, 8) != 0) {
        TEST_FAIL("byte_stream_split_generic", "decoded data mismatch");
    }

    TEST_PASS("byte_stream_split_generic");
    return 0;
}

static int test_byte_stream_split_special_floats(void) {
    float input[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN};
    int count = 5;

    uint8_t encoded[256];
    size_t bytes_written;

    carquet_status_t status = carquet_byte_stream_split_encode_float(
        input, count, encoded, sizeof(encoded), &bytes_written);
    if (status != CARQUET_OK) {
        TEST_FAIL("byte_stream_split_special_floats", "encode failed");
    }

    float output[5];
    status = carquet_byte_stream_split_decode_float(
        encoded, bytes_written, output, count);
    if (status != CARQUET_OK) {
        TEST_FAIL("byte_stream_split_special_floats", "decode failed");
    }

    /* Check zero, negative zero, infinity, negative infinity */
    if (output[0] != 0.0f) TEST_FAIL("byte_stream_split_special_floats", "zero mismatch");
    if (output[2] != INFINITY) TEST_FAIL("byte_stream_split_special_floats", "inf mismatch");
    if (output[3] != -INFINITY) TEST_FAIL("byte_stream_split_special_floats", "-inf mismatch");
    if (!isnan(output[4])) TEST_FAIL("byte_stream_split_special_floats", "nan mismatch");

    TEST_PASS("byte_stream_split_special_floats");
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================
 */

int main(void) {
    int failures = 0;

    printf("=== Extended Encoding Tests ===\n\n");

    printf("--- Delta Encoding Tests ---\n");
    failures += test_delta_int32_sequential();
    failures += test_delta_int32_negative();
    failures += test_delta_int32_large_jumps();
    failures += test_delta_int64_sequential();
    failures += test_delta_int64_timestamps();
    failures += test_delta_single_value();

    printf("\n--- Dictionary Encoding Tests ---\n");
    failures += test_dictionary_int32_unique();
    failures += test_dictionary_int32_repeated();
    failures += test_dictionary_int64();
    failures += test_dictionary_float();
    failures += test_dictionary_double();
    failures += test_dictionary_many_values();

    printf("\n--- Byte Stream Split Tests ---\n");
    failures += test_byte_stream_split_float();
    failures += test_byte_stream_split_double();
    failures += test_byte_stream_split_generic();
    failures += test_byte_stream_split_special_floats();

    printf("\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
