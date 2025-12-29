/**
 * @file test_encodings.c
 * @brief Tests for Parquet encodings
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "encoding/plain.h"
#include "encoding/rle.h"
#include "core/buffer.h"

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* ============================================================================
 * PLAIN Encoding Tests
 * ============================================================================
 */

static int test_plain_int32(void) {
    int32_t input[] = {1, 2, 3, 4, 5, -100, 0, 2147483647};
    int count = sizeof(input) / sizeof(input[0]);

    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    /* Encode */
    assert(carquet_encode_plain_int32(input, count, &buf) == CARQUET_OK);
    assert(carquet_buffer_size(&buf) == (size_t)(count * 4));

    /* Decode */
    int32_t output[8];
    int64_t bytes = carquet_decode_plain_int32(
        carquet_buffer_data_const(&buf), carquet_buffer_size(&buf),
        output, count);
    (void)bytes;

    assert(bytes == count * 4);
    for (int i = 0; i < count; i++) {
        assert(output[i] == input[i]);
    }

    carquet_buffer_destroy(&buf);
    TEST_PASS("plain_int32");
    return 0;
}

static int test_plain_int64(void) {
    int64_t input[] = {1, -1, 0x7FFFFFFFFFFFFFFFLL, (int64_t)0x8000000000000000LL};
    int count = sizeof(input) / sizeof(input[0]);

    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    assert(carquet_encode_plain_int64(input, count, &buf) == CARQUET_OK);

    int64_t output[4];
    int64_t bytes = carquet_decode_plain_int64(
        carquet_buffer_data_const(&buf), carquet_buffer_size(&buf),
        output, count);
    (void)bytes;

    assert(bytes == count * 8);
    for (int i = 0; i < count; i++) {
        assert(output[i] == input[i]);
    }

    carquet_buffer_destroy(&buf);
    TEST_PASS("plain_int64");
    return 0;
}

static int test_plain_boolean(void) {
    uint8_t input[] = {1, 0, 1, 1, 0, 0, 1, 0, 1};  /* 9 booleans */
    int count = sizeof(input);

    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    assert(carquet_encode_plain_boolean(input, count, &buf) == CARQUET_OK);
    assert(carquet_buffer_size(&buf) == 2);  /* 9 bits = 2 bytes */

    uint8_t output[9];
    int64_t bytes = carquet_decode_plain_boolean(
        carquet_buffer_data_const(&buf), carquet_buffer_size(&buf),
        output, count);
    (void)bytes;

    assert(bytes == 2);
    for (int i = 0; i < count; i++) {
        assert(output[i] == input[i]);
    }

    carquet_buffer_destroy(&buf);
    TEST_PASS("plain_boolean");
    return 0;
}

static int test_plain_double(void) {
    double input[] = {0.0, 1.0, -1.0, 3.14159265359, 1e100};
    int count = sizeof(input) / sizeof(input[0]);

    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    assert(carquet_encode_plain_double(input, count, &buf) == CARQUET_OK);

    double output[5];
    int64_t bytes = carquet_decode_plain_double(
        carquet_buffer_data_const(&buf), carquet_buffer_size(&buf),
        output, count);
    (void)bytes;

    assert(bytes == count * 8);
    for (int i = 0; i < count; i++) {
        assert(output[i] == input[i]);
    }

    carquet_buffer_destroy(&buf);
    TEST_PASS("plain_double");
    return 0;
}

/* ============================================================================
 * RLE Encoding Tests
 * ============================================================================
 */

static int test_rle_repeated_values(void) {
    /* 100 repeated zeros */
    uint32_t input[100];
    memset(input, 0, sizeof(input));

    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    assert(carquet_rle_encode_all(input, 100, 1, &buf) == CARQUET_OK);

    /* Should be very compact */
    assert(carquet_buffer_size(&buf) < 10);

    uint32_t output[100];
    int64_t count = carquet_rle_decode_all(
        carquet_buffer_data_const(&buf), carquet_buffer_size(&buf),
        1, output, 100);
    (void)count;

    assert(count == 100);
    for (int i = 0; i < 100; i++) {
        assert(output[i] == 0);
    }

    carquet_buffer_destroy(&buf);
    TEST_PASS("rle_repeated_values");
    return 0;
}

static int test_rle_alternating(void) {
    /* Alternating 0 and 1 */
    uint32_t input[16];
    for (int i = 0; i < 16; i++) {
        input[i] = i % 2;
    }

    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    assert(carquet_rle_encode_all(input, 16, 1, &buf) == CARQUET_OK);

    uint32_t output[16];
    int64_t count = carquet_rle_decode_all(
        carquet_buffer_data_const(&buf), carquet_buffer_size(&buf),
        1, output, 16);
    (void)count;

    assert(count == 16);
    for (int i = 0; i < 16; i++) {
        assert(output[i] == (uint32_t)(i % 2));
    }

    carquet_buffer_destroy(&buf);
    TEST_PASS("rle_alternating");
    return 0;
}

static int test_rle_decoder_skip(void) {
    /* Create RLE data with repeated values */
    uint32_t input[100];
    for (int i = 0; i < 100; i++) {
        input[i] = i / 10;  /* 0,0,0,...,1,1,1,...,9,9,9,... */
    }

    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    assert(carquet_rle_encode_all(input, 100, 4, &buf) == CARQUET_OK);

    carquet_rle_decoder_t dec;
    carquet_rle_decoder_init(&dec,
        carquet_buffer_data_const(&buf), carquet_buffer_size(&buf), 4);

    /* Skip first 25 values */
    int64_t skipped = carquet_rle_decoder_skip(&dec, 25);
    (void)skipped;
    assert(skipped == 25);

    /* Read next 10 values (should be 2s and 3s) */
    uint32_t output[10];
    int64_t read = carquet_rle_decoder_get_batch(&dec, output, 10);
    (void)read;
    assert(read == 10);

    for (int i = 0; i < 5; i++) {
        assert(output[i] == 2);
    }
    for (int i = 5; i < 10; i++) {
        assert(output[i] == 3);
    }

    carquet_buffer_destroy(&buf);
    TEST_PASS("rle_decoder_skip");
    return 0;
}

static int test_rle_levels(void) {
    int16_t input[] = {0, 0, 1, 0, 1, 1, 0, 0, 1, 0};
    int count = sizeof(input) / sizeof(input[0]);

    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    assert(carquet_rle_encode_levels(input, count, 1, &buf) == CARQUET_OK);

    int16_t output[10];
    int64_t decoded = carquet_rle_decode_levels(
        carquet_buffer_data_const(&buf), carquet_buffer_size(&buf),
        1, output, count);
    (void)decoded;

    assert(decoded == count);
    for (int i = 0; i < count; i++) {
        assert(output[i] == input[i]);
    }

    carquet_buffer_destroy(&buf);
    TEST_PASS("rle_levels");
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================
 */

int main(void) {
    int failures = 0;

    printf("=== Encoding Tests ===\n\n");

    /* PLAIN encoding tests */
    failures += test_plain_int32();
    failures += test_plain_int64();
    failures += test_plain_boolean();
    failures += test_plain_double();

    /* RLE encoding tests */
    failures += test_rle_repeated_values();
    failures += test_rle_alternating();
    failures += test_rle_decoder_skip();
    failures += test_rle_levels();

    printf("\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
