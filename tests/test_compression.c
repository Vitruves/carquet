/**
 * @file test_compression.c
 * @brief Tests for compression codecs
 *
 * Tests for:
 * - LZ4 compression/decompression
 * - Snappy compression/decompression
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <carquet/error.h>

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* ============================================================================
 * LZ4 Function Declarations
 * ============================================================================
 */

carquet_status_t carquet_lz4_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size);

carquet_status_t carquet_lz4_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size);

size_t carquet_lz4_compress_bound(size_t src_size);

/* ============================================================================
 * Snappy Function Declarations
 * ============================================================================
 */

carquet_status_t carquet_snappy_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size);

carquet_status_t carquet_snappy_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size);

size_t carquet_snappy_compress_bound(size_t src_size);

carquet_status_t carquet_snappy_get_uncompressed_length(
    const uint8_t* src,
    size_t src_size,
    size_t* length);

/* ============================================================================
 * GZIP Function Declarations
 * ============================================================================
 */

int carquet_gzip_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size,
    int level);

int carquet_gzip_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size);

size_t carquet_gzip_compress_bound(size_t src_size);

/* ============================================================================
 * ZSTD Function Declarations
 * ============================================================================
 */

int carquet_zstd_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size,
    int level);

int carquet_zstd_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size);

size_t carquet_zstd_compress_bound(size_t src_size);

/* ============================================================================
 * Test Helpers
 * ============================================================================
 */

static void fill_random_data(uint8_t* data, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; i++) {
        data[i] = (uint8_t)(rand() % 256);
    }
}

static void fill_compressible_data(uint8_t* data, size_t size) {
    /* Create highly compressible data with repeated patterns */
    const char* pattern = "Hello, World! This is a test pattern. ";
    size_t pattern_len = strlen(pattern);

    for (size_t i = 0; i < size; i++) {
        data[i] = (uint8_t)pattern[i % pattern_len];
    }
}

static void fill_zeros(uint8_t* data, size_t size) {
    memset(data, 0, size);
}

/* ============================================================================
 * LZ4 Tests
 * ============================================================================
 */

static int test_lz4_small_literal(void) {
    /* Very small input that won't have matches */
    uint8_t input[] = "Hello";
    size_t input_size = 5;

    size_t bound = carquet_lz4_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    carquet_status_t status = carquet_lz4_compress(
        input, input_size, compressed, bound, &compressed_size);
    if (status != CARQUET_OK) {
        free(compressed);
        TEST_FAIL("lz4_small_literal", "compress failed");
    }

    printf("  [DEBUG] LZ4 small: %zu -> %zu bytes\n", input_size, compressed_size);

    uint8_t output[16];
    size_t output_size;
    status = carquet_lz4_decompress(
        compressed, compressed_size, output, sizeof(output), &output_size);
    if (status != CARQUET_OK) {
        free(compressed);
        TEST_FAIL("lz4_small_literal", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(compressed);
        TEST_FAIL("lz4_small_literal", "data mismatch");
    }

    free(compressed);
    TEST_PASS("lz4_small_literal");
    return 0;
}

static int test_lz4_compressible(void) {
    size_t input_size = 4096;
    uint8_t* input = malloc(input_size);
    fill_compressible_data(input, input_size);

    size_t bound = carquet_lz4_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    carquet_status_t status = carquet_lz4_compress(
        input, input_size, compressed, bound, &compressed_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("lz4_compressible", "compress failed");
    }

    printf("  [DEBUG] LZ4 compressible: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    /* Should compress well */
    if (compressed_size >= input_size) {
        printf("  [WARNING] LZ4 did not compress compressible data\n");
    }

    uint8_t* output = malloc(input_size + 1024); /* Extra space for safety */
    memset(output, 0xAA, input_size + 1024); /* Fill with pattern */
    size_t output_size;
    status = carquet_lz4_decompress(
        compressed, compressed_size, output, input_size + 1024, &output_size);
    if (status != CARQUET_OK) {
        printf("  [DEBUG] LZ4 decompress failed with status %d\n", status);
        /* Dump first bytes of compressed data */
        printf("  [DEBUG] Compressed data (first 32 bytes): ");
        for (size_t i = 0; i < (compressed_size < 32 ? compressed_size : 32); i++) {
            printf("%02x ", compressed[i]);
        }
        printf("\n");
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("lz4_compressible", "decompress failed");
    }

    printf("  [DEBUG] Decompressed size: %zu (expected %zu)\n", output_size, input_size);

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("lz4_compressible", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("lz4_compressible");
    return 0;
}

static int test_lz4_random(void) {
    size_t input_size = 2048;
    uint8_t* input = malloc(input_size);
    fill_random_data(input, input_size, 12345);

    size_t bound = carquet_lz4_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    carquet_status_t status = carquet_lz4_compress(
        input, input_size, compressed, bound, &compressed_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("lz4_random", "compress failed");
    }

    printf("  [DEBUG] LZ4 random: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    uint8_t* output = malloc(input_size);
    size_t output_size;
    status = carquet_lz4_decompress(
        compressed, compressed_size, output, input_size, &output_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("lz4_random", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("lz4_random", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("lz4_random");
    return 0;
}

static int test_lz4_zeros(void) {
    size_t input_size = 8192;
    uint8_t* input = malloc(input_size);
    fill_zeros(input, input_size);

    size_t bound = carquet_lz4_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    carquet_status_t status = carquet_lz4_compress(
        input, input_size, compressed, bound, &compressed_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("lz4_zeros", "compress failed");
    }

    printf("  [DEBUG] LZ4 zeros: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    /* Zeros should compress extremely well */
    if (compressed_size > input_size / 10) {
        printf("  [WARNING] LZ4 compression of zeros was poor\n");
    }

    uint8_t* output = malloc(input_size);
    size_t output_size;
    status = carquet_lz4_decompress(
        compressed, compressed_size, output, input_size, &output_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("lz4_zeros", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("lz4_zeros", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("lz4_zeros");
    return 0;
}

static int test_lz4_empty(void) {
    uint8_t* input = NULL;
    size_t input_size = 0;

    uint8_t compressed[64];
    size_t compressed_size;

    /* Empty input should work */
    carquet_status_t status = carquet_lz4_compress(
        input, input_size, compressed, sizeof(compressed), &compressed_size);
    if (status != CARQUET_OK) {
        TEST_FAIL("lz4_empty", "compress failed");
    }

    uint8_t output[64];
    size_t output_size;
    status = carquet_lz4_decompress(
        compressed, compressed_size, output, sizeof(output), &output_size);
    if (status != CARQUET_OK) {
        TEST_FAIL("lz4_empty", "decompress failed");
    }

    if (output_size != 0) {
        TEST_FAIL("lz4_empty", "output not empty");
    }

    TEST_PASS("lz4_empty");
    return 0;
}

/* ============================================================================
 * Snappy Tests
 * ============================================================================
 */

static int test_snappy_small_literal(void) {
    uint8_t input[] = "Hello, World!";
    size_t input_size = strlen((char*)input);

    size_t bound = carquet_snappy_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    carquet_status_t status = carquet_snappy_compress(
        input, input_size, compressed, bound, &compressed_size);
    if (status != CARQUET_OK) {
        free(compressed);
        TEST_FAIL("snappy_small_literal", "compress failed");
    }

    printf("  [DEBUG] Snappy small: %zu -> %zu bytes\n", input_size, compressed_size);

    /* Verify uncompressed length */
    size_t uncompressed_len;
    status = carquet_snappy_get_uncompressed_length(
        compressed, compressed_size, &uncompressed_len);
    if (status != CARQUET_OK) {
        free(compressed);
        TEST_FAIL("snappy_small_literal", "get_uncompressed_length failed");
    }
    if (uncompressed_len != input_size) {
        free(compressed);
        TEST_FAIL("snappy_small_literal", "uncompressed length mismatch");
    }

    uint8_t output[64];
    size_t output_size;
    status = carquet_snappy_decompress(
        compressed, compressed_size, output, sizeof(output), &output_size);
    if (status != CARQUET_OK) {
        free(compressed);
        TEST_FAIL("snappy_small_literal", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(compressed);
        TEST_FAIL("snappy_small_literal", "data mismatch");
    }

    free(compressed);
    TEST_PASS("snappy_small_literal");
    return 0;
}

static int test_snappy_compressible(void) {
    size_t input_size = 4096;
    uint8_t* input = malloc(input_size);
    fill_compressible_data(input, input_size);

    size_t bound = carquet_snappy_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    carquet_status_t status = carquet_snappy_compress(
        input, input_size, compressed, bound, &compressed_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("snappy_compressible", "compress failed");
    }

    printf("  [DEBUG] Snappy compressible: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    uint8_t* output = malloc(input_size);
    size_t output_size;
    status = carquet_snappy_decompress(
        compressed, compressed_size, output, input_size, &output_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("snappy_compressible", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("snappy_compressible", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("snappy_compressible");
    return 0;
}

static int test_snappy_random(void) {
    size_t input_size = 2048;
    uint8_t* input = malloc(input_size);
    fill_random_data(input, input_size, 54321);

    size_t bound = carquet_snappy_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    carquet_status_t status = carquet_snappy_compress(
        input, input_size, compressed, bound, &compressed_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("snappy_random", "compress failed");
    }

    printf("  [DEBUG] Snappy random: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    uint8_t* output = malloc(input_size);
    size_t output_size;
    status = carquet_snappy_decompress(
        compressed, compressed_size, output, input_size, &output_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("snappy_random", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("snappy_random", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("snappy_random");
    return 0;
}

static int test_snappy_zeros(void) {
    size_t input_size = 8192;
    uint8_t* input = malloc(input_size);
    fill_zeros(input, input_size);

    size_t bound = carquet_snappy_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    carquet_status_t status = carquet_snappy_compress(
        input, input_size, compressed, bound, &compressed_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("snappy_zeros", "compress failed");
    }

    printf("  [DEBUG] Snappy zeros: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    uint8_t* output = malloc(input_size);
    size_t output_size;
    status = carquet_snappy_decompress(
        compressed, compressed_size, output, input_size, &output_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("snappy_zeros", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("snappy_zeros", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("snappy_zeros");
    return 0;
}

static int test_snappy_empty(void) {
    uint8_t* input = NULL;
    size_t input_size = 0;

    uint8_t compressed[64];
    size_t compressed_size;

    carquet_status_t status = carquet_snappy_compress(
        input, input_size, compressed, sizeof(compressed), &compressed_size);
    if (status != CARQUET_OK) {
        TEST_FAIL("snappy_empty", "compress failed");
    }

    uint8_t output[64];
    size_t output_size;
    status = carquet_snappy_decompress(
        compressed, compressed_size, output, sizeof(output), &output_size);
    if (status != CARQUET_OK) {
        TEST_FAIL("snappy_empty", "decompress failed");
    }

    if (output_size != 0) {
        TEST_FAIL("snappy_empty", "output not empty");
    }

    TEST_PASS("snappy_empty");
    return 0;
}

static int test_snappy_large(void) {
    size_t input_size = 65536;
    uint8_t* input = malloc(input_size);

    /* Mix of compressible and random data */
    fill_compressible_data(input, input_size / 2);
    fill_random_data(input + input_size / 2, input_size / 2, 99999);

    size_t bound = carquet_snappy_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    carquet_status_t status = carquet_snappy_compress(
        input, input_size, compressed, bound, &compressed_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("snappy_large", "compress failed");
    }

    printf("  [DEBUG] Snappy large: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    uint8_t* output = malloc(input_size);
    size_t output_size;
    status = carquet_snappy_decompress(
        compressed, compressed_size, output, input_size, &output_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("snappy_large", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("snappy_large", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("snappy_large");
    return 0;
}

/* ============================================================================
 * GZIP Tests
 * ============================================================================
 */

static int test_gzip_small_literal(void) {
    uint8_t input[] = "Hello, World!";
    size_t input_size = strlen((char*)input);

    size_t bound = carquet_gzip_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    int status = carquet_gzip_compress(
        input, input_size, compressed, bound, &compressed_size, 6);
    if (status != CARQUET_OK) {
        free(compressed);
        TEST_FAIL("gzip_small_literal", "compress failed");
    }

    printf("  [DEBUG] GZIP small: %zu -> %zu bytes\n", input_size, compressed_size);

    uint8_t output[64];
    size_t output_size;
    status = carquet_gzip_decompress(
        compressed, compressed_size, output, sizeof(output), &output_size);
    if (status != CARQUET_OK) {
        free(compressed);
        TEST_FAIL("gzip_small_literal", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(compressed);
        TEST_FAIL("gzip_small_literal", "data mismatch");
    }

    free(compressed);
    TEST_PASS("gzip_small_literal");
    return 0;
}

static int test_gzip_compressible(void) {
    size_t input_size = 4096;
    uint8_t* input = malloc(input_size);
    fill_compressible_data(input, input_size);

    size_t bound = carquet_gzip_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    int status = carquet_gzip_compress(
        input, input_size, compressed, bound, &compressed_size, 6);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("gzip_compressible", "compress failed");
    }

    printf("  [DEBUG] GZIP compressible: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    uint8_t* output = malloc(input_size);
    size_t output_size;
    status = carquet_gzip_decompress(
        compressed, compressed_size, output, input_size, &output_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("gzip_compressible", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("gzip_compressible", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("gzip_compressible");
    return 0;
}

static int test_gzip_zeros(void) {
    size_t input_size = 8192;
    uint8_t* input = malloc(input_size);
    fill_zeros(input, input_size);

    size_t bound = carquet_gzip_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    int status = carquet_gzip_compress(
        input, input_size, compressed, bound, &compressed_size, 6);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("gzip_zeros", "compress failed");
    }

    printf("  [DEBUG] GZIP zeros: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    uint8_t* output = malloc(input_size);
    size_t output_size;
    status = carquet_gzip_decompress(
        compressed, compressed_size, output, input_size, &output_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("gzip_zeros", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("gzip_zeros", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("gzip_zeros");
    return 0;
}

/* ============================================================================
 * ZSTD Tests
 * ============================================================================
 */

static int test_zstd_small_literal(void) {
    uint8_t input[] = "Hello, World!";
    size_t input_size = strlen((char*)input);

    size_t bound = carquet_zstd_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    int status = carquet_zstd_compress(
        input, input_size, compressed, bound, &compressed_size, 3);
    if (status != CARQUET_OK) {
        free(compressed);
        TEST_FAIL("zstd_small_literal", "compress failed");
    }

    printf("  [DEBUG] ZSTD small: %zu -> %zu bytes\n", input_size, compressed_size);

    uint8_t output[64];
    size_t output_size;
    status = carquet_zstd_decompress(
        compressed, compressed_size, output, sizeof(output), &output_size);
    if (status != CARQUET_OK) {
        free(compressed);
        TEST_FAIL("zstd_small_literal", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(compressed);
        TEST_FAIL("zstd_small_literal", "data mismatch");
    }

    free(compressed);
    TEST_PASS("zstd_small_literal");
    return 0;
}

static int test_zstd_compressible(void) {
    size_t input_size = 4096;
    uint8_t* input = malloc(input_size);
    fill_compressible_data(input, input_size);

    size_t bound = carquet_zstd_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    int status = carquet_zstd_compress(
        input, input_size, compressed, bound, &compressed_size, 3);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("zstd_compressible", "compress failed");
    }

    printf("  [DEBUG] ZSTD compressible: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    uint8_t* output = malloc(input_size);
    size_t output_size;
    status = carquet_zstd_decompress(
        compressed, compressed_size, output, input_size, &output_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("zstd_compressible", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("zstd_compressible", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("zstd_compressible");
    return 0;
}

static int test_zstd_zeros(void) {
    size_t input_size = 8192;
    uint8_t* input = malloc(input_size);
    fill_zeros(input, input_size);

    size_t bound = carquet_zstd_compress_bound(input_size);
    uint8_t* compressed = malloc(bound);
    size_t compressed_size;

    int status = carquet_zstd_compress(
        input, input_size, compressed, bound, &compressed_size, 3);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        TEST_FAIL("zstd_zeros", "compress failed");
    }

    printf("  [DEBUG] ZSTD zeros: %zu -> %zu bytes (%.1f%%)\n",
           input_size, compressed_size, 100.0 * compressed_size / input_size);

    uint8_t* output = malloc(input_size);
    size_t output_size;
    status = carquet_zstd_decompress(
        compressed, compressed_size, output, input_size, &output_size);
    if (status != CARQUET_OK) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("zstd_zeros", "decompress failed");
    }

    if (output_size != input_size || memcmp(output, input, input_size) != 0) {
        free(input);
        free(compressed);
        free(output);
        TEST_FAIL("zstd_zeros", "data mismatch");
    }

    free(input);
    free(compressed);
    free(output);
    TEST_PASS("zstd_zeros");
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================
 */

int main(void) {
    int failures = 0;

    printf("=== Compression Tests ===\n\n");

    printf("--- LZ4 Tests ---\n");
    failures += test_lz4_small_literal();
    failures += test_lz4_compressible();
    failures += test_lz4_random();
    failures += test_lz4_zeros();
    failures += test_lz4_empty();

    printf("\n--- Snappy Tests ---\n");
    failures += test_snappy_small_literal();
    failures += test_snappy_compressible();
    failures += test_snappy_random();
    failures += test_snappy_zeros();
    failures += test_snappy_empty();
    failures += test_snappy_large();

    printf("\n--- GZIP Tests ---\n");
    failures += test_gzip_small_literal();
    failures += test_gzip_compressible();
    failures += test_gzip_zeros();

    printf("\n--- ZSTD Tests ---\n");
    failures += test_zstd_small_literal();
    failures += test_zstd_compressible();
    failures += test_zstd_zeros();

    printf("\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
