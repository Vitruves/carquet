/**
 * @file test_core.c
 * @brief Tests for core utilities
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "core/arena.h"
#include "core/buffer.h"
#include "core/endian.h"
#include "core/bitpack.h"

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* ============================================================================
 * Arena Tests
 * ============================================================================
 */

static int test_arena_basic(void) {
    carquet_arena_t arena;
    if (carquet_arena_init(&arena) != CARQUET_OK) {
        TEST_FAIL("arena_basic", "failed to initialize arena");
    }

    /* Test basic allocation */
    void* p1 = carquet_arena_alloc(&arena, 100);
    if (p1 == NULL) {
        carquet_arena_destroy(&arena);
        TEST_FAIL("arena_basic", "first allocation returned NULL");
    }

    void* p2 = carquet_arena_alloc(&arena, 200);
    assert(p2 != NULL);
    assert(p2 != p1);

    /* Test calloc */
    int* arr = carquet_arena_calloc(&arena, 10, sizeof(int));
    assert(arr != NULL);
    for (int i = 0; i < 10; i++) {
        assert(arr[i] == 0);
    }

    /* Test strdup */
    char* s = carquet_arena_strdup(&arena, "Hello, World!");
    assert(s != NULL);
    assert(strcmp(s, "Hello, World!") == 0);

    /* Test reset */
    size_t allocated_before = carquet_arena_allocated(&arena);
    assert(allocated_before > 0);

    carquet_arena_reset(&arena);
    assert(carquet_arena_allocated(&arena) == 0);

    /* Can still allocate after reset */
    void* p3 = carquet_arena_alloc(&arena, 50);
    assert(p3 != NULL);

    carquet_arena_destroy(&arena);
    TEST_PASS("arena_basic");
    return 0;
}

static int test_arena_large_allocation(void) {
    carquet_arena_t arena;
    if (carquet_arena_init(&arena) != CARQUET_OK) {
        TEST_FAIL("arena_large_allocation", "failed to initialize arena");
    }

    /* Allocate something larger than default block size */
    void* p = carquet_arena_alloc(&arena, 256 * 1024);
    if (p == NULL) {
        carquet_arena_destroy(&arena);
        TEST_FAIL("arena_large_allocation", "large allocation returned NULL");
    }

    carquet_arena_destroy(&arena);
    TEST_PASS("arena_large_allocation");
    return 0;
}

static int test_arena_save_restore(void) {
    carquet_arena_t arena;
    if (carquet_arena_init(&arena) != CARQUET_OK) {
        TEST_FAIL("arena_save_restore", "failed to initialize arena");
    }

    void* p1 = carquet_arena_alloc(&arena, 100);
    if (p1 == NULL) {
        carquet_arena_destroy(&arena);
        TEST_FAIL("arena_save_restore", "first allocation returned NULL");
    }

    carquet_arena_mark_t mark = carquet_arena_save(&arena);
    size_t allocated_at_mark = carquet_arena_allocated(&arena);

    void* p2 = carquet_arena_alloc(&arena, 200);
    assert(p2 != NULL);

    carquet_arena_restore(&arena, mark);
    assert(carquet_arena_allocated(&arena) == allocated_at_mark);

    carquet_arena_destroy(&arena);
    TEST_PASS("arena_save_restore");
    return 0;
}

/* ============================================================================
 * Buffer Tests
 * ============================================================================
 */

static int test_buffer_basic(void) {
    carquet_buffer_t buf;
    if (carquet_buffer_init(&buf) != CARQUET_OK) {
        TEST_FAIL("buffer_basic", "failed to initialize buffer");
    }

    /* Test append */
    uint8_t data[] = {1, 2, 3, 4, 5};
    if (carquet_buffer_append(&buf, data, 5) != CARQUET_OK) {
        carquet_buffer_destroy(&buf);
        TEST_FAIL("buffer_basic", "failed to append data");
    }
    assert(carquet_buffer_size(&buf) == 5);

    /* Test append byte */
    assert(carquet_buffer_append_byte(&buf, 6) == CARQUET_OK);
    assert(carquet_buffer_size(&buf) == 6);

    /* Verify contents */
    const uint8_t* ptr = carquet_buffer_data_const(&buf);
    assert(ptr[0] == 1 && ptr[5] == 6);

    carquet_buffer_destroy(&buf);
    TEST_PASS("buffer_basic");
    return 0;
}

static int test_buffer_integers(void) {
    carquet_buffer_t buf;
    if (carquet_buffer_init(&buf) != CARQUET_OK) {
        TEST_FAIL("buffer_integers", "failed to initialize buffer");
    }

    if (carquet_buffer_append_u16_le(&buf, 0x1234) != CARQUET_OK) {
        carquet_buffer_destroy(&buf);
        TEST_FAIL("buffer_integers", "failed to append u16");
    }
    assert(carquet_buffer_append_u32_le(&buf, 0x12345678) == CARQUET_OK);
    assert(carquet_buffer_append_u64_le(&buf, 0x123456789ABCDEF0ULL) == CARQUET_OK);

    const uint8_t* ptr = carquet_buffer_data_const(&buf);

    /* Verify u16 */
    assert(ptr[0] == 0x34 && ptr[1] == 0x12);

    /* Verify u32 */
    assert(carquet_read_u32_le(ptr + 2) == 0x12345678);

    /* Verify u64 */
    assert(carquet_read_u64_le(ptr + 6) == 0x123456789ABCDEF0ULL);

    carquet_buffer_destroy(&buf);
    TEST_PASS("buffer_integers");
    return 0;
}

static int test_buffer_reader(void) {
    uint8_t data[] = {0x34, 0x12, 0x78, 0x56, 0x34, 0x12};
    carquet_buffer_reader_t reader;
    carquet_buffer_reader_init_data(&reader, data, sizeof(data));

    uint16_t u16;
    assert(carquet_buffer_reader_read_u16_le(&reader, &u16) == CARQUET_OK);
    assert(u16 == 0x1234);

    uint32_t u32;
    assert(carquet_buffer_reader_read_u32_le(&reader, &u32) == CARQUET_OK);
    assert(u32 == 0x12345678);

    assert(carquet_buffer_reader_remaining(&reader) == 0);

    TEST_PASS("buffer_reader");
    return 0;
}

/* ============================================================================
 * Endian Tests
 * ============================================================================
 */

static int test_endian_read_write(void) {
    uint8_t buf[8];

    /* Test 16-bit */
    carquet_write_u16_le(buf, 0x1234);
    assert(carquet_read_u16_le(buf) == 0x1234);

    /* Test 32-bit */
    carquet_write_u32_le(buf, 0x12345678);
    assert(carquet_read_u32_le(buf) == 0x12345678);

    /* Test 64-bit */
    carquet_write_u64_le(buf, 0x123456789ABCDEF0ULL);
    assert(carquet_read_u64_le(buf) == 0x123456789ABCDEF0ULL);

    TEST_PASS("endian_read_write");
    return 0;
}

static int test_varint(void) {
    uint8_t buf[10];
    uint32_t val32;
    uint64_t val64;

    /* Test small value */
    int len = carquet_encode_varint32(buf, 127);
    assert(len == 1);
    int consumed = carquet_decode_varint32(buf, 10, &val32);
    assert(consumed == 1 && val32 == 127);

    /* Test larger value */
    len = carquet_encode_varint32(buf, 300);
    assert(len == 2);
    consumed = carquet_decode_varint32(buf, 10, &val32);
    assert(consumed == 2 && val32 == 300);

    /* Test 64-bit */
    len = carquet_encode_varint64(buf, 0x123456789ULL);
    consumed = carquet_decode_varint64(buf, 10, &val64);
    assert(val64 == 0x123456789ULL);

    TEST_PASS("varint");
    return 0;
}

static int test_zigzag(void) {
    /* Positive numbers */
    assert(carquet_zigzag_encode32(0) == 0);
    assert(carquet_zigzag_encode32(1) == 2);
    assert(carquet_zigzag_encode32(2) == 4);

    /* Negative numbers */
    assert(carquet_zigzag_encode32(-1) == 1);
    assert(carquet_zigzag_encode32(-2) == 3);

    /* Roundtrip */
    for (int32_t i = -1000; i <= 1000; i++) {
        uint32_t encoded = carquet_zigzag_encode32(i);
        int32_t decoded = carquet_zigzag_decode32(encoded);
        assert(decoded == i);
    }

    TEST_PASS("zigzag");
    return 0;
}

/* ============================================================================
 * Bitpack Tests
 * ============================================================================
 */

static int test_bitpack_1bit(void) {
    uint8_t input = 0xAA;  /* 0b10101010 */
    uint32_t output[8];

    carquet_bitunpack8_1bit(&input, output);

    assert(output[0] == 0);
    assert(output[1] == 1);
    assert(output[2] == 0);
    assert(output[3] == 1);
    assert(output[4] == 0);
    assert(output[5] == 1);
    assert(output[6] == 0);
    assert(output[7] == 1);

    TEST_PASS("bitpack_1bit");
    return 0;
}

static int test_bitpack_4bit(void) {
    uint8_t input[4] = {0x21, 0x43, 0x65, 0x87};
    uint32_t output[8];

    carquet_bitunpack8_4bit(input, output);

    assert(output[0] == 1);
    assert(output[1] == 2);
    assert(output[2] == 3);
    assert(output[3] == 4);
    assert(output[4] == 5);
    assert(output[5] == 6);
    assert(output[6] == 7);
    assert(output[7] == 8);

    TEST_PASS("bitpack_4bit");
    return 0;
}

static int test_bitpack_roundtrip(void) {
    uint32_t original[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t packed[32];
    uint32_t unpacked[8];

    for (int bit_width = 1; bit_width <= 8; bit_width++) {
        /* Clear buffers */
        memset(packed, 0, sizeof(packed));
        memset(unpacked, 0, sizeof(unpacked));

        /* Pack */
        carquet_bitpack8_32(original, bit_width, packed);

        /* Unpack */
        carquet_bitunpack8_32(packed, bit_width, unpacked);

        /* Verify */
        uint32_t mask = (1U << bit_width) - 1;
        for (int i = 0; i < 8; i++) {
            assert((original[i] & mask) == unpacked[i]);
        }
    }

    TEST_PASS("bitpack_roundtrip");
    return 0;
}

static int test_bit_reader(void) {
    uint8_t data[] = {0xD2, 0xB4};  /* 0b11010010, 0b10110100, LSB first */
    carquet_bit_reader_t reader;
    carquet_bit_reader_init(&reader, data, 2);

    /* Read individual bits (LSB first) */
    assert(carquet_bit_reader_read_bit(&reader) == 0);
    assert(carquet_bit_reader_read_bit(&reader) == 1);
    assert(carquet_bit_reader_read_bit(&reader) == 0);
    assert(carquet_bit_reader_read_bit(&reader) == 0);

    /* Read 4 bits */
    uint32_t nibble = carquet_bit_reader_read_bits(&reader, 4);
    assert(nibble == 0xD);  /* bits 4-7 of first byte: 0b1101 = 13 */

    TEST_PASS("bit_reader");
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================
 */

int main(void) {
    int failures = 0;

    printf("=== Core Utility Tests ===\n\n");

    /* Arena tests */
    failures += test_arena_basic();
    failures += test_arena_large_allocation();
    failures += test_arena_save_restore();

    /* Buffer tests */
    failures += test_buffer_basic();
    failures += test_buffer_integers();
    failures += test_buffer_reader();

    /* Endian tests */
    failures += test_endian_read_write();
    failures += test_varint();
    failures += test_zigzag();

    /* Bitpack tests */
    failures += test_bitpack_1bit();
    failures += test_bitpack_4bit();
    failures += test_bitpack_roundtrip();
    failures += test_bit_reader();

    printf("\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
