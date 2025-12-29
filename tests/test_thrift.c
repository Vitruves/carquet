/**
 * @file test_thrift.c
 * @brief Tests for Thrift encoding/decoding
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "thrift/thrift_decode.h"
#include "thrift/thrift_encode.h"
#include "core/buffer.h"

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); return 1; } while(0)

static int test_thrift_varint_roundtrip(void) {
    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, &buf);

    /* Write various integers */
    thrift_write_i32(&enc, 0);
    thrift_write_i32(&enc, 1);
    thrift_write_i32(&enc, -1);
    thrift_write_i32(&enc, 127);
    thrift_write_i32(&enc, 128);
    thrift_write_i32(&enc, 12345);
    thrift_write_i32(&enc, -12345);
    thrift_write_i32(&enc, 0x7FFFFFFF);
    thrift_write_i32(&enc, (int32_t)0x80000000);

    /* Read them back */
    thrift_decoder_t dec;
    thrift_decoder_init(&dec, carquet_buffer_data_const(&buf), carquet_buffer_size(&buf));

    assert(thrift_read_i32(&dec) == 0);
    assert(thrift_read_i32(&dec) == 1);
    assert(thrift_read_i32(&dec) == -1);
    assert(thrift_read_i32(&dec) == 127);
    assert(thrift_read_i32(&dec) == 128);
    assert(thrift_read_i32(&dec) == 12345);
    assert(thrift_read_i32(&dec) == -12345);
    assert(thrift_read_i32(&dec) == 0x7FFFFFFF);
    assert(thrift_read_i32(&dec) == (int32_t)0x80000000);

    assert(!thrift_decoder_has_error(&dec));

    carquet_buffer_destroy(&buf);
    TEST_PASS("thrift_varint_roundtrip");
    return 0;
}

static int test_thrift_string_roundtrip(void) {
    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, &buf);

    thrift_write_string(&enc, "Hello, World!");
    thrift_write_string(&enc, "");
    thrift_write_string(&enc, "A longer string with more characters");

    thrift_decoder_t dec;
    thrift_decoder_init(&dec, carquet_buffer_data_const(&buf), carquet_buffer_size(&buf));

    int32_t len;
    const uint8_t* data;

    data = thrift_read_binary(&dec, &len);
    assert(len == 13);
    assert(memcmp(data, "Hello, World!", 13) == 0);

    data = thrift_read_binary(&dec, &len);
    assert(len == 0);

    data = thrift_read_binary(&dec, &len);
    assert(len == 36);

    assert(!thrift_decoder_has_error(&dec));

    carquet_buffer_destroy(&buf);
    TEST_PASS("thrift_string_roundtrip");
    return 0;
}

static int test_thrift_struct(void) {
    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, &buf);

    /* Write a simple struct */
    thrift_write_struct_begin(&enc);
    THRIFT_WRITE_FIELD_I32(&enc, 1, 42);
    THRIFT_WRITE_FIELD_STRING(&enc, 2, "test");
    THRIFT_WRITE_FIELD_BOOL(&enc, 3, true);
    thrift_write_struct_end(&enc);

    /* Read it back */
    thrift_decoder_t dec;
    thrift_decoder_init(&dec, carquet_buffer_data_const(&buf), carquet_buffer_size(&buf));

    thrift_read_struct_begin(&dec);

    thrift_type_t type;
    int16_t field_id;

    assert(thrift_read_field_begin(&dec, &type, &field_id));
    assert(field_id == 1);
    assert(type == THRIFT_TYPE_I32);
    assert(thrift_read_i32(&dec) == 42);

    assert(thrift_read_field_begin(&dec, &type, &field_id));
    assert(field_id == 2);
    assert(type == THRIFT_TYPE_BINARY);
    int32_t len;
    const uint8_t* data = thrift_read_binary(&dec, &len);
    assert(len == 4 && memcmp(data, "test", 4) == 0);

    assert(thrift_read_field_begin(&dec, &type, &field_id));
    assert(field_id == 3);
    assert(type == THRIFT_TYPE_TRUE);
    assert(thrift_read_bool(&dec) == true);

    assert(!thrift_read_field_begin(&dec, &type, &field_id));  /* STOP */

    thrift_read_struct_end(&dec);

    assert(!thrift_decoder_has_error(&dec));

    carquet_buffer_destroy(&buf);
    TEST_PASS("thrift_struct");
    return 0;
}

static int test_thrift_list(void) {
    carquet_buffer_t buf;
    carquet_buffer_init(&buf);

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, &buf);

    /* Write a list of integers */
    thrift_write_list_begin(&enc, THRIFT_TYPE_I32, 5);
    thrift_write_i32(&enc, 1);
    thrift_write_i32(&enc, 2);
    thrift_write_i32(&enc, 3);
    thrift_write_i32(&enc, 4);
    thrift_write_i32(&enc, 5);

    /* Read it back */
    thrift_decoder_t dec;
    thrift_decoder_init(&dec, carquet_buffer_data_const(&buf), carquet_buffer_size(&buf));

    thrift_type_t elem_type;
    int32_t count;
    thrift_read_list_begin(&dec, &elem_type, &count);

    assert(elem_type == THRIFT_TYPE_I32);
    assert(count == 5);

    for (int i = 1; i <= 5; i++) {
        assert(thrift_read_i32(&dec) == i);
    }

    assert(!thrift_decoder_has_error(&dec));

    carquet_buffer_destroy(&buf);
    TEST_PASS("thrift_list");
    return 0;
}

int main(void) {
    int failures = 0;

    printf("=== Thrift Tests ===\n\n");

    failures += test_thrift_varint_roundtrip();
    failures += test_thrift_string_roundtrip();
    failures += test_thrift_struct();
    failures += test_thrift_list();

    printf("\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
