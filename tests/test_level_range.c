/**
 * @file test_level_range.c
 * @brief Data page decoding rejects levels above the column maximum
 */

#include <carquet/carquet.h>
#include "reader/reader_internal.h"
#include "thrift/parquet_types.h"
#include "test_helpers.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern carquet_status_t carquet_read_data_page_v1(
    carquet_column_reader_t* reader, const uint8_t* page_data, size_t page_size,
    const parquet_data_page_header_t* header, void* values, int64_t max_values,
    int16_t* def_levels, int16_t* rep_levels, int64_t* values_read, carquet_error_t* error);
extern carquet_status_t carquet_read_data_page_v2(
    carquet_column_reader_t* reader, const uint8_t* page_data, size_t page_size,
    const parquet_data_page_header_v2_t* header, void* values, int64_t max_values,
    int16_t* def_levels, int16_t* rep_levels, int64_t* values_read, carquet_error_t* error);

#define N_LEVELS 8

/* One bit-packed group of eight 2-bit levels (hybrid header 0x03), LSB first. */
static size_t pack_levels8(const int16_t* lv, uint8_t* out) {
    out[0] = 0x03;
    out[1] = (uint8_t)(lv[0] | (lv[1] << 2) | (lv[2] << 4) | (lv[3] << 6));
    out[2] = (uint8_t)(lv[4] | (lv[5] << 2) | (lv[6] << 4) | (lv[7] << 6));
    return 3;
}

static size_t put_plain_int32(uint8_t* out, int32_t count) {
    for (int32_t i = 0; i < count; i++) {
        int32_t v = 100 + i;
        memcpy(out + (size_t)i * 4, &v, 4);
    }
    return (size_t)count * 4;
}

static void init_int32_reader(carquet_column_reader_t* cr, int16_t max_def, int16_t max_rep) {
    memset(cr, 0, sizeof(*cr));
    cr->type = CARQUET_PHYSICAL_INT32;
    cr->max_def_level = max_def;
    cr->max_rep_level = max_rep;
}

/* V1 page: [u32 length][def levels][PLAIN values]. */
static carquet_status_t decode_v1_def(const int16_t* def, int16_t* out_def, int64_t* got) {
    uint8_t page[64];
    size_t pos = 4;
    pos += pack_levels8(def, page + pos);
    uint32_t len = (uint32_t)(pos - 4);
    memcpy(page, &len, 4);
    int32_t present = 0;
    for (int i = 0; i < N_LEVELS; i++) present += def[i] == 2;
    pos += put_plain_int32(page + pos, present);

    carquet_column_reader_t cr;
    init_int32_reader(&cr, 2, 0);
    parquet_data_page_header_t h;
    memset(&h, 0, sizeof(h));
    h.num_values = N_LEVELS;
    h.encoding = CARQUET_ENCODING_PLAIN;
    h.definition_level_encoding = CARQUET_ENCODING_RLE;
    h.repetition_level_encoding = CARQUET_ENCODING_RLE;

    int32_t values[N_LEVELS];
    carquet_error_t err = CARQUET_ERROR_INIT;
    return carquet_read_data_page_v1(&cr, page, pos, &h, values, N_LEVELS,
                                     out_def, NULL, got, &err);
}

static int test_level_range_v1(void) {
    int16_t out[N_LEVELS];
    int64_t got = 0;

    /* Control: every level within [0, 2]. */
    const int16_t good[N_LEVELS] = {2, 1, 2, 0, 2, 2, 1, 2};
    if (decode_v1_def(good, out, &got) != CARQUET_OK || got != N_LEVELS)
        TEST_FAIL("level_range_v1", "valid levels rejected");
    if (memcmp(out, good, sizeof(good)) != 0)
        TEST_FAIL("level_range_v1", "valid levels changed");

    const int16_t bad[N_LEVELS] = {2, 3, 2, 2, 2, 2, 2, 2};
    got = 0;
    if (decode_v1_def(bad, out, &got) == CARQUET_OK)
        TEST_FAIL("level_range_v1", "definition level above max_def accepted");

    TEST_PASS("level_range_v1");
    return 0;
}

static int test_level_range_v2(void) {
    for (int pass = 0; pass < 2; pass++) {
        const int16_t good[N_LEVELS] = {2, 2, 0, 1, 2, 2, 2, 1};
        const int16_t bad[N_LEVELS]  = {2, 2, 0, 1, 2, 3, 2, 1};
        const int16_t* def = pass == 0 ? good : bad;

        uint8_t page[64];
        size_t pos = pack_levels8(def, page);
        size_t def_bytes = pos;
        int32_t present = 0;
        for (int i = 0; i < N_LEVELS; i++) present += def[i] == 2;
        pos += put_plain_int32(page + pos, present);

        carquet_column_reader_t cr;
        init_int32_reader(&cr, 2, 0);
        parquet_data_page_header_v2_t h;
        memset(&h, 0, sizeof(h));
        h.num_values = N_LEVELS;
        h.num_rows = N_LEVELS;
        h.encoding = CARQUET_ENCODING_PLAIN;
        h.definition_levels_byte_length = (int32_t)def_bytes;
        h.repetition_levels_byte_length = 0;

        int32_t values[N_LEVELS];
        int16_t out[N_LEVELS];
        int64_t got = 0;
        carquet_error_t err = CARQUET_ERROR_INIT;
        carquet_status_t st = carquet_read_data_page_v2(
            &cr, page, pos, &h, values, N_LEVELS, out, NULL, &got, &err);
        if (pass == 0 && st != CARQUET_OK)
            TEST_FAIL("level_range_v2", "valid levels rejected");
        if (pass == 1 && st == CARQUET_OK)
            TEST_FAIL("level_range_v2", "definition level above max_def accepted");
    }
    TEST_PASS("level_range_v2");
    return 0;
}

static int test_level_range_rep(void) {
    for (int pass = 0; pass < 2; pass++) {
        const int16_t good[N_LEVELS] = {0, 2, 1, 0, 0, 2, 2, 0};
        const int16_t bad[N_LEVELS]  = {0, 3, 1, 0, 0, 2, 2, 0};
        const int16_t def[N_LEVELS]  = {2, 2, 2, 2, 2, 2, 2, 2};

        uint8_t page[96];
        size_t pos = 0;
        uint32_t len = 3;
        memcpy(page + pos, &len, 4); pos += 4;
        pos += pack_levels8(pass == 0 ? good : bad, page + pos);
        memcpy(page + pos, &len, 4); pos += 4;
        pos += pack_levels8(def, page + pos);
        pos += put_plain_int32(page + pos, N_LEVELS);

        carquet_column_reader_t cr;
        init_int32_reader(&cr, 2, 2);
        parquet_data_page_header_t h;
        memset(&h, 0, sizeof(h));
        h.num_values = N_LEVELS;
        h.encoding = CARQUET_ENCODING_PLAIN;
        h.definition_level_encoding = CARQUET_ENCODING_RLE;
        h.repetition_level_encoding = CARQUET_ENCODING_RLE;

        int32_t values[N_LEVELS];
        int16_t out_def[N_LEVELS], out_rep[N_LEVELS];
        int64_t got = 0;
        carquet_error_t err = CARQUET_ERROR_INIT;
        carquet_status_t st = carquet_read_data_page_v1(
            &cr, page, pos, &h, values, N_LEVELS, out_def, out_rep, &got, &err);
        if (pass == 0 && st != CARQUET_OK)
            TEST_FAIL("level_range_rep", "valid levels rejected");
        if (pass == 1 && st == CARQUET_OK)
            TEST_FAIL("level_range_rep", "repetition level above max_rep accepted");
    }
    TEST_PASS("level_range_rep");
    return 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int failures = 0;
    failures += test_level_range_v1();
    failures += test_level_range_v2();
    failures += test_level_range_rep();
    if (failures) {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
    printf("All tests passed\n");
    return 0;
}
