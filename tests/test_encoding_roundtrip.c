/**
 * @file test_encoding_roundtrip.c
 * @brief carquet-reads-carquet roundtrip for every opt-in (Phase 3) encoding.
 *
 * Regression guard for the write/read asymmetry where the writer could emit
 * DELTA_* / BYTE_STREAM_SPLIT (incl. INT32/INT64/FLBA) but the reader could
 * not decode them. Every encoding here is written with carquet AND read back
 * through carquet's own reader, asserting exact value equality — including a
 * nullable column and a multi-column single-batch read that exercises the
 * DELTA_BYTE_ARRAY reconstruction-scratch lifetime across pages.
 */

#include <carquet/carquet.h>
#include "test_helpers.h"

#define N 4000

static carquet_reader_t* write_then_open(
    const char* path,
    carquet_physical_type_t phys,
    int32_t type_length,
    carquet_field_repetition_t rep,
    carquet_encoding_t enc,
    const void* values,
    int64_t nvalues,
    const int16_t* def_levels) {

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) return NULL;
    if (carquet_schema_add_column(s, "c", phys, NULL, rep, type_length, 0) != CARQUET_OK) {
        carquet_schema_free(s);
        return NULL;
    }
    carquet_writer_options_t wo;
    carquet_writer_options_init(&wo);
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); return NULL; }
    if (carquet_writer_set_column_encoding(w, 0, enc) != CARQUET_OK) {
        carquet_writer_close(w); carquet_schema_free(s); return NULL;
    }
    if (carquet_writer_write_batch(w, 0, values, nvalues, def_levels, NULL) != CARQUET_OK) {
        carquet_writer_close(w); carquet_schema_free(s); return NULL;
    }
    if (carquet_writer_close(w) != CARQUET_OK) { carquet_schema_free(s); return NULL; }
    carquet_schema_free(s);
    return carquet_reader_open(path, NULL, &err);
}

static int test_delta_binary_packed_i32(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "dbp_i32");
    int32_t in[N], out[N];
    for (int i = 0; i < N; i++) in[i] = (i * 7) - 1000 + (i % 13);
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* r = write_then_open(path, CARQUET_PHYSICAL_INT32, 0,
        CARQUET_REPETITION_REQUIRED, CARQUET_ENCODING_DELTA_BINARY_PACKED, in, N, NULL);
    if (!r) TEST_FAIL("delta_binary_packed_i32", "write/open failed");
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    int64_t n = c ? carquet_column_read_batch(c, out, N, NULL, NULL) : -1;
    if (n != N || memcmp(in, out, sizeof(in)) != 0)
        { carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
          TEST_FAIL("delta_binary_packed_i32", "value mismatch"); }
    carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
    TEST_PASS("delta_binary_packed_i32");
    return 0;
}

static int test_delta_binary_packed_i64(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "dbp_i64");
    int64_t in[N], out[N];
    for (int i = 0; i < N; i++) in[i] = ((int64_t)i * 2654435761ULL) ^ (i % 7);
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* r = write_then_open(path, CARQUET_PHYSICAL_INT64, 0,
        CARQUET_REPETITION_REQUIRED, CARQUET_ENCODING_DELTA_BINARY_PACKED, in, N, NULL);
    if (!r) TEST_FAIL("delta_binary_packed_i64", "write/open failed");
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    int64_t n = c ? carquet_column_read_batch(c, out, N, NULL, NULL) : -1;
    if (n != N || memcmp(in, out, sizeof(in)) != 0)
        { carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
          TEST_FAIL("delta_binary_packed_i64", "value mismatch"); }
    carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
    TEST_PASS("delta_binary_packed_i64");
    return 0;
}

static int check_byte_arrays(const char* name, const char* path,
                             carquet_reader_t* r,
                             const carquet_byte_array_t* expect, int64_t cnt) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    static carquet_byte_array_t out[N];
    int64_t n = c ? carquet_column_read_batch(c, out, N, NULL, NULL) : -1;
    int ok = (n == cnt);
    for (int64_t i = 0; ok && i < cnt; i++)
        ok = (out[i].length == expect[i].length) &&
             (memcmp(out[i].data, expect[i].data, (size_t)expect[i].length) == 0);
    carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL(name, "byte array mismatch");
    TEST_PASS(name);
    return 0;
}

static int test_delta_length_byte_array(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "dlba");
    static char buf[N][24];
    static carquet_byte_array_t in[N];
    for (int i = 0; i < N; i++) {
        int len = snprintf(buf[i], sizeof(buf[i]), "row-%d-val", i);
        in[i].data = (uint8_t*)buf[i]; in[i].length = len;
    }
    carquet_reader_t* r = write_then_open(path, CARQUET_PHYSICAL_BYTE_ARRAY, 0,
        CARQUET_REPETITION_REQUIRED, CARQUET_ENCODING_DELTA_LENGTH_BYTE_ARRAY, in, N, NULL);
    if (!r) TEST_FAIL("delta_length_byte_array", "write/open failed");
    return check_byte_arrays("delta_length_byte_array", path, r, in, N);
}

static int test_delta_byte_array(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "dba");
    static char buf[N][32];
    static carquet_byte_array_t in[N];
    /* Strong shared prefixes to exercise prefix reconstruction. */
    for (int i = 0; i < N; i++) {
        int len = snprintf(buf[i], sizeof(buf[i]), "common/prefix/path/item_%05d", i);
        in[i].data = (uint8_t*)buf[i]; in[i].length = len;
    }
    carquet_reader_t* r = write_then_open(path, CARQUET_PHYSICAL_BYTE_ARRAY, 0,
        CARQUET_REPETITION_REQUIRED, CARQUET_ENCODING_DELTA_BYTE_ARRAY, in, N, NULL);
    if (!r) TEST_FAIL("delta_byte_array", "write/open failed");
    return check_byte_arrays("delta_byte_array", path, r, in, N);
}

static int test_delta_byte_array_flba(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "dba_flba");
    enum { L = 8 };
    static uint8_t in[N * L], out[N * L];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < L; j++) in[i * L + j] = (uint8_t)((i + j * 31) & 0xFF);
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* r = write_then_open(path, CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY, L,
        CARQUET_REPETITION_REQUIRED, CARQUET_ENCODING_DELTA_BYTE_ARRAY, in, N, NULL);
    if (!r) TEST_FAIL("delta_byte_array_flba", "write/open failed");
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    int64_t n = c ? carquet_column_read_batch(c, out, N, NULL, NULL) : -1;
    int ok = (n == N) && (memcmp(in, out, sizeof(in)) == 0);
    carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("delta_byte_array_flba", "value mismatch");
    TEST_PASS("delta_byte_array_flba");
    return 0;
}

static int test_bss(const char* name, const char* base,
                     carquet_physical_type_t phys, int32_t tl, size_t esz) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), base);
    static uint8_t in[N * 8], out[N * 8];
    size_t bytes = (size_t)N * esz;
    for (size_t i = 0; i < bytes; i++) in[i] = (uint8_t)((i * 131 + 7) & 0xFF);
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* r = write_then_open(path, phys, tl,
        CARQUET_REPETITION_REQUIRED, CARQUET_ENCODING_BYTE_STREAM_SPLIT, in, N, NULL);
    if (!r) TEST_FAIL(name, "write/open failed");
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    int64_t n = c ? carquet_column_read_batch(c, out, N, NULL, NULL) : -1;
    int ok = (n == N) && (memcmp(in, out, bytes) == 0);
    carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL(name, "value mismatch");
    TEST_PASS(name);
    return 0;
}

/* BYTE_STREAM_SPLIT transposes a whole page into byte planes, so it is only
 * correct when applied to the finished page. Writing the same page in several
 * carquet_writer_write_batch() calls must therefore produce byte-identical
 * output to writing it in one. The chunk sizes below are deliberately not
 * multiples of any value width, so a per-call split leaves independently
 * transposed regions that the reader de-splits as a single stride. */
static int test_bss_chunked(const char* name, const char* base,
                            carquet_physical_type_t phys, int32_t tl, size_t esz) {
    static const int64_t chunks[] = { 7, 1, 1992, 2000 };
    char path[512]; carquet_test_temp_path(path, sizeof(path), base);
    static uint8_t in[N * 8], out[N * 8];
    size_t bytes = (size_t)N * esz;
    for (size_t i = 0; i < bytes; i++) in[i] = (uint8_t)((i * 131 + 7) & 0xFF);

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL(name, "schema create failed");
    if (carquet_schema_add_column(s, "c", phys, NULL,
                                  CARQUET_REPETITION_REQUIRED, tl, 0) != CARQUET_OK) {
        carquet_schema_free(s); TEST_FAIL(name, "add column failed");
    }
    carquet_writer_options_t wo;
    carquet_writer_options_init(&wo);
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL(name, "writer create failed"); }
    if (carquet_writer_set_column_encoding(
            w, 0, CARQUET_ENCODING_BYTE_STREAM_SPLIT) != CARQUET_OK) {
        carquet_writer_close(w); carquet_schema_free(s);
        TEST_FAIL(name, "set encoding failed");
    }
    int64_t done = 0;
    for (size_t k = 0; k < sizeof(chunks) / sizeof(chunks[0]); k++) {
        if (carquet_writer_write_batch(w, 0, in + (size_t)done * esz,
                                       chunks[k], NULL, NULL) != CARQUET_OK) {
            carquet_writer_close(w); carquet_schema_free(s);
            TEST_FAIL(name, "write batch failed");
        }
        done += chunks[k];
    }
    if (carquet_writer_close(w) != CARQUET_OK) {
        carquet_schema_free(s); TEST_FAIL(name, "writer close failed");
    }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) TEST_FAIL(name, "open failed");
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    int64_t n = c ? carquet_column_read_batch(c, out, N, NULL, NULL) : -1;
    int ok = (done == N) && (n == N) && (memcmp(in, out, bytes) == 0);
    carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL(name, "value mismatch");
    TEST_PASS(name);
    return 0;
}

static int test_delta_byte_array_nullable(void) {
    /* Nullable DELTA_BYTE_ARRAY: exercises the reconstruction scratch together
     * with definition levels (packed non-null value stream). */
    char path[512]; carquet_test_temp_path(path, sizeof(path), "dba_null");
    static char buf[N][32];
    static carquet_byte_array_t in[N];
    static int16_t def[N];
    int64_t nn = 0;
    for (int i = 0; i < N; i++) {
        if (i % 3 == 0) { def[i] = 0; continue; }
        def[i] = 1;
        int len = snprintf(buf[nn], sizeof(buf[nn]), "shared-prefix-%04d", i);
        in[nn].data = (uint8_t*)buf[nn]; in[nn].length = len; nn++;
    }
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* r = write_then_open(path, CARQUET_PHYSICAL_BYTE_ARRAY, 0,
        CARQUET_REPETITION_OPTIONAL, CARQUET_ENCODING_DELTA_BYTE_ARRAY, in, N, def);
    if (!r) TEST_FAIL("delta_byte_array_nullable", "write/open failed");
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    static carquet_byte_array_t out[N];
    static int16_t outdef[N];
    int64_t got = c ? carquet_column_read_batch(c, out, N, outdef, NULL) : -1;
    int ok = (got == N);
    int64_t vi = 0;
    for (int64_t i = 0; ok && i < N; i++) {
        if (outdef[i] == 0) continue;
        ok = (out[vi].length == in[vi].length) &&
             (memcmp(out[vi].data, in[vi].data, (size_t)in[vi].length) == 0);
        vi++;
    }
    ok = ok && (vi == nn);
    carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("delta_byte_array_nullable", "nullable mismatch");
    TEST_PASS("delta_byte_array_nullable");
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_delta_binary_packed_i32();
    failures += test_delta_binary_packed_i64();
    failures += test_delta_length_byte_array();
    failures += test_delta_byte_array();
    failures += test_delta_byte_array_flba();
    failures += test_delta_byte_array_nullable();
    failures += test_bss("bss_float",  "bss_f32", CARQUET_PHYSICAL_FLOAT, 0, 4);
    failures += test_bss("bss_double", "bss_f64", CARQUET_PHYSICAL_DOUBLE, 0, 8);
    failures += test_bss("bss_int32",  "bss_i32", CARQUET_PHYSICAL_INT32, 0, 4);
    failures += test_bss("bss_int64",  "bss_i64", CARQUET_PHYSICAL_INT64, 0, 8);
    failures += test_bss("bss_flba",   "bss_flba", CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY, 6, 6);
    failures += test_bss_chunked("bss_float_chunked",  "bssc_f32", CARQUET_PHYSICAL_FLOAT, 0, 4);
    failures += test_bss_chunked("bss_double_chunked", "bssc_f64", CARQUET_PHYSICAL_DOUBLE, 0, 8);
    failures += test_bss_chunked("bss_int32_chunked",  "bssc_i32", CARQUET_PHYSICAL_INT32, 0, 4);
    failures += test_bss_chunked("bss_int64_chunked",  "bssc_i64", CARQUET_PHYSICAL_INT64, 0, 8);
    failures += test_bss_chunked("bss_flba_chunked",   "bssc_flba", CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY, 6, 6);
    if (failures) { printf("\n%d test(s) FAILED\n", failures); return 1; }
    printf("\nAll encoding roundtrip tests passed\n");
    return 0;
}
