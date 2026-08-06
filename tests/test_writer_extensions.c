/**
 * @file test_writer_extensions.c
 * @brief Roundtrip + structural tests for the v0.4.5 writer extensions:
 *        INT96 writing, opt-in Data Page V2, "ARROW:schema" metadata,
 *        FLOAT16 statistics ordering, deprecated BIT_PACKED level decoding,
 *        and GEOMETRY/GEOGRAPHY GeospatialStatistics.
 *
 * Everything written here is also read back through carquet's own reader and
 * asserted exact. The Data Page V2 test additionally parses the on-disk page
 * header to assert PageType==DATA_PAGE_V2 (not just that the bytes happen to
 * decode), and the ARROW:schema test checks the encapsulated-message framing.
 */

#include <carquet/carquet.h>
#include "thrift/parquet_types.h"
#include "core/bitpack.h"
#include "core/compat.h"
#include "test_helpers.h"
#include <stdio.h>
#include <string.h>

#define N 5000

/* ---- INT96 ---- */
static int test_int96_roundtrip(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_int96");
    carquet_error_t err = CARQUET_ERROR_INIT;

    static carquet_int96_t in[N], out[N];
    for (int i = 0; i < N; i++) {
        in[i].value[0] = (uint32_t)(i * 2654435761u);
        in[i].value[1] = (uint32_t)(i ^ 0xABCD1234u);
        in[i].value[2] = (uint32_t)(2451545 + i);   /* Julian-day-ish */
    }

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("int96_roundtrip", "schema create");
    if (carquet_schema_add_column(s, "ts96", CARQUET_PHYSICAL_INT96, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("int96_roundtrip", "add col"); }

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.compression = CARQUET_COMPRESSION_ZSTD;   /* exercise the compressed path */
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("int96_roundtrip", "writer create"); }
    if (carquet_writer_write_batch(w, 0, in, N, NULL, NULL) != CARQUET_OK)
        { carquet_writer_close(w); carquet_schema_free(s);
          TEST_FAIL("int96_roundtrip", "write batch"); }
    if (carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("int96_roundtrip", "close"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL("int96_roundtrip", "open"); }
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    int64_t n = c ? carquet_column_read_batch(c, out, N, NULL, NULL) : -1;
    int ok = (n == N) && (memcmp(in, out, sizeof(in)) == 0);
    carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("int96_roundtrip", "value mismatch");
    TEST_PASS("int96_roundtrip");
    return 0;
}

/* ---- Data Page V2 ---- */
static int read_file(const char* path, uint8_t** buf, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    carquet_fseek64(f, 0, SEEK_END);
    int64_t sz = carquet_ftell64(f);
    carquet_fseek64(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return 0; }
    *buf = (uint8_t*)malloc((size_t)sz);
    if (!*buf) { fclose(f); return 0; }
    size_t rd = fread(*buf, 1, (size_t)sz, f);
    fclose(f);
    *size = rd;
    return rd == (size_t)sz;
}

static int test_data_page_v2(int nullable) {
    const char* name = nullable ? "data_page_v2_nullable" : "data_page_v2";
    char path[512]; carquet_test_temp_path(path, sizeof(path), name);
    carquet_error_t err = CARQUET_ERROR_INIT;

    static int32_t in[N], out[N];
    static int16_t def[N], outdef[N];
    int64_t nn = 0;
    for (int i = 0; i < N; i++) {
        if (nullable && (i % 4 == 0)) { def[i] = 0; continue; }
        if (nullable) def[i] = 1;
        in[nn++] = (i * 31) - 7000 + (i % 17);
    }
    int64_t logical = nullable ? N : N;
    int64_t valcount = nullable ? nn : N;

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("data_page_v2", "schema create");
    carquet_field_repetition_t rep = nullable ? CARQUET_REPETITION_OPTIONAL
                                              : CARQUET_REPETITION_REQUIRED;
    if (carquet_schema_add_column(s, "v", CARQUET_PHYSICAL_INT32, NULL,
            rep, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("data_page_v2", "add col"); }

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.data_page_version = 2;
    wo.compression = CARQUET_COMPRESSION_ZSTD;
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("data_page_v2", "writer create"); }
    if (carquet_writer_write_batch(w, 0, in, logical, nullable ? def : NULL, NULL)
            != CARQUET_OK)
        { carquet_writer_close(w); carquet_schema_free(s);
          TEST_FAIL("data_page_v2", "write batch"); }
    if (carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("data_page_v2", "close"); }
    carquet_schema_free(s);

    /* Structural assertion: parse the first data page header on disk. */
    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL("data_page_v2", "open"); }
    carquet_column_chunk_metadata_t cm;
    if (carquet_reader_column_chunk_metadata(r, 0, 0, &cm) != CARQUET_OK)
        { carquet_reader_close(r); carquet_test_cleanup(path);
          TEST_FAIL("data_page_v2", "chunk meta"); }

    uint8_t* fb = NULL; size_t fsz = 0;
    if (!read_file(path, &fb, &fsz))
        { carquet_reader_close(r); carquet_test_cleanup(path);
          TEST_FAIL("data_page_v2", "read file"); }
    parquet_page_header_t ph; size_t consumed = 0;
    carquet_status_t pst = parquet_parse_page_header(
        fb + cm.data_page_offset, fsz - (size_t)cm.data_page_offset,
        &ph, &consumed, &err);
    int v2_ok = (pst == CARQUET_OK) && (ph.type == CARQUET_PAGE_DATA_V2) &&
                (ph.data_page_header_v2.num_values == logical) &&
                (ph.data_page_header_v2.num_rows == logical) &&
                (ph.data_page_header_v2.num_nulls == (logical - valcount));
    free(fb);
    if (!v2_ok) { carquet_reader_close(r); carquet_test_cleanup(path);
        TEST_FAIL("data_page_v2", "page is not a well-formed DATA_PAGE_V2"); }

    /* Value roundtrip through carquet's own reader. */
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    int64_t got = c ? carquet_column_read_batch(c, out, N,
                        nullable ? outdef : NULL, NULL) : -1;
    int ok = (got == logical);
    if (ok && nullable) {
        int64_t vi = 0;
        for (int64_t i = 0; ok && i < N; i++) {
            if (outdef[i] == 0) continue;
            ok = (out[vi] == in[vi]); vi++;
        }
        ok = ok && (vi == nn);
    } else if (ok) {
        ok = (memcmp(in, out, sizeof(in)) == 0);
    }
    carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("data_page_v2", "value mismatch");
    TEST_PASS(name);
    return 0;
}

/* ---- ARROW:schema metadata ---- */
static int test_arrow_schema_metadata(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_arrow_schema");
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("arrow_schema_metadata", "schema create");
    carquet_logical_type_t str_lt = { .id = CARQUET_LOGICAL_STRING };
    if (carquet_schema_add_column(s, "id", CARQUET_PHYSICAL_INT64, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(s, "name", CARQUET_PHYSICAL_BYTE_ARRAY, &str_lt,
            CARQUET_REPETITION_OPTIONAL, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(s, "score", CARQUET_PHYSICAL_DOUBLE, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("arrow_schema_metadata", "add cols"); }

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.write_arrow_schema = true;
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("arrow_schema_metadata", "writer"); }
    int64_t ids[3] = { 1, 2, 3 };
    double sc[3] = { 1.5, 2.5, 3.5 };
    carquet_byte_array_t nm[3];
    const char* names[3] = { "a", "bb", "ccc" };
    for (int i = 0; i < 3; i++)
        { nm[i].data = (uint8_t*)names[i]; nm[i].length = (int32_t)strlen(names[i]); }
    int16_t ndef[3] = { 1, 1, 1 };
    if (carquet_writer_write_batch(w, 0, ids, 3, NULL, NULL) != CARQUET_OK ||
        carquet_writer_write_batch(w, 1, nm, 3, ndef, NULL) != CARQUET_OK ||
        carquet_writer_write_batch(w, 2, sc, 3, NULL, NULL) != CARQUET_OK)
        { carquet_writer_close(w); carquet_schema_free(s);
          TEST_FAIL("arrow_schema_metadata", "write"); }
    if (carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("arrow_schema_metadata", "close"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL("arrow_schema_metadata", "open"); }
    const char* v = carquet_reader_find_metadata(r, "ARROW:schema");
    /* Base64 of the encapsulated message begins with the 0xFFFFFFFF
     * continuation marker => the first 3 bytes (FF FF FF) base64 to "////". */
    int ok = v && strlen(v) > 8 && strncmp(v, "////", 4) == 0;
    carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("arrow_schema_metadata", "ARROW:schema missing/malformed");
    TEST_PASS("arrow_schema_metadata");
    return 0;
}

/* Tolerant base64 decode into a heap buffer (test-local). */
static uint8_t* fm_b64_decode(const char* in, size_t* out_len) {
    signed char T[256];
    for (int i = 0; i < 256; i++) T[i] = -1;
    const char* A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; i++) T[(unsigned char)A[i]] = (signed char)i;
    size_t n = strlen(in);
    uint8_t* out = (uint8_t*)malloc(n / 4 * 3 + 4);
    if (!out) return NULL;
    size_t o = 0; int q[4], qn = 0;
    for (size_t i = 0; i < n; i++) {
        int v = T[(unsigned char)in[i]];
        if (v < 0) continue;
        q[qn++] = v;
        if (qn == 4) {
            out[o++] = (uint8_t)((q[0] << 2) | (q[1] >> 4));
            out[o++] = (uint8_t)((q[1] << 4) | (q[2] >> 2));
            out[o++] = (uint8_t)((q[2] << 6) | q[3]);
            qn = 0;
        }
    }
    if (qn >= 2) { out[o++] = (uint8_t)((q[0] << 2) | (q[1] >> 4));
        if (qn >= 3) out[o++] = (uint8_t)((q[1] << 4) | (q[2] >> 2)); }
    *out_len = o;
    return out;
}

static int mem_contains(const uint8_t* hay, size_t hn, const char* needle) {
    size_t nn = strlen(needle);
    if (nn == 0 || nn > hn) return 0;
    for (size_t i = 0; i + nn <= hn; i++)
        if (memcmp(hay + i, needle, nn) == 0) return 1;
    return 0;
}

/* ---- Arrow per-field custom_metadata (variable labels) ---- */
static int test_field_metadata_roundtrip(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_field_meta");
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("field_metadata", "schema create");
    if (carquet_schema_add_column(s, "SurveyVar", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_OPTIONAL, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("field_metadata", "add c0"); }
    int32_t c0 = carquet_schema_num_elements(s) - 1;
    if (carquet_schema_set_field_metadata(s, c0, "Label",
            "Numeric With Labels and Missing") != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("field_metadata", "set c0 label"); }
    if (carquet_schema_set_field_metadata(s, c0, "Note", "n1") != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("field_metadata", "set c0 note"); }

    if (carquet_schema_add_column(s, "Sex", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("field_metadata", "add c1"); }
    int32_t c1 = carquet_schema_num_elements(s) - 1;
    /* Set twice with the same key: replace, must not duplicate. */
    if (carquet_schema_set_field_metadata(s, c1, "Label", "WRONG") != CARQUET_OK ||
        carquet_schema_set_field_metadata(s, c1, "Label",
            "Sex of Respondent") != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("field_metadata", "set c1 label"); }

    /* Third column intentionally has no metadata. */
    if (carquet_schema_add_column(s, "Name", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_OPTIONAL, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("field_metadata", "add c2"); }

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.write_arrow_schema = true;
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("field_metadata", "writer"); }
    int32_t v[3] = { 1, 2, 3 };
    int16_t def[3] = { 1, 1, 1 };
    if (carquet_writer_write_batch(w, 0, v, 3, def, NULL) != CARQUET_OK ||
        carquet_writer_write_batch(w, 1, v, 3, NULL, NULL) != CARQUET_OK ||
        carquet_writer_write_batch(w, 2, v, 3, def, NULL) != CARQUET_OK)
        { carquet_writer_close(w); carquet_schema_free(s);
          TEST_FAIL("field_metadata", "write"); }
    if (carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("field_metadata", "close"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL("field_metadata", "open"); }

    /* Byte-level: the exact label strings must appear in the decoded blob. */
    const char* b64 = carquet_reader_find_metadata(r, "ARROW:schema");
    if (!b64) { carquet_reader_close(r); carquet_test_cleanup(path);
        TEST_FAIL("field_metadata", "no ARROW:schema"); }
    size_t rawlen = 0;
    uint8_t* raw = fm_b64_decode(b64, &rawlen);
    int bytes_ok = raw &&
        mem_contains(raw, rawlen, "Label") &&
        mem_contains(raw, rawlen, "Numeric With Labels and Missing") &&
        mem_contains(raw, rawlen, "Sex of Respondent") &&
        !mem_contains(raw, rawlen, "WRONG");
    free(raw);
    if (!bytes_ok) { carquet_reader_close(r); carquet_test_cleanup(path);
        TEST_FAIL("field_metadata", "label bytes missing in ARROW:schema"); }

    /* API round-trip through carquet's own reader. */
    int api_ok =
        carquet_reader_column_num_metadata(r, 0) == 2 &&
        carquet_reader_column_num_metadata(r, 1) == 1 &&
        carquet_reader_column_num_metadata(r, 2) == 0;
    const char* l0 = carquet_reader_column_find_metadata(r, 0, "Label");
    const char* l1 = carquet_reader_column_find_metadata(r, 1, "Label");
    const char* n0 = carquet_reader_column_find_metadata(r, 0, "Note");
    api_ok = api_ok && l0 && strcmp(l0, "Numeric With Labels and Missing") == 0;
    api_ok = api_ok && l1 && strcmp(l1, "Sex of Respondent") == 0;
    api_ok = api_ok && n0 && strcmp(n0, "n1") == 0;
    api_ok = api_ok && carquet_reader_column_find_metadata(r, 2, "Label") == NULL;
    /* get-by-index bounds */
    const char *k, *val;
    api_ok = api_ok &&
        carquet_reader_column_get_metadata(r, 1, 0, &k, &val) == CARQUET_OK &&
        carquet_reader_column_get_metadata(r, 1, 1, &k, &val) != CARQUET_OK;

    carquet_reader_close(r); carquet_test_cleanup(path);
    if (!api_ok) TEST_FAIL("field_metadata", "reader API mismatch");
    TEST_PASS("field_metadata");
    return 0;
}

static int test_field_metadata_errors(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("field_metadata_errors", "schema create");
    if (carquet_schema_add_column(s, "x", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("field_metadata_errors", "add col"); }
    int32_t idx = carquet_schema_num_elements(s) - 1;

    int ok = 1;
    /* Root (index 0) is rejected. */
    ok = ok && carquet_schema_set_field_metadata(s, 0, "Label", "v")
               == CARQUET_ERROR_INVALID_ARGUMENT;
    /* Negative / out-of-range indices rejected. */
    ok = ok && carquet_schema_set_field_metadata(s, -1, "Label", "v")
               == CARQUET_ERROR_INVALID_ARGUMENT;
    ok = ok && carquet_schema_set_field_metadata(s, idx + 5, "Label", "v")
               == CARQUET_ERROR_INVALID_ARGUMENT;
    /* Valid element, NULL value permitted; valid key required (NONNULL). */
    ok = ok && carquet_schema_set_field_metadata(s, idx, "Label", NULL)
               == CARQUET_OK;
    ok = ok && carquet_schema_set_field_metadata(s, idx, "Label", "ok")
               == CARQUET_OK;

    carquet_schema_free(s);
    if (!ok) TEST_FAIL("field_metadata_errors", "unexpected status");
    TEST_PASS("field_metadata_errors");
    return 0;
}

static int test_arrow_schema_skipped_when_off(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_arrow_off");
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("arrow_schema_off", "schema create");
    if (carquet_schema_add_column(s, "id", CARQUET_PHYSICAL_INT64, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("arrow_schema_off", "add col"); }
    carquet_writer_options_t wo; carquet_writer_options_init(&wo);  /* default: off */
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    int64_t ids[2] = { 10, 20 };
    if (!w || carquet_writer_write_batch(w, 0, ids, 2, NULL, NULL) != CARQUET_OK ||
        carquet_writer_close(w) != CARQUET_OK)
        { if (w) carquet_writer_close(w); carquet_schema_free(s);
          TEST_FAIL("arrow_schema_off", "write"); }
    carquet_schema_free(s);
    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    int ok = r && carquet_reader_find_metadata(r, "ARROW:schema") == NULL;
    if (r) carquet_reader_close(r);
    carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("arrow_schema_off", "ARROW:schema present when disabled");
    TEST_PASS("arrow_schema_off");
    return 0;
}

/* ---- Nested ARROW:schema emission (LIST / STRUCT / MAP) ---- */
static int test_arrow_schema_nested(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_arrow_nested");
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("arrow_schema_nested", "schema create");
    /* id INT64 REQUIRED, tags LIST<int32>, addr STRUCT{street,zip}, props MAP */
    carquet_schema_add_column(s, "id", CARQUET_PHYSICAL_INT64, NULL,
                              CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_schema_add_list(s, "tags", CARQUET_PHYSICAL_INT32, NULL,
                            CARQUET_REPETITION_OPTIONAL, 0, 0);
    int32_t addr = carquet_schema_add_group(s, "addr", CARQUET_REPETITION_OPTIONAL, 0);
    carquet_logical_type_t str_lt = { .id = CARQUET_LOGICAL_STRING };
    carquet_schema_add_column(s, "street", CARQUET_PHYSICAL_BYTE_ARRAY, &str_lt,
                              CARQUET_REPETITION_OPTIONAL, 0, addr);
    carquet_schema_add_column(s, "zip", CARQUET_PHYSICAL_INT32, NULL,
                              CARQUET_REPETITION_REQUIRED, 0, addr);
    carquet_schema_add_map(s, "props", CARQUET_PHYSICAL_INT32, NULL, 0,
                           CARQUET_PHYSICAL_INT64, NULL, 0,
                           CARQUET_REPETITION_OPTIONAL, 0);

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.write_arrow_schema = true;
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("arrow_schema_nested", "writer"); }
    int64_t id = 1;
    int32_t toff[2] = {0, 2}, tval[2] = {10, 20};
    carquet_byte_array_t street = { (uint8_t*)"x", 1 };
    int16_t sdef[1] = {2};
    int32_t zip = 90210; int16_t zdef[1] = {1};
    int32_t poff[2] = {0, 1}, keys[1] = {5}; int64_t vals[1] = {500};
    int ok_w =
        carquet_writer_write_batch(w, 0, &id, 1, NULL, NULL) == CARQUET_OK &&
        carquet_writer_write_list_column(w, 1, 1, toff, NULL, tval, NULL, &err) == CARQUET_OK &&
        carquet_writer_write_batch(w, 2, &street, 1, sdef, NULL) == CARQUET_OK &&
        carquet_writer_write_batch(w, 3, &zip, 1, zdef, NULL) == CARQUET_OK &&
        carquet_writer_write_list_column(w, 4, 1, poff, NULL, keys, NULL, &err) == CARQUET_OK &&
        carquet_writer_write_list_column(w, 5, 1, poff, NULL, vals, NULL, &err) == CARQUET_OK;
    if (!ok_w || carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("arrow_schema_nested", "write"); }
    carquet_schema_free(s);

    /* A nested schema now emits ARROW:schema (previously flat-only ⇒ absent).
     * Verify the encapsulated-message framing survived (base64 "////" prefix). */
    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL("arrow_schema_nested", "open"); }
    const char* v = carquet_reader_find_metadata(r, "ARROW:schema");
    int ok = v && strlen(v) > 8 && strncmp(v, "////", 4) == 0;
    carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("arrow_schema_nested", "ARROW:schema missing for nested schema");
    TEST_PASS("arrow_schema_nested");
    return 0;
}

/* ---- Read Arrow type refinements (LargeUtf8 / LargeBinary) from ARROW:schema.
 * Uses a real PyArrow-emitted "ARROW:schema" blob for a table
 * {ls: large_string, lb: large_binary, reg: string}, injected as file metadata
 * onto a carquet file whose leaf columns are named to match. On read, carquet
 * must recover the 64-bit-offset refinement per matched leaf. ---- */
static int test_arrow_type_refinement_read(void) {
    /* PyArrow 23 output for pa.table({'ls':large_string,'lb':large_binary,'reg':string}). */
    static const char* PYARROW_LARGE_BLOB =
        "/////8AAAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAMAAAACAAIAAAABAAIAAAABAAAAAMA"
        "AABkAAAALAAAAAQAAAC4////AAABBRAAAAAUAAAABAAAAAAAAAADAAAAcmVnAKj////c////"
        "AAABExAAAAAUAAAABAAAAAAAAAACAAAAbGIAAMz///8QABQACAAGAAcADAAAABAAEAAAAAAA"
        "ARQQAAAAGAAAAAQAAAAAAAAAAgAAAGxzAAAEAAQABAAAAAAAAAA=";

    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_arrow_refine");
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("arrow_type_refinement_read", "schema create");
    carquet_logical_type_t str_lt = { .id = CARQUET_LOGICAL_STRING };
    /* Names must match the Arrow field names in the blob: ls, lb, reg. */
    carquet_schema_add_column(s, "ls", CARQUET_PHYSICAL_BYTE_ARRAY, &str_lt,
                              CARQUET_REPETITION_OPTIONAL, 0, 0);
    carquet_schema_add_column(s, "lb", CARQUET_PHYSICAL_BYTE_ARRAY, NULL,
                              CARQUET_REPETITION_OPTIONAL, 0, 0);
    carquet_schema_add_column(s, "reg", CARQUET_PHYSICAL_BYTE_ARRAY, &str_lt,
                              CARQUET_REPETITION_OPTIONAL, 0, 0);

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);  /* our own emit OFF */
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("arrow_type_refinement_read", "writer"); }
    if (carquet_writer_add_metadata(w, "ARROW:schema", PYARROW_LARGE_BLOB) != CARQUET_OK)
        { carquet_writer_close(w); carquet_schema_free(s);
          TEST_FAIL("arrow_type_refinement_read", "add_metadata"); }
    carquet_byte_array_t a = { (uint8_t*)"a", 1 };
    int16_t d1[1] = {1};
    int ok_w =
        carquet_writer_write_batch(w, 0, &a, 1, d1, NULL) == CARQUET_OK &&
        carquet_writer_write_batch(w, 1, &a, 1, d1, NULL) == CARQUET_OK &&
        carquet_writer_write_batch(w, 2, &a, 1, d1, NULL) == CARQUET_OK;
    if (!ok_w || carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("arrow_type_refinement_read", "write"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL("arrow_type_refinement_read", "open"); }
    carquet_arrow_type_refinement_t ref_ls =
        carquet_reader_column_arrow_type_refinement(r, 0);
    carquet_arrow_type_refinement_t ref_lb =
        carquet_reader_column_arrow_type_refinement(r, 1);
    carquet_arrow_type_refinement_t ref_reg =
        carquet_reader_column_arrow_type_refinement(r, 2);
    carquet_reader_close(r); carquet_test_cleanup(path);

    if (ref_ls != CARQUET_ARROW_REFINE_LARGE_UTF8)
        TEST_FAIL("arrow_type_refinement_read", "ls not LARGE_UTF8");
    if (ref_lb != CARQUET_ARROW_REFINE_LARGE_BINARY)
        TEST_FAIL("arrow_type_refinement_read", "lb not LARGE_BINARY");
    if (ref_reg != CARQUET_ARROW_REFINE_NONE)
        TEST_FAIL("arrow_type_refinement_read", "reg should be NONE");
    TEST_PASS("arrow_type_refinement_read");
    return 0;
}

/* ---- FLOAT16 statistics ordering ---- */
static uint16_t f16_le(const uint8_t* b) { return (uint16_t)(b[0] | (b[1] << 8)); }

static int test_float16_stats(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_f16");
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Halfs whose lexicographic byte order differs from numeric order:
     * -2.0=0xC000, 1.0=0x3C00, 0.5=0x3800, plus a NaN (0x7E00) to skip.
     * Numeric min = -2.0, max = 1.0. Lexicographic would pick 0.5 / -2.0. */
    static const uint16_t H[4] = { 0xC000, 0x3C00, 0x3800, 0x7E00 };
    uint8_t vals[4 * 2];
    for (int i = 0; i < 4; i++) { vals[i*2] = H[i] & 0xFF; vals[i*2+1] = H[i] >> 8; }

    carquet_schema_t* s = carquet_schema_create(&err);
    carquet_logical_type_t f16 = { .id = CARQUET_LOGICAL_FLOAT16 };
    if (!s || carquet_schema_add_column(s, "h", CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY,
            &f16, CARQUET_REPETITION_REQUIRED, 2, 0) != CARQUET_OK)
        { if (s) carquet_schema_free(s); TEST_FAIL("float16_stats", "schema"); }
    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w || carquet_writer_write_batch(w, 0, vals, 4, NULL, NULL) != CARQUET_OK ||
        carquet_writer_close(w) != CARQUET_OK)
        { if (w) carquet_writer_close(w); carquet_schema_free(s);
          TEST_FAIL("float16_stats", "write"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    carquet_column_statistics_t st;
    int ok = r && carquet_reader_column_statistics(r, 0, 0, &st) == CARQUET_OK &&
             st.has_min_max && st.min_value_size == 2 && st.max_value_size == 2 &&
             f16_le((const uint8_t*)st.min_value) == 0xC000 &&   /* -2.0 */
             f16_le((const uint8_t*)st.max_value) == 0x3C00;     /*  1.0 */
    if (r) carquet_reader_close(r);
    carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("float16_stats", "min/max not numeric-ordered / NaN not skipped");
    TEST_PASS("float16_stats");
    return 0;
}

/* ---- deprecated BIT_PACKED level decoding ---- */
static int test_bitpacked_levels(void) {
    /* Spec worked example: values 0..7 at 3-bit width MSB-first pack to
     * 0x05 0x39 0x77. */
    const uint8_t packed[3] = { 0x05, 0x39, 0x77 };
    int16_t out[8];
    size_t consumed = 0;
    if (carquet_decode_bitpacked_levels(packed, sizeof(packed), 3, 8,
                                        out, &consumed) != 0)
        TEST_FAIL("bitpacked_levels", "decode failed");
    for (int i = 0; i < 8; i++)
        if (out[i] != i) TEST_FAIL("bitpacked_levels", "wrong value");
    if (consumed != 3) TEST_FAIL("bitpacked_levels", "wrong byte count");

    /* bit_width 0 => all zeros, no input consumed. */
    int16_t z[5];
    if (carquet_decode_bitpacked_levels(NULL, 0, 0, 5, z, &consumed) != 0 ||
        consumed != 0)
        TEST_FAIL("bitpacked_levels", "zero-width");
    for (int i = 0; i < 5; i++)
        if (z[i] != 0) TEST_FAIL("bitpacked_levels", "zero-width value");

    /* Truncated input must fail, not over-read. */
    if (carquet_decode_bitpacked_levels(packed, 1, 3, 8, out, &consumed) == 0)
        TEST_FAIL("bitpacked_levels", "truncation not detected");
    TEST_PASS("bitpacked_levels");
    return 0;
}

/* ---- GEOMETRY GeospatialStatistics ---- */
static void put_wkb_point(uint8_t* b, double x, double y) {
    b[0] = 1;                     /* little-endian */
    b[1] = 1; b[2] = 0; b[3] = 0; b[4] = 0;   /* type 1 = Point XY */
    memcpy(b + 5, &x, 8);
    memcpy(b + 13, &y, 8);
}

static int test_geospatial_stats(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_geo");
    carquet_error_t err = CARQUET_ERROR_INIT;

    static uint8_t p0[21], p1[21], p2[21];
    put_wkb_point(p0, 1.0, 2.0);
    put_wkb_point(p1, 5.0, 6.0);
    put_wkb_point(p2, 3.0, -4.0);
    carquet_byte_array_t ba[3] = {
        { p0, 21 }, { p1, 21 }, { p2, 21 } };

    carquet_schema_t* s = carquet_schema_create(&err);
    carquet_logical_type_t geo = { .id = CARQUET_LOGICAL_GEOMETRY };
    if (!s || carquet_schema_add_column(s, "g", CARQUET_PHYSICAL_BYTE_ARRAY,
            &geo, CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { if (s) carquet_schema_free(s); TEST_FAIL("geospatial_stats", "schema"); }
    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w || carquet_writer_write_batch(w, 0, ba, 3, NULL, NULL) != CARQUET_OK ||
        carquet_writer_close(w) != CARQUET_OK)
        { if (w) carquet_writer_close(w); carquet_schema_free(s);
          TEST_FAIL("geospatial_stats", "write"); }
    carquet_schema_free(s);

    /* Read back through the public GeospatialStatistics API. */
    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    carquet_geospatial_statistics_t gs;
    int ok = r && carquet_reader_geospatial_statistics(r, 0, 0, &gs)
                 == CARQUET_OK &&
             gs.has_bbox &&
             gs.xmin == 1.0 && gs.xmax == 5.0 &&
             gs.ymin == -4.0 && gs.ymax == 6.0 &&
             !gs.has_z && !gs.has_m &&
             gs.num_geometry_types == 1 && gs.geometry_types[0] == 1;
    if (r) carquet_reader_close(r);
    carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("geospatial_stats", "bbox/types mismatch");
    TEST_PASS("geospatial_stats");
    return 0;
}

/* ---- TIMESTAMP coercion ---- */
static int test_timestamp_coercion(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_tscoerce");
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Nanos; the first has a sub-microsecond remainder. */
    int64_t ns[3] = { 1500000000123456789LL,
                      1500000000123456000LL,
                      2000000000000000000LL };

    /* (a) coerce to micros WITHOUT allowing truncation -> write must fail. */
    {
        carquet_schema_t* s = carquet_schema_create(&err);
        carquet_logical_type_t ts = { .id = CARQUET_LOGICAL_TIMESTAMP };
        ts.params.timestamp.unit = CARQUET_TIME_UNIT_NANOS;
        ts.params.timestamp.is_adjusted_to_utc = true;
        carquet_schema_add_column(s, "t", CARQUET_PHYSICAL_INT64, &ts,
                                  CARQUET_REPETITION_REQUIRED, 0, 0);
        carquet_writer_options_t wo; carquet_writer_options_init(&wo);
        wo.coerce_timestamps = true;
        wo.coerce_timestamp_unit = CARQUET_TIME_UNIT_MICROS;
        wo.allow_timestamp_truncation = false;
        carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
        carquet_status_t st = w ? carquet_writer_write_batch(w, 0, ns, 3, NULL, NULL)
                                : CARQUET_OK;
        if (w) carquet_writer_close(w);
        carquet_schema_free(s);
        carquet_test_cleanup(path);
        if (st == CARQUET_OK)
            TEST_FAIL("timestamp_coercion", "lossy write should have failed");
    }

    /* (b) allow truncation -> values divided by 1000, schema unit = MICROS. */
    carquet_schema_t* s = carquet_schema_create(&err);
    carquet_logical_type_t ts = { .id = CARQUET_LOGICAL_TIMESTAMP };
    ts.params.timestamp.unit = CARQUET_TIME_UNIT_NANOS;
    ts.params.timestamp.is_adjusted_to_utc = true;
    carquet_schema_add_column(s, "t", CARQUET_PHYSICAL_INT64, &ts,
                              CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.coerce_timestamps = true;
    wo.coerce_timestamp_unit = CARQUET_TIME_UNIT_MICROS;
    wo.allow_timestamp_truncation = true;
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w || carquet_writer_write_batch(w, 0, ns, 3, NULL, NULL) != CARQUET_OK ||
        carquet_writer_close(w) != CARQUET_OK)
        { if (w) carquet_writer_close(w); carquet_schema_free(s);
          carquet_test_cleanup(path);
          TEST_FAIL("timestamp_coercion", "truncated write failed"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    int64_t out[3];
    carquet_column_reader_t* c = r ? carquet_reader_get_column(r, 0, 0, &err) : NULL;
    int64_t n = c ? carquet_column_read_batch(c, out, 3, NULL, NULL) : -1;
    int ok = (n == 3) &&
             out[0] == 1500000000123456LL &&
             out[1] == 1500000000123456LL &&
             out[2] == 2000000000000000LL;
    /* Emitted schema must advertise the coerced unit. */
    const carquet_schema_t* rs = r ? carquet_reader_schema(r) : NULL;
    const carquet_schema_node_t* node = rs ?
        carquet_schema_get_element(rs, 1) : NULL;
    const carquet_logical_type_t* rlt = node ?
        carquet_schema_node_logical_type(node) : NULL;
    ok = ok && rlt && rlt->id == CARQUET_LOGICAL_TIMESTAMP &&
         rlt->params.timestamp.unit == CARQUET_TIME_UNIT_MICROS;
    carquet_column_reader_free(c); if (r) carquet_reader_close(r);
    carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("timestamp_coercion", "value/unit mismatch");
    TEST_PASS("timestamp_coercion");
    return 0;
}

/* ---- write_batch_size correctness under tiny batches ---- */
static int test_write_batch_size(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_wbs");
    carquet_error_t err = CARQUET_ERROR_INIT;
    enum { M = 5000 };
    static int32_t in[M], out[M];
    /* LCG-style spread; compute in unsigned to avoid signed overflow UB. */
    for (int i = 0; i < M; i++)
        in[i] = (int32_t)((uint32_t)i * 1103515245u + 12345u);

    carquet_schema_t* s = carquet_schema_create(&err);
    carquet_schema_add_column(s, "v", CARQUET_PHYSICAL_INT32, NULL,
                              CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.write_batch_size = 64;       /* force many tiny internal chunks */
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w || carquet_writer_write_batch(w, 0, in, M, NULL, NULL) != CARQUET_OK ||
        carquet_writer_close(w) != CARQUET_OK)
        { if (w) carquet_writer_close(w); carquet_schema_free(s);
          carquet_test_cleanup(path);
          TEST_FAIL("write_batch_size", "write failed"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    carquet_column_reader_t* c = r ? carquet_reader_get_column(r, 0, 0, &err) : NULL;
    int64_t n = c ? carquet_column_read_batch(c, out, M, NULL, NULL) : -1;
    int ok = (n == M) && (memcmp(in, out, sizeof(in)) == 0);
    carquet_column_reader_free(c); if (r) carquet_reader_close(r);
    carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("write_batch_size", "value mismatch");
    TEST_PASS("write_batch_size");
    return 0;
}

/* ---- Mixing the two bloom-filter APIs ----
 * Regression: once the ndv/fpp options API is used for ANY column, a column
 * explicitly enabled through the legacy set_column_bloom_filter() must still
 * get its bloom filter (it used to be silently dropped). A column left
 * untouched must NOT gain one from the options API's global-flag side effect. */
static int test_bloom_api_mix(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_bloom_mix");
    carquet_error_t err = CARQUET_ERROR_INIT;
    enum { M = 200 };
    static int64_t a[M], b[M], c[M];
    for (int i = 0; i < M; i++) { a[i] = i; b[i] = i * 2; c[i] = i * 3; }

    carquet_schema_t* s = carquet_schema_create(&err);
    carquet_schema_add_column(s, "a", CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_schema_add_column(s, "b", CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_schema_add_column(s, "c", CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.write_bloom_filters = false;   /* global off; per-column opt-in only */
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); carquet_test_cleanup(path);
              TEST_FAIL("bloom_api_mix", "writer create"); }

    /* a: legacy per-column enable; b: new options API; c: untouched. */
    carquet_writer_set_column_bloom_filter(w, 0, true);
    carquet_writer_set_column_bloom_filter_options(w, 1, true, 1024, 0.01);

    if (carquet_writer_write_batch(w, 0, a, M, NULL, NULL) != CARQUET_OK ||
        carquet_writer_write_batch(w, 1, b, M, NULL, NULL) != CARQUET_OK ||
        carquet_writer_write_batch(w, 2, c, M, NULL, NULL) != CARQUET_OK ||
        carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); carquet_test_cleanup(path);
          TEST_FAIL("bloom_api_mix", "write failed"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL("bloom_api_mix", "reader open"); }
    carquet_column_chunk_metadata_t ma, mb, mc;
    carquet_reader_column_chunk_metadata(r, 0, 0, &ma);
    carquet_reader_column_chunk_metadata(r, 0, 1, &mb);
    carquet_reader_column_chunk_metadata(r, 0, 2, &mc);
    int ok = ma.has_bloom_filter && mb.has_bloom_filter && !mc.has_bloom_filter;
    carquet_reader_close(r);
    carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("bloom_api_mix",
        "legacy-enabled column lost its bloom, or untouched column gained one");
    TEST_PASS("bloom_api_mix");
    return 0;
}

/* ---- Append row groups to existing file ---- */
static int test_append_row_groups(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_append");
    carquet_error_t err = CARQUET_ERROR_INIT;
    enum { N1 = 1000, N2 = 1500, N3 = 800 };
    static int32_t v1[N1], v2[N2], v3[N3], out[N1 + N2 + N3];
    for (int i = 0; i < N1; i++) v1[i] = i;
    for (int i = 0; i < N2; i++) v2[i] = N1 + i;
    for (int i = 0; i < N3; i++) v3[i] = N1 + N2 + i;

    /* Build schema reused across the three writers. */
    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("append", "schema");
    if (carquet_schema_add_column(s, "v", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("append", "add col"); }

    /* 1) Initial write: one row group with v1. */
    {
        carquet_writer_t* w = carquet_writer_create(path, s, NULL, &err);
        if (!w) { carquet_schema_free(s); TEST_FAIL("append", "create"); }
        if (carquet_writer_add_metadata(w, "origin", "v0.5") != CARQUET_OK ||
            carquet_writer_write_batch(w, 0, v1, N1, NULL, NULL) != CARQUET_OK ||
            carquet_writer_close(w) != CARQUET_OK)
            { carquet_schema_free(s); carquet_test_cleanup(path);
              TEST_FAIL("append", "initial write"); }
    }

    /* 2) Append: add a second row group with v2. */
    {
        carquet_writer_t* w = carquet_writer_open_append(path, s, NULL, &err);
        if (!w) { carquet_schema_free(s); carquet_test_cleanup(path);
                  TEST_FAIL("append", "open_append"); }
        if (carquet_writer_write_batch(w, 0, v2, N2, NULL, NULL) != CARQUET_OK ||
            carquet_writer_close(w) != CARQUET_OK)
            { carquet_schema_free(s); carquet_test_cleanup(path);
              TEST_FAIL("append", "append1 write"); }
    }

    /* 3) Append again: a third row group with v3 plus extra metadata. */
    {
        carquet_writer_t* w = carquet_writer_open_append(path, s, NULL, &err);
        if (!w) { carquet_schema_free(s); carquet_test_cleanup(path);
                  TEST_FAIL("append", "open_append2"); }
        if (carquet_writer_add_metadata(w, "appended", "yes") != CARQUET_OK ||
            carquet_writer_write_batch(w, 0, v3, N3, NULL, NULL) != CARQUET_OK ||
            carquet_writer_close(w) != CARQUET_OK)
            { carquet_schema_free(s); carquet_test_cleanup(path);
              TEST_FAIL("append", "append2 write"); }
    }
    carquet_schema_free(s);

    /* Verify: three row groups, correct total, values intact in order. */
    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL("append", "open reader"); }
    int32_t num_rg = carquet_reader_num_row_groups(r);
    int64_t num_rows = carquet_reader_num_rows(r);
    int ok = (num_rg == 3) && (num_rows == N1 + N2 + N3);
    if (ok) {
        int64_t got = 0;
        for (int32_t rg = 0; rg < num_rg && ok; rg++) {
            carquet_column_reader_t* c =
                carquet_reader_get_column(r, rg, 0, &err);
            if (!c) { ok = 0; break; }
            int64_t n = carquet_column_read_batch(c, out + got,
                                                  N1 + N2 + N3 - got,
                                                  NULL, NULL);
            carquet_column_reader_free(c);
            if (n < 0) { ok = 0; break; }
            got += n;
        }
        ok = ok && (got == N1 + N2 + N3);
        if (ok) {
            for (int i = 0; i < N1 + N2 + N3 && ok; i++) ok = (out[i] == i);
        }
    }
    carquet_reader_close(r);

    /* Schema-mismatch rejection: open append with a schema that has the
     * wrong physical type. */
    if (ok) {
        carquet_schema_t* bad = carquet_schema_create(&err);
        carquet_schema_add_column(bad, "v", CARQUET_PHYSICAL_INT64, NULL,
                                  CARQUET_REPETITION_REQUIRED, 0, 0);
        carquet_writer_t* w = carquet_writer_open_append(path, bad, NULL, &err);
        ok = ok && (w == NULL);
        if (w) carquet_writer_close(w);
        carquet_schema_free(bad);

        /* A rejected append must NOT destroy the pre-existing file: the
         * original three row groups must still be readable intact. */
        if (ok) {
            carquet_reader_t* r2 = carquet_reader_open(path, NULL, &err);
            ok = ok && (r2 != NULL)
                && (carquet_reader_num_row_groups(r2) == 3)
                && (carquet_reader_num_rows(r2) == N1 + N2 + N3);
            if (r2) carquet_reader_close(r2);
        }
    }

    /* Logical-type-mismatch rejection: same physical type (INT32) but a logical
     * annotation the existing chunks lack must be rejected, not silently
     * appended into a semantically inconsistent file. */
    if (ok) {
        carquet_logical_type_t int_lt = { .id = CARQUET_LOGICAL_INTEGER };
        int_lt.params.integer.bit_width = 32;
        int_lt.params.integer.is_signed = true;
        carquet_schema_t* bad_lt = carquet_schema_create(&err);
        carquet_schema_add_column(bad_lt, "v", CARQUET_PHYSICAL_INT32, &int_lt,
                                  CARQUET_REPETITION_REQUIRED, 0, 0);
        carquet_writer_t* w = carquet_writer_open_append(path, bad_lt, NULL, &err);
        ok = ok && (w == NULL);
        if (w) carquet_writer_close(w);
        carquet_schema_free(bad_lt);
    }
    carquet_test_cleanup(path);

    if (!ok) TEST_FAIL("append", "roundtrip / schema-check failed");
    TEST_PASS("append");
    return 0;
}

/* ---- Truncated BYTE_ARRAY statistics ---- */
/* Writes one BYTE_ARRAY value of `value_len` bytes filled with `fill`, then
 * reads back the stored min size. Returns -1 on failure. */
static int32_t write_and_read_min_size(int64_t cap, int value_len, uint8_t fill,
                                       int32_t* out_max_size) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_stats_trunc");
    carquet_error_t err = CARQUET_ERROR_INIT;
    int32_t min_size = -1;
    if (out_max_size) *out_max_size = -1;

    static uint8_t buf[512];
    if (value_len > (int)sizeof(buf)) return -1;
    memset(buf, fill, (size_t)value_len);
    carquet_byte_array_t v;
    v.data = buf;
    v.length = value_len;

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) return -1;
    carquet_logical_type_t str_lt = { .id = CARQUET_LOGICAL_STRING };
    if (carquet_schema_add_column(s, "v", CARQUET_PHYSICAL_BYTE_ARRAY, &str_lt,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); return -1; }

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); return -1; }
    if (cap > 0 && carquet_writer_set_max_statistics_size(w, cap) != CARQUET_OK)
        { carquet_writer_close(w); carquet_schema_free(s);
          carquet_test_cleanup(path); return -1; }

    if (carquet_writer_write_batch(w, 0, &v, 1, NULL, NULL) != CARQUET_OK ||
        carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); carquet_test_cleanup(path); return -1; }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (r) {
        carquet_column_statistics_t st;
        if (carquet_reader_column_statistics(r, 0, 0, &st) == CARQUET_OK &&
            st.has_min_max) {
            min_size = st.min_value_size;
            if (out_max_size) *out_max_size = st.max_value_size;
        } else if (out_max_size) {
            /* min/max suppressed — return 0 to flag the edge case. */
            *out_max_size = 0;
            min_size = 0;
        }
        carquet_reader_close(r);
    }
    carquet_test_cleanup(path);
    return min_size;
}

static int test_max_statistics_size(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    (void)err;

    /* Validation: non-positive bytes rejected. */
    {
        carquet_schema_t* s = carquet_schema_create(&err);
        carquet_schema_add_column(s, "v", CARQUET_PHYSICAL_INT32, NULL,
                                  CARQUET_REPETITION_REQUIRED, 0, 0);
        char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_stats_v");
        carquet_writer_t* w = carquet_writer_create(path, s, NULL, &err);
        int validation_ok =
            (carquet_writer_set_max_statistics_size(w, 0) ==
                CARQUET_ERROR_INVALID_ARGUMENT) &&
            (carquet_writer_set_max_statistics_size(w, -1) ==
                CARQUET_ERROR_INVALID_ARGUMENT);
        carquet_writer_close(w); carquet_schema_free(s); carquet_test_cleanup(path);
        if (!validation_ok) TEST_FAIL("max_statistics_size", "validation");
    }

    /* Default cap (32): a 100-byte value is truncated to 32. */
    int32_t n = write_and_read_min_size(0 /* unchanged */, 100, 'A', NULL);
    if (n != 32) TEST_FAIL("max_statistics_size", "default cap should produce 32-byte min");

    /* Custom cap of 64: same value truncated to 64. */
    n = write_and_read_min_size(64, 100, 'A', NULL);
    if (n != 64) TEST_FAIL("max_statistics_size", "cap=64 should produce 64-byte min");

    /* Cap larger than value: no truncation. */
    n = write_and_read_min_size(200, 50, 'A', NULL);
    if (n != 50) TEST_FAIL("max_statistics_size", "cap > value should not truncate");

    /* Cap of 1 byte: smallest legal cap, still works. */
    n = write_and_read_min_size(1, 100, 'A', NULL);
    if (n != 1) TEST_FAIL("max_statistics_size", "cap=1 should produce 1-byte min");

    /* All-0xFF input with cap < length: max prefix can't be incremented, so
     * the writer suppresses max — which currently makes has_min_max false in
     * the reader API (it only surfaces min+max as a pair). The point of the
     * test is that the file is still well-formed, not corrupted by an
     * invalid bound. */
    int32_t max_size = -1;
    n = write_and_read_min_size(32, 50, 0xFF, &max_size);
    /* n == 0 / max_size == 0 means the pair was suppressed, which is the
     * intended behavior for the all-0xFF edge case. */
    if (!(n == 0 && max_size == 0))
        TEST_FAIL("max_statistics_size", "all-0xFF max should be suppressed");

    TEST_PASS("max_statistics_size");
    return 0;
}

/* ---- Per-column page size override ---- */
/* With one column overridden to a tiny page size and another left at the
 * (large) default, the small-page column must accumulate many more pages.
 * We use the offset index to count pages exactly. */
static int test_column_page_size_override(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_col_pgsz");
    carquet_error_t err = CARQUET_ERROR_INIT;
    enum { ROWS = 20000 };
    static int32_t big_col[ROWS], small_col[ROWS];
    for (int i = 0; i < ROWS; i++) { big_col[i] = i; small_col[i] = i; }

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("column_page_size", "schema");
    if (carquet_schema_add_column(s, "big", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(s, "small", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("column_page_size", "add cols"); }

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.page_size = 1024 * 1024;        /* default: one big page per column */
    wo.write_page_index = true;        /* we need the offset index to count */
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("column_page_size", "writer"); }

    /* Validation: reject non-positive bytes and out-of-range index. */
    if (carquet_writer_set_column_page_size(w, 0, 0) != CARQUET_ERROR_INVALID_ARGUMENT ||
        carquet_writer_set_column_page_size(w, 0, -1) != CARQUET_ERROR_INVALID_ARGUMENT ||
        carquet_writer_set_column_page_size(w, 99, 1024) != CARQUET_ERROR_INVALID_ARGUMENT) {
        carquet_writer_close(w); carquet_schema_free(s); carquet_test_cleanup(path);
        TEST_FAIL("column_page_size", "validation");
    }
    /* Override column 1 to a tiny page size — forces many pages. */
    if (carquet_writer_set_column_page_size(w, 1, 1024) != CARQUET_OK)
        { carquet_writer_close(w); carquet_schema_free(s); carquet_test_cleanup(path);
          TEST_FAIL("column_page_size", "set override"); }

    if (carquet_writer_write_batch(w, 0, big_col, ROWS, NULL, NULL) != CARQUET_OK ||
        carquet_writer_write_batch(w, 1, small_col, ROWS, NULL, NULL) != CARQUET_OK ||
        carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); carquet_test_cleanup(path);
          TEST_FAIL("column_page_size", "write"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL("column_page_size", "open"); }
    carquet_offset_index_t* oi_big = carquet_reader_get_offset_index(r, 0, 0, &err);
    carquet_offset_index_t* oi_small = carquet_reader_get_offset_index(r, 0, 1, &err);
    int32_t n_big = oi_big ? carquet_offset_index_num_pages(oi_big) : -1;
    int32_t n_small = oi_small ? carquet_offset_index_num_pages(oi_small) : -1;
    carquet_offset_index_free(oi_big);
    carquet_offset_index_free(oi_small);
    carquet_reader_close(r); carquet_test_cleanup(path);

    /* big_col fits in one page at the default 1 MB page size; small_col with
     * 1 KB pages must produce many. Lower bound 5 is generous — at 1 KB / 4 B
     * stride we expect ~80, but exact count depends on header overhead. */
    if (n_big != 1)
        TEST_FAIL("column_page_size", "default column should produce 1 page");
    if (n_small < 5)
        TEST_FAIL("column_page_size", "overridden column did not split into many pages");
    TEST_PASS("column_page_size");
    return 0;
}

/* ---- Custom codec registration ---- */
/* Trivial "identity" codec: stores values uncompressed but goes through the
 * codec dispatch path so we can prove user callbacks are reached. We register
 * it against the BROTLI slot, which has no built-in — that proves the API can
 * fill an otherwise-unsupported slot. */
static int g_identity_compress_calls;
static int g_identity_decompress_calls;
static int g_identity_user_token = 0xC0DEC;

static carquet_status_t identity_compress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* out_size,
    int32_t level, void* user_data) {
    (void)level;
    if (user_data != &g_identity_user_token) return CARQUET_ERROR_INVALID_ARGUMENT;
    if (src_size > dst_capacity) return CARQUET_ERROR_INVALID_ARGUMENT;
    memcpy(dst, src, src_size);
    *out_size = src_size;
    g_identity_compress_calls++;
    return CARQUET_OK;
}

static carquet_status_t identity_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* out_size,
    void* user_data) {
    if (user_data != &g_identity_user_token) return CARQUET_ERROR_INVALID_ARGUMENT;
    if (src_size > dst_capacity) return CARQUET_ERROR_INVALID_ARGUMENT;
    memcpy(dst, src, src_size);
    *out_size = src_size;
    g_identity_decompress_calls++;
    return CARQUET_OK;
}

static size_t identity_bound(size_t src_size, void* user_data) {
    (void)user_data;
    return src_size;
}

static int test_custom_codec(void) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_custom_codec");
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Validation: NULL function pointer, UNCOMPRESSED slot, out-of-range
     * slot all rejected; NULL impl is the documented unregister path. */
    carquet_custom_codec_t bad = { 0 };
    if (carquet_register_codec(CARQUET_COMPRESSION_BROTLI, &bad) !=
            CARQUET_ERROR_INVALID_ARGUMENT)
        TEST_FAIL("custom_codec", "NULL fn ptrs should be rejected");
    carquet_custom_codec_t ok_codec = {
        .compress = identity_compress,
        .decompress = identity_decompress,
        .compress_bound = identity_bound,
        .user_data = &g_identity_user_token,
    };
    if (carquet_register_codec(CARQUET_COMPRESSION_UNCOMPRESSED, &ok_codec) !=
            CARQUET_ERROR_INVALID_ARGUMENT)
        TEST_FAIL("custom_codec", "UNCOMPRESSED override should be rejected");
    if (carquet_register_codec((carquet_compression_t)99, &ok_codec) !=
            CARQUET_ERROR_INVALID_ARGUMENT)
        TEST_FAIL("custom_codec", "out-of-range slot should be rejected");

    /* Register on the BROTLI slot — no built-in, so this proves the slot is
     * truly reachable through the user callback. */
    if (carquet_register_codec(CARQUET_COMPRESSION_BROTLI, &ok_codec) != CARQUET_OK)
        TEST_FAIL("custom_codec", "register failed");

    g_identity_compress_calls = 0;
    g_identity_decompress_calls = 0;

    enum { M = 1024 };
    static int32_t in[M], out[M];
    for (int i = 0; i < M; i++) in[i] = i * 7 - 100;

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("custom_codec", "schema");
    if (carquet_schema_add_column(s, "v", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("custom_codec", "add col"); }

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.compression = CARQUET_COMPRESSION_BROTLI;
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); carquet_test_cleanup(path);
              TEST_FAIL("custom_codec", "writer create"); }
    if (carquet_writer_write_batch(w, 0, in, M, NULL, NULL) != CARQUET_OK ||
        carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); carquet_test_cleanup(path);
          TEST_FAIL("custom_codec", "write"); }
    carquet_schema_free(s);

    if (g_identity_compress_calls == 0)
        { carquet_test_cleanup(path); TEST_FAIL("custom_codec", "compress not called"); }

    /* Read back through carquet's reader: must hit the registered decompress. */
    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL("custom_codec", "open"); }
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    int64_t n = c ? carquet_column_read_batch(c, out, M, NULL, NULL) : -1;
    int ok = (n == M) && (memcmp(in, out, sizeof(in)) == 0) &&
             (g_identity_decompress_calls > 0);
    carquet_column_reader_free(c); carquet_reader_close(r);
    carquet_test_cleanup(path);

    /* Unregister; subsequent BROTLI writes must now fail with UNSUPPORTED_CODEC,
     * proving NULL really clears the slot. */
    if (carquet_register_codec(CARQUET_COMPRESSION_BROTLI, NULL) != CARQUET_OK)
        TEST_FAIL("custom_codec", "unregister failed");

    char path2[512]; carquet_test_temp_path(path2, sizeof(path2), "ext_custom_codec2");
    carquet_schema_t* s2 = carquet_schema_create(&err);
    carquet_schema_add_column(s2, "v", CARQUET_PHYSICAL_INT32, NULL,
                              CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_writer_options_t wo2; carquet_writer_options_init(&wo2);
    wo2.compression = CARQUET_COMPRESSION_BROTLI;
    carquet_writer_t* w2 = carquet_writer_create(path2, s2, &wo2, &err);
    int unregister_ok = 1;
    if (w2) {
        /* Writer creation may succeed and defer the codec call until first
         * page emit; either the write or the close should now surface the
         * missing-codec error. */
        carquet_status_t wst = carquet_writer_write_batch(w2, 0, in, M, NULL, NULL);
        carquet_status_t cst = carquet_writer_close(w2);
        unregister_ok = (wst == CARQUET_ERROR_UNSUPPORTED_CODEC) ||
                        (cst == CARQUET_ERROR_UNSUPPORTED_CODEC);
    }
    carquet_schema_free(s2);
    carquet_test_cleanup(path2);

    if (!ok) TEST_FAIL("custom_codec", "roundtrip mismatch or decompress not called");
    if (!unregister_ok) TEST_FAIL("custom_codec", "unregister did not clear slot");
    TEST_PASS("custom_codec");
    return 0;
}

/* ---- File format version (FileMetaData.version) ---- */
static int test_file_format_version_one(int requested, int expected) {
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_ver");
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL("file_format_version", "schema create");
    if (carquet_schema_add_column(s, "v", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL("file_format_version", "add col"); }

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.file_format_version = requested;
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("file_format_version", "writer"); }
    int32_t in[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    if (carquet_writer_write_batch(w, 0, in, 8, NULL, NULL) != CARQUET_OK ||
        carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); carquet_test_cleanup(path);
          TEST_FAIL("file_format_version", "write"); }
    carquet_schema_free(s);

    carquet_file_info_t info;
    carquet_status_t st = carquet_get_file_info(path, &info, &err);
    int ok = (st == CARQUET_OK) && (info.version == expected);
    carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("file_format_version", "footer version mismatch");
    return 0;
}

static int test_file_format_version(void) {
    if (test_file_format_version_one(1, 1)) return 1;   /* explicit v1 */
    if (test_file_format_version_one(2, 2)) return 1;   /* explicit v2 */
    if (test_file_format_version_one(0, 2)) return 1;   /* invalid -> v2 */
    if (test_file_format_version_one(99, 2)) return 1;  /* invalid -> v2 */
    TEST_PASS("file_format_version");
    return 0;
}

/* ---- PLAIN BOOLEAN across several write_batch calls ---- */
/* Parquet packs a page's booleans as one continuous bit stream, so a page
 * written in several calls must produce the same bytes as one written in a
 * single call. The chunk sizes below are deliberately not multiples of 8, so
 * every batch after the first starts mid-byte. */
static int test_boolean_chunked_writes(void) {
    const char* name = "boolean_chunked_writes";
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_bool_chunk");
    carquet_error_t err = CARQUET_ERROR_INIT;
    static const int64_t chunks[] = { 5, 3, 991, 1, 4000 };

    static uint8_t in[N], out[N];
    for (int i = 0; i < N; i++) in[i] = (uint8_t)((i * 7 + (i / 3)) & 1);

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL(name, "schema create");
    if (carquet_schema_add_column(s, "b", CARQUET_PHYSICAL_BOOLEAN, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL(name, "add col"); }

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL(name, "writer create"); }
    int64_t done = 0;
    for (size_t k = 0; k < sizeof(chunks) / sizeof(chunks[0]); k++) {
        if (carquet_writer_write_batch(w, 0, in + done, chunks[k],
                                       NULL, NULL) != CARQUET_OK)
            { carquet_writer_close(w); carquet_schema_free(s);
              TEST_FAIL(name, "write batch"); }
        done += chunks[k];
    }
    if (carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL(name, "close"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL(name, "open"); }
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    int64_t n = c ? carquet_column_read_batch(c, out, N, NULL, NULL) : -1;
    int ok = (done == N) && (n == N);
    int64_t wrong = 0;
    for (int64_t i = 0; i < N && ok; i++) if (in[i] != out[i]) wrong++;
    ok = ok && (wrong == 0);
    carquet_column_reader_free(c); carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL(name, "value mismatch");
    TEST_PASS(name);
    return 0;
}

/* ---- PLAIN BOOLEAN: exact on-disk bytes ---- */
/* A round-trip through carquet's own decoder cannot distinguish a correct page
 * from a self-consistently wrong one, so this asserts the literal page payload.
 *
 * Twelve booleans, written as 5 + 7 so the second batch starts at bit 5 --
 * mid-byte, which is the case that was broken:
 *
 *   index  0 1 2 3 4 | 5 6 7 8 9 10 11
 *   value  1 0 1 1 0 | 0 1 0 1 1  0  1
 *
 * PLAIN packs them LSB-first into a continuous stream:
 *
 *   byte 0 = bits 0..7  = 1,0,1,1,0,0,1,0 -> 0x4D  (1 + 4 + 8 + 64)
 *   byte 1 = bits 8..11 = 1,1,0,1         -> 0x0B  (1 + 2 + 8)
 *
 * so the payload is exactly 4D 0B. Restarting the stream at the second call
 * instead emits the first 5 bits, pads to a byte, and starts again, giving
 * 0D 5A -- which is what this pins against. */
static int test_boolean_exact_page_bytes(void) {
    const char* name = "boolean_exact_page_bytes";
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ext_bool_bytes");
    carquet_error_t err = CARQUET_ERROR_INIT;

    static const uint8_t in[12] = { 1,0,1,1,0, 0,1,0,1,1,0,1 };
    static const uint8_t expect[2] = { 0x4D, 0x0B };

    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) TEST_FAIL(name, "schema create");
    if (carquet_schema_add_column(s, "b", CARQUET_PHYSICAL_BOOLEAN, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL(name, "add col"); }

    carquet_writer_options_t wo; carquet_writer_options_init(&wo);
    wo.compression = CARQUET_COMPRESSION_UNCOMPRESSED;  /* payload is the values */
    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL(name, "writer create"); }
    if (carquet_writer_write_batch(w, 0, in, 5, NULL, NULL) != CARQUET_OK ||
        carquet_writer_write_batch(w, 0, in + 5, 7, NULL, NULL) != CARQUET_OK)
        { carquet_writer_close(w); carquet_schema_free(s);
          TEST_FAIL(name, "write batch"); }
    if (carquet_writer_close(w) != CARQUET_OK)
        { carquet_schema_free(s); TEST_FAIL(name, "close"); }
    carquet_schema_free(s);

    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { carquet_test_cleanup(path); TEST_FAIL(name, "open"); }
    carquet_column_chunk_metadata_t cm;
    if (carquet_reader_column_chunk_metadata(r, 0, 0, &cm) != CARQUET_OK)
        { carquet_reader_close(r); carquet_test_cleanup(path);
          TEST_FAIL(name, "chunk meta"); }

    uint8_t* fb = NULL; size_t fsz = 0;
    if (!read_file(path, &fb, &fsz))
        { carquet_reader_close(r); carquet_test_cleanup(path);
          TEST_FAIL(name, "read file"); }

    parquet_page_header_t ph; size_t consumed = 0;
    carquet_status_t pst = parquet_parse_page_header(
        fb + cm.data_page_offset, fsz - (size_t)cm.data_page_offset,
        &ph, &consumed, &err);
    int ok = (pst == CARQUET_OK) &&
             (ph.uncompressed_page_size == (int32_t)sizeof(expect));
    if (ok) {
        const uint8_t* payload = fb + cm.data_page_offset + consumed;
        ok = (memcmp(payload, expect, sizeof(expect)) == 0);
        if (!ok) {
            fprintf(stderr, "  expected: %02X %02X\n  actual:   %02X %02X\n",
                    expect[0], expect[1], payload[0], payload[1]);
        }
    } else if (pst == CARQUET_OK) {
        fprintf(stderr, "  payload is %d bytes, expected %d\n",
                ph.uncompressed_page_size, (int)sizeof(expect));
    }
    free(fb);
    carquet_reader_close(r); carquet_test_cleanup(path);
    if (!ok) TEST_FAIL(name, "page payload is not the expected bit stream");
    TEST_PASS(name);
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_int96_roundtrip();
    failures += test_boolean_chunked_writes();
    failures += test_boolean_exact_page_bytes();
    failures += test_data_page_v2(0);
    failures += test_data_page_v2(1);
    failures += test_arrow_schema_metadata();
    failures += test_field_metadata_roundtrip();
    failures += test_field_metadata_errors();
    failures += test_arrow_schema_skipped_when_off();
    failures += test_arrow_schema_nested();
    failures += test_arrow_type_refinement_read();
    failures += test_float16_stats();
    failures += test_bitpacked_levels();
    failures += test_geospatial_stats();
    failures += test_timestamp_coercion();
    failures += test_write_batch_size();
    failures += test_file_format_version();
    failures += test_custom_codec();
    failures += test_column_page_size_override();
    failures += test_max_statistics_size();
    failures += test_bloom_api_mix();
    failures += test_append_row_groups();
    if (failures) { printf("\n%d test(s) FAILED\n", failures); return 1; }
    printf("\nAll writer-extension tests passed\n");
    return 0;
}
