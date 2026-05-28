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

int main(void) {
    int failures = 0;
    failures += test_int96_roundtrip();
    failures += test_data_page_v2(0);
    failures += test_data_page_v2(1);
    failures += test_arrow_schema_metadata();
    failures += test_arrow_schema_skipped_when_off();
    failures += test_float16_stats();
    failures += test_bitpacked_levels();
    failures += test_geospatial_stats();
    failures += test_timestamp_coercion();
    failures += test_write_batch_size();
    if (failures) { printf("\n%d test(s) FAILED\n", failures); return 1; }
    printf("\nAll writer-extension tests passed\n");
    return 0;
}
