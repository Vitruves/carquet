/**
 * @file test_parallel_write.c
 * @brief Page-granular parallel encode + background I/O in the writer.
 *
 * The row-group finalize schedules one task per page for flat fixed-stride
 * columns (src/writer/page_tasks.h) and hands finished row groups to an I/O
 * thread (carquet_writer_options_t.async_io). Neither may change a single
 * output byte: this test writes the same columns through the serial path
 * (a one-column file never finalizes in parallel) and through the parallel
 * path, then compares the column chunk bytes, and writes the same file with
 * async_io on and off and compares the whole files. It also drives several
 * row groups through the buffer recycling of the I/O thread and reads every
 * value back. Nullable and dictionary columns are covered too: nullable
 * flat columns are cut into pages in parallel, and the pipelined row-group
 * encode finalizes the columns it cannot cut (dictionary, ...) whole on its
 * pool, which must not change a byte either.
 */

#include <carquet/carquet.h>
#include "test_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ROWS 300000   /* > 2 pages of 8-byte values at the 1 MB default */

static uint32_t lcg_state;
static uint32_t lcg(void) {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return (lcg_state >> 8) & 0xFFFFFFu;
}

typedef struct {
    int64_t* ids;
    double* values;
    int32_t* cats;
} data_t;

static void data_init(data_t* d) {
    d->ids = malloc(ROWS * sizeof(int64_t));
    d->values = malloc(ROWS * sizeof(double));
    d->cats = malloc(ROWS * sizeof(int32_t));
    lcg_state = 7;
    for (int i = 0; i < ROWS; i++) {
        d->ids[i] = 1000000 + (int64_t)lcg();
        d->values[i] = (double)lcg() / 1024.0 - 3000.0;
        d->cats[i] = (int32_t)(lcg() % 97);
    }
}

static void data_free(data_t* d) {
    free(d->ids); free(d->values); free(d->cats);
}

typedef struct {
    carquet_compression_t codec;
    int64_t write_batch_size;
    int64_t max_rows_per_page;
    int data_page_version;
    bool async_io;
    int64_t rg_rows;   /* 0 = one row group */
} cfg_t;

static void apply_cfg(carquet_writer_options_t* o, const cfg_t* c) {
    carquet_writer_options_init(o);
    o->compression = c->codec;
    o->compression_level = c->codec == CARQUET_COMPRESSION_ZSTD ? 1 : 0;
    o->write_batch_size = c->write_batch_size;
    o->max_rows_per_page = c->max_rows_per_page;
    o->data_page_version = c->data_page_version;
    o->async_io = c->async_io;
}

/* Write `ncols` of the three columns (in order id, value, cat). */
static int write_file(const char* path, const data_t* d, int ncols, const cfg_t* c) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) return 0;
    static const char* names[3] = {"id", "value", "cat"};
    static const carquet_physical_type_t types[3] = {
        CARQUET_PHYSICAL_INT64, CARQUET_PHYSICAL_DOUBLE, CARQUET_PHYSICAL_INT32};
    for (int i = 0; i < ncols; i++) {
        if (carquet_schema_add_column(s, names[i], types[i], NULL,
                CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK) {
            carquet_schema_free(s); return 0;
        }
    }
    carquet_writer_options_t o; apply_cfg(&o, c);
    carquet_writer_t* w = carquet_writer_create(path, s, &o, &err);
    if (!w) { carquet_schema_free(s); return 0; }
    int ok = 1;
    int64_t step = c->rg_rows > 0 ? c->rg_rows : ROWS;
    for (int64_t off = 0; off < ROWS && ok; off += step) {
        int64_t n = ROWS - off < step ? ROWS - off : step;
        const void* cols[3] = {d->ids + off, d->values + off, d->cats + off};
        for (int i = 0; i < ncols && ok; i++) {
            if (carquet_writer_write_batch(w, i, cols[i], n, NULL, NULL) != CARQUET_OK) ok = 0;
        }
        if (ok && off + n < ROWS && carquet_writer_new_row_group(w) != CARQUET_OK) ok = 0;
    }
    if (carquet_writer_close(w) != CARQUET_OK) ok = 0;
    carquet_schema_free(s);
    return ok;
}

static uint8_t* read_range(const char* path, int64_t off, int64_t len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t* buf = malloc((size_t)len);
    if (!buf || fseek(f, (long)off, SEEK_SET) != 0 ||
        fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf); buf = NULL;
    }
    fclose(f);
    return buf;
}

static long file_size(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}

/* Compare column `col` of the 3-column file against column 0 of a file that
 * holds only that column: same chunk size, same bytes. */
static int chunk_matches(const char* multi, const char* single, int col, const char* tag) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* rm = carquet_reader_open(multi, NULL, &err);
    carquet_reader_t* rs = carquet_reader_open(single, NULL, &err);
    if (!rm || !rs) {
        if (rm) carquet_reader_close(rm);
        if (rs) carquet_reader_close(rs);
        printf("  [%s] open failed\n", tag); return 0;
    }
    int ok = 0;
    carquet_column_chunk_metadata_t mm, ms;
    if (carquet_reader_column_chunk_metadata(rm, 0, col, &mm) == CARQUET_OK &&
        carquet_reader_column_chunk_metadata(rs, 0, 0, &ms) == CARQUET_OK) {
        if (mm.total_compressed_size != ms.total_compressed_size ||
            mm.num_values != ms.num_values || mm.num_values != ROWS) {
            printf("  [%s] col %d size/count differ: %lld/%lld vs %lld/%lld\n", tag, col,
                   (long long)mm.total_compressed_size, (long long)mm.num_values,
                   (long long)ms.total_compressed_size, (long long)ms.num_values);
        } else {
            /* A chunk starts at its dictionary page when it has one (the
             * integer columns may get a dictionary by default). */
            int64_t om = mm.has_dictionary_page ? mm.dictionary_page_offset : mm.data_page_offset;
            int64_t os = ms.has_dictionary_page ? ms.dictionary_page_offset : ms.data_page_offset;
            uint8_t* a = read_range(multi, om, mm.total_compressed_size);
            uint8_t* b = read_range(single, os, ms.total_compressed_size);
            ok = a && b && mm.has_dictionary_page == ms.has_dictionary_page &&
                 memcmp(a, b, (size_t)mm.total_compressed_size) == 0;
            if (!ok) printf("  [%s] col %d chunk bytes differ\n", tag, col);
            free(a); free(b);
        }
    }
    carquet_reader_close(rm);
    carquet_reader_close(rs);
    return ok;
}

static int values_match(const char* path, const data_t* d, int ncols, const char* tag) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_options_t ro; carquet_reader_options_init(&ro);
    ro.use_mmap = true;
    carquet_reader_t* r = carquet_reader_open(path, &ro, &err);
    if (!r) { printf("  [%s] open failed\n", tag); return 0; }
    carquet_batch_reader_config_t c; carquet_batch_reader_config_init(&c);
    c.batch_size = 65536;
    carquet_batch_reader_t* br = carquet_batch_reader_create(r, &c, &err);
    int ok = br != NULL;
    int64_t row = 0;
    carquet_row_batch_t* b = NULL;
    while (ok && carquet_batch_reader_next(br, &b) == CARQUET_OK && b) {
        int64_t n = carquet_row_batch_num_rows(b);
        const void* p; const uint8_t* nulls; int64_t cnt;
        if (ncols > 0 && carquet_row_batch_column(b, 0, &p, &nulls, &cnt) == CARQUET_OK) {
            if (cnt != n || memcmp(p, d->ids + row, (size_t)n * 8) != 0) ok = 0;
        }
        if (ncols > 1 && carquet_row_batch_column(b, 1, &p, &nulls, &cnt) == CARQUET_OK) {
            if (cnt != n || memcmp(p, d->values + row, (size_t)n * 8) != 0) ok = 0;
        }
        if (ncols > 2 && carquet_row_batch_column(b, 2, &p, &nulls, &cnt) == CARQUET_OK) {
            if (cnt != n || memcmp(p, d->cats + row, (size_t)n * 4) != 0) ok = 0;
        }
        row += n;
        carquet_row_batch_free(b); b = NULL;
    }
    if (row != ROWS) ok = 0;
    if (!ok) printf("  [%s] value mismatch (rows=%lld)\n", tag, (long long)row);
    if (br) carquet_batch_reader_free(br);
    carquet_reader_close(r);
    return ok;
}

static int test_page_parallel_identical(const cfg_t* c, const char* tag) {
    data_t d; data_init(&d);
    char multi[512], single[3][512];
    carquet_test_temp_path(multi, sizeof(multi), "ppw_multi");
    carquet_test_temp_path(single[0], sizeof(single[0]), "ppw_s0");
    carquet_test_temp_path(single[1], sizeof(single[1]), "ppw_s1");
    carquet_test_temp_path(single[2], sizeof(single[2]), "ppw_s2");

    int ok = write_file(multi, &d, 3, c);
    /* Column i alone: the one-column row group finalizes serially. */
    for (int i = 0; ok && i < 3; i++) {
        /* Build the one-column file by hand to keep the physical type. */
        carquet_error_t err = CARQUET_ERROR_INIT;
        carquet_schema_t* s = carquet_schema_create(&err);
        static const carquet_physical_type_t types[3] = {
            CARQUET_PHYSICAL_INT64, CARQUET_PHYSICAL_DOUBLE, CARQUET_PHYSICAL_INT32};
        ok = s && carquet_schema_add_column(s, "c", types[i], NULL,
                CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK;
        carquet_writer_options_t o; apply_cfg(&o, c);
        carquet_writer_t* w = ok ? carquet_writer_create(single[i], s, &o, &err) : NULL;
        ok = w != NULL;
        const void* src = i == 0 ? (const void*)d.ids : i == 1 ? (const void*)d.values : (const void*)d.cats;
        if (ok && carquet_writer_write_batch(w, 0, src, ROWS, NULL, NULL) != CARQUET_OK) ok = 0;
        if (w && carquet_writer_close(w) != CARQUET_OK) ok = 0;
        if (s) carquet_schema_free(s);
    }
    for (int i = 0; ok && i < 3; i++) ok = chunk_matches(multi, single[i], i, tag);
    if (ok) ok = values_match(multi, &d, 3, tag);

    carquet_test_cleanup(multi);
    for (int i = 0; i < 3; i++) carquet_test_cleanup(single[i]);
    data_free(&d);
    if (!ok) TEST_FAIL(tag, "parallel page output differs from serial output");
    TEST_PASS(tag);
    return 0;
}

static int test_async_io_identical(carquet_compression_t codec, int64_t rg_rows,
                                   const char* tag) {
    data_t d; data_init(&d);
    char a[512], b[512];
    carquet_test_temp_path(a, sizeof(a), "aio_on");
    carquet_test_temp_path(b, sizeof(b), "aio_off");
    cfg_t on = {codec, 0, 0, 1, true, rg_rows};
    cfg_t off = on; off.async_io = false;
    int ok = write_file(a, &d, 3, &on) && write_file(b, &d, 3, &off);
    if (ok) {
        long la = file_size(a), lb = file_size(b);
        ok = la > 0 && la == lb;
        if (ok) {
            uint8_t* pa = read_range(a, 0, la);
            uint8_t* pb = read_range(b, 0, lb);
            ok = pa && pb && memcmp(pa, pb, (size_t)la) == 0;
            free(pa); free(pb);
        }
        if (!ok) printf("  [%s] files differ (%ld vs %ld bytes)\n", tag, la, lb);
    }
    if (ok) ok = values_match(a, &d, 3, tag);
    carquet_test_cleanup(a); carquet_test_cleanup(b);
    data_free(&d);
    if (!ok) TEST_FAIL(tag, "async_io changed the output");
    TEST_PASS(tag);
    return 0;
}

/* Several row groups through the buffer writer: the I/O thread appends to
 * the writer's memory sink while later groups are encoded, and get_buffer
 * must see the complete file. */
static int test_async_buffer_writer(void) {
    data_t d; data_init(&d);
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    int ok = s &&
        carquet_schema_add_column(s, "id", CARQUET_PHYSICAL_INT64, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK &&
        carquet_schema_add_column(s, "value", CARQUET_PHYSICAL_DOUBLE, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK;
    cfg_t c = {CARQUET_COMPRESSION_SNAPPY, 0, 0, 1, true, 40000};
    carquet_writer_options_t o; apply_cfg(&o, &c);
    carquet_writer_t* w = ok ? carquet_writer_create_buffer(s, &o, &err) : NULL;
    ok = w != NULL;
    for (int64_t off = 0; ok && off < ROWS; off += c.rg_rows) {
        int64_t n = ROWS - off < c.rg_rows ? ROWS - off : c.rg_rows;
        ok = carquet_writer_write_batch(w, 0, d.ids + off, n, NULL, NULL) == CARQUET_OK &&
             carquet_writer_write_batch(w, 1, d.values + off, n, NULL, NULL) == CARQUET_OK;
        if (ok && off + n < ROWS) ok = carquet_writer_new_row_group(w) == CARQUET_OK;
    }
    if (ok) ok = carquet_writer_close(w) == CARQUET_OK;
    void* buf = NULL; size_t size = 0;
    if (ok) ok = carquet_writer_get_buffer(w, &buf, &size) == CARQUET_OK && buf && size > 0;
    else if (w) carquet_writer_abort(w);
    if (s) carquet_schema_free(s);

    if (ok) {
        char path[512]; carquet_test_temp_path(path, sizeof(path), "aio_buf");
        FILE* f = fopen(path, "wb");
        ok = f && fwrite(buf, 1, size, f) == size;
        if (f) fclose(f);
        if (ok) ok = values_match(path, &d, 2, "async_buffer_writer");
        carquet_test_cleanup(path);
    }
    free(buf);
    data_free(&d);
    if (!ok) TEST_FAIL("async_buffer_writer", "multi-row-group buffer write failed");
    TEST_PASS("async_buffer_writer");
    return 0;
}

/* The first `rows` rows of the three columns, to `path` or, when it is NULL,
 * through the buffer writer into *buf / *size (carquet_free). */
static int write_sink(const char* path, const data_t* d, int64_t rows, const cfg_t* c,
                      bool bloom, bool page_index, void** buf, size_t* size) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) return 0;
    static const char* names[3] = {"id", "value", "cat"};
    static const carquet_physical_type_t types[3] = {
        CARQUET_PHYSICAL_INT64, CARQUET_PHYSICAL_DOUBLE, CARQUET_PHYSICAL_INT32};
    for (int i = 0; i < 3; i++) {
        if (carquet_schema_add_column(s, names[i], types[i], NULL,
                CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK) {
            carquet_schema_free(s); return 0;
        }
    }
    carquet_writer_options_t o; apply_cfg(&o, c);
    o.write_bloom_filters = bloom;
    o.write_page_index = page_index;
    carquet_writer_t* w = path ? carquet_writer_create(path, s, &o, &err)
                               : carquet_writer_create_buffer(s, &o, &err);
    carquet_schema_free(s);
    if (!w) return 0;
    int ok = 1;
    int64_t step = c->rg_rows > 0 ? c->rg_rows : ROWS;
    for (int64_t off = 0; off < rows && ok; off += step) {
        int64_t n = rows - off < step ? rows - off : step;
        const void* cols[3] = {d->ids + off, d->values + off, d->cats + off};
        for (int i = 0; i < 3 && ok; i++) {
            if (carquet_writer_write_batch(w, i, cols[i], n, NULL, NULL) != CARQUET_OK) ok = 0;
        }
        if (ok && off + n < rows && carquet_writer_new_row_group(w) != CARQUET_OK) ok = 0;
    }
    if (!ok) { carquet_writer_abort(w); return 0; }
    if (carquet_writer_close(w) != CARQUET_OK) {
        /* A buffer writer outlives a failed close. */
        if (!path) carquet_writer_abort(w);
        return 0;
    }
    if (!path && carquet_writer_get_buffer(w, buf, size) != CARQUET_OK) {
        carquet_writer_abort(w);
        return 0;
    }
    return 1;
}

/* The buffer writer assembles the file in memory instead of going through a
 * stream: for the same input it must produce the bytes the file writer does,
 * row groups, bloom filters, page indexes and footer included. */
static int test_buffer_matches_file(const cfg_t* c, bool bloom, bool page_index,
                                    int64_t rows, const char* tag) {
    data_t d; data_init(&d);
    char path[512];
    carquet_test_temp_path(path, sizeof(path), "buf_vs_file");
    void* buf = NULL; size_t size = 0;
    int ok = write_sink(path, &d, rows, c, bloom, page_index, NULL, NULL) &&
             write_sink(NULL, &d, rows, c, bloom, page_index, &buf, &size);
    if (ok) {
        long fl = file_size(path);
        ok = fl > 0 && (size_t)fl == size && buf != NULL;
        if (ok) {
            uint8_t* pf = read_range(path, 0, fl);
            ok = pf && memcmp(pf, buf, size) == 0;
            free(pf);
        }
        if (!ok) printf("  [%s] buffer differs from file (%llu vs %ld bytes)\n", tag,
                        (unsigned long long)size, fl);
    }
    if (ok && size >= 8) {
        ok = memcmp(buf, "PAR1", 4) == 0 &&
             memcmp((const uint8_t*)buf + size - 4, "PAR1", 4) == 0;
    }
    if (ok && rows == ROWS) ok = values_match(path, &d, 3, tag);
    if (ok) {
        /* The buffer is a readable file of `rows` rows on its own. */
        carquet_error_t err = CARQUET_ERROR_INIT;
        carquet_reader_t* r = carquet_reader_open_buffer(buf, size, NULL, &err);
        ok = r != NULL && carquet_reader_num_rows(r) == rows;
        if (r) carquet_reader_close(r);
    }
    carquet_free(buf);
    carquet_test_cleanup(path);
    data_free(&d);
    if (!ok) TEST_FAIL(tag, "buffer writer output differs from the file writer's");
    TEST_PASS(tag);
    return 0;
}

/* Buffer writers that never hand their buffer out: aborted with row groups
 * written and pending, and closed then aborted (close -> abort is the way to
 * drop the result). Nothing may leak or be freed twice (sanitizer builds). */
static int test_buffer_writer_abandoned(void) {
    data_t d; data_init(&d);
    int ok = 1;
    for (int async = 0; async < 2 && ok; async++) {
        for (int closed = 0; closed < 2 && ok; closed++) {
            carquet_error_t err = CARQUET_ERROR_INIT;
            carquet_schema_t* s = carquet_schema_create(&err);
            ok = s && carquet_schema_add_column(s, "id", CARQUET_PHYSICAL_INT64, NULL,
                          CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK;
            cfg_t c = {CARQUET_COMPRESSION_SNAPPY, 0, 0, 1, async != 0, 0};
            carquet_writer_options_t o; apply_cfg(&o, &c);
            carquet_writer_t* w = ok ? carquet_writer_create_buffer(s, &o, &err) : NULL;
            if (s) carquet_schema_free(s);
            if (!w) { ok = 0; break; }
            for (int g = 0; ok && g < 3; g++) {
                ok = carquet_writer_write_batch(w, 0, d.ids + g * 50000, 50000,
                                                NULL, NULL) == CARQUET_OK;
                if (ok && g < 2) ok = carquet_writer_new_row_group(w) == CARQUET_OK;
            }
            if (ok && closed) ok = carquet_writer_close(w) == CARQUET_OK;
            carquet_writer_abort(w);
        }
    }
    data_free(&d);
    if (!ok) TEST_FAIL("buffer_writer_abandoned", "buffer writer failed");
    TEST_PASS("buffer_writer_abandoned");
    return 0;
}

/* A nullable column next to REQUIRED ones, all page-parallel; every value and
 * level must land correctly in the same file. */
static int test_mixed_nullable(void) {
    data_t d; data_init(&d);
    carquet_error_t err = CARQUET_ERROR_INIT;
    char path[512]; carquet_test_temp_path(path, sizeof(path), "ppw_mixed");
    carquet_schema_t* s = carquet_schema_create(&err);
    int ok = s &&
        carquet_schema_add_column(s, "id", CARQUET_PHYSICAL_INT64, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK &&
        carquet_schema_add_column(s, "opt", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_OPTIONAL, 0, 0) == CARQUET_OK &&
        carquet_schema_add_column(s, "value", CARQUET_PHYSICAL_DOUBLE, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK;
    int16_t* defs = malloc(ROWS * sizeof(int16_t));
    int32_t* packed = malloc(ROWS * sizeof(int32_t));
    int64_t np = 0;
    for (int i = 0; i < ROWS; i++) {
        defs[i] = (i % 5 == 0) ? 0 : 1;
        if (defs[i]) packed[np++] = d.cats[i];
    }
    cfg_t c = {CARQUET_COMPRESSION_LZ4_RAW, 0, 0, 1, true, 0};
    carquet_writer_options_t o; apply_cfg(&o, &c);
    carquet_writer_t* w = ok ? carquet_writer_create(path, s, &o, &err) : NULL;
    ok = w != NULL;
    if (ok) ok = carquet_writer_write_batch(w, 0, d.ids, ROWS, NULL, NULL) == CARQUET_OK &&
                 carquet_writer_write_batch(w, 1, packed, ROWS, defs, NULL) == CARQUET_OK &&
                 carquet_writer_write_batch(w, 2, d.values, ROWS, NULL, NULL) == CARQUET_OK;
    if (w && carquet_writer_close(w) != CARQUET_OK) ok = 0;
    if (s) carquet_schema_free(s);

    if (ok) {
        carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
        ok = r != NULL;
        if (ok) {
            int64_t* ids = malloc(ROWS * sizeof(int64_t));
            double* vals = malloc(ROWS * sizeof(double));
            int32_t* got = malloc(ROWS * sizeof(int32_t));
            int16_t* gdef = malloc(ROWS * sizeof(int16_t));
            carquet_column_reader_t* c0 = carquet_reader_get_column(r, 0, 0, &err);
            carquet_column_reader_t* c1 = carquet_reader_get_column(r, 0, 1, &err);
            carquet_column_reader_t* c2 = carquet_reader_get_column(r, 0, 2, &err);
            ok = c0 && c1 && c2 &&
                 carquet_column_read_batch(c0, ids, ROWS, NULL, NULL) == ROWS &&
                 carquet_column_read_batch(c1, got, ROWS, gdef, NULL) == ROWS &&
                 carquet_column_read_batch(c2, vals, ROWS, NULL, NULL) == ROWS &&
                 memcmp(ids, d.ids, ROWS * 8) == 0 &&
                 memcmp(vals, d.values, ROWS * 8) == 0 &&
                 memcmp(gdef, defs, ROWS * sizeof(int16_t)) == 0 &&
                 memcmp(got, packed, (size_t)np * sizeof(int32_t)) == 0;
            if (c0) carquet_column_reader_free(c0);
            if (c1) carquet_column_reader_free(c1);
            if (c2) carquet_column_reader_free(c2);
            free(ids); free(vals); free(got); free(gdef);
            carquet_reader_close(r);
        }
    }
    carquet_test_cleanup(path);
    free(defs); free(packed);
    data_free(&d);
    if (!ok) TEST_FAIL("mixed_nullable", "roundtrip failed");
    TEST_PASS("mixed_nullable");
    return 0;
}

/* A background write failure must surface from close() rather than vanish.
 * Only runnable where a stream that rejects writes exists (/dev/full). */
static int test_async_write_error(void) {
#ifdef _WIN32
    /* Windows has no always-full stream. Do not even try the fopen: under
     * Wine "/dev/full" resolves through the Z: drive to the real Unix device
     * and the UCRT terminates the process inside fopen() with exit status 0,
     * which would silently cut the run short and hide every later test. */
    TEST_PASS("async_write_error (skipped: no /dev/full on Windows)");
    return 0;
#else
    FILE* full = fopen("/dev/full", "wb");
    if (!full) { TEST_PASS("async_write_error (skipped: no /dev/full)"); return 0; }
    data_t d; data_init(&d);
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    int ok = s &&
        carquet_schema_add_column(s, "id", CARQUET_PHYSICAL_INT64, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK &&
        carquet_schema_add_column(s, "value", CARQUET_PHYSICAL_DOUBLE, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK;
    carquet_writer_options_t o; carquet_writer_options_init(&o);
    carquet_writer_t* w = ok ? carquet_writer_create_file(full, s, &o, &err) : NULL;
    ok = w != NULL;
    carquet_status_t st = CARQUET_OK;
    if (ok) {
        (void)carquet_writer_write_batch(w, 0, d.ids, ROWS, NULL, NULL);
        (void)carquet_writer_write_batch(w, 1, d.values, ROWS, NULL, NULL);
        (void)carquet_writer_new_row_group(w);
        (void)carquet_writer_write_batch(w, 0, d.ids, 1000, NULL, NULL);
        (void)carquet_writer_write_batch(w, 1, d.values, 1000, NULL, NULL);
        st = carquet_writer_close(w);
    }
    if (s) carquet_schema_free(s);
    fclose(full);
    data_free(&d);
    if (!ok) TEST_FAIL("async_write_error", "setup failed");
    if (st == CARQUET_OK) TEST_FAIL("async_write_error", "close() hid the write failure");
    TEST_PASS("async_write_error");
    return 0;
#endif
}


/* ----------------------------------------------------------------------------
 * Column kinds: nullable and dictionary columns next to plain ones.
 * ------------------------------------------------------------------------- */

typedef enum {
    K_REQ,        /* REQUIRED, PLAIN */
    K_OPT,        /* OPTIONAL, ~10% random nulls */
    K_OPT_RUNS,   /* OPTIONAL, nulls in long runs (RLE-friendly levels) */
    K_OPT_NONE,   /* OPTIONAL, no null at all */
    K_ALL_NULL,   /* OPTIONAL, every row null */
    K_DICT,       /* REQUIRED, RLE_DICTIONARY, low cardinality */
    K_DICT_OPT,   /* OPTIONAL, RLE_DICTIONARY, low cardinality, ~10% nulls */
    K_BSS_OPT     /* OPTIONAL, BYTE_STREAM_SPLIT, ~10% nulls */
} kind_t;

static const carquet_physical_type_t kind_types[3] = {
    CARQUET_PHYSICAL_INT64, CARQUET_PHYSICAL_DOUBLE, CARQUET_PHYSICAL_INT32};

typedef struct {
    kind_t kind[3];
    void* full[3];       /* ROWS logical values (row i's value, nulls included) */
    int16_t* defs[3];    /* ROWS levels, NULL for REQUIRED */
} kdata_t;

static bool kind_optional(kind_t k) {
    return k == K_OPT || k == K_OPT_RUNS || k == K_OPT_NONE || k == K_ALL_NULL ||
           k == K_DICT_OPT || k == K_BSS_OPT;
}
static bool kind_dict(kind_t k) { return k == K_DICT || k == K_DICT_OPT; }

static void kdata_init(kdata_t* d, const kind_t kinds[3]) {
    lcg_state = 11;
    for (int c = 0; c < 3; c++) {
        d->kind[c] = kinds[c];
        size_t w = c == 2 ? 4 : 8;
        d->full[c] = malloc(ROWS * w);
        d->defs[c] = kind_optional(kinds[c]) ? malloc(ROWS * sizeof(int16_t)) : NULL;
        for (int i = 0; i < ROWS; i++) {
            uint32_t r = lcg();
            bool dict = kind_dict(kinds[c]);
            if (c == 0) ((int64_t*)d->full[c])[i] = dict ? (int64_t)(r % 300) * 1000003 : 1000000 + (int64_t)r;
            if (c == 1) ((double*)d->full[c])[i] = dict ? (double)(r % 50) * 0.25 : (double)r / 1024.0 - 3000.0;
            if (c == 2) ((int32_t*)d->full[c])[i] = dict ? (int32_t)(r % 97) - 40 : (int32_t)r;
            if (d->defs[c]) {
                int16_t def = 1;
                switch (kinds[c]) {
                    case K_OPT: case K_DICT_OPT: case K_BSS_OPT: def = (lcg() % 10) != 0; break;
                    case K_OPT_RUNS: def = ((i / 5000) % 3) != 1; break;
                    case K_ALL_NULL: def = 0; break;
                    default: break;
                }
                d->defs[c][i] = def;
            }
        }
    }
}

static void kdata_free(kdata_t* d) {
    for (int c = 0; c < 3; c++) { free(d->full[c]); free(d->defs[c]); }
}

/* Pack rows [off, off + n) of column c: the non-null values, in order. */
static int64_t kdata_pack(const kdata_t* d, int c, int64_t off, int64_t n, void* out) {
    size_t w = c == 2 ? 4 : 8;
    int64_t k = 0;
    for (int64_t i = off; i < off + n; i++) {
        if (d->defs[c] && !d->defs[c][i]) continue;
        memcpy((uint8_t*)out + (size_t)k * w, (const uint8_t*)d->full[c] + (size_t)i * w, w);
        k++;
    }
    return k;
}

/* Write the columns `cols` (ncols of them, indices into d) as one file. */
static int write_kinds(const char* path, const kdata_t* d, const int* cols, int ncols,
                       const cfg_t* c) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) return 0;
    for (int j = 0; j < ncols; j++) {
        char name[8]; snprintf(name, sizeof(name), "c%d", cols[j]);
        if (carquet_schema_add_column(s, name, kind_types[cols[j]], NULL,
                kind_optional(d->kind[cols[j]]) ? CARQUET_REPETITION_OPTIONAL
                                                : CARQUET_REPETITION_REQUIRED,
                0, 0) != CARQUET_OK) {
            carquet_schema_free(s); return 0;
        }
    }
    carquet_writer_options_t o; apply_cfg(&o, c);
    carquet_writer_t* w = carquet_writer_create(path, s, &o, &err);
    if (!w) { carquet_schema_free(s); return 0; }
    int ok = 1;
    for (int j = 0; j < ncols && ok; j++) {
        kind_t k = d->kind[cols[j]];
        if (kind_dict(k)) {
            ok = carquet_writer_set_column_encoding(w, j, CARQUET_ENCODING_RLE_DICTIONARY) == CARQUET_OK;
        } else if (k == K_BSS_OPT) {
            ok = carquet_writer_set_column_encoding(w, j, CARQUET_ENCODING_BYTE_STREAM_SPLIT) == CARQUET_OK;
        }
    }
    void* packed = malloc(ROWS * 8);
    int64_t step = c->rg_rows > 0 ? c->rg_rows : ROWS;
    for (int64_t off = 0; off < ROWS && ok; off += step) {
        int64_t n = ROWS - off < step ? ROWS - off : step;
        for (int j = 0; j < ncols && ok; j++) {
            int col = cols[j];
            (void)kdata_pack(d, col, off, n, packed);
            if (carquet_writer_write_batch(w, j, packed, n,
                    d->defs[col] ? d->defs[col] + off : NULL, NULL) != CARQUET_OK) ok = 0;
        }
        if (ok && off + n < ROWS && carquet_writer_new_row_group(w) != CARQUET_OK) ok = 0;
    }
    free(packed);
    if (carquet_writer_close(w) != CARQUET_OK) ok = 0;
    carquet_schema_free(s);
    return ok;
}

/* Chunk (rg, col) of `multi` against chunk (rg, 0) of `single`: same size,
 * same bytes from the first page (dictionary page included) on. */
static int kind_chunk_matches(const char* multi, const char* single, int col, const char* tag) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* rm = carquet_reader_open(multi, NULL, &err);
    carquet_reader_t* rs = carquet_reader_open(single, NULL, &err);
    int ok = rm && rs &&
        carquet_reader_num_row_groups(rm) == carquet_reader_num_row_groups(rs);
    for (int32_t rg = 0; ok && rg < carquet_reader_num_row_groups(rm); rg++) {
        carquet_column_chunk_metadata_t mm, ms;
        ok = carquet_reader_column_chunk_metadata(rm, rg, col, &mm) == CARQUET_OK &&
             carquet_reader_column_chunk_metadata(rs, rg, 0, &ms) == CARQUET_OK;
        if (!ok) break;
        if (mm.total_compressed_size != ms.total_compressed_size ||
            mm.num_values != ms.num_values ||
            mm.has_dictionary_page != ms.has_dictionary_page) {
            printf("  [%s] rg %d col %d size/count differ: %lld/%lld vs %lld/%lld\n",
                   tag, rg, col,
                   (long long)mm.total_compressed_size, (long long)mm.num_values,
                   (long long)ms.total_compressed_size, (long long)ms.num_values);
            ok = 0; break;
        }
        int64_t om = mm.has_dictionary_page ? mm.dictionary_page_offset : mm.data_page_offset;
        int64_t os = ms.has_dictionary_page ? ms.dictionary_page_offset : ms.data_page_offset;
        uint8_t* a = read_range(multi, om, mm.total_compressed_size);
        uint8_t* b = read_range(single, os, ms.total_compressed_size);
        ok = a && b && memcmp(a, b, (size_t)mm.total_compressed_size) == 0;
        if (!ok) printf("  [%s] rg %d col %d chunk bytes differ\n", tag, rg, col);
        free(a); free(b);
    }
    if (!ok && !(rm && rs)) printf("  [%s] open failed\n", tag);
    if (rm) carquet_reader_close(rm);
    if (rs) carquet_reader_close(rs);
    return ok;
}

/* Every value and level of the 3-column file, row group by row group. */
static int kind_values_match(const char* path, const kdata_t* d, const char* tag) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { printf("  [%s] open failed\n", tag); return 0; }
    int ok = 1;
    int64_t row = 0;
    void* want = malloc(ROWS * 8);
    void* got = malloc(ROWS * 8);
    int16_t* gdef = malloc(ROWS * sizeof(int16_t));
    for (int32_t rg = 0; ok && rg < carquet_reader_num_row_groups(r); rg++) {
        int64_t n = 0;
        for (int c = 0; ok && c < 3; c++) {
            carquet_column_chunk_metadata_t m;
            ok = carquet_reader_column_chunk_metadata(r, rg, c, &m) == CARQUET_OK;
            if (!ok) break;
            n = m.num_values;
            carquet_column_reader_t* cr = carquet_reader_get_column(r, rg, c, &err);
            ok = cr && carquet_column_read_batch(cr, got, n, d->defs[c] ? gdef : NULL, NULL) == n;
            if (ok && d->defs[c]) ok = memcmp(gdef, d->defs[c] + row, (size_t)n * sizeof(int16_t)) == 0;
            int64_t np = kdata_pack(d, c, row, n, want);
            if (ok) ok = memcmp(got, want, (size_t)np * (c == 2 ? 4 : 8)) == 0;
            if (cr) carquet_column_reader_free(cr);
            if (!ok) printf("  [%s] rg %d col %d value mismatch\n", tag, rg, c);
        }
        row += n;
    }
    if (ok && row != ROWS) { ok = 0; printf("  [%s] read %lld rows\n", tag, (long long)row); }
    free(want); free(got); free(gdef);
    carquet_reader_close(r);
    return ok;
}

/* Serial (one-column files) against parallel / pipelined (three columns). */
static int test_kinds_identical(kind_t k0, kind_t k1, kind_t k2, const cfg_t* c,
                                const char* tag) {
    kind_t kinds[3] = {k0, k1, k2};
    kdata_t d; kdata_init(&d, kinds);
    char multi[512], single[512];
    carquet_test_temp_path(multi, sizeof(multi), "kind_multi");
    carquet_test_temp_path(single, sizeof(single), "kind_single");
    static const int all[3] = {0, 1, 2};
    int ok = write_kinds(multi, &d, all, 3, c);
    if (!ok) printf("  [%s] multi-column write failed\n", tag);
    for (int i = 0; ok && i < 3; i++) {
        ok = write_kinds(single, &d, &i, 1, c);
        if (!ok) printf("  [%s] single-column write %d failed\n", tag, i);
        if (ok) ok = kind_chunk_matches(multi, single, i, tag);
    }
    if (ok) ok = kind_values_match(multi, &d, tag);
    carquet_test_cleanup(multi);
    carquet_test_cleanup(single);
    kdata_free(&d);
    if (!ok) TEST_FAIL(tag, "parallel output of nullable / dictionary columns differs from serial");
    TEST_PASS(tag);
    return 0;
}

/* Whole files with async_io (pipelined encode) on and off. */
static int test_kinds_async_identical(kind_t k0, kind_t k1, kind_t k2, const cfg_t* c,
                                      const char* tag) {
    kind_t kinds[3] = {k0, k1, k2};
    kdata_t d; kdata_init(&d, kinds);
    char a[512], b[512];
    carquet_test_temp_path(a, sizeof(a), "kaio_on");
    carquet_test_temp_path(b, sizeof(b), "kaio_off");
    static const int all[3] = {0, 1, 2};
    cfg_t on = *c; on.async_io = true;
    cfg_t off = *c; off.async_io = false;
    int ok = write_kinds(a, &d, all, 3, &on) && write_kinds(b, &d, all, 3, &off);
    if (ok) {
        long la = file_size(a), lb = file_size(b);
        ok = la > 0 && la == lb;
        if (ok) {
            uint8_t* pa = read_range(a, 0, la);
            uint8_t* pb = read_range(b, 0, lb);
            ok = pa && pb && memcmp(pa, pb, (size_t)la) == 0;
            free(pa); free(pb);
        }
        if (!ok) printf("  [%s] files differ (%ld vs %ld bytes)\n", tag, la, lb);
    }
    if (ok) ok = kind_values_match(a, &d, tag);
    carquet_test_cleanup(a); carquet_test_cleanup(b);
    kdata_free(&d);
    if (!ok) TEST_FAIL(tag, "async_io changed the output of nullable / dictionary columns");
    TEST_PASS(tag);
    return 0;
}

/* RLE_DICTIONARY chunks are cut into data pages by size (and row cap) like
 * PLAIN ones, each page carrying the statistics of its own values: checked
 * through the page index, for an INT32 column with nulls and a DOUBLE column
 * with NaN and signed zeros. */
static int test_dict_pages(int64_t page_size, int64_t max_rows, const char* tag) {
    enum { N = 300000 };
    char path[512]; carquet_test_temp_path(path, sizeof(path), "dict_pages");
    int32_t* iv = malloc(N * sizeof(int32_t));
    double* dv = malloc(N * sizeof(double));
    int16_t* idef = malloc(N * sizeof(int16_t));
    int32_t* ipacked = malloc(N * sizeof(int32_t));
    static const double dset[8] = {1.5, -2.5, 0.0, -0.0, 7.25, -100.0, 3.0, 0.0};
    int64_t np = 0;
    lcg_state = 5;
    for (int i = 0; i < N; i++) {
        /* Value ranges drift across the chunk so page bounds differ. */
        iv[i] = (int32_t)(lcg() % 200) - 100 + (i / 50000) * 7;
        idef[i] = (lcg() % 7) != 0;
        if (idef[i]) ipacked[np++] = iv[i];
        uint32_t r = lcg() % 9;
        dv[i] = r == 8 ? NAN : dset[r] + (double)(i / 60000);
        /* A NaN run shorter than any page: an all-NaN page would withhold
         * the whole ColumnIndex. */
        if (i >= 120000 && i < 130000) dv[i] = NAN;
    }
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    int ok = s &&
        carquet_schema_add_column(s, "i", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_OPTIONAL, 0, 0) == CARQUET_OK &&
        carquet_schema_add_column(s, "d", CARQUET_PHYSICAL_DOUBLE, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK;
    carquet_writer_options_t o; carquet_writer_options_init(&o);
    o.compression = CARQUET_COMPRESSION_SNAPPY;
    o.write_page_index = true;
    o.page_size = page_size;
    o.max_rows_per_page = max_rows;
    carquet_writer_t* w = ok ? carquet_writer_create(path, s, &o, &err) : NULL;
    ok = w != NULL;
    if (ok) ok = carquet_writer_set_column_encoding(w, 0, CARQUET_ENCODING_RLE_DICTIONARY) == CARQUET_OK &&
                 carquet_writer_set_column_encoding(w, 1, CARQUET_ENCODING_RLE_DICTIONARY) == CARQUET_OK &&
                 carquet_writer_write_batch(w, 0, ipacked, N, idef, NULL) == CARQUET_OK &&
                 carquet_writer_write_batch(w, 1, dv, N, NULL, NULL) == CARQUET_OK;
    if (w && carquet_writer_close(w) != CARQUET_OK) ok = 0;
    if (s) carquet_schema_free(s);

    carquet_reader_t* r = ok ? carquet_reader_open(path, NULL, &err) : NULL;
    ok = r != NULL;
    for (int c = 0; ok && c < 2; c++) {
        carquet_column_chunk_metadata_t m;
        ok = carquet_reader_column_chunk_metadata(r, 0, c, &m) == CARQUET_OK &&
             m.has_dictionary_page;
        carquet_offset_index_t* oi = ok ? carquet_reader_get_offset_index(r, 0, c, &err) : NULL;
        carquet_column_index_t* ci = ok ? carquet_reader_get_column_index(r, 0, c, &err) : NULL;
        ok = oi && ci;
        if (!ok) printf("  [%s] col %d: no page index\n", tag, c);
        int32_t pages = ok ? carquet_offset_index_num_pages(oi) : 0;
        if (ok && (pages < 2 || carquet_column_index_num_pages(ci) != pages)) {
            printf("  [%s] col %d: %d data pages\n", tag, c, pages);
            ok = 0;
        }
        for (int32_t p = 0; ok && p < pages; p++) {
            carquet_page_location_t loc, next;
            carquet_page_stats_t st;
            ok = carquet_offset_index_get_page_location(oi, p, &loc) == CARQUET_OK &&
                 carquet_column_index_get_page_stats(ci, p, &st) == CARQUET_OK;
            int64_t end = N;
            if (ok && p + 1 < pages) {
                ok = carquet_offset_index_get_page_location(oi, p + 1, &next) == CARQUET_OK;
                end = next.first_row_index;
            }
            if (!ok) break;
            if (max_rows > 0 && end - loc.first_row_index > max_rows) {
                printf("  [%s] col %d page %d: %lld rows\n", tag, c, p,
                       (long long)(end - loc.first_row_index));
                ok = 0; break;
            }
            /* Expected bounds of the page's present, non-NaN values. */
            int64_t nulls = 0; bool any = false;
            int32_t imn = 0, imx = 0; double dmn = 0, dmx = 0;
            for (int64_t i = loc.first_row_index; i < end; i++) {
                if (c == 0) {
                    if (!idef[i]) { nulls++; continue; }
                    if (!any || iv[i] < imn) imn = iv[i];
                    if (!any || iv[i] > imx) imx = iv[i];
                } else {
                    if (dv[i] != dv[i]) continue;
                    if (!any || dv[i] < dmn) dmn = dv[i];
                    if (!any || dv[i] > dmx) dmx = dv[i];
                }
                any = true;
            }
            if (st.null_count != nulls) { ok = 0; printf("  [%s] col %d page %d null_count\n", tag, c, p); break; }
            if (!any) {
                /* No bounds: a null page, or (DOUBLE) all NaN, which withholds the index. */
                continue;
            }
            if (c == 0) {
                ok = st.min_value_size == 4 && st.max_value_size == 4 &&
                     memcmp(st.min_value, &imn, 4) == 0 && memcmp(st.max_value, &imx, 4) == 0;
            } else {
                if (dmn == 0.0) dmn = -0.0;
                if (dmx == 0.0) dmx = 0.0;
                ok = st.min_value_size == 8 && st.max_value_size == 8 &&
                     memcmp(st.min_value, &dmn, 8) == 0 && memcmp(st.max_value, &dmx, 8) == 0;
            }
            if (!ok) printf("  [%s] col %d page %d bounds\n", tag, c, p);
        }
        if (oi) carquet_offset_index_free(oi);
        if (ci) carquet_column_index_free(ci);
    }
    /* Values round trip. */
    if (ok) {
        int32_t* gi = malloc(N * sizeof(int32_t));
        int16_t* gd = malloc(N * sizeof(int16_t));
        double* gv = malloc(N * sizeof(double));
        carquet_column_reader_t* c0 = carquet_reader_get_column(r, 0, 0, &err);
        carquet_column_reader_t* c1 = carquet_reader_get_column(r, 0, 1, &err);
        ok = c0 && c1 &&
             carquet_column_read_batch(c0, gi, N, gd, NULL) == N &&
             carquet_column_read_batch(c1, gv, N, NULL, NULL) == N &&
             memcmp(gd, idef, N * sizeof(int16_t)) == 0 &&
             memcmp(gi, ipacked, (size_t)np * sizeof(int32_t)) == 0 &&
             memcmp(gv, dv, N * sizeof(double)) == 0;
        if (!ok) printf("  [%s] value mismatch\n", tag);
        if (c0) carquet_column_reader_free(c0);
        if (c1) carquet_column_reader_free(c1);
        free(gi); free(gd); free(gv);
    }
    if (r) carquet_reader_close(r);
    carquet_test_cleanup(path);
    free(iv); free(dv); free(idef); free(ipacked);
    if (!ok) TEST_FAIL(tag, "dictionary data pages");
    TEST_PASS(tag);
    return 0;
}

/* Regression: a write that adds no value bytes to the row-group size
 * estimate (an all-null batch of an OPTIONAL BYTE_ARRAY column; any INT96
 * batch, which the estimate did not size) saturated the estimate, and every
 * later aligned write then flushed a row group of its own. Three row groups
 * written in several calls each must stay three row groups. */
static int test_row_groups_all_null_batches(void) {
    enum { RG = 50000, STEP = 7001, NRG = 3 };
    char path[512]; carquet_test_temp_path(path, sizeof(path), "rg_all_null");
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    carquet_logical_type_t str; memset(&str, 0, sizeof(str));
    str.id = CARQUET_LOGICAL_STRING;
    int ok = s &&
        carquet_schema_add_column(s, "s", CARQUET_PHYSICAL_BYTE_ARRAY, &str,
            CARQUET_REPETITION_OPTIONAL, 0, 0) == CARQUET_OK &&
        carquet_schema_add_column(s, "k", CARQUET_PHYSICAL_INT32, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK &&
        carquet_schema_add_column(s, "t", CARQUET_PHYSICAL_INT96, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK;
    carquet_writer_options_t o; carquet_writer_options_init(&o);
    o.compression = CARQUET_COMPRESSION_ZSTD;
    carquet_writer_t* w = ok ? carquet_writer_create(path, s, &o, &err) : NULL;
    ok = w != NULL;
    static uint8_t text[] = "abcdefgh";
    carquet_byte_array_t* v = malloc(STEP * sizeof(*v));
    int16_t* def = malloc(STEP * sizeof(int16_t));
    int32_t* k = malloc(STEP * sizeof(int32_t));
    carquet_int96_t* t = calloc(STEP, sizeof(carquet_int96_t));
    for (int i = 0; i < STEP; i++) {
        v[i].data = text; v[i].length = 1 + i % 8; k[i] = i;
    }
    for (int rg = 0; ok && rg < NRG; rg++) {
        for (int off = 0; ok && off < RG; off += STEP) {
            int m = RG - off < STEP ? RG - off : STEP;
            for (int i = 0; i < m; i++) def[i] = rg == 1 ? 0 : (i % 3 != 0);
            ok = carquet_writer_write_batch(w, 0, v, m, def, NULL) == CARQUET_OK &&
                 carquet_writer_write_batch(w, 1, k, m, NULL, NULL) == CARQUET_OK &&
                 carquet_writer_write_batch(w, 2, t, m, NULL, NULL) == CARQUET_OK;
        }
        if (ok && rg + 1 < NRG) ok = carquet_writer_new_row_group(w) == CARQUET_OK;
    }
    if (w && carquet_writer_close(w) != CARQUET_OK) ok = 0;
    if (s) carquet_schema_free(s);
    free(v); free(def); free(k); free(t);
    if (ok) {
        carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
        ok = r != NULL;
        int32_t n = ok ? carquet_reader_num_row_groups(r) : -1;
        if (ok && n != NRG) {
            printf("  [rg_count_all_null_batches] %d row groups, want %d\n", n, NRG);
            ok = 0;
        }
        for (int32_t i = 0; ok && i < n; i++) {
            carquet_column_chunk_metadata_t m;
            ok = carquet_reader_column_chunk_metadata(r, i, 1, &m) == CARQUET_OK &&
                 m.num_values == RG;
        }
        if (r) carquet_reader_close(r);
    }
    carquet_test_cleanup(path);
    if (!ok) TEST_FAIL("rg_count_all_null_batches", "row groups split by all-null / INT96 batches");
    TEST_PASS("rg_count_all_null_batches");
    return 0;
}

/* ----------------------------------------------------------------------------
 * Automatic dictionary: with a codec, an INT32 / INT64 column gets a
 * dictionary chunk by chunk, when a sample of the chunk shows repeated values.
 * ------------------------------------------------------------------------- */

#define AD_COLS 4   /* low-cardinality i64, distinct i64, mid-cardinality i32, low-cardinality double */

typedef struct {
    int64_t* low;     /* 300 distinct values */
    int64_t* uniq;    /* all distinct */
    int32_t* mid;     /* 20000 distinct values */
    double* dbl;      /* 50 distinct values */
} adata_t;

static void adata_init(adata_t* d) {
    d->low = malloc(ROWS * sizeof(int64_t));
    d->uniq = malloc(ROWS * sizeof(int64_t));
    d->mid = malloc(ROWS * sizeof(int32_t));
    d->dbl = malloc(ROWS * sizeof(double));
    lcg_state = 23;
    for (int i = 0; i < ROWS; i++) {
        d->low[i] = (int64_t)(lcg() % 300) * 1000003;
        d->uniq[i] = (int64_t)i * 7919 + 13;
        d->mid[i] = (int32_t)(lcg() % 20000) - 5000;
        d->dbl[i] = (double)(lcg() % 50) * 0.25;
    }
}

static void adata_free(adata_t* d) {
    free(d->low); free(d->uniq); free(d->mid); free(d->dbl);
}

/* The four columns in one row group; `plain` pins the integer columns to
 * PLAIN. */
static int write_auto(const char* path, const adata_t* d, carquet_compression_t codec,
                      bool plain) {
    static const carquet_physical_type_t types[AD_COLS] = {
        CARQUET_PHYSICAL_INT64, CARQUET_PHYSICAL_INT64,
        CARQUET_PHYSICAL_INT32, CARQUET_PHYSICAL_DOUBLE};
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) return 0;
    int ok = 1;
    for (int i = 0; ok && i < AD_COLS; i++) {
        char name[8]; snprintf(name, sizeof(name), "c%d", i);
        ok = carquet_schema_add_column(s, name, types[i], NULL,
                CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK;
    }
    carquet_writer_options_t o; carquet_writer_options_init(&o);
    o.compression = codec;
    carquet_writer_t* w = ok ? carquet_writer_create(path, s, &o, &err) : NULL;
    if (!w) { carquet_schema_free(s); return 0; }
    for (int i = 0; ok && plain && i < 3; i++) {
        ok = carquet_writer_set_column_encoding(w, i, CARQUET_ENCODING_PLAIN) == CARQUET_OK;
    }
    const void* cols[AD_COLS] = {d->low, d->uniq, d->mid, d->dbl};
    for (int i = 0; ok && i < AD_COLS; i++) {
        ok = carquet_writer_write_batch(w, i, cols[i], ROWS, NULL, NULL) == CARQUET_OK;
    }
    if (carquet_writer_close(w) != CARQUET_OK) ok = 0;
    carquet_schema_free(s);
    return ok;
}

/* has_dictionary_page of the first row group's chunks against `want`, and
 * the values read back. */
static int auto_file_ok(const char* path, const adata_t* d, const bool want[AD_COLS],
                        const char* tag) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
    if (!r) { printf("  [%s] open failed\n", tag); return 0; }
    const void* cols[AD_COLS] = {d->low, d->uniq, d->mid, d->dbl};
    static const size_t widths[AD_COLS] = {8, 8, 4, 8};
    void* got = malloc(ROWS * 8);
    int ok = got != NULL;
    for (int c = 0; ok && c < AD_COLS; c++) {
        carquet_column_chunk_metadata_t m;
        ok = carquet_reader_column_chunk_metadata(r, 0, c, &m) == CARQUET_OK;
        if (ok && m.has_dictionary_page != want[c]) {
            printf("  [%s] col %d: dictionary page %s\n", tag, c,
                   m.has_dictionary_page ? "present" : "missing");
            ok = 0;
        }
        /* The chunk lists RLE_DICTIONARY exactly when it has a dictionary. */
        bool listed = false;
        for (int32_t e = 0; ok && e < m.num_encodings; e++) {
            if (m.encodings[e] == CARQUET_ENCODING_RLE_DICTIONARY) listed = true;
        }
        if (ok && listed != want[c]) {
            printf("  [%s] col %d: encodings list disagrees\n", tag, c);
            ok = 0;
        }
        carquet_column_reader_t* cr = ok ? carquet_reader_get_column(r, 0, c, &err) : NULL;
        if (ok) ok = cr && carquet_column_read_batch(cr, got, ROWS, NULL, NULL) == ROWS &&
                     memcmp(got, cols[c], ROWS * widths[c]) == 0;
        if (cr) carquet_column_reader_free(cr);
        if (!ok) printf("  [%s] col %d failed\n", tag, c);
    }
    free(got);
    carquet_reader_close(r);
    return ok;
}

static int test_auto_dictionary_choice(void) {
    adata_t d; adata_init(&d);
    char pa[512], pp[512], pu[512];
    carquet_test_temp_path(pa, sizeof(pa), "ad_auto");
    carquet_test_temp_path(pp, sizeof(pp), "ad_plain");
    carquet_test_temp_path(pu, sizeof(pu), "ad_unc");

    /* With a codec: the repeated integer columns get a dictionary, the
     * distinct one and the DOUBLE do not. */
    static const bool want_auto[AD_COLS] = {true, false, true, false};
    static const bool want_none[AD_COLS] = {false, false, false, false};
    int ok = write_auto(pa, &d, CARQUET_COMPRESSION_SNAPPY, false) &&
             auto_file_ok(pa, &d, want_auto, "auto_dictionary/codec");
    /* An explicit encoding is never overridden. */
    ok = ok && write_auto(pp, &d, CARQUET_COMPRESSION_SNAPPY, true) &&
         auto_file_ok(pp, &d, want_none, "auto_dictionary/explicit_plain");
    /* Uncompressed columns stay PLAIN (zero-copy reads). */
    ok = ok && write_auto(pu, &d, CARQUET_COMPRESSION_UNCOMPRESSED, false) &&
         auto_file_ok(pu, &d, want_none, "auto_dictionary/uncompressed");

    /* A chunk whose sample rejects the dictionary is the PLAIN chunk, byte
     * for byte (and is encoded page-parallel like one). */
    if (ok) {
        carquet_error_t err = CARQUET_ERROR_INIT;
        carquet_reader_t* ra = carquet_reader_open(pa, NULL, &err);
        carquet_reader_t* rp = carquet_reader_open(pp, NULL, &err);
        carquet_column_chunk_metadata_t ma, mp;
        ok = ra && rp &&
             carquet_reader_column_chunk_metadata(ra, 0, 1, &ma) == CARQUET_OK &&
             carquet_reader_column_chunk_metadata(rp, 0, 1, &mp) == CARQUET_OK &&
             ma.total_compressed_size == mp.total_compressed_size;
        if (ok) {
            uint8_t* a = read_range(pa, ma.data_page_offset, ma.total_compressed_size);
            uint8_t* b = read_range(pp, mp.data_page_offset, mp.total_compressed_size);
            ok = a && b && memcmp(a, b, (size_t)ma.total_compressed_size) == 0;
            free(a); free(b);
        }
        if (!ok) printf("  [auto_dictionary] rejected chunk differs from the PLAIN chunk\n");
        if (ra) carquet_reader_close(ra);
        if (rp) carquet_reader_close(rp);
    }

    carquet_test_cleanup(pa); carquet_test_cleanup(pp); carquet_test_cleanup(pu);
    adata_free(&d);
    if (!ok) TEST_FAIL("auto_dictionary_choice", "wrong encoding choice or values");
    TEST_PASS("auto_dictionary_choice");
    return 0;
}

/* The choice is made per chunk: one column (REQUIRED, and OPTIONAL with
 * ~10% nulls) alternates between repeated and distinct row groups, more row
 * groups than the pipeline has slots. */
static int test_auto_dictionary_per_chunk(void) {
    enum { RG = 20000, NRG = ROWS / RG };
    int64_t* vals = malloc(ROWS * sizeof(int64_t));
    int64_t* packed = malloc(ROWS * sizeof(int64_t));
    int16_t* defs = malloc(ROWS * sizeof(int16_t));
    lcg_state = 31;
    for (int i = 0; i < ROWS; i++) {
        bool repeated = ((i / RG) % 2) == 0;
        vals[i] = repeated ? (int64_t)(lcg() % 64) - 20 : (int64_t)i * 104729 - 7;
        defs[i] = (lcg() % 10) != 0;
    }
    char path[512];
    carquet_test_temp_path(path, sizeof(path), "ad_chunks");

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    int ok = s &&
        carquet_schema_add_column(s, "req", CARQUET_PHYSICAL_INT64, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK &&
        carquet_schema_add_column(s, "opt", CARQUET_PHYSICAL_INT64, NULL,
            CARQUET_REPETITION_OPTIONAL, 0, 0) == CARQUET_OK;
    carquet_writer_options_t o; carquet_writer_options_init(&o);
    o.compression = CARQUET_COMPRESSION_ZSTD;
    o.compression_level = 1;
    carquet_writer_t* w = ok ? carquet_writer_create(path, s, &o, &err) : NULL;
    ok = w != NULL;
    for (int64_t off = 0; ok && off < ROWS; off += RG) {
        int64_t k = 0;
        for (int64_t i = off; i < off + RG; i++) {
            if (defs[i]) packed[k++] = vals[i];
        }
        ok = carquet_writer_write_batch(w, 0, vals + off, RG, NULL, NULL) == CARQUET_OK &&
             carquet_writer_write_batch(w, 1, packed, RG, defs + off, NULL) == CARQUET_OK;
        if (ok && off + RG < ROWS) ok = carquet_writer_new_row_group(w) == CARQUET_OK;
    }
    if (w && carquet_writer_close(w) != CARQUET_OK) ok = 0;
    if (s) carquet_schema_free(s);

    carquet_reader_t* r = ok ? carquet_reader_open(path, NULL, &err) : NULL;
    ok = r && carquet_reader_num_row_groups(r) == NRG;
    int64_t* got = malloc(RG * sizeof(int64_t));
    int16_t* gdef = malloc(RG * sizeof(int16_t));
    for (int32_t rg = 0; ok && rg < NRG; rg++) {
        int64_t off = (int64_t)rg * RG;
        for (int c = 0; ok && c < 2; c++) {
            carquet_column_chunk_metadata_t m;
            ok = carquet_reader_column_chunk_metadata(r, rg, c, &m) == CARQUET_OK &&
                 m.has_dictionary_page == ((rg % 2) == 0);
            if (!ok) { printf("  [auto_dictionary_per_chunk] rg %d col %d: dictionary\n", rg, c); break; }
            carquet_column_reader_t* cr = carquet_reader_get_column(r, rg, c, &err);
            ok = cr && carquet_column_read_batch(cr, got, RG, c ? gdef : NULL, NULL) == RG;
            if (ok && c == 0) ok = memcmp(got, vals + off, RG * sizeof(int64_t)) == 0;
            if (ok && c == 1) {
                int64_t k = 0;
                for (int64_t i = off; ok && i < off + RG; i++) {
                    if (gdef[i - off] != defs[i]) ok = 0;
                    else if (defs[i] && got[k++] != vals[i]) ok = 0;
                }
            }
            if (cr) carquet_column_reader_free(cr);
            if (!ok) printf("  [auto_dictionary_per_chunk] rg %d col %d: values\n", rg, c);
        }
    }
    free(got); free(gdef);
    if (r) carquet_reader_close(r);
    carquet_test_cleanup(path);
    free(vals); free(packed); free(defs);
    if (!ok) TEST_FAIL("auto_dictionary_per_chunk", "per-chunk dictionary choice or values");
    TEST_PASS("auto_dictionary_per_chunk");
    return 0;
}

/* Abort with row groups still in the pipeline: the queued writes are
 * dropped, nothing is left behind and nothing is freed twice. */
static int test_abort_in_flight(void) {
    data_t d; data_init(&d);
    char path[512];
    carquet_test_temp_path(path, sizeof(path), "ppw_abort");
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    int ok = s &&
        carquet_schema_add_column(s, "id", CARQUET_PHYSICAL_INT64, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK &&
        carquet_schema_add_column(s, "value", CARQUET_PHYSICAL_DOUBLE, NULL,
            CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK;
    carquet_writer_options_t o; carquet_writer_options_init(&o);
    o.compression = CARQUET_COMPRESSION_ZSTD;
    carquet_writer_t* w = ok ? carquet_writer_create(path, s, &o, &err) : NULL;
    ok = w != NULL;
    for (int64_t off = 0; ok && off + 30000 <= ROWS; off += 30000) {
        ok = carquet_writer_write_batch(w, 0, d.ids + off, 30000, NULL, NULL) == CARQUET_OK &&
             carquet_writer_write_batch(w, 1, d.values + off, 30000, NULL, NULL) == CARQUET_OK &&
             carquet_writer_new_row_group(w) == CARQUET_OK;
    }
    if (w) carquet_writer_abort(w);
    if (s) carquet_schema_free(s);
    if (ok && file_size(path) >= 0) {
        printf("  [abort_in_flight] the aborted file is still there\n");
        ok = 0;
    }
    carquet_test_cleanup(path);
    data_free(&d);
    if (!ok) TEST_FAIL("abort_in_flight", "abort with row groups in the pipeline");
    TEST_PASS("abort_in_flight");
    return 0;
}

int main(void) {
    int failures = 0;
    cfg_t base = {CARQUET_COMPRESSION_LZ4_RAW, 0, 0, 1, true, 0};
    failures += test_page_parallel_identical(&base, "page_parallel_lz4");
    cfg_t z = base; z.codec = CARQUET_COMPRESSION_ZSTD;
    failures += test_page_parallel_identical(&z, "page_parallel_zstd");
    cfg_t sn = base; sn.codec = CARQUET_COMPRESSION_SNAPPY; sn.write_batch_size = 50000;
    failures += test_page_parallel_identical(&sn, "page_parallel_snappy_batch50k");
    cfg_t rows = base; rows.max_rows_per_page = 70000;
    failures += test_page_parallel_identical(&rows, "page_parallel_max_rows_70k");
    cfg_t v2 = base; v2.data_page_version = 2; v2.codec = CARQUET_COMPRESSION_GZIP;
    failures += test_page_parallel_identical(&v2, "page_parallel_gzip_v2");
    cfg_t unc = base; unc.codec = CARQUET_COMPRESSION_UNCOMPRESSED;
    failures += test_page_parallel_identical(&unc, "page_parallel_uncompressed");
    failures += test_async_io_identical(CARQUET_COMPRESSION_ZSTD, 70000, "async_io_identical_zstd");
    failures += test_async_io_identical(CARQUET_COMPRESSION_UNCOMPRESSED, 70000, "async_io_identical_none");
    /* 20 small row groups: more than the pipeline's slots stay in flight. */
    failures += test_async_io_identical(CARQUET_COMPRESSION_SNAPPY, 15000, "async_io_identical_snappy_20rg");
    failures += test_async_buffer_writer();
    {
        cfg_t b = {CARQUET_COMPRESSION_ZSTD, 0, 0, 1, true, 70000};
        failures += test_buffer_matches_file(&b, false, false, ROWS, "buffer_matches_file_zstd_async");
        b.async_io = false;
        failures += test_buffer_matches_file(&b, false, false, ROWS, "buffer_matches_file_zstd_sync");
        b.codec = CARQUET_COMPRESSION_UNCOMPRESSED;
        failures += test_buffer_matches_file(&b, false, false, ROWS, "buffer_matches_file_none_sync");
        b.async_io = true;
        failures += test_buffer_matches_file(&b, false, false, ROWS, "buffer_matches_file_none_async");
        /* One row group, and 20: more than the pipeline keeps in flight. */
        b.rg_rows = 0;
        failures += test_buffer_matches_file(&b, false, false, ROWS, "buffer_matches_file_one_rg");
        b.codec = CARQUET_COMPRESSION_SNAPPY; b.rg_rows = 15000; b.data_page_version = 2;
        failures += test_buffer_matches_file(&b, false, false, ROWS, "buffer_matches_file_snappy_20rg_v2");
        /* Bloom filters and page indexes are written between row groups,
         * from the caller's thread. */
        b.rg_rows = 70000; b.data_page_version = 1;
        failures += test_buffer_matches_file(&b, true, true, ROWS, "buffer_matches_file_bloom_index_async");
        failures += test_buffer_matches_file(&b, true, false, ROWS, "buffer_matches_file_bloom_async");
        failures += test_buffer_matches_file(&b, false, true, ROWS, "buffer_matches_file_index_async");
        b.async_io = false;
        failures += test_buffer_matches_file(&b, true, true, ROWS, "buffer_matches_file_bloom_index_sync");
        /* No rows at all: header, footer, nothing else. */
        failures += test_buffer_matches_file(&b, false, false, 0, "buffer_matches_file_empty_sync");
        failures += test_buffer_matches_file(&b, true, true, 0, "buffer_matches_file_empty_bloom_index");
        b.async_io = true;
        failures += test_buffer_matches_file(&b, false, false, 0, "buffer_matches_file_empty_async");
    }
    failures += test_buffer_writer_abandoned();
    failures += test_mixed_nullable();
    failures += test_async_write_error();

    /* Nullable and dictionary columns: page-parallel nullable pages, whole
     * columns on the pipeline pool, several row groups. */
    cfg_t k = base;
    failures += test_kinds_identical(K_OPT, K_OPT, K_OPT, &k, "kinds_opt_lz4");
    failures += test_kinds_identical(K_OPT_RUNS, K_BSS_OPT, K_OPT_NONE, &k, "kinds_opt_runs_bss");
    failures += test_kinds_identical(K_ALL_NULL, K_OPT, K_REQ, &k, "kinds_all_null");
    failures += test_kinds_identical(K_REQ, K_REQ, K_DICT, &k, "kinds_dict_lz4");
    failures += test_kinds_identical(K_DICT, K_DICT_OPT, K_DICT_OPT, &k, "kinds_dict_opt");
    cfg_t kz = base; kz.codec = CARQUET_COMPRESSION_ZSTD; kz.rg_rows = 70000;
    failures += test_kinds_identical(K_OPT, K_DICT_OPT, K_REQ, &kz, "kinds_zstd_5rg");
    cfg_t kr = base; kr.codec = CARQUET_COMPRESSION_SNAPPY; kr.max_rows_per_page = 30000;
    kr.rg_rows = 110000;
    failures += test_kinds_identical(K_OPT, K_OPT_RUNS, K_DICT_OPT, &kr, "kinds_snappy_max_rows_30k");
    cfg_t kb = base; kb.write_batch_size = 7000; kb.data_page_version = 2;
    failures += test_kinds_identical(K_OPT, K_BSS_OPT, K_DICT, &kb, "kinds_batch7k_v2");
    cfg_t ku = base; ku.codec = CARQUET_COMPRESSION_UNCOMPRESSED; ku.rg_rows = 70000;
    failures += test_kinds_identical(K_OPT, K_DICT_OPT, K_REQ, &ku, "kinds_uncompressed_5rg");
    cfg_t ka = base; ka.codec = CARQUET_COMPRESSION_SNAPPY; ka.rg_rows = 15000;
    failures += test_kinds_async_identical(K_OPT, K_DICT_OPT, K_REQ, &ka, "kinds_async_snappy_20rg");
    ka.codec = CARQUET_COMPRESSION_ZSTD; ka.rg_rows = 70000;
    failures += test_kinds_async_identical(K_DICT, K_OPT_RUNS, K_ALL_NULL, &ka, "kinds_async_zstd_5rg");
    failures += test_dict_pages(16 * 1024, 0, "dict_pages_by_size");
    failures += test_dict_pages(1024 * 1024, 25000, "dict_pages_max_rows_25k");
    failures += test_row_groups_all_null_batches();
    failures += test_auto_dictionary_choice();
    failures += test_auto_dictionary_per_chunk();
    failures += test_abort_in_flight();
    if (failures) { printf("\n%d test(s) FAILED\n", failures); return 1; }
    printf("\nAll parallel-write tests passed\n");
    return 0;
}
