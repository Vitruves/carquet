/**
 * @file test_dictionary_fallback.c
 * @brief preserve_dictionaries over a chunk falling back to PLAIN pages
 *
 * A dictionary-encoded column chunk may fall back to PLAIN pages after its
 * dictionary pages (as writers do once the dictionary is full). Read with
 * preserve_dictionaries, the batch reader returns the dictionary part's
 * indices, then fails at the PLAIN page: never the PLAIN values' bytes as
 * indices. A column reader reading indices (carquet_column_reader_reset_indices)
 * stops there too, and carquet_column_reader_materialize() reads the PLAIN
 * values from there. carquet's writer falls back for whole chunks only, so the
 * mixed chunk is spliced from two files' chunks: a dictionary-encoded one, then
 * a PLAIN one's data pages, under the first's footer, grown to both.
 */

#include <carquet/carquet.h>
#include "test_helpers.h"
#include "core/arena.h"
#include "core/buffer.h"
#include "thrift/parquet_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DICT_ROWS 3000
#define PLAIN_ROWS 2000
#define DICT_VALUES 7

static int64_t dict_value(int64_t i) { return 100 + i % DICT_VALUES; }
static int64_t plain_value(int64_t i) { return 1000000 + i; }

/* one INT64 REQUIRED column v, uncompressed, without statistics: rows of
 * value(i), dictionary-encoded or PLAIN */
static int write_file(const char* path, int64_t rows, int64_t (*value)(int64_t), bool dict) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema || carquet_schema_add_column(schema, "v", CARQUET_PHYSICAL_INT64, NULL,
                                             CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK) {
        return 1;
    }
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_UNCOMPRESSED;
    opts.write_statistics = false;
    carquet_writer_t* w = carquet_writer_create(path, schema, &opts, &err);
    if (!w) {
        carquet_schema_free(schema);
        return 1;
    }
    int64_t* vs = malloc(sizeof(int64_t) * (size_t)rows);
    for (int64_t i = 0; i < rows; i++) vs[i] = value(i);
    int rc = (dict && carquet_writer_set_column_encoding(w, 0, CARQUET_ENCODING_RLE_DICTIONARY) != CARQUET_OK) ||
             carquet_writer_write_batch(w, 0, vs, rows, NULL, NULL) != CARQUET_OK;
    rc |= carquet_writer_close(w) != CARQUET_OK;
    free(vs);
    carquet_schema_free(schema);
    return rc;
}

static uint8_t* read_file(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* b = malloc((size_t)n);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        b = NULL;
    }
    fclose(f);
    *size = (size_t)n;
    return b;
}

/* a file's footer parsed into arena */
static int parse_footer(const uint8_t* file, size_t size, carquet_arena_t* arena,
                        parquet_file_metadata_t* md) {
    uint32_t len = (uint32_t)file[size - 8] | ((uint32_t)file[size - 7] << 8) |
                   ((uint32_t)file[size - 6] << 16) | ((uint32_t)file[size - 5] << 24);
    carquet_error_t err = CARQUET_ERROR_INIT;
    return parquet_parse_file_metadata(file + size - 8 - len, len, arena, md, &err) != CARQUET_OK;
}

/* the mixed file: the dictionary file's chunk, then the PLAIN file's data
 * pages, under the dictionary file's footer counting both */
static int splice(const char* dict_path, const char* plain_path, const char* out_path) {
    size_t ds = 0, ps = 0;
    uint8_t* d = read_file(dict_path, &ds);
    uint8_t* p = read_file(plain_path, &ps);
    carquet_arena_t arena;
    parquet_file_metadata_t dm, pm;
    int rc = 1;
    if (!d || !p || carquet_arena_init(&arena) != CARQUET_OK) goto out;
    if (parse_footer(d, ds, &arena, &dm) || parse_footer(p, ps, &arena, &pm)) goto out_arena;
    {
        parquet_column_metadata_t* dc = &dm.row_groups[0].columns[0].metadata;
        const parquet_column_metadata_t* pc = &pm.row_groups[0].columns[0].metadata;
        if (!dc->has_dictionary_page_offset || pc->has_dictionary_page_offset) goto out_arena;
        size_t dict_end = (size_t)(dc->dictionary_page_offset + dc->total_compressed_size);
        size_t plain_begin = (size_t)pc->data_page_offset;
        size_t plain_bytes = (size_t)pc->total_compressed_size;

        dc->num_values += pc->num_values;
        dc->total_compressed_size += pc->total_compressed_size;
        dc->total_uncompressed_size += pc->total_uncompressed_size;
        dc->num_encoding_stats = 0;
        dm.row_groups[0].num_rows += pm.row_groups[0].num_rows;
        dm.row_groups[0].total_byte_size += pm.row_groups[0].total_byte_size;
        if (dm.row_groups[0].has_total_compressed_size) {
            dm.row_groups[0].total_compressed_size += pc->total_compressed_size;
        }
        dm.num_rows += pm.num_rows;

        carquet_buffer_t footer;
        carquet_buffer_init(&footer);
        carquet_error_t err = CARQUET_ERROR_INIT;
        FILE* f = fopen(out_path, "wb");
        if (f && parquet_write_file_metadata(&dm, &footer, &err) == CARQUET_OK) {
            size_t n = carquet_buffer_size(&footer);
            uint8_t len[4] = {(uint8_t)n, (uint8_t)(n >> 8), (uint8_t)(n >> 16), (uint8_t)(n >> 24)};
            rc = fwrite(d, 1, dict_end, f) != dict_end ||
                 fwrite(p + plain_begin, 1, plain_bytes, f) != plain_bytes ||
                 fwrite(carquet_buffer_data(&footer), 1, n, f) != n ||
                 fwrite(len, 1, 4, f) != 4 || fwrite("PAR1", 1, 4, f) != 4;
        }
        if (f && fclose(f) != 0) rc = 1;
        carquet_buffer_destroy(&footer);
    }
out_arena:
    carquet_arena_destroy(&arena);
out:
    free(d);
    free(p);
    return rc;
}

#define STRING_ROWS 1000
static const char* const strings[] = {"", "a", "bb", "a longer string value", "ccc"};

/* one BYTE_ARRAY REQUIRED column s, dictionary-encoded: rows of strings[i % 5] */
static int write_strings(const char* path) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema || carquet_schema_add_column(schema, "s", CARQUET_PHYSICAL_BYTE_ARRAY, NULL,
                                             CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK) {
        return 1;
    }
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    carquet_writer_t* w = carquet_writer_create(path, schema, &opts, &err);
    if (!w) {
        carquet_schema_free(schema);
        return 1;
    }
    static carquet_byte_array_t vs[STRING_ROWS];
    for (int i = 0; i < STRING_ROWS; i++) {
        vs[i].data = (uint8_t*)strings[i % 5];
        vs[i].length = (int32_t)strlen(strings[i % 5]);
    }
    int rc = carquet_writer_set_column_encoding(w, 0, CARQUET_ENCODING_RLE_DICTIONARY) != CARQUET_OK ||
             carquet_writer_write_batch(w, 0, vs, STRING_ROWS, NULL, NULL) != CARQUET_OK;
    rc |= carquet_writer_close(w) != CARQUET_OK;
    carquet_schema_free(schema);
    return rc;
}

/* the next n reads of c, as indices, are the dictionary part's first n rows */
static int check_indices(carquet_column_reader_t* c, int64_t n) {
    static uint32_t indices[DICT_ROWS];
    const uint8_t* dict = NULL;
    const uint32_t* offsets = NULL;
    int32_t count = 0;
    if (carquet_column_read_batch(c, indices, n, NULL, NULL) != n ||
        carquet_column_reader_dictionary(c, &dict, &count, &offsets) != CARQUET_OK ||
        count != DICT_VALUES || offsets)
        return 1;
    for (int64_t i = 0; i < n; i++) {
        int64_t v;
        if (indices[i] >= (uint32_t)count) return 1;
        memcpy(&v, dict + (size_t)indices[i] * sizeof(int64_t), sizeof(v));
        if (v != dict_value(i)) return 1;
    }
    return 0;
}

/* column readers reading indices: the mixed chunk to its PLAIN page, then values;
 * resets between indices and values; a PLAIN chunk; a BYTE_ARRAY dictionary */
static int column_reader_indices(carquet_reader_t* r, const char* plain_path, const char* strings_path) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    static int64_t values[DICT_ROWS + PLAIN_ROWS];
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    bool indices = false;
    if (!c || carquet_column_reader_reset_indices(c, 0, &indices, &err) != CARQUET_OK || !indices)
        TEST_FAIL("column_reader_indices", "reset_indices");
    const uint8_t* dict = NULL;
    int32_t count = 0;
    if (carquet_column_reader_dictionary(c, &dict, &count, NULL) != CARQUET_ERROR_DICTIONARY_NOT_FOUND)
        TEST_FAIL("column_reader_indices", "a dictionary before the first read");
    if (check_indices(c, 10)) TEST_FAIL("column_reader_indices", "the first indices");
    /* in the middle of a page: refused */
    carquet_error_t e = CARQUET_ERROR_INIT;
    if (carquet_column_reader_materialize(c, &e) != CARQUET_ERROR_INVALID_STATE)
        TEST_FAIL("column_reader_indices", "materialize in the middle of a page");

    /* on to the PLAIN page: a partial read of the indices before it, stopped there */
    static uint32_t rest[DICT_ROWS];
    int64_t got = 10;
    for (;;) {
        carquet_error_t re = CARQUET_ERROR_INIT;
        int64_t n = carquet_column_read_batch_ex(c, rest, 700, NULL, NULL, &re);
        if (carquet_error_is_set(&re)) {
            if (re.code != CARQUET_ERROR_INVALID_ENCODING)
                TEST_FAIL("column_reader_indices", re.message);
            got += n > 0 ? n : 0;
            break;
        }
        if (n <= 0) TEST_FAIL("column_reader_indices", "the indices ended before the PLAIN page");
        got += n;
    }
    if (got != DICT_ROWS) TEST_FAIL("column_reader_indices", "not stopped at the PLAIN page");

    /* the rest as values, from the PLAIN page */
    if (carquet_column_reader_materialize(c, &err) != CARQUET_OK)
        TEST_FAIL("column_reader_indices", "materialize at the PLAIN page");
    if (carquet_column_read_batch_ex(c, values, PLAIN_ROWS, NULL, NULL, &err) != PLAIN_ROWS ||
        carquet_column_has_next(c) || values[0] != plain_value(0) ||
        values[PLAIN_ROWS - 1] != plain_value(PLAIN_ROWS - 1))
        TEST_FAIL("column_reader_indices", "the PLAIN page's values");

    /* a reset after materialize reads indices again; reset reads values; and back */
    if (carquet_column_reader_reset_indices(c, 0, &indices, &err) != CARQUET_OK || !indices ||
        check_indices(c, DICT_ROWS))
        TEST_FAIL("column_reader_indices", "indices again after materialize");
    if (carquet_column_reader_reset(c, 0, &err) != CARQUET_OK ||
        carquet_column_read_batch(c, values, DICT_ROWS + PLAIN_ROWS, NULL, NULL) != DICT_ROWS + PLAIN_ROWS ||
        values[DICT_ROWS - 1] != dict_value(DICT_ROWS - 1) || values[DICT_ROWS] != plain_value(0))
        TEST_FAIL("column_reader_indices", "values after reading indices");
    if (carquet_column_reader_reset_indices(c, 0, &indices, &err) != CARQUET_OK || !indices ||
        check_indices(c, 100))
        TEST_FAIL("column_reader_indices", "indices after reading values");
    carquet_column_reader_free(c);

    /* a PLAIN chunk: no indices, its values read as ever */
    carquet_reader_t* pr = carquet_reader_open(plain_path, NULL, &err);
    carquet_column_reader_t* pc = pr ? carquet_reader_get_column(pr, 0, 0, &err) : NULL;
    if (!pc || carquet_column_reader_reset_indices(pc, 0, &indices, &err) != CARQUET_OK || indices ||
        carquet_column_read_batch(pc, values, PLAIN_ROWS, NULL, NULL) != PLAIN_ROWS || values[5] != plain_value(5))
        TEST_FAIL("column_reader_indices", "a PLAIN chunk");
    carquet_column_reader_free(pc);
    carquet_reader_close(pr);

    /* BYTE_ARRAY: the layout of carquet_row_batch_column_dictionary() */
    carquet_reader_t* sr = carquet_reader_open(strings_path, NULL, &err);
    carquet_column_reader_t* sc = sr ? carquet_reader_get_column(sr, 0, 0, &err) : NULL;
    static uint32_t sidx[STRING_ROWS];
    const uint32_t* offsets = NULL;
    if (!sc || carquet_column_reader_reset_indices(sc, 0, &indices, &err) != CARQUET_OK || !indices ||
        carquet_column_read_batch(sc, sidx, STRING_ROWS, NULL, NULL) != STRING_ROWS ||
        carquet_column_reader_dictionary(sc, &dict, &count, &offsets) != CARQUET_OK || count != 5 || !offsets)
        TEST_FAIL("column_reader_indices", "a BYTE_ARRAY dictionary");
    for (int i = 0; i < STRING_ROWS; i++) {
        if (sidx[i] >= (uint32_t)count) TEST_FAIL("column_reader_indices", "a string index");
        const uint8_t* entry = dict + offsets[sidx[i]];
        uint32_t len;
        memcpy(&len, entry, sizeof(len));
        if (len != strlen(strings[i % 5]) || memcmp(entry + 4, strings[i % 5], len) != 0)
            TEST_FAIL("column_reader_indices", "a string");
    }
    carquet_column_reader_free(sc);
    carquet_reader_close(sr);
    TEST_PASS("column_reader_indices");
    return 0;
}

int main(void) {
    carquet_init();
    char dict_path[512], plain_path[512], mixed_path[512], strings_path[512];
    carquet_test_temp_path(dict_path, sizeof(dict_path), "test_dictionary_fallback_dict");
    carquet_test_temp_path(plain_path, sizeof(plain_path), "test_dictionary_fallback_plain");
    carquet_test_temp_path(mixed_path, sizeof(mixed_path), "test_dictionary_fallback_mixed");
    carquet_test_temp_path(strings_path, sizeof(strings_path), "test_dictionary_fallback_strings");
    if (write_file(dict_path, DICT_ROWS, dict_value, true) ||
        write_file(plain_path, PLAIN_ROWS, plain_value, false) ||
        splice(dict_path, plain_path, mixed_path) || write_strings(strings_path))
        TEST_FAIL("dictionary_fallback", "cannot write the files");

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* r = carquet_reader_open(mixed_path, NULL, &err);
    if (!r) TEST_FAIL("dictionary_fallback", err.message);

    /* values: the spliced chunk reads whole */
    static int64_t values[DICT_ROWS + PLAIN_ROWS];
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    if (!c || carquet_column_read_batch(c, values, DICT_ROWS + PLAIN_ROWS, NULL, NULL) != DICT_ROWS + PLAIN_ROWS)
        TEST_FAIL("dictionary_fallback", "values of the mixed chunk");
    for (int64_t i = 0; i < DICT_ROWS + PLAIN_ROWS; i++) {
        if (values[i] != (i < DICT_ROWS ? dict_value(i) : plain_value(i - DICT_ROWS)))
            TEST_FAIL("dictionary_fallback", "a value of the mixed chunk");
    }
    carquet_column_reader_free(c);
    if (column_reader_indices(r, plain_path, strings_path)) return 1;

    /* preserved dictionaries: the dictionary part's indices, then an error */
    carquet_batch_reader_config_t config;
    carquet_batch_reader_config_init(&config);
    config.preserve_dictionaries = true;
    config.batch_size = 1000;
    carquet_batch_reader_t* br = carquet_batch_reader_create(r, &config, &err);
    if (!br) TEST_FAIL("dictionary_fallback", err.message);
    int64_t rows = 0;
    carquet_status_t status;
    for (;;) {
        carquet_row_batch_t* batch = NULL;
        status = carquet_batch_reader_next(br, &batch);
        if (status != CARQUET_OK || !batch) break;
        const uint32_t* indices;
        const uint8_t* nulls;
        const uint8_t* dict;
        const uint32_t* offsets;
        int64_t n = 0;
        int32_t count = 0;
        if (carquet_row_batch_column_dictionary(batch, 0, &indices, &nulls, &n, &dict, &count, &offsets) != CARQUET_OK)
            TEST_FAIL("dictionary_fallback", "a batch not dictionary-preserved");
        if (count != DICT_VALUES) TEST_FAIL("dictionary_fallback", "the dictionary");
        for (int64_t i = 0; i < n; i++) {
            if (rows + i >= DICT_ROWS) TEST_FAIL("dictionary_fallback", "PLAIN values read as indices");
            if (indices[i] >= (uint32_t)count)
                TEST_FAIL("dictionary_fallback", "an index past the dictionary");
            int64_t v;
            memcpy(&v, dict + (size_t)indices[i] * sizeof(int64_t), sizeof(v));
            if (v != dict_value(rows + i)) TEST_FAIL("dictionary_fallback", "an index");
        }
        rows += n;
        carquet_row_batch_free(batch);
    }
    if (rows != DICT_ROWS) TEST_FAIL("dictionary_fallback", "not every dictionary-encoded row read");
    if (status == CARQUET_OK || status == CARQUET_ERROR_END_OF_DATA)
        TEST_FAIL("dictionary_fallback", "no error at the PLAIN page");

    carquet_batch_reader_free(br);
    carquet_reader_close(r);
    remove(dict_path);
    remove(plain_path);
    remove(mixed_path);
    remove(strings_path);
    TEST_PASS("dictionary_fallback");
    printf("\nAll dictionary fallback tests passed.\n");
    return 0;
}
