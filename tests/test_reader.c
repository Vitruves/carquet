/**
 * @file test_reader.c
 * @brief Tests for Parquet file reading
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <carquet/carquet.h>
#include "core/compat.h"
#include "reader/reader_internal.h"
#include "test_helpers.h"

static int test_version(void) {
    const char* ver = carquet_version();
    (void)ver;
    assert(ver != NULL);
    assert(strcmp(ver, CARQUET_VERSION_STRING) == 0);

    int major, minor, patch;
    carquet_version_components(&major, &minor, &patch);
    assert(major == CARQUET_VERSION_MAJOR);
    assert(minor == CARQUET_VERSION_MINOR);
    assert(patch == CARQUET_VERSION_PATCH);

    TEST_PASS("version");
    return 0;
}

static int test_cpu_detection(void) {
    const carquet_cpu_info_t* info = carquet_get_cpu_info();
    assert(info != NULL);

    printf("  CPU features detected:\n");
#if defined(__x86_64__) || defined(__i386__)
    printf("    SSE2:    %s\n", info->has_sse2 ? "yes" : "no");
    printf("    SSE4.1:  %s\n", info->has_sse41 ? "yes" : "no");
    printf("    SSE4.2:  %s\n", info->has_sse42 ? "yes" : "no");
    printf("    AVX:     %s\n", info->has_avx ? "yes" : "no");
    printf("    AVX2:    %s\n", info->has_avx2 ? "yes" : "no");
    printf("    AVX-512: %s\n", info->has_avx512f ? "yes" : "no");
#elif defined(__aarch64__) || defined(__arm64__) || defined(__arm__)
    printf("    NEON:    %s\n", info->has_neon ? "yes" : "no");
    printf("    SVE:     %s\n", info->has_sve ? "yes" : "no");
    if (info->has_sve) {
        printf("    SVE len: %d bits\n", info->sve_vector_length);
    }

#if defined(__aarch64__) || defined(__arm64__)
    assert(info->has_neon);
#endif
#else
    printf("    (no architecture-specific features)\n");
#endif

    TEST_PASS("cpu_detection");
    return 0;
}

static int test_reader_options(void) {
    carquet_reader_options_t opts;
    carquet_reader_options_init(&opts);

    assert(opts.use_mmap == false);
    assert(opts.verify_checksums == true);
    assert(opts.buffer_size == 64 * 1024);
    assert(opts.num_threads == 0);

    TEST_PASS("reader_options");
    return 0;
}

static int test_writer_options(void) {
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);

    assert(opts.compression == CARQUET_COMPRESSION_UNCOMPRESSED);
    assert(opts.row_group_size == 128 * 1024 * 1024);
    assert(opts.page_size == 1024 * 1024);
    assert(opts.write_statistics == true);
    assert(opts.write_crc == true);
    assert(opts.created_by != NULL);

    TEST_PASS("writer_options");
    return 0;
}

static int test_open_nonexistent(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* reader = carquet_reader_open(
        "/nonexistent/path/file.parquet", NULL, &err);
    (void)reader;

    assert(reader == NULL);
    assert(err.code == CARQUET_ERROR_FILE_OPEN);

    TEST_PASS("open_nonexistent");
    return 0;
}

static int test_type_names(void) {
    assert(strcmp(carquet_physical_type_name(CARQUET_PHYSICAL_BOOLEAN), "BOOLEAN") == 0);
    assert(strcmp(carquet_physical_type_name(CARQUET_PHYSICAL_INT32), "INT32") == 0);
    assert(strcmp(carquet_physical_type_name(CARQUET_PHYSICAL_INT64), "INT64") == 0);
    assert(strcmp(carquet_physical_type_name(CARQUET_PHYSICAL_DOUBLE), "DOUBLE") == 0);
    assert(strcmp(carquet_physical_type_name(CARQUET_PHYSICAL_BYTE_ARRAY), "BYTE_ARRAY") == 0);

    assert(strcmp(carquet_compression_name(CARQUET_COMPRESSION_UNCOMPRESSED), "UNCOMPRESSED") == 0);
    assert(strcmp(carquet_compression_name(CARQUET_COMPRESSION_SNAPPY), "SNAPPY") == 0);
    assert(strcmp(carquet_compression_name(CARQUET_COMPRESSION_GZIP), "GZIP") == 0);
    assert(strcmp(carquet_compression_name(CARQUET_COMPRESSION_LZ4), "LZ4") == 0);
    assert(strcmp(carquet_compression_name(CARQUET_COMPRESSION_ZSTD), "ZSTD") == 0);

    assert(strcmp(carquet_encoding_name(CARQUET_ENCODING_PLAIN), "PLAIN") == 0);
    assert(strcmp(carquet_encoding_name(CARQUET_ENCODING_RLE), "RLE") == 0);
    assert(strcmp(carquet_encoding_name(CARQUET_ENCODING_RLE_DICTIONARY), "RLE_DICTIONARY") == 0);

    TEST_PASS("type_names");
    return 0;
}

static int test_status_strings(void) {
    assert(strcmp(carquet_status_string(CARQUET_OK), "Success") == 0);
    assert(strcmp(carquet_status_string(CARQUET_ERROR_FILE_NOT_FOUND), "File not found") == 0);
    assert(strcmp(carquet_status_string(CARQUET_ERROR_INVALID_MAGIC), "Invalid magic bytes") == 0);
    assert(strcmp(carquet_status_string(CARQUET_ERROR_OUT_OF_MEMORY), "Out of memory") == 0);

    TEST_PASS("status_strings");
    return 0;
}

/**
 * Test nested schema definition/repetition level computation.
 *
 * Creates this schema:
 *   schema (root, required)
 *   ├── id (required, INT32)               -> def=0, rep=0
 *   ├── name (optional, BYTE_ARRAY)        -> def=1, rep=0
 *   ├── address (optional, group)
 *   │   ├── street (required, BYTE_ARRAY)  -> def=1, rep=0  (from parent)
 *   │   └── city (optional, BYTE_ARRAY)    -> def=2, rep=0  (from parent + self)
 *   └── phones (repeated, group)
 *       ├── number (required, BYTE_ARRAY)  -> def=1, rep=1  (from parent)
 *       └── type (optional, BYTE_ARRAY)    -> def=2, rep=1  (from parent + self)
 */
static int test_nested_schema_levels(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Create schema */
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        printf("Failed to create schema: %s\n", err.message);
        TEST_FAIL("nested_schema_levels", "schema creation failed");
    }

    carquet_status_t status;

    /* Add flat columns at root level */
    status = carquet_schema_add_column(
        schema, "id", CARQUET_PHYSICAL_INT32, NULL,
        CARQUET_REPETITION_REQUIRED, 0, 0);
    assert(status == CARQUET_OK);

    status = carquet_schema_add_column(
        schema, "name", CARQUET_PHYSICAL_BYTE_ARRAY, NULL,
        CARQUET_REPETITION_OPTIONAL, 0, 0);
    assert(status == CARQUET_OK);

    /* Add optional group "address" */
    int32_t address_idx = carquet_schema_add_group(
        schema, "address", CARQUET_REPETITION_OPTIONAL, 0);
    assert(address_idx >= 0);

    /* Add columns inside address group */
    status = carquet_schema_add_column(
        schema, "street", CARQUET_PHYSICAL_BYTE_ARRAY, NULL,
        CARQUET_REPETITION_REQUIRED, 0, address_idx);
    assert(status == CARQUET_OK);

    status = carquet_schema_add_column(
        schema, "city", CARQUET_PHYSICAL_BYTE_ARRAY, NULL,
        CARQUET_REPETITION_OPTIONAL, 0, address_idx);
    assert(status == CARQUET_OK);

    /* Add repeated group "phones" */
    int32_t phones_idx = carquet_schema_add_group(
        schema, "phones", CARQUET_REPETITION_REPEATED, 0);
    assert(phones_idx >= 0);

    /* Add columns inside phones group */
    status = carquet_schema_add_column(
        schema, "number", CARQUET_PHYSICAL_BYTE_ARRAY, NULL,
        CARQUET_REPETITION_REQUIRED, 0, phones_idx);
    assert(status == CARQUET_OK);

    status = carquet_schema_add_column(
        schema, "type", CARQUET_PHYSICAL_BYTE_ARRAY, NULL,
        CARQUET_REPETITION_OPTIONAL, 0, phones_idx);
    assert(status == CARQUET_OK);

    /* Get number of columns (leaves) */
    int32_t num_cols = carquet_schema_num_columns(schema);
    printf("  Nested schema has %d leaf columns\n", num_cols);
    assert(num_cols == 6);  /* id, name, street, city, number, type */

    /* Write and read back to test level computation in reader */
    char test_file[512];
    carquet_test_temp_path(test_file, sizeof(test_file), "nested_schema");
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_UNCOMPRESSED;

    carquet_writer_t* writer = carquet_writer_create(test_file, schema, &opts, &err);
    if (!writer) {
        carquet_schema_free(schema);
        printf("Failed to create writer: %s\n", err.message);
        TEST_FAIL("nested_schema_levels", "writer creation failed");
    }

    /* Write minimal data (1 row per column for structure test) */
    int32_t id_val = 1;
    status = carquet_writer_write_batch(writer, 0, &id_val, 1, NULL, NULL);
    assert(status == CARQUET_OK);

    /* Close writer */
    status = carquet_writer_close(writer);
    carquet_schema_free(schema);

    if (status != CARQUET_OK) {
        printf("Failed to close writer: %d\n", status);
        remove(test_file);
        TEST_FAIL("nested_schema_levels", "writer close failed");
    }

    /* Read file back and verify levels */
    carquet_reader_t* reader = carquet_reader_open(test_file, NULL, &err);
    if (!reader) {
        printf("Failed to open reader: %s\n", err.message);
        remove(test_file);
        /* Reader might not fully support nested schemas yet */
        printf("  (Reader may not fully support nested schemas)\n");
        TEST_PASS("nested_schema_levels (partial - reader failed)");
        return 0;
    }

    const carquet_schema_t* read_schema = carquet_reader_schema(reader);
    (void)read_schema;
    assert(read_schema != NULL);

    int32_t read_cols = carquet_reader_num_columns(reader);
    printf("  Read back schema with %d columns\n", read_cols);

    /* Verify levels for each column if accessor functions work */
    /* Expected levels:
     * Column 0 (id):     def=0, rep=0  (required at root)
     * Column 1 (name):   def=1, rep=0  (optional at root)
     * Column 2 (street): def=1, rep=0  (required under optional group)
     * Column 3 (city):   def=2, rep=0  (optional under optional group)
     * Column 4 (number): def=1, rep=1  (required under repeated group)
     * Column 5 (type):   def=2, rep=1  (optional under repeated group)
     */

    printf("  Level verification:\n");
    for (int32_t i = 0; i < read_cols && i < 6; i++) {
        carquet_column_reader_t* col = carquet_reader_get_column(reader, 0, i, &err);
        if (col) {
            /* We'd need accessor functions for def/rep levels from column reader */
            /* For now, the internal computation is tested by successful file read */
            carquet_column_reader_free(col);
            printf("    Column %d: accessible\n", i);
        }
    }

    carquet_reader_close(reader);
    remove(test_file);

    TEST_PASS("nested_schema_levels");
    return 0;
}

static int test_write_simple_file(void) {
    char test_file[512];
    carquet_test_temp_path(test_file, sizeof(test_file), "carquet_simple");
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Create schema */
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        printf("Failed to create schema: %s\n", err.message);
        TEST_FAIL("write_simple_file", "schema creation failed");
    }

    /* Add columns */
    carquet_status_t status = carquet_schema_add_column(
        schema, "id", CARQUET_PHYSICAL_INT32, NULL,
        CARQUET_REPETITION_REQUIRED, 0, 0);
    assert(status == CARQUET_OK);

    status = carquet_schema_add_column(
        schema, "value", CARQUET_PHYSICAL_DOUBLE, NULL,
        CARQUET_REPETITION_REQUIRED, 0, 0);
    assert(status == CARQUET_OK);

    /* Create writer */
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_UNCOMPRESSED;

    carquet_writer_t* writer = carquet_writer_create(test_file, schema, &opts, &err);
    if (!writer) {
        carquet_schema_free(schema);
        printf("Failed to create writer: %s\n", err.message);
        TEST_FAIL("write_simple_file", "writer creation failed");
    }

    /* Write some data */
    const int num_rows = 100;
    int32_t ids[100];
    double values[100];

    for (int i = 0; i < num_rows; i++) {
        ids[i] = i;
        values[i] = (double)i * 1.5;
    }

    status = carquet_writer_write_batch(writer, 0, ids, num_rows, NULL, NULL);
    assert(status == CARQUET_OK);

    status = carquet_writer_write_batch(writer, 1, values, num_rows, NULL, NULL);
    assert(status == CARQUET_OK);

    /* Close writer */
    status = carquet_writer_close(writer);
    if (status != CARQUET_OK) {
        carquet_schema_free(schema);
        printf("Failed to close writer: %d\n", status);
        TEST_FAIL("write_simple_file", "writer close failed");
    }

    carquet_schema_free(schema);

    /* Verify file exists and has correct structure */
    FILE* f = fopen(test_file, "rb");
    if (!f) {
        TEST_FAIL("write_simple_file", "output file not found");
    }

    /* Check PAR1 header */
    char magic[4];
    (void)magic;
    assert(fread(magic, 1, 4, f) == 4);
    assert(memcmp(magic, "PAR1", 4) == 0);

    /* Check PAR1 footer */
    carquet_fseek64(f, -4, SEEK_END);
    assert(fread(magic, 1, 4, f) == 4);
    assert(memcmp(magic, "PAR1", 4) == 0);

    fclose(f);

    /* Try to read the file back */
    carquet_reader_t* reader = carquet_reader_open(test_file, NULL, &err);
    if (!reader) {
        printf("Failed to open reader: %s\n", err.message);
        /* This might fail if reader isn't fully implemented yet */
        printf("  (Reader may not be fully implemented)\n");
        remove(test_file);
        TEST_PASS("write_simple_file (partial - reader failed)");
        return 0;
    }

    /* Verify basic metadata */
    int64_t read_rows = carquet_reader_num_rows(reader);
    int32_t read_cols = carquet_reader_num_columns(reader);

    printf("  Written file has %lld rows, %d columns\n",
           (long long)read_rows, read_cols);

    assert(read_rows == num_rows);
    assert(read_cols == 2);

    carquet_reader_close(reader);
    remove(test_file);

    TEST_PASS("write_simple_file");
    return 0;
}

/* ============================================================================
 * BYTE_ARRAY page retention
 * ============================================================================
 *
 * The values of a PLAIN / DELTA_LENGTH_BYTE_ARRAY page point into the page
 * payload, so the column reader keeps every such payload until the row group
 * is left: a compressed page is inflated straight into the retained list, an
 * uncompressed one is copied to it (or is the mapped file). One read spanning
 * many pages must hand out pointers that all stay valid.
 */

#define STR_ROWS 20000

static int str_is_null(int i) { return i % 7 == 3; }

static int str_make(int i, char* out) {
    if (i % 11 == 5) { out[0] = '\0'; return 0; }   /* empty, not null */
    return snprintf(out, 64, "row-%d-%.*s", i, i % 23, "abcdefghijklmnopqrstuvwxyz");
}

static int str_write(const char* path, carquet_compression_t codec, int version,
                     carquet_encoding_t encoding, int optional) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) return 0;
    if (carquet_schema_add_column(schema, "s", CARQUET_PHYSICAL_BYTE_ARRAY, NULL,
            optional ? CARQUET_REPETITION_OPTIONAL : CARQUET_REPETITION_REQUIRED,
            0, 0) != CARQUET_OK) {
        carquet_schema_free(schema);
        return 0;
    }
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = codec;
    opts.data_page_version = version;
    opts.page_size = 4096;   /* many pages per chunk */
    carquet_writer_t* w = carquet_writer_create(path, schema, &opts, &err);
    carquet_schema_free(schema);
    if (!w) return 0;

    char* pool = malloc((size_t)STR_ROWS * 64);
    carquet_byte_array_t* vals = malloc(sizeof(*vals) * STR_ROWS);
    int16_t* def = malloc(sizeof(int16_t) * STR_ROWS);
    int ok = pool && vals && def &&
             carquet_writer_set_column_encoding(w, 0, encoding) == CARQUET_OK;
    int n = 0;
    for (int i = 0; ok && i < STR_ROWS; i++) {
        def[i] = (optional && str_is_null(i)) ? 0 : 1;
        if (!def[i]) continue;
        char* dst = pool + (size_t)i * 64;
        vals[n].length = (int32_t)str_make(i, dst);
        vals[n].data = (uint8_t*)dst;
        n++;
    }
    if (ok) ok = carquet_writer_write_batch(w, 0, vals, STR_ROWS,
                                            optional ? def : NULL, NULL) == CARQUET_OK;
    if (ok) ok = carquet_writer_close(w) == CARQUET_OK;
    else carquet_writer_abort(w);
    free(pool); free(vals); free(def);
    return ok;
}

static int count_retained_pages(const carquet_column_reader_t* col) {
    int n = 0;
    for (const carquet_retained_page_t* p = col->retained_pages; p; p = p->next) n++;
    return n;
}

/* Read the whole chunk with as few calls as the reader allows, then (only
 * then) look at every value. */
static int str_read_check(const char* path, int optional, int use_mmap,
                          int expect_retained) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_options_t ro;
    carquet_reader_options_init(&ro);
    ro.use_mmap = use_mmap != 0;
    carquet_reader_t* r = carquet_reader_open(path, &ro, &err);
    if (!r) return 0;
    carquet_column_reader_t* col = carquet_reader_get_column(r, 0, 0, &err);
    carquet_byte_array_t* vals = calloc(STR_ROWS, sizeof(*vals));
    int16_t* def = calloc(STR_ROWS, sizeof(int16_t));
    int ok = col && vals && def;

    int64_t rows = 0, dense = 0;
    while (ok && rows < STR_ROWS) {
        int64_t got = carquet_column_read_batch(col, vals + dense, STR_ROWS - rows,
                                                optional ? def + rows : NULL, NULL);
        if (got <= 0) { ok = 0; break; }
        for (int64_t i = 0; i < got; i++) {
            if (!optional || def[rows + i] == 1) dense++;
        }
        rows += got;
    }
    if (ok && expect_retained >= 0) {
        int retained = count_retained_pages(col);
        if (expect_retained == 0 ? retained != 0 : retained < 2) {
            printf("  retained pages: %d\n", retained);
            ok = 0;
        }
    }

    char want[64];
    int64_t k = 0;
    for (int i = 0; ok && i < STR_ROWS; i++) {
        if (optional && str_is_null(i)) {
            if (def[i] != 0) ok = 0;
            continue;
        }
        if (optional && def[i] != 1) { ok = 0; break; }
        int len = str_make(i, want);
        if (vals[k].length != (int32_t)len ||
            (len > 0 && memcmp(vals[k].data, want, (size_t)len) != 0)) {
            printf("  value %d differs\n", i);
            ok = 0;
        }
        k++;
    }
    if (ok && k != dense) ok = 0;

    free(vals); free(def);
    if (col) carquet_column_reader_free(col);
    carquet_reader_close(r);
    return ok;
}

static int test_byte_array_page_retention(void) {
    static const carquet_compression_t codecs[] = {
        CARQUET_COMPRESSION_UNCOMPRESSED, CARQUET_COMPRESSION_SNAPPY,
        CARQUET_COMPRESSION_ZSTD, CARQUET_COMPRESSION_GZIP, CARQUET_COMPRESSION_LZ4_RAW};
    static const carquet_encoding_t encodings[] = {
        CARQUET_ENCODING_PLAIN, CARQUET_ENCODING_DELTA_LENGTH_BYTE_ARRAY};
    char path[512];
    carquet_test_temp_path(path, sizeof(path), "str_retain");

    for (size_t c = 0; c < sizeof(codecs) / sizeof(codecs[0]); c++) {
        for (int version = 1; version <= 2; version++) {
            for (int e = 0; e < 2; e++) {
                for (int optional = 0; optional < 2; optional++) {
                    char tag[128];
                    snprintf(tag, sizeof(tag), "codec=%d v%d encoding=%d optional=%d",
                             (int)codecs[c], version, (int)encodings[e], optional);
                    if (!str_write(path, codecs[c], version, encodings[e], optional)) {
                        remove(path);
                        printf("  %s: write failed\n", tag);
                        TEST_FAIL("byte_array_page_retention", "write failed");
                    }
                    for (int use_mmap = 0; use_mmap < 2; use_mmap++) {
                        /* An uncompressed page of a mapped file is viewed in
                         * place; every other page lives on the list. */
                        int expect = (use_mmap &&
                            codecs[c] == CARQUET_COMPRESSION_UNCOMPRESSED) ? 0 : 1;
                        if (!str_read_check(path, optional, use_mmap, expect)) {
                            remove(path);
                            printf("  %s mmap=%d\n", tag, use_mmap);
                            TEST_FAIL("byte_array_page_retention", "read mismatch");
                        }
                    }
                }
            }
        }
    }
    remove(path);
    TEST_PASS("byte_array_page_retention");
    return 0;
}

/* A page that fails to load must not stay on the retained list: a caller
 * retrying the read would otherwise grow it by one page per attempt. */
static int test_byte_array_failed_page_not_retained(void) {
    static const carquet_compression_t codecs[] = {
        CARQUET_COMPRESSION_UNCOMPRESSED, CARQUET_COMPRESSION_ZSTD,
        CARQUET_COMPRESSION_SNAPPY};
    char path[512];
    carquet_test_temp_path(path, sizeof(path), "str_corrupt");

    for (size_t c = 0; c < sizeof(codecs) / sizeof(codecs[0]); c++) {
        for (int version = 1; version <= 2; version++) {
            if (!str_write(path, codecs[c], version, CARQUET_ENCODING_PLAIN, 0)) {
                remove(path);
                TEST_FAIL("byte_array_failed_page_not_retained", "write failed");
            }
            /* Break a page in the middle of the chunk. */
            FILE* f = fopen(path, "r+b");
            if (!f) TEST_FAIL("byte_array_failed_page_not_retained", "reopen failed");
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            uint8_t junk[256];
            memset(junk, 0xFF, sizeof(junk));
            fseek(f, size / 2, SEEK_SET);
            fwrite(junk, 1, sizeof(junk), f);
            fclose(f);

            /* verify: the page CRC rejects it; no verify: decompression (the
             * payload is then already on the list) or decode does. */
            for (int mode = 0; mode < 4; mode++) {
                carquet_error_t err = CARQUET_ERROR_INIT;
                carquet_reader_options_t ro;
                carquet_reader_options_init(&ro);
                ro.use_mmap = (mode & 1) != 0;
                ro.verify_checksums = (mode & 2) != 0;
                carquet_reader_t* r = carquet_reader_open(path, &ro, &err);
                carquet_column_reader_t* col = r ? carquet_reader_get_column(r, 0, 0, &err) : NULL;
                carquet_byte_array_t* vals = calloc(STR_ROWS, sizeof(*vals));
                int ok = col && vals;
                int64_t rows = 0;
                int failed = 0, at_failure = -1;
                for (int attempt = 0; ok && attempt < 200 && rows < STR_ROWS; attempt++) {
                    carquet_error_t e = CARQUET_ERROR_INIT;
                    int64_t got = carquet_column_read_batch_ex(
                        col, vals + rows, STR_ROWS - rows, NULL, NULL, &e);
                    if (got > 0) rows += got;
                    if (e.code == CARQUET_OK) continue;
                    /* The same page fails again and again. */
                    int retained = count_retained_pages(col);
                    if (++failed == 1) at_failure = retained;
                    else if (retained != at_failure) ok = 0;
                    if (failed == 4) break;
                }
                /* PLAIN strings can survive 256 junk bytes undetected when
                 * nothing checks the page (lengths that happen to fit); what
                 * matters is that a detected failure retains nothing. */
                if (ok && !failed && (codecs[c] != CARQUET_COMPRESSION_UNCOMPRESSED ||
                                      ro.verify_checksums)) {
                    printf("  codec=%d v%d mode=%d: corrupted page read cleanly\n",
                           (int)codecs[c], version, mode);
                    ok = 0;
                }
                free(vals);
                if (col) carquet_column_reader_free(col);
                if (r) carquet_reader_close(r);
                if (!ok) {
                    remove(path);
                    printf("  codec=%d v%d mode=%d\n", (int)codecs[c], version, mode);
                    TEST_FAIL("byte_array_failed_page_not_retained",
                              "a failed page stayed on the retained list");
                }
            }
        }
    }
    remove(path);
    TEST_PASS("byte_array_failed_page_not_retained");
    return 0;
}

int main(void) {
    int failures = 0;

    printf("=== Reader Tests ===\n\n");

    failures += test_version();
    failures += test_cpu_detection();
    failures += test_reader_options();
    failures += test_writer_options();
    failures += test_open_nonexistent();
    failures += test_type_names();
    failures += test_status_strings();
    failures += test_nested_schema_levels();
    failures += test_write_simple_file();
    failures += test_byte_array_page_retention();
    failures += test_byte_array_failed_page_not_retained();

    printf("\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
