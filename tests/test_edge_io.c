/**
 * @file test_edge_io.c
 * @brief Edge case tests for Parquet file I/O
 *
 * Tests schema edge cases, empty files, invalid paths,
 * and various writer/reader boundary conditions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include <carquet/carquet.h>

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* Temporary file helper - portable across platforms */
static const char* get_temp_dir(void) {
#ifdef _WIN32
    static char temp_dir[256] = {0};
    if (temp_dir[0] == 0) {
        const char* tmp = getenv("TEMP");
        if (!tmp) tmp = getenv("TMP");
        if (!tmp) tmp = ".";
        snprintf(temp_dir, sizeof(temp_dir), "%s", tmp);
    }
    return temp_dir;
#else
    return "/tmp";
#endif
}

static const char* get_temp_file(const char* suffix) {
    static char path[512];
    snprintf(path, sizeof(path), "%s/carquet_test_%d_%s.parquet", get_temp_dir(), getpid(), suffix);
    return path;
}

static void cleanup_file(const char* path) {
    remove(path);
}

/* ============================================================================
 * Schema Edge Cases
 * ============================================================================
 */

static int test_schema_empty(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        TEST_FAIL("schema_empty", "Failed to create schema");
    }

    /* Empty schema should have 0 columns */
    int32_t num_cols = carquet_schema_num_columns(schema);
    if (num_cols != 0) {
        carquet_schema_free(schema);
        TEST_FAIL("schema_empty", "Empty schema should have 0 columns");
    }

    carquet_schema_free(schema);
    TEST_PASS("schema_empty");
    return 0;
}

static int test_schema_single_column(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        TEST_FAIL("schema_single_column", "Failed to create schema");
    }

    carquet_status_t status = carquet_schema_add_column(
        schema, "value", CARQUET_PHYSICAL_INT32, NULL,
        CARQUET_REPETITION_REQUIRED, 0, 0);

    if (status != CARQUET_OK) {
        carquet_schema_free(schema);
        TEST_FAIL("schema_single_column", "Failed to add column");
    }

    int32_t num_cols = carquet_schema_num_columns(schema);
    if (num_cols != 1) {
        carquet_schema_free(schema);
        TEST_FAIL("schema_single_column", "Should have 1 column");
    }

    carquet_schema_free(schema);
    TEST_PASS("schema_single_column");
    return 0;
}

static int test_schema_all_types(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        TEST_FAIL("schema_all_types", "Failed to create schema");
    }

    carquet_status_t status;

    /* Add one column of each physical type */
    status = carquet_schema_add_column(schema, "bool_col",
        CARQUET_PHYSICAL_BOOLEAN, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    if (status != CARQUET_OK) goto fail;

    status = carquet_schema_add_column(schema, "int32_col",
        CARQUET_PHYSICAL_INT32, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    if (status != CARQUET_OK) goto fail;

    status = carquet_schema_add_column(schema, "int64_col",
        CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    if (status != CARQUET_OK) goto fail;

    status = carquet_schema_add_column(schema, "float_col",
        CARQUET_PHYSICAL_FLOAT, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    if (status != CARQUET_OK) goto fail;

    status = carquet_schema_add_column(schema, "double_col",
        CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    if (status != CARQUET_OK) goto fail;

    status = carquet_schema_add_column(schema, "binary_col",
        CARQUET_PHYSICAL_BYTE_ARRAY, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    if (status != CARQUET_OK) goto fail;

    int32_t num_cols = carquet_schema_num_columns(schema);
    if (num_cols != 6) {
        carquet_schema_free(schema);
        TEST_FAIL("schema_all_types", "Should have 6 columns");
    }

    carquet_schema_free(schema);
    TEST_PASS("schema_all_types");
    return 0;

fail:
    carquet_schema_free(schema);
    TEST_FAIL("schema_all_types", "Failed to add column");
    return 1;
}

static int test_schema_repetition_types(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        TEST_FAIL("schema_repetition_types", "Failed to create schema");
    }

    carquet_status_t status;

    /* REQUIRED - must be present */
    status = carquet_schema_add_column(schema, "required_col",
        CARQUET_PHYSICAL_INT32, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    if (status != CARQUET_OK) goto fail;

    /* OPTIONAL - may be null */
    status = carquet_schema_add_column(schema, "optional_col",
        CARQUET_PHYSICAL_INT32, NULL, CARQUET_REPETITION_OPTIONAL, 0, 0);
    if (status != CARQUET_OK) goto fail;

    /* REPEATED - array of values */
    status = carquet_schema_add_column(schema, "repeated_col",
        CARQUET_PHYSICAL_INT32, NULL, CARQUET_REPETITION_REPEATED, 0, 0);
    if (status != CARQUET_OK) goto fail;

    carquet_schema_free(schema);
    TEST_PASS("schema_repetition_types");
    return 0;

fail:
    carquet_schema_free(schema);
    TEST_FAIL("schema_repetition_types", "Failed to add column");
    return 1;
}

static int test_schema_logical_type_validation(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        TEST_FAIL("schema_logical_type_validation", "Failed to create schema");
    }

    carquet_logical_type_t string_type = { .id = CARQUET_LOGICAL_STRING };
    carquet_logical_type_t date_type = { .id = CARQUET_LOGICAL_DATE };
    carquet_logical_type_t uuid_type = { .id = CARQUET_LOGICAL_UUID };
    carquet_logical_type_t float16_type = { .id = CARQUET_LOGICAL_FLOAT16 };
    carquet_logical_type_t geometry_type = { .id = CARQUET_LOGICAL_GEOMETRY };
    carquet_logical_type_t geography_type = { .id = CARQUET_LOGICAL_GEOGRAPHY };
    carquet_logical_type_t variant_type = { .id = CARQUET_LOGICAL_VARIANT };
    carquet_logical_type_t decimal_bad_scale = {
        .id = CARQUET_LOGICAL_DECIMAL,
        .params.decimal = { .precision = 4, .scale = 5 }
    };
    carquet_logical_type_t decimal_i32_too_wide = {
        .id = CARQUET_LOGICAL_DECIMAL,
        .params.decimal = { .precision = 10, .scale = 0 }
    };
    carquet_logical_type_t int16_type = {
        .id = CARQUET_LOGICAL_INTEGER,
        .params.integer = { .bit_width = 16, .is_signed = true }
    };
    carquet_logical_type_t bad_int_width = {
        .id = CARQUET_LOGICAL_INTEGER,
        .params.integer = { .bit_width = 24, .is_signed = true }
    };
    carquet_logical_type_t time_millis = {
        .id = CARQUET_LOGICAL_TIME,
        .params.time = { .unit = CARQUET_TIME_UNIT_MILLIS, .is_adjusted_to_utc = true }
    };
    carquet_logical_type_t time_micros = {
        .id = CARQUET_LOGICAL_TIME,
        .params.time = { .unit = CARQUET_TIME_UNIT_MICROS, .is_adjusted_to_utc = true }
    };

    if (carquet_schema_add_column(schema, "string_ok", CARQUET_PHYSICAL_BYTE_ARRAY,
            &string_type, CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(schema, "date_ok", CARQUET_PHYSICAL_INT32,
            &date_type, CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(schema, "uuid_ok", CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY,
            &uuid_type, CARQUET_REPETITION_REQUIRED, 16, 0) != CARQUET_OK ||
        carquet_schema_add_column(schema, "float16_ok", CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY,
            &float16_type, CARQUET_REPETITION_REQUIRED, 2, 0) != CARQUET_OK ||
        carquet_schema_add_column(schema, "int16_ok", CARQUET_PHYSICAL_INT32,
            &int16_type, CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(schema, "time_millis_ok", CARQUET_PHYSICAL_INT32,
            &time_millis, CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(schema, "time_micros_ok", CARQUET_PHYSICAL_INT64,
            &time_micros, CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(schema, "geometry_ok", CARQUET_PHYSICAL_BYTE_ARRAY,
            &geometry_type, CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(schema, "geography_ok", CARQUET_PHYSICAL_BYTE_ARRAY,
            &geography_type, CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK ||
        carquet_schema_add_variant(schema, "variant_ok", CARQUET_REPETITION_OPTIONAL, 0) < 0) {
        carquet_schema_free(schema);
        TEST_FAIL("schema_logical_type_validation", "valid logical type rejected");
    }

    if (carquet_schema_add_column(schema, "bad_string", CARQUET_PHYSICAL_INT32,
            &string_type, CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK ||
        carquet_schema_add_column(schema, "bad_uuid", CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY,
            &uuid_type, CARQUET_REPETITION_REQUIRED, 15, 0) == CARQUET_OK ||
        carquet_schema_add_column(schema, "bad_float16", CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY,
            &float16_type, CARQUET_REPETITION_REQUIRED, 4, 0) == CARQUET_OK ||
        carquet_schema_add_column(schema, "bad_decimal_scale", CARQUET_PHYSICAL_INT32,
            &decimal_bad_scale, CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK ||
        carquet_schema_add_column(schema, "bad_decimal_i32", CARQUET_PHYSICAL_INT32,
            &decimal_i32_too_wide, CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK ||
        carquet_schema_add_column(schema, "bad_int_width", CARQUET_PHYSICAL_INT32,
            &bad_int_width, CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK ||
        carquet_schema_add_column(schema, "bad_time_millis", CARQUET_PHYSICAL_INT64,
            &time_millis, CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK ||
        carquet_schema_add_column(schema, "bad_time_micros", CARQUET_PHYSICAL_INT32,
            &time_micros, CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK ||
        carquet_schema_add_column(schema, "bad_geometry", CARQUET_PHYSICAL_INT32,
            &geometry_type, CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK ||
        carquet_schema_add_column(schema, "bad_geography", CARQUET_PHYSICAL_INT32,
            &geography_type, CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK ||
        carquet_schema_add_column(schema, "bad_variant", CARQUET_PHYSICAL_BYTE_ARRAY,
            &variant_type, CARQUET_REPETITION_REQUIRED, 0, 0) == CARQUET_OK) {
        carquet_schema_free(schema);
        TEST_FAIL("schema_logical_type_validation", "invalid logical type accepted");
    }

    carquet_schema_free(schema);
    TEST_PASS("schema_logical_type_validation");
    return 0;
}

/* ============================================================================
 * Reader Edge Cases
 * ============================================================================
 */

static int test_reader_empty_path(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Passing empty path */
    carquet_reader_t* reader = carquet_reader_open("", NULL, &err);
    if (reader != NULL) {
        carquet_reader_close(reader);
    }

    printf("  [DEBUG] Empty path handled: code=%d\n", err.code);
    TEST_PASS("reader_empty_path");
    return 0;
}

static int test_reader_nonexistent(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_reader_t* reader = carquet_reader_open(
        "/nonexistent/path/file.parquet", NULL, &err);
    if (reader != NULL) {
        carquet_reader_close(reader);
        TEST_FAIL("reader_nonexistent", "Should fail with nonexistent path");
    }

    printf("  [DEBUG] Nonexistent path error: code=%d\n", err.code);
    TEST_PASS("reader_nonexistent");
    return 0;
}

static int test_reader_invalid_parquet(void) {
    /* Create a file with invalid content */
    const char* path = get_temp_file("invalid");
    FILE* f = fopen(path, "wb");
    if (!f) {
        TEST_FAIL("reader_invalid_parquet", "Could not create temp file");
    }
    fprintf(f, "This is not a parquet file!");
    fclose(f);

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* reader = carquet_reader_open(path, NULL, &err);
    if (reader != NULL) {
        carquet_reader_close(reader);
        cleanup_file(path);
        TEST_FAIL("reader_invalid_parquet", "Should fail with invalid file");
    }

    printf("  [DEBUG] Invalid parquet error: code=%d, msg=%s\n", err.code, err.message);
    cleanup_file(path);
    TEST_PASS("reader_invalid_parquet");
    return 0;
}

static int test_reader_truncated_file(void) {
    /* Create a file with just PAR1 magic */
    const char* path = get_temp_file("truncated");
    FILE* f = fopen(path, "wb");
    if (!f) {
        TEST_FAIL("reader_truncated_file", "Could not create temp file");
    }
    fwrite("PAR1", 1, 4, f);  /* Just the magic, truncated */
    fclose(f);

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* reader = carquet_reader_open(path, NULL, &err);
    if (reader != NULL) {
        carquet_reader_close(reader);
        cleanup_file(path);
        TEST_FAIL("reader_truncated_file", "Should fail with truncated file");
    }

    printf("  [DEBUG] Truncated file error: code=%d\n", err.code);
    cleanup_file(path);
    TEST_PASS("reader_truncated_file");
    return 0;
}

/* ============================================================================
 * Writer Edge Cases
 * ============================================================================
 */

/* NOTE: test_writer_null_schema removed - carquet_writer_create requires non-NULL
 * schema per API contract (CARQUET_NONNULL). Passing NULL is undefined behavior.
 */

static int test_writer_invalid_path(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* schema = carquet_schema_create(&err);
    carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_INT32, NULL,
                              CARQUET_REPETITION_REQUIRED, 0, 0);

    /* Try to write to non-existent directory */
    carquet_writer_t* writer = carquet_writer_create(
        "/nonexistent/directory/file.parquet", schema, NULL, &err);

    if (writer != NULL) {
        carquet_writer_close(writer);
        carquet_schema_free(schema);
        TEST_FAIL("writer_invalid_path", "Should fail with invalid path");
    }

    printf("  [DEBUG] Invalid path error: code=%d\n", err.code);
    carquet_schema_free(schema);
    TEST_PASS("writer_invalid_path");
    return 0;
}

/* ============================================================================
 * Roundtrip Edge Cases
 * ============================================================================
 */

static int test_roundtrip_single_row(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    const char* path = get_temp_file("single_row");

    /* Create schema */
    carquet_schema_t* schema = carquet_schema_create(&err);
    carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_INT32, NULL,
                              CARQUET_REPETITION_REQUIRED, 0, 0);

    /* Write single row */
    carquet_writer_t* writer = carquet_writer_create(path, schema, NULL, &err);
    if (!writer) {
        carquet_schema_free(schema);
        cleanup_file(path);
        TEST_FAIL("roundtrip_single_row", "Failed to open writer");
    }

    int32_t value = 42;
    carquet_status_t status = carquet_writer_write_batch(writer, 0, &value, 1, NULL, NULL);
    if (status != CARQUET_OK) {
        carquet_writer_close(writer);
        carquet_schema_free(schema);
        cleanup_file(path);
        TEST_FAIL("roundtrip_single_row", "Failed to write batch");
    }

    carquet_writer_close(writer);
    carquet_schema_free(schema);

    /* Read back */
    carquet_reader_t* reader = carquet_reader_open(path, NULL, &err);
    if (!reader) {
        cleanup_file(path);
        TEST_FAIL("roundtrip_single_row", "Failed to open reader");
    }

    int64_t num_rows = carquet_reader_num_rows(reader);
    if (num_rows != 1) {
        carquet_reader_close(reader);
        cleanup_file(path);
        printf("  [DEBUG] Expected 1 row, got %lld\n", (long long)num_rows);
        TEST_FAIL("roundtrip_single_row", "Wrong row count");
    }

    carquet_reader_close(reader);
    cleanup_file(path);
    TEST_PASS("roundtrip_single_row");
    return 0;
}

/* Write `rows` INT64 rows (value i * 3) uncompressed to `path`. */
static int write_int64_file(const char* path, int64_t rows) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = carquet_schema_create(&err);
    carquet_schema_add_column(schema, "v", CARQUET_PHYSICAL_INT64, NULL,
                              CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_writer_t* writer = carquet_writer_create(path, schema, NULL, &err);
    if (!writer) {
        carquet_schema_free(schema);
        return 1;
    }
    int64_t* values = malloc((size_t)rows * sizeof(int64_t));
    for (int64_t i = 0; i < rows; i++) values[i] = i * 3;
    carquet_status_t status = carquet_writer_write_batch(writer, 0, values, rows, NULL, NULL);
    free(values);
    if (carquet_writer_close(writer) != CARQUET_OK) status = CARQUET_ERROR_FILE_WRITE;
    carquet_schema_free(schema);
    return status == CARQUET_OK ? 0 : 1;
}

static uint8_t* read_whole_file(const char* path, long* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* data = malloc(*size > 0 ? (size_t)*size : 1);
    if (fread(data, 1, (size_t)*size, f) != (size_t)*size) {
        free(data);
        data = NULL;
    }
    fclose(f);
    return data;
}

/* A writer created over an existing file replaces it: the result is byte for
 * byte the file a fresh path gets, whether the old file was larger (large
 * enough to be written over in place, its tail cut at close) or smaller, and
 * an aborted writer leaves no file. */
static int test_overwrite_existing_file(void) {
    char path[512], fresh[512];
    snprintf(path, sizeof(path), "%s", get_temp_file("overwrite"));
    snprintf(fresh, sizeof(fresh), "%s", get_temp_file("overwrite_fresh"));
    static const int64_t sizes[][2] = {
        {2000000, 1000},      /* 16 MB file replaced by a small one */
        {2000000, 1500000},   /* ... by one that ends inside it */
        {1000, 2000000},      /* small file replaced by a large one */
        {1000000, 2000000},   /* large file replaced by a larger one */
    };
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++) {
        cleanup_file(path);
        cleanup_file(fresh);
        if (write_int64_file(path, sizes[k][0]) || write_int64_file(path, sizes[k][1]) ||
            write_int64_file(fresh, sizes[k][1])) {
            cleanup_file(path);
            cleanup_file(fresh);
            TEST_FAIL("overwrite_existing_file", "write failed");
        }
        long size_a = 0, size_b = 0;
        uint8_t* a = read_whole_file(path, &size_a);
        uint8_t* b = read_whole_file(fresh, &size_b);
        int same = a && b && size_a == size_b && memcmp(a, b, (size_t)size_a) == 0;
        free(a);
        free(b);
        if (!same) {
            cleanup_file(path);
            cleanup_file(fresh);
            printf("  [DEBUG] case %zu: %ld vs %ld bytes\n", k, size_a, size_b);
            TEST_FAIL("overwrite_existing_file", "overwritten file differs from a fresh one");
        }
        carquet_error_t err = CARQUET_ERROR_INIT;
        carquet_reader_t* reader = carquet_reader_open(path, NULL, &err);
        int64_t rows = reader ? carquet_reader_num_rows(reader) : -1;
        if (reader) carquet_reader_close(reader);
        if (rows != sizes[k][1]) {
            cleanup_file(path);
            cleanup_file(fresh);
            TEST_FAIL("overwrite_existing_file", "wrong row count after overwrite");
        }
    }

    /* Abort over a large existing file: the file is removed, as before. */
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = carquet_schema_create(&err);
    carquet_schema_add_column(schema, "v", CARQUET_PHYSICAL_INT64, NULL,
                              CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_writer_t* writer = carquet_writer_create(path, schema, NULL, &err);
    if (writer) carquet_writer_abort(writer);
    carquet_schema_free(schema);
    FILE* gone = fopen(path, "rb");
    if (gone) fclose(gone);
    cleanup_file(path);
    cleanup_file(fresh);
    if (!writer || gone) {
        TEST_FAIL("overwrite_existing_file", "aborted writer left the file");
    }
    TEST_PASS("overwrite_existing_file");
    return 0;
}

/* ============================================================================
 * Options Edge Cases
 * ============================================================================
 */

static int test_reader_options_defaults(void) {
    carquet_reader_options_t opts;
    carquet_reader_options_init(&opts);

    if (opts.verify_checksums != true) {
        TEST_FAIL("reader_options_defaults", "verify_checksums should default to true");
    }

    if (opts.use_mmap != false) {
        TEST_FAIL("reader_options_defaults", "use_mmap should default to false");
    }

    TEST_PASS("reader_options_defaults");
    return 0;
}

static int test_writer_options_defaults(void) {
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);

    if (opts.compression != CARQUET_COMPRESSION_UNCOMPRESSED) {
        TEST_FAIL("writer_options_defaults", "compression should default to UNCOMPRESSED");
    }

    TEST_PASS("writer_options_defaults");
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================
 */

int main(void) {
    printf("=== I/O Edge Case Tests ===\n\n");

    /* Initialize carquet */
    carquet_init();

    int failures = 0;

    printf("--- Schema Edge Cases ---\n");
    failures += test_schema_empty();
    failures += test_schema_single_column();
    failures += test_schema_all_types();
    failures += test_schema_repetition_types();
    failures += test_schema_logical_type_validation();

    printf("\n--- Reader Edge Cases ---\n");
    failures += test_reader_empty_path();
    failures += test_reader_nonexistent();
    failures += test_reader_invalid_parquet();
    failures += test_reader_truncated_file();

    printf("\n--- Writer Edge Cases ---\n");
    failures += test_writer_invalid_path();

    printf("\n--- Roundtrip Edge Cases ---\n");
    failures += test_roundtrip_single_row();
    failures += test_overwrite_existing_file();

    printf("\n--- Options Edge Cases ---\n");
    failures += test_reader_options_defaults();
    failures += test_writer_options_defaults();

    printf("\n");
    if (failures == 0) {
        printf("All I/O edge case tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed.\n", failures);
        return 1;
    }
}
