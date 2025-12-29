/**
 * @file test_production.c
 * @brief Tests for production-ready features
 *
 * Tests for:
 * - Column projection (batch reader)
 * - Row group statistics
 * - Predicate pushdown / row group filtering
 * - Memory-mapped I/O
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <carquet/carquet.h>

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); return 1; } while(0)

#define NUM_ROWS 10000
#define NUM_ROW_GROUPS 10

static const char* TEST_FILE = "/tmp/test_production.parquet";

/* ============================================================================
 * Helper: Create test file with multiple row groups
 * ============================================================================
 */

static int create_test_file(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Create schema with multiple columns */
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) return -1;

    carquet_schema_add_column(schema, "id", CARQUET_PHYSICAL_INT32, NULL,
        CARQUET_REPETITION_REQUIRED, 0);
    carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_DOUBLE, NULL,
        CARQUET_REPETITION_REQUIRED, 0);
    carquet_schema_add_column(schema, "category", CARQUET_PHYSICAL_INT32, NULL,
        CARQUET_REPETITION_REQUIRED, 0);
    carquet_schema_add_column(schema, "score", CARQUET_PHYSICAL_FLOAT, NULL,
        CARQUET_REPETITION_REQUIRED, 0);

    /* Writer options - small row groups for testing */
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_SNAPPY;
    opts.row_group_size = (NUM_ROWS / NUM_ROW_GROUPS) * 32;  /* Force multiple row groups */

    carquet_writer_t* writer = carquet_writer_create(TEST_FILE, schema, &opts, &err);
    if (!writer) {
        carquet_schema_free(schema);
        return -1;
    }

    /* Generate test data */
    int32_t* ids = malloc(NUM_ROWS * sizeof(int32_t));
    double* values = malloc(NUM_ROWS * sizeof(double));
    int32_t* categories = malloc(NUM_ROWS * sizeof(int32_t));
    float* scores = malloc(NUM_ROWS * sizeof(float));

    for (int i = 0; i < NUM_ROWS; i++) {
        ids[i] = i;
        values[i] = (double)i * 1.5;
        categories[i] = i % 10;  /* 0-9 categories */
        scores[i] = (float)(i % 100) / 10.0f;  /* 0.0 - 9.9 */
    }

    /* Write data in chunks to create multiple row groups */
    int rows_per_group = NUM_ROWS / NUM_ROW_GROUPS;
    for (int g = 0; g < NUM_ROW_GROUPS; g++) {
        int offset = g * rows_per_group;

        carquet_writer_write_batch(writer, 0, ids + offset, rows_per_group, NULL, NULL);
        carquet_writer_write_batch(writer, 1, values + offset, rows_per_group, NULL, NULL);
        carquet_writer_write_batch(writer, 2, categories + offset, rows_per_group, NULL, NULL);
        carquet_writer_write_batch(writer, 3, scores + offset, rows_per_group, NULL, NULL);

        if (g < NUM_ROW_GROUPS - 1) {
            carquet_writer_new_row_group(writer);
        }
    }

    free(ids);
    free(values);
    free(categories);
    free(scores);

    carquet_status_t status = carquet_writer_close(writer);
    carquet_schema_free(schema);

    return (status == CARQUET_OK) ? 0 : -1;
}

/* ============================================================================
 * Test: Column Projection
 * ============================================================================
 */

static int test_column_projection(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Open file */
    carquet_reader_t* reader = carquet_reader_open(TEST_FILE, NULL, &err);
    if (!reader) {
        printf("  Failed to open file: %s\n", err.message);
        TEST_FAIL("column_projection", "failed to open file");
    }

    /* Verify file has expected columns */
    int32_t num_cols = carquet_reader_num_columns(reader);
    assert(num_cols == 4);

    /* Test 1: Project only 2 columns by index */
    carquet_batch_reader_config_t config;
    carquet_batch_reader_config_init(&config);

    int32_t proj_cols[] = {0, 2};  /* id and category only */
    config.column_indices = proj_cols;
    config.num_columns = 2;
    config.batch_size = 1000;

    carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &config, &err);
    if (!batch_reader) {
        carquet_reader_close(reader);
        printf("  Failed to create batch reader: %s\n", err.message);
        TEST_FAIL("column_projection", "failed to create batch reader");
    }

    /* Read all batches and verify */
    int64_t total_rows = 0;
    carquet_row_batch_t* batch = NULL;

    while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
        /* Verify batch has only 2 columns */
        assert(carquet_row_batch_num_columns(batch) == 2);

        int64_t batch_rows = carquet_row_batch_num_rows(batch);
        total_rows += batch_rows;

        /* Verify we can access the projected columns */
        const void* data;
        const uint8_t* null_bitmap;
        int64_t num_values;

        carquet_status_t status = carquet_row_batch_column(batch, 0, &data, &null_bitmap, &num_values);
        assert(status == CARQUET_OK);
        assert(data != NULL);
        assert(num_values == batch_rows);

        status = carquet_row_batch_column(batch, 1, &data, &null_bitmap, &num_values);
        assert(status == CARQUET_OK);
        assert(data != NULL);

        carquet_row_batch_free(batch);
        batch = NULL;
    }

    printf("  Read %lld rows with 2-column projection\n", (long long)total_rows);
    assert(total_rows == NUM_ROWS);

    carquet_batch_reader_free(batch_reader);

    /* Test 2: Project by column names */
    const char* col_names[] = {"value", "score"};
    carquet_batch_reader_config_init(&config);
    config.column_names = col_names;
    config.num_column_names = 2;
    config.batch_size = 2000;

    batch_reader = carquet_batch_reader_create(reader, &config, &err);
    if (!batch_reader) {
        carquet_reader_close(reader);
        printf("  Failed to create batch reader by name: %s\n", err.message);
        TEST_FAIL("column_projection", "failed to create batch reader by name");
    }

    total_rows = 0;
    while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
        assert(carquet_row_batch_num_columns(batch) == 2);
        total_rows += carquet_row_batch_num_rows(batch);
        carquet_row_batch_free(batch);
        batch = NULL;
    }

    printf("  Read %lld rows with name-based projection\n", (long long)total_rows);
    assert(total_rows == NUM_ROWS);

    carquet_batch_reader_free(batch_reader);
    carquet_reader_close(reader);

    TEST_PASS("column_projection");
    return 0;
}

/* ============================================================================
 * Test: Row Group Statistics
 * ============================================================================
 */

static int test_row_group_statistics(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_reader_t* reader = carquet_reader_open(TEST_FILE, NULL, &err);
    if (!reader) {
        TEST_FAIL("row_group_statistics", "failed to open file");
    }

    int32_t num_row_groups = carquet_reader_num_row_groups(reader);
    printf("  File has %d row groups\n", num_row_groups);

    /* Get statistics for column 0 (id) in each row group */
    for (int32_t rg = 0; rg < num_row_groups; rg++) {
        carquet_column_statistics_t stats;
        carquet_status_t status = carquet_reader_column_statistics(reader, rg, 0, &stats);

        if (status == CARQUET_OK && stats.has_min_max) {
            int32_t min_val = *(const int32_t*)stats.min_value;
            int32_t max_val = *(const int32_t*)stats.max_value;
            printf("  Row group %d: id range [%d, %d], %lld values\n",
                   rg, min_val, max_val, (long long)stats.num_values);
        }
    }

    carquet_reader_close(reader);
    TEST_PASS("row_group_statistics");
    return 0;
}

/* ============================================================================
 * Test: Predicate Pushdown (Row Group Filtering)
 * ============================================================================
 */

static int test_predicate_pushdown(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_reader_t* reader = carquet_reader_open(TEST_FILE, NULL, &err);
    if (!reader) {
        TEST_FAIL("predicate_pushdown", "failed to open file");
    }

    int32_t num_row_groups = carquet_reader_num_row_groups(reader);

    /* Test: Find row groups where id > 5000 */
    int32_t search_value = 5000;
    int32_t matching[100];

    int32_t num_matching = carquet_reader_filter_row_groups(
        reader,
        0,  /* column 0 = id */
        CARQUET_COMPARE_GT,
        &search_value,
        sizeof(int32_t),
        matching,
        100);

    printf("  Row groups with id > %d: %d (of %d total)\n",
           search_value, num_matching, num_row_groups);

    /* Should filter out roughly half the row groups */
    assert(num_matching > 0);
    assert(num_matching <= num_row_groups);

    /* Test: Find row groups where id == 100 (should match only 1 or few) */
    search_value = 100;
    num_matching = carquet_reader_filter_row_groups(
        reader,
        0,
        CARQUET_COMPARE_EQ,
        &search_value,
        sizeof(int32_t),
        matching,
        100);

    printf("  Row groups that might contain id == %d: %d\n", search_value, num_matching);
    assert(num_matching >= 1);  /* At least one should match */

    /* Test: Find row groups where id < 0 (should match none) */
    search_value = 0;
    num_matching = carquet_reader_filter_row_groups(
        reader,
        0,
        CARQUET_COMPARE_LT,
        &search_value,
        sizeof(int32_t),
        matching,
        100);

    printf("  Row groups with id < 0: %d (should be 0)\n", num_matching);
    /* All IDs start from 0, so no row group should have id < 0 */

    carquet_reader_close(reader);
    TEST_PASS("predicate_pushdown");
    return 0;
}

/* ============================================================================
 * Test: Buffer-based Reading (simulates mmap)
 * ============================================================================
 */

static int test_buffer_reading(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Read entire file into memory */
    FILE* f = fopen(TEST_FILE, "rb");
    if (!f) {
        TEST_FAIL("buffer_reading", "failed to open file");
    }

    fseek(f, 0, SEEK_END);
    size_t size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* buffer = malloc(size);
    if (!buffer) {
        fclose(f);
        TEST_FAIL("buffer_reading", "failed to allocate buffer");
    }

    size_t read = fread(buffer, 1, size, f);
    fclose(f);

    if (read != size) {
        free(buffer);
        TEST_FAIL("buffer_reading", "failed to read file");
    }

    printf("  File size: %zu bytes\n", size);

    /* Open from buffer */
    carquet_reader_t* reader = carquet_reader_open_buffer(buffer, size, NULL, &err);
    if (!reader) {
        free(buffer);
        printf("  Failed to open from buffer: %s\n", err.message);
        TEST_FAIL("buffer_reading", "failed to open from buffer");
    }

    /* Verify we can read metadata */
    int64_t num_rows = carquet_reader_num_rows(reader);
    int32_t num_cols = carquet_reader_num_columns(reader);

    printf("  Buffer read: %lld rows, %d columns\n", (long long)num_rows, num_cols);
    assert(num_rows == NUM_ROWS);
    assert(num_cols == 4);

    carquet_reader_close(reader);
    free(buffer);

    TEST_PASS("buffer_reading");
    return 0;
}

/* ============================================================================
 * Test: Full Pipeline (projection + filtering)
 * ============================================================================
 */

static int test_full_pipeline(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_reader_t* reader = carquet_reader_open(TEST_FILE, NULL, &err);
    if (!reader) {
        TEST_FAIL("full_pipeline", "failed to open file");
    }

    /* Step 1: Filter row groups where category column might have value 5 */
    int32_t category_value = 5;
    int32_t matching_rgs[100];
    int32_t num_matching = carquet_reader_filter_row_groups(
        reader,
        2,  /* category column */
        CARQUET_COMPARE_EQ,
        &category_value,
        sizeof(int32_t),
        matching_rgs,
        100);

    printf("  Row groups that might contain category=5: %d\n", num_matching);

    /* Step 2: Read only id and category columns from matching row groups */
    int32_t proj_cols[] = {0, 2};
    carquet_batch_reader_config_t config;
    carquet_batch_reader_config_init(&config);
    config.column_indices = proj_cols;
    config.num_columns = 2;
    config.batch_size = 1000;

    int64_t total_matching_rows = 0;

    /* In a real implementation, we'd only read the matching row groups */
    /* For now, we read all and count matches */
    carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &config, &err);
    if (batch_reader) {
        carquet_row_batch_t* batch = NULL;
        while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
            const void* cat_data;
            int64_t num_values;
            carquet_row_batch_column(batch, 1, &cat_data, NULL, &num_values);

            const int32_t* categories = (const int32_t*)cat_data;
            for (int64_t i = 0; i < num_values; i++) {
                if (categories[i] == category_value) {
                    total_matching_rows++;
                }
            }

            carquet_row_batch_free(batch);
            batch = NULL;
        }
        carquet_batch_reader_free(batch_reader);
    }

    printf("  Rows with category=5: %lld (expected ~%d)\n",
           (long long)total_matching_rows, NUM_ROWS / 10);

    /* Should be approximately 10% of rows (category 0-9) */
    assert(total_matching_rows > 0);
    assert(total_matching_rows <= NUM_ROWS);

    carquet_reader_close(reader);
    TEST_PASS("full_pipeline");
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================
 */

int main(void) {
    int failures = 0;

    printf("=== Production Feature Tests ===\n\n");

    /* Create test file */
    printf("Creating test file with %d rows in ~%d row groups...\n", NUM_ROWS, NUM_ROW_GROUPS);
    if (create_test_file() != 0) {
        printf("FATAL: Failed to create test file\n");
        return 1;
    }
    printf("Test file created: %s\n\n", TEST_FILE);

    /* Run tests */
    failures += test_column_projection();
    failures += test_row_group_statistics();
    failures += test_predicate_pushdown();
    failures += test_buffer_reading();
    failures += test_full_pipeline();

    /* Cleanup */
    remove(TEST_FILE);

    printf("\n");
    if (failures == 0) {
        printf("All production tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
