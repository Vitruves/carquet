/**
 * @file roundtrip_comprehensive.c
 * @brief Comprehensive roundtrip test covering all types and compressions
 *
 * Tests: All physical types, all compressions, nullability, edge cases
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <carquet/carquet.h>

#define NUM_ROWS 5000
#define NUM_COMPRESSIONS 5

static const char* COMPRESSION_NAMES[] = {
    "uncompressed", "snappy", "gzip", "lz4_raw", "zstd"
};

/* Use LZ4_RAW (codec 7) for Parquet - uses LZ4 block format */
static carquet_compression_t COMPRESSIONS[] = {
    CARQUET_COMPRESSION_UNCOMPRESSED,
    CARQUET_COMPRESSION_SNAPPY,
    CARQUET_COMPRESSION_GZIP,
    CARQUET_COMPRESSION_LZ4_RAW,
    CARQUET_COMPRESSION_ZSTD
};

/* Test data generation (sparse encoding: values arrays contain only non-null values) */
static void generate_test_data(
    uint8_t* bools, int32_t* int32s, int64_t* int64s,
    float* floats, double* doubles,
    carquet_byte_array_t* strings, int16_t* string_def_levels,
    int32_t* nullable_ints, int16_t* nullable_def_levels,
    int n
) {
    const char* sample_strings[] = {
        "hello", "world", "carquet", "parquet", "test",
        "alpha", "beta", "gamma", "delta", "epsilon"
    };

    int string_value_count = 0;
    int nullable_int_count = 0;

    for (int i = 0; i < n; i++) {
        bools[i] = (i % 2 == 0) ? 1 : 0;
        int32s[i] = i * 10 - 5000;  /* Negative and positive */
        int64s[i] = (int64_t)i * 1000000LL - 2500000000LL;
        floats[i] = (float)i * 0.5f - 1250.0f;
        doubles[i] = (double)i * 0.125 - 312.5;

        /* Strings: every 7th is null (sparse: only non-null in values array) */
        if (i % 7 == 0) {
            string_def_levels[i] = 0;
        } else {
            string_def_levels[i] = 1;
            const char* s = sample_strings[i % 10];
            strings[string_value_count].length = (int32_t)strlen(s);
            strings[string_value_count].data = (uint8_t*)s;
            string_value_count++;
        }

        /* Nullable ints: every 5th is null (sparse: only non-null in values array) */
        if (i % 5 == 0) {
            nullable_def_levels[i] = 0;
        } else {
            nullable_def_levels[i] = 1;
            nullable_ints[nullable_int_count] = i * 100;
            nullable_int_count++;
        }
    }
}

static int write_test_file(const char* path, carquet_compression_t codec) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) return 1;

    /* All major types */
    (void)carquet_schema_add_column(schema, "bool_col", CARQUET_PHYSICAL_BOOLEAN,
                                    NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "int32_col", CARQUET_PHYSICAL_INT32,
                                    NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "int64_col", CARQUET_PHYSICAL_INT64,
                                    NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "float_col", CARQUET_PHYSICAL_FLOAT,
                                    NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "double_col", CARQUET_PHYSICAL_DOUBLE,
                                    NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "string_col", CARQUET_PHYSICAL_BYTE_ARRAY,
                                    NULL, CARQUET_REPETITION_OPTIONAL, 0);
    (void)carquet_schema_add_column(schema, "nullable_int", CARQUET_PHYSICAL_INT32,
                                    NULL, CARQUET_REPETITION_OPTIONAL, 0);

    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = codec;
    opts.row_group_size = 2000 * 100;  /* Force multiple row groups */

    carquet_writer_t* writer = carquet_writer_create(path, schema, &opts, &err);
    if (!writer) {
        fprintf(stderr, "Failed to create writer: %s\n", err.message);
        carquet_schema_free(schema);
        return 1;
    }

    /* Allocate test data */
    uint8_t* bools = malloc(NUM_ROWS * sizeof(uint8_t));
    int32_t* int32s = malloc(NUM_ROWS * sizeof(int32_t));
    int64_t* int64s = malloc(NUM_ROWS * sizeof(int64_t));
    float* floats = malloc(NUM_ROWS * sizeof(float));
    double* doubles = malloc(NUM_ROWS * sizeof(double));
    carquet_byte_array_t* strings = malloc(NUM_ROWS * sizeof(carquet_byte_array_t));
    int16_t* string_def_levels = malloc(NUM_ROWS * sizeof(int16_t));
    int32_t* nullable_ints = malloc(NUM_ROWS * sizeof(int32_t));
    int16_t* nullable_def_levels = malloc(NUM_ROWS * sizeof(int16_t));

    generate_test_data(bools, int32s, int64s, floats, doubles,
                       strings, string_def_levels,
                       nullable_ints, nullable_def_levels, NUM_ROWS);

    /* Write all columns */
    (void)carquet_writer_write_batch(writer, 0, bools, NUM_ROWS, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 1, int32s, NUM_ROWS, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 2, int64s, NUM_ROWS, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 3, floats, NUM_ROWS, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 4, doubles, NUM_ROWS, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 5, strings, NUM_ROWS, string_def_levels, NULL);
    (void)carquet_writer_write_batch(writer, 6, nullable_ints, NUM_ROWS, nullable_def_levels, NULL);

    carquet_status_t status = carquet_writer_close(writer);

    free(bools);
    free(int32s);
    free(int64s);
    free(floats);
    free(doubles);
    free(strings);
    free(string_def_levels);
    free(nullable_ints);
    free(nullable_def_levels);
    carquet_schema_free(schema);

    return (status == CARQUET_OK) ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* output_dir = (argc > 1) ? argv[1] : "/tmp";

    if (carquet_init() != CARQUET_OK) {
        fprintf(stderr, "Failed to init carquet\n");
        return 1;
    }

    /* Output JSON with expected values for verification */
    printf("{\n");
    printf("  \"num_rows\": %d,\n", NUM_ROWS);
    printf("  \"files\": [\n");

    int first = 1;
    for (int c = 0; c < NUM_COMPRESSIONS; c++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/carquet_%s.parquet",
                 output_dir, COMPRESSION_NAMES[c]);

        if (write_test_file(path, COMPRESSIONS[c]) != 0) {
            fprintf(stderr, "Failed to write %s\n", path);
            continue;
        }

        if (!first) printf(",\n");
        first = 0;

        printf("    {\n");
        printf("      \"path\": \"%s\",\n", path);
        printf("      \"compression\": \"%s\",\n", COMPRESSION_NAMES[c]);
        printf("      \"columns\": {\n");
        printf("        \"bool_col\": { \"first\": [true, false, true, false, true], \"type\": \"bool\" },\n");
        printf("        \"int32_col\": { \"first\": [-5000, -4990, -4980, -4970, -4960], \"type\": \"int32\" },\n");
        printf("        \"int64_col\": { \"first\": [-2500000000, -2499000000, -2498000000, -2497000000, -2496000000], \"type\": \"int64\" },\n");
        printf("        \"float_col\": { \"first\": [-1250.0, -1249.5, -1249.0, -1248.5, -1248.0], \"type\": \"float\" },\n");
        printf("        \"double_col\": { \"first\": [-312.5, -312.375, -312.25, -312.125, -312.0], \"type\": \"double\" },\n");
        printf("        \"string_col\": { \"first\": [null, \"world\", \"carquet\", \"parquet\", \"test\"], \"null_pattern\": \"every_7th\", \"type\": \"string\" },\n");
        printf("        \"nullable_int\": { \"first\": [null, 100, 200, 300, 400], \"null_pattern\": \"every_5th\", \"type\": \"int32\" }\n");
        printf("      }\n");
        printf("    }");
    }

    printf("\n  ],\n");
    printf("  \"verification\": {\n");
    printf("    \"row_counts\": %d,\n", NUM_ROWS);
    printf("    \"null_count_string_col\": %d,\n", (NUM_ROWS + 6) / 7);
    printf("    \"null_count_nullable_int\": %d,\n", (NUM_ROWS + 4) / 5);
    printf("    \"bool_true_count\": %d,\n", (NUM_ROWS + 1) / 2);
    printf("    \"int32_sum\": %lld,\n", (long long)((NUM_ROWS - 1) * NUM_ROWS / 2 * 10 - 5000LL * NUM_ROWS));
    printf("    \"last_int32\": %d\n", (NUM_ROWS - 1) * 10 - 5000);
    printf("  }\n");
    printf("}\n");

    return 0;
}
