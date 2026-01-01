/**
 * @file roundtrip_writer.c
 * @brief Write test data with known values for roundtrip verification
 *
 * This program writes a Parquet file with predictable values that can be
 * verified by reading with PyArrow, DuckDB, or other Parquet implementations.
 *
 * Build:
 *   gcc -O2 -I../include -o roundtrip_writer roundtrip_writer.c \
 *       ../build/libcarquet.a -lzstd -lz
 *
 * Usage:
 *   ./roundtrip_writer output.parquet
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <carquet/carquet.h>

#define NUM_ROWS 1000

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <output.parquet>\n", argv[0]);
        return 1;
    }

    const char* output_path = argv[1];
    carquet_error_t err = CARQUET_ERROR_INIT;

    if (carquet_init() != CARQUET_OK) {
        fprintf(stderr, "Failed to init carquet\n");
        return 1;
    }

    /* Create schema with various types including nullable */
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        fprintf(stderr, "Failed to create schema: %s\n", err.message);
        return 1;
    }

    (void)carquet_schema_add_column(schema, "int32_col", CARQUET_PHYSICAL_INT32,
                                    NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "int64_col", CARQUET_PHYSICAL_INT64,
                                    NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "float_col", CARQUET_PHYSICAL_FLOAT,
                                    NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "double_col", CARQUET_PHYSICAL_DOUBLE,
                                    NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "nullable_int", CARQUET_PHYSICAL_INT32,
                                    NULL, CARQUET_REPETITION_OPTIONAL, 0);

    /* Create writer with ZSTD compression */
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_ZSTD;

    carquet_writer_t* writer = carquet_writer_create(output_path, schema, &opts, &err);
    if (!writer) {
        fprintf(stderr, "Failed to create writer: %s\n", err.message);
        carquet_schema_free(schema);
        return 1;
    }

    /* Generate test data with predictable values */
    int32_t* int32_data = malloc(NUM_ROWS * sizeof(int32_t));
    int64_t* int64_data = malloc(NUM_ROWS * sizeof(int64_t));
    float* float_data = malloc(NUM_ROWS * sizeof(float));
    double* double_data = malloc(NUM_ROWS * sizeof(double));
    int32_t* nullable_data = malloc(NUM_ROWS * sizeof(int32_t));
    int16_t* def_levels = malloc(NUM_ROWS * sizeof(int16_t));

    for (int i = 0; i < NUM_ROWS; i++) {
        int32_data[i] = i * 10;
        int64_data[i] = (int64_t)i * 1000000LL;
        float_data[i] = (float)i * 0.5f;
        double_data[i] = (double)i * 0.125;
        nullable_data[i] = i * 100;
        /* Every 5th value is NULL (def_level=0 means null, 1 means present) */
        def_levels[i] = (i % 5 == 0) ? 0 : 1;
    }

    /* Write all columns */
    (void)carquet_writer_write_batch(writer, 0, int32_data, NUM_ROWS, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 1, int64_data, NUM_ROWS, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 2, float_data, NUM_ROWS, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 3, double_data, NUM_ROWS, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 4, nullable_data, NUM_ROWS, def_levels, NULL);

    carquet_status_t status = carquet_writer_close(writer);

    free(int32_data);
    free(int64_data);
    free(float_data);
    free(double_data);
    free(nullable_data);
    free(def_levels);
    carquet_schema_free(schema);

    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to close writer\n");
        return 1;
    }

    printf("Wrote %d rows to %s\n", NUM_ROWS, output_path);
    return 0;
}
