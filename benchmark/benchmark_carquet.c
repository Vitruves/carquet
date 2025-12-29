/**
 * @file benchmark_carquet.c
 * @brief Performance benchmarks for Carquet
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <carquet/carquet.h>

#define BENCH_ROWS 1000000
#define BENCH_ITERATIONS 3

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void benchmark_write(const char* filename, carquet_compression_t codec) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* schema = carquet_schema_create(&err);
(void)carquet_schema_add_column(schema, "id", CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0);
(void)carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_REQUIRED, 0);

    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = codec;

    int64_t* ids = malloc(BENCH_ROWS * sizeof(int64_t));
    double* values = malloc(BENCH_ROWS * sizeof(double));
    for (int i = 0; i < BENCH_ROWS; i++) {
        ids[i] = i;
        values[i] = (double)i * 1.1;
    }

    double total_time = 0;
    for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
        double start = get_time_ms();

        carquet_writer_t* writer = carquet_writer_create(filename, schema, &opts, &err);
(void)carquet_writer_write_batch(writer, 0, ids, BENCH_ROWS, NULL, NULL);
(void)carquet_writer_write_batch(writer, 1, values, BENCH_ROWS, NULL, NULL);
(void)carquet_writer_close(writer);

        double elapsed = get_time_ms() - start;
        total_time += elapsed;
    }

    printf("  Write %s: %.2f ms (%.2f M rows/sec)\n",
           carquet_compression_name(codec),
           total_time / BENCH_ITERATIONS,
           (BENCH_ROWS / 1000000.0) / (total_time / BENCH_ITERATIONS / 1000.0));

    free(ids);
    free(values);
    carquet_schema_free(schema);
}

static void benchmark_read(const char* filename) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    double total_time = 0;
    for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
        double start = get_time_ms();

        carquet_reader_t* reader = carquet_reader_open(filename, NULL, &err);
        if (!reader) continue;

        carquet_batch_reader_config_t config;
        carquet_batch_reader_config_init(&config);
        config.batch_size = 65536;

        carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &config, &err);
        if (batch_reader) {
            carquet_row_batch_t* batch = NULL;
            int64_t total_rows = 0;
            while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
                total_rows += carquet_row_batch_num_rows(batch);
                carquet_row_batch_free(batch);
                batch = NULL;
            }
            (void)total_rows;
            carquet_batch_reader_free(batch_reader);
        }

        carquet_reader_close(reader);
        double elapsed = get_time_ms() - start;
        total_time += elapsed;
    }

    printf("  Read: %.2f ms (%.2f M rows/sec)\n",
           total_time / BENCH_ITERATIONS,
           (BENCH_ROWS / 1000000.0) / (total_time / BENCH_ITERATIONS / 1000.0));
}

int main(void) {
    printf("=== Carquet Benchmarks ===\n");
    printf("Rows: %d, Iterations: %d\n\n", BENCH_ROWS, BENCH_ITERATIONS);

    const char* filename = "/tmp/bench_carquet.parquet";

    carquet_compression_t codecs[] = {
        CARQUET_COMPRESSION_UNCOMPRESSED,
        CARQUET_COMPRESSION_SNAPPY,
        CARQUET_COMPRESSION_LZ4,
        CARQUET_COMPRESSION_ZSTD
    };

    for (size_t i = 0; i < sizeof(codecs)/sizeof(codecs[0]); i++) {
        printf("\n%s:\n", carquet_compression_name(codecs[i]));
        benchmark_write(filename, codecs[i]);
        benchmark_read(filename);
    }

    remove(filename);
    printf("\nBenchmarks complete.\n");
    return 0;
}
