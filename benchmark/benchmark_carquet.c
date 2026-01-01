/**
 * @file benchmark_carquet.c
 * @brief Performance benchmarks for Carquet
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#include <carquet/carquet.h>

#define WARMUP_ITERATIONS 2
#define BENCH_ITERATIONS 5

typedef struct {
    const char* name;
    int rows;
} dataset_t;

typedef struct {
    carquet_compression_t codec;
    const char* name;
} compression_config_t;

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static long get_file_size(const char* filename) {
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size;
    }
    return 0;
}

static double benchmark_write(const char* filename, int num_rows, carquet_compression_t codec) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_schema_t* schema = carquet_schema_create(&err);
    (void)carquet_schema_add_column(schema, "id", CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_REQUIRED, 0);
    (void)carquet_schema_add_column(schema, "category", CARQUET_PHYSICAL_INT32, NULL, CARQUET_REPETITION_REQUIRED, 0);

    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = codec;
    opts.row_group_size = 100000;

    int64_t* ids = malloc(num_rows * sizeof(int64_t));
    double* values = malloc(num_rows * sizeof(double));
    int32_t* categories = malloc(num_rows * sizeof(int32_t));

    for (int i = 0; i < num_rows; i++) {
        ids[i] = i;
        values[i] = (double)i * 1.5 + 0.123456789;
        categories[i] = i % 100;
    }

    double start = get_time_ms();

    carquet_writer_t* writer = carquet_writer_create(filename, schema, &opts, &err);
    (void)carquet_writer_write_batch(writer, 0, ids, num_rows, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 1, values, num_rows, NULL, NULL);
    (void)carquet_writer_write_batch(writer, 2, categories, num_rows, NULL, NULL);
    (void)carquet_writer_close(writer);

    double elapsed = get_time_ms() - start;

    free(ids);
    free(values);
    free(categories);
    carquet_schema_free(schema);

    return elapsed;
}

static double benchmark_read(const char* filename, int expected_rows) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    double start = get_time_ms();

    /* Fair benchmark: enable mmap but also verify checksums */
    carquet_reader_options_t opts;
    carquet_reader_options_init(&opts);
    opts.use_mmap = true;
    opts.verify_checksums = true;  /* Fair comparison with PyArrow */

    carquet_reader_t* reader = carquet_reader_open(filename, &opts, &err);
    if (!reader) return 0;

    carquet_batch_reader_config_t config;
    carquet_batch_reader_config_init(&config);
    config.batch_size = 262144;  /* 256K rows per batch */

    carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &config, &err);
    if (batch_reader) {
        carquet_row_batch_t* batch = NULL;
        int64_t total_rows = 0;
        volatile int64_t checksum = 0;  /* Prevent optimizer from skipping reads */

        while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
            int64_t batch_rows = carquet_row_batch_num_rows(batch);
            total_rows += batch_rows;

            /* Actually read and verify data like PyArrow does */
            const void* data;
            const uint8_t* nulls;
            int64_t count;

            /* Read column 0 (ids - INT64) */
            if (carquet_row_batch_column(batch, 0, &data, &nulls, &count) == CARQUET_OK && data) {
                const int64_t* ids = (const int64_t*)data;
                for (int64_t i = 0; i < count; i += 1000) {
                    checksum += ids[i];  /* Sample every 1000th value */
                }
            }

            /* Read column 1 (values - DOUBLE) */
            if (carquet_row_batch_column(batch, 1, &data, &nulls, &count) == CARQUET_OK && data) {
                const double* values = (const double*)data;
                for (int64_t i = 0; i < count; i += 1000) {
                    checksum += (int64_t)values[i];
                }
            }

            /* Read column 2 (categories - INT32) */
            if (carquet_row_batch_column(batch, 2, &data, &nulls, &count) == CARQUET_OK && data) {
                const int32_t* cats = (const int32_t*)data;
                for (int64_t i = 0; i < count; i += 1000) {
                    checksum += cats[i];
                }
            }

            carquet_row_batch_free(batch);
            batch = NULL;
        }

        (void)checksum;  /* Use it to prevent optimization */
        if (total_rows != expected_rows) {
            fprintf(stderr, "Warning: row count mismatch %lld vs %d\n",
                    (long long)total_rows, expected_rows);
        }
        carquet_batch_reader_free(batch_reader);
    }

    carquet_reader_close(reader);

    return get_time_ms() - start;
}

static void run_benchmark(const char* dataset_name, int num_rows,
                          carquet_compression_t codec, const char* compression_name) {
    char filename[256];
    snprintf(filename, sizeof(filename), "/tmp/benchmark_%s_%s_carquet.parquet",
             dataset_name, compression_name);

    printf("\n=== %s (%d rows, %s) ===\n", dataset_name, num_rows, compression_name);

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        benchmark_write(filename, num_rows, codec);
        benchmark_read(filename, num_rows);
    }

    // Benchmark
    double write_times[BENCH_ITERATIONS];
    double read_times[BENCH_ITERATIONS];
    long file_size = 0;

    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        write_times[i] = benchmark_write(filename, num_rows, codec);
        file_size = get_file_size(filename);
        read_times[i] = benchmark_read(filename, num_rows);
    }

    double write_sum = 0, read_sum = 0;
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        write_sum += write_times[i];
        read_sum += read_times[i];
    }
    double write_avg = write_sum / BENCH_ITERATIONS;
    double read_avg = read_sum / BENCH_ITERATIONS;

    double rows_per_sec_write = (num_rows / write_avg) * 1000;
    double rows_per_sec_read = (num_rows / read_avg) * 1000;

    printf("  Write: %.2f ms (%.2f M rows/sec)\n", write_avg, rows_per_sec_write / 1e6);
    printf("  Read:  %.2f ms (%.2f M rows/sec)\n", read_avg, rows_per_sec_read / 1e6);
    printf("  File:  %.2f MB (%.2f bytes/row)\n",
           file_size / (1024.0 * 1024.0), (double)file_size / num_rows);

    // Output CSV line for parsing
    printf("CSV:carquet,%s,%s,%d,%.2f,%.2f,%ld\n",
           dataset_name, compression_name, num_rows, write_avg, read_avg, file_size);

    remove(filename);
}

int main(void) {
    printf("Carquet Benchmark\n");
    printf("=================\n");

    dataset_t datasets[] = {
        {"small", 100000},
        {"medium", 1000000},
        {"large", 10000000}
    };

    compression_config_t compressions[] = {
        {CARQUET_COMPRESSION_UNCOMPRESSED, "none"},
        {CARQUET_COMPRESSION_SNAPPY, "snappy"},
        {CARQUET_COMPRESSION_ZSTD, "zstd"}
    };

    for (size_t d = 0; d < sizeof(datasets) / sizeof(datasets[0]); d++) {
        for (size_t c = 0; c < sizeof(compressions) / sizeof(compressions[0]); c++) {
            run_benchmark(datasets[d].name, datasets[d].rows,
                         compressions[c].codec, compressions[c].name);
        }
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
