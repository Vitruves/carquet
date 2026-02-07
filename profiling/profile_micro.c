/**
 * @file profile_micro.c
 * @brief Micro-benchmarks for isolated component profiling
 *
 * Isolates specific bottlenecks for detailed perf analysis:
 * - RLE decoding (levels)
 * - Dictionary gather operations
 * - Null bitmap construction
 * - Compression/decompression
 * - SIMD dispatch overhead
 *
 * Usage:
 *   ./profile_micro --component rle --iterations 1000000
 *   perf stat ./profile_micro --component gather
 *   perf record -g ./profile_micro --component all
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

/* Include internal headers for direct component access */
#include "carquet/carquet.h"
#include "encoding/rle.h"
#include "core/buffer.h"

/* Forward declarations for internal SIMD dispatch */
extern void carquet_dispatch_gather_i32(const int32_t* dict, const uint32_t* indices,
                                         int64_t count, int32_t* output);
extern void carquet_dispatch_gather_i64(const int64_t* dict, const uint32_t* indices,
                                         int64_t count, int64_t* output);
extern int64_t carquet_dispatch_count_non_nulls(const int16_t* def_levels, int64_t count,
                                                 int16_t max_def);
extern void carquet_dispatch_build_null_bitmap(const int16_t* def_levels, int64_t count,
                                                int16_t max_def, uint8_t* bitmap);

/* LZ4 internal functions */
extern int carquet_lz4_compress(const uint8_t* src, size_t src_len,
                                 uint8_t* dst, size_t dst_capacity);
extern int carquet_lz4_decompress(const uint8_t* src, size_t src_len,
                                   uint8_t* dst, size_t dst_capacity);

/* Snappy internal functions */
extern size_t carquet_snappy_max_compressed_length(size_t source_length);
extern int carquet_snappy_compress(const uint8_t* input, size_t input_length,
                                    uint8_t* output, size_t* output_length);
extern int carquet_snappy_decompress(const uint8_t* input, size_t input_length,
                                      uint8_t* output, size_t* output_length);

/* ============================================================================
 * Timing
 * ============================================================================ */

#ifdef _WIN32
static double get_time_ns(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1e9 / (double)freq.QuadPart;
}
#else
static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

#define BENCH_START() double _bench_start = get_time_ns()
#define BENCH_END() (get_time_ns() - _bench_start)

/* Prevent compiler from optimizing away */
volatile int64_t g_sink;

#ifdef __GNUC__
#define NOINLINE __attribute__((noinline))
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define NOINLINE
#define ALWAYS_INLINE inline
#endif

/* ============================================================================
 * RLE Micro-benchmark
 * ============================================================================ */

/*
 * Generates RLE-encoded data that mimics real Parquet definition levels.
 * Pattern: mostly non-null (1) with occasional null (0) values.
 */
static void generate_rle_test_data(uint8_t** out_data, size_t* out_size,
                                    int64_t num_values, double null_ratio) {
    /* Encode using the RLE encoder */
    carquet_buffer_t buf;
    carquet_buffer_init_capacity(&buf, 4096);

    carquet_rle_encoder_t enc;
    carquet_rle_encoder_init(&enc, &buf, 1); /* bit_width=1 for def levels */

    uint32_t seed = 42;
    for (int64_t i = 0; i < num_values; i++) {
        seed = seed * 1103515245 + 12345;
        double r = (double)(seed >> 16) / 32768.0;
        uint32_t val = (r >= null_ratio) ? 1 : 0;
        carquet_rle_encoder_put(&enc, val);
    }
    carquet_rle_encoder_flush(&enc);

    *out_data = malloc(buf.size);
    memcpy(*out_data, buf.data, buf.size);
    *out_size = buf.size;

    carquet_buffer_destroy(&buf);
}

NOINLINE static void bench_rle_decode_single(const uint8_t* data, size_t size,
                                              int bit_width, int64_t count,
                                              int64_t iterations) {
    int16_t* output = malloc(count * sizeof(int16_t));

    printf("  RLE single-value decode: ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        carquet_rle_decoder_t dec;
        carquet_rle_decoder_init(&dec, data, size, bit_width);

        for (int64_t i = 0; i < count; i++) {
            output[i] = (int16_t)carquet_rle_decoder_get(&dec);
        }
        g_sink = output[count / 2];
    }
    double elapsed = BENCH_END();

    double ns_per_value = elapsed / (iterations * count);
    double values_per_sec = 1e9 / ns_per_value;

    printf("%.2f ns/value, %.2f M values/sec\n", ns_per_value, values_per_sec / 1e6);

    free(output);
}

NOINLINE static void bench_rle_decode_batch(const uint8_t* data, size_t size,
                                             int bit_width, int64_t count,
                                             int64_t iterations) {
    uint32_t* output = malloc(count * sizeof(uint32_t));

    printf("  RLE batch decode:        ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        carquet_rle_decoder_t dec;
        carquet_rle_decoder_init(&dec, data, size, bit_width);

        int64_t decoded = carquet_rle_decoder_get_batch(&dec, output, count);
        g_sink = output[decoded / 2];
    }
    double elapsed = BENCH_END();

    double ns_per_value = elapsed / (iterations * count);
    double values_per_sec = 1e9 / ns_per_value;

    printf("%.2f ns/value, %.2f M values/sec\n", ns_per_value, values_per_sec / 1e6);

    free(output);
}

NOINLINE static void bench_rle_decode_levels(const uint8_t* data, size_t size,
                                              int bit_width, int64_t count,
                                              int64_t iterations) {
    int16_t* output = malloc(count * sizeof(int16_t));

    printf("  RLE decode_levels API:   ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        int64_t decoded = carquet_rle_decode_levels(data, size, bit_width,
                                                     output, count);
        g_sink = output[decoded / 2];
    }
    double elapsed = BENCH_END();

    double ns_per_value = elapsed / (iterations * count);
    double values_per_sec = 1e9 / ns_per_value;

    printf("%.2f ns/value, %.2f M values/sec\n", ns_per_value, values_per_sec / 1e6);

    free(output);
}

static void run_rle_benchmarks(int64_t count, int64_t iterations) {
    printf("\n=== RLE Decoding Benchmarks ===\n");
    printf("Values: %ld, Iterations: %ld\n\n", (long)count, (long)iterations);

    /* Generate test data with 10% nulls */
    uint8_t* rle_data;
    size_t rle_size;
    generate_rle_test_data(&rle_data, &rle_size, count, 0.10);

    printf("RLE data size: %zu bytes (%.2f bytes/value)\n\n",
           rle_size, (double)rle_size / count);

    bench_rle_decode_single(rle_data, rle_size, 1, count, iterations);
    bench_rle_decode_batch(rle_data, rle_size, 1, count, iterations);
    bench_rle_decode_levels(rle_data, rle_size, 1, count, iterations);

    free(rle_data);
}

/* ============================================================================
 * Dictionary Gather Micro-benchmark
 * ============================================================================ */

NOINLINE static void bench_gather_i32_scalar(const int32_t* dict, const uint32_t* indices,
                                              int64_t count, int32_t* output,
                                              int64_t iterations) {
    printf("  Gather i32 (scalar):     ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        for (int64_t i = 0; i < count; i++) {
            output[i] = dict[indices[i]];
        }
        g_sink = output[count / 2];
    }
    double elapsed = BENCH_END();

    double ns_per_value = elapsed / (iterations * count);
    printf("%.2f ns/value, %.2f M values/sec\n", ns_per_value, 1e9 / ns_per_value / 1e6);
}

NOINLINE static void bench_gather_i32_dispatch(const int32_t* dict, const uint32_t* indices,
                                                int64_t count, int32_t* output,
                                                int64_t iterations) {
    printf("  Gather i32 (dispatch):   ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        carquet_dispatch_gather_i32(dict, indices, count, output);
        g_sink = output[count / 2];
    }
    double elapsed = BENCH_END();

    double ns_per_value = elapsed / (iterations * count);
    printf("%.2f ns/value, %.2f M values/sec\n", ns_per_value, 1e9 / ns_per_value / 1e6);
}

NOINLINE static void bench_gather_i64_dispatch(const int64_t* dict, const uint32_t* indices,
                                                int64_t count, int64_t* output,
                                                int64_t iterations) {
    printf("  Gather i64 (dispatch):   ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        carquet_dispatch_gather_i64(dict, indices, count, output);
        g_sink = output[count / 2];
    }
    double elapsed = BENCH_END();

    double ns_per_value = elapsed / (iterations * count);
    printf("%.2f ns/value, %.2f M values/sec\n", ns_per_value, 1e9 / ns_per_value / 1e6);
}

static void run_gather_benchmark_sized(int64_t count, int64_t iterations, int dict_size) {
    int32_t* dict_i32 = malloc((size_t)dict_size * sizeof(int32_t));
    int64_t* dict_i64 = malloc((size_t)dict_size * sizeof(int64_t));
    for (int i = 0; i < dict_size; i++) {
        dict_i32[i] = i * 7 + 13;
        dict_i64[i] = (int64_t)i * 7 + 13;
    }

    /* Generate random indices (xorshift for full 32-bit range) */
    uint32_t* indices = malloc((size_t)count * sizeof(uint32_t));
    uint32_t seed = 42;
    for (int64_t i = 0; i < count; i++) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        indices[i] = seed % (uint32_t)dict_size;
    }

    int32_t* output_i32 = malloc((size_t)count * sizeof(int32_t));
    int64_t* output_i64 = malloc((size_t)count * sizeof(int64_t));

    double dict_mb = (double)dict_size * sizeof(int32_t) / (1024.0 * 1024.0);
    printf("\nDictionary size: %d entries (%.2f MB)\n", dict_size, dict_mb);

    bench_gather_i32_scalar(dict_i32, indices, count, output_i32, iterations);
    bench_gather_i32_dispatch(dict_i32, indices, count, output_i32, iterations);
    bench_gather_i64_dispatch(dict_i64, indices, count, output_i64, iterations);

    free(dict_i32);
    free(dict_i64);
    free(indices);
    free(output_i32);
    free(output_i64);
}

static void run_gather_benchmarks(int64_t count, int64_t iterations) {
    printf("\n=== Dictionary Gather Benchmarks ===\n");
    printf("Values: %ld, Iterations: %ld\n", (long)count, (long)iterations);

    /* Test with different dictionary sizes to measure cache effects:
     * - 1K entries (4KB) - fits in L1 cache
     * - 100K entries (400KB) - fits in L2/L3 cache
     * - 10M entries (40MB) - exceeds L3 cache (tests memory bandwidth)
     */
    run_gather_benchmark_sized(count, iterations, 1000);      /* L1 cache */
    run_gather_benchmark_sized(count, iterations, 100000);    /* L2/L3 cache */
    run_gather_benchmark_sized(count, iterations / 10, 10000000); /* Main memory */
}

/* ============================================================================
 * Null Bitmap Micro-benchmark
 * ============================================================================ */

NOINLINE static void bench_count_nulls_scalar(const int16_t* def_levels, int64_t count,
                                               int64_t iterations) {
    printf("  Count non-nulls (scalar):   ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        int64_t non_null = 0;
        for (int64_t i = 0; i < count; i++) {
            if (def_levels[i] == 1) non_null++;
        }
        g_sink = non_null;
    }
    double elapsed = BENCH_END();

    double ns_per_value = elapsed / (iterations * count);
    printf("%.2f ns/value, %.2f M values/sec\n", ns_per_value, 1e9 / ns_per_value / 1e6);
}

NOINLINE static void bench_count_nulls_dispatch(const int16_t* def_levels, int64_t count,
                                                  int64_t iterations) {
    printf("  Count non-nulls (dispatch): ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        int64_t non_null = carquet_dispatch_count_non_nulls(def_levels, count, 1);
        g_sink = non_null;
    }
    double elapsed = BENCH_END();

    double ns_per_value = elapsed / (iterations * count);
    printf("%.2f ns/value, %.2f M values/sec\n", ns_per_value, 1e9 / ns_per_value / 1e6);
}

NOINLINE static void bench_build_bitmap_scalar(const int16_t* def_levels, int64_t count,
                                                uint8_t* bitmap, int64_t iterations) {
    printf("  Build bitmap (scalar):      ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        int64_t bytes = (count + 7) / 8;
        memset(bitmap, 0, bytes);
        for (int64_t i = 0; i < count; i++) {
            if (def_levels[i] < 1) {
                bitmap[i / 8] |= (1 << (i % 8));
            }
        }
        g_sink = bitmap[0];
    }
    double elapsed = BENCH_END();

    double ns_per_value = elapsed / (iterations * count);
    printf("%.2f ns/value, %.2f M values/sec\n", ns_per_value, 1e9 / ns_per_value / 1e6);
}

NOINLINE static void bench_build_bitmap_dispatch(const int16_t* def_levels, int64_t count,
                                                   uint8_t* bitmap, int64_t iterations) {
    printf("  Build bitmap (dispatch):    ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        carquet_dispatch_build_null_bitmap(def_levels, count, 1, bitmap);
        g_sink = bitmap[0];
    }
    double elapsed = BENCH_END();

    double ns_per_value = elapsed / (iterations * count);
    printf("%.2f ns/value, %.2f M values/sec\n", ns_per_value, 1e9 / ns_per_value / 1e6);
}

static void run_null_bitmap_benchmarks(int64_t count, int64_t iterations) {
    printf("\n=== Null Bitmap Benchmarks ===\n");
    printf("Values: %ld, Iterations: %ld\n\n", (long)count, (long)iterations);

    /* Generate def levels with 10% nulls */
    int16_t* def_levels = malloc(count * sizeof(int16_t));
    uint32_t seed = 42;
    int64_t null_count = 0;
    for (int64_t i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        double r = (double)(seed >> 16) / 32768.0;
        def_levels[i] = (r >= 0.10) ? 1 : 0;
        if (def_levels[i] == 0) null_count++;
    }

    uint8_t* bitmap = malloc((count + 7) / 8);

    printf("Null ratio: %.1f%%\n\n", (double)null_count / count * 100);

    bench_count_nulls_scalar(def_levels, count, iterations);
    bench_count_nulls_dispatch(def_levels, count, iterations);
    bench_build_bitmap_scalar(def_levels, count, bitmap, iterations);
    bench_build_bitmap_dispatch(def_levels, count, bitmap, iterations);

    free(def_levels);
    free(bitmap);
}

/* ============================================================================
 * Compression Micro-benchmark
 * ============================================================================ */

NOINLINE static void bench_lz4_compress(const uint8_t* input, size_t input_size,
                                         uint8_t* output, size_t output_size,
                                         int64_t iterations) {
    printf("  LZ4 compress:    ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        int compressed = carquet_lz4_compress(input, input_size, output, output_size);
        g_sink = compressed;
    }
    double elapsed = BENCH_END();

    double mb_per_sec = (double)input_size * iterations / elapsed * 1e3;
    printf("%.2f MB/sec\n", mb_per_sec);
}

NOINLINE static void bench_lz4_decompress(const uint8_t* compressed, size_t comp_size,
                                           uint8_t* output, size_t output_size,
                                           int64_t iterations) {
    printf("  LZ4 decompress:  ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        int decompressed = carquet_lz4_decompress(compressed, comp_size, output, output_size);
        g_sink = decompressed;
    }
    double elapsed = BENCH_END();

    double mb_per_sec = (double)output_size * iterations / elapsed * 1e3;
    printf("%.2f MB/sec\n", mb_per_sec);
}

NOINLINE static void bench_snappy_compress(const uint8_t* input, size_t input_size,
                                            uint8_t* output, size_t output_size,
                                            int64_t iterations) {
    printf("  Snappy compress: ");
    fflush(stdout);

    BENCH_START();
    for (int64_t iter = 0; iter < iterations; iter++) {
        size_t out_len = output_size;
        int ret = carquet_snappy_compress(input, input_size, output, &out_len);
        g_sink = ret;
    }
    double elapsed = BENCH_END();

    double mb_per_sec = (double)input_size * iterations / elapsed * 1e3;
    printf("%.2f MB/sec\n", mb_per_sec);
}

static void run_compression_benchmarks(size_t size, int64_t iterations) {
    printf("\n=== Compression Benchmarks ===\n");
    printf("Data size: %zu bytes, Iterations: %ld\n\n", size, (long)iterations);

    /* Generate compressible data (simulating Parquet column data) */
    uint8_t* input = malloc(size);
    uint32_t seed = 42;
    for (size_t i = 0; i < size; i++) {
        seed = seed * 1103515245 + 12345;
        /* Mix of patterns: some repetition for compressibility */
        input[i] = (uint8_t)((seed >> 16) % 64);
    }

    size_t max_compressed = size * 2;
    uint8_t* compressed = malloc(max_compressed);
    uint8_t* decompressed = malloc(size);

    /* LZ4 */
    int comp_size = carquet_lz4_compress(input, size, compressed, max_compressed);
    printf("LZ4 ratio: %.2fx\n", (double)size / comp_size);
    bench_lz4_compress(input, size, compressed, max_compressed, iterations);
    bench_lz4_decompress(compressed, comp_size, decompressed, size, iterations);

    printf("\n");

    /* Snappy */
    size_t snappy_size = max_compressed;
    carquet_snappy_compress(input, size, compressed, &snappy_size);
    printf("Snappy ratio: %.2fx\n", (double)size / snappy_size);
    bench_snappy_compress(input, size, compressed, max_compressed, iterations);

    free(input);
    free(compressed);
    free(decompressed);
}

/* ============================================================================
 * Dispatch Overhead Benchmark
 * ============================================================================ */

static void run_dispatch_overhead_benchmark(int64_t iterations) {
    printf("\n=== Dispatch Overhead Benchmark ===\n");
    printf("Iterations: %ld\n\n", (long)iterations);

    /* Minimal data to isolate function call overhead */
    int32_t dict[4] = {1, 2, 3, 4};
    uint32_t indices[4] = {0, 1, 2, 3};
    int32_t output[4];

    double elapsed_direct, elapsed_dispatch;

    printf("  Direct call (4 values): ");
    fflush(stdout);

    {
        BENCH_START();
        for (int64_t iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < 4; i++) {
                output[i] = dict[indices[i]];
            }
            g_sink = output[0];
        }
        elapsed_direct = BENCH_END();
    }
    printf("%.2f ns/call\n", elapsed_direct / iterations);

    printf("  Dispatch call (4 values): ");
    fflush(stdout);

    {
        BENCH_START();
        for (int64_t iter = 0; iter < iterations; iter++) {
            carquet_dispatch_gather_i32(dict, indices, 4, output);
            g_sink = output[0];
        }
        elapsed_dispatch = BENCH_END();
    }
    printf("%.2f ns/call\n", elapsed_dispatch / iterations);

    printf("  Overhead: %.2f ns (%.1fx)\n",
           (elapsed_dispatch - elapsed_direct) / iterations,
           elapsed_dispatch / elapsed_direct);
}

/* ============================================================================
 * Main
 * ============================================================================ */

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --component NAME   Component to benchmark:\n");
    printf("                     rle, gather, null, compression, dispatch, all\n");
    printf("  --count N          Number of values (default: 1000000)\n");
    printf("  --iterations N     Number of iterations (default: 100)\n");
    printf("  -h, --help         Show this help\n");
    printf("\nExample:\n");
    printf("  perf record -g %s --component rle --iterations 1000\n", prog);
}

int main(int argc, char** argv) {
    const char* component = "all";
    int64_t count = 1000000;
    int64_t iterations = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--component") == 0 && i + 1 < argc) {
            component = argv[++i];
        }
        else if (strcmp(argv[i], "--count") == 0 && i + 1 < argc) {
            count = atol(argv[++i]);
        }
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atol(argv[++i]);
        }
    }

    printf("=== Carquet Micro-benchmarks ===\n");
    printf("Component: %s\n", component);

    if (strcmp(component, "rle") == 0 || strcmp(component, "all") == 0) {
        run_rle_benchmarks(count, iterations);
    }
    if (strcmp(component, "gather") == 0 || strcmp(component, "all") == 0) {
        run_gather_benchmarks(count, iterations);
    }
    if (strcmp(component, "null") == 0 || strcmp(component, "all") == 0) {
        run_null_bitmap_benchmarks(count, iterations);
    }
    if (strcmp(component, "compression") == 0 || strcmp(component, "all") == 0) {
        run_compression_benchmarks(1024 * 1024, iterations / 10);  /* 1MB blocks */
    }
    if (strcmp(component, "dispatch") == 0 || strcmp(component, "all") == 0) {
        run_dispatch_overhead_benchmark(iterations * 10000);
    }

    printf("\nDone.\n");
    return 0;
}
