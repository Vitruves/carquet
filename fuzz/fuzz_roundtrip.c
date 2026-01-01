/**
 * @file fuzz_roundtrip.c
 * @brief Fuzz target for encode-decode roundtrip testing
 *
 * Tests that encode(decode(input)) produces consistent results and
 * that decode(encode(data)) == data for valid inputs.
 * This catches bugs where encode and decode paths have mismatched logic.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <carquet/carquet.h>

/* Delta encoding declarations */
carquet_status_t carquet_delta_encode_int32(
    const int32_t* input, int32_t count,
    uint8_t* output, size_t output_size, size_t* bytes_written);

carquet_status_t carquet_delta_decode_int32(
    const uint8_t* data, size_t size,
    int32_t* output, int32_t num_values, size_t* bytes_consumed);

carquet_status_t carquet_delta_encode_int64(
    const int64_t* input, int64_t count,
    uint8_t* output, size_t output_size, size_t* bytes_written);

carquet_status_t carquet_delta_decode_int64(
    const uint8_t* data, size_t size,
    int64_t* output, int32_t num_values, size_t* bytes_consumed);

/* LZ4 compression declarations */
carquet_status_t carquet_lz4_compress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);

carquet_status_t carquet_lz4_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);

size_t carquet_lz4_compress_bound(size_t src_size);

/* Byte stream split declarations */
carquet_status_t carquet_byte_stream_split_encode_float(
    const float* values, int64_t count,
    uint8_t* output, size_t output_capacity, size_t* bytes_written);

carquet_status_t carquet_byte_stream_split_decode_float(
    const uint8_t* data, size_t data_size,
    float* values, int64_t count);

carquet_status_t carquet_byte_stream_split_encode_double(
    const double* values, int64_t count,
    uint8_t* output, size_t output_capacity, size_t* bytes_written);

carquet_status_t carquet_byte_stream_split_decode_double(
    const uint8_t* data, size_t data_size,
    double* values, int64_t count);

/**
 * Test delta int32 roundtrip: interpret fuzz input as int32 values,
 * encode them, decode, and verify match.
 */
static void fuzz_delta_int32_roundtrip(const uint8_t* data, size_t size) {
    if (size < 4 || size > 4000) return;  /* Reasonable bounds */

    int32_t count = (int32_t)(size / 4);
    if (count < 1 || count > 1000) return;

    /* Interpret input as int32 values */
    int32_t* input = (int32_t*)malloc((size_t)count * sizeof(int32_t));
    if (!input) return;
    memcpy(input, data, (size_t)count * sizeof(int32_t));

    /* Encode */
    size_t encoded_capacity = (size_t)count * 10 + 100;  /* Generous buffer */
    uint8_t* encoded = (uint8_t*)malloc(encoded_capacity);
    if (!encoded) { free(input); return; }

    size_t encoded_size = 0;
    carquet_status_t status = carquet_delta_encode_int32(
        input, count, encoded, encoded_capacity, &encoded_size);

    if (status == CARQUET_OK && encoded_size > 0) {
        /* Decode */
        int32_t* decoded = (int32_t*)malloc((size_t)count * sizeof(int32_t));
        if (decoded) {
            size_t consumed = 0;
            status = carquet_delta_decode_int32(
                encoded, encoded_size, decoded, count, &consumed);

            /* Verify roundtrip */
            if (status == CARQUET_OK) {
                for (int32_t i = 0; i < count; i++) {
                    if (input[i] != decoded[i]) {
                        /* Mismatch - this would be a bug! */
                        __builtin_trap();
                    }
                }
            }
            free(decoded);
        }
    }

    free(encoded);
    free(input);
}

/**
 * Test delta int64 roundtrip
 */
static void fuzz_delta_int64_roundtrip(const uint8_t* data, size_t size) {
    if (size < 8 || size > 8000) return;

    int64_t count = (int64_t)(size / 8);
    if (count < 1 || count > 1000) return;

    int64_t* input = (int64_t*)malloc((size_t)count * sizeof(int64_t));
    if (!input) return;
    memcpy(input, data, (size_t)count * sizeof(int64_t));

    size_t encoded_capacity = (size_t)count * 20 + 100;
    uint8_t* encoded = (uint8_t*)malloc(encoded_capacity);
    if (!encoded) { free(input); return; }

    size_t encoded_size = 0;
    carquet_status_t status = carquet_delta_encode_int64(
        input, count, encoded, encoded_capacity, &encoded_size);

    if (status == CARQUET_OK && encoded_size > 0) {
        int64_t* decoded = (int64_t*)malloc((size_t)count * sizeof(int64_t));
        if (decoded) {
            size_t consumed = 0;
            status = carquet_delta_decode_int64(
                encoded, encoded_size, decoded, (int32_t)count, &consumed);

            if (status == CARQUET_OK) {
                for (int64_t i = 0; i < count; i++) {
                    if (input[i] != decoded[i]) {
                        __builtin_trap();
                    }
                }
            }
            free(decoded);
        }
    }

    free(encoded);
    free(input);
}

/**
 * Test LZ4 compression roundtrip
 */
static void fuzz_lz4_roundtrip(const uint8_t* data, size_t size) {
    if (size < 1 || size > 100000) return;

    size_t compressed_capacity = carquet_lz4_compress_bound(size);
    uint8_t* compressed = (uint8_t*)malloc(compressed_capacity);
    if (!compressed) return;

    size_t compressed_size = 0;
    carquet_status_t status = carquet_lz4_compress(
        data, size, compressed, compressed_capacity, &compressed_size);

    if (status == CARQUET_OK && compressed_size > 0) {
        uint8_t* decompressed = (uint8_t*)malloc(size);
        if (decompressed) {
            size_t decompressed_size = 0;
            status = carquet_lz4_decompress(
                compressed, compressed_size,
                decompressed, size, &decompressed_size);

            if (status == CARQUET_OK) {
                if (decompressed_size != size) {
                    __builtin_trap();
                }
                if (memcmp(data, decompressed, size) != 0) {
                    __builtin_trap();
                }
            }
            free(decompressed);
        }
    }

    free(compressed);
}

/**
 * Test byte stream split float roundtrip
 */
static void fuzz_bss_float_roundtrip(const uint8_t* data, size_t size) {
    if (size < 4 || size > 40000) return;

    int64_t count = (int64_t)(size / 4);
    if (count < 1 || count > 10000) return;

    float* input = (float*)malloc((size_t)count * sizeof(float));
    if (!input) return;
    memcpy(input, data, (size_t)count * sizeof(float));

    size_t output_capacity = (size_t)count * sizeof(float);
    uint8_t* encoded = (uint8_t*)malloc(output_capacity);
    if (!encoded) { free(input); return; }

    size_t bytes_written = 0;
    carquet_status_t status = carquet_byte_stream_split_encode_float(
        input, count, encoded, output_capacity, &bytes_written);

    if (status == CARQUET_OK && bytes_written > 0) {
        float* decoded = (float*)malloc((size_t)count * sizeof(float));
        if (decoded) {
            status = carquet_byte_stream_split_decode_float(
                encoded, (size_t)count * sizeof(float), decoded, count);

            if (status == CARQUET_OK) {
                if (memcmp(input, decoded, (size_t)count * sizeof(float)) != 0) {
                    __builtin_trap();
                }
            }
            free(decoded);
        }
    }

    free(encoded);
    free(input);
}

/**
 * Test byte stream split double roundtrip
 */
static void fuzz_bss_double_roundtrip(const uint8_t* data, size_t size) {
    if (size < 8 || size > 80000) return;

    int64_t count = (int64_t)(size / 8);
    if (count < 1 || count > 10000) return;

    double* input = (double*)malloc((size_t)count * sizeof(double));
    if (!input) return;
    memcpy(input, data, (size_t)count * sizeof(double));

    size_t output_capacity = (size_t)count * sizeof(double);
    uint8_t* encoded = (uint8_t*)malloc(output_capacity);
    if (!encoded) { free(input); return; }

    size_t bytes_written = 0;
    carquet_status_t status = carquet_byte_stream_split_encode_double(
        input, count, encoded, output_capacity, &bytes_written);

    if (status == CARQUET_OK && bytes_written > 0) {
        double* decoded = (double*)malloc((size_t)count * sizeof(double));
        if (decoded) {
            status = carquet_byte_stream_split_decode_double(
                encoded, (size_t)count * sizeof(double), decoded, count);

            if (status == CARQUET_OK) {
                if (memcmp(input, decoded, (size_t)count * sizeof(double)) != 0) {
                    __builtin_trap();
                }
            }
            free(decoded);
        }
    }

    free(encoded);
    free(input);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 2) {
        return 0;
    }

    (void)carquet_init();

    /* First byte selects test mode */
    uint8_t mode = data[0] % 5;
    const uint8_t* payload = data + 1;
    size_t payload_size = size - 1;

    switch (mode) {
        case 0:
            fuzz_delta_int32_roundtrip(payload, payload_size);
            break;
        case 1:
            fuzz_delta_int64_roundtrip(payload, payload_size);
            break;
        case 2:
            fuzz_lz4_roundtrip(payload, payload_size);
            break;
        case 3:
            fuzz_bss_float_roundtrip(payload, payload_size);
            break;
        case 4:
            fuzz_bss_double_roundtrip(payload, payload_size);
            break;
    }

    return 0;
}

#ifdef AFL_MAIN
#include <stdio.h>
#include <sys/stat.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        perror("fopen");
        return 1;
    }

    struct stat st;
    fstat(fileno(f), &st);
    size_t file_size = (size_t)st.st_size;

    uint8_t* file_data = malloc(file_size);
    if (!file_data) {
        fclose(f);
        return 1;
    }

    fread(file_data, 1, file_size, f);
    fclose(f);

    int result = LLVMFuzzerTestOneInput(file_data, file_size);
    free(file_data);
    return result;
}
#endif
