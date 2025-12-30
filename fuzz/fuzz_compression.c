/**
 * @file fuzz_compression.c
 * @brief Fuzz target for carquet compression decoders
 *
 * Tests all compression decoders with arbitrary input.
 * The first byte of input selects the codec.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <carquet/carquet.h>

/* Compression function declarations */
int carquet_snappy_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);

carquet_status_t carquet_snappy_get_uncompressed_length(
    const uint8_t* src, size_t src_size, size_t* length);

int carquet_lz4_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);

int carquet_gzip_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);

int carquet_zstd_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 2) {
        return 0;
    }

    carquet_init();

    /* First byte selects codec */
    uint8_t codec = data[0] % 4;
    const uint8_t* payload = data + 1;
    size_t payload_size = size - 1;

    /* Reasonable output buffer - not too large to avoid OOM */
    size_t dst_capacity = 1024 * 1024;  /* 1MB max */
    uint8_t* dst = malloc(dst_capacity);
    if (!dst) {
        return 0;
    }

    size_t dst_size = 0;

    switch (codec) {
        case 0: {
            /* Snappy */
            size_t uncompressed_len = 0;
            carquet_status_t status = carquet_snappy_get_uncompressed_length(
                payload, payload_size, &uncompressed_len);
            if (status == CARQUET_OK && uncompressed_len <= dst_capacity) {
                carquet_snappy_decompress(payload, payload_size,
                                          dst, uncompressed_len, &dst_size);
            }
            break;
        }

        case 1: {
            /* LZ4 - need to extract uncompressed size from header if present */
            /* LZ4 block format doesn't include size, so we try with max buffer */
            carquet_lz4_decompress(payload, payload_size,
                                   dst, dst_capacity, &dst_size);
            break;
        }

        case 2: {
            /* GZIP/DEFLATE */
            carquet_gzip_decompress(payload, payload_size,
                                    dst, dst_capacity, &dst_size);
            break;
        }

        case 3: {
            /* ZSTD */
            carquet_zstd_decompress(payload, payload_size,
                                    dst, dst_capacity, &dst_size);
            break;
        }
    }

    free(dst);
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
    size_t size = st.st_size;

    uint8_t* data = malloc(size);
    if (!data) {
        fclose(f);
        return 1;
    }

    fread(data, 1, size, f);
    fclose(f);

    int result = LLVMFuzzerTestOneInput(data, size);
    free(data);
    return result;
}
#endif
