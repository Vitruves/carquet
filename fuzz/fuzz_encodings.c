/**
 * @file fuzz_encodings.c
 * @brief Fuzz target for carquet encoding decoders
 *
 * Tests RLE, Delta, and other encoding decoders with arbitrary input.
 * The first byte selects the encoding type.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <carquet/carquet.h>

/* Encoding function declarations */
int64_t carquet_rle_decode_all(
    const uint8_t* input, size_t input_size, int bit_width,
    uint32_t* output, int64_t max_values);

carquet_status_t carquet_delta_decode_int32(
    const uint8_t* data, size_t data_size,
    int32_t* values, int32_t num_values, size_t* bytes_consumed);

carquet_status_t carquet_delta_decode_int64(
    const uint8_t* data, size_t data_size,
    int64_t* values, int32_t num_values, size_t* bytes_consumed);

carquet_status_t carquet_plain_decode_int32(
    const uint8_t* data, size_t size, int32_t count, int32_t* values);

carquet_status_t carquet_plain_decode_int64(
    const uint8_t* data, size_t size, int32_t count, int64_t* values);

carquet_status_t carquet_plain_decode_float(
    const uint8_t* data, size_t size, int32_t count, float* values);

carquet_status_t carquet_plain_decode_double(
    const uint8_t* data, size_t size, int32_t count, double* values);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 3) {
        return 0;
    }

    carquet_init();

    /* First byte: encoding type, second byte: parameters */
    uint8_t encoding = data[0] % 6;
    uint8_t param = data[1];
    const uint8_t* payload = data + 2;
    size_t payload_size = size - 2;

    /* Output buffer - limit size to avoid OOM */
    size_t max_values = 10000;
    void* output = malloc(max_values * 8);  /* 8 bytes per value max */
    if (!output) {
        return 0;
    }

    switch (encoding) {
        case 0: {
            /* RLE - bit_width from param (1-32) */
            int bit_width = (param % 32) + 1;
            carquet_rle_decode_all(payload, payload_size, bit_width,
                                   (uint32_t*)output, max_values);
            break;
        }

        case 1: {
            /* Delta INT32 */
            int32_t num_values = (param % 100) + 1;
            size_t consumed;
            carquet_delta_decode_int32(payload, payload_size,
                                       (int32_t*)output, num_values, &consumed);
            break;
        }

        case 2: {
            /* Delta INT64 */
            int32_t num_values = (param % 100) + 1;
            size_t consumed;
            carquet_delta_decode_int64(payload, payload_size,
                                       (int64_t*)output, num_values, &consumed);
            break;
        }

        case 3: {
            /* Plain INT32 */
            int32_t count = payload_size / 4;
            if (count > (int32_t)max_values) count = max_values;
            carquet_plain_decode_int32(payload, payload_size,
                                       count, (int32_t*)output);
            break;
        }

        case 4: {
            /* Plain INT64 */
            int32_t count = payload_size / 8;
            if (count > (int32_t)max_values) count = max_values;
            carquet_plain_decode_int64(payload, payload_size,
                                       count, (int64_t*)output);
            break;
        }

        case 5: {
            /* Plain DOUBLE */
            int32_t count = payload_size / 8;
            if (count > (int32_t)max_values) count = max_values;
            carquet_plain_decode_double(payload, payload_size,
                                        count, (double*)output);
            break;
        }
    }

    free(output);
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
