/**
 * @file fuzz_encodings.c
 * @brief Fuzz target for carquet encoding decoders
 *
 * Tests RLE, Delta, Dictionary, Byte Stream Split, and other encoding
 * decoders with arbitrary input. The first byte selects the encoding type.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <carquet/carquet.h>

/* Internal encoding headers */
#include "encoding/rle.h"
#include "encoding/plain.h"

/* Delta decode declarations (internal) */
carquet_status_t carquet_delta_decode_int32(
    const uint8_t* data, size_t size,
    int32_t* output, int32_t num_values,
    size_t* bytes_consumed);

carquet_status_t carquet_delta_decode_int64(
    const uint8_t* data, size_t size,
    int64_t* output, int32_t num_values,
    size_t* bytes_consumed);

/* Dictionary decode declarations */
carquet_status_t carquet_dictionary_decode_int32(
    const uint8_t* dict_data, size_t dict_size,
    const uint8_t* indices_data, size_t indices_size,
    int32_t* output, int64_t num_values);

carquet_status_t carquet_dictionary_decode_int64(
    const uint8_t* dict_data, size_t dict_size,
    const uint8_t* indices_data, size_t indices_size,
    int64_t* output, int64_t num_values);

carquet_status_t carquet_dictionary_decode_float(
    const uint8_t* dict_data, size_t dict_size,
    const uint8_t* indices_data, size_t indices_size,
    float* output, int64_t num_values);

carquet_status_t carquet_dictionary_decode_double(
    const uint8_t* dict_data, size_t dict_size,
    const uint8_t* indices_data, size_t indices_size,
    double* output, int64_t num_values);

/* Byte stream split decode declarations */
carquet_status_t carquet_byte_stream_split_decode_float(
    const uint8_t* input, size_t input_size,
    float* output, int64_t num_values);

carquet_status_t carquet_byte_stream_split_decode_double(
    const uint8_t* input, size_t input_size,
    double* output, int64_t num_values);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 3) {
        return 0;
    }

    (void)carquet_init();

    /* First byte: encoding type, second byte: parameters */
    uint8_t encoding = data[0] % 12;  /* 12 test modes */
    uint8_t param = data[1];
    const uint8_t* payload = data + 2;
    size_t payload_size = size - 2;

    /* Output buffer - limit size to avoid OOM */
    int64_t max_values = 10000;
    void* output = malloc((size_t)max_values * 8);  /* 8 bytes per value max */
    if (!output) {
        return 0;
    }

    switch (encoding) {
        case 0: {
            /* RLE - bit_width from param (1-32) */
            int bit_width = (param % 32) + 1;
            (void)carquet_rle_decode_all(payload, payload_size, bit_width,
                                         (uint32_t*)output, max_values);
            break;
        }

        case 1: {
            /* Delta INT32 */
            int32_t num_values = (param % 100) + 1;
            size_t consumed = 0;
            (void)carquet_delta_decode_int32(payload, payload_size,
                                             (int32_t*)output, num_values, &consumed);
            break;
        }

        case 2: {
            /* Delta INT64 */
            int32_t num_values = (param % 100) + 1;
            size_t consumed = 0;
            (void)carquet_delta_decode_int64(payload, payload_size,
                                             (int64_t*)output, num_values, &consumed);
            break;
        }

        case 3: {
            /* Plain INT32 */
            int64_t count = payload_size / 4;
            if (count > max_values) count = max_values;
            (void)carquet_decode_plain_int32(payload, payload_size,
                                             (int32_t*)output, count);
            break;
        }

        case 4: {
            /* Plain INT64 */
            int64_t count = payload_size / 8;
            if (count > max_values) count = max_values;
            (void)carquet_decode_plain_int64(payload, payload_size,
                                             (int64_t*)output, count);
            break;
        }

        case 5: {
            /* Plain DOUBLE */
            int64_t count = payload_size / 8;
            if (count > max_values) count = max_values;
            (void)carquet_decode_plain_double(payload, payload_size,
                                              (double*)output, count);
            break;
        }

        case 6: {
            /* Dictionary INT32 - split payload into dict and indices */
            if (payload_size < 4) break;
            size_t dict_size = (param % 64 + 1) * 4;  /* 4-256 bytes dict */
            if (dict_size >= payload_size) dict_size = payload_size / 2;
            size_t indices_size = payload_size - dict_size;
            int64_t num_values = (param % 100) + 1;
            if (num_values > max_values) num_values = max_values;
            (void)carquet_dictionary_decode_int32(
                payload, dict_size,
                payload + dict_size, indices_size,
                (int32_t*)output, num_values);
            break;
        }

        case 7: {
            /* Dictionary INT64 */
            if (payload_size < 8) break;
            size_t dict_size = (param % 32 + 1) * 8;  /* 8-256 bytes dict */
            if (dict_size >= payload_size) dict_size = payload_size / 2;
            size_t indices_size = payload_size - dict_size;
            int64_t num_values = (param % 100) + 1;
            if (num_values > max_values) num_values = max_values;
            (void)carquet_dictionary_decode_int64(
                payload, dict_size,
                payload + dict_size, indices_size,
                (int64_t*)output, num_values);
            break;
        }

        case 8: {
            /* Dictionary FLOAT */
            if (payload_size < 4) break;
            size_t dict_size = (param % 64 + 1) * 4;
            if (dict_size >= payload_size) dict_size = payload_size / 2;
            size_t indices_size = payload_size - dict_size;
            int64_t num_values = (param % 100) + 1;
            if (num_values > max_values) num_values = max_values;
            (void)carquet_dictionary_decode_float(
                payload, dict_size,
                payload + dict_size, indices_size,
                (float*)output, num_values);
            break;
        }

        case 9: {
            /* Dictionary DOUBLE */
            if (payload_size < 8) break;
            size_t dict_size = (param % 32 + 1) * 8;
            if (dict_size >= payload_size) dict_size = payload_size / 2;
            size_t indices_size = payload_size - dict_size;
            int64_t num_values = (param % 100) + 1;
            if (num_values > max_values) num_values = max_values;
            (void)carquet_dictionary_decode_double(
                payload, dict_size,
                payload + dict_size, indices_size,
                (double*)output, num_values);
            break;
        }

        case 10: {
            /* Byte stream split FLOAT */
            int64_t num_values = payload_size / 4;
            if (num_values > max_values) num_values = max_values;
            if (num_values > 0) {
                (void)carquet_byte_stream_split_decode_float(
                    payload, (size_t)num_values * 4,
                    (float*)output, num_values);
            }
            break;
        }

        case 11: {
            /* Byte stream split DOUBLE */
            int64_t num_values = payload_size / 8;
            if (num_values > max_values) num_values = max_values;
            if (num_values > 0) {
                (void)carquet_byte_stream_split_decode_double(
                    payload, (size_t)num_values * 8,
                    (double*)output, num_values);
            }
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
