/**
 * @file fuzz_thrift.c
 * @brief Fuzz target for carquet Thrift compact protocol decoder
 *
 * The Thrift decoder parses Parquet file metadata.
 * This is a critical attack surface for malicious files.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <carquet/carquet.h>

/* Thrift internal headers */
#include "thrift/parquet_types.h"
#include "thrift/thrift_decode.h"
#include "core/arena.h"

/**
 * Test mode 0: Low-level Thrift primitives
 */
static void fuzz_thrift_primitives(const uint8_t* data, size_t size) {
    thrift_decoder_t dec;
    thrift_decoder_init(&dec, data, size);

    /* Try reading various primitive types */
    while (!thrift_decoder_has_error(&dec) && thrift_decoder_remaining(&dec) > 0) {
        size_t remaining = thrift_decoder_remaining(&dec);

        if (remaining >= 1) {
            (void)thrift_read_byte(&dec);
            if (thrift_decoder_has_error(&dec)) break;
        }

        if (remaining >= 1) {
            (void)thrift_read_varint(&dec);
            if (thrift_decoder_has_error(&dec)) break;
        }

        if (remaining >= 1) {
            (void)thrift_read_zigzag(&dec);
            if (thrift_decoder_has_error(&dec)) break;
        }

        if (remaining >= 2) {
            (void)thrift_read_i16(&dec);
            if (thrift_decoder_has_error(&dec)) break;
        }

        if (remaining >= 4) {
            (void)thrift_read_i32(&dec);
            if (thrift_decoder_has_error(&dec)) break;
        }

        if (remaining >= 8) {
            (void)thrift_read_i64(&dec);
            if (thrift_decoder_has_error(&dec)) break;
        }

        if (remaining >= 8) {
            (void)thrift_read_double(&dec);
            if (thrift_decoder_has_error(&dec)) break;
        }

        /* Try reading a binary/string */
        int32_t len = 0;
        const uint8_t* bin = thrift_read_binary(&dec, &len);
        (void)bin;
        if (thrift_decoder_has_error(&dec)) break;

        /* Exit after one round to avoid infinite loops */
        break;
    }
}

/**
 * Test mode 1: Thrift struct parsing
 */
static void fuzz_thrift_struct(const uint8_t* data, size_t size) {
    thrift_decoder_t dec;
    thrift_decoder_init(&dec, data, size);

    thrift_read_struct_begin(&dec);
    if (thrift_decoder_has_error(&dec)) return;

    int field_count = 0;
    const int max_fields = 100;

    thrift_type_t type;
    int16_t field_id;

    while (!thrift_decoder_has_error(&dec) &&
           thrift_read_field_begin(&dec, &type, &field_id) &&
           field_count < max_fields) {

        /* Skip the field value */
        thrift_skip_field(&dec, type);
        field_count++;
    }

    thrift_read_struct_end(&dec);
}

/**
 * Test mode 2: Thrift containers (list/set/map)
 */
static void fuzz_thrift_containers(const uint8_t* data, size_t size) {
    thrift_decoder_t dec;
    thrift_decoder_init(&dec, data, size);

    /* Try list */
    thrift_type_t elem_type;
    int32_t count;

    thrift_read_list_begin(&dec, &elem_type, &count);
    if (!thrift_decoder_has_error(&dec)) {
        /* Limit iterations */
        int max_iter = (count > 100) ? 100 : count;
        for (int i = 0; i < max_iter && !thrift_decoder_has_error(&dec); i++) {
            thrift_skip(&dec, elem_type);
        }
    }

    /* Try map with fresh decoder */
    thrift_decoder_init(&dec, data, size);

    thrift_type_t key_type, value_type;
    thrift_read_map_begin(&dec, &key_type, &value_type, &count);
    if (!thrift_decoder_has_error(&dec)) {
        int max_iter = (count > 50) ? 50 : count;
        for (int i = 0; i < max_iter && !thrift_decoder_has_error(&dec); i++) {
            thrift_skip(&dec, key_type);
            thrift_skip(&dec, value_type);
        }
    }
}

/**
 * Test mode 3: Parquet file metadata parsing
 */
static void fuzz_parquet_metadata(const uint8_t* data, size_t size) {
    carquet_arena_t arena;
    if (carquet_arena_init(&arena) != CARQUET_OK) return;

    parquet_file_metadata_t metadata;
    memset(&metadata, 0, sizeof(metadata));

    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_status_t status = parquet_parse_file_metadata(
        data, size, &arena, &metadata, &err);

    /* Access parsed metadata to ensure it's valid */
    if (status == CARQUET_OK) {
        (void)metadata.version;
        (void)metadata.num_rows;
        (void)metadata.num_row_groups;
        (void)metadata.num_schema_elements;

        /* Access schema elements */
        for (int32_t i = 0; i < metadata.num_schema_elements && i < 100; i++) {
            if (metadata.schema) {
                (void)metadata.schema[i].name;
                (void)metadata.schema[i].type;
            }
        }

        /* Access row groups */
        for (int32_t i = 0; i < metadata.num_row_groups && i < 10; i++) {
            if (metadata.row_groups) {
                (void)metadata.row_groups[i].num_rows;
                (void)metadata.row_groups[i].num_columns;
            }
        }
    }

    parquet_file_metadata_free(&metadata);
    carquet_arena_destroy(&arena);
}

/**
 * Test mode 4: Parquet page header parsing
 */
static void fuzz_parquet_page_header(const uint8_t* data, size_t size) {
    parquet_page_header_t header;
    memset(&header, 0, sizeof(header));

    size_t bytes_read = 0;
    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_status_t status = parquet_parse_page_header(
        data, size, &header, &bytes_read, &err);

    (void)status;

    if (status == CARQUET_OK) {
        (void)header.type;
        (void)header.compressed_page_size;
        (void)header.uncompressed_page_size;
    }
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
            fuzz_thrift_primitives(payload, payload_size);
            break;
        case 1:
            fuzz_thrift_struct(payload, payload_size);
            break;
        case 2:
            fuzz_thrift_containers(payload, payload_size);
            break;
        case 3:
            fuzz_parquet_metadata(payload, payload_size);
            break;
        case 4:
            fuzz_parquet_page_header(payload, payload_size);
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
