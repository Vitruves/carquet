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

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 4) {
        return 0;
    }

    carquet_init();

    /* First byte selects which Thrift structure to decode */
    uint8_t struct_type = data[0] % 4;
    const uint8_t* payload = data + 1;
    size_t payload_size = size - 1;

    /* Create arena for allocations */
    carquet_arena_t arena;
    carquet_arena_init(&arena, 64 * 1024);  /* 64KB arena */

    switch (struct_type) {
        case 0: {
            /* FileMetaData - the main metadata structure */
            parquet_file_metadata_t metadata;
            memset(&metadata, 0, sizeof(metadata));
            carquet_thrift_decode_file_metadata(payload, payload_size,
                                                 &metadata, &arena);
            break;
        }

        case 1: {
            /* PageHeader */
            parquet_page_header_t header;
            memset(&header, 0, sizeof(header));
            size_t consumed;
            carquet_thrift_decode_page_header(payload, payload_size,
                                               &header, &consumed);
            break;
        }

        case 2: {
            /* ColumnMetaData */
            parquet_column_metadata_t col_meta;
            memset(&col_meta, 0, sizeof(col_meta));
            carquet_thrift_decode_column_metadata(payload, payload_size,
                                                   &col_meta, &arena);
            break;
        }

        case 3: {
            /* SchemaElement */
            parquet_schema_element_t element;
            memset(&element, 0, sizeof(element));
            size_t consumed;
            carquet_thrift_decode_schema_element(payload, payload_size,
                                                  &element, &arena, &consumed);
            break;
        }
    }

    carquet_arena_destroy(&arena);
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
