/**
 * @file fuzz_reader.c
 * @brief Fuzz target for carquet Parquet reader
 *
 * This fuzz target tests the Parquet file reader with arbitrary input.
 * It's designed to work with both libFuzzer and AFL++.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <carquet/carquet.h>

/**
 * libFuzzer entry point
 */
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    /* Skip trivially small inputs */
    if (size < 12) {
        return 0;
    }

    /* Initialize library (idempotent) */
    (void)carquet_init();

    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Try to open from memory buffer */
    carquet_reader_t* reader = carquet_reader_open_buffer(data, size, NULL, &err);
    if (!reader) {
        /* Invalid file format - expected for most fuzz inputs */
        return 0;
    }

    /* Try to read metadata */
    int64_t num_rows = carquet_reader_num_rows(reader);
    int32_t num_cols = carquet_reader_num_columns(reader);
    int32_t num_row_groups = carquet_reader_num_row_groups(reader);

    (void)num_rows;

    /* Get schema info */
    const carquet_schema_t* schema = carquet_reader_schema(reader);
    if (schema) {
        int32_t num_elements = carquet_schema_num_elements(schema);
        for (int32_t i = 0; i < num_elements && i < 100; i++) {
            const carquet_schema_node_t* node = carquet_schema_get_element(schema, i);
            if (node) {
                const char* name = carquet_schema_node_name(node);
                carquet_physical_type_t ptype = carquet_schema_node_physical_type(node);
                (void)name;
                (void)ptype;
            }
        }
    }

    /* Try to read data using batch reader (simpler API) */
    carquet_batch_reader_config_t config;
    carquet_batch_reader_config_init(&config);
    config.batch_size = 1000;

    carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &config, &err);
    if (batch_reader) {
        carquet_row_batch_t* batch = NULL;
        int batch_count = 0;

        while (batch_count < 10 &&
               carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
            for (int32_t col = 0; col < num_cols && col < 100; col++) {
                const void* data_ptr;
                const uint8_t* nulls;
                int64_t count;
                (void)carquet_row_batch_column(batch, col, &data_ptr, &nulls, &count);
            }
            carquet_row_batch_free(batch);
            batch = NULL;
            batch_count++;
        }

        carquet_batch_reader_free(batch_reader);
    }

    /* Also try low-level column reader API */
    for (int32_t rg = 0; rg < num_row_groups && rg < 5; rg++) {
        carquet_row_group_metadata_t rg_meta;
        if (carquet_reader_row_group_metadata(reader, rg, &rg_meta) != CARQUET_OK) {
            continue;
        }

        int64_t rg_rows = rg_meta.num_rows;
        /* Clamp to reasonable bounds (negative or huge values from malicious files) */
        if (rg_rows <= 0 || rg_rows > 10000) rg_rows = 10000;

        for (int32_t col = 0; col < num_cols && col < 50; col++) {
            carquet_column_reader_t* col_reader =
                carquet_reader_get_column(reader, rg, col, &err);
            if (!col_reader) {
                continue;
            }

            /* Allocate generic buffer (use 16 bytes per value to cover all types) */
            void* values = malloc((size_t)rg_rows * 16);
            int16_t* def_levels = malloc((size_t)rg_rows * sizeof(int16_t));
            int16_t* rep_levels = malloc((size_t)rg_rows * sizeof(int16_t));

            if (values && def_levels && rep_levels) {
                int64_t values_read = carquet_column_read_batch(
                    col_reader, values, rg_rows, def_levels, rep_levels);
                (void)values_read;
            }

            free(values);
            free(def_levels);
            free(rep_levels);
            carquet_column_reader_free(col_reader);
        }
    }

    carquet_reader_close(reader);
    return 0;
}

#ifdef AFL_MAIN
/* AFL++ standalone main */
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
