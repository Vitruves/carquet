/**
 * @file fuzz_writer.c
 * @brief Fuzz target for carquet Parquet writer
 *
 * Tests the writer with random schemas and data, then verifies
 * the output is readable (roundtrip test).
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <carquet/carquet.h>

/* Maximum limits to prevent OOM */
#define MAX_COLUMNS 16
#define MAX_ROWS 1000
#define MAX_STRING_LEN 256

/**
 * Consume bytes from fuzz input
 */
typedef struct {
    const uint8_t* data;
    size_t size;
    size_t pos;
} fuzz_input_t;

static uint8_t consume_byte(fuzz_input_t* input) {
    if (input->pos >= input->size) return 0;
    return input->data[input->pos++];
}

static uint16_t consume_u16(fuzz_input_t* input) {
    uint16_t lo = consume_byte(input);
    uint16_t hi = consume_byte(input);
    return (hi << 8) | lo;
}

static uint32_t consume_u32(fuzz_input_t* input) {
    uint32_t a = consume_byte(input);
    uint32_t b = consume_byte(input);
    uint32_t c = consume_byte(input);
    uint32_t d = consume_byte(input);
    return a | (b << 8) | (c << 16) | (d << 24);
}

static uint64_t consume_u64(fuzz_input_t* input) {
    uint64_t lo = consume_u32(input);
    uint64_t hi = consume_u32(input);
    return lo | (hi << 32);
}

static float consume_float(fuzz_input_t* input) {
    union { uint32_t u; float f; } v;
    v.u = consume_u32(input);
    return v.f;
}

static double consume_double(fuzz_input_t* input) {
    union { uint64_t u; double d; } v;
    v.u = consume_u64(input);
    return v.d;
}

/**
 * Physical types we support for fuzzing (subset for simplicity)
 */
static const carquet_physical_type_t FUZZ_TYPES[] = {
    CARQUET_PHYSICAL_BOOLEAN,
    CARQUET_PHYSICAL_INT32,
    CARQUET_PHYSICAL_INT64,
    CARQUET_PHYSICAL_FLOAT,
    CARQUET_PHYSICAL_DOUBLE,
    CARQUET_PHYSICAL_BYTE_ARRAY,
};
#define NUM_FUZZ_TYPES (sizeof(FUZZ_TYPES) / sizeof(FUZZ_TYPES[0]))

/**
 * Compression codecs to test
 */
#define NUM_FUZZ_CODECS 5
static const carquet_compression_t FUZZ_CODECS[NUM_FUZZ_CODECS] = {
    CARQUET_COMPRESSION_UNCOMPRESSED,
    CARQUET_COMPRESSION_SNAPPY,
    CARQUET_COMPRESSION_GZIP,
    CARQUET_COMPRESSION_LZ4,
    CARQUET_COMPRESSION_ZSTD,
};

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 10) {
        return 0;
    }

    (void)carquet_init();

    fuzz_input_t input = { data, size, 0 };
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Parse fuzzer-controlled parameters */
    uint8_t num_columns = (consume_byte(&input) % MAX_COLUMNS) + 1;
    uint16_t num_rows = (consume_u16(&input) % MAX_ROWS) + 1;
    uint8_t codec_idx = consume_byte(&input) % NUM_FUZZ_CODECS;
    uint8_t nullable_mask = consume_byte(&input);  /* Which columns are nullable */

    carquet_compression_t codec = FUZZ_CODECS[codec_idx];

    /* Create schema */
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        return 0;
    }

    /* Track column types */
    carquet_physical_type_t* col_types = malloc(num_columns * sizeof(carquet_physical_type_t));
    if (!col_types) {
        carquet_schema_free(schema);
        return 0;
    }

    /* Add columns */
    for (int i = 0; i < num_columns; i++) {
        char col_name[32];
        snprintf(col_name, sizeof(col_name), "col_%d", i);

        uint8_t type_idx = consume_byte(&input) % NUM_FUZZ_TYPES;
        col_types[i] = FUZZ_TYPES[type_idx];

        carquet_field_repetition_t rep = (nullable_mask & (1 << (i % 8)))
            ? CARQUET_REPETITION_OPTIONAL
            : CARQUET_REPETITION_REQUIRED;

        carquet_status_t status = carquet_schema_add_column(
            schema, col_name, col_types[i], NULL, rep, 0);

        if (status != CARQUET_OK) {
            free(col_types);
            carquet_schema_free(schema);
            return 0;
        }
    }

    /* Create temp file for output */
    char tmp_path[] = "/tmp/fuzz_writer_XXXXXX";
    int fd = mkstemp(tmp_path);
    if (fd < 0) {
        free(col_types);
        carquet_schema_free(schema);
        return 0;
    }
    close(fd);

    /* Configure writer */
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = codec;
    opts.row_group_size = 64 * 1024;  /* Small row groups for testing */
    opts.page_size = 4 * 1024;

    /* Create writer */
    carquet_writer_t* writer = carquet_writer_create(tmp_path, schema, &opts, &err);
    if (!writer) {
        remove(tmp_path);
        free(col_types);
        carquet_schema_free(schema);
        return 0;
    }

    /* Allocate buffers for each type */
    void* values = malloc(num_rows * 16);  /* 16 bytes max per value */
    int16_t* def_levels = malloc(num_rows * sizeof(int16_t));

    if (!values || !def_levels) {
        free(values);
        free(def_levels);
        carquet_writer_abort(writer);
        remove(tmp_path);
        free(col_types);
        carquet_schema_free(schema);
        return 0;
    }

    /* Write data for each column */
    for (int col = 0; col < num_columns; col++) {
        bool is_nullable = nullable_mask & (1 << (col % 8));

        /* Generate data based on type */
        switch (col_types[col]) {
            case CARQUET_PHYSICAL_BOOLEAN: {
                uint8_t* bools = (uint8_t*)values;
                for (int i = 0; i < num_rows; i++) {
                    bools[i] = consume_byte(&input) & 1;
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                break;
            }
            case CARQUET_PHYSICAL_INT32: {
                int32_t* ints = (int32_t*)values;
                for (int i = 0; i < num_rows; i++) {
                    ints[i] = (int32_t)consume_u32(&input);
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                break;
            }
            case CARQUET_PHYSICAL_INT64: {
                int64_t* longs = (int64_t*)values;
                for (int i = 0; i < num_rows; i++) {
                    longs[i] = (int64_t)consume_u64(&input);
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                break;
            }
            case CARQUET_PHYSICAL_FLOAT: {
                float* floats = (float*)values;
                for (int i = 0; i < num_rows; i++) {
                    floats[i] = consume_float(&input);
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                break;
            }
            case CARQUET_PHYSICAL_DOUBLE: {
                double* doubles = (double*)values;
                for (int i = 0; i < num_rows; i++) {
                    doubles[i] = consume_double(&input);
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                break;
            }
            case CARQUET_PHYSICAL_BYTE_ARRAY: {
                /* For BYTE_ARRAY, we need to use carquet_byte_array_t or similar */
                /* For simplicity, skip this type in the fuzzer or write empty */
                int32_t* ints = (int32_t*)values;
                for (int i = 0; i < num_rows; i++) {
                    ints[i] = (int32_t)consume_u32(&input);
                    def_levels[i] = is_nullable ? (consume_byte(&input) & 1) : 1;
                }
                /* Skip BYTE_ARRAY for now - use INT32 instead */
                col_types[col] = CARQUET_PHYSICAL_INT32;
                break;
            }
            default:
                break;
        }

        /* Write the batch */
        carquet_status_t status = carquet_writer_write_batch(
            writer,
            col,
            values,
            num_rows,
            is_nullable ? def_levels : NULL,
            NULL);

        if (status != CARQUET_OK) {
            /* Write error - abort */
            free(values);
            free(def_levels);
            carquet_writer_abort(writer);
            remove(tmp_path);
            free(col_types);
            carquet_schema_free(schema);
            return 0;
        }
    }

    /* Close writer */
    carquet_status_t close_status = carquet_writer_close(writer);

    free(values);
    free(def_levels);
    free(col_types);
    carquet_schema_free(schema);

    if (close_status != CARQUET_OK) {
        remove(tmp_path);
        return 0;
    }

    /* Roundtrip test: try to read the file back */
    carquet_reader_t* reader = carquet_reader_open(tmp_path, NULL, &err);
    if (reader) {
        /* Verify basic metadata */
        int64_t read_rows = carquet_reader_num_rows(reader);
        int32_t read_cols = carquet_reader_num_columns(reader);

        /* If metadata doesn't match, that's a bug */
        if (read_rows != num_rows || read_cols != num_columns) {
            /* Mismatch! This would indicate a bug */
            carquet_reader_close(reader);
            remove(tmp_path);
            __builtin_trap();
        }

        /* Try to read some data */
        carquet_batch_reader_config_t config;
        carquet_batch_reader_config_init(&config);
        config.batch_size = 100;

        carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &config, &err);
        if (batch_reader) {
            carquet_row_batch_t* batch = NULL;
            while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
                (void)carquet_row_batch_num_rows(batch);
                carquet_row_batch_free(batch);
                batch = NULL;
            }
            carquet_batch_reader_free(batch_reader);
        }

        carquet_reader_close(reader);
    }

    /* Clean up temp file */
    remove(tmp_path);

    return 0;
}

#ifdef AFL_MAIN
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
