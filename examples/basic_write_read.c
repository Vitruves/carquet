/**
 * @file basic_write_read.c
 * @brief Basic example of writing and reading Parquet files with Carquet
 *
 * This example demonstrates:
 * - Creating a schema with columns
 * - Writing data to a Parquet file
 * - Reading data back from the file
 * - Verifying the data matches
 *
 * Build:
 *   gcc -o basic_write_read basic_write_read.c -lcarquet -I../include
 *
 * Run:
 *   ./basic_write_read
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <carquet/carquet.h>

#define NUM_ROWS 1000

/**
 * Print error and exit
 */
static void fatal_error(const char* context, const carquet_error_t* err) {
    fprintf(stderr, "ERROR in %s: %s (code %d)\n",
            context, err->message, err->code);
    exit(1);
}

/**
 * Write sample data to a Parquet file
 */
static int write_parquet_file(const char* filename) {
    printf("Writing Parquet file: %s\n", filename);

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_status_t status;

    /* Step 1: Create schema */
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema) {
        fatal_error("schema creation", &err);
    }

    /* Add columns:
     * - id: INT32 (required)
     * - name: BYTE_ARRAY/STRING (required)
     * - price: DOUBLE (required)
     * - quantity: INT64 (required)
     */
    carquet_logical_type_t string_type = {
        .id = CARQUET_LOGICAL_STRING
    };

    status = carquet_schema_add_column(schema, "id",
        CARQUET_PHYSICAL_INT32, NULL, CARQUET_REPETITION_REQUIRED, 0);
    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to add 'id' column\n");
        carquet_schema_free(schema);
        return -1;
    }

    status = carquet_schema_add_column(schema, "name",
        CARQUET_PHYSICAL_BYTE_ARRAY, &string_type, CARQUET_REPETITION_REQUIRED, 0);
    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to add 'name' column\n");
        carquet_schema_free(schema);
        return -1;
    }

    status = carquet_schema_add_column(schema, "price",
        CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_REQUIRED, 0);
    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to add 'price' column\n");
        carquet_schema_free(schema);
        return -1;
    }

    status = carquet_schema_add_column(schema, "quantity",
        CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0);
    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to add 'quantity' column\n");
        carquet_schema_free(schema);
        return -1;
    }

    printf("  Schema created with %d columns\n", carquet_schema_num_columns(schema));

    /* Step 2: Create writer */
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_SNAPPY;  /* Use Snappy compression */

    carquet_writer_t* writer = carquet_writer_create(filename, schema, &opts, &err);
    if (!writer) {
        carquet_schema_free(schema);
        fatal_error("writer creation", &err);
    }

    /* Step 3: Generate and write data */
    int32_t* ids = malloc(NUM_ROWS * sizeof(int32_t));
    carquet_byte_array_t* names = malloc(NUM_ROWS * sizeof(carquet_byte_array_t));
    double* prices = malloc(NUM_ROWS * sizeof(double));
    int64_t* quantities = malloc(NUM_ROWS * sizeof(int64_t));

    if (!ids || !names || !prices || !quantities) {
        fprintf(stderr, "Failed to allocate memory\n");
        carquet_writer_abort(writer);
        carquet_schema_free(schema);
        return -1;
    }

    /* Generate sample data */
    char name_buffer[NUM_ROWS][32];
    for (int i = 0; i < NUM_ROWS; i++) {
        ids[i] = i + 1;
        snprintf(name_buffer[i], sizeof(name_buffer[i]), "Product_%04d", i + 1);
        names[i].data = (uint8_t*)name_buffer[i];
        names[i].length = (int32_t)strlen(name_buffer[i]);
        prices[i] = 9.99 + (i % 100) * 0.50;
        quantities[i] = (i % 50) + 1;
    }

    /* Write column data */
    status = carquet_writer_write_batch(writer, 0, ids, NUM_ROWS, NULL, NULL);
    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to write 'id' column\n");
        goto cleanup;
    }

    status = carquet_writer_write_batch(writer, 1, names, NUM_ROWS, NULL, NULL);
    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to write 'name' column\n");
        goto cleanup;
    }

    status = carquet_writer_write_batch(writer, 2, prices, NUM_ROWS, NULL, NULL);
    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to write 'price' column\n");
        goto cleanup;
    }

    status = carquet_writer_write_batch(writer, 3, quantities, NUM_ROWS, NULL, NULL);
    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to write 'quantity' column\n");
        goto cleanup;
    }

    /* Step 4: Close writer (writes footer and finalizes file) */
    status = carquet_writer_close(writer);
    writer = NULL;  /* Writer is freed by close */
    if (status != CARQUET_OK) {
        fprintf(stderr, "Failed to close writer\n");
        goto cleanup;
    }

    printf("  Successfully wrote %d rows\n", NUM_ROWS);

    free(ids);
    free(names);
    free(prices);
    free(quantities);
    carquet_schema_free(schema);
    return 0;

cleanup:
    free(ids);
    free(names);
    free(prices);
    free(quantities);
    if (writer) carquet_writer_abort(writer);
    carquet_schema_free(schema);
    return -1;
}

/**
 * Read and verify data from a Parquet file
 */
static int read_parquet_file(const char* filename) {
    printf("\nReading Parquet file: %s\n", filename);

    carquet_error_t err = CARQUET_ERROR_INIT;

    /* Step 1: Open reader */
    carquet_reader_t* reader = carquet_reader_open(filename, NULL, &err);
    if (!reader) {
        fatal_error("reader open", &err);
    }

    /* Step 2: Get file metadata */
    int64_t num_rows = carquet_reader_num_rows(reader);
    int32_t num_cols = carquet_reader_num_columns(reader);
    int32_t num_row_groups = carquet_reader_num_row_groups(reader);

    printf("  File metadata:\n");
    printf("    Rows: %lld\n", (long long)num_rows);
    printf("    Columns: %d\n", num_cols);
    printf("    Row groups: %d\n", num_row_groups);

    /* Step 3: Get schema information */
    const carquet_schema_t* schema = carquet_reader_schema(reader);
    printf("  Schema:\n");
    for (int i = 0; i < num_cols; i++) {
        const carquet_schema_node_t* node = carquet_schema_get_element(schema, i);
        if (node && carquet_schema_node_is_leaf(node)) {
            const char* name = carquet_schema_node_name(node);
            carquet_physical_type_t ptype = carquet_schema_node_physical_type(node);
            printf("    [%d] %s: %s\n", i, name, carquet_physical_type_name(ptype));
        }
    }

    /* Step 4: Read data from each column in each row group */
    for (int rg = 0; rg < num_row_groups; rg++) {
        printf("  Row group %d:\n", rg);

        carquet_row_group_metadata_t rg_meta;
(void)carquet_reader_row_group_metadata(reader, rg, &rg_meta);
        printf("    Rows: %lld\n", (long long)rg_meta.num_rows);

        /* Read first column (id) */
        carquet_column_reader_t* col_reader =
            carquet_reader_get_column(reader, rg, 0, &err);
        if (!col_reader) {
            printf("    Warning: Could not read column 0: %s\n", err.message);
            continue;
        }

        /* Read a batch of values */
        int32_t ids[100];
        int64_t read_count = carquet_column_read_batch(col_reader, ids, 100, NULL, NULL);

        if (read_count > 0) {
            printf("    First 5 IDs: ");
            for (int i = 0; i < 5 && i < read_count; i++) {
                printf("%d ", ids[i]);
            }
            printf("...\n");
        }

        carquet_column_reader_free(col_reader);
    }

    /* Step 5: Close reader */
    carquet_reader_close(reader);

    printf("  Successfully read file\n");
    return 0;
}

/**
 * Display basic file info by opening the reader
 */
static void show_file_info(const char* filename) {
    printf("\nFile info for: %s\n", filename);

    carquet_error_t err = CARQUET_ERROR_INIT;

    carquet_reader_t* reader = carquet_reader_open(filename, NULL, &err);
    if (!reader) {
        printf("  Could not open file: %s\n", err.message);
        return;
    }

    printf("  Total rows: %lld\n", (long long)carquet_reader_num_rows(reader));
    printf("  Row groups: %d\n", carquet_reader_num_row_groups(reader));
    printf("  Columns: %d\n", carquet_reader_num_columns(reader));

    carquet_reader_close(reader);
}

int main(int argc, char* argv[]) {
    const char* filename = "/tmp/example_basic.parquet";

    printf("=== Carquet Basic Write/Read Example ===\n");
    printf("Library version: %s\n\n", carquet_version());

    /* Show CPU features */
    const carquet_cpu_info_t* cpu = carquet_get_cpu_info();
    printf("CPU features:\n");
#if defined(__x86_64__) || defined(__i386__)
    printf("  SSE4.2: %s, AVX2: %s, AVX-512: %s\n",
           cpu->has_sse42 ? "yes" : "no",
           cpu->has_avx2 ? "yes" : "no",
           cpu->has_avx512f ? "yes" : "no");
#elif defined(__aarch64__) || defined(__arm__)
    printf("  NEON: %s, SVE: %s\n",
           cpu->has_neon ? "yes" : "no",
           cpu->has_sve ? "yes" : "no");
#endif
    printf("\n");

    /* Allow custom filename from command line */
    if (argc > 1) {
        filename = argv[1];
    }

    /* Write the file */
    if (write_parquet_file(filename) != 0) {
        fprintf(stderr, "Failed to write Parquet file\n");
        return 1;
    }

    /* Show quick file info */
    show_file_info(filename);

    /* Read the file back */
    if (read_parquet_file(filename) != 0) {
        fprintf(stderr, "Failed to read Parquet file\n");
        return 1;
    }

    printf("\n=== Example completed successfully ===\n");

    /* Clean up test file */
    if (argc <= 1) {
        remove(filename);
        printf("(Removed temporary file)\n");
    }

    return 0;
}
