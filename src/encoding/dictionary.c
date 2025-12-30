/**
 * @file dictionary.c
 * @brief Dictionary encoding implementation
 *
 * Dictionary encoding stores unique values in a dictionary page,
 * and data pages contain RLE-encoded indices into the dictionary.
 */

#include <carquet/error.h>
#include <carquet/types.h>
#include "rle.h"
#include "core/buffer.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Dictionary Builder
 * ============================================================================
 */

typedef struct dict_entry {
    uint8_t* data;
    size_t size;
    uint32_t index;
    struct dict_entry* next;
} dict_entry_t;

typedef struct {
    dict_entry_t** buckets;
    size_t num_buckets;
    size_t count;

    carquet_buffer_t dict_buffer;  /* Stores dictionary values */
    uint32_t* indices;             /* Maps input index to dict index */
    size_t indices_capacity;
    size_t indices_count;

    size_t value_size;             /* For fixed-size types */
    bool is_variable_length;
} dict_builder_t;

static uint32_t dict_hash(const uint8_t* data, size_t size) {
    uint32_t h = 0x811c9dc5;
    for (size_t i = 0; i < size; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

static carquet_status_t dict_builder_init(dict_builder_t* builder,
                                           size_t value_size,
                                           bool is_variable_length) {
    memset(builder, 0, sizeof(*builder));

    builder->num_buckets = 1024;
    builder->buckets = calloc(builder->num_buckets, sizeof(dict_entry_t*));
    if (!builder->buckets) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    carquet_status_t status = carquet_buffer_init_capacity(&builder->dict_buffer, 4096);
    if (status != CARQUET_OK) {
        free(builder->buckets);
        return status;
    }

    builder->indices_capacity = 1024;
    builder->indices = malloc(builder->indices_capacity * sizeof(uint32_t));
    if (!builder->indices) {
        carquet_buffer_destroy(&builder->dict_buffer);
        free(builder->buckets);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    builder->value_size = value_size;
    builder->is_variable_length = is_variable_length;

    return CARQUET_OK;
}

static void dict_builder_destroy(dict_builder_t* builder) {
    if (builder->buckets) {
        for (size_t i = 0; i < builder->num_buckets; i++) {
            dict_entry_t* entry = builder->buckets[i];
            while (entry) {
                dict_entry_t* next = entry->next;
                free(entry->data);
                free(entry);
                entry = next;
            }
        }
        free(builder->buckets);
    }
    carquet_buffer_destroy(&builder->dict_buffer);
    free(builder->indices);
}

static carquet_status_t dict_builder_add(dict_builder_t* builder,
                                          const uint8_t* value,
                                          size_t value_size) {
    /* Ensure indices array has space */
    if (builder->indices_count >= builder->indices_capacity) {
        size_t new_cap = builder->indices_capacity * 2;
        uint32_t* new_indices = realloc(builder->indices, new_cap * sizeof(uint32_t));
        if (!new_indices) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        builder->indices = new_indices;
        builder->indices_capacity = new_cap;
    }

    /* Look up in hash table */
    uint32_t hash = dict_hash(value, value_size);
    size_t bucket = hash % builder->num_buckets;

    for (dict_entry_t* entry = builder->buckets[bucket]; entry; entry = entry->next) {
        if (entry->size == value_size && memcmp(entry->data, value, value_size) == 0) {
            /* Found existing entry */
            builder->indices[builder->indices_count++] = entry->index;
            return CARQUET_OK;
        }
    }

    /* Add new entry */
    dict_entry_t* new_entry = malloc(sizeof(dict_entry_t));
    if (!new_entry) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    new_entry->data = malloc(value_size);
    if (!new_entry->data) {
        free(new_entry);
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    memcpy(new_entry->data, value, value_size);
    new_entry->size = value_size;
    new_entry->index = (uint32_t)builder->count;
    new_entry->next = builder->buckets[bucket];
    builder->buckets[bucket] = new_entry;

    /* Add to dictionary buffer */
    if (builder->is_variable_length) {
        /* Write length prefix */
        uint32_t len = (uint32_t)value_size;
        carquet_buffer_append_u32_le(&builder->dict_buffer, len);
    }
    carquet_buffer_append(&builder->dict_buffer, value, value_size);

    builder->indices[builder->indices_count++] = new_entry->index;
    builder->count++;

    return CARQUET_OK;
}

/* ============================================================================
 * Dictionary Encoding
 * ============================================================================
 */

static int bit_width_for_count(uint32_t count) {
    if (count == 0) return 0;
    count--; /* Max index */
    int width = 0;
    while (count > 0) {
        width++;
        count >>= 1;
    }
    return width > 0 ? width : 1;
}

carquet_status_t carquet_dictionary_encode_int32(
    const int32_t* values,
    int64_t count,
    carquet_buffer_t* dict_output,
    carquet_buffer_t* indices_output) {

    dict_builder_t builder;
    carquet_status_t status = dict_builder_init(&builder, sizeof(int32_t), false);
    if (status != CARQUET_OK) {
        return status;
    }

    /* Build dictionary */
    for (int64_t i = 0; i < count; i++) {
        status = dict_builder_add(&builder, (const uint8_t*)&values[i], sizeof(int32_t));
        if (status != CARQUET_OK) {
            dict_builder_destroy(&builder);
            return status;
        }
    }

    /* Copy dictionary */
    carquet_buffer_append(dict_output, builder.dict_buffer.data, builder.dict_buffer.size);

    /* Encode indices with RLE */
    int bit_width = bit_width_for_count((uint32_t)builder.count);

    /* Write bit width byte */
    uint8_t bw = (uint8_t)bit_width;
    carquet_buffer_append_byte(indices_output, bw);

    /* RLE encode indices */
    status = carquet_rle_encode_all(builder.indices, count, bit_width, indices_output);

    dict_builder_destroy(&builder);
    return status;
}

carquet_status_t carquet_dictionary_encode_int64(
    const int64_t* values,
    int64_t count,
    carquet_buffer_t* dict_output,
    carquet_buffer_t* indices_output) {

    dict_builder_t builder;
    carquet_status_t status = dict_builder_init(&builder, sizeof(int64_t), false);
    if (status != CARQUET_OK) {
        return status;
    }

    for (int64_t i = 0; i < count; i++) {
        status = dict_builder_add(&builder, (const uint8_t*)&values[i], sizeof(int64_t));
        if (status != CARQUET_OK) {
            dict_builder_destroy(&builder);
            return status;
        }
    }

    carquet_buffer_append(dict_output, builder.dict_buffer.data, builder.dict_buffer.size);

    int bit_width = bit_width_for_count((uint32_t)builder.count);
    uint8_t bw = (uint8_t)bit_width;
    carquet_buffer_append_byte(indices_output, bw);
    status = carquet_rle_encode_all(builder.indices, count, bit_width, indices_output);

    dict_builder_destroy(&builder);
    return status;
}

carquet_status_t carquet_dictionary_encode_float(
    const float* values,
    int64_t count,
    carquet_buffer_t* dict_output,
    carquet_buffer_t* indices_output) {

    dict_builder_t builder;
    carquet_status_t status = dict_builder_init(&builder, sizeof(float), false);
    if (status != CARQUET_OK) {
        return status;
    }

    for (int64_t i = 0; i < count; i++) {
        status = dict_builder_add(&builder, (const uint8_t*)&values[i], sizeof(float));
        if (status != CARQUET_OK) {
            dict_builder_destroy(&builder);
            return status;
        }
    }

    carquet_buffer_append(dict_output, builder.dict_buffer.data, builder.dict_buffer.size);

    int bit_width = bit_width_for_count((uint32_t)builder.count);
    uint8_t bw = (uint8_t)bit_width;
    carquet_buffer_append_byte(indices_output, bw);
    status = carquet_rle_encode_all(builder.indices, count, bit_width, indices_output);

    dict_builder_destroy(&builder);
    return status;
}

carquet_status_t carquet_dictionary_encode_double(
    const double* values,
    int64_t count,
    carquet_buffer_t* dict_output,
    carquet_buffer_t* indices_output) {

    dict_builder_t builder;
    carquet_status_t status = dict_builder_init(&builder, sizeof(double), false);
    if (status != CARQUET_OK) {
        return status;
    }

    for (int64_t i = 0; i < count; i++) {
        status = dict_builder_add(&builder, (const uint8_t*)&values[i], sizeof(double));
        if (status != CARQUET_OK) {
            dict_builder_destroy(&builder);
            return status;
        }
    }

    carquet_buffer_append(dict_output, builder.dict_buffer.data, builder.dict_buffer.size);

    int bit_width = bit_width_for_count((uint32_t)builder.count);
    uint8_t bw = (uint8_t)bit_width;
    carquet_buffer_append_byte(indices_output, bw);
    status = carquet_rle_encode_all(builder.indices, count, bit_width, indices_output);

    dict_builder_destroy(&builder);
    return status;
}

carquet_status_t carquet_dictionary_encode_byte_array(
    const carquet_byte_array_t* values,
    int64_t count,
    carquet_buffer_t* dict_output,
    carquet_buffer_t* indices_output) {

    dict_builder_t builder;
    carquet_status_t status = dict_builder_init(&builder, 0, true);
    if (status != CARQUET_OK) {
        return status;
    }

    for (int64_t i = 0; i < count; i++) {
        status = dict_builder_add(&builder, values[i].data, values[i].length);
        if (status != CARQUET_OK) {
            dict_builder_destroy(&builder);
            return status;
        }
    }

    carquet_buffer_append(dict_output, builder.dict_buffer.data, builder.dict_buffer.size);

    int bit_width = bit_width_for_count((uint32_t)builder.count);
    uint8_t bw = (uint8_t)bit_width;
    carquet_buffer_append_byte(indices_output, bw);
    status = carquet_rle_encode_all(builder.indices, count, bit_width, indices_output);

    dict_builder_destroy(&builder);
    return status;
}

/* ============================================================================
 * Dictionary Decoding
 * ============================================================================
 */

carquet_status_t carquet_dictionary_decode_int32(
    const uint8_t* dict_data,
    size_t dict_size,
    int32_t dict_count,
    const uint8_t* indices_data,
    size_t indices_size,
    int32_t* output,
    int64_t output_count) {

    /* Early validation */
    if (output_count <= 0) {
        return CARQUET_OK;
    }

    if (dict_count <= 0 || dict_data == NULL) {
        return CARQUET_ERROR_DECODE;
    }

    if (dict_size < (size_t)dict_count * sizeof(int32_t)) {
        return CARQUET_ERROR_DECODE;
    }

    /* Read bit width */
    if (indices_size < 1) {
        return CARQUET_ERROR_DECODE;
    }
    int bit_width = indices_data[0];

    /* Decode RLE indices */
    uint32_t* indices = malloc(output_count * sizeof(uint32_t));
    if (!indices) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    int64_t decoded = carquet_rle_decode_all(
        indices_data + 1, indices_size - 1, bit_width, indices, output_count);

    if (decoded < 0 || decoded < output_count) {
        free(indices);
        return CARQUET_ERROR_DECODE;
    }

    /* Look up values */
    const int32_t* dict = (const int32_t*)dict_data;
    for (int64_t i = 0; i < decoded; i++) {
        if ((int32_t)indices[i] >= dict_count) {
            free(indices);
            return CARQUET_ERROR_DECODE;
        }
        output[i] = dict[indices[i]];
    }

    free(indices);
    return CARQUET_OK;
}

carquet_status_t carquet_dictionary_decode_int64(
    const uint8_t* dict_data,
    size_t dict_size,
    int32_t dict_count,
    const uint8_t* indices_data,
    size_t indices_size,
    int64_t* output,
    int64_t output_count) {

    /* Early validation */
    if (output_count <= 0) {
        return CARQUET_OK;
    }

    if (dict_count <= 0 || dict_data == NULL) {
        return CARQUET_ERROR_DECODE;
    }

    if (dict_size < (size_t)dict_count * sizeof(int64_t)) {
        return CARQUET_ERROR_DECODE;
    }

    if (indices_size < 1) {
        return CARQUET_ERROR_DECODE;
    }
    int bit_width = indices_data[0];

    uint32_t* indices = malloc(output_count * sizeof(uint32_t));
    if (!indices) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    int64_t decoded = carquet_rle_decode_all(
        indices_data + 1, indices_size - 1, bit_width, indices, output_count);

    if (decoded < 0 || decoded < output_count) {
        free(indices);
        return CARQUET_ERROR_DECODE;
    }

    const int64_t* dict = (const int64_t*)dict_data;
    for (int64_t i = 0; i < decoded; i++) {
        if ((int32_t)indices[i] >= dict_count) {
            free(indices);
            return CARQUET_ERROR_DECODE;
        }
        output[i] = dict[indices[i]];
    }

    free(indices);
    return CARQUET_OK;
}

carquet_status_t carquet_dictionary_decode_float(
    const uint8_t* dict_data,
    size_t dict_size,
    int32_t dict_count,
    const uint8_t* indices_data,
    size_t indices_size,
    float* output,
    int64_t output_count) {

    /* Early validation */
    if (output_count <= 0) {
        return CARQUET_OK;
    }

    if (dict_count <= 0 || dict_data == NULL) {
        return CARQUET_ERROR_DECODE;
    }

    if (dict_size < (size_t)dict_count * sizeof(float)) {
        return CARQUET_ERROR_DECODE;
    }

    if (indices_size < 1) {
        return CARQUET_ERROR_DECODE;
    }
    int bit_width = indices_data[0];

    uint32_t* indices = malloc(output_count * sizeof(uint32_t));
    if (!indices) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    int64_t decoded = carquet_rle_decode_all(
        indices_data + 1, indices_size - 1, bit_width, indices, output_count);

    if (decoded < 0 || decoded < output_count) {
        free(indices);
        return CARQUET_ERROR_DECODE;
    }

    const float* dict = (const float*)dict_data;
    for (int64_t i = 0; i < decoded; i++) {
        if ((int32_t)indices[i] >= dict_count) {
            free(indices);
            return CARQUET_ERROR_DECODE;
        }
        output[i] = dict[indices[i]];
    }

    free(indices);
    return CARQUET_OK;
}

carquet_status_t carquet_dictionary_decode_double(
    const uint8_t* dict_data,
    size_t dict_size,
    int32_t dict_count,
    const uint8_t* indices_data,
    size_t indices_size,
    double* output,
    int64_t output_count) {

    /* Early validation */
    if (output_count <= 0) {
        return CARQUET_OK;
    }

    if (dict_count <= 0 || dict_data == NULL) {
        return CARQUET_ERROR_DECODE;
    }

    if (dict_size < (size_t)dict_count * sizeof(double)) {
        return CARQUET_ERROR_DECODE;
    }

    if (indices_size < 1) {
        return CARQUET_ERROR_DECODE;
    }
    int bit_width = indices_data[0];

    uint32_t* indices = malloc(output_count * sizeof(uint32_t));
    if (!indices) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    int64_t decoded = carquet_rle_decode_all(
        indices_data + 1, indices_size - 1, bit_width, indices, output_count);

    if (decoded < 0 || decoded < output_count) {
        free(indices);
        return CARQUET_ERROR_DECODE;
    }

    const double* dict = (const double*)dict_data;
    for (int64_t i = 0; i < decoded; i++) {
        if ((int32_t)indices[i] >= dict_count) {
            free(indices);
            return CARQUET_ERROR_DECODE;
        }
        output[i] = dict[indices[i]];
    }

    free(indices);
    return CARQUET_OK;
}
