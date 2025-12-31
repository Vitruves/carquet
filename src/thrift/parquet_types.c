/**
 * @file parquet_types.c
 * @brief Parquet Thrift structure parsing implementation
 */

#include "parquet_types.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Security Limits
 * ============================================================================
 * These limits prevent OOM attacks from malicious files that claim huge counts.
 * Real Parquet files rarely exceed these limits.
 */

#define CARQUET_MAX_SCHEMA_ELEMENTS   10000   /* Max columns/groups in schema */
#define CARQUET_MAX_ROW_GROUPS        100000  /* Max row groups in file */
#define CARQUET_MAX_COLUMNS_PER_RG    10000   /* Max columns per row group */
#define CARQUET_MAX_KEY_VALUE_PAIRS   10000   /* Max metadata key-value pairs */
#define CARQUET_MAX_ENCODINGS         100     /* Max encodings per column */
#define CARQUET_MAX_PATH_ELEMENTS     100     /* Max path depth */
#define CARQUET_MAX_ENCODING_STATS    100     /* Max encoding stats entries */

/* Validate count is within reasonable bounds before allocation */
#define VALIDATE_COUNT(count, max, dec) \
    do { \
        if ((count) < 0 || (count) > (max)) { \
            (dec)->status = CARQUET_ERROR_THRIFT_DECODE; \
            snprintf((dec)->error_message, sizeof((dec)->error_message), \
                "Invalid count %d (max %d)", (int)(count), (int)(max)); \
            return; \
        } \
    } while(0)

#define VALIDATE_COUNT_STATUS(count, max, error) \
    do { \
        if ((count) < 0 || (count) > (max)) { \
            CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_METADATA, \
                "Invalid count %d exceeds limit %d", (int)(count), (int)(max)); \
            return CARQUET_ERROR_INVALID_METADATA; \
        } \
    } while(0)

/* ============================================================================
 * Internal Helpers
 * ============================================================================
 */

static char* arena_strdup_thrift(carquet_arena_t* arena, thrift_decoder_t* dec) {
    int32_t len;
    const uint8_t* data = thrift_read_binary(dec, &len);
    if (!data && len == 0) {
        return carquet_arena_strdup(arena, "");
    }
    if (!data) return NULL;
    return carquet_arena_strndup(arena, (const char*)data, (size_t)len);
}

static uint8_t* arena_bindup_thrift(carquet_arena_t* arena, thrift_decoder_t* dec, int32_t* out_len) {
    int32_t len;
    const uint8_t* data = thrift_read_binary(dec, &len);
    *out_len = len;
    if (!data || len == 0) return NULL;
    return carquet_arena_memdup(arena, data, (size_t)len);
}

/* ============================================================================
 * Statistics Parsing
 * ============================================================================
 */

static void parse_statistics(thrift_decoder_t* dec, carquet_arena_t* arena,
                              parquet_statistics_t* stats) {
    memset(stats, 0, sizeof(*stats));
    thrift_read_struct_begin(dec);

    thrift_type_t type;
    int16_t field_id;

    while (thrift_read_field_begin(dec, &type, &field_id)) {
        switch (field_id) {
            case 1:  /* max (deprecated) */
                stats->max_deprecated = arena_bindup_thrift(arena, dec,
                    &stats->max_deprecated_len);
                break;
            case 2:  /* min (deprecated) */
                stats->min_deprecated = arena_bindup_thrift(arena, dec,
                    &stats->min_deprecated_len);
                break;
            case 3:  /* null_count */
                stats->has_null_count = true;
                stats->null_count = thrift_read_i64(dec);
                break;
            case 4:  /* distinct_count */
                stats->has_distinct_count = true;
                stats->distinct_count = thrift_read_i64(dec);
                break;
            case 5:  /* max_value */
                stats->max_value = arena_bindup_thrift(arena, dec,
                    &stats->max_value_len);
                break;
            case 6:  /* min_value */
                stats->min_value = arena_bindup_thrift(arena, dec,
                    &stats->min_value_len);
                break;
            case 7:  /* is_max_value_exact */
                stats->has_is_max_value_exact = true;
                stats->is_max_value_exact = thrift_read_bool(dec);
                break;
            case 8:  /* is_min_value_exact */
                stats->has_is_min_value_exact = true;
                stats->is_min_value_exact = thrift_read_bool(dec);
                break;
            default:
                thrift_skip(dec, type);
                break;
        }
    }

    thrift_read_struct_end(dec);
}

/* ============================================================================
 * Logical Type Parsing
 * ============================================================================
 */

static void parse_logical_type(thrift_decoder_t* dec, carquet_logical_type_t* lt) {
    memset(lt, 0, sizeof(*lt));
    thrift_read_struct_begin(dec);

    thrift_type_t type;
    int16_t field_id;

    while (thrift_read_field_begin(dec, &type, &field_id)) {
        switch (field_id) {
            case 1:  /* STRING */
                lt->id = CARQUET_LOGICAL_STRING;
                thrift_skip(dec, type);
                break;
            case 2:  /* MAP */
                lt->id = CARQUET_LOGICAL_MAP;
                thrift_skip(dec, type);
                break;
            case 3:  /* LIST */
                lt->id = CARQUET_LOGICAL_LIST;
                thrift_skip(dec, type);
                break;
            case 4:  /* ENUM */
                lt->id = CARQUET_LOGICAL_ENUM;
                thrift_skip(dec, type);
                break;
            case 5:  /* DECIMAL */
                lt->id = CARQUET_LOGICAL_DECIMAL;
                thrift_read_struct_begin(dec);
                while (thrift_read_field_begin(dec, &type, &field_id)) {
                    if (field_id == 1) lt->params.decimal.scale = thrift_read_i32(dec);
                    else if (field_id == 2) lt->params.decimal.precision = thrift_read_i32(dec);
                    else thrift_skip(dec, type);
                }
                thrift_read_struct_end(dec);
                break;
            case 6:  /* DATE */
                lt->id = CARQUET_LOGICAL_DATE;
                thrift_skip(dec, type);
                break;
            case 7:  /* TIME */
                lt->id = CARQUET_LOGICAL_TIME;
                thrift_read_struct_begin(dec);
                while (thrift_read_field_begin(dec, &type, &field_id)) {
                    if (field_id == 1) lt->params.time.is_adjusted_to_utc = thrift_read_bool(dec);
                    else if (field_id == 2) {
                        /* TimeUnit is a union struct */
                        thrift_read_struct_begin(dec);
                        thrift_type_t ut;
                        int16_t uf;
                        while (thrift_read_field_begin(dec, &ut, &uf)) {
                            if (uf == 1) lt->params.time.unit = CARQUET_TIME_UNIT_MILLIS;
                            else if (uf == 2) lt->params.time.unit = CARQUET_TIME_UNIT_MICROS;
                            else if (uf == 3) lt->params.time.unit = CARQUET_TIME_UNIT_NANOS;
                            thrift_skip(dec, ut);
                        }
                        thrift_read_struct_end(dec);
                    }
                    else thrift_skip(dec, type);
                }
                thrift_read_struct_end(dec);
                break;
            case 8:  /* TIMESTAMP */
                lt->id = CARQUET_LOGICAL_TIMESTAMP;
                thrift_read_struct_begin(dec);
                while (thrift_read_field_begin(dec, &type, &field_id)) {
                    if (field_id == 1) lt->params.timestamp.is_adjusted_to_utc = thrift_read_bool(dec);
                    else if (field_id == 2) {
                        thrift_read_struct_begin(dec);
                        thrift_type_t ut;
                        int16_t uf;
                        while (thrift_read_field_begin(dec, &ut, &uf)) {
                            if (uf == 1) lt->params.timestamp.unit = CARQUET_TIME_UNIT_MILLIS;
                            else if (uf == 2) lt->params.timestamp.unit = CARQUET_TIME_UNIT_MICROS;
                            else if (uf == 3) lt->params.timestamp.unit = CARQUET_TIME_UNIT_NANOS;
                            thrift_skip(dec, ut);
                        }
                        thrift_read_struct_end(dec);
                    }
                    else thrift_skip(dec, type);
                }
                thrift_read_struct_end(dec);
                break;
            case 10:  /* INTEGER */
                lt->id = CARQUET_LOGICAL_INTEGER;
                thrift_read_struct_begin(dec);
                while (thrift_read_field_begin(dec, &type, &field_id)) {
                    if (field_id == 1) lt->params.integer.bit_width = (int8_t)thrift_read_byte(dec);
                    else if (field_id == 2) lt->params.integer.is_signed = thrift_read_bool(dec);
                    else thrift_skip(dec, type);
                }
                thrift_read_struct_end(dec);
                break;
            case 11:  /* NULL */
                lt->id = CARQUET_LOGICAL_NULL;
                thrift_skip(dec, type);
                break;
            case 12:  /* JSON */
                lt->id = CARQUET_LOGICAL_JSON;
                thrift_skip(dec, type);
                break;
            case 13:  /* BSON */
                lt->id = CARQUET_LOGICAL_BSON;
                thrift_skip(dec, type);
                break;
            case 14:  /* UUID */
                lt->id = CARQUET_LOGICAL_UUID;
                thrift_skip(dec, type);
                break;
            case 15:  /* FLOAT16 */
                lt->id = CARQUET_LOGICAL_FLOAT16;
                thrift_skip(dec, type);
                break;
            default:
                thrift_skip(dec, type);
                break;
        }
    }

    thrift_read_struct_end(dec);
}

/* ============================================================================
 * Schema Element Parsing
 * ============================================================================
 */

static void parse_schema_element(thrift_decoder_t* dec, carquet_arena_t* arena,
                                  parquet_schema_element_t* elem) {
    memset(elem, 0, sizeof(*elem));
    thrift_read_struct_begin(dec);

    thrift_type_t type;
    int16_t field_id;

    while (thrift_read_field_begin(dec, &type, &field_id)) {
        switch (field_id) {
            case 1:  /* type */
                elem->has_type = true;
                elem->type = (carquet_physical_type_t)thrift_read_i32(dec);
                break;
            case 2:  /* type_length */
                elem->type_length = thrift_read_i32(dec);
                break;
            case 3:  /* repetition_type */
                elem->has_repetition = true;
                elem->repetition_type = (carquet_field_repetition_t)thrift_read_i32(dec);
                break;
            case 4:  /* name */
                elem->name = arena_strdup_thrift(arena, dec);
                break;
            case 5:  /* num_children */
                elem->num_children = thrift_read_i32(dec);
                break;
            case 6:  /* converted_type */
                elem->has_converted_type = true;
                elem->converted_type = (carquet_converted_type_t)thrift_read_i32(dec);
                break;
            case 7:  /* scale */
                elem->scale = thrift_read_i32(dec);
                break;
            case 8:  /* precision */
                elem->precision = thrift_read_i32(dec);
                break;
            case 9:  /* field_id */
                elem->has_field_id = true;
                elem->field_id = thrift_read_i32(dec);
                break;
            case 10:  /* logicalType */
                elem->has_logical_type = true;
                parse_logical_type(dec, &elem->logical_type);
                break;
            default:
                thrift_skip(dec, type);
                break;
        }
    }

    thrift_read_struct_end(dec);
}

/* ============================================================================
 * Column Metadata Parsing
 * ============================================================================
 */

static void parse_column_metadata(thrift_decoder_t* dec, carquet_arena_t* arena,
                                   parquet_column_metadata_t* meta) {
    memset(meta, 0, sizeof(*meta));
    thrift_read_struct_begin(dec);

    thrift_type_t type;
    int16_t field_id;

    while (thrift_read_field_begin(dec, &type, &field_id)) {
        switch (field_id) {
            case 1:  /* type */
                meta->type = (carquet_physical_type_t)thrift_read_i32(dec);
                break;
            case 2: {  /* encodings */
                thrift_type_t elem_type;
                int32_t count;
                thrift_read_list_begin(dec, &elem_type, &count);
                VALIDATE_COUNT(count, CARQUET_MAX_ENCODINGS, dec);
                meta->num_encodings = count;
                meta->encodings = carquet_arena_calloc(arena, count, sizeof(carquet_encoding_t));
                for (int32_t i = 0; i < count; i++) {
                    meta->encodings[i] = (carquet_encoding_t)thrift_read_i32(dec);
                }
                break;
            }
            case 3: {  /* path_in_schema */
                thrift_type_t elem_type;
                int32_t count;
                thrift_read_list_begin(dec, &elem_type, &count);
                VALIDATE_COUNT(count, CARQUET_MAX_PATH_ELEMENTS, dec);
                meta->path_len = count;
                meta->path_in_schema = carquet_arena_calloc(arena, count, sizeof(char*));
                for (int32_t i = 0; i < count; i++) {
                    meta->path_in_schema[i] = arena_strdup_thrift(arena, dec);
                }
                break;
            }
            case 4:  /* codec */
                meta->codec = (carquet_compression_t)thrift_read_i32(dec);
                break;
            case 5:  /* num_values */
                meta->num_values = thrift_read_i64(dec);
                break;
            case 6:  /* total_uncompressed_size */
                meta->total_uncompressed_size = thrift_read_i64(dec);
                break;
            case 7:  /* total_compressed_size */
                meta->total_compressed_size = thrift_read_i64(dec);
                break;
            case 8: {  /* key_value_metadata */
                thrift_type_t elem_type;
                int32_t count;
                thrift_read_list_begin(dec, &elem_type, &count);
                VALIDATE_COUNT(count, CARQUET_MAX_KEY_VALUE_PAIRS, dec);
                meta->num_key_value = count;
                meta->key_value_metadata = carquet_arena_calloc(arena, count,
                    sizeof(parquet_key_value_t));
                for (int32_t i = 0; i < count; i++) {
                    thrift_read_struct_begin(dec);
                    thrift_type_t ft;
                    int16_t fid;
                    while (thrift_read_field_begin(dec, &ft, &fid)) {
                        if (fid == 1) meta->key_value_metadata[i].key = arena_strdup_thrift(arena, dec);
                        else if (fid == 2) meta->key_value_metadata[i].value = arena_strdup_thrift(arena, dec);
                        else thrift_skip(dec, ft);
                    }
                    thrift_read_struct_end(dec);
                }
                break;
            }
            case 9:  /* data_page_offset */
                meta->data_page_offset = thrift_read_i64(dec);
                break;
            case 10:  /* index_page_offset */
                meta->has_index_page_offset = true;
                meta->index_page_offset = thrift_read_i64(dec);
                break;
            case 11:  /* dictionary_page_offset */
                meta->has_dictionary_page_offset = true;
                meta->dictionary_page_offset = thrift_read_i64(dec);
                break;
            case 12:  /* statistics */
                meta->has_statistics = true;
                parse_statistics(dec, arena, &meta->statistics);
                break;
            case 13: {  /* encoding_stats */
                thrift_type_t elem_type;
                int32_t count;
                thrift_read_list_begin(dec, &elem_type, &count);
                VALIDATE_COUNT(count, CARQUET_MAX_ENCODING_STATS, dec);
                meta->num_encoding_stats = count;
                meta->encoding_stats = carquet_arena_calloc(arena, count,
                    sizeof(parquet_page_encoding_stats_t));
                for (int32_t i = 0; i < count; i++) {
                    thrift_read_struct_begin(dec);
                    thrift_type_t ft;
                    int16_t fid;
                    while (thrift_read_field_begin(dec, &ft, &fid)) {
                        if (fid == 1) meta->encoding_stats[i].page_type =
                            (carquet_page_type_t)thrift_read_i32(dec);
                        else if (fid == 2) meta->encoding_stats[i].encoding =
                            (carquet_encoding_t)thrift_read_i32(dec);
                        else if (fid == 3) meta->encoding_stats[i].count = thrift_read_i32(dec);
                        else thrift_skip(dec, ft);
                    }
                    thrift_read_struct_end(dec);
                }
                break;
            }
            case 14:  /* bloom_filter_offset */
                meta->has_bloom_filter_offset = true;
                meta->bloom_filter_offset = thrift_read_i64(dec);
                break;
            case 15:  /* bloom_filter_length */
                meta->has_bloom_filter_length = true;
                meta->bloom_filter_length = thrift_read_i32(dec);
                break;
            default:
                thrift_skip(dec, type);
                break;
        }
    }

    thrift_read_struct_end(dec);
}

/* ============================================================================
 * Column Chunk Parsing
 * ============================================================================
 */

static void parse_column_chunk(thrift_decoder_t* dec, carquet_arena_t* arena,
                                parquet_column_chunk_t* chunk) {
    memset(chunk, 0, sizeof(*chunk));
    thrift_read_struct_begin(dec);

    thrift_type_t type;
    int16_t field_id;

    while (thrift_read_field_begin(dec, &type, &field_id)) {
        switch (field_id) {
            case 1:  /* file_path */
                chunk->file_path = arena_strdup_thrift(arena, dec);
                break;
            case 2:  /* file_offset */
                chunk->file_offset = thrift_read_i64(dec);
                break;
            case 3:  /* meta_data */
                chunk->has_metadata = true;
                parse_column_metadata(dec, arena, &chunk->metadata);
                break;
            case 4:  /* offset_index_offset */
                chunk->has_offset_index_offset = true;
                chunk->offset_index_offset = thrift_read_i64(dec);
                break;
            case 5:  /* offset_index_length */
                chunk->has_offset_index_length = true;
                chunk->offset_index_length = thrift_read_i32(dec);
                break;
            case 6:  /* column_index_offset */
                chunk->has_column_index_offset = true;
                chunk->column_index_offset = thrift_read_i64(dec);
                break;
            case 7:  /* column_index_length */
                chunk->has_column_index_length = true;
                chunk->column_index_length = thrift_read_i32(dec);
                break;
            default:
                thrift_skip(dec, type);
                break;
        }
    }

    thrift_read_struct_end(dec);
}

/* ============================================================================
 * Row Group Parsing
 * ============================================================================
 */

static void parse_row_group(thrift_decoder_t* dec, carquet_arena_t* arena,
                             parquet_row_group_t* rg) {
    memset(rg, 0, sizeof(*rg));
    thrift_read_struct_begin(dec);

    thrift_type_t type;
    int16_t field_id;

    while (thrift_read_field_begin(dec, &type, &field_id)) {
        switch (field_id) {
            case 1: {  /* columns */
                thrift_type_t elem_type;
                int32_t count;
                thrift_read_list_begin(dec, &elem_type, &count);
                VALIDATE_COUNT(count, CARQUET_MAX_COLUMNS_PER_RG, dec);
                rg->num_columns = count;
                rg->columns = carquet_arena_calloc(arena, count,
                    sizeof(parquet_column_chunk_t));
                for (int32_t i = 0; i < count; i++) {
                    parse_column_chunk(dec, arena, &rg->columns[i]);
                }
                break;
            }
            case 2:  /* total_byte_size */
                rg->total_byte_size = thrift_read_i64(dec);
                break;
            case 3:  /* num_rows */
                rg->num_rows = thrift_read_i64(dec);
                break;
            case 4:  /* sorting_columns */
                thrift_skip(dec, type);
                break;
            case 5:  /* file_offset */
                rg->has_file_offset = true;
                rg->file_offset = thrift_read_i64(dec);
                break;
            case 6:  /* total_compressed_size */
                rg->has_total_compressed_size = true;
                rg->total_compressed_size = thrift_read_i64(dec);
                break;
            case 7:  /* ordinal */
                rg->has_ordinal = true;
                rg->ordinal = thrift_read_i16(dec);
                break;
            default:
                thrift_skip(dec, type);
                break;
        }
    }

    thrift_read_struct_end(dec);
}

/* ============================================================================
 * File Metadata Parsing
 * ============================================================================
 */

carquet_status_t parquet_parse_file_metadata(
    const uint8_t* data,
    size_t size,
    carquet_arena_t* arena,
    parquet_file_metadata_t* metadata,
    carquet_error_t* error) {

    if (!data || !arena || !metadata) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ARGUMENT, "NULL argument");
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    memset(metadata, 0, sizeof(*metadata));

    thrift_decoder_t dec;
    thrift_decoder_init(&dec, data, size);

    thrift_read_struct_begin(&dec);

    thrift_type_t type;
    int16_t field_id;

    while (thrift_read_field_begin(&dec, &type, &field_id)) {
        if (thrift_decoder_has_error(&dec)) {
            CARQUET_SET_ERROR(error, dec.status, "%s", dec.error_message);
            return dec.status;
        }

        switch (field_id) {
            case 1:  /* version */
                metadata->version = thrift_read_i32(&dec);
                break;
            case 2: {  /* schema */
                thrift_type_t elem_type;
                int32_t count;
                thrift_read_list_begin(&dec, &elem_type, &count);
                VALIDATE_COUNT_STATUS(count, CARQUET_MAX_SCHEMA_ELEMENTS, error);
                metadata->num_schema_elements = count;
                metadata->schema = carquet_arena_calloc(arena, count,
                    sizeof(parquet_schema_element_t));
                for (int32_t i = 0; i < count; i++) {
                    parse_schema_element(&dec, arena, &metadata->schema[i]);
                }
                break;
            }
            case 3:  /* num_rows */
                metadata->num_rows = thrift_read_i64(&dec);
                break;
            case 4: {  /* row_groups */
                thrift_type_t elem_type;
                int32_t count;
                thrift_read_list_begin(&dec, &elem_type, &count);
                VALIDATE_COUNT_STATUS(count, CARQUET_MAX_ROW_GROUPS, error);
                metadata->num_row_groups = count;
                metadata->row_groups = carquet_arena_calloc(arena, count,
                    sizeof(parquet_row_group_t));
                for (int32_t i = 0; i < count; i++) {
                    parse_row_group(&dec, arena, &metadata->row_groups[i]);
                }
                break;
            }
            case 5: {  /* key_value_metadata */
                thrift_type_t elem_type;
                int32_t count;
                thrift_read_list_begin(&dec, &elem_type, &count);
                VALIDATE_COUNT_STATUS(count, CARQUET_MAX_KEY_VALUE_PAIRS, error);
                metadata->num_key_value = count;
                metadata->key_value_metadata = carquet_arena_calloc(arena, count,
                    sizeof(parquet_key_value_t));
                for (int32_t i = 0; i < count; i++) {
                    thrift_read_struct_begin(&dec);
                    thrift_type_t ft;
                    int16_t fid;
                    while (thrift_read_field_begin(&dec, &ft, &fid)) {
                        if (fid == 1) metadata->key_value_metadata[i].key =
                            arena_strdup_thrift(arena, &dec);
                        else if (fid == 2) metadata->key_value_metadata[i].value =
                            arena_strdup_thrift(arena, &dec);
                        else thrift_skip(&dec, ft);
                    }
                    thrift_read_struct_end(&dec);
                }
                break;
            }
            case 6:  /* created_by */
                metadata->created_by = arena_strdup_thrift(arena, &dec);
                break;
            case 7:  /* column_orders */
            case 8:  /* encryption_algorithm */
            case 9:  /* footer_signing_key_metadata */
                thrift_skip(&dec, type);
                break;
            default:
                thrift_skip(&dec, type);
                break;
        }
    }

    thrift_read_struct_end(&dec);

    if (thrift_decoder_has_error(&dec)) {
        CARQUET_SET_ERROR(error, dec.status, "%s", dec.error_message);
        return dec.status;
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Page Header Parsing
 * ============================================================================
 */

carquet_status_t parquet_parse_page_header(
    const uint8_t* data,
    size_t size,
    parquet_page_header_t* header,
    size_t* bytes_read,
    carquet_error_t* error) {

    if (!data || !header || !bytes_read) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ARGUMENT, "NULL argument");
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    memset(header, 0, sizeof(*header));
    *bytes_read = 0;

    thrift_decoder_t dec;
    thrift_decoder_init(&dec, data, size);

    thrift_read_struct_begin(&dec);

    thrift_type_t type;
    int16_t field_id;

    while (thrift_read_field_begin(&dec, &type, &field_id)) {
        if (thrift_decoder_has_error(&dec)) {
            CARQUET_SET_ERROR(error, dec.status, "%s", dec.error_message);
            return dec.status;
        }

        switch (field_id) {
            case 1:  /* type */
                header->type = (carquet_page_type_t)thrift_read_i32(&dec);
                break;
            case 2:  /* uncompressed_page_size */
                header->uncompressed_page_size = thrift_read_i32(&dec);
                break;
            case 3:  /* compressed_page_size */
                header->compressed_page_size = thrift_read_i32(&dec);
                break;
            case 4:  /* crc */
                header->has_crc = true;
                header->crc = thrift_read_i32(&dec);
                break;
            case 5: {  /* data_page_header */
                thrift_read_struct_begin(&dec);
                thrift_type_t ft;
                int16_t fid;
                while (thrift_read_field_begin(&dec, &ft, &fid)) {
                    switch (fid) {
                        case 1:
                            header->data_page_header.num_values = thrift_read_i32(&dec);
                            break;
                        case 2:
                            header->data_page_header.encoding =
                                (carquet_encoding_t)thrift_read_i32(&dec);
                            break;
                        case 3:
                            header->data_page_header.definition_level_encoding =
                                (carquet_encoding_t)thrift_read_i32(&dec);
                            break;
                        case 4:
                            header->data_page_header.repetition_level_encoding =
                                (carquet_encoding_t)thrift_read_i32(&dec);
                            break;
                        case 5:
                            header->data_page_header.has_statistics = true;
                            /* Skip statistics for now - arena needed */
                            thrift_skip(&dec, ft);
                            break;
                        default:
                            thrift_skip(&dec, ft);
                            break;
                    }
                }
                thrift_read_struct_end(&dec);
                break;
            }
            case 7: {  /* dictionary_page_header */
                thrift_read_struct_begin(&dec);
                thrift_type_t ft;
                int16_t fid;
                while (thrift_read_field_begin(&dec, &ft, &fid)) {
                    switch (fid) {
                        case 1:
                            header->dictionary_page_header.num_values = thrift_read_i32(&dec);
                            break;
                        case 2:
                            header->dictionary_page_header.encoding =
                                (carquet_encoding_t)thrift_read_i32(&dec);
                            break;
                        case 3:
                            header->dictionary_page_header.is_sorted = thrift_read_bool(&dec);
                            break;
                        default:
                            thrift_skip(&dec, ft);
                            break;
                    }
                }
                thrift_read_struct_end(&dec);
                break;
            }
            case 8: {  /* data_page_header_v2 */
                thrift_read_struct_begin(&dec);
                thrift_type_t ft;
                int16_t fid;
                header->data_page_header_v2.is_compressed = true;  /* default */
                while (thrift_read_field_begin(&dec, &ft, &fid)) {
                    switch (fid) {
                        case 1:
                            header->data_page_header_v2.num_values = thrift_read_i32(&dec);
                            break;
                        case 2:
                            header->data_page_header_v2.num_nulls = thrift_read_i32(&dec);
                            break;
                        case 3:
                            header->data_page_header_v2.num_rows = thrift_read_i32(&dec);
                            break;
                        case 4:
                            header->data_page_header_v2.encoding =
                                (carquet_encoding_t)thrift_read_i32(&dec);
                            break;
                        case 5:
                            header->data_page_header_v2.definition_levels_byte_length =
                                thrift_read_i32(&dec);
                            break;
                        case 6:
                            header->data_page_header_v2.repetition_levels_byte_length =
                                thrift_read_i32(&dec);
                            break;
                        case 7:
                            header->data_page_header_v2.is_compressed = thrift_read_bool(&dec);
                            break;
                        case 8:
                            header->data_page_header_v2.has_statistics = true;
                            thrift_skip(&dec, ft);
                            break;
                        default:
                            thrift_skip(&dec, ft);
                            break;
                    }
                }
                thrift_read_struct_end(&dec);
                break;
            }
            default:
                thrift_skip(&dec, type);
                break;
        }
    }

    thrift_read_struct_end(&dec);

    if (thrift_decoder_has_error(&dec)) {
        CARQUET_SET_ERROR(error, dec.status, "%s", dec.error_message);
        return dec.status;
    }

    *bytes_read = dec.reader.pos;
    return CARQUET_OK;
}

/* ============================================================================
 * Cleanup
 * ============================================================================
 */

void parquet_file_metadata_free(parquet_file_metadata_t* metadata) {
    /* Arena handles all allocations, nothing to free here */
    (void)metadata;
}

/* ============================================================================
 * Writing Functions
 * ============================================================================
 */

/**
 * Write statistics to Thrift buffer.
 */
static void write_statistics(thrift_encoder_t* enc, const parquet_statistics_t* stats) {
    thrift_write_struct_begin(enc);

    /* Field 1: max (deprecated) */
    if (stats->max_deprecated && stats->max_deprecated_len > 0) {
        thrift_write_field_header(enc, THRIFT_TYPE_BINARY, 1);
        thrift_write_binary(enc, stats->max_deprecated, stats->max_deprecated_len);
    }

    /* Field 2: min (deprecated) */
    if (stats->min_deprecated && stats->min_deprecated_len > 0) {
        thrift_write_field_header(enc, THRIFT_TYPE_BINARY, 2);
        thrift_write_binary(enc, stats->min_deprecated, stats->min_deprecated_len);
    }

    /* Field 3: null_count */
    if (stats->has_null_count) {
        thrift_write_field_header(enc, THRIFT_TYPE_I64, 3);
        thrift_write_i64(enc, stats->null_count);
    }

    /* Field 4: distinct_count */
    if (stats->has_distinct_count) {
        thrift_write_field_header(enc, THRIFT_TYPE_I64, 4);
        thrift_write_i64(enc, stats->distinct_count);
    }

    /* Field 5: max_value */
    if (stats->max_value && stats->max_value_len > 0) {
        thrift_write_field_header(enc, THRIFT_TYPE_BINARY, 5);
        thrift_write_binary(enc, stats->max_value, stats->max_value_len);
    }

    /* Field 6: min_value */
    if (stats->min_value && stats->min_value_len > 0) {
        thrift_write_field_header(enc, THRIFT_TYPE_BINARY, 6);
        thrift_write_binary(enc, stats->min_value, stats->min_value_len);
    }

    thrift_write_struct_end(enc);
}

/**
 * Write logical type to Thrift buffer.
 */
static void write_logical_type(thrift_encoder_t* enc, const carquet_logical_type_t* lt) {
    thrift_write_struct_begin(enc);

    switch (lt->id) {
        case CARQUET_LOGICAL_STRING:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 1);
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_MAP:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 2);
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_LIST:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 3);
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_ENUM:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 4);
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_DECIMAL:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 5);
            thrift_write_struct_begin(enc);
            thrift_write_field_header(enc, THRIFT_TYPE_I32, 1);
            thrift_write_i32(enc, lt->params.decimal.scale);
            thrift_write_field_header(enc, THRIFT_TYPE_I32, 2);
            thrift_write_i32(enc, lt->params.decimal.precision);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_DATE:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 6);
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_TIME:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 7);
            thrift_write_struct_begin(enc);
            /* Field 1: isAdjustedToUTC */
            thrift_write_field_header(enc, lt->params.time.is_adjusted_to_utc ? 1 : 2, 1);
            /* Field 2: unit (TimeUnit union) */
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 2);
            thrift_write_struct_begin(enc);
            if (lt->params.time.unit == CARQUET_TIME_UNIT_MILLIS) {
                thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 1);
            } else if (lt->params.time.unit == CARQUET_TIME_UNIT_MICROS) {
                thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 2);
            } else {
                thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 3);
            }
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            thrift_write_struct_end(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_TIMESTAMP:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 8);
            thrift_write_struct_begin(enc);
            /* Field 1: isAdjustedToUTC */
            thrift_write_field_header(enc, lt->params.timestamp.is_adjusted_to_utc ? 1 : 2, 1);
            /* Field 2: unit (TimeUnit union) */
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 2);
            thrift_write_struct_begin(enc);
            if (lt->params.timestamp.unit == CARQUET_TIME_UNIT_MILLIS) {
                thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 1);
            } else if (lt->params.timestamp.unit == CARQUET_TIME_UNIT_MICROS) {
                thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 2);
            } else {
                thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 3);
            }
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            thrift_write_struct_end(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_INTEGER:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 10);
            thrift_write_struct_begin(enc);
            thrift_write_field_header(enc, THRIFT_TYPE_BYTE, 1);
            thrift_write_byte(enc, lt->params.integer.bit_width);
            thrift_write_field_header(enc, lt->params.integer.is_signed ? 1 : 2, 2);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_NULL:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 11);
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_JSON:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 12);
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_BSON:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 13);
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_UUID:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 14);
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            break;

        case CARQUET_LOGICAL_FLOAT16:
            thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 15);
            thrift_write_struct_begin(enc);
            thrift_write_struct_end(enc);
            break;

        default:
            break;
    }

    thrift_write_struct_end(enc);
}

/**
 * Write schema element to Thrift buffer.
 */
static void write_schema_element(thrift_encoder_t* enc, const parquet_schema_element_t* elem) {
    thrift_write_struct_begin(enc);

    /* Field 1: type (optional for groups) */
    if (elem->has_type) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 1);
        thrift_write_i32(enc, (int32_t)elem->type);
    }

    /* Field 2: type_length */
    if (elem->type_length > 0) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 2);
        thrift_write_i32(enc, elem->type_length);
    }

    /* Field 3: repetition_type */
    if (elem->has_repetition) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 3);
        thrift_write_i32(enc, (int32_t)elem->repetition_type);
    }

    /* Field 4: name */
    if (elem->name) {
        thrift_write_field_header(enc, THRIFT_TYPE_BINARY, 4);
        thrift_write_string(enc, elem->name);
    }

    /* Field 5: num_children */
    if (elem->num_children > 0) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 5);
        thrift_write_i32(enc, elem->num_children);
    }

    /* Field 6: converted_type */
    if (elem->has_converted_type) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 6);
        thrift_write_i32(enc, (int32_t)elem->converted_type);
    }

    /* Field 7: scale */
    if (elem->scale != 0) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 7);
        thrift_write_i32(enc, elem->scale);
    }

    /* Field 8: precision */
    if (elem->precision != 0) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 8);
        thrift_write_i32(enc, elem->precision);
    }

    /* Field 9: field_id */
    if (elem->has_field_id) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 9);
        thrift_write_i32(enc, elem->field_id);
    }

    /* Field 10: logicalType */
    if (elem->has_logical_type && elem->logical_type.id != CARQUET_LOGICAL_UNKNOWN) {
        thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 10);
        write_logical_type(enc, &elem->logical_type);
    }

    thrift_write_struct_end(enc);
}

/**
 * Write column metadata to Thrift buffer.
 */
static void write_column_metadata(thrift_encoder_t* enc, const parquet_column_metadata_t* meta) {
    thrift_write_struct_begin(enc);

    /* Field 1: type */
    thrift_write_field_header(enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(enc, (int32_t)meta->type);

    /* Field 2: encodings */
    thrift_write_field_header(enc, THRIFT_TYPE_LIST, 2);
    thrift_write_list_begin(enc, THRIFT_TYPE_I32, meta->num_encodings);
    for (int32_t i = 0; i < meta->num_encodings; i++) {
        thrift_write_i32(enc, (int32_t)meta->encodings[i]);
    }

    /* Field 3: path_in_schema */
    thrift_write_field_header(enc, THRIFT_TYPE_LIST, 3);
    thrift_write_list_begin(enc, THRIFT_TYPE_BINARY, meta->path_len);
    for (int32_t i = 0; i < meta->path_len; i++) {
        thrift_write_string(enc, meta->path_in_schema[i]);
    }

    /* Field 4: codec */
    thrift_write_field_header(enc, THRIFT_TYPE_I32, 4);
    thrift_write_i32(enc, (int32_t)meta->codec);

    /* Field 5: num_values */
    thrift_write_field_header(enc, THRIFT_TYPE_I64, 5);
    thrift_write_i64(enc, meta->num_values);

    /* Field 6: total_uncompressed_size */
    thrift_write_field_header(enc, THRIFT_TYPE_I64, 6);
    thrift_write_i64(enc, meta->total_uncompressed_size);

    /* Field 7: total_compressed_size */
    thrift_write_field_header(enc, THRIFT_TYPE_I64, 7);
    thrift_write_i64(enc, meta->total_compressed_size);

    /* Field 9: data_page_offset */
    thrift_write_field_header(enc, THRIFT_TYPE_I64, 9);
    thrift_write_i64(enc, meta->data_page_offset);

    /* Field 10: index_page_offset (optional) */
    if (meta->has_index_page_offset) {
        thrift_write_field_header(enc, THRIFT_TYPE_I64, 10);
        thrift_write_i64(enc, meta->index_page_offset);
    }

    /* Field 11: dictionary_page_offset (optional) */
    if (meta->has_dictionary_page_offset) {
        thrift_write_field_header(enc, THRIFT_TYPE_I64, 11);
        thrift_write_i64(enc, meta->dictionary_page_offset);
    }

    /* Field 12: statistics (optional) */
    if (meta->has_statistics) {
        thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 12);
        write_statistics(enc, &meta->statistics);
    }

    /* Field 14: bloom_filter_offset (optional) */
    if (meta->has_bloom_filter_offset) {
        thrift_write_field_header(enc, THRIFT_TYPE_I64, 14);
        thrift_write_i64(enc, meta->bloom_filter_offset);
    }

    /* Field 15: bloom_filter_length (optional) */
    if (meta->has_bloom_filter_length) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 15);
        thrift_write_i32(enc, meta->bloom_filter_length);
    }

    thrift_write_struct_end(enc);
}

/**
 * Write column chunk to Thrift buffer.
 */
static void write_column_chunk(thrift_encoder_t* enc, const parquet_column_chunk_t* chunk) {
    thrift_write_struct_begin(enc);

    /* Field 1: file_path (optional) */
    if (chunk->file_path) {
        thrift_write_field_header(enc, THRIFT_TYPE_BINARY, 1);
        thrift_write_string(enc, chunk->file_path);
    }

    /* Field 2: file_offset */
    thrift_write_field_header(enc, THRIFT_TYPE_I64, 2);
    thrift_write_i64(enc, chunk->file_offset);

    /* Field 3: meta_data */
    if (chunk->has_metadata) {
        thrift_write_field_header(enc, THRIFT_TYPE_STRUCT, 3);
        write_column_metadata(enc, &chunk->metadata);
    }

    /* Field 4: offset_index_offset (optional) */
    if (chunk->has_offset_index_offset) {
        thrift_write_field_header(enc, THRIFT_TYPE_I64, 4);
        thrift_write_i64(enc, chunk->offset_index_offset);
    }

    /* Field 5: offset_index_length (optional) */
    if (chunk->has_offset_index_length) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 5);
        thrift_write_i32(enc, chunk->offset_index_length);
    }

    /* Field 6: column_index_offset (optional) */
    if (chunk->has_column_index_offset) {
        thrift_write_field_header(enc, THRIFT_TYPE_I64, 6);
        thrift_write_i64(enc, chunk->column_index_offset);
    }

    /* Field 7: column_index_length (optional) */
    if (chunk->has_column_index_length) {
        thrift_write_field_header(enc, THRIFT_TYPE_I32, 7);
        thrift_write_i32(enc, chunk->column_index_length);
    }

    thrift_write_struct_end(enc);
}

/**
 * Write row group to Thrift buffer.
 */
static void write_row_group(thrift_encoder_t* enc, const parquet_row_group_t* rg) {
    thrift_write_struct_begin(enc);

    /* Field 1: columns */
    thrift_write_field_header(enc, THRIFT_TYPE_LIST, 1);
    thrift_write_list_begin(enc, THRIFT_TYPE_STRUCT, rg->num_columns);
    for (int32_t i = 0; i < rg->num_columns; i++) {
        write_column_chunk(enc, &rg->columns[i]);
    }

    /* Field 2: total_byte_size */
    thrift_write_field_header(enc, THRIFT_TYPE_I64, 2);
    thrift_write_i64(enc, rg->total_byte_size);

    /* Field 3: num_rows */
    thrift_write_field_header(enc, THRIFT_TYPE_I64, 3);
    thrift_write_i64(enc, rg->num_rows);

    /* Field 5: file_offset (optional) */
    if (rg->has_file_offset) {
        thrift_write_field_header(enc, THRIFT_TYPE_I64, 5);
        thrift_write_i64(enc, rg->file_offset);
    }

    /* Field 6: total_compressed_size (optional) */
    if (rg->has_total_compressed_size) {
        thrift_write_field_header(enc, THRIFT_TYPE_I64, 6);
        thrift_write_i64(enc, rg->total_compressed_size);
    }

    /* Field 7: ordinal (optional) */
    if (rg->has_ordinal) {
        thrift_write_field_header(enc, THRIFT_TYPE_I16, 7);
        thrift_write_i16(enc, rg->ordinal);
    }

    thrift_write_struct_end(enc);
}

carquet_status_t parquet_write_file_metadata(
    const parquet_file_metadata_t* metadata,
    carquet_buffer_t* buffer,
    carquet_error_t* error) {

    if (!metadata || !buffer) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ARGUMENT, "NULL argument");
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, buffer);

    thrift_write_struct_begin(&enc);

    /* Field 1: version */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, metadata->version);

    /* Field 2: schema */
    thrift_write_field_header(&enc, THRIFT_TYPE_LIST, 2);
    thrift_write_list_begin(&enc, THRIFT_TYPE_STRUCT, metadata->num_schema_elements);
    for (int32_t i = 0; i < metadata->num_schema_elements; i++) {
        write_schema_element(&enc, &metadata->schema[i]);
    }

    /* Field 3: num_rows */
    thrift_write_field_header(&enc, THRIFT_TYPE_I64, 3);
    thrift_write_i64(&enc, metadata->num_rows);

    /* Field 4: row_groups */
    thrift_write_field_header(&enc, THRIFT_TYPE_LIST, 4);
    thrift_write_list_begin(&enc, THRIFT_TYPE_STRUCT, metadata->num_row_groups);
    for (int32_t i = 0; i < metadata->num_row_groups; i++) {
        write_row_group(&enc, &metadata->row_groups[i]);
    }

    /* Field 5: key_value_metadata (optional) */
    if (metadata->key_value_metadata && metadata->num_key_value > 0) {
        thrift_write_field_header(&enc, THRIFT_TYPE_LIST, 5);
        thrift_write_list_begin(&enc, THRIFT_TYPE_STRUCT, metadata->num_key_value);
        for (int32_t i = 0; i < metadata->num_key_value; i++) {
            thrift_write_struct_begin(&enc);
            thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 1);
            thrift_write_string(&enc, metadata->key_value_metadata[i].key);
            if (metadata->key_value_metadata[i].value) {
                thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 2);
                thrift_write_string(&enc, metadata->key_value_metadata[i].value);
            }
            thrift_write_struct_end(&enc);
        }
    }

    /* Field 6: created_by */
    if (metadata->created_by) {
        thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 6);
        thrift_write_string(&enc, metadata->created_by);
    }

    thrift_write_struct_end(&enc);

    if (thrift_encoder_has_error(&enc)) {
        CARQUET_SET_ERROR(error, enc.status, "Failed to encode file metadata");
        return enc.status;
    }

    return CARQUET_OK;
}

carquet_status_t parquet_write_page_header(
    const parquet_page_header_t* header,
    carquet_buffer_t* buffer,
    carquet_error_t* error) {

    if (!header || !buffer) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ARGUMENT, "NULL argument");
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, buffer);

    thrift_write_struct_begin(&enc);

    /* Field 1: type */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, (int32_t)header->type);

    /* Field 2: uncompressed_page_size */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, header->uncompressed_page_size);

    /* Field 3: compressed_page_size */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
    thrift_write_i32(&enc, header->compressed_page_size);

    /* Field 4: crc (optional) */
    if (header->has_crc) {
        thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
        thrift_write_i32(&enc, header->crc);
    }

    /* Type-specific header */
    switch (header->type) {
        case CARQUET_PAGE_DATA:
            thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 5);
            thrift_write_struct_begin(&enc);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
            thrift_write_i32(&enc, header->data_page_header.num_values);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
            thrift_write_i32(&enc, (int32_t)header->data_page_header.encoding);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
            thrift_write_i32(&enc, (int32_t)header->data_page_header.definition_level_encoding);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
            thrift_write_i32(&enc, (int32_t)header->data_page_header.repetition_level_encoding);
            if (header->data_page_header.has_statistics) {
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 5);
                write_statistics(&enc, &header->data_page_header.statistics);
            }
            thrift_write_struct_end(&enc);
            break;

        case CARQUET_PAGE_DATA_V2:
            thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 8);
            thrift_write_struct_begin(&enc);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
            thrift_write_i32(&enc, header->data_page_header_v2.num_values);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
            thrift_write_i32(&enc, header->data_page_header_v2.num_nulls);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
            thrift_write_i32(&enc, header->data_page_header_v2.num_rows);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
            thrift_write_i32(&enc, (int32_t)header->data_page_header_v2.encoding);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 5);
            thrift_write_i32(&enc, header->data_page_header_v2.definition_levels_byte_length);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 6);
            thrift_write_i32(&enc, header->data_page_header_v2.repetition_levels_byte_length);
            thrift_write_field_header(&enc, header->data_page_header_v2.is_compressed ? 1 : 2, 7);
            thrift_write_struct_end(&enc);
            break;

        case CARQUET_PAGE_DICTIONARY:
            thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 7);
            thrift_write_struct_begin(&enc);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
            thrift_write_i32(&enc, header->dictionary_page_header.num_values);
            thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
            thrift_write_i32(&enc, (int32_t)header->dictionary_page_header.encoding);
            thrift_write_field_header(&enc, header->dictionary_page_header.is_sorted ? 1 : 2, 3);
            thrift_write_struct_end(&enc);
            break;

        default:
            break;
    }

    thrift_write_struct_end(&enc);

    if (thrift_encoder_has_error(&enc)) {
        CARQUET_SET_ERROR(error, enc.status, "Failed to encode page header");
        return enc.status;
    }

    return CARQUET_OK;
}
