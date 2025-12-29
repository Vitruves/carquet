/**
 * @file page_reader.c
 * @brief Page reading implementation
 *
 * Handles reading and decoding of Parquet data pages.
 */

#include <carquet/carquet.h>
#include "reader_internal.h"
#include "thrift/parquet_types.h"
#include "encoding/plain.h"
#include "encoding/rle.h"
#include "core/endian.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* CRC32 verification */
extern uint32_t carquet_crc32(const uint8_t* data, size_t length);

/* Forward declarations for compression functions */
extern carquet_status_t carquet_lz4_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);
extern carquet_status_t carquet_snappy_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);
extern int carquet_gzip_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);
extern int carquet_zstd_decompress(
    const uint8_t* src, size_t src_size,
    uint8_t* dst, size_t dst_capacity, size_t* dst_size);

/* ============================================================================
 * Decompression
 * ============================================================================
 */

static carquet_status_t decompress_page(
    carquet_compression_t codec,
    const uint8_t* compressed,
    size_t compressed_size,
    uint8_t* decompressed,
    size_t decompressed_capacity,
    size_t* decompressed_size) {

    switch (codec) {
        case CARQUET_COMPRESSION_UNCOMPRESSED:
            if (compressed_size > decompressed_capacity) {
                return CARQUET_ERROR_DECOMPRESSION;
            }
            memcpy(decompressed, compressed, compressed_size);
            *decompressed_size = compressed_size;
            return CARQUET_OK;

        case CARQUET_COMPRESSION_SNAPPY:
            return carquet_snappy_decompress(
                compressed, compressed_size,
                decompressed, decompressed_capacity, decompressed_size);

        case CARQUET_COMPRESSION_LZ4:
        case CARQUET_COMPRESSION_LZ4_RAW:
            return carquet_lz4_decompress(
                compressed, compressed_size,
                decompressed, decompressed_capacity, decompressed_size);

        case CARQUET_COMPRESSION_GZIP:
            return carquet_gzip_decompress(
                compressed, compressed_size,
                decompressed, decompressed_capacity, decompressed_size);

        case CARQUET_COMPRESSION_ZSTD:
            return carquet_zstd_decompress(
                compressed, compressed_size,
                decompressed, decompressed_capacity, decompressed_size);

        default:
            return CARQUET_ERROR_UNSUPPORTED_CODEC;
    }
}

/* ============================================================================
 * Utility Functions
 * ============================================================================
 */

static inline int bit_width_for_max(int max_val) {
    if (max_val == 0) return 0;
    int width = 0;
    while (max_val > 0) {
        width++;
        max_val >>= 1;
    }
    return width;
}

/* ============================================================================
 * Level Decoding
 * ============================================================================
 */

static carquet_status_t decode_levels_rle(
    const uint8_t* data,
    size_t data_size,
    int bit_width,
    int32_t num_values,
    int16_t* levels,
    size_t* bytes_consumed) {

    if (bit_width == 0) {
        /* All zeros */
        memset(levels, 0, num_values * sizeof(int16_t));
        *bytes_consumed = 0;
        return CARQUET_OK;
    }

    /* Use the convenience function for decoding levels */
    int64_t decoded = carquet_rle_decode_levels(
        data, data_size, bit_width, levels, num_values);

    if (decoded < 0) {
        return CARQUET_ERROR_DECODE;
    }

    /* Estimate bytes consumed (not perfect, but good enough) */
    *bytes_consumed = data_size;
    return CARQUET_OK;
}

/* ============================================================================
 * Dictionary Page Reading
 * ============================================================================
 */

carquet_status_t carquet_read_dictionary_page(
    carquet_column_reader_t* reader,
    const uint8_t* page_data,
    size_t page_size,
    const parquet_dictionary_page_header_t* header,
    carquet_error_t* error) {

    if (!reader || !page_data || !header) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ARGUMENT, "NULL argument");
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Allocate dictionary storage */
    size_t value_size = 0;
    switch (reader->type) {
        case CARQUET_PHYSICAL_INT32:
        case CARQUET_PHYSICAL_FLOAT:
            value_size = 4;
            break;
        case CARQUET_PHYSICAL_INT64:
        case CARQUET_PHYSICAL_DOUBLE:
            value_size = 8;
            break;
        case CARQUET_PHYSICAL_INT96:
            value_size = 12;
            break;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            value_size = reader->type_length;
            break;
        case CARQUET_PHYSICAL_BYTE_ARRAY:
            /* Variable length - will be handled differently */
            break;
        default:
            break;
    }

    reader->dictionary_count = header->num_values;

    if (reader->type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        /* For variable length, store raw dictionary data */
        reader->dictionary_data = malloc(page_size);
        if (!reader->dictionary_data) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate dictionary");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        memcpy(reader->dictionary_data, page_data, page_size);
        reader->dictionary_size = page_size;
    } else {
        /* Fixed size values */
        size_t dict_size = value_size * header->num_values;
        reader->dictionary_data = malloc(dict_size);
        if (!reader->dictionary_data) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate dictionary");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        memcpy(reader->dictionary_data, page_data, dict_size);
        reader->dictionary_size = dict_size;
    }

    reader->has_dictionary = true;
    return CARQUET_OK;
}

/* ============================================================================
 * Data Page Reading
 * ============================================================================
 */

carquet_status_t carquet_read_data_page_v1(
    carquet_column_reader_t* reader,
    const uint8_t* page_data,
    size_t page_size,
    const parquet_data_page_header_t* header,
    void* values,
    int64_t max_values,
    int16_t* def_levels,
    int16_t* rep_levels,
    int64_t* values_read,
    carquet_error_t* error) {

    const uint8_t* ptr = page_data;
    size_t remaining = page_size;
    size_t bytes_consumed;

    int32_t num_values = header->num_values;
    if (num_values > max_values) {
        num_values = (int32_t)max_values;
    }

    /* Decode repetition levels if needed */
    if (reader->max_rep_level > 0 && rep_levels) {
        /* Read 4-byte length prefix */
        if (remaining < 4) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Truncated rep levels");
            return CARQUET_ERROR_DECODE;
        }
        uint32_t rep_size = carquet_read_u32_le(ptr);
        ptr += 4;
        remaining -= 4;

        if (rep_size > remaining) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Invalid rep level size");
            return CARQUET_ERROR_DECODE;
        }

        int bit_width = bit_width_for_max(reader->max_rep_level);
        carquet_status_t status = decode_levels_rle(
            ptr, rep_size, bit_width, num_values, rep_levels, &bytes_consumed);
        if (status != CARQUET_OK) {
            CARQUET_SET_ERROR(error, status, "Failed to decode rep levels");
            return status;
        }
        ptr += rep_size;
        remaining -= rep_size;
    } else if (rep_levels) {
        memset(rep_levels, 0, num_values * sizeof(int16_t));
    }

    /* Decode definition levels if needed */
    if (reader->max_def_level > 0 && def_levels) {
        /* Read 4-byte length prefix */
        if (remaining < 4) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Truncated def levels");
            return CARQUET_ERROR_DECODE;
        }
        uint32_t def_size = carquet_read_u32_le(ptr);
        ptr += 4;
        remaining -= 4;

        if (def_size > remaining) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Invalid def level size");
            return CARQUET_ERROR_DECODE;
        }

        int bit_width = bit_width_for_max(reader->max_def_level);
        carquet_status_t status = decode_levels_rle(
            ptr, def_size, bit_width, num_values, def_levels, &bytes_consumed);
        if (status != CARQUET_OK) {
            CARQUET_SET_ERROR(error, status, "Failed to decode def levels");
            return status;
        }
        ptr += def_size;
        remaining -= def_size;
    } else if (def_levels) {
        /* Set all to max level (all values present) */
        for (int32_t i = 0; i < num_values; i++) {
            def_levels[i] = reader->max_def_level;
        }
    }

    /* Count non-null values */
    int32_t non_null_count = num_values;
    if (def_levels && reader->max_def_level > 0) {
        non_null_count = 0;
        for (int32_t i = 0; i < num_values; i++) {
            if (def_levels[i] == reader->max_def_level) {
                non_null_count++;
            }
        }
    }

    /* Decode values based on encoding */
    carquet_status_t status = CARQUET_OK;

    switch (header->encoding) {
        case CARQUET_ENCODING_PLAIN:
            {
                int64_t bytes = carquet_decode_plain(
                    ptr, remaining, reader->type, reader->type_length,
                    values, non_null_count);
                if (bytes < 0) {
                    status = CARQUET_ERROR_DECODE;
                }
            }
            break;

        case CARQUET_ENCODING_RLE_DICTIONARY:
        case CARQUET_ENCODING_PLAIN_DICTIONARY:
            if (!reader->has_dictionary) {
                CARQUET_SET_ERROR(error, CARQUET_ERROR_DICTIONARY_NOT_FOUND,
                    "Dictionary encoding without dictionary");
                return CARQUET_ERROR_DICTIONARY_NOT_FOUND;
            }
            /* Decode dictionary indices using RLE */
            {
                /* Read bit width byte */
                if (remaining < 1) {
                    CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Missing bit width");
                    return CARQUET_ERROR_DECODE;
                }
                int bit_width = ptr[0];
                ptr++;
                remaining--;

                /* Decode indices using RLE */
                uint32_t* indices = malloc(non_null_count * sizeof(uint32_t));
                if (!indices) {
                    CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate indices");
                    return CARQUET_ERROR_OUT_OF_MEMORY;
                }

                int64_t decoded = carquet_rle_decode_all(
                    ptr, remaining, bit_width, indices, non_null_count);

                if (decoded < 0) {
                    free(indices);
                    CARQUET_SET_ERROR(error, CARQUET_ERROR_DECODE, "Failed to decode dictionary indices");
                    return CARQUET_ERROR_DECODE;
                }

                /* Look up values from dictionary */
                size_t value_size = 0;
                switch (reader->type) {
                    case CARQUET_PHYSICAL_INT32:
                    case CARQUET_PHYSICAL_FLOAT:
                        value_size = 4;
                        break;
                    case CARQUET_PHYSICAL_INT64:
                    case CARQUET_PHYSICAL_DOUBLE:
                        value_size = 8;
                        break;
                    default:
                        break;
                }

                if (value_size > 0) {
                    uint8_t* out = (uint8_t*)values;
                    for (int32_t i = 0; i < non_null_count; i++) {
                        int32_t idx = (int32_t)indices[i];
                        if (idx < 0 || idx >= reader->dictionary_count) {
                            status = CARQUET_ERROR_DECODE;
                            break;
                        }
                        memcpy(out + i * value_size,
                               reader->dictionary_data + idx * value_size,
                               value_size);
                    }
                }

                free(indices);
            }
            break;

        default:
            CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ENCODING,
                "Unsupported encoding: %d", header->encoding);
            return CARQUET_ERROR_INVALID_ENCODING;
    }

    if (status != CARQUET_OK) {
        CARQUET_SET_ERROR(error, status, "Failed to decode values");
        return status;
    }

    *values_read = num_values;
    return CARQUET_OK;
}

/* ============================================================================
 * Page Reading Entry Point
 * ============================================================================
 */

carquet_status_t carquet_read_next_page(
    carquet_column_reader_t* reader,
    void* values,
    int64_t max_values,
    int16_t* def_levels,
    int16_t* rep_levels,
    int64_t* values_read,
    carquet_error_t* error) {

    if (!reader || !values || !values_read) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_ARGUMENT, "NULL argument");
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_reader_t* file_reader = reader->file_reader;
    FILE* file = file_reader->file;

    /* Get column chunk info */
    const parquet_column_metadata_t* col_meta = reader->col_meta;
    int64_t data_offset = col_meta->data_page_offset;

    /* Check for dictionary page */
    if (col_meta->has_dictionary_page_offset && !reader->has_dictionary) {
        /* Seek to dictionary page */
        if (fseek(file, col_meta->dictionary_page_offset, SEEK_SET) != 0) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_SEEK, "Failed to seek to dictionary");
            return CARQUET_ERROR_FILE_SEEK;
        }

        /* Read page header */
        uint8_t header_buf[256];
        size_t header_read = fread(header_buf, 1, sizeof(header_buf), file);
        if (header_read < 8) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to read dictionary header");
            return CARQUET_ERROR_FILE_READ;
        }

        parquet_page_header_t page_header;
        size_t header_size;
        carquet_status_t status = parquet_parse_page_header(
            header_buf, header_read, &page_header, &header_size, error);
        if (status != CARQUET_OK) {
            return status;
        }

        if (page_header.type != CARQUET_PAGE_DICTIONARY) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_PAGE, "Expected dictionary page");
            return CARQUET_ERROR_INVALID_PAGE;
        }

        /* Seek past header and read page data */
        if (fseek(file, col_meta->dictionary_page_offset + (long)header_size, SEEK_SET) != 0) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_SEEK, "Failed to seek past dict header");
            return CARQUET_ERROR_FILE_SEEK;
        }

        /* Allocate and read compressed data */
        uint8_t* compressed = malloc(page_header.compressed_page_size);
        if (!compressed) {
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate compressed buffer");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        if (fread(compressed, 1, page_header.compressed_page_size, file) !=
            (size_t)page_header.compressed_page_size) {
            free(compressed);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to read dictionary data");
            return CARQUET_ERROR_FILE_READ;
        }

        /* Verify CRC32 if present */
        if (page_header.has_crc) {
            uint32_t computed_crc = carquet_crc32(compressed, page_header.compressed_page_size);
            uint32_t expected_crc = (uint32_t)page_header.crc;
            if (computed_crc != expected_crc) {
                free(compressed);
                CARQUET_SET_ERROR(error, CARQUET_ERROR_CRC_MISMATCH,
                    "Dictionary page CRC mismatch: expected 0x%08X, got 0x%08X",
                    expected_crc, computed_crc);
                return CARQUET_ERROR_CRC_MISMATCH;
            }
        }

        /* Decompress if needed */
        uint8_t* page_data;
        size_t page_size;

        if (col_meta->codec == CARQUET_COMPRESSION_UNCOMPRESSED) {
            page_data = compressed;
            page_size = page_header.compressed_page_size;
        } else {
            page_data = malloc(page_header.uncompressed_page_size);
            if (!page_data) {
                free(compressed);
                CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate decompress buffer");
                return CARQUET_ERROR_OUT_OF_MEMORY;
            }

            status = decompress_page(col_meta->codec,
                compressed, page_header.compressed_page_size,
                page_data, page_header.uncompressed_page_size, &page_size);
            free(compressed);

            if (status != CARQUET_OK) {
                free(page_data);
                CARQUET_SET_ERROR(error, status, "Failed to decompress dictionary");
                return status;
            }
        }

        /* Parse dictionary */
        status = carquet_read_dictionary_page(
            reader, page_data, page_size,
            &page_header.dictionary_page_header, error);

        if (page_data != compressed) {
            free(page_data);
        } else {
            free(compressed);
        }

        if (status != CARQUET_OK) {
            return status;
        }
    }

    /* Seek to data page */
    if (fseek(file, data_offset + reader->current_page, SEEK_SET) != 0) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_SEEK, "Failed to seek to data page");
        return CARQUET_ERROR_FILE_SEEK;
    }

    /* Read page header */
    uint8_t header_buf[256];
    size_t header_read = fread(header_buf, 1, sizeof(header_buf), file);
    if (header_read < 8) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to read page header");
        return CARQUET_ERROR_FILE_READ;
    }

    parquet_page_header_t page_header;
    size_t header_size;
    carquet_status_t status = parquet_parse_page_header(
        header_buf, header_read, &page_header, &header_size, error);
    if (status != CARQUET_OK) {
        return status;
    }

    if (page_header.type != CARQUET_PAGE_DATA && page_header.type != CARQUET_PAGE_DATA_V2) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_INVALID_PAGE, "Expected data page");
        return CARQUET_ERROR_INVALID_PAGE;
    }

    /* Seek past header and read page data */
    if (fseek(file, data_offset + reader->current_page + (long)header_size, SEEK_SET) != 0) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_SEEK, "Failed to seek past header");
        return CARQUET_ERROR_FILE_SEEK;
    }

    /* Allocate and read compressed data */
    uint8_t* compressed = malloc(page_header.compressed_page_size);
    if (!compressed) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate compressed buffer");
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    if (fread(compressed, 1, page_header.compressed_page_size, file) !=
        (size_t)page_header.compressed_page_size) {
        free(compressed);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_READ, "Failed to read page data");
        return CARQUET_ERROR_FILE_READ;
    }

    /* Verify CRC32 if present */
    if (page_header.has_crc) {
        uint32_t computed_crc = carquet_crc32(compressed, page_header.compressed_page_size);
        uint32_t expected_crc = (uint32_t)page_header.crc;
        if (computed_crc != expected_crc) {
            free(compressed);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_CRC_MISMATCH,
                "Page CRC mismatch: expected 0x%08X, got 0x%08X at offset %ld",
                expected_crc, computed_crc, data_offset + reader->current_page);
            return CARQUET_ERROR_CRC_MISMATCH;
        }
    }

    /* Decompress if needed */
    uint8_t* page_data;
    size_t page_size;

    if (col_meta->codec == CARQUET_COMPRESSION_UNCOMPRESSED) {
        page_data = compressed;
        page_size = page_header.compressed_page_size;
    } else {
        page_data = malloc(page_header.uncompressed_page_size);
        if (!page_data) {
            free(compressed);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate decompress buffer");
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        status = decompress_page(col_meta->codec,
            compressed, page_header.compressed_page_size,
            page_data, page_header.uncompressed_page_size, &page_size);
        free(compressed);
        compressed = NULL;

        if (status != CARQUET_OK) {
            free(page_data);
            CARQUET_SET_ERROR(error, status, "Failed to decompress page");
            return status;
        }
    }

    /* Decode page */
    status = carquet_read_data_page_v1(
        reader, page_data, page_size,
        &page_header.data_page_header,
        values, max_values, def_levels, rep_levels,
        values_read, error);

    if (page_data != compressed) {
        free(page_data);
    }
    if (compressed) {
        free(compressed);
    }

    if (status == CARQUET_OK) {
        /* Update position */
        reader->current_page += header_size + page_header.compressed_page_size;
        reader->values_remaining -= *values_read;
    }

    return status;
}
