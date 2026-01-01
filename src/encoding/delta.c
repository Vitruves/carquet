/**
 * @file delta.c
 * @brief DELTA_BINARY_PACKED encoding implementation
 *
 * Reference: https://parquet.apache.org/docs/file-format/data-pages/encodings/
 */

#include <carquet/error.h>
#include <carquet/types.h>
#include "core/bitpack.h"
#include "core/endian.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Constants
 * ============================================================================
 */

#define DELTA_BLOCK_SIZE      128
#define DELTA_MINI_BLOCKS     4
#define DELTA_MINI_BLOCK_SIZE (DELTA_BLOCK_SIZE / DELTA_MINI_BLOCKS)

/* ============================================================================
 * Delta Decoder State
 * ============================================================================
 */

typedef struct {
    const uint8_t* data;
    size_t size;
    size_t pos;

    int32_t block_size;
    int32_t mini_blocks_per_block;
    int32_t total_values;
    int32_t values_decoded;

    int64_t first_value;
    int64_t last_value;

    /* Current block state */
    int64_t min_delta;
    uint8_t bit_widths[DELTA_MINI_BLOCKS];
    int32_t current_mini_block;
    int32_t values_in_mini_block;

    /* Mini-block buffer */
    int64_t mini_block_values[DELTA_MINI_BLOCK_SIZE];
    int32_t mini_block_pos;
} delta_decoder_t;

/* ============================================================================
 * Varint Reading
 * ============================================================================
 */

static size_t read_uleb128(const uint8_t* data, size_t size, uint64_t* value) {
    *value = 0;
    int shift = 0;
    size_t i = 0;

    while (i < size && i < 10) {
        uint8_t b = data[i++];
        *value |= ((uint64_t)(b & 0x7F)) << shift;
        if ((b & 0x80) == 0) {
            return i;
        }
        shift += 7;
    }
    return 0;
}

static int64_t zigzag_decode64(uint64_t n) {
    return (int64_t)((n >> 1) ^ (~(n & 1) + 1));
}

/* ============================================================================
 * Delta Decoder Implementation
 * ============================================================================
 */

static carquet_status_t delta_decoder_init(delta_decoder_t* dec,
                                            const uint8_t* data, size_t size) {
    memset(dec, 0, sizeof(*dec));
    dec->data = data;
    dec->size = size;
    dec->pos = 0;

    /* Read header */
    uint64_t val;
    size_t bytes;

    /* Block size */
    bytes = read_uleb128(data + dec->pos, size - dec->pos, &val);
    if (bytes == 0) return CARQUET_ERROR_DECODE;
    dec->block_size = (int32_t)val;
    dec->pos += bytes;

    /* Mini-blocks per block */
    bytes = read_uleb128(data + dec->pos, size - dec->pos, &val);
    if (bytes == 0) return CARQUET_ERROR_DECODE;
    dec->mini_blocks_per_block = (int32_t)val;
    dec->pos += bytes;

    /* Validate header values to prevent buffer overflows */
    if (dec->mini_blocks_per_block <= 0 || dec->mini_blocks_per_block > DELTA_MINI_BLOCKS) {
        return CARQUET_ERROR_DECODE;
    }
    if (dec->block_size <= 0 || dec->block_size > DELTA_BLOCK_SIZE) {
        return CARQUET_ERROR_DECODE;
    }
    /* mini_block_size = block_size / mini_blocks_per_block must fit in buffer */
    if (dec->block_size / dec->mini_blocks_per_block > DELTA_MINI_BLOCK_SIZE) {
        return CARQUET_ERROR_DECODE;
    }

    /* Total value count */
    bytes = read_uleb128(data + dec->pos, size - dec->pos, &val);
    if (bytes == 0) return CARQUET_ERROR_DECODE;
    dec->total_values = (int32_t)val;
    dec->pos += bytes;

    /* First value (zigzag encoded) */
    bytes = read_uleb128(data + dec->pos, size - dec->pos, &val);
    if (bytes == 0) return CARQUET_ERROR_DECODE;
    dec->first_value = zigzag_decode64(val);
    dec->pos += bytes;

    dec->last_value = dec->first_value;
    dec->current_mini_block = dec->mini_blocks_per_block; /* Force block read */
    dec->mini_block_pos = DELTA_MINI_BLOCK_SIZE; /* Force mini-block read */

    return CARQUET_OK;
}

static carquet_status_t delta_decoder_read_block(delta_decoder_t* dec) {
    if (dec->pos >= dec->size) {
        return CARQUET_ERROR_END_OF_DATA;
    }

    /* Read min delta (zigzag encoded) */
    uint64_t val;
    size_t bytes = read_uleb128(dec->data + dec->pos, dec->size - dec->pos, &val);
    if (bytes == 0) return CARQUET_ERROR_DECODE;
    dec->min_delta = zigzag_decode64(val);
    dec->pos += bytes;

    /* Read bit widths for each mini-block */
    if (dec->pos + dec->mini_blocks_per_block > dec->size) {
        return CARQUET_ERROR_DECODE;
    }
    memcpy(dec->bit_widths, dec->data + dec->pos, dec->mini_blocks_per_block);
    dec->pos += dec->mini_blocks_per_block;

    dec->current_mini_block = 0;
    return CARQUET_OK;
}

static carquet_status_t delta_decoder_read_mini_block(delta_decoder_t* dec) {
    if (dec->current_mini_block >= dec->mini_blocks_per_block) {
        carquet_status_t status = delta_decoder_read_block(dec);
        if (status != CARQUET_OK) return status;
    }

    int bit_width = dec->bit_widths[dec->current_mini_block];
    int mini_block_size = dec->block_size / dec->mini_blocks_per_block;

    if (bit_width == 0) {
        /* All deltas are min_delta */
        for (int i = 0; i < mini_block_size; i++) {
            dec->mini_block_values[i] = dec->min_delta;
        }
    } else if (bit_width <= 32) {
        /* Unpack bit-packed deltas (32-bit) */
        size_t packed_size = (mini_block_size * bit_width + 7) / 8;
        if (dec->pos + packed_size > dec->size) {
            return CARQUET_ERROR_DECODE;
        }

        uint32_t unpacked[DELTA_MINI_BLOCK_SIZE];
        carquet_bitunpack_32(dec->data + dec->pos, mini_block_size, bit_width, unpacked);

        for (int i = 0; i < mini_block_size; i++) {
            /* Use unsigned addition to avoid overflow UB */
            dec->mini_block_values[i] = (int64_t)((uint64_t)dec->min_delta + (uint64_t)unpacked[i]);
        }

        dec->pos += packed_size;
    } else {
        /* Unpack 64-bit values (stored as little-endian bytes) */
        int bytes_per_value = (bit_width + 7) / 8;
        size_t packed_size = mini_block_size * bytes_per_value;
        if (dec->pos + packed_size > dec->size) {
            return CARQUET_ERROR_DECODE;
        }

        for (int i = 0; i < mini_block_size; i++) {
            uint64_t val = 0;
            for (int b = 0; b < bytes_per_value; b++) {
                val |= (uint64_t)dec->data[dec->pos++] << (b * 8);
            }
            /* Use unsigned addition to avoid overflow UB */
            dec->mini_block_values[i] = (int64_t)((uint64_t)dec->min_delta + val);
        }
    }

    dec->current_mini_block++;
    dec->mini_block_pos = 0;
    dec->values_in_mini_block = mini_block_size;

    return CARQUET_OK;
}

static carquet_status_t delta_decoder_next(delta_decoder_t* dec, int64_t* value) {
    if (dec->values_decoded >= dec->total_values) {
        return CARQUET_ERROR_END_OF_DATA;
    }

    /* First value is special */
    if (dec->values_decoded == 0) {
        *value = dec->first_value;
        dec->values_decoded++;
        return CARQUET_OK;
    }

    /* Read from mini-block */
    if (dec->mini_block_pos >= dec->values_in_mini_block) {
        carquet_status_t status = delta_decoder_read_mini_block(dec);
        if (status != CARQUET_OK) return status;
    }

    int64_t delta = dec->mini_block_values[dec->mini_block_pos++];
    /* Use unsigned addition to avoid overflow UB, then reinterpret as signed */
    dec->last_value = (int64_t)((uint64_t)dec->last_value + (uint64_t)delta);
    *value = dec->last_value;
    dec->values_decoded++;

    return CARQUET_OK;
}

/* ============================================================================
 * Public API
 * ============================================================================
 */

carquet_status_t carquet_delta_decode_int32(
    const uint8_t* data,
    size_t data_size,
    int32_t* values,
    int32_t num_values,
    size_t* bytes_consumed) {

    delta_decoder_t dec;
    carquet_status_t status = delta_decoder_init(&dec, data, data_size);
    if (status != CARQUET_OK) {
        return status;
    }

    for (int32_t i = 0; i < num_values; i++) {
        int64_t val;
        status = delta_decoder_next(&dec, &val);
        if (status != CARQUET_OK) {
            return status;
        }
        values[i] = (int32_t)val;
    }

    if (bytes_consumed) {
        *bytes_consumed = dec.pos;
    }

    return CARQUET_OK;
}

carquet_status_t carquet_delta_decode_int64(
    const uint8_t* data,
    size_t data_size,
    int64_t* values,
    int32_t num_values,
    size_t* bytes_consumed) {

    delta_decoder_t dec;
    carquet_status_t status = delta_decoder_init(&dec, data, data_size);
    if (status != CARQUET_OK) {
        return status;
    }

    for (int32_t i = 0; i < num_values; i++) {
        status = delta_decoder_next(&dec, &values[i]);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    if (bytes_consumed) {
        *bytes_consumed = dec.pos;
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Delta Encoder Implementation
 * ============================================================================
 */

typedef struct {
    uint8_t* data;
    size_t capacity;
    size_t pos;

    int32_t block_size;
    int32_t mini_blocks_per_block;
    int32_t values_written;

    int64_t first_value;
    int64_t last_value;

    /* Current block buffer */
    int64_t deltas[DELTA_BLOCK_SIZE];
    int32_t delta_count;
} delta_encoder_t;

static size_t write_uleb128(uint8_t* data, uint64_t value) {
    size_t i = 0;
    while (value >= 0x80) {
        data[i++] = (uint8_t)(value | 0x80);
        value >>= 7;
    }
    data[i++] = (uint8_t)value;
    return i;
}

static uint64_t zigzag_encode64(int64_t n) {
    return ((uint64_t)n << 1) ^ (n >> 63);
}

static int bit_width_required(uint64_t value) {
    if (value == 0) return 0;
    int width = 0;
    while (value > 0) {
        width++;
        value >>= 1;
    }
    return width;
}

static carquet_status_t delta_encoder_init(delta_encoder_t* enc,
                                            uint8_t* data, size_t capacity) {
    memset(enc, 0, sizeof(*enc));
    enc->data = data;
    enc->capacity = capacity;
    enc->block_size = DELTA_BLOCK_SIZE;
    enc->mini_blocks_per_block = DELTA_MINI_BLOCKS;
    return CARQUET_OK;
}

static carquet_status_t delta_encoder_flush_block(delta_encoder_t* enc) {
    if (enc->delta_count == 0) return CARQUET_OK;

    /* Find min delta */
    int64_t min_delta = enc->deltas[0];
    for (int32_t i = 1; i < enc->delta_count; i++) {
        if (enc->deltas[i] < min_delta) {
            min_delta = enc->deltas[i];
        }
    }

    /* Calculate bit widths for each mini-block first to determine space needed */
    int mini_block_size = enc->block_size / enc->mini_blocks_per_block;
    uint8_t bit_widths[DELTA_MINI_BLOCKS];
    size_t packed_bytes_needed = 0;

    for (int mb = 0; mb < enc->mini_blocks_per_block; mb++) {
        uint64_t max_val = 0;
        int start = mb * mini_block_size;
        int end = start + mini_block_size;
        if (end > enc->delta_count) end = enc->delta_count;

        for (int i = start; i < end; i++) {
            /* Use unsigned subtraction to avoid overflow UB */
            uint64_t adjusted = (uint64_t)enc->deltas[i] - (uint64_t)min_delta;
            if (adjusted > max_val) max_val = adjusted;
        }

        bit_widths[mb] = (uint8_t)bit_width_required(max_val);
        if (bit_widths[mb] > 0) {
            /* Calculate bytes needed for this mini-block */
            if (bit_widths[mb] <= 32) {
                /* Bitpacked: mini_block_size values * bit_width / 8 */
                packed_bytes_needed += (size_t)mini_block_size * bit_widths[mb] / 8;
            } else {
                /* Byte-by-byte: mini_block_size values * bytes_per_value */
                packed_bytes_needed += (size_t)mini_block_size * ((bit_widths[mb] + 7) / 8);
            }
        }
    }

    /* Check capacity: min_delta varint (max 10) + bit_widths + packed data */
    size_t bytes_needed = 10 + (size_t)enc->mini_blocks_per_block + packed_bytes_needed;
    if (enc->pos + bytes_needed > enc->capacity) {
        return CARQUET_ERROR_ENCODE;
    }

    /* Write min delta */
    enc->pos += write_uleb128(enc->data + enc->pos, zigzag_encode64(min_delta));

    /* Write bit widths */
    memcpy(enc->data + enc->pos, bit_widths, enc->mini_blocks_per_block);
    enc->pos += enc->mini_blocks_per_block;

    /* Write packed deltas for each mini-block */
    for (int mb = 0; mb < enc->mini_blocks_per_block; mb++) {
        int start = mb * mini_block_size;
        int end = start + mini_block_size;
        if (end > enc->delta_count) end = enc->delta_count;

        if (bit_widths[mb] == 0) continue;

        /* Pack values - use 64-bit packing for large bit widths */
        if (bit_widths[mb] <= 32) {
            uint32_t to_pack[DELTA_MINI_BLOCK_SIZE];
            for (int i = start; i < end; i++) {
                /* Use unsigned subtraction to avoid overflow UB */
                to_pack[i - start] = (uint32_t)((uint64_t)enc->deltas[i] - (uint64_t)min_delta);
            }
            /* Pad with zeros */
            for (int i = end - start; i < mini_block_size; i++) {
                to_pack[i] = 0;
            }
            enc->pos += carquet_bitpack_32(to_pack, mini_block_size,
                                            bit_widths[mb], enc->data + enc->pos);
        } else {
            /* For bit widths > 32, pack directly as bytes (little-endian) */
            int bytes_per_value = (bit_widths[mb] + 7) / 8;
            for (int i = start; i < end; i++) {
                /* Use unsigned subtraction to avoid overflow UB */
                uint64_t adjusted = (uint64_t)enc->deltas[i] - (uint64_t)min_delta;
                for (int b = 0; b < bytes_per_value; b++) {
                    enc->data[enc->pos++] = (uint8_t)(adjusted >> (b * 8));
                }
            }
            /* Pad with zeros */
            for (int i = end - start; i < mini_block_size; i++) {
                for (int b = 0; b < bytes_per_value; b++) {
                    enc->data[enc->pos++] = 0;
                }
            }
        }
    }

    enc->delta_count = 0;
    return CARQUET_OK;
}

carquet_status_t carquet_delta_encode_int32(
    const int32_t* values,
    int32_t num_values,
    uint8_t* data,
    size_t data_capacity,
    size_t* bytes_written) {

    if (num_values == 0) {
        *bytes_written = 0;
        return CARQUET_OK;
    }

    delta_encoder_t enc;
    delta_encoder_init(&enc, data, data_capacity);

    /* Check capacity for header (max 40 bytes for 4 varints) */
    if (data_capacity < 40) {
        return CARQUET_ERROR_ENCODE;
    }

    /* Write header */
    enc.pos += write_uleb128(data + enc.pos, DELTA_BLOCK_SIZE);
    enc.pos += write_uleb128(data + enc.pos, DELTA_MINI_BLOCKS);
    enc.pos += write_uleb128(data + enc.pos, (uint64_t)num_values);
    enc.pos += write_uleb128(data + enc.pos, zigzag_encode64(values[0]));

    enc.first_value = values[0];
    enc.last_value = values[0];
    enc.values_written = 1;

    /* Encode remaining values */
    for (int32_t i = 1; i < num_values; i++) {
        /* Use unsigned subtraction to avoid overflow UB, then reinterpret as signed */
        int64_t delta = (int64_t)((uint64_t)(int64_t)values[i] - (uint64_t)enc.last_value);
        enc.deltas[enc.delta_count++] = delta;
        enc.last_value = values[i];

        if (enc.delta_count == enc.block_size) {
            carquet_status_t status = delta_encoder_flush_block(&enc);
            if (status != CARQUET_OK) return status;
        }
    }

    /* Flush remaining */
    carquet_status_t status = delta_encoder_flush_block(&enc);
    if (status != CARQUET_OK) return status;

    *bytes_written = enc.pos;
    return CARQUET_OK;
}

carquet_status_t carquet_delta_encode_int64(
    const int64_t* values,
    int32_t num_values,
    uint8_t* data,
    size_t data_capacity,
    size_t* bytes_written) {

    if (num_values == 0) {
        *bytes_written = 0;
        return CARQUET_OK;
    }

    delta_encoder_t enc;
    delta_encoder_init(&enc, data, data_capacity);

    /* Check capacity for header (max 40 bytes for 4 varints) */
    if (data_capacity < 40) {
        return CARQUET_ERROR_ENCODE;
    }

    /* Write header */
    enc.pos += write_uleb128(data + enc.pos, DELTA_BLOCK_SIZE);
    enc.pos += write_uleb128(data + enc.pos, DELTA_MINI_BLOCKS);
    enc.pos += write_uleb128(data + enc.pos, (uint64_t)num_values);
    enc.pos += write_uleb128(data + enc.pos, zigzag_encode64(values[0]));

    enc.first_value = values[0];
    enc.last_value = values[0];
    enc.values_written = 1;

    /* Encode remaining values */
    for (int32_t i = 1; i < num_values; i++) {
        /* Use unsigned subtraction to avoid overflow UB, then reinterpret as signed */
        int64_t delta = (int64_t)((uint64_t)values[i] - (uint64_t)enc.last_value);
        enc.deltas[enc.delta_count++] = delta;
        enc.last_value = values[i];

        if (enc.delta_count == enc.block_size) {
            carquet_status_t status = delta_encoder_flush_block(&enc);
            if (status != CARQUET_OK) return status;
        }
    }

    /* Flush remaining */
    carquet_status_t status = delta_encoder_flush_block(&enc);
    if (status != CARQUET_OK) return status;

    *bytes_written = enc.pos;
    return CARQUET_OK;
}
