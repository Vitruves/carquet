/**
 * @file rle.c
 * @brief RLE/Bit-packing hybrid encoding implementation
 */

#include "rle.h"
#include "core/endian.h"
#include "core/bitpack.h"
#include <string.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

/* ============================================================================
 * Internal Helpers
 * ============================================================================
 */

static int read_varint(const uint8_t* data, size_t size, size_t* pos, uint32_t* out) {
    uint32_t result = 0;
    int shift = 0;
    size_t p = *pos;

    while (p < size && shift < 32) {
        uint8_t byte = data[p++];
        result |= (uint32_t)(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            *pos = p;
            *out = result;
            return 0;
        }
        shift += 7;
    }

    return -1;  /* Truncated or overflow */
}

static bool start_new_run(carquet_rle_decoder_t* dec) {
    if (dec->pos >= dec->size) {
        return false;
    }

    /* Read header */
    uint32_t header;
    if (read_varint(dec->data, dec->size, &dec->pos, &header) < 0) {
        dec->status = CARQUET_ERROR_INVALID_RLE;
        return false;
    }

    if ((header & 1) == 0) {
        /* RLE run */
        dec->in_rle_run = true;
        dec->run_remaining = (int64_t)(header >> 1);

        if (dec->run_remaining == 0) {
            /* Empty run, try next */
            return start_new_run(dec);
        }

        /* Read the repeated value (ceil(bit_width/8) bytes) */
        int value_bytes = (dec->bit_width + 7) / 8;
        if (dec->pos + (size_t)value_bytes > dec->size) {
            dec->status = CARQUET_ERROR_INVALID_RLE;
            return false;
        }

        dec->rle_value = 0;
        for (int i = 0; i < value_bytes; i++) {
            dec->rle_value |= (uint32_t)dec->data[dec->pos++] << (i * 8);
        }
        dec->rle_value &= dec->value_mask;

    } else {
        /* Bit-packed run */
        dec->in_rle_run = false;
        int num_groups = (int)(header >> 1);  /* Number of 8-value groups */
        dec->run_remaining = (int64_t)num_groups * 8;

        if (dec->run_remaining == 0) {
            return start_new_run(dec);
        }

        /* We'll decode 8 values at a time into the buffer */
        dec->bitpack_pos = 0;
        dec->bitpack_count = 0;
    }

    return true;
}

static bool fill_bitpack_buffer(carquet_rle_decoder_t* dec) {
    if (dec->run_remaining <= 0) {
        return false;
    }

    /* Read 8 packed values */
    size_t bytes_needed = (size_t)dec->bit_width;  /* 8 values * bit_width bits = bit_width bytes */
    if (dec->pos + bytes_needed > dec->size) {
        dec->status = CARQUET_ERROR_INVALID_RLE;
        return false;
    }

    carquet_bitunpack8_32(dec->data + dec->pos, dec->bit_width, dec->bitpack_buffer);
    dec->pos += bytes_needed;
    dec->bitpack_pos = 0;
    dec->bitpack_count = 8;

    return true;
}

/* ============================================================================
 * RLE Decoder
 * ============================================================================
 */

void carquet_rle_decoder_init(
    carquet_rle_decoder_t* dec,
    const uint8_t* data,
    size_t size,
    int bit_width) {

    memset(dec, 0, sizeof(*dec));
    dec->data = data;
    dec->size = size;
    dec->bit_width = bit_width;
    dec->value_mask = bit_width >= 32 ? ~0U : (1U << bit_width) - 1;
    dec->status = CARQUET_OK;
}

bool carquet_rle_decoder_has_next(const carquet_rle_decoder_t* dec) {
    if (dec->status != CARQUET_OK) {
        return false;
    }
    if (dec->run_remaining > 0) {
        return true;
    }
    return dec->pos < dec->size;
}

uint32_t carquet_rle_decoder_get(carquet_rle_decoder_t* dec) {
    if (dec->status != CARQUET_OK) {
        return 0;
    }

    /* Need new run? */
    if (dec->run_remaining <= 0) {
        if (!start_new_run(dec)) {
            return 0;
        }
    }

    if (dec->in_rle_run) {
        dec->run_remaining--;
        return dec->rle_value;
    } else {
        /* Bit-packed run */
        if (dec->bitpack_pos >= dec->bitpack_count) {
            if (!fill_bitpack_buffer(dec)) {
                return 0;
            }
        }

        dec->run_remaining--;
        return dec->bitpack_buffer[dec->bitpack_pos++];
    }
}

int64_t carquet_rle_decoder_get_batch(
    carquet_rle_decoder_t* dec,
    uint32_t* output,
    int64_t count) {

    int64_t read = 0;

    while (read < count && carquet_rle_decoder_has_next(dec)) {
        /* Need new run? */
        if (dec->run_remaining <= 0) {
            if (!start_new_run(dec)) {
                break;
            }
        }

        if (dec->in_rle_run) {
            /* Fill with repeated value */
            int64_t to_fill = count - read;
            if (to_fill > dec->run_remaining) {
                to_fill = dec->run_remaining;
            }

            for (int64_t i = 0; i < to_fill; i++) {
                output[read++] = dec->rle_value;
            }
            dec->run_remaining -= to_fill;

        } else {
            /* Bit-packed run */
            while (read < count && dec->run_remaining > 0) {
                if (dec->bitpack_pos >= dec->bitpack_count) {
                    if (!fill_bitpack_buffer(dec)) {
                        break;
                    }
                }

                /* Copy from buffer */
                while (read < count && dec->bitpack_pos < dec->bitpack_count &&
                       dec->run_remaining > 0) {
                    output[read++] = dec->bitpack_buffer[dec->bitpack_pos++];
                    dec->run_remaining--;
                }
            }
        }
    }

    return read;
}

int64_t carquet_rle_decoder_skip(
    carquet_rle_decoder_t* dec,
    int64_t count) {

    int64_t skipped = 0;

    while (skipped < count && carquet_rle_decoder_has_next(dec)) {
        if (dec->run_remaining <= 0) {
            if (!start_new_run(dec)) {
                break;
            }
        }

        int64_t to_skip = count - skipped;
        if (to_skip > dec->run_remaining) {
            to_skip = dec->run_remaining;
        }

        if (dec->in_rle_run) {
            /* Easy - just reduce count */
            skipped += to_skip;
            dec->run_remaining -= to_skip;
        } else {
            /* Need to actually advance through bit-packed data */
            while (to_skip > 0 && dec->run_remaining > 0) {
                if (dec->bitpack_pos >= dec->bitpack_count) {
                    if (!fill_bitpack_buffer(dec)) {
                        break;
                    }
                }

                int64_t can_skip = dec->bitpack_count - dec->bitpack_pos;
                if (can_skip > to_skip) can_skip = to_skip;
                if (can_skip > dec->run_remaining) can_skip = dec->run_remaining;

                dec->bitpack_pos += (int)can_skip;
                dec->run_remaining -= can_skip;
                skipped += can_skip;
                to_skip -= can_skip;
            }
        }
    }

    return skipped;
}

/* ============================================================================
 * RLE Encoder
 * ============================================================================
 */

static void write_varint(carquet_buffer_t* buf, uint32_t value) {
    uint8_t bytes[5];
    int len = 0;

    while (value >= 0x80) {
        bytes[len++] = (uint8_t)((value & 0x7F) | 0x80);
        value >>= 7;
    }
    bytes[len++] = (uint8_t)value;

    carquet_buffer_append(buf, bytes, (size_t)len);
}

static void flush_rle(carquet_rle_encoder_t* enc) {
    if (enc->repeat_count == 0) return;

    /* Write RLE header: (count << 1) | 0 */
    write_varint(enc->buffer, (uint32_t)(enc->repeat_count << 1));

    /* Write value (ceil(bit_width/8) bytes) */
    int value_bytes = (enc->bit_width + 7) / 8;
    uint8_t bytes[4];
    for (int i = 0; i < value_bytes; i++) {
        bytes[i] = (uint8_t)(enc->prev_value >> (i * 8));
    }
    carquet_buffer_append(enc->buffer, bytes, (size_t)value_bytes);

    enc->repeat_count = 0;
}

static void flush_bitpack(carquet_rle_encoder_t* enc) {
    if (enc->bitpack_count == 0) return;

    /* Pad buffer to 8 values */
    while (enc->bitpack_count < 8) {
        enc->bitpack_buffer[enc->bitpack_count++] = 0;
    }

    /* Write bit-packed header: (num_groups << 1) | 1 */
    int num_groups = (int)((enc->bitpack_total + 7) / 8);
    write_varint(enc->buffer, (uint32_t)((num_groups << 1) | 1));

    /* Write packed data for all groups */
    uint8_t packed[32];  /* Max for 32-bit values, 8 values */
    for (int g = 0; g < num_groups; g++) {
        carquet_bitpack8_32(enc->bitpack_buffer, enc->bit_width, packed);
        carquet_buffer_append(enc->buffer, packed, (size_t)enc->bit_width);

        /* Shift remaining values */
        /* Note: This simplified impl assumes we flush after each group */
    }

    enc->bitpack_count = 0;
    enc->bitpack_total = 0;
}

void carquet_rle_encoder_init(
    carquet_rle_encoder_t* enc,
    carquet_buffer_t* buffer,
    int bit_width) {

    memset(enc, 0, sizeof(*enc));
    enc->buffer = buffer;
    enc->bit_width = bit_width;
    enc->status = CARQUET_OK;
}

carquet_status_t carquet_rle_encoder_put(
    carquet_rle_encoder_t* enc,
    uint32_t value) {

    if (enc->status != CARQUET_OK) {
        return enc->status;
    }

    if (!enc->has_prev) {
        enc->prev_value = value;
        enc->repeat_count = 1;
        enc->has_prev = true;
        return CARQUET_OK;
    }

    if (value == enc->prev_value) {
        enc->repeat_count++;
        return CARQUET_OK;
    }

    /* Value changed */
    if (enc->repeat_count >= 8) {
        /* Flush as RLE */
        flush_bitpack(enc);  /* Flush any pending bit-pack */
        flush_rle(enc);
    } else {
        /* Add to bit-pack buffer */
        for (int64_t i = 0; i < enc->repeat_count; i++) {
            enc->bitpack_buffer[enc->bitpack_count++] = enc->prev_value;
            enc->bitpack_total++;

            if (enc->bitpack_count == 8) {
                flush_bitpack(enc);
            }
        }
        enc->repeat_count = 0;
    }

    enc->prev_value = value;
    enc->repeat_count = 1;
    return CARQUET_OK;
}

carquet_status_t carquet_rle_encoder_put_repeat(
    carquet_rle_encoder_t* enc,
    uint32_t value,
    int64_t count) {

    for (int64_t i = 0; i < count; i++) {
        carquet_status_t status = carquet_rle_encoder_put(enc, value);
        if (status != CARQUET_OK) return status;
    }
    return CARQUET_OK;
}

carquet_status_t carquet_rle_encoder_flush(carquet_rle_encoder_t* enc) {
    if (enc->status != CARQUET_OK) {
        return enc->status;
    }

    if (enc->repeat_count >= 8) {
        flush_bitpack(enc);
        flush_rle(enc);
    } else if (enc->repeat_count > 0) {
        for (int64_t i = 0; i < enc->repeat_count; i++) {
            enc->bitpack_buffer[enc->bitpack_count++] = enc->prev_value;
            enc->bitpack_total++;

            if (enc->bitpack_count == 8) {
                flush_bitpack(enc);
            }
        }
        enc->repeat_count = 0;

        /* Flush remaining bit-pack buffer */
        if (enc->bitpack_count > 0) {
            flush_bitpack(enc);
        }
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Convenience Functions
 * ============================================================================
 */

int64_t carquet_rle_decode_all(
    const uint8_t* input,
    size_t input_size,
    int bit_width,
    uint32_t* output,
    int64_t max_values) {

    carquet_rle_decoder_t dec;
    carquet_rle_decoder_init(&dec, input, input_size, bit_width);
    return carquet_rle_decoder_get_batch(&dec, output, max_values);
}

int64_t carquet_rle_decode_levels(
    const uint8_t* input,
    size_t input_size,
    int bit_width,
    int16_t* output,
    int64_t max_values) {

    if (max_values <= 0 || input_size == 0) {
        return 0;
    }

    /* Fast path: decode directly without per-value function calls */
    size_t pos = 0;
    int64_t count = 0;
    uint32_t value_mask = bit_width >= 32 ? ~0U : (1U << bit_width) - 1;
    int value_bytes = (bit_width + 7) / 8;

    while (count < max_values && pos < input_size) {
        /* Read varint header inline */
        uint32_t header = 0;
        int shift = 0;
        while (pos < input_size && shift < 32) {
            uint8_t byte = input[pos++];
            header |= (uint32_t)(byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) break;
            shift += 7;
        }

        if ((header & 1) == 0) {
            /* RLE run: fill output with repeated value */
            int64_t run_length = (int64_t)(header >> 1);
            if (run_length == 0) continue;

            if (pos + (size_t)value_bytes > input_size) break;

            /* Read the repeated value */
            uint32_t rle_value = 0;
            for (int i = 0; i < value_bytes; i++) {
                rle_value |= (uint32_t)input[pos++] << (i * 8);
            }
            rle_value &= value_mask;
            int16_t val16 = (int16_t)rle_value;

            /* Fill output in bulk */
            int64_t to_fill = run_length;
            if (count + to_fill > max_values) {
                to_fill = max_values - count;
            }

            /* Use optimized fill for common case of small values */
            int16_t* dst = output + count;
            int64_t i = 0;

#if defined(__SSE2__)
            /* SSE2: fill 8 int16_t at a time */
            if (to_fill >= 8) {
                __m128i vval = _mm_set1_epi16(val16);
                for (; i + 8 <= to_fill; i += 8) {
                    _mm_storeu_si128((__m128i*)(dst + i), vval);
                }
            }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
            /* NEON: fill 8 int16_t at a time */
            if (to_fill >= 8) {
                int16x8_t vval = vdupq_n_s16(val16);
                for (; i + 8 <= to_fill; i += 8) {
                    vst1q_s16(dst + i, vval);
                }
            }
#endif
            /* Scalar remainder */
            for (; i < to_fill; i++) {
                dst[i] = val16;
            }
            count += to_fill;

        } else {
            /* Bit-packed run: decode 8 values at a time */
            int num_groups = (int)(header >> 1);
            int64_t run_length = (int64_t)num_groups * 8;
            if (run_length == 0) continue;

            size_t bytes_per_group = (size_t)bit_width;

            for (int g = 0; g < num_groups && count < max_values; g++) {
                if (pos + bytes_per_group > input_size) break;

                /* Unpack 8 values */
                uint32_t temp[8];
                carquet_bitunpack8_32(input + pos, bit_width, temp);
                pos += bytes_per_group;

                /* Convert to int16_t and store */
                int64_t to_store = 8;
                if (count + to_store > max_values) {
                    to_store = max_values - count;
                }

#if defined(__SSE2__)
                if (to_store == 8) {
                    /* SSE2: load 2x4 int32_t, pack-saturate to 8 int16_t */
                    __m128i v0 = _mm_loadu_si128((const __m128i*)temp);
                    __m128i v1 = _mm_loadu_si128((const __m128i*)(temp + 4));
                    __m128i packed = _mm_packs_epi32(v0, v1);
                    _mm_storeu_si128((__m128i*)(output + count), packed);
                    count += 8;
                    continue;
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                if (to_store == 8) {
                    /* NEON: load 8 uint32_t, narrow to int16_t */
                    uint32x4_t v0 = vld1q_u32(temp);
                    uint32x4_t v1 = vld1q_u32(temp + 4);
                    int16x4_t n0 = vmovn_s32(vreinterpretq_s32_u32(v0));
                    int16x4_t n1 = vmovn_s32(vreinterpretq_s32_u32(v1));
                    int16x8_t result = vcombine_s16(n0, n1);
                    vst1q_s16(output + count, result);
                    count += 8;
                    continue;
                }
#endif
                for (int64_t i = 0; i < to_store; i++) {
                    output[count++] = (int16_t)temp[i];
                }
            }
        }
    }

    return count;
}

carquet_status_t carquet_rle_encode_all(
    const uint32_t* input,
    int64_t count,
    int bit_width,
    carquet_buffer_t* output) {

    carquet_rle_encoder_t enc;
    carquet_rle_encoder_init(&enc, output, bit_width);

    for (int64_t i = 0; i < count; i++) {
        carquet_status_t status = carquet_rle_encoder_put(&enc, input[i]);
        if (status != CARQUET_OK) return status;
    }

    return carquet_rle_encoder_flush(&enc);
}

carquet_status_t carquet_rle_encode_levels(
    const int16_t* input,
    int64_t count,
    int bit_width,
    carquet_buffer_t* output) {

    carquet_rle_encoder_t enc;
    carquet_rle_encoder_init(&enc, output, bit_width);

    for (int64_t i = 0; i < count; i++) {
        carquet_status_t status = carquet_rle_encoder_put(&enc, (uint32_t)input[i]);
        if (status != CARQUET_OK) return status;
    }

    return carquet_rle_encoder_flush(&enc);
}

int64_t carquet_rle_decode_levels_prefixed(
    const uint8_t* input,
    size_t input_size,
    int bit_width,
    int16_t* output,
    int64_t max_values,
    size_t* bytes_consumed) {

    if (input_size < 4) {
        if (bytes_consumed) *bytes_consumed = 0;
        return -1;
    }

    /* Read 4-byte length prefix (little-endian) */
    uint32_t rle_length = carquet_read_u32_le(input);
    if (4 + rle_length > input_size) {
        if (bytes_consumed) *bytes_consumed = 0;
        return -1;
    }

    int64_t count = carquet_rle_decode_levels(
        input + 4, rle_length, bit_width, output, max_values);

    if (bytes_consumed) {
        *bytes_consumed = 4 + rle_length;
    }

    return count;
}
