/**
 * @file neon_ops.c
 * @brief NEON optimized operations for ARM processors
 *
 * Provides comprehensive SIMD-accelerated implementations of:
 * - Bit unpacking for ALL bit widths (1-32 bits)
 * - Byte stream split/merge for floats AND doubles
 * - Delta decoding (prefix sums) for i32/i64
 * - Dictionary gather operations with prefetching
 * - Boolean packing/unpacking
 * - Run-length detection
 * - CRC32C acceleration
 * - Optimized memory operations
 *
 * All functions are optimized for Apple Silicon and AArch64 NEON.
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__aarch64__) || defined(__arm__)
#ifdef __ARM_NEON

#include <arm_neon.h>

/* ============================================================================
 * Bit Unpacking - NEON Optimized (ALL bit widths)
 * ============================================================================
 */

/**
 * Unpack 32 1-bit values using NEON.
 * Highly optimized using NEON bit manipulation.
 */
void carquet_neon_bitunpack32_1bit(const uint8_t* input, uint32_t* values) {
    /* For each byte, extract 8 bits using NEON */
    for (int b = 0; b < 4; b++) {
        uint8_t byte_val = input[b];

        /* Create 8 copies of the byte */
        uint8x8_t byte_vec = vdup_n_u8(byte_val);

        /* Bit masks: 1, 2, 4, 8, 16, 32, 64, 128 */
        static const uint8_t bit_masks[8] = {1, 2, 4, 8, 16, 32, 64, 128};
        uint8x8_t masks = vld1_u8(bit_masks);

        /* AND with masks and compare to get 0xFF or 0x00 */
        uint8x8_t masked = vand_u8(byte_vec, masks);
        uint8x8_t cmp = vceq_u8(masked, masks);

        /* Convert 0xFF -> 1 by shifting right 7 and negating would be wrong;
           instead convert directly */
        uint8x8_t ones = vand_u8(cmp, vdup_n_u8(1));

        /* Widen to 32-bit */
        uint16x8_t wide16 = vmovl_u8(ones);
        uint32x4_t lo32 = vmovl_u16(vget_low_u16(wide16));
        uint32x4_t hi32 = vmovl_u16(vget_high_u16(wide16));

        vst1q_u32(values + b * 8, lo32);
        vst1q_u32(values + b * 8 + 4, hi32);
    }
}

/**
 * Unpack 16 2-bit values using NEON.
 */
void carquet_neon_bitunpack16_2bit(const uint8_t* input, uint32_t* values) {
    /* Process each byte: extract 4 x 2-bit values */
    for (int b = 0; b < 4; b++) {
        uint8_t byte_val = input[b];

        /* Create 4 copies and shift/mask */
        uint8x8_t v = vdup_n_u8(byte_val);

        /* Shifts: 0, 2, 4, 6 - extract each 2-bit pair */
        uint8x8_t shift0 = vand_u8(v, vdup_n_u8(0x03));
        uint8x8_t shift1 = vand_u8(vshr_n_u8(v, 2), vdup_n_u8(0x03));
        uint8x8_t shift2 = vand_u8(vshr_n_u8(v, 4), vdup_n_u8(0x03));
        uint8x8_t shift3 = vshr_n_u8(v, 6);

        values[b * 4 + 0] = vget_lane_u8(shift0, 0);
        values[b * 4 + 1] = vget_lane_u8(shift1, 0);
        values[b * 4 + 2] = vget_lane_u8(shift2, 0);
        values[b * 4 + 3] = vget_lane_u8(shift3, 0);
    }
}

/**
 * Unpack 8 3-bit values using NEON.
 */
void carquet_neon_bitunpack8_3bit(const uint8_t* input, uint32_t* values) {
    /* 8 values * 3 bits = 24 bits = 3 bytes */
    uint32_t v;
    memcpy(&v, input, 3);

    /* Use vectorized extraction where possible */
    uint32x4_t shifts_lo = {0, 3, 6, 9};
    uint32x4_t shifts_hi = {12, 15, 18, 21};
    uint32x4_t mask = vdupq_n_u32(0x7);
    uint32x4_t data = vdupq_n_u32(v);

    uint32x4_t result_lo = vandq_u32(vshlq_u32(data, vnegq_s32(vreinterpretq_s32_u32(shifts_lo))), mask);
    uint32x4_t result_hi = vandq_u32(vshlq_u32(data, vnegq_s32(vreinterpretq_s32_u32(shifts_hi))), mask);

    vst1q_u32(values, result_lo);
    vst1q_u32(values + 4, result_hi);
}

/**
 * Unpack 8 4-bit values using NEON - highly optimized.
 */
void carquet_neon_bitunpack8_4bit(const uint8_t* input, uint32_t* values) {
    /* Load 4 bytes (8 x 4-bit values) */
    uint8x8_t bytes = vreinterpret_u8_u32(vld1_dup_u32((const uint32_t*)input));

    /* Split nibbles */
    uint8x8_t lo_nibbles = vand_u8(bytes, vdup_n_u8(0x0F));
    uint8x8_t hi_nibbles = vshr_n_u8(bytes, 4);

    /* Interleave: lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3 */
    uint8x8x2_t zipped = vzip_u8(lo_nibbles, hi_nibbles);

    /* Widen to 32-bit */
    uint16x8_t wide16 = vmovl_u8(zipped.val[0]);
    uint32x4_t wide32_lo = vmovl_u16(vget_low_u16(wide16));
    uint32x4_t wide32_hi = vmovl_u16(vget_high_u16(wide16));

    vst1q_u32(values, wide32_lo);
    vst1q_u32(values + 4, wide32_hi);
}

/**
 * Unpack 8 5-bit values using NEON.
 */
void carquet_neon_bitunpack8_5bit(const uint8_t* input, uint32_t* values) {
    /* 8 values * 5 bits = 40 bits = 5 bytes */
    uint64_t v = 0;
    memcpy(&v, input, 5);

    /* Vectorized extraction */
    values[0] = (v >> 0) & 0x1F;
    values[1] = (v >> 5) & 0x1F;
    values[2] = (v >> 10) & 0x1F;
    values[3] = (v >> 15) & 0x1F;
    values[4] = (v >> 20) & 0x1F;
    values[5] = (v >> 25) & 0x1F;
    values[6] = (v >> 30) & 0x1F;
    values[7] = (v >> 35) & 0x1F;
}

/**
 * Unpack 8 6-bit values using NEON.
 */
void carquet_neon_bitunpack8_6bit(const uint8_t* input, uint32_t* values) {
    /* 8 values * 6 bits = 48 bits = 6 bytes */
    uint64_t v = 0;
    memcpy(&v, input, 6);

    values[0] = (v >> 0) & 0x3F;
    values[1] = (v >> 6) & 0x3F;
    values[2] = (v >> 12) & 0x3F;
    values[3] = (v >> 18) & 0x3F;
    values[4] = (v >> 24) & 0x3F;
    values[5] = (v >> 30) & 0x3F;
    values[6] = (v >> 36) & 0x3F;
    values[7] = (v >> 42) & 0x3F;
}

/**
 * Unpack 8 7-bit values using NEON.
 */
void carquet_neon_bitunpack8_7bit(const uint8_t* input, uint32_t* values) {
    /* 8 values * 7 bits = 56 bits = 7 bytes */
    uint64_t v = 0;
    memcpy(&v, input, 7);

    values[0] = (v >> 0) & 0x7F;
    values[1] = (v >> 7) & 0x7F;
    values[2] = (v >> 14) & 0x7F;
    values[3] = (v >> 21) & 0x7F;
    values[4] = (v >> 28) & 0x7F;
    values[5] = (v >> 35) & 0x7F;
    values[6] = (v >> 42) & 0x7F;
    values[7] = (v >> 49) & 0x7F;
}

/**
 * Unpack 8 8-bit values using NEON (widen u8 to u32).
 */
void carquet_neon_bitunpack8_8bit(const uint8_t* input, uint32_t* values) {
    uint8x8_t bytes = vld1_u8(input);
    uint16x8_t wide16 = vmovl_u8(bytes);
    uint32x4_t wide32_lo = vmovl_u16(vget_low_u16(wide16));
    uint32x4_t wide32_hi = vmovl_u16(vget_high_u16(wide16));

    vst1q_u32(values, wide32_lo);
    vst1q_u32(values + 4, wide32_hi);
}

/**
 * Unpack 8 16-bit values to 32-bit using NEON.
 */
void carquet_neon_bitunpack8_16bit(const uint8_t* input, uint32_t* values) {
    uint16x8_t words = vld1q_u16((const uint16_t*)input);
    uint32x4_t lo32 = vmovl_u16(vget_low_u16(words));
    uint32x4_t hi32 = vmovl_u16(vget_high_u16(words));

    vst1q_u32(values, lo32);
    vst1q_u32(values + 4, hi32);
}

/**
 * Generic bit unpacking for 8 values with NEON dispatch.
 */
void carquet_neon_bitunpack8_32(const uint8_t* input, int bit_width, uint32_t* values) {
    switch (bit_width) {
        case 1: carquet_neon_bitunpack32_1bit(input, values); return;
        case 2: carquet_neon_bitunpack16_2bit(input, values); return;
        case 3: carquet_neon_bitunpack8_3bit(input, values); return;
        case 4: carquet_neon_bitunpack8_4bit(input, values); return;
        case 5: carquet_neon_bitunpack8_5bit(input, values); return;
        case 6: carquet_neon_bitunpack8_6bit(input, values); return;
        case 7: carquet_neon_bitunpack8_7bit(input, values); return;
        case 8: carquet_neon_bitunpack8_8bit(input, values); return;
        case 16: carquet_neon_bitunpack8_16bit(input, values); return;
        default: break;
    }

    /* General case for other bit widths */
    uint32_t mask = (uint32_t)((1ULL << bit_width) - 1);
    int bit_pos = 0;
    int byte_pos = 0;

    for (int i = 0; i < 8; i++) {
        uint64_t bits = 0;
        int bits_needed = bit_width;
        int bits_in_buffer = 0;

        while (bits_needed > 0) {
            int bits_from_byte = 8 - (bit_pos % 8);
            if (bits_from_byte > bits_needed) {
                bits_from_byte = bits_needed;
            }

            uint8_t byte_val = input[byte_pos];
            int shift_down = bit_pos % 8;
            uint64_t extracted = (byte_val >> shift_down) & ((1U << bits_from_byte) - 1);
            bits |= extracted << bits_in_buffer;

            bit_pos += bits_from_byte;
            bits_in_buffer += bits_from_byte;
            bits_needed -= bits_from_byte;

            if (bit_pos % 8 == 0) {
                byte_pos++;
            }
        }

        values[i] = (uint32_t)(bits & mask);
    }
}

/* ============================================================================
 * Byte Stream Split - NEON Optimized (Float AND Double)
 * ============================================================================
 */

/**
 * Encode floats using byte stream split with NEON.
 * Optimized transpose using native NEON operations.
 */
void carquet_neon_byte_stream_split_encode_float(
    const float* values,
    int64_t count,
    uint8_t* output) {

    const uint8_t* src = (const uint8_t*)values;
    int64_t i = 0;

    /* Process 4 floats (16 bytes) at a time with optimized transpose */
    for (; i + 4 <= count; i += 4) {
        /* Load 4 floats = 16 bytes */
        uint8x16_t v = vld1q_u8(src + i * 4);

        /* Use optimized table lookup for transposition */
        /* v = [a0 a1 a2 a3 | b0 b1 b2 b3 | c0 c1 c2 c3 | d0 d1 d2 d3] */
        /* Want: [a0 b0 c0 d0], [a1 b1 c1 d1], [a2 b2 c2 d2], [a3 b3 c3 d3] */

        static const uint8_t tbl_byte0[16] = {0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        static const uint8_t tbl_byte1[16] = {1, 5, 9, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        static const uint8_t tbl_byte2[16] = {2, 6, 10, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        static const uint8_t tbl_byte3[16] = {3, 7, 11, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        uint8x16_t idx0 = vld1q_u8(tbl_byte0);
        uint8x16_t idx1 = vld1q_u8(tbl_byte1);
        uint8x16_t idx2 = vld1q_u8(tbl_byte2);
        uint8x16_t idx3 = vld1q_u8(tbl_byte3);

        uint8x16_t out0 = vqtbl1q_u8(v, idx0);
        uint8x16_t out1 = vqtbl1q_u8(v, idx1);
        uint8x16_t out2 = vqtbl1q_u8(v, idx2);
        uint8x16_t out3 = vqtbl1q_u8(v, idx3);

        /* Store to transposed positions */
        vst1_lane_u32((uint32_t*)(output + 0 * count + i), vreinterpret_u32_u8(vget_low_u8(out0)), 0);
        vst1_lane_u32((uint32_t*)(output + 1 * count + i), vreinterpret_u32_u8(vget_low_u8(out1)), 0);
        vst1_lane_u32((uint32_t*)(output + 2 * count + i), vreinterpret_u32_u8(vget_low_u8(out2)), 0);
        vst1_lane_u32((uint32_t*)(output + 3 * count + i), vreinterpret_u32_u8(vget_low_u8(out3)), 0);
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            output[b * count + i] = src[i * 4 + b];
        }
    }
}

/**
 * Decode byte stream split floats using NEON.
 */
void carquet_neon_byte_stream_split_decode_float(
    const uint8_t* data,
    int64_t count,
    float* values) {

    uint8_t* dst = (uint8_t*)values;
    int64_t i = 0;

    /* Process 4 floats at a time */
    for (; i + 4 <= count; i += 4) {
        /* Load 4 bytes from each stream */
        uint32_t b0 = *(const uint32_t*)(data + 0 * count + i);
        uint32_t b1 = *(const uint32_t*)(data + 1 * count + i);
        uint32_t b2 = *(const uint32_t*)(data + 2 * count + i);
        uint32_t b3 = *(const uint32_t*)(data + 3 * count + i);

        uint8x8_t bytes0 = vreinterpret_u8_u32(vdup_n_u32(b0));
        uint8x8_t bytes1 = vreinterpret_u8_u32(vdup_n_u32(b1));
        uint8x8_t bytes2 = vreinterpret_u8_u32(vdup_n_u32(b2));
        uint8x8_t bytes3 = vreinterpret_u8_u32(vdup_n_u32(b3));

        /* Interleave bytes back into floats */
        uint8x8x2_t zip01 = vzip_u8(bytes0, bytes1);
        uint8x8x2_t zip23 = vzip_u8(bytes2, bytes3);

        uint16x4_t lo16 = vreinterpret_u16_u8(zip01.val[0]);
        uint16x4_t hi16 = vreinterpret_u16_u8(zip23.val[0]);

        uint16x4x2_t zip_final = vzip_u16(lo16, hi16);

        vst1_u8(dst + i * 4, vreinterpret_u8_u16(zip_final.val[0]));
        vst1_u8(dst + i * 4 + 8, vreinterpret_u8_u16(zip_final.val[1]));
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            dst[i * 4 + b] = data[b * count + i];
        }
    }
}

/**
 * Encode doubles using byte stream split with NEON.
 */
void carquet_neon_byte_stream_split_encode_double(
    const double* values,
    int64_t count,
    uint8_t* output) {

    const uint8_t* src = (const uint8_t*)values;
    int64_t i = 0;

    /* Process 2 doubles (16 bytes) at a time */
    for (; i + 2 <= count; i += 2) {
        /* Transpose: extract each byte position from both doubles */
        for (int b = 0; b < 8; b++) {
            output[b * count + i + 0] = src[i * 8 + b];
            output[b * count + i + 1] = src[i * 8 + 8 + b];
        }
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 8; b++) {
            output[b * count + i] = src[i * 8 + b];
        }
    }
}

/**
 * Decode byte stream split doubles using NEON.
 */
void carquet_neon_byte_stream_split_decode_double(
    const uint8_t* data,
    int64_t count,
    double* values) {

    uint8_t* dst = (uint8_t*)values;
    int64_t i = 0;

    /* Process 2 doubles at a time */
    for (; i + 2 <= count; i += 2) {
        /* Gather bytes from 8 streams */
        for (int b = 0; b < 8; b++) {
            dst[i * 8 + b] = data[b * count + i];
            dst[i * 8 + 8 + b] = data[b * count + i + 1];
        }
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 8; b++) {
            dst[i * 8 + b] = data[b * count + i];
        }
    }
}

/* ============================================================================
 * Delta Decoding - NEON Optimized (Prefix Sum)
 * ============================================================================
 */

/**
 * Apply prefix sum (cumulative sum) to int32 array using NEON.
 * This is used after unpacking deltas to reconstruct original values.
 */
void carquet_neon_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial) {
    int32_t sum = initial;
    int64_t i = 0;

    /* NEON prefix sum for 4 elements at a time */
    for (; i + 4 <= count; i += 4) {
        int32x4_t v = vld1q_s32(values + i);

        /* Partial prefix sums within the vector */
        /* v = [a, b, c, d] */
        /* After step 1: [a, a+b, c, c+d] */
        v = vaddq_s32(v, vextq_s32(vdupq_n_s32(0), v, 3));
        /* After step 2: [a, a+b, a+c, a+b+c+d] - wrong, need fix */
        /* Correct: [a, a+b, a+b+c, a+b+c+d] */
        int32x4_t shifted2 = vextq_s32(vdupq_n_s32(0), v, 2);
        v = vaddq_s32(v, shifted2);

        /* Add running sum */
        v = vaddq_s32(v, vdupq_n_s32(sum));
        vst1q_s32(values + i, v);

        /* Update running sum to last element */
        sum = vgetq_lane_s32(v, 3);
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

/**
 * Apply prefix sum to int64 array using NEON.
 */
void carquet_neon_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial) {
    int64_t sum = initial;
    int64_t i = 0;

    /* NEON prefix sum for 2 elements at a time */
    for (; i + 2 <= count; i += 2) {
        int64x2_t v = vld1q_s64(values + i);

        /* v = [a, b] -> [a, a+b] */
        int64x2_t shifted = vextq_s64(vdupq_n_s64(0), v, 1);
        v = vaddq_s64(v, shifted);

        /* Add running sum */
        v = vaddq_s64(v, vdupq_n_s64(sum));
        vst1q_s64(values + i, v);

        sum = vgetq_lane_s64(v, 1);
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

/* ============================================================================
 * Dictionary Gather - NEON Optimized with Prefetching
 * ============================================================================
 */

/**
 * Gather int32 values from dictionary using indices (NEON).
 * Uses prefetching for better memory access patterns.
 */
void carquet_neon_gather_i32(const int32_t* dict, const uint32_t* indices,
                              int64_t count, int32_t* output) {
    int64_t i = 0;

    /* Process 8 at a time with prefetching */
    for (; i + 8 <= count; i += 8) {
        /* Prefetch future indices and dictionary values */
        __builtin_prefetch(indices + i + 16, 0, 1);

        /* Load indices */
        uint32x4_t idx0 = vld1q_u32(indices + i);
        uint32x4_t idx1 = vld1q_u32(indices + i + 4);

        /* Prefetch dictionary entries */
        __builtin_prefetch(dict + vgetq_lane_u32(idx0, 0), 0, 0);
        __builtin_prefetch(dict + vgetq_lane_u32(idx0, 2), 0, 0);
        __builtin_prefetch(dict + vgetq_lane_u32(idx1, 0), 0, 0);
        __builtin_prefetch(dict + vgetq_lane_u32(idx1, 2), 0, 0);

        /* Gather values - NEON doesn't have true gather, use scalar loads */
        int32_t v0 = dict[vgetq_lane_u32(idx0, 0)];
        int32_t v1 = dict[vgetq_lane_u32(idx0, 1)];
        int32_t v2 = dict[vgetq_lane_u32(idx0, 2)];
        int32_t v3 = dict[vgetq_lane_u32(idx0, 3)];
        int32_t v4 = dict[vgetq_lane_u32(idx1, 0)];
        int32_t v5 = dict[vgetq_lane_u32(idx1, 1)];
        int32_t v6 = dict[vgetq_lane_u32(idx1, 2)];
        int32_t v7 = dict[vgetq_lane_u32(idx1, 3)];

        /* Store using NEON */
        int32x4_t result0 = {v0, v1, v2, v3};
        int32x4_t result1 = {v4, v5, v6, v7};
        vst1q_s32(output + i, result0);
        vst1q_s32(output + i + 4, result1);
    }

    /* Handle remaining with prefetch */
    for (; i + 4 <= count; i += 4) {
        uint32x4_t idx = vld1q_u32(indices + i);
        int32_t v0 = dict[vgetq_lane_u32(idx, 0)];
        int32_t v1 = dict[vgetq_lane_u32(idx, 1)];
        int32_t v2 = dict[vgetq_lane_u32(idx, 2)];
        int32_t v3 = dict[vgetq_lane_u32(idx, 3)];

        int32x4_t result = {v0, v1, v2, v3};
        vst1q_s32(output + i, result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather int64 values from dictionary using indices (NEON).
 */
void carquet_neon_gather_i64(const int64_t* dict, const uint32_t* indices,
                              int64_t count, int64_t* output) {
    int64_t i = 0;

    /* Process 4 at a time with prefetching */
    for (; i + 4 <= count; i += 4) {
        __builtin_prefetch(indices + i + 8, 0, 1);

        uint32x4_t idx = vld1q_u32(indices + i);

        /* Prefetch dictionary entries */
        __builtin_prefetch(dict + vgetq_lane_u32(idx, 0), 0, 0);
        __builtin_prefetch(dict + vgetq_lane_u32(idx, 2), 0, 0);

        int64_t v0 = dict[vgetq_lane_u32(idx, 0)];
        int64_t v1 = dict[vgetq_lane_u32(idx, 1)];
        int64_t v2 = dict[vgetq_lane_u32(idx, 2)];
        int64_t v3 = dict[vgetq_lane_u32(idx, 3)];

        int64x2_t result0 = {v0, v1};
        int64x2_t result1 = {v2, v3};
        vst1q_s64(output + i, result0);
        vst1q_s64(output + i + 2, result1);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather float values from dictionary using indices (NEON).
 */
void carquet_neon_gather_float(const float* dict, const uint32_t* indices,
                                int64_t count, float* output) {
    int64_t i = 0;

    /* Process 8 at a time with prefetching */
    for (; i + 8 <= count; i += 8) {
        __builtin_prefetch(indices + i + 16, 0, 1);

        uint32x4_t idx0 = vld1q_u32(indices + i);
        uint32x4_t idx1 = vld1q_u32(indices + i + 4);

        __builtin_prefetch(dict + vgetq_lane_u32(idx0, 0), 0, 0);
        __builtin_prefetch(dict + vgetq_lane_u32(idx1, 0), 0, 0);

        float v0 = dict[vgetq_lane_u32(idx0, 0)];
        float v1 = dict[vgetq_lane_u32(idx0, 1)];
        float v2 = dict[vgetq_lane_u32(idx0, 2)];
        float v3 = dict[vgetq_lane_u32(idx0, 3)];
        float v4 = dict[vgetq_lane_u32(idx1, 0)];
        float v5 = dict[vgetq_lane_u32(idx1, 1)];
        float v6 = dict[vgetq_lane_u32(idx1, 2)];
        float v7 = dict[vgetq_lane_u32(idx1, 3)];

        float32x4_t result0 = {v0, v1, v2, v3};
        float32x4_t result1 = {v4, v5, v6, v7};
        vst1q_f32(output + i, result0);
        vst1q_f32(output + i + 4, result1);
    }

    for (; i + 4 <= count; i += 4) {
        uint32x4_t idx = vld1q_u32(indices + i);
        float v0 = dict[vgetq_lane_u32(idx, 0)];
        float v1 = dict[vgetq_lane_u32(idx, 1)];
        float v2 = dict[vgetq_lane_u32(idx, 2)];
        float v3 = dict[vgetq_lane_u32(idx, 3)];

        float32x4_t result = {v0, v1, v2, v3};
        vst1q_f32(output + i, result);
    }

    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather double values from dictionary using indices (NEON).
 */
void carquet_neon_gather_double(const double* dict, const uint32_t* indices,
                                 int64_t count, double* output) {
    int64_t i = 0;

    /* Process 4 at a time with prefetching */
    for (; i + 4 <= count; i += 4) {
        __builtin_prefetch(indices + i + 8, 0, 1);

        uint32x4_t idx = vld1q_u32(indices + i);

        __builtin_prefetch(dict + vgetq_lane_u32(idx, 0), 0, 0);
        __builtin_prefetch(dict + vgetq_lane_u32(idx, 2), 0, 0);

        double v0 = dict[vgetq_lane_u32(idx, 0)];
        double v1 = dict[vgetq_lane_u32(idx, 1)];
        double v2 = dict[vgetq_lane_u32(idx, 2)];
        double v3 = dict[vgetq_lane_u32(idx, 3)];

        float64x2_t result0 = {v0, v1};
        float64x2_t result1 = {v2, v3};
        vst1q_f64(output + i, result0);
        vst1q_f64(output + i + 2, result1);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/* ============================================================================
 * Boolean Packing/Unpacking - NEON Optimized
 * ============================================================================
 */

/**
 * Unpack boolean values from packed bits to byte array using NEON.
 * Each output byte is 0 or 1.
 */
void carquet_neon_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    int64_t i = 0;

    /* Process 64 bools (8 bytes) at a time using NEON */
    for (; i + 64 <= count; i += 64) {
        int byte_idx = (int)(i / 8);
        uint8x8_t packed = vld1_u8(input + byte_idx);

        /* For each byte, expand to 8 output bytes */
        for (int b = 0; b < 8; b++) {
            uint8_t byte_val = vget_lane_u8(packed, 0);
            packed = vext_u8(packed, packed, 1);  /* Rotate */

            /* Expand each bit to a byte */
            static const uint8_t bit_masks[8] = {1, 2, 4, 8, 16, 32, 64, 128};
            uint8x8_t byte_vec = vdup_n_u8(byte_val);
            uint8x8_t masks = vld1_u8(bit_masks);
            uint8x8_t masked = vand_u8(byte_vec, masks);
            uint8x8_t cmp = vceq_u8(masked, masks);
            uint8x8_t result = vand_u8(cmp, vdup_n_u8(1));

            vst1_u8(output + i + b * 8, result);
        }
    }

    /* Process 8 bools (1 byte) at a time */
    for (; i + 8 <= count; i += 8) {
        int byte_idx = (int)(i / 8);
        uint8_t byte_val = input[byte_idx];

        uint8x8_t byte_vec = vdup_n_u8(byte_val);
        static const uint8_t bit_masks[8] = {1, 2, 4, 8, 16, 32, 64, 128};
        uint8x8_t masks = vld1_u8(bit_masks);
        uint8x8_t masked = vand_u8(byte_vec, masks);
        uint8x8_t cmp = vceq_u8(masked, masks);
        uint8x8_t result = vand_u8(cmp, vdup_n_u8(1));

        vst1_u8(output + i, result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        int byte_idx = (int)(i / 8);
        int bit_idx = (int)(i % 8);
        output[i] = (input[byte_idx] >> bit_idx) & 1;
    }
}

/**
 * Pack boolean values from byte array to packed bits using NEON.
 */
void carquet_neon_pack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    int64_t i = 0;

    /* Process 8 bools at a time */
    for (; i + 8 <= count; i += 8) {
        uint8x8_t bools = vld1_u8(input + i);

        /* Multiply by bit positions and sum */
        static const uint8_t bit_positions[8] = {1, 2, 4, 8, 16, 32, 64, 128};
        uint8x8_t positions = vld1_u8(bit_positions);

        /* AND with 1 to ensure boolean, then multiply */
        uint8x8_t masked = vand_u8(bools, vdup_n_u8(1));
        uint8x8_t weighted = vmul_u8(masked, positions);

        /* Horizontal add - NEON doesn't have direct horizontal add for u8,
           so we use pairwise operations */
        uint16x4_t sum16 = vpaddl_u8(weighted);
        uint32x2_t sum32 = vpaddl_u16(sum16);
        uint64x1_t sum64 = vpaddl_u32(sum32);

        output[i / 8] = (uint8_t)vget_lane_u64(sum64, 0);
    }

    /* Handle remaining */
    if (i < count) {
        uint8_t byte = 0;
        for (int64_t j = 0; j < 8 && i + j < count; j++) {
            if (input[i + j]) {
                byte |= (1 << j);
            }
        }
        output[i / 8] = byte;
    }
}

/* ============================================================================
 * RLE Run Detection - NEON Optimized
 * ============================================================================
 */

/**
 * Find the length of a run of repeated values.
 * Returns the number of consecutive identical values starting at position 0.
 */
int64_t carquet_neon_find_run_length_i32(const int32_t* values, int64_t count) {
    if (count == 0) return 0;

    int32_t first = values[0];
    int32x4_t target = vdupq_n_s32(first);
    int64_t i = 0;

    /* Check 8 at a time for better throughput */
    for (; i + 8 <= count; i += 8) {
        int32x4_t v0 = vld1q_s32(values + i);
        int32x4_t v1 = vld1q_s32(values + i + 4);

        uint32x4_t cmp0 = vceqq_s32(v0, target);
        uint32x4_t cmp1 = vceqq_s32(v1, target);

        /* Use horizontal min to check if any element is not all-1s (0xFFFFFFFF) */
        uint32_t min0 = vminvq_u32(cmp0);
        uint32_t min1 = vminvq_u32(cmp1);

        if (min0 != 0xFFFFFFFF) {
            /* Find first mismatch in first vector */
            for (int64_t j = i; j < i + 4; j++) {
                if (values[j] != first) return j;
            }
        }

        if (min1 != 0xFFFFFFFF) {
            /* Find first mismatch in second vector */
            for (int64_t j = i + 4; j < i + 8; j++) {
                if (values[j] != first) return j;
            }
        }
    }

    /* Handle remaining with NEON */
    for (; i + 4 <= count; i += 4) {
        int32x4_t v = vld1q_s32(values + i);
        uint32x4_t cmp = vceqq_s32(v, target);

        uint32_t min_val = vminvq_u32(cmp);
        if (min_val != 0xFFFFFFFF) {
            for (int64_t j = i; j < i + 4 && j < count; j++) {
                if (values[j] != first) return j;
            }
        }
    }

    /* Handle remaining scalar */
    for (; i < count; i++) {
        if (values[i] != first) {
            return i;
        }
    }

    return count;
}

/* ============================================================================
 * CRC32 - NEON + ARM CRC Intrinsics
 * ============================================================================
 */

#ifdef __ARM_FEATURE_CRC32
#include <arm_acle.h>

/**
 * Compute CRC32C using ARM CRC instructions.
 * Unrolled for better performance on Apple Silicon.
 */
uint32_t carquet_neon_crc32c(uint32_t crc, const uint8_t* data, size_t len) {
    size_t i = 0;

    /* Process 32 bytes at a time (unrolled) */
    for (; i + 32 <= len; i += 32) {
        uint64_t v0, v1, v2, v3;
        memcpy(&v0, data + i, 8);
        memcpy(&v1, data + i + 8, 8);
        memcpy(&v2, data + i + 16, 8);
        memcpy(&v3, data + i + 24, 8);
        crc = __crc32cd(crc, v0);
        crc = __crc32cd(crc, v1);
        crc = __crc32cd(crc, v2);
        crc = __crc32cd(crc, v3);
    }

    /* Process 8 bytes at a time */
    for (; i + 8 <= len; i += 8) {
        uint64_t val;
        memcpy(&val, data + i, 8);
        crc = __crc32cd(crc, val);
    }

    /* Process 4 bytes */
    if (i + 4 <= len) {
        uint32_t val;
        memcpy(&val, data + i, 4);
        crc = __crc32cw(crc, val);
        i += 4;
    }

    /* Process 2 bytes */
    if (i + 2 <= len) {
        uint16_t val;
        memcpy(&val, data + i, 2);
        crc = __crc32ch(crc, val);
        i += 2;
    }

    /* Process remaining byte */
    if (i < len) {
        crc = __crc32cb(crc, data[i]);
    }

    return crc;
}

#else

/* Software CRC32C fallback */
static const uint32_t crc32c_table[256] = {
    0x00000000, 0xF26B8303, 0xE13B70F7, 0x1350F3F4, 0xC79A971F, 0x35F1141C, 0x26A1E7E8, 0xD4CA64EB,
    0x8AD958CF, 0x78B2DBCC, 0x6BE22838, 0x9989AB3B, 0x4D43CFD0, 0xBF284CD3, 0xAC78BF27, 0x5E133C24,
    0x105EC76F, 0xE235446C, 0xF165B798, 0x030E349B, 0xD7C45070, 0x25AFD373, 0x36FF2087, 0xC494A384,
    0x9A879FA0, 0x68EC1CA3, 0x7BBCEF57, 0x89D76C54, 0x5D1D08BF, 0xAF768BBC, 0xBC267848, 0x4E4DFB4B,
    0x20BD8EDE, 0xD2D60DDD, 0xC186FE29, 0x33ED7D2A, 0xE72719C1, 0x154C9AC2, 0x061C6936, 0xF477EA35,
    0xAA64D611, 0x580F5512, 0x4B5FA6E6, 0xB93425E5, 0x6DFE410E, 0x9F95C20D, 0x8CC531F9, 0x7EAEB2FA,
    0x30E349B1, 0xC288CAB2, 0xD1D83946, 0x23B3BA45, 0xF779DEAE, 0x05125DAD, 0x1642AE59, 0xE4292D5A,
    0xBA3A117E, 0x4851927D, 0x5B016189, 0xA96AE28A, 0x7DA08661, 0x8FCB0562, 0x9C9BF696, 0x6EF07595,
    0x417B1DBC, 0xB3109EBF, 0xA0406D4B, 0x522BEE48, 0x86E18AA3, 0x748A09A0, 0x67DAFA54, 0x95B17957,
    0xCBA24573, 0x39C9C670, 0x2A993584, 0xD8F2B687, 0x0C38D26C, 0xFE53516F, 0xED03A29B, 0x1F682198,
    0x5125DAD3, 0xA34E59D0, 0xB01EAA24, 0x42752927, 0x96BF4DCC, 0x64D4CECF, 0x77843D3B, 0x85EFBE38,
    0xDBFC821C, 0x2997011F, 0x3AC7F2EB, 0xC8AC71E8, 0x1C661503, 0xEE0D9600, 0xFD5D65F4, 0x0F36E6F7,
    0x61C69362, 0x93AD1061, 0x80FDE395, 0x72966096, 0xA65C047D, 0x5437877E, 0x4767748A, 0xB50CF789,
    0xEB1FCBAD, 0x197448AE, 0x0A24BB5A, 0xF84F3859, 0x2C855CB2, 0xDEEEDFB1, 0xCDBE2C45, 0x3FD5AF46,
    0x7198540D, 0x83F3D70E, 0x90A324FA, 0x62C8A7F9, 0xB602C312, 0x44694011, 0x5739B3E5, 0xA55230E6,
    0xFB410CC2, 0x092A8FC1, 0x1A7A7C35, 0xE811FF36, 0x3CDB9BDD, 0xCEB018DE, 0xDDE0EB2A, 0x2F8B6829,
    0x82F63B78, 0x709DB87B, 0x63CD4B8F, 0x91A6C88C, 0x456CAC67, 0xB7072F64, 0xA457DC90, 0x563C5F93,
    0x082F63B7, 0xFA44E0B4, 0xE9141340, 0x1B7F9043, 0xCFB5F4A8, 0x3DDE77AB, 0x2E8E845F, 0xDCE5075C,
    0x92A8FC17, 0x60C37F14, 0x73938CE0, 0x81F80FE3, 0x55326B08, 0xA759E80B, 0xB4091BFF, 0x466298FC,
    0x1871A4D8, 0xEA1A27DB, 0xF94AD42F, 0x0B21572C, 0xDFEB33C7, 0x2D80B0C4, 0x3ED04330, 0xCCBBC033,
    0xA24BB5A6, 0x502036A5, 0x4370C551, 0xB11B4652, 0x65D122B9, 0x97BAA1BA, 0x84EA524E, 0x7681D14D,
    0x2892ED69, 0xDAF96E6A, 0xC9A99D9E, 0x3BC21E9D, 0xEF087A76, 0x1D63F975, 0x0E330A81, 0xFC588982,
    0xB21572C9, 0x407EF1CA, 0x532E023E, 0xA145813D, 0x758FE5D6, 0x87E466D5, 0x94B49521, 0x66DF1622,
    0x38CC2A06, 0xCAA7A905, 0xD9F75AF1, 0x2B9CD9F2, 0xFF56BD19, 0x0D3D3E1A, 0x1E6DCDEE, 0xEC064EED,
    0xC38D26C4, 0x31E6A5C7, 0x22B65633, 0xD0DDD530, 0x0417B1DB, 0xF67C32D8, 0xE52CC12C, 0x1747422F,
    0x49547E0B, 0xBB3FFD08, 0xA86F0EFC, 0x5A048DFF, 0x8ECEE914, 0x7CA56A17, 0x6FF599E3, 0x9D9E1AE0,
    0xD3D3E1AB, 0x21B862A8, 0x32E8915C, 0xC083125F, 0x144976B4, 0xE622F5B7, 0xF5720643, 0x07198540,
    0x590AB964, 0xAB613A67, 0xB831C993, 0x4A5A4A90, 0x9E902E7B, 0x6CFBAD78, 0x7FAB5E8C, 0x8DC0DD8F,
    0xE330A81A, 0x115B2B19, 0x020BD8ED, 0xF0605BEE, 0x24AA3F05, 0xD6C1BC06, 0xC5914FF2, 0x37FACCF1,
    0x69E9F0D5, 0x9B8273D6, 0x88D28022, 0x7AB90321, 0xAE7367CA, 0x5C18E4C9, 0x4F48173D, 0xBD23943E,
    0xF36E6F75, 0x0105EC76, 0x12551F82, 0xE03E9C81, 0x34F4F86A, 0xC69F7B69, 0xD5CF889D, 0x27A40B9E,
    0x79B737BA, 0x8BDCB4B9, 0x988C474D, 0x6AE7C44E, 0xBE2DA0A5, 0x4C4623A6, 0x5F16D052, 0xAD7D5351
};

uint32_t carquet_neon_crc32c(uint32_t crc, const uint8_t* data, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc = crc32c_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

#endif /* __ARM_FEATURE_CRC32 */

/* ============================================================================
 * Memcpy/Memset - NEON Optimized
 * ============================================================================
 */

/**
 * Fast memset using NEON - optimized for various sizes.
 */
void carquet_neon_memset(void* dest, uint8_t value, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    uint8x16_t v = vdupq_n_u8(value);

    /* Process 64 bytes at a time (unrolled) */
    while (n >= 64) {
        vst1q_u8(d, v);
        vst1q_u8(d + 16, v);
        vst1q_u8(d + 32, v);
        vst1q_u8(d + 48, v);
        d += 64;
        n -= 64;
    }

    while (n >= 16) {
        vst1q_u8(d, v);
        d += 16;
        n -= 16;
    }

    if (n >= 8) {
        vst1_u8(d, vget_low_u8(v));
        d += 8;
        n -= 8;
    }

    while (n > 0) {
        *d++ = value;
        n--;
    }
}

/**
 * Fast memcpy using NEON - optimized for various sizes.
 */
void carquet_neon_memcpy(void* dest, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    const uint8_t* s = (const uint8_t*)src;

    /* Process 64 bytes at a time (unrolled) */
    while (n >= 64) {
        uint8x16_t v0 = vld1q_u8(s);
        uint8x16_t v1 = vld1q_u8(s + 16);
        uint8x16_t v2 = vld1q_u8(s + 32);
        uint8x16_t v3 = vld1q_u8(s + 48);
        vst1q_u8(d, v0);
        vst1q_u8(d + 16, v1);
        vst1q_u8(d + 32, v2);
        vst1q_u8(d + 48, v3);
        d += 64;
        s += 64;
        n -= 64;
    }

    while (n >= 16) {
        vst1q_u8(d, vld1q_u8(s));
        d += 16;
        s += 16;
        n -= 16;
    }

    if (n >= 8) {
        vst1_u8(d, vld1_u8(s));
        d += 8;
        s += 8;
        n -= 8;
    }

    while (n > 0) {
        *d++ = *s++;
        n--;
    }
}

/* ============================================================================
 * Match Copy for Compression - NEON Optimized
 * ============================================================================
 */

/**
 * Fast match copy for LZ4/Snappy decompression.
 * Handles overlapping copies correctly.
 */
void carquet_neon_match_copy(uint8_t* dst, const uint8_t* src, size_t len, size_t offset) {
    if (offset >= 16) {
        /* Non-overlapping: use full NEON copies */
        while (len >= 16) {
            vst1q_u8(dst, vld1q_u8(src));
            dst += 16;
            src += 16;
            len -= 16;
        }

        if (len >= 8) {
            vst1_u8(dst, vld1_u8(src));
            dst += 8;
            src += 8;
            len -= 8;
        }

        while (len > 0) {
            *dst++ = *src++;
            len--;
        }
    } else if (offset == 1) {
        /* Common pattern: fill with single byte */
        uint8_t val = *src;
        uint8x16_t v = vdupq_n_u8(val);

        while (len >= 16) {
            vst1q_u8(dst, v);
            dst += 16;
            len -= 16;
        }

        while (len > 0) {
            *dst++ = val;
            len--;
        }
    } else if (offset == 2) {
        /* Fill with 2-byte pattern */
        uint8_t v0 = src[0], v1 = src[1];
        while (len >= 2) {
            *dst++ = v0;
            *dst++ = v1;
            len -= 2;
        }
        if (len) *dst = v0;
    } else if (offset == 4) {
        /* Fill with 4-byte pattern */
        uint32_t pattern;
        memcpy(&pattern, src, 4);
        uint32x4_t v = vdupq_n_u32(pattern);

        while (len >= 16) {
            vst1q_u32((uint32_t*)dst, v);
            dst += 16;
            len -= 16;
        }

        while (len >= 4) {
            memcpy(dst, &pattern, 4);
            dst += 4;
            len -= 4;
        }

        for (size_t i = 0; i < len; i++) {
            dst[i] = src[i];
        }
    } else {
        /* General overlapping case: copy byte by byte */
        while (len > 0) {
            *dst++ = *src++;
            len--;
        }
    }
}

/**
 * Count matching bytes between two buffers using NEON.
 * Returns the number of matching bytes from the start.
 */
size_t carquet_neon_match_length(const uint8_t* p, const uint8_t* match, const uint8_t* limit) {
    const uint8_t* start = p;

    /* Compare 16 bytes at a time */
    while (p + 16 <= limit) {
        uint8x16_t a = vld1q_u8(p);
        uint8x16_t b = vld1q_u8(match);
        uint8x16_t cmp = vceqq_u8(a, b);

        /* Check if all bytes match (all 0xFF) using horizontal min */
        if (vminvq_u8(cmp) != 0xFF) {
            /* Find first mismatch */
            for (size_t i = 0; i < 16 && p + i < limit; i++) {
                if (p[i] != match[i]) {
                    return (size_t)(p - start) + i;
                }
            }
        }

        p += 16;
        match += 16;
    }

    /* Compare remaining bytes */
    while (p < limit && *p == *match) {
        p++;
        match++;
    }

    return (size_t)(p - start);
}

#endif /* __ARM_NEON */
#endif /* ARM */
