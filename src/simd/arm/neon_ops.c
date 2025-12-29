/**
 * @file neon_ops.c
 * @brief NEON optimized operations for ARM processors
 *
 * Provides SIMD-accelerated implementations of:
 * - Bit unpacking for common bit widths
 * - Byte stream split/merge (for BYTE_STREAM_SPLIT encoding)
 * - Delta decoding (prefix sums)
 * - Dictionary gather operations
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__aarch64__) || defined(__arm__)
#ifdef __ARM_NEON

#include <arm_neon.h>

/* ============================================================================
 * Bit Unpacking - NEON Optimized
 * ============================================================================
 */

/**
 * Unpack 32 1-bit values using NEON.
 */
void carquet_neon_bitunpack32_1bit(const uint8_t* input, uint32_t* values) {
    /* Expand each byte to 8 single-bit values */
    for (int b = 0; b < 4; b++) {
        uint8_t byte = input[b];
        for (int i = 0; i < 8; i++) {
            values[b * 8 + i] = (byte >> i) & 1;
        }
    }
}

/**
 * Unpack 16 2-bit values using NEON.
 */
void carquet_neon_bitunpack16_2bit(const uint8_t* input, uint32_t* values) {
    /* Expand each byte to 4 values */
    for (int b = 0; b < 4; b++) {
        uint8_t byte = input[b];
        values[b * 4 + 0] = (byte >> 0) & 0x3;
        values[b * 4 + 1] = (byte >> 2) & 0x3;
        values[b * 4 + 2] = (byte >> 4) & 0x3;
        values[b * 4 + 3] = (byte >> 6) & 0x3;
    }
}

/**
 * Unpack 8 4-bit values using NEON.
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

/* ============================================================================
 * Byte Stream Split - NEON Optimized
 * ============================================================================
 */

/**
 * Encode floats using byte stream split with NEON.
 * Transposes: puts all byte 0s together, then all byte 1s, etc.
 */
void carquet_neon_byte_stream_split_encode_float(
    const float* values,
    int64_t count,
    uint8_t* output) {

    const uint8_t* src = (const uint8_t*)values;
    int64_t i = 0;

    /* Process 4 floats (16 bytes) at a time */
    for (; i + 4 <= count; i += 4) {
        /* Load 4 floats = 16 bytes */
        uint8x16_t v = vld1q_u8(src + i * 4);

        /* Transpose: extract every 4th byte */
        /* v = [a0 a1 a2 a3 | b0 b1 b2 b3 | c0 c1 c2 c3 | d0 d1 d2 d3] */
        /* Want: [a0 b0 c0 d0], [a1 b1 c1 d1], [a2 b2 c2 d2], [a3 b3 c3 d3] */

        /* Use table lookup for transposition */
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
        /* After step 2: [a, a+b, a+c, a+b+c+d] */
        v = vaddq_s32(v, vextq_s32(vdupq_n_s32(0), v, 2));

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
 * Dictionary Gather - NEON Optimized
 * ============================================================================
 */

/**
 * Gather int32 values from dictionary using indices (NEON).
 */
void carquet_neon_gather_i32(const int32_t* dict, const uint32_t* indices,
                              int64_t count, int32_t* output) {
    int64_t i = 0;

    /* Process 4 at a time - unfortunately NEON doesn't have true gather,
     * so we use scalar loads but vectorized stores */
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
 * Gather float values from dictionary using indices (NEON).
 */
void carquet_neon_gather_float(const float* dict, const uint32_t* indices,
                                int64_t count, float* output) {
    int64_t i = 0;

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

/* ============================================================================
 * CRC32 - NEON + ARM CRC Intrinsics
 * ============================================================================
 */

#ifdef __ARM_FEATURE_CRC32
#include <arm_acle.h>

/**
 * Compute CRC32C using ARM CRC instructions.
 */
uint32_t carquet_neon_crc32c(uint32_t crc, const uint8_t* data, size_t len) {
    size_t i = 0;

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

#endif /* __ARM_FEATURE_CRC32 */

/* ============================================================================
 * Memcpy/Memset - NEON Optimized for small sizes
 * ============================================================================
 */

/**
 * Fast memset for small buffers using NEON.
 */
void carquet_neon_memset_small(void* dest, uint8_t value, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    uint8x16_t v = vdupq_n_u8(value);

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
 * Fast memcpy for small buffers using NEON.
 */
void carquet_neon_memcpy_small(void* dest, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    const uint8_t* s = (const uint8_t*)src;

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

#endif /* __ARM_NEON */
#endif /* ARM */
