/**
 * @file sse_ops.c
 * @brief SSE4.2 optimized operations for x86 processors
 *
 * Provides SIMD-accelerated implementations of:
 * - Bit unpacking for common bit widths
 * - Byte stream split/merge (for BYTE_STREAM_SPLIT encoding)
 * - Delta decoding (prefix sums)
 * - Dictionary gather operations
 * - CRC32C computation
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
/* SSE4.2 is always available on x64 MSVC, check __SSE4_2__ for GCC/Clang */
#if defined(__SSE4_2__) || defined(_M_X64) || defined(_M_IX86)

#ifdef _MSC_VER
#include <intrin.h>

/* MSVC doesn't have __builtin_prefetch, use _mm_prefetch instead */
#define __builtin_prefetch(addr, rw, locality) \
    _mm_prefetch((const char*)(addr), _MM_HINT_T0)

/* MSVC doesn't have __builtin_ctz (count trailing zeros) */
static inline int msvc_ctz(unsigned int x) {
    unsigned long index;
    _BitScanForward(&index, x);
    return (int)index;
}
#define __builtin_ctz(x) msvc_ctz(x)

/* MSVC doesn't have __builtin_popcount */
#define __builtin_popcount(x) __popcnt(x)

#endif
#include <smmintrin.h>
#include <nmmintrin.h>

/* ============================================================================
 * Bit Unpacking - SSE Optimized
 * ============================================================================
 */

/**
 * Unpack 32 1-bit values using SSE.
 * Input: 4 bytes, Output: 32 x uint32_t
 */
void carquet_sse_bitunpack32_1bit(const uint8_t* input, uint32_t* values) {
    /* Load 4 bytes and expand */
    __m128i bytes = _mm_cvtsi32_si128(*(const int32_t*)input);

    /* Shuffle to repeat each byte 8 times for masking */
    static const int8_t shuffle_mask[16] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1
    };
    __m128i shuf = _mm_loadu_si128((const __m128i*)shuffle_mask);

    /* Process first 16 bits */
    __m128i expanded = _mm_shuffle_epi8(bytes, shuf);
    __m128i bit_mask = _mm_set_epi8(
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01
    );

    __m128i masked = _mm_and_si128(expanded, bit_mask);
    __m128i result = _mm_min_epu8(masked, _mm_set1_epi8(1));

    /* Unpack to 32-bit */
    __m128i zero = _mm_setzero_si128();
    __m128i lo8 = _mm_unpacklo_epi8(result, zero);
    __m128i hi8 = _mm_unpackhi_epi8(result, zero);

    __m128i v0 = _mm_unpacklo_epi16(lo8, zero);
    __m128i v1 = _mm_unpackhi_epi16(lo8, zero);
    __m128i v2 = _mm_unpacklo_epi16(hi8, zero);
    __m128i v3 = _mm_unpackhi_epi16(hi8, zero);

    _mm_storeu_si128((__m128i*)(values + 0), v0);
    _mm_storeu_si128((__m128i*)(values + 4), v1);
    _mm_storeu_si128((__m128i*)(values + 8), v2);
    _mm_storeu_si128((__m128i*)(values + 12), v3);

    /* Process bytes 2-3 for values 16-31 */
    static const int8_t shuffle_mask2[16] = {
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3
    };
    shuf = _mm_loadu_si128((const __m128i*)shuffle_mask2);
    expanded = _mm_shuffle_epi8(bytes, shuf);
    masked = _mm_and_si128(expanded, bit_mask);
    result = _mm_min_epu8(masked, _mm_set1_epi8(1));

    lo8 = _mm_unpacklo_epi8(result, zero);
    hi8 = _mm_unpackhi_epi8(result, zero);

    v0 = _mm_unpacklo_epi16(lo8, zero);
    v1 = _mm_unpackhi_epi16(lo8, zero);
    v2 = _mm_unpacklo_epi16(hi8, zero);
    v3 = _mm_unpackhi_epi16(hi8, zero);

    _mm_storeu_si128((__m128i*)(values + 16), v0);
    _mm_storeu_si128((__m128i*)(values + 20), v1);
    _mm_storeu_si128((__m128i*)(values + 24), v2);
    _mm_storeu_si128((__m128i*)(values + 28), v3);
}

/**
 * Unpack 8 4-bit values using SSE.
 */
void carquet_sse_bitunpack8_4bit(const uint8_t* input, uint32_t* values) {
    /* Load 4 bytes containing 8 x 4-bit values */
    __m128i bytes = _mm_cvtsi32_si128(*(const int32_t*)input);

    /* Split into low and high nibbles */
    __m128i lo_nibbles = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
    __m128i hi_nibbles = _mm_srli_epi16(bytes, 4);
    hi_nibbles = _mm_and_si128(hi_nibbles, _mm_set1_epi8(0x0F));

    /* Interleave */
    __m128i interleaved = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);

    /* Expand to 32-bit */
    __m128i zero = _mm_setzero_si128();
    __m128i words = _mm_unpacklo_epi8(interleaved, zero);

    __m128i v0 = _mm_unpacklo_epi16(words, zero);
    __m128i v1 = _mm_unpackhi_epi16(words, zero);

    _mm_storeu_si128((__m128i*)(values + 0), v0);
    _mm_storeu_si128((__m128i*)(values + 4), v1);
}

/**
 * Unpack 8 8-bit values using SSE (widen u8 to u32).
 */
void carquet_sse_bitunpack8_8bit(const uint8_t* input, uint32_t* values) {
    /* Load 8 bytes */
    __m128i bytes = _mm_loadl_epi64((const __m128i*)input);

    /* Expand to 32-bit */
    __m128i zero = _mm_setzero_si128();
    __m128i words = _mm_unpacklo_epi8(bytes, zero);

    __m128i v0 = _mm_unpacklo_epi16(words, zero);
    __m128i v1 = _mm_unpackhi_epi16(words, zero);

    _mm_storeu_si128((__m128i*)(values + 0), v0);
    _mm_storeu_si128((__m128i*)(values + 4), v1);
}

/* ============================================================================
 * Byte Stream Split - SSE Optimized
 * ============================================================================
 */

/**
 * Encode floats using byte stream split with SSE.
 * Transposes: puts all byte 0s together, then all byte 1s, etc.
 */
void carquet_sse_byte_stream_split_encode_float(
    const float* values,
    int64_t count,
    uint8_t* output) {

    const uint8_t* src = (const uint8_t*)values;
    int64_t i = 0;

    /* Process 4 floats (16 bytes) at a time */
    for (; i + 4 <= count; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i*)(src + i * 4));

        /* Transpose using shuffles */
        /* v = [a0 a1 a2 a3 | b0 b1 b2 b3 | c0 c1 c2 c3 | d0 d1 d2 d3] */
        static const int8_t shuf_b0[16] = {0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        static const int8_t shuf_b1[16] = {1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        static const int8_t shuf_b2[16] = {2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        static const int8_t shuf_b3[16] = {3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

        __m128i s0 = _mm_loadu_si128((const __m128i*)shuf_b0);
        __m128i s1 = _mm_loadu_si128((const __m128i*)shuf_b1);
        __m128i s2 = _mm_loadu_si128((const __m128i*)shuf_b2);
        __m128i s3 = _mm_loadu_si128((const __m128i*)shuf_b3);

        __m128i out0 = _mm_shuffle_epi8(v, s0);
        __m128i out1 = _mm_shuffle_epi8(v, s1);
        __m128i out2 = _mm_shuffle_epi8(v, s2);
        __m128i out3 = _mm_shuffle_epi8(v, s3);

        /* Store to transposed positions (use memcpy for unaligned access) */
        uint32_t t0 = (uint32_t)_mm_cvtsi128_si32(out0);
        uint32_t t1 = (uint32_t)_mm_cvtsi128_si32(out1);
        uint32_t t2 = (uint32_t)_mm_cvtsi128_si32(out2);
        uint32_t t3 = (uint32_t)_mm_cvtsi128_si32(out3);
        memcpy(output + 0 * count + i, &t0, sizeof(uint32_t));
        memcpy(output + 1 * count + i, &t1, sizeof(uint32_t));
        memcpy(output + 2 * count + i, &t2, sizeof(uint32_t));
        memcpy(output + 3 * count + i, &t3, sizeof(uint32_t));
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            output[b * count + i] = src[i * 4 + b];
        }
    }
}

/**
 * Decode byte stream split floats using SSE.
 */
void carquet_sse_byte_stream_split_decode_float(
    const uint8_t* data,
    int64_t count,
    float* values) {

    uint8_t* dst = (uint8_t*)values;
    int64_t i = 0;

    /* Process 4 floats at a time */
    for (; i + 4 <= count; i += 4) {
        /* Load 4 bytes from each stream (use memcpy for unaligned access) */
        uint32_t b0, b1, b2, b3;
        memcpy(&b0, data + 0 * count + i, sizeof(uint32_t));
        memcpy(&b1, data + 1 * count + i, sizeof(uint32_t));
        memcpy(&b2, data + 2 * count + i, sizeof(uint32_t));
        memcpy(&b3, data + 3 * count + i, sizeof(uint32_t));

        __m128i v0 = _mm_cvtsi32_si128((int)b0);
        __m128i v1 = _mm_cvtsi32_si128((int)b1);
        __m128i v2 = _mm_cvtsi32_si128((int)b2);
        __m128i v3 = _mm_cvtsi32_si128((int)b3);

        /* Interleave bytes back into floats */
        __m128i lo01 = _mm_unpacklo_epi8(v0, v1);  /* a0b0 a1b1 a2b2 a3b3 ... */
        __m128i lo23 = _mm_unpacklo_epi8(v2, v3);  /* c0d0 c1d1 c2d2 c3d3 ... */
        __m128i result = _mm_unpacklo_epi16(lo01, lo23);  /* a0b0c0d0 a1b1c1d1 ... */

        _mm_storeu_si128((__m128i*)(dst + i * 4), result);
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            dst[i * 4 + b] = data[b * count + i];
        }
    }
}

/**
 * Encode doubles using byte stream split with SSE.
 */
void carquet_sse_byte_stream_split_encode_double(
    const double* values,
    int64_t count,
    uint8_t* output) {

    const uint8_t* src = (const uint8_t*)values;
    int64_t i = 0;

    /* Process 2 doubles (16 bytes) at a time */
    for (; i + 2 <= count; i += 2) {
        /* Transpose using shuffles for 8 byte streams */
        for (int b = 0; b < 8; b++) {
            output[b * count + i + 0] = ((const uint8_t*)(src + i * 8))[0 + b];
            output[b * count + i + 1] = ((const uint8_t*)(src + i * 8))[8 + b];
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
 * Decode byte stream split doubles using SSE.
 */
void carquet_sse_byte_stream_split_decode_double(
    const uint8_t* data,
    int64_t count,
    double* values) {

    uint8_t* dst = (uint8_t*)values;
    int64_t i = 0;

    /* Handle values */
    for (; i < count; i++) {
        for (int b = 0; b < 8; b++) {
            dst[i * 8 + b] = data[b * count + i];
        }
    }
}

/* ============================================================================
 * Delta Decoding - SSE Optimized (Prefix Sum)
 * ============================================================================
 */

/**
 * Apply prefix sum (cumulative sum) to int32 array using SSE.
 */
void carquet_sse_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial) {
    int32_t sum = initial;
    int64_t i = 0;

    /* SSE prefix sum for 4 elements at a time */
    for (; i + 4 <= count; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i*)(values + i));

        /* Partial prefix sums within the vector */
        /* v = [a, b, c, d] */
        /* After step 1: [a, a+b, c, c+d] */
        __m128i shifted1 = _mm_slli_si128(v, 4);
        v = _mm_add_epi32(v, shifted1);

        /* After step 2: [a, a+b, a+c, a+b+c+d] */
        __m128i shifted2 = _mm_slli_si128(v, 8);
        v = _mm_add_epi32(v, shifted2);

        /* Add running sum */
        __m128i sums = _mm_set1_epi32(sum);
        v = _mm_add_epi32(v, sums);
        _mm_storeu_si128((__m128i*)(values + i), v);

        /* Update running sum to last element */
        sum = _mm_extract_epi32(v, 3);
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

/**
 * Apply prefix sum to int64 array using SSE.
 */
void carquet_sse_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial) {
    int64_t sum = initial;
    int64_t i = 0;

    /* SSE prefix sum for 2 elements at a time */
    for (; i + 2 <= count; i += 2) {
        __m128i v = _mm_loadu_si128((const __m128i*)(values + i));

        /* v = [a, b] -> [a, a+b] */
        __m128i shifted = _mm_slli_si128(v, 8);
        v = _mm_add_epi64(v, shifted);

        /* Add running sum */
        __m128i sums = _mm_set1_epi64x(sum);
        v = _mm_add_epi64(v, sums);
        _mm_storeu_si128((__m128i*)(values + i), v);

        /* Update running sum */
        int64_t result[2];
        _mm_storeu_si128((__m128i*)result, v);
        sum = result[1];
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

/* ============================================================================
 * Dictionary Gather - SSE Optimized
 * ============================================================================
 */

/**
 * Gather int32 values from dictionary using indices (SSE).
 * Uses prefetching for better memory access patterns (matching NEON implementation).
 */
void carquet_sse_gather_i32(const int32_t* dict, const uint32_t* indices,
                             int64_t count, int32_t* output) {
    int64_t i = 0;

    /* Process 8 at a time with prefetching (like NEON) */
    for (; i + 8 <= count; i += 8) {
        /* Prefetch future indices */
        __builtin_prefetch(indices + i + 16, 0, 1);

        /* Prefetch dictionary entries for current batch */
        __builtin_prefetch(dict + indices[i], 0, 0);
        __builtin_prefetch(dict + indices[i + 2], 0, 0);
        __builtin_prefetch(dict + indices[i + 4], 0, 0);
        __builtin_prefetch(dict + indices[i + 6], 0, 0);

        /* First 4 values */
        int32_t v0 = dict[indices[i + 0]];
        int32_t v1 = dict[indices[i + 1]];
        int32_t v2 = dict[indices[i + 2]];
        int32_t v3 = dict[indices[i + 3]];
        __m128i result0 = _mm_set_epi32(v3, v2, v1, v0);
        _mm_storeu_si128((__m128i*)(output + i), result0);

        /* Second 4 values */
        int32_t v4 = dict[indices[i + 4]];
        int32_t v5 = dict[indices[i + 5]];
        int32_t v6 = dict[indices[i + 6]];
        int32_t v7 = dict[indices[i + 7]];
        __m128i result1 = _mm_set_epi32(v7, v6, v5, v4);
        _mm_storeu_si128((__m128i*)(output + i + 4), result1);
    }

    /* Process remaining 4 at a time */
    for (; i + 4 <= count; i += 4) {
        int32_t v0 = dict[indices[i + 0]];
        int32_t v1 = dict[indices[i + 1]];
        int32_t v2 = dict[indices[i + 2]];
        int32_t v3 = dict[indices[i + 3]];

        __m128i result = _mm_set_epi32(v3, v2, v1, v0);
        _mm_storeu_si128((__m128i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather float values from dictionary using indices (SSE).
 * Uses prefetching for better memory access patterns.
 */
void carquet_sse_gather_float(const float* dict, const uint32_t* indices,
                               int64_t count, float* output) {
    int64_t i = 0;

    /* Process 8 at a time with prefetching */
    for (; i + 8 <= count; i += 8) {
        /* Prefetch future indices */
        __builtin_prefetch(indices + i + 16, 0, 1);

        /* Prefetch dictionary entries */
        __builtin_prefetch(dict + indices[i], 0, 0);
        __builtin_prefetch(dict + indices[i + 2], 0, 0);
        __builtin_prefetch(dict + indices[i + 4], 0, 0);
        __builtin_prefetch(dict + indices[i + 6], 0, 0);

        /* First 4 values */
        float v0 = dict[indices[i + 0]];
        float v1 = dict[indices[i + 1]];
        float v2 = dict[indices[i + 2]];
        float v3 = dict[indices[i + 3]];
        __m128 result0 = _mm_set_ps(v3, v2, v1, v0);
        _mm_storeu_ps(output + i, result0);

        /* Second 4 values */
        float v4 = dict[indices[i + 4]];
        float v5 = dict[indices[i + 5]];
        float v6 = dict[indices[i + 6]];
        float v7 = dict[indices[i + 7]];
        __m128 result1 = _mm_set_ps(v7, v6, v5, v4);
        _mm_storeu_ps(output + i + 4, result1);
    }

    /* Process remaining 4 at a time */
    for (; i + 4 <= count; i += 4) {
        float v0 = dict[indices[i + 0]];
        float v1 = dict[indices[i + 1]];
        float v2 = dict[indices[i + 2]];
        float v3 = dict[indices[i + 3]];

        __m128 result = _mm_set_ps(v3, v2, v1, v0);
        _mm_storeu_ps(output + i, result);
    }

    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather int64 values from dictionary using indices (SSE).
 * Uses prefetching for better memory access patterns.
 */
void carquet_sse_gather_i64(const int64_t* dict, const uint32_t* indices,
                             int64_t count, int64_t* output) {
    int64_t i = 0;

    /* Process 4 at a time with prefetching */
    for (; i + 4 <= count; i += 4) {
        /* Prefetch future indices */
        __builtin_prefetch(indices + i + 8, 0, 1);

        /* Prefetch dictionary entries */
        __builtin_prefetch(dict + indices[i], 0, 0);
        __builtin_prefetch(dict + indices[i + 2], 0, 0);

        int64_t v0 = dict[indices[i + 0]];
        int64_t v1 = dict[indices[i + 1]];
        int64_t v2 = dict[indices[i + 2]];
        int64_t v3 = dict[indices[i + 3]];

        __m128i result0 = _mm_set_epi64x(v1, v0);
        __m128i result1 = _mm_set_epi64x(v3, v2);
        _mm_storeu_si128((__m128i*)(output + i), result0);
        _mm_storeu_si128((__m128i*)(output + i + 2), result1);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather double values from dictionary using indices (SSE).
 * Uses prefetching for better memory access patterns.
 */
void carquet_sse_gather_double(const double* dict, const uint32_t* indices,
                                int64_t count, double* output) {
    int64_t i = 0;

    /* Process 4 at a time with prefetching */
    for (; i + 4 <= count; i += 4) {
        /* Prefetch future indices */
        __builtin_prefetch(indices + i + 8, 0, 1);

        /* Prefetch dictionary entries */
        __builtin_prefetch(dict + indices[i], 0, 0);
        __builtin_prefetch(dict + indices[i + 2], 0, 0);

        double v0 = dict[indices[i + 0]];
        double v1 = dict[indices[i + 1]];
        double v2 = dict[indices[i + 2]];
        double v3 = dict[indices[i + 3]];

        __m128d result0 = _mm_set_pd(v1, v0);
        __m128d result1 = _mm_set_pd(v3, v2);
        _mm_storeu_pd(output + i, result0);
        _mm_storeu_pd(output + i + 2, result1);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/* ============================================================================
 * CRC32C - SSE4.2 Hardware Acceleration
 * ============================================================================
 */

/**
 * Compute CRC32C using SSE4.2 hardware instructions.
 */
uint32_t carquet_sse_crc32c(uint32_t crc, const uint8_t* data, size_t len) {
    size_t i = 0;

#ifdef __x86_64__
    /* Process 8 bytes at a time on 64-bit */
    for (; i + 8 <= len; i += 8) {
        uint64_t val;
        memcpy(&val, data + i, 8);
        crc = (uint32_t)_mm_crc32_u64(crc, val);
    }
#endif

    /* Process 4 bytes at a time */
    for (; i + 4 <= len; i += 4) {
        uint32_t val;
        memcpy(&val, data + i, 4);
        crc = _mm_crc32_u32(crc, val);
    }

    /* Process 2 bytes */
    if (i + 2 <= len) {
        uint16_t val;
        memcpy(&val, data + i, 2);
        crc = _mm_crc32_u16(crc, val);
        i += 2;
    }

    /* Process remaining byte */
    if (i < len) {
        crc = _mm_crc32_u8(crc, data[i]);
    }

    return crc;
}

/* ============================================================================
 * Memcpy/Memset - SSE Optimized
 * ============================================================================
 */

/**
 * Fast memset for small-medium buffers using SSE.
 */
void carquet_sse_memset_small(void* dest, uint8_t value, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    __m128i v = _mm_set1_epi8((char)value);

    while (n >= 64) {
        _mm_storeu_si128((__m128i*)(d + 0), v);
        _mm_storeu_si128((__m128i*)(d + 16), v);
        _mm_storeu_si128((__m128i*)(d + 32), v);
        _mm_storeu_si128((__m128i*)(d + 48), v);
        d += 64;
        n -= 64;
    }

    while (n >= 16) {
        _mm_storeu_si128((__m128i*)d, v);
        d += 16;
        n -= 16;
    }

    while (n > 0) {
        *d++ = value;
        n--;
    }
}

/**
 * Fast memcpy for small-medium buffers using SSE.
 */
void carquet_sse_memcpy_small(void* dest, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    const uint8_t* s = (const uint8_t*)src;

    while (n >= 64) {
        __m128i v0 = _mm_loadu_si128((const __m128i*)(s + 0));
        __m128i v1 = _mm_loadu_si128((const __m128i*)(s + 16));
        __m128i v2 = _mm_loadu_si128((const __m128i*)(s + 32));
        __m128i v3 = _mm_loadu_si128((const __m128i*)(s + 48));
        _mm_storeu_si128((__m128i*)(d + 0), v0);
        _mm_storeu_si128((__m128i*)(d + 16), v1);
        _mm_storeu_si128((__m128i*)(d + 32), v2);
        _mm_storeu_si128((__m128i*)(d + 48), v3);
        d += 64;
        s += 64;
        n -= 64;
    }

    while (n >= 16) {
        _mm_storeu_si128((__m128i*)d, _mm_loadu_si128((const __m128i*)s));
        d += 16;
        s += 16;
        n -= 16;
    }

    while (n > 0) {
        *d++ = *s++;
        n--;
    }
}

/* ============================================================================
 * Boolean Unpacking - SSE Optimized
 * ============================================================================
 */

/**
 * Unpack boolean values from packed bits to byte array.
 * Each output byte is 0 or 1.
 */
void carquet_sse_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    int64_t i = 0;

    /* Process 16 bools (2 bytes) at a time */
    for (; i + 16 <= count; i += 16) {
        int byte_idx = (int)(i / 8);
        uint16_t packed;
        memcpy(&packed, input + byte_idx, 2);

        __m128i bits = _mm_set1_epi16(packed);

        /* Create masks for each bit position */
        __m128i mask = _mm_set_epi8(
            (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
            (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01
        );

        /* Expand each byte for its corresponding bits */
        static const int8_t shuf[16] = {
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1
        };
        __m128i shuffled = _mm_shuffle_epi8(bits, _mm_loadu_si128((const __m128i*)shuf));

        /* AND with mask and normalize to 0/1 */
        __m128i masked = _mm_and_si128(shuffled, mask);
        __m128i result = _mm_min_epu8(masked, _mm_set1_epi8(1));

        _mm_storeu_si128((__m128i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        int byte_idx = (int)(i / 8);
        int bit_idx = (int)(i % 8);
        output[i] = (input[byte_idx] >> bit_idx) & 1;
    }
}

/**
 * Pack boolean values from byte array to packed bits.
 * Input bytes should be 0 or 1.
 */
void carquet_sse_pack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    int64_t i = 0;

    /* Process 8 bools (1 output byte) at a time using movemask trick:
     * 1. Load 8 bytes (each 0 or 1)
     * 2. Shift left by 7 within 32-bit lanes: moves bit 0 of each byte to bit 7
     * 3. movemask extracts bit 7 from each byte position
     */
    for (; i + 8 <= count; i += 8) {
        __m128i bools = _mm_loadl_epi64((const __m128i*)(input + i));
        __m128i shifted = _mm_slli_epi32(bools, 7);
        output[i / 8] = (uint8_t)_mm_movemask_epi8(shifted);
    }

    /* Handle remaining */
    if (i < count) {
        uint8_t byte = 0;
        for (int64_t j = 0; j < count - i && j < 8; j++) {
            if (input[i + j]) {
                byte |= (1 << j);
            }
        }
        output[i / 8] = byte;
    }
}

/* ============================================================================
 * Compression Helpers
 * ============================================================================
 */

/**
 * Fast match copy for LZ4/Snappy decompression.
 * Handles overlapping copies correctly using SSE optimizations.
 */
void carquet_sse_match_copy(uint8_t* dst, const uint8_t* src, size_t len, size_t offset) {
    if (offset >= 16) {
        /* Non-overlapping: use full SSE copies */
        while (len >= 16) {
            _mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((const __m128i*)src));
            dst += 16;
            src += 16;
            len -= 16;
        }

        if (len >= 8) {
            _mm_storel_epi64((__m128i*)dst, _mm_loadl_epi64((const __m128i*)src));
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
        __m128i v = _mm_set1_epi8((char)val);

        while (len >= 16) {
            _mm_storeu_si128((__m128i*)dst, v);
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
        __m128i v = _mm_set1_epi32((int32_t)pattern);

        while (len >= 16) {
            _mm_storeu_si128((__m128i*)dst, v);
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
 * Count matching bytes between two buffers using SSE.
 * Returns the number of matching bytes from the start.
 */
size_t carquet_sse_match_length(const uint8_t* p, const uint8_t* match, const uint8_t* limit) {
    const uint8_t* start = p;

    /* Fast path: compare 16 bytes at a time */
    while (p + 16 <= limit) {
        __m128i a = _mm_loadu_si128((const __m128i*)p);
        __m128i b = _mm_loadu_si128((const __m128i*)match);
        __m128i cmp = _mm_cmpeq_epi8(a, b);
        int mask = _mm_movemask_epi8(cmp);

        if (mask != 0xFFFF) {
            /* Find first differing byte */
            int first_diff = __builtin_ctz(~mask);
            return (size_t)(p - start) + (size_t)first_diff;
        }

        p += 16;
        match += 16;
    }

    /* Byte-by-byte for remaining */
    while (p < limit && *p == *match) {
        p++;
        match++;
    }

    return (size_t)(p - start);
}

/* ============================================================================
 * Definition Level Processing (Critical for Read Performance)
 * ============================================================================
 */

/**
 * Count non-null values using SIMD.
 * Counts how many def_levels[i] == max_def_level.
 */
int64_t carquet_sse_count_non_nulls(const int16_t* def_levels, int64_t count, int16_t max_def_level) {
    int64_t non_null_count = 0;
    int64_t i = 0;

    __m128i max_vec = _mm_set1_epi16(max_def_level);

    /* Process 8 int16_t values at a time */
    for (; i + 8 <= count; i += 8) {
        __m128i levels = _mm_loadu_si128((const __m128i*)(def_levels + i));
        __m128i cmp = _mm_cmpeq_epi16(levels, max_vec);
        int mask = _mm_movemask_epi8(cmp);
        /* Each matching int16 produces 2 bits set, so count and divide by 2 */
        non_null_count += __builtin_popcount(mask) >> 1;
    }

    /* Handle remaining */
    for (; i < count; i++) {
        if (def_levels[i] == max_def_level) {
            non_null_count++;
        }
    }

    return non_null_count;
}

/**
 * Build null bitmap from definition levels using SIMD.
 * Sets bit to 1 if def_levels[i] < max_def_level (null).
 */
void carquet_sse_build_null_bitmap(const int16_t* def_levels, int64_t count,
                                    int16_t max_def_level, uint8_t* null_bitmap) {
    int64_t i = 0;

    __m128i max_vec = _mm_set1_epi16(max_def_level);
    __m128i zero = _mm_setzero_si128();

    /* Process 8 int16_t values -> 1 byte of bitmap */
    int64_t full_bytes = count / 8;
    for (int64_t b = 0; b < full_bytes; b++) {
        __m128i levels = _mm_loadu_si128((const __m128i*)(def_levels + b * 8));
        /* levels < max_def means null: result is 0x0000 or 0xFFFF per lane */
        __m128i cmp = _mm_cmplt_epi16(levels, max_vec);
        /* Pack 8 int16 results (0x0000 or 0xFFFF) to 8 int8 (0x00 or 0xFF) */
        __m128i packed = _mm_packs_epi16(cmp, zero);
        /* movemask extracts bit 7 from each byte -> 8-bit result in low byte */
        null_bitmap[b] = (uint8_t)_mm_movemask_epi8(packed);
        i += 8;
    }

    /* Handle remaining bits */
    if (i < count) {
        uint8_t null_bits = 0;
        for (int64_t j = 0; i + j < count && j < 8; j++) {
            if (def_levels[i + j] < max_def_level) {
                null_bits |= (1 << j);
            }
        }
        null_bitmap[full_bytes] = null_bits;
    }
}

/**
 * Fill definition levels with a constant value using SIMD.
 */
void carquet_sse_fill_def_levels(int16_t* def_levels, int64_t count, int16_t value) {
    int64_t i = 0;
    __m128i val_vec = _mm_set1_epi16(value);

    /* Process 8 int16_t values at a time */
    for (; i + 8 <= count; i += 8) {
        _mm_storeu_si128((__m128i*)(def_levels + i), val_vec);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        def_levels[i] = value;
    }
}

#endif /* __SSE4_2__ */
#endif /* x86 */
