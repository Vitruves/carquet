/**
 * @file avx2_ops.c
 * @brief AVX2 optimized operations for x86-64 processors
 *
 * Provides SIMD-accelerated implementations using 256-bit vectors:
 * - Bit unpacking for common bit widths
 * - Byte stream split/merge (for BYTE_STREAM_SPLIT encoding)
 * - Delta decoding (prefix sums)
 * - Dictionary gather operations (using AVX2 gather instructions)
 * - Boolean packing/unpacking
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
/* Check for AVX2 support - MSVC defines __AVX2__ when /arch:AVX2 is used */
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))

#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <immintrin.h>

/* ============================================================================
 * Bit Unpacking - AVX2 Optimized
 * ============================================================================
 */

/**
 * Unpack 64 1-bit values using AVX2.
 * Input: 8 bytes, Output: 64 x uint32_t
 */
void carquet_avx2_bitunpack64_1bit(const uint8_t* input, uint32_t* values) {
    /* For each byte, extract bits */
    for (int b = 0; b < 8; b++) {
        uint8_t byte_val = input[b];
        for (int i = 0; i < 8; i++) {
            values[b * 8 + i] = (byte_val >> i) & 1;
        }
    }
}

/**
 * Unpack 16 4-bit values using AVX2.
 */
void carquet_avx2_bitunpack16_4bit(const uint8_t* input, uint32_t* values) {
    /* Load 8 bytes containing 16 x 4-bit values */
    __m128i bytes = _mm_loadl_epi64((const __m128i*)input);

    /* Split nibbles */
    __m128i lo_nibbles = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
    __m128i hi_nibbles = _mm_srli_epi16(bytes, 4);
    hi_nibbles = _mm_and_si128(hi_nibbles, _mm_set1_epi8(0x0F));

    /* Interleave */
    __m128i interleaved = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);

    /* Expand to 32-bit using AVX2 */
    __m256i result = _mm256_cvtepu8_epi32(interleaved);
    _mm256_storeu_si256((__m256i*)values, result);

    /* Process second half */
    __m128i second_half = _mm_unpackhi_epi64(interleaved, interleaved);
    result = _mm256_cvtepu8_epi32(second_half);
    _mm256_storeu_si256((__m256i*)(values + 8), result);
}

/**
 * Unpack 16 8-bit values using AVX2 (widen u8 to u32).
 */
void carquet_avx2_bitunpack16_8bit(const uint8_t* input, uint32_t* values) {
    /* Load 16 bytes */
    __m128i bytes = _mm_loadu_si128((const __m128i*)input);

    /* Expand low 8 bytes to 8 x 32-bit */
    __m256i lo = _mm256_cvtepu8_epi32(bytes);
    _mm256_storeu_si256((__m256i*)values, lo);

    /* Expand high 8 bytes to 8 x 32-bit */
    __m128i hi_bytes = _mm_srli_si128(bytes, 8);
    __m256i hi = _mm256_cvtepu8_epi32(hi_bytes);
    _mm256_storeu_si256((__m256i*)(values + 8), hi);
}

/**
 * Unpack 8 16-bit values to 32-bit using AVX2.
 */
void carquet_avx2_bitunpack8_16bit(const uint8_t* input, uint32_t* values) {
    __m128i words = _mm_loadu_si128((const __m128i*)input);
    __m256i result = _mm256_cvtepu16_epi32(words);
    _mm256_storeu_si256((__m256i*)values, result);
}

/* ============================================================================
 * Byte Stream Split - AVX2 Optimized
 * ============================================================================
 */

/**
 * Encode floats using byte stream split with AVX2.
 * Processes 8 floats (32 bytes) at a time.
 */
void carquet_avx2_byte_stream_split_encode_float(
    const float* values,
    int64_t count,
    uint8_t* output) {

    const uint8_t* src = (const uint8_t*)values;
    int64_t i = 0;

    /* Process 8 floats (32 bytes) at a time */
    for (; i + 8 <= count; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(src + i * 4));

        /* Transpose using shuffles - extract byte 0 from each float */
        static const int8_t shuf_b0[32] = {
            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        };
        static const int8_t shuf_b1[32] = {
            1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        };
        static const int8_t shuf_b2[32] = {
            2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        };
        static const int8_t shuf_b3[32] = {
            3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        };

        __m256i s0 = _mm256_loadu_si256((const __m256i*)shuf_b0);
        __m256i s1 = _mm256_loadu_si256((const __m256i*)shuf_b1);
        __m256i s2 = _mm256_loadu_si256((const __m256i*)shuf_b2);
        __m256i s3 = _mm256_loadu_si256((const __m256i*)shuf_b3);

        __m256i out0 = _mm256_shuffle_epi8(v, s0);
        __m256i out1 = _mm256_shuffle_epi8(v, s1);
        __m256i out2 = _mm256_shuffle_epi8(v, s2);
        __m256i out3 = _mm256_shuffle_epi8(v, s3);

        /* Extract and combine low and high 128-bit lanes */
        uint32_t b0_lo = _mm256_extract_epi32(out0, 0);
        uint32_t b0_hi = _mm256_extract_epi32(out0, 4);
        uint32_t b1_lo = _mm256_extract_epi32(out1, 0);
        uint32_t b1_hi = _mm256_extract_epi32(out1, 4);
        uint32_t b2_lo = _mm256_extract_epi32(out2, 0);
        uint32_t b2_hi = _mm256_extract_epi32(out2, 4);
        uint32_t b3_lo = _mm256_extract_epi32(out3, 0);
        uint32_t b3_hi = _mm256_extract_epi32(out3, 4);

        /* Store to transposed positions (use memcpy for unaligned access) */
        memcpy(output + 0 * count + i, &b0_lo, sizeof(uint32_t));
        memcpy(output + 0 * count + i + 4, &b0_hi, sizeof(uint32_t));
        memcpy(output + 1 * count + i, &b1_lo, sizeof(uint32_t));
        memcpy(output + 1 * count + i + 4, &b1_hi, sizeof(uint32_t));
        memcpy(output + 2 * count + i, &b2_lo, sizeof(uint32_t));
        memcpy(output + 2 * count + i + 4, &b2_hi, sizeof(uint32_t));
        memcpy(output + 3 * count + i, &b3_lo, sizeof(uint32_t));
        memcpy(output + 3 * count + i + 4, &b3_hi, sizeof(uint32_t));
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            output[b * count + i] = src[i * 4 + b];
        }
    }
}

/**
 * Decode byte stream split floats using AVX2.
 */
void carquet_avx2_byte_stream_split_decode_float(
    const uint8_t* data,
    int64_t count,
    float* values) {

    uint8_t* dst = (uint8_t*)values;
    int64_t i = 0;

    /* Process 8 floats at a time */
    for (; i + 8 <= count; i += 8) {
        /* Load 8 bytes from each of the 4 streams (use memcpy for unaligned access) */
        uint64_t t0, t1, t2, t3;
        memcpy(&t0, data + 0 * count + i, sizeof(uint64_t));
        memcpy(&t1, data + 1 * count + i, sizeof(uint64_t));
        memcpy(&t2, data + 2 * count + i, sizeof(uint64_t));
        memcpy(&t3, data + 3 * count + i, sizeof(uint64_t));
        __m128i b0 = _mm_cvtsi64_si128((long long)t0);
        __m128i b1 = _mm_cvtsi64_si128((long long)t1);
        __m128i b2 = _mm_cvtsi64_si128((long long)t2);
        __m128i b3 = _mm_cvtsi64_si128((long long)t3);

        /* Interleave bytes to reconstruct floats */
        __m128i lo01 = _mm_unpacklo_epi8(b0, b1);  /* a0b0 a1b1 a2b2 ... */
        __m128i lo23 = _mm_unpacklo_epi8(b2, b3);  /* c0d0 c1d1 c2d2 ... */

        __m128i result_lo = _mm_unpacklo_epi16(lo01, lo23);  /* a0b0c0d0 a1b1c1d1 ... */
        __m128i result_hi = _mm_unpackhi_epi16(lo01, lo23);

        _mm_storeu_si128((__m128i*)(dst + i * 4), result_lo);
        _mm_storeu_si128((__m128i*)(dst + i * 4 + 16), result_hi);
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            dst[i * 4 + b] = data[b * count + i];
        }
    }
}

/**
 * Encode doubles using byte stream split with AVX2.
 */
void carquet_avx2_byte_stream_split_encode_double(
    const double* values,
    int64_t count,
    uint8_t* output) {

    const uint8_t* src = (const uint8_t*)values;
    int64_t i = 0;

    /* Process 4 doubles (32 bytes) at a time */
    for (; i + 4 <= count; i += 4) {
        /* Transpose - extract each byte position */
        for (int b = 0; b < 8; b++) {
            output[b * count + i + 0] = src[i * 8 + 0 + b];
            output[b * count + i + 1] = src[i * 8 + 8 + b];
            output[b * count + i + 2] = src[i * 8 + 16 + b];
            output[b * count + i + 3] = src[i * 8 + 24 + b];
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
 * Decode byte stream split doubles using AVX2.
 */
void carquet_avx2_byte_stream_split_decode_double(
    const uint8_t* data,
    int64_t count,
    double* values) {

    uint8_t* dst = (uint8_t*)values;
    int64_t i = 0;

    /* Process values */
    for (; i < count; i++) {
        for (int b = 0; b < 8; b++) {
            dst[i * 8 + b] = data[b * count + i];
        }
    }
}

/* ============================================================================
 * Delta Decoding - AVX2 Optimized (Prefix Sum)
 * ============================================================================
 */

/**
 * Apply prefix sum (cumulative sum) to int32 array using AVX2.
 */
void carquet_avx2_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial) {
    int32_t sum = initial;
    int64_t i = 0;

    /* AVX2 prefix sum for 8 elements at a time */
    for (; i + 8 <= count; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(values + i));

        /* Partial prefix sums within the vector */
        /* Step 1: Add adjacent pairs */
        __m256i shifted1 = _mm256_slli_si256(v, 4);
        v = _mm256_add_epi32(v, shifted1);

        /* Step 2: Add pairs that are 2 apart */
        __m256i shifted2 = _mm256_slli_si256(v, 8);
        v = _mm256_add_epi32(v, shifted2);

        /* Step 3: Handle cross-lane (bit tricky with AVX2) */
        /* Extract lane 0's last value and add to all of lane 1 */
        __m128i lo = _mm256_extracti128_si256(v, 0);
        __m128i hi = _mm256_extracti128_si256(v, 1);

        int32_t lane0_sum = _mm_extract_epi32(lo, 3);
        __m128i lane0_broadcast = _mm_set1_epi32(lane0_sum);
        hi = _mm_add_epi32(hi, lane0_broadcast);

        v = _mm256_inserti128_si256(v, hi, 1);

        /* Add running sum */
        __m256i sums = _mm256_set1_epi32(sum);
        v = _mm256_add_epi32(v, sums);
        _mm256_storeu_si256((__m256i*)(values + i), v);

        /* Update running sum to last element */
        sum = _mm256_extract_epi32(v, 7);
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

/**
 * Apply prefix sum to int64 array using AVX2.
 */
void carquet_avx2_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial) {
    int64_t sum = initial;
    int64_t i = 0;

    /* AVX2 prefix sum for 4 elements at a time */
    for (; i + 4 <= count; i += 4) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(values + i));

        /* Partial prefix sums */
        __m256i shifted1 = _mm256_slli_si256(v, 8);
        v = _mm256_add_epi64(v, shifted1);

        /* Cross-lane fixup */
        __m128i lo = _mm256_extracti128_si256(v, 0);
        __m128i hi = _mm256_extracti128_si256(v, 1);

        int64_t lane0_last;
        _mm_storel_epi64((__m128i*)&lane0_last, _mm_srli_si128(lo, 8));
        __m128i lane0_broadcast = _mm_set1_epi64x(lane0_last);
        hi = _mm_add_epi64(hi, lane0_broadcast);

        v = _mm256_inserti128_si256(v, hi, 1);

        /* Add running sum */
        __m256i sums = _mm256_set1_epi64x(sum);
        v = _mm256_add_epi64(v, sums);
        _mm256_storeu_si256((__m256i*)(values + i), v);

        /* Update running sum */
        int64_t result[4];
        _mm256_storeu_si256((__m256i*)result, v);
        sum = result[3];
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

/* ============================================================================
 * Dictionary Gather - AVX2 Optimized (True Hardware Gather)
 * ============================================================================
 */

/**
 * Gather int32 values from dictionary using AVX2 gather instructions.
 */
void carquet_avx2_gather_i32(const int32_t* dict, const uint32_t* indices,
                              int64_t count, int32_t* output) {
    int64_t i = 0;

    /* Process 8 at a time using AVX2 gather */
    for (; i + 8 <= count; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i*)(indices + i));
        __m256i result = _mm256_i32gather_epi32(dict, idx, 4);  /* Scale = 4 bytes per int32 */
        _mm256_storeu_si256((__m256i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather int64 values from dictionary using AVX2 gather instructions.
 */
void carquet_avx2_gather_i64(const int64_t* dict, const uint32_t* indices,
                              int64_t count, int64_t* output) {
    int64_t i = 0;

    /* Process 4 at a time using AVX2 gather */
    for (; i + 4 <= count; i += 4) {
        __m128i idx = _mm_loadu_si128((const __m128i*)(indices + i));
        __m256i result = _mm256_i32gather_epi64((const long long*)dict, idx, 8);
        _mm256_storeu_si256((__m256i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather float values from dictionary using AVX2 gather instructions.
 * Note: float and int32 are both 4 bytes, so we reuse gather_i32 via cast.
 */
void carquet_avx2_gather_float(const float* dict, const uint32_t* indices,
                                int64_t count, float* output) {
    /* Data movement doesn't care about type - reuse int32 implementation */
    carquet_avx2_gather_i32((const int32_t*)dict, indices, count, (int32_t*)output);
}

/**
 * Gather double values from dictionary using AVX2 gather instructions.
 * Note: double and int64 are both 8 bytes, so we reuse gather_i64 via cast.
 */
void carquet_avx2_gather_double(const double* dict, const uint32_t* indices,
                                 int64_t count, double* output) {
    /* Data movement doesn't care about type - reuse int64 implementation */
    carquet_avx2_gather_i64((const int64_t*)dict, indices, count, (int64_t*)output);
}

/* ============================================================================
 * Memcpy/Memset - AVX2 Optimized
 * ============================================================================
 */

/**
 * Fast memset for buffers using AVX2.
 */
void carquet_avx2_memset(void* dest, uint8_t value, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    __m256i v = _mm256_set1_epi8((char)value);

    while (n >= 128) {
        _mm256_storeu_si256((__m256i*)(d + 0), v);
        _mm256_storeu_si256((__m256i*)(d + 32), v);
        _mm256_storeu_si256((__m256i*)(d + 64), v);
        _mm256_storeu_si256((__m256i*)(d + 96), v);
        d += 128;
        n -= 128;
    }

    while (n >= 32) {
        _mm256_storeu_si256((__m256i*)d, v);
        d += 32;
        n -= 32;
    }

    /* Handle tail with SSE */
    __m128i v128 = _mm_set1_epi8((char)value);
    while (n >= 16) {
        _mm_storeu_si128((__m128i*)d, v128);
        d += 16;
        n -= 16;
    }

    while (n > 0) {
        *d++ = value;
        n--;
    }
}

/**
 * Fast memcpy for buffers using AVX2.
 */
void carquet_avx2_memcpy(void* dest, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    const uint8_t* s = (const uint8_t*)src;

    while (n >= 128) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)(s + 0));
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(s + 32));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(s + 64));
        __m256i v3 = _mm256_loadu_si256((const __m256i*)(s + 96));
        _mm256_storeu_si256((__m256i*)(d + 0), v0);
        _mm256_storeu_si256((__m256i*)(d + 32), v1);
        _mm256_storeu_si256((__m256i*)(d + 64), v2);
        _mm256_storeu_si256((__m256i*)(d + 96), v3);
        d += 128;
        s += 128;
        n -= 128;
    }

    while (n >= 32) {
        _mm256_storeu_si256((__m256i*)d, _mm256_loadu_si256((const __m256i*)s));
        d += 32;
        s += 32;
        n -= 32;
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
 * Boolean Unpacking - AVX2 Optimized
 * ============================================================================
 */

/**
 * Unpack boolean values from packed bits to byte array using AVX2.
 * Each output byte is 0 or 1.
 */
void carquet_avx2_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    int64_t i = 0;

    /* Process 32 bools (4 bytes) at a time */
    for (; i + 32 <= count; i += 32) {
        int byte_idx = (int)(i / 8);
        uint32_t packed;
        memcpy(&packed, input + byte_idx, 4);

        __m256i bits = _mm256_set1_epi32(packed);

        /* Create masks for each bit position */
        __m256i mask = _mm256_set_epi8(
            (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
            (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
            (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
            (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01
        );

        /* Shuffle to put the right byte in each position */
        static const int8_t shuf[32] = {
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3
        };
        __m256i shuffled = _mm256_shuffle_epi8(bits, _mm256_loadu_si256((const __m256i*)shuf));

        /* AND with mask and normalize to 0/1 */
        __m256i masked = _mm256_and_si256(shuffled, mask);
        __m256i result = _mm256_min_epu8(masked, _mm256_set1_epi8(1));

        _mm256_storeu_si256((__m256i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        int byte_idx = (int)(i / 8);
        int bit_idx = (int)(i % 8);
        output[i] = (input[byte_idx] >> bit_idx) & 1;
    }
}

/**
 * Pack boolean values from byte array to packed bits using AVX2.
 */
void carquet_avx2_pack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    int64_t i = 0;

    /* Process 8 bools at a time using movemask */
    for (; i + 8 <= count; i += 8) {
        __m128i bools = _mm_loadl_epi64((const __m128i*)(input + i));

        /* Actually simpler: multiply by bit positions */
        __m128i mult = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                     (char)128, 64, 32, 16, 8, 4, 2, 1);
        __m128i zero = _mm_setzero_si128();
        __m128i words = _mm_unpacklo_epi8(bools, zero);
        __m128i mwords = _mm_unpacklo_epi8(mult, zero);

        __m128i prod = _mm_mullo_epi16(words, mwords);
        prod = _mm_add_epi16(prod, _mm_srli_si128(prod, 2));
        prod = _mm_add_epi16(prod, _mm_srli_si128(prod, 4));
        prod = _mm_add_epi16(prod, _mm_srli_si128(prod, 8));

        output[i / 8] = (uint8_t)_mm_extract_epi16(prod, 0);
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
 * RLE Run Detection - AVX2 Optimized
 * ============================================================================
 */

/**
 * Find the length of a run of repeated values.
 * Returns the number of consecutive identical values starting at the given position.
 */
int64_t carquet_avx2_find_run_length_i32(const int32_t* values, int64_t count) {
    if (count == 0) return 0;

    int32_t first = values[0];
    __m256i target = _mm256_set1_epi32(first);
    int64_t i = 0;

    /* Check 8 at a time */
    for (; i + 8 <= count; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(values + i));
        __m256i cmp = _mm256_cmpeq_epi32(v, target);
        int mask = _mm256_movemask_epi8(cmp);

        if (mask != -1) {  /* Not all equal */
            /* Find first mismatch */
            for (int64_t j = i; j < i + 8 && j < count; j++) {
                if (values[j] != first) {
                    return j;
                }
            }
        }
    }

    /* Handle remaining */
    for (; i < count; i++) {
        if (values[i] != first) {
            return i;
        }
    }

    return count;
}

#endif /* __AVX2__ */
#endif /* x86 */
