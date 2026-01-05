/**
 * @file avx512_ops.c
 * @brief AVX-512 optimized operations for x86-64 processors
 *
 * Provides SIMD-accelerated implementations using 512-bit vectors:
 * - Bit unpacking for various bit widths
 * - Byte stream split/merge (for BYTE_STREAM_SPLIT encoding)
 * - Delta decoding (prefix sums)
 * - Dictionary gather operations (using AVX-512 scatter/gather)
 * - Boolean packing/unpacking
 * - Masked operations for predicated processing
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
/* Check for AVX-512 support */
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(__AVX512F__))

#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <immintrin.h>

/* Portable count trailing zeros */
static inline int portable_ctz(unsigned int v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(v);
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, v);
    return (int)index;
#else
    int n = 0;
    if (!(v & 0xFFFF)) { n += 16; v >>= 16; }
    if (!(v & 0xFF)) { n += 8; v >>= 8; }
    if (!(v & 0xF)) { n += 4; v >>= 4; }
    if (!(v & 0x3)) { n += 2; v >>= 2; }
    if (!(v & 0x1)) { n += 1; }
    return n;
#endif
}

/* ============================================================================
 * Bit Unpacking - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Unpack 32 8-bit values to 32-bit using AVX-512.
 */
void carquet_avx512_bitunpack32_8bit(const uint8_t* input, uint32_t* values) {
    /* Load 32 bytes as two 128-bit halves */
    __m128i bytes_lo = _mm_loadu_si128((const __m128i*)input);
    __m128i bytes_hi = _mm_loadu_si128((const __m128i*)(input + 16));

    /* Expand each half to 32-bit using AVX-512 (16 x 8-bit -> 16 x 32-bit) */
    __m512i result_lo = _mm512_cvtepu8_epi32(bytes_lo);
    __m512i result_hi = _mm512_cvtepu8_epi32(bytes_hi);

    _mm512_storeu_si512((__m512i*)values, result_lo);
    _mm512_storeu_si512((__m512i*)(values + 16), result_hi);
}

/**
 * Unpack 16 16-bit values to 32-bit using AVX-512.
 */
void carquet_avx512_bitunpack16_16bit(const uint8_t* input, uint32_t* values) {
    __m256i words = _mm256_loadu_si256((const __m256i*)input);
    __m512i result = _mm512_cvtepu16_epi32(words);
    _mm512_storeu_si512((__m512i*)values, result);
}

/**
 * Unpack 32 4-bit values to 32-bit using AVX-512.
 */
void carquet_avx512_bitunpack32_4bit(const uint8_t* input, uint32_t* values) {
    /* Load 16 bytes containing 32 x 4-bit values */
    __m128i bytes = _mm_loadu_si128((const __m128i*)input);

    /* Split nibbles */
    __m128i lo_nibbles = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
    __m128i hi_nibbles = _mm_srli_epi16(bytes, 4);
    hi_nibbles = _mm_and_si128(hi_nibbles, _mm_set1_epi8(0x0F));

    /* Interleave to get correct order - produces two 128-bit results */
    __m128i interleaved_lo = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
    __m128i interleaved_hi = _mm_unpackhi_epi8(lo_nibbles, hi_nibbles);

    /* Expand each half to 32-bit using AVX-512 (16 x 8-bit -> 16 x 32-bit) */
    __m512i result_lo = _mm512_cvtepu8_epi32(interleaved_lo);
    __m512i result_hi = _mm512_cvtepu8_epi32(interleaved_hi);

    _mm512_storeu_si512((__m512i*)values, result_lo);
    _mm512_storeu_si512((__m512i*)(values + 16), result_hi);
}

/* ============================================================================
 * Byte Stream Split - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Encode floats using byte stream split with AVX-512.
 * Processes 16 floats (64 bytes) at a time using VBMI byte permutation.
 */
void carquet_avx512_byte_stream_split_encode_float(
    const float* values,
    int64_t count,
    uint8_t* output) {

    const uint8_t* src = (const uint8_t*)values;
    int64_t i = 0;

#ifdef __AVX512VBMI__
    /* Single permutation that places all 4 byte streams in the 4 128-bit lanes:
     * Lane 0 (bits 0-127):   byte 0 from each of 16 floats
     * Lane 1 (bits 128-255): byte 1 from each of 16 floats
     * Lane 2 (bits 256-383): byte 2 from each of 16 floats
     * Lane 3 (bits 384-511): byte 3 from each of 16 floats
     */
    const __m512i perm_all = _mm512_set_epi8(
        63, 59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11, 7, 3,  /* byte 3s */
        62, 58, 54, 50, 46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2,  /* byte 2s */
        61, 57, 53, 49, 45, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1,   /* byte 1s */
        60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0);  /* byte 0s */

    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(src + i * 4));

        /* Single permutation gathers all 4 streams */
        __m512i transposed = _mm512_permutexvar_epi8(perm_all, v);

        /* Extract and store each 128-bit lane to its stream */
        _mm_storeu_si128((__m128i*)(output + 0 * count + i), _mm512_castsi512_si128(transposed));
        _mm_storeu_si128((__m128i*)(output + 1 * count + i), _mm512_extracti32x4_epi32(transposed, 1));
        _mm_storeu_si128((__m128i*)(output + 2 * count + i), _mm512_extracti32x4_epi32(transposed, 2));
        _mm_storeu_si128((__m128i*)(output + 3 * count + i), _mm512_extracti32x4_epi32(transposed, 3));
    }
#else
    /* Fallback without VBMI: use shuffle + permutexvar approach
     * Step 1: shuffle_epi8 transposes within each 128-bit lane (4 floats -> 4 bytes per stream)
     * Step 2: permutexvar_epi32 rearranges dwords to group all byte 0s, byte 1s, etc.
     */
    const __m512i intra_lane_shuf = _mm512_set_epi8(
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
    const __m512i cross_lane_perm = _mm512_set_epi32(
        15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);

    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(src + i * 4));

        /* Transpose within each 128-bit lane */
        __m512i shuffled = _mm512_shuffle_epi8(v, intra_lane_shuf);

        /* Rearrange dwords across lanes to group streams */
        __m512i transposed = _mm512_permutexvar_epi32(cross_lane_perm, shuffled);

        /* Extract and store each 128-bit lane to its stream */
        _mm_storeu_si128((__m128i*)(output + 0 * count + i), _mm512_castsi512_si128(transposed));
        _mm_storeu_si128((__m128i*)(output + 1 * count + i), _mm512_extracti32x4_epi32(transposed, 1));
        _mm_storeu_si128((__m128i*)(output + 2 * count + i), _mm512_extracti32x4_epi32(transposed, 2));
        _mm_storeu_si128((__m128i*)(output + 3 * count + i), _mm512_extracti32x4_epi32(transposed, 3));
    }
#endif

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            output[b * count + i] = src[i * 4 + b];
        }
    }
}

/**
 * Decode byte stream split floats using AVX-512.
 */
void carquet_avx512_byte_stream_split_decode_float(
    const uint8_t* data,
    int64_t count,
    float* values) {

    uint8_t* dst = (uint8_t*)values;
    int64_t i = 0;

    /* Process 16 floats at a time */
    for (; i + 16 <= count; i += 16) {
        /* Load 16 bytes from each of the 4 streams */
        __m128i b0 = _mm_loadu_si128((const __m128i*)(data + 0 * count + i));
        __m128i b1 = _mm_loadu_si128((const __m128i*)(data + 1 * count + i));
        __m128i b2 = _mm_loadu_si128((const __m128i*)(data + 2 * count + i));
        __m128i b3 = _mm_loadu_si128((const __m128i*)(data + 3 * count + i));

        /* Interleave to reconstruct floats */
        __m128i lo01_lo = _mm_unpacklo_epi8(b0, b1);
        __m128i lo01_hi = _mm_unpackhi_epi8(b0, b1);
        __m128i lo23_lo = _mm_unpacklo_epi8(b2, b3);
        __m128i lo23_hi = _mm_unpackhi_epi8(b2, b3);

        __m128i result0 = _mm_unpacklo_epi16(lo01_lo, lo23_lo);
        __m128i result1 = _mm_unpackhi_epi16(lo01_lo, lo23_lo);
        __m128i result2 = _mm_unpacklo_epi16(lo01_hi, lo23_hi);
        __m128i result3 = _mm_unpackhi_epi16(lo01_hi, lo23_hi);

        _mm_storeu_si128((__m128i*)(dst + i * 4 + 0), result0);
        _mm_storeu_si128((__m128i*)(dst + i * 4 + 16), result1);
        _mm_storeu_si128((__m128i*)(dst + i * 4 + 32), result2);
        _mm_storeu_si128((__m128i*)(dst + i * 4 + 48), result3);
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            dst[i * 4 + b] = data[b * count + i];
        }
    }
}

/* ============================================================================
 * Delta Decoding - AVX-512 Optimized (Prefix Sum)
 * ============================================================================
 */

/**
 * Apply prefix sum (cumulative sum) to int32 array using AVX-512.
 */
void carquet_avx512_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial) {
    int32_t sum = initial;
    int64_t i = 0;

    /* AVX-512 prefix sum for 16 elements at a time */
    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(values + i));

        /* Multi-step prefix sum within vector */
        /* Step 1: Add adjacent pairs */
        __m512i shifted1 = _mm512_maskz_alignr_epi32(0xFFFE, v, _mm512_setzero_si512(), 15);
        v = _mm512_add_epi32(v, shifted1);

        /* Step 2: Add elements 2 apart */
        __m512i shifted2 = _mm512_maskz_alignr_epi32(0xFFFC, v, _mm512_setzero_si512(), 14);
        v = _mm512_add_epi32(v, shifted2);

        /* Step 3: Add elements 4 apart */
        __m512i shifted4 = _mm512_maskz_alignr_epi32(0xFFF0, v, _mm512_setzero_si512(), 12);
        v = _mm512_add_epi32(v, shifted4);

        /* Step 4: Add elements 8 apart */
        __m512i shifted8 = _mm512_maskz_alignr_epi32(0xFF00, v, _mm512_setzero_si512(), 8);
        v = _mm512_add_epi32(v, shifted8);

        /* Add running sum */
        __m512i sums = _mm512_set1_epi32(sum);
        v = _mm512_add_epi32(v, sums);
        _mm512_storeu_si512((__m512i*)(values + i), v);

        /* Update running sum to last element */
        sum = values[i + 15];
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

/**
 * Apply prefix sum to int64 array using AVX-512.
 */
void carquet_avx512_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial) {
    int64_t sum = initial;
    int64_t i = 0;

    /* AVX-512 prefix sum for 8 elements at a time */
    for (; i + 8 <= count; i += 8) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(values + i));

        /* Multi-step prefix sum */
        __m512i shifted1 = _mm512_maskz_alignr_epi64(0xFE, v, _mm512_setzero_si512(), 7);
        v = _mm512_add_epi64(v, shifted1);

        __m512i shifted2 = _mm512_maskz_alignr_epi64(0xFC, v, _mm512_setzero_si512(), 6);
        v = _mm512_add_epi64(v, shifted2);

        __m512i shifted4 = _mm512_maskz_alignr_epi64(0xF0, v, _mm512_setzero_si512(), 4);
        v = _mm512_add_epi64(v, shifted4);

        /* Add running sum */
        __m512i sums = _mm512_set1_epi64(sum);
        v = _mm512_add_epi64(v, sums);
        _mm512_storeu_si512((__m512i*)(values + i), v);

        /* Update running sum */
        sum = values[i + 7];
    }

    /* Handle remaining values */
    for (; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

/* ============================================================================
 * Dictionary Gather - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Gather int32 values from dictionary using AVX-512 gather instructions.
 */
void carquet_avx512_gather_i32(const int32_t* dict, const uint32_t* indices,
                               int64_t count, int32_t* output) {
    int64_t i = 0;

    /* Process 16 at a time using AVX-512 gather */
    for (; i + 16 <= count; i += 16) {
        __m512i idx = _mm512_loadu_si512((const __m512i*)(indices + i));
        __m512i result = _mm512_i32gather_epi32(idx, dict, 4);
        _mm512_storeu_si512((__m512i*)(output + i), result);
    }

    /* Handle remaining with AVX2 */
    for (; i + 8 <= count; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i*)(indices + i));
        __m256i result = _mm256_i32gather_epi32(dict, idx, 4);
        _mm256_storeu_si256((__m256i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather int64 values from dictionary using AVX-512 gather instructions.
 */
void carquet_avx512_gather_i64(const int64_t* dict, const uint32_t* indices,
                               int64_t count, int64_t* output) {
    int64_t i = 0;

    /* Process 8 at a time using AVX-512 gather */
    for (; i + 8 <= count; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i*)(indices + i));
        __m512i result = _mm512_i32gather_epi64(idx, dict, 8);
        _mm512_storeu_si512((__m512i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

/**
 * Gather float values from dictionary using AVX-512 gather instructions.
 * Note: float and int32 are both 4 bytes, so we reuse gather_i32 via cast.
 */
void carquet_avx512_gather_float(const float* dict, const uint32_t* indices,
                                  int64_t count, float* output) {
    /* Data movement doesn't care about type - reuse int32 implementation */
    carquet_avx512_gather_i32((const int32_t*)dict, indices, count, (int32_t*)output);
}

/**
 * Gather double values from dictionary using AVX-512 gather instructions.
 * Note: double and int64 are both 8 bytes, so we reuse gather_i64 via cast.
 */
void carquet_avx512_gather_double(const double* dict, const uint32_t* indices,
                                   int64_t count, double* output) {
    /* Data movement doesn't care about type - reuse int64 implementation */
    carquet_avx512_gather_i64((const int64_t*)dict, indices, count, (int64_t*)output);
}

/* ============================================================================
 * Memcpy/Memset - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Fast memset for large buffers using AVX-512.
 */
void carquet_avx512_memset(void* dest, uint8_t value, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    __m512i v = _mm512_set1_epi8((char)value);

    while (n >= 256) {
        _mm512_storeu_si512((__m512i*)(d + 0), v);
        _mm512_storeu_si512((__m512i*)(d + 64), v);
        _mm512_storeu_si512((__m512i*)(d + 128), v);
        _mm512_storeu_si512((__m512i*)(d + 192), v);
        d += 256;
        n -= 256;
    }

    while (n >= 64) {
        _mm512_storeu_si512((__m512i*)d, v);
        d += 64;
        n -= 64;
    }

    /* Handle tail with AVX2/SSE */
    __m256i v256 = _mm256_set1_epi8((char)value);
    while (n >= 32) {
        _mm256_storeu_si256((__m256i*)d, v256);
        d += 32;
        n -= 32;
    }

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
 * Fast memcpy for large buffers using AVX-512.
 */
void carquet_avx512_memcpy(void* dest, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    const uint8_t* s = (const uint8_t*)src;

    while (n >= 256) {
        __m512i v0 = _mm512_loadu_si512((const __m512i*)(s + 0));
        __m512i v1 = _mm512_loadu_si512((const __m512i*)(s + 64));
        __m512i v2 = _mm512_loadu_si512((const __m512i*)(s + 128));
        __m512i v3 = _mm512_loadu_si512((const __m512i*)(s + 192));
        _mm512_storeu_si512((__m512i*)(d + 0), v0);
        _mm512_storeu_si512((__m512i*)(d + 64), v1);
        _mm512_storeu_si512((__m512i*)(d + 128), v2);
        _mm512_storeu_si512((__m512i*)(d + 192), v3);
        d += 256;
        s += 256;
        n -= 256;
    }

    while (n >= 64) {
        _mm512_storeu_si512((__m512i*)d, _mm512_loadu_si512((const __m512i*)s));
        d += 64;
        s += 64;
        n -= 64;
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
 * Boolean Operations - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Unpack boolean values from packed bits to byte array using AVX-512.
 */
void carquet_avx512_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    int64_t i = 0;

    /* Process 64 bools (8 bytes) at a time using AVX-512 mask */
    for (; i + 64 <= count; i += 64) {
        int byte_idx = (int)(i / 8);
        uint64_t packed;
        memcpy(&packed, input + byte_idx, 8);

        /* Convert to mask and create result with maskz_set1 (1 where set, 0 otherwise) */
        __m512i result = _mm512_maskz_set1_epi8((__mmask64)packed, 1);

        _mm512_storeu_si512((__m512i*)(output + i), result);
    }

    /* Handle remaining */
    for (; i < count; i++) {
        int byte_idx = (int)(i / 8);
        int bit_idx = (int)(i % 8);
        output[i] = (input[byte_idx] >> bit_idx) & 1;
    }
}

/**
 * Pack boolean values from byte array to packed bits using AVX-512.
 */
void carquet_avx512_pack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    int64_t i = 0;

    /* Process 64 bools at a time */
    for (; i + 64 <= count; i += 64) {
        __m512i bools = _mm512_loadu_si512((const __m512i*)(input + i));

        /* Use test_epi8_mask: bit is set if (a & b) != 0, i.e., if bool is non-zero */
        __mmask64 mask = _mm512_test_epi8_mask(bools, bools);

        /* Store mask as 8 bytes */
        uint64_t packed = (uint64_t)mask;
        memcpy(output + i / 8, &packed, 8);
    }

    /* Handle remaining elements with masked load */
    if (i < count) {
        int64_t remaining = count - i;
        /* Create mask for remaining elements: set bits 0..(remaining-1) */
        __mmask64 load_mask = (remaining >= 64) ? ~0ULL : ((1ULL << remaining) - 1);

        /* Masked load zeros out elements beyond the mask */
        __m512i bools = _mm512_maskz_loadu_epi8(load_mask, input + i);

        /* Test for non-zero values */
        __mmask64 result_mask = _mm512_test_epi8_mask(bools, bools);

        /* Write only the bytes we need */
        int64_t bytes_to_write = (remaining + 7) / 8;
        uint64_t packed = (uint64_t)result_mask;
        memcpy(output + i / 8, &packed, (size_t)bytes_to_write);
    }
}

/* ============================================================================
 * Run Detection - AVX-512 Optimized
 * ============================================================================
 */

/**
 * Find the length of a run of repeated int32 values.
 */
int64_t carquet_avx512_find_run_length_i32(const int32_t* values, int64_t count) {
    if (count == 0) return 0;

    int32_t first = values[0];
    __m512i target = _mm512_set1_epi32(first);
    int64_t i = 0;

    /* Check 16 at a time */
    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(values + i));
        __mmask16 cmp = _mm512_cmpeq_epi32_mask(v, target);

        if (cmp != 0xFFFF) {  /* Not all equal */
            /* Find first mismatch using trailing zeros */
            int tz = portable_ctz(~cmp);
            return i + tz;
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

/* ============================================================================
 * Conflict Detection - AVX-512 Specific
 * ============================================================================
 */

#ifdef __AVX512CD__

/**
 * Detect conflicts in indices for scatter operations.
 * Returns a mask where bit i is set if indices[i] conflicts with any earlier index.
 */
__mmask16 carquet_avx512_detect_conflicts_i32(const uint32_t* indices) {
    __m512i idx = _mm512_loadu_si512((const __m512i*)indices);
    __m512i conflicts = _mm512_conflict_epi32(idx);

    /* Non-zero conflict value means there's a conflict */
    return _mm512_cmpneq_epi32_mask(conflicts, _mm512_setzero_si512());
}

#endif /* __AVX512CD__ */

#endif /* __AVX512F__ */
#endif /* x86_64 */
