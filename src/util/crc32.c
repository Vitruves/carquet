/**
 * @file crc32.c
 * @brief CRC32 checksum for Parquet page checksums
 *
 * Parquet page CRCs use the standard IEEE CRC32 polynomial (0x04C11DB7,
 * reflected 0xEDB88320), the one zlib and gzip use, computed over the
 * serialized page body and excluding the page header.
 *
 * Upstream zlib computes it in software (a few GB/s). Where the CPU has the
 * instructions, it is computed here instead:
 *   - x86-64 with PCLMULQDQ + SSE4.1: carry-less multiplication folding
 *     (Intel, "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ
 *     Instruction"; the same constants as the Linux kernel and Chromium's
 *     zlib), checked at run time;
 *   - ARMv8 Linux with the CRC32 extension (checked at run time): the
 *     crc32b/x instructions compute this polynomial (the crc32c* ones
 *     compute Castagnoli), about 7.5 GB/s against 1-2 for software zlib.
 *     Apple's zlib already folds with PMULL (30 GB/s on an M3), so macOS
 *     keeps it.
 * Every other case, and the tails the fold leaves, go to zlib. The result is
 * bit-identical to zlib's crc32() (tests/test_utils.c compares them).
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <zlib.h>

static uint32_t crc32_zlib(uint32_t crc, const uint8_t* data, size_t length) {
    /* zlib's crc32 takes uInt (32-bit) lengths; chunk if needed. */
    uLong c = (uLong)crc;
    while (length > 0) {
        uInt chunk = length > (size_t)0xFFFFFFF0u ? (uInt)0xFFFFFFF0u : (uInt)length;
        c = crc32(c, (const Bytef*)data, chunk);
        data += chunk;
        length -= chunk;
    }
    return (uint32_t)c;
}

/* ============================================================================
 * x86-64: PCLMULQDQ folding
 * ============================================================================ */

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER))
#define CARQUET_CRC32_PCLMUL 1
#include <immintrin.h>
#include <wmmintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define CRC32_TARGET
#else
#include <cpuid.h>
#define CRC32_TARGET __attribute__((target("pclmul,sse4.1")))
#endif

static int cpu_has_pclmul_sse41(void) {
#if defined(_MSC_VER) && !defined(__clang__)
    int info[4];
    __cpuid(info, 1);
    unsigned ecx = (unsigned)info[2];
#else
    unsigned eax, ebx, ecx = 0, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return 0;
#endif
    return ((ecx >> 1) & 1) && ((ecx >> 19) & 1);   /* PCLMULQDQ, SSE4.1 */
}

/* CRC of `len` bytes (len >= 64, a multiple of 16), on the inverted register
 * as zlib keeps it internally (pass ~crc, get ~crc back). */
CRC32_TARGET
static uint32_t crc32_fold_pclmul(const uint8_t* buf, size_t len, uint32_t crc) {
    const __m128i k1k2 = _mm_set_epi64x(0x01c6e41596LL, 0x0154442bd4LL);
    const __m128i k3k4 = _mm_set_epi64x(0x00ccaa009eLL, 0x01751997d0LL);
    const __m128i k5k0 = _mm_set_epi64x(0, 0x0163cd6124LL);
    const __m128i poly = _mm_set_epi64x(0x01f7011641LL, 0x01db710641LL);
    __m128i x0, x1, x2, x3, x4, x5, x6, x7, x8, y5, y6, y7, y8;

    x1 = _mm_loadu_si128((const __m128i*)(buf + 0x00));
    x2 = _mm_loadu_si128((const __m128i*)(buf + 0x10));
    x3 = _mm_loadu_si128((const __m128i*)(buf + 0x20));
    x4 = _mm_loadu_si128((const __m128i*)(buf + 0x30));
    x1 = _mm_xor_si128(x1, _mm_cvtsi32_si128((int)crc));
    x0 = k1k2;
    buf += 64;
    len -= 64;

    /* Four 128-bit lanes folded 64 bytes at a time. */
    while (len >= 64) {
        x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
        x6 = _mm_clmulepi64_si128(x2, x0, 0x00);
        x7 = _mm_clmulepi64_si128(x3, x0, 0x00);
        x8 = _mm_clmulepi64_si128(x4, x0, 0x00);
        x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
        x2 = _mm_clmulepi64_si128(x2, x0, 0x11);
        x3 = _mm_clmulepi64_si128(x3, x0, 0x11);
        x4 = _mm_clmulepi64_si128(x4, x0, 0x11);
        y5 = _mm_loadu_si128((const __m128i*)(buf + 0x00));
        y6 = _mm_loadu_si128((const __m128i*)(buf + 0x10));
        y7 = _mm_loadu_si128((const __m128i*)(buf + 0x20));
        y8 = _mm_loadu_si128((const __m128i*)(buf + 0x30));
        x1 = _mm_xor_si128(_mm_xor_si128(x1, x5), y5);
        x2 = _mm_xor_si128(_mm_xor_si128(x2, x6), y6);
        x3 = _mm_xor_si128(_mm_xor_si128(x3, x7), y7);
        x4 = _mm_xor_si128(_mm_xor_si128(x4, x8), y8);
        buf += 64;
        len -= 64;
    }

    /* Fold the four lanes into one. */
    x0 = k3k4;
    x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
    x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
    x1 = _mm_xor_si128(_mm_xor_si128(x1, x2), x5);
    x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
    x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
    x1 = _mm_xor_si128(_mm_xor_si128(x1, x3), x5);
    x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
    x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
    x1 = _mm_xor_si128(_mm_xor_si128(x1, x4), x5);

    /* Remaining 16-byte blocks. */
    while (len >= 16) {
        x2 = _mm_loadu_si128((const __m128i*)buf);
        x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
        x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
        x1 = _mm_xor_si128(_mm_xor_si128(x1, x2), x5);
        buf += 16;
        len -= 16;
    }

    /* 128 -> 64 bits. */
    x2 = _mm_clmulepi64_si128(x1, x0, 0x10);
    x3 = _mm_setr_epi32(~0, 0, ~0, 0);
    x1 = _mm_srli_si128(x1, 8);
    x1 = _mm_xor_si128(x1, x2);
    x0 = k5k0;
    x2 = _mm_srli_si128(x1, 4);
    x1 = _mm_and_si128(x1, x3);
    x1 = _mm_clmulepi64_si128(x1, x0, 0x00);
    x1 = _mm_xor_si128(x1, x2);

    /* Barrett reduction to 32 bits. */
    x0 = poly;
    x2 = _mm_and_si128(x1, x3);
    x2 = _mm_clmulepi64_si128(x2, x0, 0x10);
    x2 = _mm_and_si128(x2, x3);
    x2 = _mm_clmulepi64_si128(x2, x0, 0x00);
    x1 = _mm_xor_si128(x1, x2);
    return (uint32_t)_mm_extract_epi32(x1, 1);
}
#endif

/* ============================================================================
 * ARMv8: CRC32 instructions
 * ============================================================================ */

#if defined(__aarch64__) && defined(__linux__) && \
    (defined(__GNUC__) || defined(__clang__))
#define CARQUET_CRC32_ARMV8 1
#include <arm_acle.h>
#include <sys/auxv.h>
#ifndef HWCAP_CRC32
#define HWCAP_CRC32 (1 << 7)
#endif

static int cpu_has_armv8_crc(void) {
    return (getauxval(AT_HWCAP) & HWCAP_CRC32) != 0;
}

__attribute__((target("+crc")))
static uint32_t crc32_armv8(uint32_t crc, const uint8_t* p, size_t n) {
    crc = ~crc;
    while (n > 0 && ((uintptr_t)p & 7)) {
        crc = __crc32b(crc, *p++);
        n--;
    }
    while (n >= 32) {
        uint64_t a, b, c, d;
        memcpy(&a, p, 8); memcpy(&b, p + 8, 8);
        memcpy(&c, p + 16, 8); memcpy(&d, p + 24, 8);
        crc = __crc32d(crc, a);
        crc = __crc32d(crc, b);
        crc = __crc32d(crc, c);
        crc = __crc32d(crc, d);
        p += 32;
        n -= 32;
    }
    while (n >= 8) {
        uint64_t a;
        memcpy(&a, p, 8);
        crc = __crc32d(crc, a);
        p += 8;
        n -= 8;
    }
    while (n > 0) {
        crc = __crc32b(crc, *p++);
        n--;
    }
    return ~crc;
}
#endif

/* ============================================================================
 * Dispatch
 * ============================================================================ */

/* 0 = not checked yet, 1 = zlib, 2 = PCLMUL, 3 = ARMv8. Threads checking at
 * the same time store the same value. */
static volatile int g_crc32_impl;

static int crc32_impl(void) {
    int impl = g_crc32_impl;
    if (impl == 0) {
        impl = 1;
#if defined(CARQUET_CRC32_PCLMUL)
        if (cpu_has_pclmul_sse41()) impl = 2;
#elif defined(CARQUET_CRC32_ARMV8)
        if (cpu_has_armv8_crc()) impl = 3;
#endif
        g_crc32_impl = impl;
    }
    return impl;
}

uint32_t carquet_crc32_update(uint32_t crc, const uint8_t* data, size_t length) {
    int impl = crc32_impl();
#if defined(CARQUET_CRC32_PCLMUL)
    if (impl == 2 && length >= 64) {
        size_t chunk = length & ~(size_t)15;
        crc = ~crc32_fold_pclmul(data, chunk, ~crc);
        data += chunk;
        length -= chunk;
    }
#endif
#if defined(CARQUET_CRC32_ARMV8)
    if (impl == 3) {
        return crc32_armv8(crc, data, length);
    }
#endif
    (void)impl;
    return length ? crc32_zlib(crc, data, length) : crc;
}

uint32_t carquet_crc32(const uint8_t* data, size_t length) {
    return carquet_crc32_update(0, data, length);
}
