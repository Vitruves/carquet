/**
 * @file crc32_arm.c
 * @brief ARM hardware-accelerated CRC32 implementation
 *
 * Uses ARMv8 CRC32 instructions for ~10x speedup over table-based.
 */

#include <stdint.h>
#include <stddef.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>

uint32_t carquet_crc32_arm(const uint8_t* data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;

    /* Process 8 bytes at a time */
    while (length >= 8) {
        uint64_t val;
        __builtin_memcpy(&val, data, 8);
        crc = __crc32d(crc, val);
        data += 8;
        length -= 8;
    }

    /* Process 4 bytes */
    if (length >= 4) {
        uint32_t val;
        __builtin_memcpy(&val, data, 4);
        crc = __crc32w(crc, val);
        data += 4;
        length -= 4;
    }

    /* Process 2 bytes */
    if (length >= 2) {
        uint16_t val;
        __builtin_memcpy(&val, data, 2);
        crc = __crc32h(crc, val);
        data += 2;
        length -= 2;
    }

    /* Process remaining byte */
    if (length >= 1) {
        crc = __crc32b(crc, *data);
    }

    return crc ^ 0xFFFFFFFF;
}

int carquet_has_arm_crc32(void) {
    return 1;
}

#else

/* Fallback stubs when CRC32 is not available */
uint32_t carquet_crc32_arm(const uint8_t* data, size_t length) {
    (void)data;
    (void)length;
    return 0;
}

int carquet_has_arm_crc32(void) {
    return 0;
}

#endif
