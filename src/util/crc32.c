/**
 * @file crc32.c
 * @brief CRC32 checksum implementation with hardware acceleration
 *
 * Uses slicing-by-8 algorithm on x86 for ~5-8x speedup over byte-at-a-time.
 * Uses hardware CRC32 instructions on ARM when available.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ARM hardware CRC32 (when available) */
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
extern uint32_t carquet_crc32_arm(const uint8_t* data, size_t length);
extern int carquet_has_arm_crc32(void);

/* Runtime check for ARM CRC32 support */
static int use_arm_crc32 = -1;  /* -1 = not checked yet */

static int check_arm_crc32(void) {
    if (use_arm_crc32 < 0) {
        use_arm_crc32 = carquet_has_arm_crc32();
    }
    return use_arm_crc32;
}
#endif

/* ============================================================================
 * Slicing-by-8 CRC32 Implementation (IEEE polynomial 0xEDB88320)
 *
 * Processes 8 bytes per iteration using 8 precomputed lookup tables,
 * giving ~5-8x speedup over the naive byte-at-a-time approach.
 * ============================================================================
 */

#define CRC32_POLY 0xEDB88320u

/* 8 tables of 256 entries for slicing-by-8 */
static uint32_t crc32_tables[8][256];
static volatile int crc32_tables_initialized = 0;

static void crc32_init_tables(void) {
    if (crc32_tables_initialized) return;

    /* Generate base table (standard reflected CRC32) */
    for (int i = 0; i < 256; i++) {
        uint32_t crc = (uint32_t)i;
        for (int j = 0; j < 8; j++) {
            crc = (crc & 1) ? ((crc >> 1) ^ CRC32_POLY) : (crc >> 1);
        }
        crc32_tables[0][i] = crc;
    }

    /* Generate extended tables for slicing-by-8.
     * table[k][i] represents the CRC contribution of byte i
     * when it's k positions ahead in the 8-byte window. */
    for (int k = 1; k < 8; k++) {
        for (int i = 0; i < 256; i++) {
            crc32_tables[k][i] = (crc32_tables[k - 1][i] >> 8) ^
                                  crc32_tables[0][crc32_tables[k - 1][i] & 0xFF];
        }
    }

    crc32_tables_initialized = 1;
}

static uint32_t crc32_slicing_by_8(uint32_t crc, const uint8_t* data, size_t length) {
    if (!crc32_tables_initialized) crc32_init_tables();

    crc = ~crc;

    /* Process 8 bytes at a time */
    while (length >= 8) {
        uint32_t one, two;
        memcpy(&one, data, 4);
        memcpy(&two, data + 4, 4);
        one ^= crc;

        crc = crc32_tables[7][ one        & 0xFF] ^
              crc32_tables[6][(one >>  8) & 0xFF] ^
              crc32_tables[5][(one >> 16) & 0xFF] ^
              crc32_tables[4][(one >> 24)       ] ^
              crc32_tables[3][ two        & 0xFF] ^
              crc32_tables[2][(two >>  8) & 0xFF] ^
              crc32_tables[1][(two >> 16) & 0xFF] ^
              crc32_tables[0][(two >> 24)       ];

        data += 8;
        length -= 8;
    }

    /* Process remaining bytes one at a time */
    while (length--) {
        crc = crc32_tables[0][(crc ^ *data++) & 0xFF] ^ (crc >> 8);
    }

    return ~crc;
}

uint32_t carquet_crc32(const uint8_t* data, size_t length) {
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
    /* Use hardware CRC32 if available (ARM) */
    if (check_arm_crc32()) {
        return carquet_crc32_arm(data, length);
    }
#endif
    return crc32_slicing_by_8(0, data, length);
}

uint32_t carquet_crc32_update(uint32_t crc, const uint8_t* data, size_t length) {
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
    /* ARM hardware path doesn't support incremental updates in the current
     * implementation, fall through to software slicing-by-8 */
#endif
    return crc32_slicing_by_8(crc, data, length);
}
