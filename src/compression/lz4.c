/**
 * @file lz4.c
 * @brief Pure C LZ4 block compression/decompression
 *
 * Implements LZ4 block format (not frame format).
 * Reference: https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ============================================================================
 * Constants
 * ============================================================================
 */

#define LZ4_MIN_MATCH        4
#define LZ4_MAX_MATCH_LEN    (15 + 255 * 255)
#define LZ4_HASH_LOG         12
#define LZ4_HASH_SIZE        (1 << LZ4_HASH_LOG)
#define LZ4_SKIP_TRIGGER     6
#define LZ4_MIN_LENGTH       13
#define LZ4_LAST_LITERALS    5

/* ============================================================================
 * LZ4 Decompression
 * ============================================================================
 */

carquet_status_t carquet_lz4_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    const uint8_t* ip = src;
    const uint8_t* const iend = src + src_size;
    uint8_t* op = dst;
    uint8_t* const oend = dst + dst_capacity;

    while (ip < iend) {
        /* Read token */
        uint8_t token = *ip++;

        /* Literal length */
        size_t lit_len = token >> 4;
        if (lit_len == 15) {
            uint8_t s;
            do {
                if (ip >= iend) {
                    return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
                }
                s = *ip++;
                lit_len += s;
            } while (s == 255);
        }

        /* Copy literals */
        if (lit_len > 0) {
            if (ip + lit_len > iend || op + lit_len > oend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            memcpy(op, ip, lit_len);
            ip += lit_len;
            op += lit_len;
        }

        /* Check for end of block */
        if (ip >= iend) {
            break;
        }

        /* Read match offset (little-endian 16-bit) */
        if (ip + 2 > iend) {
            return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
        }
        size_t offset = ip[0] | ((size_t)ip[1] << 8);
        ip += 2;

        if (offset == 0 || offset > (size_t)(op - dst)) {
            return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
        }

        /* Match length */
        size_t match_len = (token & 0x0F) + LZ4_MIN_MATCH;
        if ((token & 0x0F) == 15) {
            uint8_t s;
            do {
                if (ip >= iend) {
                    return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
                }
                s = *ip++;
                match_len += s;
            } while (s == 255);
        }

        /* Copy match */
        if (op + match_len > oend) {
            return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
        }

        const uint8_t* match = op - offset;

        /* Handle overlapping copies */
        if (offset >= 8) {
            /* Fast path: no overlap, copy 8 bytes at a time */
            while (match_len >= 8) {
                memcpy(op, match, 8);
                op += 8;
                match += 8;
                match_len -= 8;
            }
            while (match_len-- > 0) {
                *op++ = *match++;
            }
        } else {
            /* Slow path: potential overlap */
            while (match_len-- > 0) {
                *op++ = *match++;
            }
        }
    }

    *dst_size = (size_t)(op - dst);
    return CARQUET_OK;
}

/* ============================================================================
 * LZ4 Compression
 * ============================================================================
 */

static inline uint32_t lz4_hash(uint32_t val) {
    return (val * 2654435761U) >> (32 - LZ4_HASH_LOG);
}

static inline uint32_t lz4_read32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline size_t lz4_count(const uint8_t* p, const uint8_t* match,
                                const uint8_t* limit) {
    const uint8_t* start = p;
    while (p < limit - 7) {
        uint64_t diff;
        uint64_t a, b;
        memcpy(&a, p, 8);
        memcpy(&b, match, 8);
        diff = a ^ b;
        if (diff) {
            /* Count trailing zeros to find first difference */
            int tz = 0;
            while ((diff & 0xFF) == 0) {
                tz++;
                diff >>= 8;
            }
            return (size_t)(p - start) + tz;
        }
        p += 8;
        match += 8;
    }
    while (p < limit && *p == *match) {
        p++;
        match++;
    }
    return (size_t)(p - start);
}

carquet_status_t carquet_lz4_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Handle empty input */
    if (src_size == 0) {
        if (dst_capacity < 1) {
            return CARQUET_ERROR_COMPRESSION;
        }
        *dst = 0;  /* Token with 0 literals, no match */
        *dst_size = 1;
        return CARQUET_OK;
    }

    if (!src) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* For very small inputs, just store as literals */
    if (src_size < LZ4_MIN_LENGTH) {
        if (dst_capacity < src_size + 1) {
            return CARQUET_ERROR_COMPRESSION;
        }
        /* Single literal run */
        if (src_size < 15) {
            *dst = (uint8_t)(src_size << 4);
            memcpy(dst + 1, src, src_size);
            *dst_size = src_size + 1;
        } else {
            *dst = 0xF0;
            dst[1] = (uint8_t)(src_size - 15);
            memcpy(dst + 2, src, src_size);
            *dst_size = src_size + 2;
        }
        return CARQUET_OK;
    }

    uint16_t hash_table[LZ4_HASH_SIZE];
    memset(hash_table, 0, sizeof(hash_table));

    const uint8_t* ip = src;
    const uint8_t* const iend = src + src_size;
    const uint8_t* const mflimit = iend - LZ4_MIN_MATCH;
    const uint8_t* const matchlimit = iend - LZ4_LAST_LITERALS;
    const uint8_t* anchor = src;

    uint8_t* op = dst;
    uint8_t* const oend = dst + dst_capacity;

    /* Main loop */
    while (ip < mflimit) {
        /* Find match */
        uint32_t h = lz4_hash(lz4_read32(ip));
        const uint8_t* ref = src + hash_table[h];
        hash_table[h] = (uint16_t)(ip - src);

        /* Check match validity - offset must be > 0 and <= 65535 */
        if (ip <= ref || ip - ref > 65535 || lz4_read32(ref) != lz4_read32(ip)) {
            ip++;
            continue;
        }

        /* Found a match - encode literals first */
        size_t lit_len = (size_t)(ip - anchor);
        size_t match_len = lz4_count(ip + LZ4_MIN_MATCH, ref + LZ4_MIN_MATCH,
                                      matchlimit) + LZ4_MIN_MATCH;

        /* Check output space */
        size_t max_out = 1 + (lit_len / 255) + lit_len + 2 + (match_len / 255);
        if (op + max_out > oend) {
            return CARQUET_ERROR_COMPRESSION;
        }

        /* Write token */
        uint8_t* token = op++;
        *token = 0;

        /* Encode literal length */
        if (lit_len >= 15) {
            *token = 0xF0;
            size_t rem = lit_len - 15;
            while (rem >= 255) {
                *op++ = 255;
                rem -= 255;
            }
            *op++ = (uint8_t)rem;
        } else {
            *token = (uint8_t)(lit_len << 4);
        }

        /* Copy literals */
        memcpy(op, anchor, lit_len);
        op += lit_len;

        /* Write offset */
        size_t offset = (size_t)(ip - ref);
        *op++ = (uint8_t)(offset & 0xFF);
        *op++ = (uint8_t)(offset >> 8);

        /* Encode match length */
        size_t ml = match_len - LZ4_MIN_MATCH;
        if (ml >= 15) {
            *token |= 0x0F;
            ml -= 15;
            while (ml >= 255) {
                *op++ = 255;
                ml -= 255;
            }
            *op++ = (uint8_t)ml;
        } else {
            *token |= (uint8_t)ml;
        }

        /* Update positions */
        ip += match_len;
        anchor = ip;

        /* Update hash for positions within match */
        if (ip < mflimit) {
            hash_table[lz4_hash(lz4_read32(ip - 2))] = (uint16_t)(ip - 2 - src);
        }
    }

    /* Write last literals */
    size_t last_run = (size_t)(iend - anchor);
    if (op + 1 + (last_run / 255) + last_run > oend) {
        return CARQUET_ERROR_COMPRESSION;
    }

    if (last_run >= 15) {
        *op++ = 0xF0;
        size_t rem = last_run - 15;
        while (rem >= 255) {
            *op++ = 255;
            rem -= 255;
        }
        *op++ = (uint8_t)rem;
    } else {
        *op++ = (uint8_t)(last_run << 4);
    }
    memcpy(op, anchor, last_run);
    op += last_run;

    *dst_size = (size_t)(op - dst);
    return CARQUET_OK;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================
 */

size_t carquet_lz4_compress_bound(size_t src_size) {
    return src_size + (src_size / 255) + 16;
}
