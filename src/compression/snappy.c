/**
 * @file snappy.c
 * @brief Pure C Snappy compression/decompression
 *
 * Reference: https://github.com/google/snappy/blob/main/format_description.txt
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ============================================================================
 * Constants
 * ============================================================================
 */

#define SNAPPY_LITERAL         0
#define SNAPPY_COPY_1          1
#define SNAPPY_COPY_2          2
#define SNAPPY_COPY_4          3

#define SNAPPY_HASH_LOG        14
#define SNAPPY_HASH_SIZE       (1 << SNAPPY_HASH_LOG)
#define SNAPPY_MAX_OFFSET      (1 << 15)
#define SNAPPY_BLOCK_SIZE      (1 << 16)

/* Forward declaration */
size_t carquet_snappy_compress_bound(size_t src_size);

/* ============================================================================
 * Varint Encoding
 * ============================================================================
 */

static size_t snappy_read_varint(const uint8_t* p, const uint8_t* end, uint32_t* value) {
    *value = 0;
    int shift = 0;
    const uint8_t* start = p;

    while (p < end) {
        uint8_t b = *p++;
        *value |= ((uint32_t)(b & 0x7F)) << shift;
        if ((b & 0x80) == 0) {
            return (size_t)(p - start);
        }
        shift += 7;
        if (shift >= 32) {
            return 0; /* Overflow */
        }
    }
    return 0; /* Truncated */
}

static size_t snappy_write_varint(uint8_t* p, uint32_t value) {
    uint8_t* start = p;
    while (value >= 0x80) {
        *p++ = (uint8_t)(value | 0x80);
        value >>= 7;
    }
    *p++ = (uint8_t)value;
    return (size_t)(p - start);
}

/* ============================================================================
 * Snappy Decompression
 * ============================================================================
 */

carquet_status_t carquet_snappy_decompress(
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

    /* Read uncompressed length */
    uint32_t uncompressed_len;
    size_t varint_len = snappy_read_varint(ip, iend, &uncompressed_len);
    if (varint_len == 0) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }
    ip += varint_len;

    if (uncompressed_len > dst_capacity) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    uint8_t* op = dst;
    uint8_t* const oend = dst + uncompressed_len;

    while (ip < iend && op < oend) {
        uint8_t tag = *ip++;
        uint8_t type = tag & 0x03;

        if (type == SNAPPY_LITERAL) {
            /* Literal */
            size_t len = (tag >> 2) + 1;

            if (len > 60) {
                /* Extended length */
                size_t extra_bytes = len - 60;
                if (ip + extra_bytes > iend) {
                    return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
                }
                len = 1;
                for (size_t i = 0; i < extra_bytes; i++) {
                    len += (size_t)ip[i] << (8 * i);
                }
                ip += extra_bytes;
            }

            if (ip + len > iend || op + len > oend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            memcpy(op, ip, len);
            ip += len;
            op += len;

        } else if (type == SNAPPY_COPY_1) {
            /* Copy with 1-byte offset */
            size_t len = ((tag >> 2) & 0x07) + 4;
            size_t offset = ((tag >> 5) << 8) | *ip++;

            if (offset == 0 || offset > (size_t)(op - dst)) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            if (op + len > oend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            const uint8_t* ref = op - offset;
            while (len-- > 0) {
                *op++ = *ref++;
            }

        } else if (type == SNAPPY_COPY_2) {
            /* Copy with 2-byte offset */
            size_t len = ((tag >> 2) & 0x3F) + 1;
            if (ip + 2 > iend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            size_t offset = ip[0] | ((size_t)ip[1] << 8);
            ip += 2;

            if (offset == 0 || offset > (size_t)(op - dst)) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            if (op + len > oend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            const uint8_t* ref = op - offset;
            if (offset >= len) {
                memcpy(op, ref, len);
                op += len;
            } else {
                while (len-- > 0) {
                    *op++ = *ref++;
                }
            }

        } else { /* SNAPPY_COPY_4 */
            /* Copy with 4-byte offset */
            size_t len = ((tag >> 2) & 0x3F) + 1;
            if (ip + 4 > iend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            size_t offset = ip[0] | ((size_t)ip[1] << 8) |
                           ((size_t)ip[2] << 16) | ((size_t)ip[3] << 24);
            ip += 4;

            if (offset == 0 || offset > (size_t)(op - dst)) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            if (op + len > oend) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            const uint8_t* ref = op - offset;
            if (offset >= len) {
                memcpy(op, ref, len);
                op += len;
            } else {
                while (len-- > 0) {
                    *op++ = *ref++;
                }
            }
        }
    }

    if ((size_t)(op - dst) != uncompressed_len) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    *dst_size = uncompressed_len;
    return CARQUET_OK;
}

/* ============================================================================
 * Snappy Compression
 * ============================================================================
 */

static inline uint32_t snappy_hash(uint32_t val) {
    return (val * 0x1e35a7bd) >> (32 - SNAPPY_HASH_LOG);
}

static inline uint32_t snappy_read32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static uint8_t* snappy_emit_literal(uint8_t* op, const uint8_t* literal, size_t len) {
    if (len <= 60) {
        *op++ = (uint8_t)((len - 1) << 2);
    } else if (len <= 256) {
        *op++ = (60 << 2);
        *op++ = (uint8_t)(len - 1);
    } else if (len <= 65536) {
        *op++ = (61 << 2);
        *op++ = (uint8_t)((len - 1) & 0xFF);
        *op++ = (uint8_t)((len - 1) >> 8);
    } else if (len <= 16777216) {
        *op++ = (62 << 2);
        *op++ = (uint8_t)((len - 1) & 0xFF);
        *op++ = (uint8_t)(((len - 1) >> 8) & 0xFF);
        *op++ = (uint8_t)(((len - 1) >> 16) & 0xFF);
    } else {
        *op++ = (63 << 2);
        *op++ = (uint8_t)((len - 1) & 0xFF);
        *op++ = (uint8_t)(((len - 1) >> 8) & 0xFF);
        *op++ = (uint8_t)(((len - 1) >> 16) & 0xFF);
        *op++ = (uint8_t)(((len - 1) >> 24) & 0xFF);
    }
    memcpy(op, literal, len);
    return op + len;
}

static uint8_t* snappy_emit_copy(uint8_t* op, size_t offset, size_t len) {
    while (len >= 68) {
        /* Emit copy of 64 bytes */
        *op++ = (uint8_t)((63 << 2) | SNAPPY_COPY_2);
        *op++ = (uint8_t)(offset & 0xFF);
        *op++ = (uint8_t)(offset >> 8);
        len -= 64;
    }

    if (len > 64) {
        /* Emit copy of 60 bytes */
        *op++ = (uint8_t)((59 << 2) | SNAPPY_COPY_2);
        *op++ = (uint8_t)(offset & 0xFF);
        *op++ = (uint8_t)(offset >> 8);
        len -= 60;
    }

    if (len >= 12 || offset >= 2048) {
        /* Use 2-byte offset copy */
        *op++ = (uint8_t)(((len - 1) << 2) | SNAPPY_COPY_2);
        *op++ = (uint8_t)(offset & 0xFF);
        *op++ = (uint8_t)(offset >> 8);
    } else {
        /* Use 1-byte offset copy (len 4-11, offset < 2048) */
        *op++ = (uint8_t)(((offset >> 8) << 5) | ((len - 4) << 2) | SNAPPY_COPY_1);
        *op++ = (uint8_t)(offset & 0xFF);
    }

    return op;
}

carquet_status_t carquet_snappy_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Check if output buffer is large enough for worst case */
    size_t max_output = carquet_snappy_compress_bound(src_size);
    if (dst_capacity < max_output) {
        return CARQUET_ERROR_COMPRESSION;
    }

    uint8_t* op = dst;

    /* Write uncompressed length */
    op += snappy_write_varint(op, (uint32_t)src_size);

    if (src_size == 0) {
        *dst_size = (size_t)(op - dst);
        return CARQUET_OK;
    }

    if (!src) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (src_size < 15) {
        /* Just emit a literal */
        op = snappy_emit_literal(op, src, src_size);
        *dst_size = (size_t)(op - dst);
        return CARQUET_OK;
    }

    uint16_t hash_table[SNAPPY_HASH_SIZE];
    memset(hash_table, 0, sizeof(hash_table));

    const uint8_t* ip = src;
    const uint8_t* const iend = src + src_size;
    const uint8_t* const ilimit = iend - 15;
    const uint8_t* anchor = src;

    while (ip < ilimit) {
        /* Find match */
        uint32_t h = snappy_hash(snappy_read32(ip));
        const uint8_t* ref = src + hash_table[h];
        hash_table[h] = (uint16_t)(ip - src);

        /* Check match validity - offset must be > 0 and <= SNAPPY_MAX_OFFSET */
        if (ip <= ref || ip - ref > SNAPPY_MAX_OFFSET || snappy_read32(ref) != snappy_read32(ip)) {
            ip++;
            continue;
        }

        /* Emit pending literal */
        if (ip > anchor) {
            op = snappy_emit_literal(op, anchor, (size_t)(ip - anchor));
        }

        /* Find match length */
        const uint8_t* match_start = ip;
        ip += 4;
        ref += 4;
        while (ip < iend && *ip == *ref) {
            ip++;
            ref++;
        }

        /* Emit copy */
        size_t offset = (size_t)(match_start - (ref - (ip - match_start)));
        size_t match_len = (size_t)(ip - match_start);
        op = snappy_emit_copy(op, offset, match_len);

        anchor = ip;

        /* Update hash */
        if (ip < ilimit) {
            hash_table[snappy_hash(snappy_read32(ip - 1))] = (uint16_t)(ip - 1 - src);
        }
    }

    /* Emit final literal */
    if (anchor < iend) {
        op = snappy_emit_literal(op, anchor, (size_t)(iend - anchor));
    }

    *dst_size = (size_t)(op - dst);
    return CARQUET_OK;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================
 */

size_t carquet_snappy_compress_bound(size_t src_size) {
    return 32 + src_size + src_size / 6;
}

carquet_status_t carquet_snappy_get_uncompressed_length(
    const uint8_t* src,
    size_t src_size,
    size_t* length) {

    if (!src || !length) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    uint32_t len;
    size_t varint_len = snappy_read_varint(src, src + src_size, &len);
    if (varint_len == 0) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    *length = len;
    return CARQUET_OK;
}
