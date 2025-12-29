/**
 * @file gzip.c
 * @brief Pure C DEFLATE compression/decompression for Parquet
 *
 * Implements RFC 1951 DEFLATE algorithm.
 * Parquet uses raw DEFLATE (no gzip/zlib headers).
 *
 * Reference: https://tools.ietf.org/html/rfc1951
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

/* ============================================================================
 * Constants
 * ============================================================================
 */

#define DEFLATE_MAX_BITS          15
#define DEFLATE_MAX_LITLEN_CODES  288
#define DEFLATE_MAX_DIST_CODES    32
#define DEFLATE_MAX_CODELEN_CODES 19
#define DEFLATE_WINDOW_SIZE       32768
#define DEFLATE_MAX_MATCH         258
#define DEFLATE_MIN_MATCH         3
#define DEFLATE_HASH_BITS         15
#define DEFLATE_HASH_SIZE         (1 << DEFLATE_HASH_BITS)
#define DEFLATE_HASH_MASK         (DEFLATE_HASH_SIZE - 1)

/* Block types */
#define DEFLATE_BLOCK_STORED      0
#define DEFLATE_BLOCK_FIXED       1
#define DEFLATE_BLOCK_DYNAMIC     2

/* End of block code */
#define DEFLATE_END_BLOCK         256

/* ============================================================================
 * Static Tables
 * ============================================================================
 */

/* Order of code length codes */
static const uint8_t codelen_order[DEFLATE_MAX_CODELEN_CODES] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
};

/* Base lengths for length codes 257-285 */
static const uint16_t length_base[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
};

/* Extra bits for length codes */
static const uint8_t length_extra[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
};

/* Base distances for distance codes 0-29 */
static const uint16_t dist_base[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577
};

/* Extra bits for distance codes */
static const uint8_t dist_extra[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
};

/* ============================================================================
 * Huffman Decoding Structure
 * ============================================================================
 */

typedef struct {
    uint16_t counts[DEFLATE_MAX_BITS + 1];  /* Count of codes at each length */
    uint16_t symbols[DEFLATE_MAX_LITLEN_CODES]; /* Symbols sorted by code */
} huffman_table_t;

/* ============================================================================
 * Bit Reader for Decompression
 * ============================================================================
 */

typedef struct {
    const uint8_t* data;
    size_t size;
    size_t pos;
    uint32_t bits;      /* Bit buffer */
    int num_bits;       /* Number of bits in buffer */
} bit_reader_t;

static void bitreader_init(bit_reader_t* br, const uint8_t* data, size_t size) {
    br->data = data;
    br->size = size;
    br->pos = 0;
    br->bits = 0;
    br->num_bits = 0;
}

static inline void bitreader_refill(bit_reader_t* br) {
    while (br->num_bits <= 24 && br->pos < br->size) {
        br->bits |= (uint32_t)br->data[br->pos++] << br->num_bits;
        br->num_bits += 8;
    }
}

static inline uint32_t bitreader_peek(bit_reader_t* br, int n) {
    return br->bits & ((1U << n) - 1);
}

static inline void bitreader_consume(bit_reader_t* br, int n) {
    br->bits >>= n;
    br->num_bits -= n;
}

static inline uint32_t bitreader_read(bit_reader_t* br, int n) {
    bitreader_refill(br);
    uint32_t val = bitreader_peek(br, n);
    bitreader_consume(br, n);
    return val;
}

static inline bool bitreader_has_bits(bit_reader_t* br, int n) {
    return br->num_bits >= n || br->pos < br->size;
}

/* ============================================================================
 * Huffman Table Building
 * ============================================================================
 */

static int build_huffman_table(huffman_table_t* table,
                               const uint8_t* lengths,
                               int num_codes) {
    uint16_t offsets[DEFLATE_MAX_BITS + 1];
    int i;

    /* Clear counts */
    memset(table->counts, 0, sizeof(table->counts));

    /* Count code lengths */
    for (i = 0; i < num_codes; i++) {
        if (lengths[i] > DEFLATE_MAX_BITS) {
            return -1;  /* Invalid code length */
        }
        table->counts[lengths[i]]++;
    }
    table->counts[0] = 0;  /* Ignore zero-length codes */

    /* Build offsets for sorting */
    offsets[0] = 0;
    offsets[1] = 0;
    for (i = 1; i < DEFLATE_MAX_BITS; i++) {
        offsets[i + 1] = offsets[i] + table->counts[i];
    }

    /* Sort symbols by code length, then by symbol value */
    for (i = 0; i < num_codes; i++) {
        if (lengths[i] > 0) {
            table->symbols[offsets[lengths[i]]++] = (uint16_t)i;
        }
    }

    return 0;
}

/* ============================================================================
 * Huffman Decoding
 * ============================================================================
 */

static int decode_symbol(bit_reader_t* br, const huffman_table_t* table) {
    bitreader_refill(br);

    uint32_t bits = br->bits;
    int code = 0;
    int first = 0;
    int index = 0;

    for (int len = 1; len <= DEFLATE_MAX_BITS; len++) {
        code |= bits & 1;
        bits >>= 1;

        int count = table->counts[len];
        if (code - count < first) {
            bitreader_consume(br, len);
            return table->symbols[index + (code - first)];
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }

    return -1;  /* Invalid code */
}

/* ============================================================================
 * Fixed Huffman Tables
 * ============================================================================
 */

static int g_fixed_tables_built = 0;
static huffman_table_t g_fixed_litlen;
static huffman_table_t g_fixed_dist;

static void build_fixed_tables(void) {
    if (g_fixed_tables_built) {
        return;
    }

    uint8_t lengths[DEFLATE_MAX_LITLEN_CODES];
    int i;

    /* Build fixed literal/length table */
    for (i = 0; i < 144; i++) lengths[i] = 8;
    for (i = 144; i < 256; i++) lengths[i] = 9;
    for (i = 256; i < 280; i++) lengths[i] = 7;
    for (i = 280; i < 288; i++) lengths[i] = 8;
    build_huffman_table(&g_fixed_litlen, lengths, 288);

    /* Build fixed distance table */
    for (i = 0; i < 32; i++) lengths[i] = 5;
    build_huffman_table(&g_fixed_dist, lengths, 32);

    g_fixed_tables_built = 1;
}

/* ============================================================================
 * Dynamic Huffman Tables
 * ============================================================================
 */

static int decode_dynamic_tables(bit_reader_t* br,
                                 huffman_table_t* litlen,
                                 huffman_table_t* dist) {
    /* Read header */
    int hlit = (int)bitreader_read(br, 5) + 257;
    int hdist = (int)bitreader_read(br, 5) + 1;
    int hclen = (int)bitreader_read(br, 4) + 4;

    if (hlit > DEFLATE_MAX_LITLEN_CODES || hdist > DEFLATE_MAX_DIST_CODES) {
        return -1;
    }

    /* Read code length code lengths */
    uint8_t codelen_lengths[DEFLATE_MAX_CODELEN_CODES] = {0};
    for (int i = 0; i < hclen; i++) {
        codelen_lengths[codelen_order[i]] = (uint8_t)bitreader_read(br, 3);
    }

    /* Build code length table */
    huffman_table_t codelen_table;
    if (build_huffman_table(&codelen_table, codelen_lengths, DEFLATE_MAX_CODELEN_CODES) < 0) {
        return -1;
    }

    /* Decode literal/length and distance code lengths */
    uint8_t lengths[DEFLATE_MAX_LITLEN_CODES + DEFLATE_MAX_DIST_CODES];
    int total = hlit + hdist;
    int i = 0;

    while (i < total) {
        int sym = decode_symbol(br, &codelen_table);
        if (sym < 0) return -1;

        if (sym < 16) {
            lengths[i++] = (uint8_t)sym;
        } else if (sym == 16) {
            /* Repeat previous length 3-6 times */
            if (i == 0) return -1;
            int repeat = (int)bitreader_read(br, 2) + 3;
            if (i + repeat > total) return -1;
            uint8_t prev = lengths[i - 1];
            while (repeat-- > 0) {
                lengths[i++] = prev;
            }
        } else if (sym == 17) {
            /* Repeat zero 3-10 times */
            int repeat = (int)bitreader_read(br, 3) + 3;
            if (i + repeat > total) return -1;
            while (repeat-- > 0) {
                lengths[i++] = 0;
            }
        } else if (sym == 18) {
            /* Repeat zero 11-138 times */
            int repeat = (int)bitreader_read(br, 7) + 11;
            if (i + repeat > total) return -1;
            while (repeat-- > 0) {
                lengths[i++] = 0;
            }
        } else {
            return -1;
        }
    }

    /* Build tables */
    if (build_huffman_table(litlen, lengths, hlit) < 0) {
        return -1;
    }
    if (build_huffman_table(dist, lengths + hlit, hdist) < 0) {
        return -1;
    }

    return 0;
}

/* ============================================================================
 * DEFLATE Decompression
 * ============================================================================
 */

static carquet_status_t deflate_decompress_block(
    bit_reader_t* br,
    uint8_t* output,
    size_t output_capacity,
    size_t* output_pos,
    const huffman_table_t* litlen,
    const huffman_table_t* dist) {

    while (1) {
        int sym = decode_symbol(br, litlen);
        if (sym < 0) {
            return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
        }

        if (sym < 256) {
            /* Literal byte */
            if (*output_pos >= output_capacity) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            output[(*output_pos)++] = (uint8_t)sym;
        } else if (sym == DEFLATE_END_BLOCK) {
            /* End of block */
            break;
        } else {
            /* Length/distance pair */
            int length_code = sym - 257;
            if (length_code >= 29) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            int length = length_base[length_code];
            int extra = length_extra[length_code];
            if (extra > 0) {
                length += (int)bitreader_read(br, extra);
            }

            int dist_code = decode_symbol(br, dist);
            if (dist_code < 0 || dist_code >= 30) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            int distance = dist_base[dist_code];
            extra = dist_extra[dist_code];
            if (extra > 0) {
                distance += (int)bitreader_read(br, extra);
            }

            /* Copy from back reference */
            if ((size_t)distance > *output_pos) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            if (*output_pos + (size_t)length > output_capacity) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            /* Handle overlapping copy */
            size_t src_pos = *output_pos - (size_t)distance;
            for (int i = 0; i < length; i++) {
                output[*output_pos + (size_t)i] = output[src_pos + (size_t)i];
            }
            *output_pos += (size_t)length;
        }
    }

    return CARQUET_OK;
}

int carquet_gzip_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    build_fixed_tables();

    bit_reader_t br;
    bitreader_init(&br, src, src_size);

    size_t output_pos = 0;
    int bfinal;

    do {
        /* Validate we have enough bits for block header */
        if (!bitreader_has_bits(&br, 3)) {
            return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
        }

        /* Read block header */
        bfinal = (int)bitreader_read(&br, 1);
        int btype = (int)bitreader_read(&br, 2);

        if (btype == DEFLATE_BLOCK_STORED) {
            /* Stored (uncompressed) block */
            /* Skip to byte boundary */
            br.bits = 0;
            br.num_bits = 0;

            if (br.pos + 4 > br.size) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            uint16_t len = br.data[br.pos] | ((uint16_t)br.data[br.pos + 1] << 8);
            uint16_t nlen = br.data[br.pos + 2] | ((uint16_t)br.data[br.pos + 3] << 8);
            br.pos += 4;

            /* Verify complement */
            if ((uint16_t)~nlen != len) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            if (br.pos + len > br.size) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }
            if (output_pos + len > dst_capacity) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            memcpy(dst + output_pos, br.data + br.pos, len);
            br.pos += len;
            output_pos += len;

        } else if (btype == DEFLATE_BLOCK_FIXED) {
            /* Fixed Huffman codes */
            carquet_status_t status = deflate_decompress_block(
                &br, dst, dst_capacity, &output_pos,
                &g_fixed_litlen, &g_fixed_dist);
            if (status != CARQUET_OK) {
                return status;
            }

        } else if (btype == DEFLATE_BLOCK_DYNAMIC) {
            /* Dynamic Huffman codes */
            huffman_table_t litlen, dist;
            if (decode_dynamic_tables(&br, &litlen, &dist) < 0) {
                return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
            }

            carquet_status_t status = deflate_decompress_block(
                &br, dst, dst_capacity, &output_pos,
                &litlen, &dist);
            if (status != CARQUET_OK) {
                return status;
            }

        } else {
            /* Invalid block type */
            return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
        }

    } while (!bfinal);

    *dst_size = output_pos;
    return CARQUET_OK;
}

/* ============================================================================
 * Bit Writer for Compression
 * ============================================================================
 */

typedef struct {
    uint8_t* data;
    size_t capacity;
    size_t pos;
    uint32_t bits;
    int num_bits;
} bit_writer_t;

static void bitwriter_init(bit_writer_t* bw, uint8_t* data, size_t capacity) {
    bw->data = data;
    bw->capacity = capacity;
    bw->pos = 0;
    bw->bits = 0;
    bw->num_bits = 0;
}

static inline int bitwriter_write(bit_writer_t* bw, uint32_t value, int n) {
    bw->bits |= value << bw->num_bits;
    bw->num_bits += n;

    while (bw->num_bits >= 8) {
        if (bw->pos >= bw->capacity) {
            return -1;
        }
        bw->data[bw->pos++] = (uint8_t)bw->bits;
        bw->bits >>= 8;
        bw->num_bits -= 8;
    }
    return 0;
}

static inline int bitwriter_flush(bit_writer_t* bw) {
    if (bw->num_bits > 0) {
        if (bw->pos >= bw->capacity) {
            return -1;
        }
        bw->data[bw->pos++] = (uint8_t)bw->bits;
        bw->bits = 0;
        bw->num_bits = 0;
    }
    return 0;
}

/* ============================================================================
 * Huffman Encoding Tables
 * ============================================================================
 */

typedef struct {
    uint16_t code;
    uint8_t len;
} huffman_code_t;

/* Fixed Huffman encoding tables */
static int g_fixed_encode_built = 0;
static huffman_code_t g_fixed_litlen_codes[DEFLATE_MAX_LITLEN_CODES];
static huffman_code_t g_fixed_dist_codes[DEFLATE_MAX_DIST_CODES];

static uint16_t reverse_bits(uint16_t v, int n) {
    uint16_t r = 0;
    for (int i = 0; i < n; i++) {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

static void build_fixed_encode_tables(void) {
    if (g_fixed_encode_built) {
        return;
    }

    /* Build fixed literal/length encoding table */
    uint16_t code = 0;
    int i;

    /* Codes 256-279: 7 bits (codes 0-23) */
    for (i = 256; i < 280; i++) {
        g_fixed_litlen_codes[i].code = reverse_bits(code++, 7);
        g_fixed_litlen_codes[i].len = 7;
    }

    /* Codes 0-143: 8 bits (codes 48-191) */
    code = 48;
    for (i = 0; i < 144; i++) {
        g_fixed_litlen_codes[i].code = reverse_bits(code++, 8);
        g_fixed_litlen_codes[i].len = 8;
    }

    /* Codes 280-287: 8 bits (codes 192-199) */
    code = 192;
    for (i = 280; i < 288; i++) {
        g_fixed_litlen_codes[i].code = reverse_bits(code++, 8);
        g_fixed_litlen_codes[i].len = 8;
    }

    /* Codes 144-255: 9 bits (codes 400-511) */
    code = 400;
    for (i = 144; i < 256; i++) {
        g_fixed_litlen_codes[i].code = reverse_bits(code++, 9);
        g_fixed_litlen_codes[i].len = 9;
    }

    /* Build fixed distance encoding table (all 5 bits) */
    for (i = 0; i < 32; i++) {
        g_fixed_dist_codes[i].code = reverse_bits((uint16_t)i, 5);
        g_fixed_dist_codes[i].len = 5;
    }

    g_fixed_encode_built = 1;
}

/* ============================================================================
 * Length/Distance Code Lookup
 * ============================================================================
 */

static int get_length_code(int length, int* extra_bits, int* extra_value) {
    for (int i = 0; i < 29; i++) {
        int base = length_base[i];
        int extra = length_extra[i];
        int max = base + (1 << extra) - 1;
        if (length <= max) {
            *extra_bits = extra;
            *extra_value = length - base;
            return i + 257;
        }
    }
    return 285;  /* Max length code */
}

static int get_dist_code(int distance, int* extra_bits, int* extra_value) {
    for (int i = 0; i < 30; i++) {
        int base = dist_base[i];
        int extra = dist_extra[i];
        int max = base + (1 << extra) - 1;
        if (distance <= max) {
            *extra_bits = extra;
            *extra_value = distance - base;
            return i;
        }
    }
    return 29;  /* Max distance code */
}

/* ============================================================================
 * LZ77 Matching
 * ============================================================================
 */

typedef struct {
    int16_t head[DEFLATE_HASH_SIZE];
    int16_t prev[DEFLATE_WINDOW_SIZE];
} match_finder_t;

static inline uint32_t hash3(const uint8_t* p) {
    return ((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16))
           * 2654435761U >> (32 - DEFLATE_HASH_BITS);
}

static void match_finder_init(match_finder_t* mf) {
    memset(mf->head, -1, sizeof(mf->head));
}

static int find_match(const match_finder_t* mf,
                      const uint8_t* src,
                      int pos,
                      int src_size,
                      int* match_pos,
                      int level) {
    if (pos + DEFLATE_MIN_MATCH > src_size) {
        return 0;
    }

    uint32_t h = hash3(src + pos);
    int chain_len = level < 4 ? 4 : (level < 7 ? 32 : 128);
    int best_len = DEFLATE_MIN_MATCH - 1;
    int best_pos = 0;
    int limit = pos - DEFLATE_WINDOW_SIZE;
    if (limit < 0) limit = 0;

    int cur = mf->head[h];
    while (cur >= limit && chain_len-- > 0) {
        if (src[cur + best_len] == src[pos + best_len]) {
            /* Check full match */
            int len = 0;
            int max_len = src_size - pos;
            if (max_len > DEFLATE_MAX_MATCH) {
                max_len = DEFLATE_MAX_MATCH;
            }

            while (len < max_len && src[cur + len] == src[pos + len]) {
                len++;
            }

            if (len > best_len) {
                best_len = len;
                best_pos = cur;
                if (len >= DEFLATE_MAX_MATCH) {
                    break;
                }
            }
        }
        cur = mf->prev[cur & (DEFLATE_WINDOW_SIZE - 1)];
    }

    if (best_len >= DEFLATE_MIN_MATCH) {
        *match_pos = best_pos;
        return best_len;
    }
    return 0;
}

static void match_finder_insert(match_finder_t* mf, const uint8_t* src, int pos) {
    uint32_t h = hash3(src + pos);
    mf->prev[pos & (DEFLATE_WINDOW_SIZE - 1)] = mf->head[h];
    mf->head[h] = (int16_t)pos;
}

/* ============================================================================
 * DEFLATE Compression
 * ============================================================================
 */

int carquet_gzip_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size,
    int level) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Clamp level */
    if (level < 1) level = 1;
    if (level > 9) level = 9;

    build_fixed_encode_tables();

    /* For very small inputs or level 1, use stored block */
    if (src_size < 10 || (level == 1 && src_size < 1024)) {
        /* Stored block */
        size_t block_size = src_size;
        if (block_size > 65535) {
            block_size = 65535;  /* Max stored block size */
        }

        size_t needed = 1 + 4 + block_size;  /* header + len/nlen + data */
        if (needed > dst_capacity) {
            return CARQUET_ERROR_COMPRESSION;
        }

        dst[0] = (src_size <= 65535) ? 0x01 : 0x00;  /* BFINAL=1 if single block, BTYPE=00 */
        dst[1] = (uint8_t)(block_size & 0xFF);
        dst[2] = (uint8_t)(block_size >> 8);
        dst[3] = (uint8_t)(~block_size & 0xFF);
        dst[4] = (uint8_t)((~block_size >> 8) & 0xFF);
        memcpy(dst + 5, src, block_size);
        *dst_size = 5 + block_size;
        return CARQUET_OK;
    }

    /* Use fixed Huffman coding */
    bit_writer_t bw;
    bitwriter_init(&bw, dst, dst_capacity);

    /* Write block header: BFINAL=1, BTYPE=01 (fixed Huffman) */
    if (bitwriter_write(&bw, 1, 1) < 0) return CARQUET_ERROR_COMPRESSION;  /* BFINAL */
    if (bitwriter_write(&bw, 1, 2) < 0) return CARQUET_ERROR_COMPRESSION;  /* BTYPE=01 */

    /* Initialize match finder */
    match_finder_t* mf = NULL;

    /* Only use LZ77 for higher levels */
    if (level >= 4) {
        /* Static allocation would be better but this works */
        static match_finder_t static_mf;
        mf = &static_mf;
        match_finder_init(mf);
    }

    int pos = 0;
    int src_len = (int)src_size;

    while (pos < src_len) {
        int match_len = 0;
        int match_pos = 0;

        /* Try to find a match */
        if (mf && pos + DEFLATE_MIN_MATCH <= src_len) {
            match_len = find_match(mf, src, pos, src_len, &match_pos, level);
        }

        if (match_len >= DEFLATE_MIN_MATCH) {
            /* Emit length/distance pair */
            int extra_bits, extra_value;
            int length_code = get_length_code(match_len, &extra_bits, &extra_value);

            huffman_code_t* lc = &g_fixed_litlen_codes[length_code];
            if (bitwriter_write(&bw, lc->code, lc->len) < 0) {
                return CARQUET_ERROR_COMPRESSION;
            }
            if (extra_bits > 0) {
                if (bitwriter_write(&bw, (uint32_t)extra_value, extra_bits) < 0) {
                    return CARQUET_ERROR_COMPRESSION;
                }
            }

            int distance = pos - match_pos;
            int dist_code = get_dist_code(distance, &extra_bits, &extra_value);

            huffman_code_t* dc = &g_fixed_dist_codes[dist_code];
            if (bitwriter_write(&bw, dc->code, dc->len) < 0) {
                return CARQUET_ERROR_COMPRESSION;
            }
            if (extra_bits > 0) {
                if (bitwriter_write(&bw, (uint32_t)extra_value, extra_bits) < 0) {
                    return CARQUET_ERROR_COMPRESSION;
                }
            }

            /* Insert all positions of the match into hash table */
            if (mf) {
                for (int i = 0; i < match_len; i++) {
                    match_finder_insert(mf, src, pos + i);
                }
            }
            pos += match_len;

        } else {
            /* Emit literal */
            huffman_code_t* lc = &g_fixed_litlen_codes[src[pos]];
            if (bitwriter_write(&bw, lc->code, lc->len) < 0) {
                return CARQUET_ERROR_COMPRESSION;
            }

            if (mf && pos + DEFLATE_MIN_MATCH <= src_len) {
                match_finder_insert(mf, src, pos);
            }
            pos++;
        }
    }

    /* Emit end of block */
    huffman_code_t* eob = &g_fixed_litlen_codes[DEFLATE_END_BLOCK];
    if (bitwriter_write(&bw, eob->code, eob->len) < 0) {
        return CARQUET_ERROR_COMPRESSION;
    }

    /* Flush remaining bits */
    if (bitwriter_flush(&bw) < 0) {
        return CARQUET_ERROR_COMPRESSION;
    }

    *dst_size = bw.pos;
    return CARQUET_OK;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================
 */

size_t carquet_gzip_compress_bound(size_t src_size) {
    /* Worst case: stored blocks with headers */
    return src_size + (src_size / 65535 + 1) * 5 + 10;
}
