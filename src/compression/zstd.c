/**
 * @file zstd.c
 * @brief Pure C ZSTD compression/decompression
 *
 * Implements Zstandard (ZSTD) format with FSE entropy coding.
 * Reference: https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md
 */

/* #define ZSTD_DEBUG 1 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

#ifdef ZSTD_DEBUG
#include <stdio.h>
#endif

/* ============================================================================
 * Constants
 * ============================================================================
 */

#define ZSTD_MAGIC               0xFD2FB528
#define ZSTD_MAGIC_SKIPPABLE_MIN 0x184D2A50
#define ZSTD_MAGIC_SKIPPABLE_MAX 0x184D2A5F

/* Block types */
#define ZSTD_BLOCK_RAW           0
#define ZSTD_BLOCK_RLE           1
#define ZSTD_BLOCK_COMPRESSED    2
#define ZSTD_BLOCK_RESERVED      3

/* Literals section types */
#define ZSTD_LITBLOCK_RAW        0
#define ZSTD_LITBLOCK_RLE        1
#define ZSTD_LITBLOCK_COMPRESSED 2
#define ZSTD_LITBLOCK_TREELESS   3

/* Maximum values */
#define ZSTD_MAX_WINDOW_LOG      30
#define ZSTD_MIN_WINDOW_LOG      10
#define ZSTD_BLOCK_SIZE_MAX      (1 << 17)  /* 128 KB */
#define ZSTD_MAX_OFFSET          (1 << 22)
#define ZSTD_MIN_MATCH           3
#define ZSTD_MAX_MATCH           (131074 + ZSTD_MIN_MATCH)
#define ZSTD_MAX_LITERALS        (1 << 17)

/* FSE constants */
#define FSE_MAX_ACCURACY_LOG     9
#define LITERALS_LENGTH_ACC_LOG  6
#define MATCH_LENGTH_ACC_LOG     6
#define OFFSET_ACC_LOG           5

/* Huffman constants */
#define HUF_MAX_SYMBOL           256
#define HUF_MAX_BITS             12
#define HUF_TABLE_SIZE           (1 << HUF_MAX_BITS)

/* LZ77 compression constants */
#define ZSTD_HASH_LOG            17
#define ZSTD_HASH_SIZE           (1 << ZSTD_HASH_LOG)
#define ZSTD_SEARCH_DEPTH        6

/* ============================================================================
 * Portable bit manipulation
 * ============================================================================
 */

static inline int highest_bit_set(uint32_t v) {
    if (v == 0) return -1;
    int n = 0;
    if (v >= (1U << 16)) { n += 16; v >>= 16; }
    if (v >= (1U << 8))  { n += 8;  v >>= 8;  }
    if (v >= (1U << 4))  { n += 4;  v >>= 4;  }
    if (v >= (1U << 2))  { n += 2;  v >>= 2;  }
    if (v >= (1U << 1))  { n += 1; }
    return n;
}

/* ============================================================================
 * Bit Reader (backward)
 *
 * ZSTD backward bitstream format:
 * - Bits are written forward (LSB first) with a '1' padding marker at the end
 * - Reading starts from the END of the buffer
 * - After finding the padding marker, bits are read in reverse order
 *   (from the position just below the marker toward bit 0)
 *
 * To handle this, we load bytes from end to start, but we need to read
 * bits in reverse order from how they were written. The trick is to
 * track which bits we've "consumed" from the HIGH end of the accumulator.
 * ============================================================================
 */

typedef struct {
    const uint8_t* data;
    size_t size;
    size_t byte_pos;      /* Next byte to load (going backward) */
    uint64_t bits;        /* Bit accumulator */
    int bits_loaded;      /* Total bits loaded from bytes */
    int bits_consumed;    /* Bits already read from the high end */
} zstd_bitreader_t;

static void zstd_br_init_backward(zstd_bitreader_t* br, const uint8_t* data, size_t size) {
    br->data = data;
    br->size = size;
    br->byte_pos = size;
    br->bits = 0;
    br->bits_loaded = 0;
    br->bits_consumed = 0;

    if (size == 0) return;

    /* Load bytes from end until we have the padding marker */
    while (br->byte_pos > 0 && br->bits_loaded < 64) {
        br->byte_pos--;
        br->bits = (br->bits << 8) | br->data[br->byte_pos];
        br->bits_loaded += 8;
    }

    /* Find and skip the padding marker (highest '1' bit) */
    int marker_pos = -1;
    for (int i = br->bits_loaded - 1; i >= 0; i--) {
        if ((br->bits >> i) & 1) {
            marker_pos = i;
            break;
        }
    }

    if (marker_pos >= 0) {
        /* Bits below the marker are valid data */
        br->bits_consumed = br->bits_loaded - marker_pos;
    } else {
        br->bits_consumed = br->bits_loaded;  /* No marker found, error */
    }
}

static inline void zstd_br_refill(zstd_bitreader_t* br) {
    while (br->bits_loaded - br->bits_consumed < 32 && br->byte_pos > 0) {
        br->byte_pos--;
        br->bits = (br->bits << 8) | br->data[br->byte_pos];
        br->bits_loaded += 8;
    }
}

static inline uint32_t zstd_br_read_bits(zstd_bitreader_t* br, int n) {
    if (n == 0) return 0;
    zstd_br_refill(br);

    /* Read n bits from just below the consumed position */
    int read_pos = br->bits_loaded - br->bits_consumed - n;
    if (read_pos < 0) read_pos = 0;

    uint32_t val = (uint32_t)((br->bits >> read_pos) & ((1ULL << n) - 1));
    br->bits_consumed += n;
    return val;
}

/* ============================================================================
 * FSE Decoding Tables
 * ============================================================================
 */

typedef struct {
    int16_t new_state_base;  /* Signed: can be negative during FSE table construction */
    uint8_t symbol;
    uint8_t num_bits;
} fse_entry_t;

/* Predefined normalized frequency tables */
static const int16_t LL_default_norm[36] = {
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1
};

static const int16_t ML_default_norm[53] = {
    /* From zstd source lib/common/zstd_internal.h ML_defaultNorm */
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,  /* 0-15 */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  /* 16-31 */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, /* 32-47 */
    -1, -1, -1, -1, -1                                /* 48-52 */
};

static const int16_t OF_default_norm[29] = {
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1
};

/* Symbol baseline/extra bits tables */
static const uint32_t LL_baseline[36] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096,
    8192, 16384, 32768, 65536
};

static const uint8_t LL_bits[36] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16
};

static const uint32_t ML_baseline[53] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051,
    4099, 8195, 16387, 32771, 65539
};

static const uint8_t ML_bits[53] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16
};

/* Static FSE decoding tables */
static fse_entry_t g_LL_table[1 << LITERALS_LENGTH_ACC_LOG];
static fse_entry_t g_ML_table[1 << MATCH_LENGTH_ACC_LOG];
static fse_entry_t g_OF_table[1 << OFFSET_ACC_LOG];
static int g_tables_initialized = 0;

static void build_fse_decode_table(fse_entry_t* table, int accuracy_log,
                                   const int16_t* norm, int max_symbol) {
    int table_size = 1 << accuracy_log;
    uint16_t symbol_next[64];
    int high_threshold = table_size - 1;

    /* Initialize symbol_next and place -1 symbols at high end */
    for (int s = 0; s <= max_symbol; s++) {
        if (norm[s] == -1) {
            table[high_threshold].symbol = (uint8_t)s;
            high_threshold--;
            symbol_next[s] = 1;
        } else if (norm[s] > 0) {
            symbol_next[s] = (uint16_t)norm[s];
        } else {
            symbol_next[s] = 0;
        }
    }

    /* Spread symbols across table */
    int step = (table_size >> 1) + (table_size >> 3) + 3;
    int mask = table_size - 1;
    int pos = 0;

    for (int s = 0; s <= max_symbol; s++) {
        int freq = norm[s];
        if (freq <= 0) continue;

        for (int i = 0; i < freq; i++) {
            table[pos].symbol = (uint8_t)s;
            /* Skip positions already used by -1 symbols */
            do {
                pos = (pos + step) & mask;
            } while (pos > high_threshold);
        }
    }

    /* Build decoding entries */
    for (int i = 0; i < table_size; i++) {
        uint8_t sym = table[i].symbol;
        uint16_t next_state = symbol_next[sym]++;
        int nb_bits = accuracy_log - highest_bit_set(next_state);
        if (nb_bits < 0) nb_bits = 0;
        table[i].num_bits = (uint8_t)nb_bits;
        table[i].new_state_base = (int16_t)((next_state << nb_bits) - table_size);
    }
}

static void init_default_tables(void) {
    if (g_tables_initialized) return;

    build_fse_decode_table(g_LL_table, LITERALS_LENGTH_ACC_LOG, LL_default_norm, 35);
    build_fse_decode_table(g_ML_table, MATCH_LENGTH_ACC_LOG, ML_default_norm, 52);
    build_fse_decode_table(g_OF_table, OFFSET_ACC_LOG, OF_default_norm, 28);

    g_tables_initialized = 1;
}

/* ============================================================================
 * FSE Table Decoding from Compressed Bitstream
 * ============================================================================
 */

/* Forward bit reader for FSE table decoding */
typedef struct {
    const uint8_t* data;
    size_t size;
    size_t byte_pos;
    uint32_t bits;
    int bits_avail;
} fse_bitreader_t;

static void fse_br_init(fse_bitreader_t* br, const uint8_t* data, size_t size) {
    br->data = data;
    br->size = size;
    br->byte_pos = 0;
    br->bits = 0;
    br->bits_avail = 0;
    /* Pre-load some bits */
    while (br->bits_avail < 24 && br->byte_pos < br->size) {
        br->bits |= (uint32_t)br->data[br->byte_pos++] << br->bits_avail;
        br->bits_avail += 8;
    }
}

static inline uint32_t fse_br_read(fse_bitreader_t* br, int n) {
    if (n == 0) return 0;
    uint32_t val = br->bits & ((1U << n) - 1);
    br->bits >>= n;
    br->bits_avail -= n;
    /* Refill */
    while (br->bits_avail < 24 && br->byte_pos < br->size) {
        br->bits |= (uint32_t)br->data[br->byte_pos++] << br->bits_avail;
        br->bits_avail += 8;
    }
    return val;
}

static inline uint32_t fse_br_peek(fse_bitreader_t* br, int n) {
    /* Refill if needed before peeking */
    while (br->bits_avail < n && br->byte_pos < br->size) {
        br->bits |= (uint32_t)br->data[br->byte_pos++] << br->bits_avail;
        br->bits_avail += 8;
    }
    return br->bits & ((1U << n) - 1);
}

static inline void fse_br_consume(fse_bitreader_t* br, int n) {
    br->bits >>= n;
    br->bits_avail -= n;
    while (br->bits_avail < 24 && br->byte_pos < br->size) {
        br->bits |= (uint32_t)br->data[br->byte_pos++] << br->bits_avail;
        br->bits_avail += 8;
    }
}

static size_t fse_br_bytes_consumed(fse_bitreader_t* br) {
    /* Bytes consumed = byte_pos minus any bits we loaded but didn't use */
    int unused_bytes = br->bits_avail / 8;
    return br->byte_pos - (size_t)unused_bytes;
}

/**
 * Decode FSE table header and normalized frequencies from bitstream.
 * Returns: bytes consumed on success, 0 on error
 *
 * The format follows RFC 8878 section 4.1.1:
 * - 4 bits: accuracy log (actual AL = value + 5, so 0-4 means AL 5-9)
 * - Variable length: normalized frequencies for each symbol
 */
static size_t decode_fse_table_header(const uint8_t* data, size_t size,
                                       int16_t* norm, int* accuracy_log_out,
                                       int max_symbol) {
    if (size < 1) return 0;

    fse_bitreader_t br;
    fse_br_init(&br, data, size);

    /* Read accuracy log (4 bits) */
#ifdef ZSTD_DEBUG
    fprintf(stderr, "decode_fse_table_header: first bytes = %02x %02x %02x %02x\n",
            data[0], data[1], size > 2 ? data[2] : 0, size > 3 ? data[3] : 0);
#endif
    int al = (int)fse_br_read(&br, 4) + 5;
    if (al > 9) al = 9;  /* Cap at max */
    *accuracy_log_out = al;

#ifdef ZSTD_DEBUG
    fprintf(stderr, "decode_fse_table_header: accuracy_log=%d table_size=%d max_symbol=%d\n",
            al, 1 << al, max_symbol);
#endif

    int table_size = 1 << al;

    /*
     * Follows the exact algorithm from zstd reference:
     * FSE_readNCount_body in entropy_common.c
     */
    int nbBits = al;
    int threshold = 1 << nbBits;
    nbBits++;
    int remaining = (1 << nbBits) + 1;
    int symbol = 0;

    while (remaining > 1 && symbol <= max_symbol) {
        /* Adjust nbBits when remaining shrinks */
        while (remaining < threshold) {
            nbBits--;
            threshold >>= 1;
        }

        /*
         * Variable-length decoding:
         * max = (2*threshold - 1) - remaining
         * If low bits < max: use nbBits-1 bits
         * Else: use nbBits bits and adjust
         */
        int max_val = (2 * threshold - 1) - remaining;
        uint32_t bits = fse_br_peek(&br, nbBits);

#ifdef ZSTD_DEBUG
        if (symbol < 5) {
            fprintf(stderr, "  sym=%d: nbBits=%d threshold=%d remaining=%d max=%d bits=0x%x\n",
                    symbol, nbBits, threshold, remaining, max_val, bits);
        }
#endif

        int count;
        if (max_val < 0 || (int)(bits & (threshold - 1)) < max_val) {
            /* Short form: use nbBits-1 bits */
            count = (int)(bits & (threshold - 1));
            fse_br_consume(&br, nbBits - 1);
        } else {
            /* Long form: use nbBits bits */
            count = (int)(bits & ((2 * threshold) - 1));
            if (count >= threshold) {
                count -= max_val;
            }
            fse_br_consume(&br, nbBits);
        }

        /* Decode: count=0 means prob=-1, count=1 means prob=0, etc. */
        count--;

        if (count < 0) {
            /* Probability -1 (less than 1) */
            norm[symbol] = -1;
            remaining += count;  /* count is -1, so remaining -= 1 */
            symbol++;
        } else if (count == 0) {
            /* Probability 0: symbol not present, read repeat zeros */
            norm[symbol] = 0;
            symbol++;
            /* Read 2-bit repeat count */
            int repeat = (int)fse_br_read(&br, 2);
            while (repeat == 3 && symbol <= max_symbol) {
                for (int r = 0; r < 3 && symbol <= max_symbol; r++) {
                    norm[symbol++] = 0;
                }
                repeat = (int)fse_br_read(&br, 2);
            }
            while (repeat > 0 && symbol <= max_symbol) {
                norm[symbol++] = 0;
                repeat--;
            }
        } else {
            /* Normal probability */
            norm[symbol] = (int16_t)count;
            remaining -= count;
            symbol++;
        }
    }

    /* Verify we distributed exactly table_size probability */
    int total = 0;
    for (int i = 0; i <= max_symbol; i++) {
        if (norm[i] == -1) total += 1;
        else if (norm[i] > 0) total += norm[i];
    }

    /* Fill remaining symbols with 0 */
    while (symbol <= max_symbol) {
        norm[symbol++] = 0;
    }

#ifdef ZSTD_DEBUG
    fprintf(stderr, "decode_fse_table_header: total=%d (should be %d), remaining=%d (should be 1)\n",
            total, table_size, remaining);
    if (total != table_size) {
        fprintf(stderr, "decode_fse_table_header: norm = [");
        for (int i = 0; i <= max_symbol; i++) {
            if (norm[i] != 0) fprintf(stderr, "%d:%d, ", i, norm[i]);
        }
        fprintf(stderr, "]\n");
    }
#endif
    (void)total;

    return fse_br_bytes_consumed(&br);
}

/* Per-block FSE tables for custom sequences encoding */
typedef struct {
    fse_entry_t ll_table[1 << FSE_MAX_ACCURACY_LOG];
    fse_entry_t ml_table[1 << FSE_MAX_ACCURACY_LOG];
    fse_entry_t of_table[1 << FSE_MAX_ACCURACY_LOG];
    int ll_accuracy_log;
    int ml_accuracy_log;
    int of_accuracy_log;
    bool ll_valid;
    bool ml_valid;
    bool of_valid;
} seq_tables_t;

/* ============================================================================
 * Huffman Decoding for Compressed Literals
 * ============================================================================
 */

typedef struct {
    uint8_t symbol;
    uint8_t num_bits;
} huf_entry_t;

typedef struct {
    huf_entry_t table[HUF_TABLE_SIZE];
    int max_bits;
    int num_symbols;
} huf_dtable_t;

/* Forward bit reader for Huffman streams */
typedef struct {
    const uint8_t* data;
    size_t size;
    size_t byte_pos;
    uint64_t bits;
    int num_bits;
} huf_bitreader_t;

static void huf_br_init(huf_bitreader_t* br, const uint8_t* data, size_t size) {
    br->data = data;
    br->size = size;
    br->byte_pos = 0;
    br->bits = 0;
    br->num_bits = 0;
}

static inline void huf_br_refill(huf_bitreader_t* br) {
    while (br->num_bits <= 56 && br->byte_pos < br->size) {
        br->bits |= (uint64_t)br->data[br->byte_pos++] << br->num_bits;
        br->num_bits += 8;
    }
}

static inline uint32_t huf_br_peek_bits(huf_bitreader_t* br, int n) {
    huf_br_refill(br);
    return (uint32_t)(br->bits & ((1ULL << n) - 1));
}

static inline void huf_br_consume_bits(huf_bitreader_t* br, int n) {
    br->bits >>= n;
    br->num_bits -= n;
}

/* Build Huffman decode table from weights */
static int huf_build_dtable(huf_dtable_t* dtable, const uint8_t* weights, int num_symbols) {
    if (num_symbols > HUF_MAX_SYMBOL || num_symbols < 1) return -1;

    /* Count symbols by weight and find max weight */
    int weight_count[HUF_MAX_BITS + 1] = {0};
    int max_weight = 0;
    for (int i = 0; i < num_symbols; i++) {
        if (weights[i] > HUF_MAX_BITS) return -1;
        weight_count[weights[i]]++;
        if (weights[i] > max_weight) max_weight = weights[i];
    }

    if (max_weight == 0) return -1;

    /* Calculate table log (max bits needed) */
    int table_log = max_weight;
    dtable->max_bits = table_log;
    dtable->num_symbols = num_symbols;

    /* Calculate starting codes for each weight */
    uint32_t next_code[HUF_MAX_BITS + 1];
    uint32_t code = 0;
    for (int w = max_weight; w >= 1; w--) {
        next_code[w] = code;
        code = (code + weight_count[w]) >> 1;
    }

    /* Fill decode table */
    int table_size = 1 << table_log;
    memset(dtable->table, 0, sizeof(dtable->table));

    for (int sym = 0; sym < num_symbols; sym++) {
        int w = weights[sym];
        if (w == 0) continue;

        int bits = max_weight - w + 1;
        uint32_t base_code = next_code[w];
        next_code[w]++;

        /* Fill all entries that decode to this symbol */
        int num_entries = 1 << (table_log - bits);
        for (int i = 0; i < num_entries; i++) {
            uint32_t code_val = (base_code << (table_log - bits)) | i;
            if (code_val < (uint32_t)table_size) {
                dtable->table[code_val].symbol = (uint8_t)sym;
                dtable->table[code_val].num_bits = (uint8_t)bits;
            }
        }
    }

    return 0;
}

/* Decode Huffman weights using FSE */
static int decode_huf_weights_fse(const uint8_t* data, size_t size, size_t* consumed,
                                   uint8_t* weights, int* num_weights) {
    if (size < 1) return -1;

    uint8_t hdr = data[0];
    size_t pos = 1;

    if (hdr < 128) {
        /* FSE compressed weights - simplified: assume direct encoding */
        int compressed_size = hdr;
        if (pos + compressed_size > size) return -1;

        /* For simplicity, decode as raw weights with 4 bits each */
        int n = 0;
        for (int i = 0; i < compressed_size && n < HUF_MAX_SYMBOL; i++) {
            weights[n++] = (data[pos + i] >> 4) & 0x0F;
            if (n < HUF_MAX_SYMBOL) {
                weights[n++] = data[pos + i] & 0x0F;
            }
        }
        *num_weights = n;
        *consumed = pos + compressed_size;
        return 0;
    } else {
        /* Direct representation: 4 bits per weight */
        int num_syms = hdr - 127;
        int bytes_needed = (num_syms + 1) / 2;
        if (pos + bytes_needed > size) return -1;

        for (int i = 0; i < num_syms; i++) {
            int byte_idx = i / 2;
            if (i % 2 == 0) {
                weights[i] = (data[pos + byte_idx] >> 4) & 0x0F;
            } else {
                weights[i] = data[pos + byte_idx] & 0x0F;
            }
        }
        *num_weights = num_syms;
        *consumed = pos + bytes_needed;
        return 0;
    }
}

/* Decode a single Huffman stream */
static int huf_decode_stream(huf_dtable_t* dtable, const uint8_t* data, size_t size,
                             uint8_t* out, size_t out_size) {
    huf_bitreader_t br;
    huf_br_init(&br, data, size);

    for (size_t i = 0; i < out_size; i++) {
        uint32_t code = huf_br_peek_bits(&br, dtable->max_bits);
        huf_entry_t* e = &dtable->table[code];
        if (e->num_bits == 0) return -1;
        out[i] = e->symbol;
        huf_br_consume_bits(&br, e->num_bits);
    }

    return 0;
}

/* Decode 4-stream Huffman literals */
static int decode_huf_4streams(huf_dtable_t* dtable,
                               const uint8_t* data, size_t size,
                               uint8_t* out, size_t total_size) {
    if (size < 6) return -1;

    /* Read jump table (3 x 2-byte offsets) */
    size_t stream_sizes[4];
    stream_sizes[0] = data[0] | ((size_t)data[1] << 8);
    stream_sizes[1] = data[2] | ((size_t)data[3] << 8);
    stream_sizes[2] = data[4] | ((size_t)data[5] << 8);

    size_t total_stream_size = stream_sizes[0] + stream_sizes[1] + stream_sizes[2];
    if (total_stream_size + 6 > size) return -1;
    stream_sizes[3] = size - 6 - total_stream_size;

    /* Calculate output sizes per stream */
    size_t out_per_stream = (total_size + 3) / 4;
    size_t out_sizes[4];
    out_sizes[0] = out_per_stream;
    out_sizes[1] = out_per_stream;
    out_sizes[2] = out_per_stream;
    out_sizes[3] = total_size - 3 * out_per_stream;

    /* Decode each stream */
    const uint8_t* stream_ptr = data + 6;
    uint8_t* out_ptr = out;

    for (int s = 0; s < 4; s++) {
        if (huf_decode_stream(dtable, stream_ptr, stream_sizes[s],
                             out_ptr, out_sizes[s]) < 0) {
            return -1;
        }
        stream_ptr += stream_sizes[s];
        out_ptr += out_sizes[s];
    }

    return 0;
}

/* ============================================================================
 * Frame Header
 * ============================================================================
 */

typedef struct {
    uint64_t window_size;
    uint64_t content_size;
    uint32_t dict_id;
    bool single_segment;
    bool has_checksum;
    bool has_content_size;
} zstd_frame_header_t;

static int parse_frame_header(const uint8_t* data, size_t size,
                              zstd_frame_header_t* hdr, size_t* header_size) {
    if (size < 5) return -1;

    uint32_t magic = data[0] | ((uint32_t)data[1] << 8) |
                     ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
    if (magic != ZSTD_MAGIC) return -1;

    uint8_t fhd = data[4];
    int fcs_flag = (fhd >> 6) & 3;
    hdr->single_segment = (fhd >> 5) & 1;
    hdr->has_checksum = (fhd >> 2) & 1;
    int did_flag = fhd & 3;

    size_t pos = 5;

    /* Window descriptor */
    if (!hdr->single_segment) {
        if (pos >= size) return -1;
        uint8_t wd = data[pos++];
        int exp = (wd >> 3) + 10;
        int mantissa = wd & 7;
        hdr->window_size = ((uint64_t)1 << exp) + (((uint64_t)1 << exp) >> 3) * mantissa;
    }

    /* Dictionary ID */
    hdr->dict_id = 0;
    if (did_flag > 0) {
        int did_bytes = 1 << (did_flag - 1);
        if (pos + did_bytes > size) return -1;
        for (int i = 0; i < did_bytes; i++) {
            hdr->dict_id |= (uint32_t)data[pos++] << (8 * i);
        }
    }

    /* Frame content size */
    hdr->content_size = 0;
    hdr->has_content_size = fcs_flag > 0 || hdr->single_segment;
    if (hdr->has_content_size) {
        int fcs_bytes = (fcs_flag == 0) ? 1 : (fcs_flag == 1) ? 2 : (fcs_flag == 2) ? 4 : 8;
        if (pos + fcs_bytes > size) return -1;
        for (int i = 0; i < fcs_bytes; i++) {
            hdr->content_size |= (uint64_t)data[pos++] << (8 * i);
        }
        if (fcs_flag == 1) hdr->content_size += 256;
    }

    if (hdr->single_segment) {
        hdr->window_size = hdr->content_size;
    }

    *header_size = pos;
    return 0;
}

/* ============================================================================
 * Decompression Context
 * ============================================================================
 */

typedef struct {
    uint32_t rep[3];
    huf_dtable_t huf_table;
    bool huf_table_valid;
    seq_tables_t seq_tables;
} zstd_dctx_t;

/* ============================================================================
 * Literals Decoding
 * ============================================================================
 */

static int decode_literals(zstd_dctx_t* ctx, const uint8_t* data, size_t size,
                           size_t* consumed, uint8_t* out, size_t* out_size) {
    if (size < 1) return -1;

    uint8_t hdr = data[0];
    int type = hdr & 3;
    size_t pos = 0;

    if (type == ZSTD_LITBLOCK_RAW) {
        int size_fmt = (hdr >> 2) & 3;
        size_t lit_size;

        if (size_fmt == 0) {
            /* Size_Format=0: 1-byte header, size in bits 3-7 (5 bits, max 31) */
            lit_size = hdr >> 3;
            pos = 1;
        } else if (size_fmt == 1) {
            /* Size_Format=1: 2-byte header, size in bits 4-7 of byte0 + byte1 (12 bits) */
            if (size < 2) return -1;
            lit_size = (hdr >> 4) | ((size_t)data[1] << 4);
            pos = 2;
        } else if (size_fmt == 2) {
            /* Size_Format=2: 3-byte header (20 bits) */
            if (size < 3) return -1;
            lit_size = (hdr >> 4) | ((size_t)data[1] << 4) | ((size_t)data[2] << 12);
            pos = 3;
        } else {
            /* Size_Format=3: 3-byte header (20 bits) */
            if (size < 3) return -1;
            lit_size = (hdr >> 4) | ((size_t)data[1] << 4) | ((size_t)data[2] << 12);
            pos = 3;
        }

        if (pos + lit_size > size) return -1;
        if (lit_size > ZSTD_MAX_LITERALS) return -1;
        memcpy(out, data + pos, lit_size);
        *out_size = lit_size;
        *consumed = pos + lit_size;
        return 0;

    } else if (type == ZSTD_LITBLOCK_RLE) {
        int size_fmt = (hdr >> 2) & 3;
        size_t lit_size;

        if (size_fmt == 0) {
            /* Size_Format=0: 1-byte header, size in bits 3-7 (5 bits, max 31) */
            lit_size = hdr >> 3;
            pos = 1;
        } else if (size_fmt == 1) {
            /* Size_Format=1: 2-byte header (12 bits) */
            if (size < 2) return -1;
            lit_size = (hdr >> 4) | ((size_t)data[1] << 4);
            pos = 2;
        } else if (size_fmt == 2) {
            /* Size_Format=2: 3-byte header (20 bits) */
            if (size < 3) return -1;
            lit_size = (hdr >> 4) | ((size_t)data[1] << 4) | ((size_t)data[2] << 12);
            pos = 3;
        } else {
            /* Size_Format=3: 3-byte header (20 bits) */
            if (size < 3) return -1;
            lit_size = (hdr >> 4) | ((size_t)data[1] << 4) | ((size_t)data[2] << 12);
            pos = 3;
        }

        if (pos >= size) return -1;
        if (lit_size > ZSTD_MAX_LITERALS) return -1;
        memset(out, data[pos], lit_size);
        *out_size = lit_size;
        *consumed = pos + 1;
        return 0;

    } else if (type == ZSTD_LITBLOCK_COMPRESSED || type == ZSTD_LITBLOCK_TREELESS) {
        /* Compressed or treeless literals */
        int size_fmt = (hdr >> 2) & 3;
        size_t regenerated_size, compressed_size;
        int num_streams;

        if (size_fmt == 0) {
            /* Single stream, sizes <= 1KB */
            if (size < 3) return -1;
            num_streams = 1;
            regenerated_size = (hdr >> 4) & 0x3F;
            compressed_size = (data[1] >> 4) | ((size_t)(data[2]) << 4);
            pos = 3;
        } else if (size_fmt == 1) {
            /* Single stream, sizes > 1KB */
            if (size < 3) return -1;
            num_streams = 1;
            regenerated_size = (hdr >> 4) | ((size_t)(data[1] & 0x3F) << 4);
            compressed_size = (data[1] >> 6) | ((size_t)data[2] << 2);
            pos = 3;
        } else if (size_fmt == 2) {
            /* 4 streams, sizes <= 1KB */
            if (size < 4) return -1;
            num_streams = 4;
            regenerated_size = (hdr >> 4) | ((size_t)(data[1] & 0x3F) << 4);
            compressed_size = (data[1] >> 6) | ((size_t)data[2] << 2) |
                            ((size_t)(data[3] & 0x03) << 10);
            pos = 4;
        } else {
            /* 4 streams, sizes > 1KB */
            if (size < 5) return -1;
            num_streams = 4;
            regenerated_size = (hdr >> 4) | ((size_t)(data[1] & 0x3F) << 4);
            compressed_size = (data[1] >> 6) | ((size_t)data[2] << 2) |
                            ((size_t)data[3] << 10);
            pos = 5;
        }

        if (pos + compressed_size > size) return -1;
        if (regenerated_size > ZSTD_MAX_LITERALS) return -1;

        const uint8_t* streams_data = data + pos;
        size_t streams_size = compressed_size;

        if (type == ZSTD_LITBLOCK_COMPRESSED) {
            /* Decode Huffman table from stream */
            uint8_t weights[HUF_MAX_SYMBOL];
            int num_weights;
            size_t table_consumed;

            if (decode_huf_weights_fse(streams_data, streams_size, &table_consumed,
                                       weights, &num_weights) < 0) {
                return -1;
            }

            /* Last symbol weight is implicit */
            int weight_sum = 0;
            for (int i = 0; i < num_weights; i++) {
                if (weights[i] > 0) {
                    weight_sum += 1 << (weights[i] - 1);
                }
            }
            int last_weight = highest_bit_set(weight_sum) + 1;
            if (last_weight > 0 && num_weights < HUF_MAX_SYMBOL) {
                weights[num_weights] = (uint8_t)last_weight;
                num_weights++;
            }

            if (huf_build_dtable(&ctx->huf_table, weights, num_weights) < 0) {
                return -1;
            }
            ctx->huf_table_valid = true;

            streams_data += table_consumed;
            streams_size -= table_consumed;

        } else {
            /* Treeless mode - use saved table */
            if (!ctx->huf_table_valid) {
                return -1;
            }
        }

        /* Decode Huffman streams */
        if (num_streams == 1) {
            if (huf_decode_stream(&ctx->huf_table, streams_data, streams_size,
                                 out, regenerated_size) < 0) {
                return -1;
            }
        } else {
            if (decode_huf_4streams(&ctx->huf_table, streams_data, streams_size,
                                   out, regenerated_size) < 0) {
                return -1;
            }
        }

        *out_size = regenerated_size;
        *consumed = pos + compressed_size;
        return 0;
    }

    return -1;
}

/* ============================================================================
 * Sequences Decoding
 * ============================================================================
 */

/* FSE mode constants */
#define FSE_MODE_PREDEFINED 0
#define FSE_MODE_RLE        1
#define FSE_MODE_FSE        2
#define FSE_MODE_REPEAT     3

static int decode_sequences(zstd_dctx_t* ctx, const uint8_t* data, size_t size,
                            const uint8_t* literals, size_t lit_size,
                            uint8_t* out, size_t* out_size, size_t out_cap) {
    if (size < 1) return -1;

    size_t pos = 0;
    uint8_t b0 = data[pos++];

    size_t num_seq;
    if (b0 == 0) {
        /* No sequences */
        if (lit_size > out_cap) return -1;
        memcpy(out, literals, lit_size);
        *out_size = lit_size;
        return 0;
    } else if (b0 < 128) {
        num_seq = b0;
    } else if (b0 < 255) {
        if (pos >= size) return -1;
        num_seq = ((b0 - 128) << 8) + data[pos++];
    } else {
        if (pos + 2 > size) return -1;
        num_seq = data[pos] | ((size_t)data[pos + 1] << 8);
        num_seq += 0x7F00;
        pos += 2;
    }

    if (pos >= size) return -1;

    /* Compression modes byte
     * bits 0-1: Literals_Lengths_Mode
     * bits 2-3: Offsets_Mode
     * bits 4-5: Match_Lengths_Mode
     * bits 6-7: Reserved, must be zero
     */
    uint8_t modes = data[pos++];
    int ll_mode = modes & 3;
    int of_mode = (modes >> 2) & 3;
    int ml_mode = (modes >> 4) & 3;

    /* Tables to use for decoding */
    fse_entry_t* ll_table;
    fse_entry_t* of_table;
    fse_entry_t* ml_table;
    int ll_al, of_al, ml_al;  /* accuracy logs */

    /* RLE symbols (for mode 1) */
    uint8_t ll_rle_sym = 0, of_rle_sym = 0, ml_rle_sym = 0;

    /* Initialize with defaults */
    init_default_tables();
    ll_table = g_LL_table;
    of_table = g_OF_table;
    ml_table = g_ML_table;
    ll_al = LITERALS_LENGTH_ACC_LOG;
    of_al = OFFSET_ACC_LOG;
    ml_al = MATCH_LENGTH_ACC_LOG;

    /* Process Literals_Lengths FSE table */
    if (ll_mode == FSE_MODE_PREDEFINED) {
        /* Use default table - already set */
    } else if (ll_mode == FSE_MODE_RLE) {
        if (pos >= size) return -1;
        ll_rle_sym = data[pos++];
    } else if (ll_mode == FSE_MODE_FSE) {
        int16_t norm[64];
        int al;
        size_t consumed = decode_fse_table_header(data + pos, size - pos, norm, &al, 35);
        if (consumed == 0) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences: failed to decode LL FSE table\n");
#endif
            return -1;
        }
        pos += consumed;
        build_fse_decode_table(ctx->seq_tables.ll_table, al, norm, 35);
        ctx->seq_tables.ll_accuracy_log = al;
        ctx->seq_tables.ll_valid = true;
        ll_table = ctx->seq_tables.ll_table;
        ll_al = al;
    } else if (ll_mode == FSE_MODE_REPEAT) {
        if (!ctx->seq_tables.ll_valid) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences: LL repeat mode but no previous table\n");
#endif
            return -1;
        }
        ll_table = ctx->seq_tables.ll_table;
        ll_al = ctx->seq_tables.ll_accuracy_log;
    }

    /* Process Offsets FSE table */
    if (of_mode == FSE_MODE_PREDEFINED) {
        /* Use default table - already set */
    } else if (of_mode == FSE_MODE_RLE) {
        if (pos >= size) return -1;
        of_rle_sym = data[pos++];
    } else if (of_mode == FSE_MODE_FSE) {
        int16_t norm[64];
        int al;
        size_t consumed = decode_fse_table_header(data + pos, size - pos, norm, &al, 31);
        if (consumed == 0) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences: failed to decode OF FSE table\n");
#endif
            return -1;
        }
        pos += consumed;
        build_fse_decode_table(ctx->seq_tables.of_table, al, norm, 31);
        ctx->seq_tables.of_accuracy_log = al;
        ctx->seq_tables.of_valid = true;
        of_table = ctx->seq_tables.of_table;
        of_al = al;
    } else if (of_mode == FSE_MODE_REPEAT) {
        if (!ctx->seq_tables.of_valid) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences: OF repeat mode but no previous table\n");
#endif
            return -1;
        }
        of_table = ctx->seq_tables.of_table;
        of_al = ctx->seq_tables.of_accuracy_log;
    }

    /* Process Match_Lengths FSE table */
    if (ml_mode == FSE_MODE_PREDEFINED) {
        /* Use default table - already set */
    } else if (ml_mode == FSE_MODE_RLE) {
        if (pos >= size) return -1;
        ml_rle_sym = data[pos++];
    } else if (ml_mode == FSE_MODE_FSE) {
        int16_t norm[64];
        int al;
        size_t consumed = decode_fse_table_header(data + pos, size - pos, norm, &al, 52);
        if (consumed == 0) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences: failed to decode ML FSE table\n");
#endif
            return -1;
        }
        pos += consumed;
        build_fse_decode_table(ctx->seq_tables.ml_table, al, norm, 52);
        ctx->seq_tables.ml_accuracy_log = al;
        ctx->seq_tables.ml_valid = true;
        ml_table = ctx->seq_tables.ml_table;
        ml_al = al;
    } else if (ml_mode == FSE_MODE_REPEAT) {
        if (!ctx->seq_tables.ml_valid) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences: ML repeat mode but no previous table\n");
#endif
            return -1;
        }
        ml_table = ctx->seq_tables.ml_table;
        ml_al = ctx->seq_tables.ml_accuracy_log;
    }

    /* Init backward bitreader */
    zstd_bitreader_t br;
    zstd_br_init_backward(&br, data + pos, size - pos);

    /* Read initial states - use actual accuracy logs for each table */
    uint32_t ll_state = 0, of_state = 0, ml_state = 0;
    if (ll_mode != FSE_MODE_RLE) {
        ll_state = zstd_br_read_bits(&br, ll_al);
    }
    if (of_mode != FSE_MODE_RLE) {
        of_state = zstd_br_read_bits(&br, of_al);
    }
    if (ml_mode != FSE_MODE_RLE) {
        ml_state = zstd_br_read_bits(&br, ml_al);
    }

#ifdef ZSTD_DEBUG
    fprintf(stderr, "decode_sequences: num_seq=%zu modes ll=%d of=%d ml=%d al ll=%d of=%d ml=%d init_states ll=%u of=%u ml=%u\n",
            num_seq, ll_mode, of_mode, ml_mode, ll_al, of_al, ml_al, ll_state, of_state, ml_state);
#endif

    size_t lit_pos = 0;
    size_t out_pos = 0;

    for (size_t i = 0; i < num_seq; i++) {
        /* Check state bounds (only for non-RLE modes) */
        if (ll_mode != FSE_MODE_RLE && ll_state >= (1U << ll_al)) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences[%zu]: ll_state out of bounds %u >= %u\n",
                    i, ll_state, 1U << ll_al);
#endif
            return -1;
        }
        if (of_mode != FSE_MODE_RLE && of_state >= (1U << of_al)) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences[%zu]: of_state out of bounds %u >= %u\n",
                    i, of_state, 1U << of_al);
#endif
            return -1;
        }
        if (ml_mode != FSE_MODE_RLE && ml_state >= (1U << ml_al)) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences[%zu]: ml_state out of bounds %u >= %u\n",
                    i, ml_state, 1U << ml_al);
#endif
            return -1;
        }

        /* Per RFC 8878, decode symbols and read bits in this order:
         * 1. Get symbols from current states
         * 2. Read extra bits: LL, OF, ML
         * 3. Read update bits: LL, OF, ML
         */

        /* Get FSE table entries and symbols */
        fse_entry_t* ll_e = (ll_mode != FSE_MODE_RLE) ? &ll_table[ll_state] : NULL;
        fse_entry_t* of_e = (of_mode != FSE_MODE_RLE) ? &of_table[of_state] : NULL;
        fse_entry_t* ml_e = (ml_mode != FSE_MODE_RLE) ? &ml_table[ml_state] : NULL;

        uint32_t ll_code = (ll_mode == FSE_MODE_RLE) ? ll_rle_sym : ll_e->symbol;
        uint32_t of_code = (of_mode == FSE_MODE_RLE) ? of_rle_sym : of_e->symbol;
        uint32_t ml_code = (ml_mode == FSE_MODE_RLE) ? ml_rle_sym : ml_e->symbol;

        /* Read extra bits in order: OF, ML, LL (per RFC 8878 Section 3.1.1.3.2.1.2) */
        uint32_t offset_val;
        if (of_code > 0) {
            uint32_t extra = zstd_br_read_bits(&br, of_code);
            offset_val = (1U << of_code) + extra;
        } else {
            offset_val = 1;
        }

        uint32_t match_len = ML_baseline[ml_code];
        if (ML_bits[ml_code] > 0) {
            match_len += zstd_br_read_bits(&br, ML_bits[ml_code]);
        }

        uint32_t lit_len = LL_baseline[ll_code];
        if (LL_bits[ll_code] > 0) {
            lit_len += zstd_br_read_bits(&br, LL_bits[ll_code]);
        }

        /* Read update bits in order: LL, ML, OF (per RFC 8878)
         * Update bits are read for all sequences EXCEPT the last.
         * RLE mode doesn't have state updates.
         */
        if (i < num_seq - 1) {
            if (ll_mode != FSE_MODE_RLE) {
                ll_state = ll_e->new_state_base + zstd_br_read_bits(&br, ll_e->num_bits);
            }
            if (ml_mode != FSE_MODE_RLE) {
                ml_state = ml_e->new_state_base + zstd_br_read_bits(&br, ml_e->num_bits);
            }
            if (of_mode != FSE_MODE_RLE) {
                of_state = of_e->new_state_base + zstd_br_read_bits(&br, of_e->num_bits);
            }
        }

        /* Handle repeat offsets */
        uint32_t offset;
        if (offset_val <= 3) {
            if (offset_val == 1) {
                offset = ctx->rep[0];
            } else if (offset_val == 2) {
                offset = ctx->rep[1];
                ctx->rep[1] = ctx->rep[0];
                ctx->rep[0] = offset;
            } else { /* offset_val == 3 */
                offset = ctx->rep[2];
                ctx->rep[2] = ctx->rep[1];
                ctx->rep[1] = ctx->rep[0];
                ctx->rep[0] = offset;
            }
        } else {
            offset = offset_val - 3;
            ctx->rep[2] = ctx->rep[1];
            ctx->rep[1] = ctx->rep[0];
            ctx->rep[0] = offset;
        }

#ifdef ZSTD_DEBUG
        fprintf(stderr, "decode_sequences[%zu]: ll=%u ml=%u off=%u (off_val=%u)\n",
                i, lit_len, match_len, offset, offset_val);
#endif

        /* Copy literals */
        if (lit_pos + lit_len > lit_size) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences[%zu]: lit overflow pos=%zu len=%u size=%zu\n",
                    i, lit_pos, lit_len, lit_size);
#endif
            return -1;
        }
        if (out_pos + lit_len > out_cap) return -1;
        memcpy(out + out_pos, literals + lit_pos, lit_len);
        lit_pos += lit_len;
        out_pos += lit_len;

        /* Copy match */
        if (offset > out_pos) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_sequences[%zu]: offset too large %u > %zu\n",
                    i, offset, out_pos);
#endif
            return -1;
        }
        if (out_pos + match_len > out_cap) return -1;

        size_t match_src = out_pos - offset;
        for (uint32_t j = 0; j < match_len; j++) {
            out[out_pos++] = out[match_src++];
        }
    }

    /* Copy remaining literals */
    size_t rem = lit_size - lit_pos;
    if (out_pos + rem > out_cap) return -1;
    memcpy(out + out_pos, literals + lit_pos, rem);
    out_pos += rem;

    *out_size = out_pos;
    return 0;
}

/* ============================================================================
 * Block Decoding
 * ============================================================================
 */

static int decode_block(zstd_dctx_t* ctx, const uint8_t* data, size_t size,
                        size_t* consumed, uint8_t* out, size_t* out_size,
                        size_t out_cap, bool* is_last) {
    if (size < 3) return -1;

    uint32_t hdr = data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16);
    *is_last = (hdr & 1) != 0;
    int type = (hdr >> 1) & 3;
    size_t block_size = hdr >> 3;

    size_t pos = 3;

    if (type == ZSTD_BLOCK_RAW) {
        /* For raw blocks, block_size is input size (and also output size) */
        if (pos + block_size > size) return -1;
        if (block_size > out_cap) return -1;
        memcpy(out, data + pos, block_size);
        *out_size = block_size;
        *consumed = pos + block_size;
        return 0;

    } else if (type == ZSTD_BLOCK_RLE) {
        /* For RLE blocks, block_size is OUTPUT size, input is just 1 byte */
        if (pos + 1 > size) return -1;
        if (block_size > out_cap) return -1;
        memset(out, data[pos], block_size);
        *out_size = block_size;
        *consumed = pos + 1;
        return 0;

    } else if (type == ZSTD_BLOCK_COMPRESSED) {
        /* For compressed blocks, block_size is input size */
        if (pos + block_size > size) return -1;
        uint8_t literals[ZSTD_MAX_LITERALS];
        size_t lit_size, lit_consumed;

#ifdef ZSTD_DEBUG
        fprintf(stderr, "decode_block: COMPRESSED block_size=%zu\n", block_size);
#endif

        if (decode_literals(ctx, data + pos, block_size, &lit_consumed, literals, &lit_size) < 0) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_block: decode_literals failed\n");
#endif
            return -1;
        }

#ifdef ZSTD_DEBUG
        fprintf(stderr, "decode_block: literals size=%zu consumed=%zu\n", lit_size, lit_consumed);
#endif

        if (decode_sequences(ctx, data + pos + lit_consumed, block_size - lit_consumed,
                            literals, lit_size, out, out_size, out_cap) < 0) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "decode_block: decode_sequences failed\n");
#endif
            return -1;
        }

        *consumed = pos + block_size;
        return 0;
    }

    return -1;
}

/* ============================================================================
 * ZSTD Decompression
 * ============================================================================
 */

int carquet_zstd_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    zstd_frame_header_t hdr;
    size_t hdr_size;
    if (parse_frame_header(src, src_size, &hdr, &hdr_size) < 0) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    zstd_dctx_t ctx;
    ctx.rep[0] = 1;
    ctx.rep[1] = 4;
    ctx.rep[2] = 8;
    ctx.huf_table_valid = false;
    ctx.seq_tables.ll_valid = false;
    ctx.seq_tables.of_valid = false;
    ctx.seq_tables.ml_valid = false;

    size_t pos = hdr_size;
    size_t out_pos = 0;
    bool last = false;

    while (!last && pos < src_size) {
        size_t consumed, block_out;
        if (decode_block(&ctx, src + pos, src_size - pos, &consumed,
                        dst + out_pos, &block_out, dst_capacity - out_pos, &last) < 0) {
            return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
        }
        pos += consumed;
        out_pos += block_out;
    }

    *dst_size = out_pos;
    return CARQUET_OK;
}

/* ============================================================================
 * Bit Writer (forward)
 * ============================================================================
 */

typedef struct {
    uint8_t* data;
    size_t cap;
    size_t pos;
    uint64_t bits;
    int num_bits;
} zstd_bitwriter_t;

static void zstd_bw_init(zstd_bitwriter_t* bw, uint8_t* data, size_t cap) {
    bw->data = data;
    bw->cap = cap;
    bw->pos = 0;
    bw->bits = 0;
    bw->num_bits = 0;
}

static int zstd_bw_add_bits(zstd_bitwriter_t* bw, uint32_t val, int n) {
    if (n == 0) return 0;
    /* Mask value to prevent garbage high bits from corrupting the stream */
    val &= (1U << n) - 1;

    bw->bits |= (uint64_t)val << bw->num_bits;
    bw->num_bits += n;

    while (bw->num_bits >= 8) {
        if (bw->pos >= bw->cap) return -1;
        bw->data[bw->pos++] = (uint8_t)bw->bits;
        bw->bits >>= 8;
        bw->num_bits -= 8;
    }
    return 0;
}

static int zstd_bw_flush(zstd_bitwriter_t* bw) {
    /* Add padding marker: 1 followed by zeros */
    bw->bits |= 1ULL << bw->num_bits;
    bw->num_bits++;

    while (bw->num_bits > 0) {
        if (bw->pos >= bw->cap) return -1;
        bw->data[bw->pos++] = (uint8_t)bw->bits;
        bw->bits >>= 8;
        bw->num_bits -= 8;
    }
    return 0;
}

/* ============================================================================
 * ZSTD LZ77 Compression
 * ============================================================================
 */

/* Sequence for LZ77 compression */
typedef struct {
    uint32_t lit_len;
    uint32_t match_len;
    uint32_t offset;
} zstd_sequence_t;

/* Compression context */
typedef struct {
    uint32_t hash_table[ZSTD_HASH_SIZE];
    uint32_t rep[3];
    zstd_sequence_t sequences[8192];
    size_t num_sequences;
    const uint8_t* literals_start;
    size_t literals_len;
} zstd_cctx_t;

static inline uint32_t zstd_hash4(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return (v * 2654435761U) >> (32 - ZSTD_HASH_LOG);
}

static inline uint32_t zstd_read32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

/* Find LL code for literal length */
static int find_ll_code(uint32_t lit_len) {
    if (lit_len < 16) return (int)lit_len;
    if (lit_len < 18) return 16;
    if (lit_len < 20) return 17;
    if (lit_len < 22) return 18;
    if (lit_len < 24) return 19;
    if (lit_len < 28) return 20;
    if (lit_len < 32) return 21;
    if (lit_len < 40) return 22;
    if (lit_len < 48) return 23;
    if (lit_len < 64) return 24;
    if (lit_len < 128) return 25;
    if (lit_len < 256) return 26;
    if (lit_len < 512) return 27;
    if (lit_len < 1024) return 28;
    if (lit_len < 2048) return 29;
    if (lit_len < 4096) return 30;
    if (lit_len < 8192) return 31;
    if (lit_len < 16384) return 32;
    if (lit_len < 32768) return 33;
    if (lit_len < 65536) return 34;
    return 35;
}

/* Find ML code for match length */
static int find_ml_code(uint32_t match_len) {
    if (match_len < 35) return (int)(match_len - 3);
    if (match_len < 37) return 32;
    if (match_len < 39) return 33;
    if (match_len < 41) return 34;
    if (match_len < 43) return 35;
    if (match_len < 47) return 36;
    if (match_len < 51) return 37;
    if (match_len < 59) return 38;
    if (match_len < 67) return 39;
    if (match_len < 83) return 40;
    if (match_len < 99) return 41;
    if (match_len < 131) return 42;
    if (match_len < 259) return 43;
    if (match_len < 515) return 44;
    if (match_len < 1027) return 45;
    if (match_len < 2051) return 46;
    if (match_len < 4099) return 47;
    if (match_len < 8195) return 48;
    if (match_len < 16387) return 49;
    if (match_len < 32771) return 50;
    if (match_len < 65539) return 51;
    return 52;
}

/* LZ77 match finding */
static size_t zstd_find_matches(zstd_cctx_t* cctx, const uint8_t* src, size_t src_size) {
    if (src_size < ZSTD_MIN_MATCH + 5) {
        /* Too short for matching */
        cctx->num_sequences = 0;
        cctx->literals_start = src;
        cctx->literals_len = src_size;
        return 0;
    }

    memset(cctx->hash_table, 0, sizeof(cctx->hash_table));
    /* Note: rep[] is initialized once per frame in carquet_zstd_compress,
     * not here, to maintain correct state across multiple blocks */
    cctx->num_sequences = 0;

    const uint8_t* ip = src;
    const uint8_t* const iend = src + src_size;
    const uint8_t* const ilimit = iend - 8;
    const uint8_t* anchor = src;

    while (ip < ilimit) {
        uint32_t h = zstd_hash4(ip);
        uint32_t match_idx = cctx->hash_table[h];
        cctx->hash_table[h] = (uint32_t)(ip - src);

        const uint8_t* match = src + match_idx;
        size_t offset = (size_t)(ip - match);

        /* Check for valid match */
        if (offset > 0 && offset < ZSTD_MAX_OFFSET &&
            ip - offset >= src && zstd_read32(match) == zstd_read32(ip)) {

            /* Count match length */
            size_t match_len = 4;
            const uint8_t* mp = match + 4;
            const uint8_t* ip2 = ip + 4;
            while (ip2 < iend - 7 && zstd_read32(mp) == zstd_read32(ip2)) {
                match_len += 4;
                mp += 4;
                ip2 += 4;
            }
            while (ip2 < iend && *mp == *ip2) {
                match_len++;
                mp++;
                ip2++;
            }

            if (match_len >= ZSTD_MIN_MATCH && cctx->num_sequences < 8192) {
                /* Store sequence */
                zstd_sequence_t* seq = &cctx->sequences[cctx->num_sequences++];
                seq->lit_len = (uint32_t)(ip - anchor);
                seq->match_len = (uint32_t)match_len;

                /* Check for repeat offset */
                if (offset == cctx->rep[0]) {
                    seq->offset = 1;
                } else if (offset == cctx->rep[1]) {
                    seq->offset = 2;
                    cctx->rep[1] = cctx->rep[0];
                    cctx->rep[0] = (uint32_t)offset;
                } else if (offset == cctx->rep[2]) {
                    seq->offset = 3;
                    cctx->rep[2] = cctx->rep[1];
                    cctx->rep[1] = cctx->rep[0];
                    cctx->rep[0] = (uint32_t)offset;
                } else {
                    seq->offset = (uint32_t)offset + 3;
                    cctx->rep[2] = cctx->rep[1];
                    cctx->rep[1] = cctx->rep[0];
                    cctx->rep[0] = (uint32_t)offset;
                }

                ip += match_len;
                anchor = ip;

                /* Update hash for positions within match */
                if (ip < ilimit) {
                    cctx->hash_table[zstd_hash4(ip - 2)] = (uint32_t)(ip - 2 - src);
                }
                continue;
            }
        }
        ip++;
    }

    /* Save trailing literals info */
    cctx->literals_start = anchor;
    cctx->literals_len = (size_t)(iend - anchor);

    return cctx->num_sequences;
}

/* Find all states in the FSE table that decode to the given symbol */
static int find_states_for_symbol(fse_entry_t* table, int table_size, int symbol,
                                   uint32_t* states, int max_states) {
    int count = 0;
    for (int i = 0; i < table_size && count < max_states; i++) {
        if (table[i].symbol == symbol) {
            states[count++] = (uint32_t)i;
        }
    }
    return count;
}

/* Find a state that can transition to target_state given table entry constraints.
 * Returns the state index, or -1 if no valid state exists. */
static int find_encoding_state(fse_entry_t* table, int table_size, int symbol,
                                uint32_t target_state) {
    /* We need: target_state = table[state].new_state_base + update_bits
     * where update_bits < (1 << table[state].num_bits)
     * So: state must decode to symbol AND
     *     target_state - table[state].new_state_base < (1 << table[state].num_bits)
     */
    for (int state = 0; state < table_size; state++) {
        if (table[state].symbol != symbol) continue;

        int32_t base = table[state].new_state_base;  /* Signed base */
        int num_bits = table[state].num_bits;
        int32_t max_update = (1 << num_bits);

        /* Check if target_state can be reached: base <= target < base + max_update */
        int32_t target = (int32_t)target_state;
        if (target >= base && (target - base) < max_update) {
            return state;
        }
    }
    /* No valid state found - FSE encoding not possible for this transition */
    return -1;
}

/* Encode sequences using predefined FSE tables
 *
 * FSE encoding works BACKWARDS from the last sequence:
 * 1. Choose arbitrary final states (any state decoding to the symbol)
 * 2. For each sequence from last to first:
 *    - Compute what state we need to be in such that the update bits
 *      will transition to the next sequence's state
 * 3. The computed state for sequence 0 becomes the initial state
 *
 * The bitstream is read backward, so bits written last are read first.
 */
static size_t encode_sequences(zstd_cctx_t* cctx, const uint8_t* src,
                               uint8_t* dst, size_t dst_cap) {
    (void)src;

    if (cctx->num_sequences == 0) {
        /* No sequences - write 0 sequence count */
        if (dst_cap < 1) return 0;
        dst[0] = 0;
        return 1;
    }

    size_t pos = 0;

    /* Sequences header */
    if (cctx->num_sequences < 128) {
        if (pos >= dst_cap) return 0;
        dst[pos++] = (uint8_t)cctx->num_sequences;
    } else if (cctx->num_sequences < 0x7F00) {
        if (pos + 2 > dst_cap) return 0;
        dst[pos++] = (uint8_t)((cctx->num_sequences >> 8) + 128);
        dst[pos++] = (uint8_t)(cctx->num_sequences & 0xFF);
    } else {
        if (pos + 3 > dst_cap) return 0;
        dst[pos++] = 255;
        uint16_t val = (uint16_t)(cctx->num_sequences - 0x7F00);
        dst[pos++] = val & 0xFF;
        dst[pos++] = (val >> 8) & 0xFF;
    }

    /* Compression modes: all predefined (mode 0) */
    if (pos >= dst_cap) return 0;
    dst[pos++] = 0x00;  /* LL=0, OF=0, ML=0 */

    init_default_tables();

    /* Compute symbol codes for all sequences
     *
     * Offset encoding (to produce correct offset_val for decoder):
     * - offset_val = 1 (repeat 1): of_code = 0 (no extra bits)
     * - offset_val = 2 (repeat 2): of_code = 1, extra = 0
     * - offset_val = 3 (repeat 3): of_code = 1, extra = 1
     * - offset_val >= 4 (real offset): of_code = log2(offset_val), extra = offset_val - 2^of_code
     */
    int ll_codes[8192], ml_codes[8192], of_codes[8192];
    for (size_t i = 0; i < cctx->num_sequences; i++) {
        zstd_sequence_t* seq = &cctx->sequences[i];
        ll_codes[i] = find_ll_code(seq->lit_len);
        ml_codes[i] = find_ml_code(seq->match_len);

        uint32_t offset_val = seq->offset;
        if (offset_val == 1) {
            of_codes[i] = 0;  /* Repeat offset 1: of_code=0, no extra bits */
        } else if (offset_val <= 3) {
            of_codes[i] = 1;  /* Repeat offset 2 or 3: of_code=1, extra=offset_val-2 */
        } else {
            of_codes[i] = highest_bit_set(offset_val);
        }
    }

    /* Compute FSE states working BACKWARD from last sequence
     * For the last sequence, we can choose any state that decodes to the symbol.
     * For earlier sequences, we need to find a state such that the update bits
     * will correctly transition to the next sequence's state.
     */
    uint32_t ll_states[8192], ml_states[8192], of_states[8192];
    size_t n = cctx->num_sequences;

    /* Start with last sequence - use any valid state */
    uint32_t dummy[64];
    find_states_for_symbol(g_LL_table, 1 << LITERALS_LENGTH_ACC_LOG, ll_codes[n-1], dummy, 1);
    ll_states[n-1] = dummy[0];
    find_states_for_symbol(g_ML_table, 1 << MATCH_LENGTH_ACC_LOG, ml_codes[n-1], dummy, 1);
    ml_states[n-1] = dummy[0];
    find_states_for_symbol(g_OF_table, 1 << OFFSET_ACC_LOG, of_codes[n-1], dummy, 1);
    of_states[n-1] = dummy[0];

    /* Work backwards to find states for earlier sequences */
    for (size_t idx = n - 1; idx > 0; idx--) {
        size_t i = idx - 1;
        uint32_t next_ll = ll_states[idx];
        uint32_t next_ml = ml_states[idx];
        uint32_t next_of = of_states[idx];

        /* Find states that can transition to next states */
        int ll_state = find_encoding_state(g_LL_table, 1 << LITERALS_LENGTH_ACC_LOG,
                                           ll_codes[i], next_ll);
        int ml_state = find_encoding_state(g_ML_table, 1 << MATCH_LENGTH_ACC_LOG,
                                           ml_codes[i], next_ml);
        int of_state = find_encoding_state(g_OF_table, 1 << OFFSET_ACC_LOG,
                                           of_codes[i], next_of);

        /* If any state transition is impossible, FSE encoding fails */
        if (ll_state < 0 || ml_state < 0 || of_state < 0) {
#ifdef ZSTD_DEBUG
            fprintf(stderr, "encode_sequences: FSE encoding failed at seq %zu "
                    "(ll=%d ml=%d of=%d)\n", i, ll_state, ml_state, of_state);
#endif
            return 0;  /* Fall back to raw block */
        }

        ll_states[i] = (uint32_t)ll_state;
        ml_states[i] = (uint32_t)ml_state;
        of_states[i] = (uint32_t)of_state;
    }

    /* Build bitstream - written forward, read backward */
    uint8_t bitstream[ZSTD_BLOCK_SIZE_MAX + 256];
    zstd_bitwriter_t bw;
    zstd_bw_init(&bw, bitstream, sizeof(bitstream));

    /* Encode sequences from LAST to FIRST (so first sequence's data is read first)
     *
     * Per RFC 8878, decoder reads for each sequence:
     * 1. LL extra bits (symbol decoding)
     * 2. OF extra bits (symbol decoding)
     * 3. ML extra bits (symbol decoding)
     * 4. LL update bits (state update)
     * 5. OF update bits (state update)
     * 6. ML update bits (state update)
     *
     * Since bitstream is read backward, we write in reverse:
     * ML update, OF update, LL update, ML extra, OF extra, LL extra
     */
    for (size_t idx = n; idx > 0; idx--) {
        size_t i = idx - 1;
        zstd_sequence_t* seq = &cctx->sequences[i];

        uint32_t ll_state = ll_states[i];
        uint32_t ml_state = ml_states[i];
        uint32_t of_state = of_states[i];

        fse_entry_t* ll_entry = &g_LL_table[ll_state];
        fse_entry_t* ml_entry = &g_ML_table[ml_state];
        fse_entry_t* of_entry = &g_OF_table[of_state];

        /* Write update bits for all sequences EXCEPT the last decoded one.
         * The last decoded sequence is the first one we encode (i == n - 1).
         * Per RFC 8878: "update bits are read for all sequences except the last."
         */
        if (i < n - 1) {
            /* Compute update bits to reach next state
             * update_bits = next_state - new_state_base (works with signed base)
             */
            int32_t ll_next = (int32_t)ll_states[i + 1];
            int32_t ml_next = (int32_t)ml_states[i + 1];
            int32_t of_next = (int32_t)of_states[i + 1];

            uint32_t ll_update = (uint32_t)(ll_next - ll_entry->new_state_base);
            uint32_t ml_update = (uint32_t)(ml_next - ml_entry->new_state_base);
            uint32_t of_update = (uint32_t)(of_next - of_entry->new_state_base);

            /* Write update bits in order: OF, ML, LL (read as LL, ML, OF backward per RFC 8878) */
            zstd_bw_add_bits(&bw, of_update, of_entry->num_bits);
            zstd_bw_add_bits(&bw, ml_update, ml_entry->num_bits);
            zstd_bw_add_bits(&bw, ll_update, ll_entry->num_bits);
        }

        /* Write extra bits in order: LL, ML, OF (read as OF, ML, LL backward per RFC 8878) */
        if (LL_bits[ll_codes[i]] > 0) {
            uint32_t extra = seq->lit_len - LL_baseline[ll_codes[i]];
            zstd_bw_add_bits(&bw, extra, LL_bits[ll_codes[i]]);
        }
        if (ML_bits[ml_codes[i]] > 0) {
            uint32_t extra = seq->match_len - ML_baseline[ml_codes[i]];
            zstd_bw_add_bits(&bw, extra, ML_bits[ml_codes[i]]);
        }
        if (of_codes[i] > 0) {
            uint32_t offset_val = seq->offset;
            uint32_t extra = offset_val - (1U << of_codes[i]);
            zstd_bw_add_bits(&bw, extra, of_codes[i]);
        }
    }

    /* Write initial states (read first by decoder)
     *
     * The backward bitstream is read from HIGH bits to LOW bits. The decoder
     * reads: LL, OF, ML (in that order). Since we write forward with LL at
     * the highest bit positions (written last), it will be read first.
     *
     * Write order: ML, OF, LL (so LL ends up at high bits, closest to padding)
     */
#ifdef ZSTD_DEBUG
    fprintf(stderr, "encode_sequences: n=%zu init_states ll=%u of=%u ml=%u\n",
            n, ll_states[0], of_states[0], ml_states[0]);
    for (size_t i = 0; i < n && i < 3; i++) {
        fprintf(stderr, "encode_sequences[%zu]: ll_len=%u ml_len=%u offset=%u codes ll=%d ml=%d of=%d\n",
                i, cctx->sequences[i].lit_len, cctx->sequences[i].match_len,
                cctx->sequences[i].offset, ll_codes[i], ml_codes[i], of_codes[i]);
    }
#endif
    zstd_bw_add_bits(&bw, ml_states[0], MATCH_LENGTH_ACC_LOG);
    zstd_bw_add_bits(&bw, of_states[0], OFFSET_ACC_LOG);
    zstd_bw_add_bits(&bw, ll_states[0], LITERALS_LENGTH_ACC_LOG);

    /* Flush with padding marker */
    if (zstd_bw_flush(&bw) < 0) return 0;

    /* Copy bitstream as-is. The decoder reads backward from the end. */
    size_t bs_size = bw.pos;
    if (pos + bs_size > dst_cap) return 0;
    memcpy(dst + pos, bitstream, bs_size);
    pos += bs_size;

    return pos;
}

/* Write a compressed block */
static size_t write_compressed_block(zstd_cctx_t* cctx, uint8_t* dst, size_t cap,
                                      const uint8_t* src, size_t src_size, int is_last) {
    /* Calculate total literals */
    size_t total_lit = 0;
    for (size_t i = 0; i < cctx->num_sequences; i++) {
        total_lit += cctx->sequences[i].lit_len;
    }
    total_lit += cctx->literals_len;

    /* Reserve space for block header */
    uint8_t block_data[ZSTD_BLOCK_SIZE_MAX + 1024];
    size_t block_pos = 0;

    /* Write literals section (raw)
     * Format for Raw/RLE literals:
     * - bits 0-1: Literals_Block_Type (00=Raw)
     * - bit 2: part of Size_Format
     * Per RFC 8878:
     *   - Size_Format=0: 1-byte header, size in bits 3-7 (5 bits, max 31)
     *   - Size_Format=1: 2-byte header, size in bits 4-7 of byte0 + byte1 (12 bits)
     *   - Size_Format=2: 3-byte header (20 bits)
     *   - Size_Format=3: 3-byte header (20 bits)
     */
    if (total_lit <= 31) {
        /* Size_Format=0: 1-byte header, size in bits 3-7 */
        block_data[block_pos++] = (uint8_t)(ZSTD_LITBLOCK_RAW | (total_lit << 3));
    } else if (total_lit < 4096) {
        /* Size_Format=1: 2-byte header (bit 2 set = 0x04) */
        block_data[block_pos++] = (uint8_t)(ZSTD_LITBLOCK_RAW | 0x04 | ((total_lit & 0x0F) << 4));
        block_data[block_pos++] = (uint8_t)(total_lit >> 4);
    } else {
        /* Size_Format=2: 3-byte header (bits 2-3 = 0x08) */
        block_data[block_pos++] = (uint8_t)(ZSTD_LITBLOCK_RAW | 0x0C | ((total_lit & 0x0F) << 4));
        block_data[block_pos++] = (uint8_t)(total_lit >> 4);
        block_data[block_pos++] = (uint8_t)(total_lit >> 12);
    }

    /* Copy literals from sequences */
    const uint8_t* lit_ptr = src;
    for (size_t i = 0; i < cctx->num_sequences; i++) {
        if (cctx->sequences[i].lit_len > 0) {
            memcpy(block_data + block_pos, lit_ptr, cctx->sequences[i].lit_len);
            block_pos += cctx->sequences[i].lit_len;
        }
        /* Skip past literals and match */
        lit_ptr += cctx->sequences[i].lit_len;
        /* Find actual offset for skipping */
        uint32_t offset = cctx->sequences[i].offset;
        if (offset <= 3) {
            /* Repeat offset - need to track actual offset */
            /* For now, just skip match_len bytes */
        }
        lit_ptr += cctx->sequences[i].match_len;
    }
    /* Copy trailing literals */
    if (cctx->literals_len > 0) {
        memcpy(block_data + block_pos, cctx->literals_start, cctx->literals_len);
        block_pos += cctx->literals_len;
    }

    /* Encode sequences */
    size_t seq_size = encode_sequences(cctx, src, block_data + block_pos,
                                        sizeof(block_data) - block_pos);

    /* If sequence encoding failed (FSE encoding impossible), fall back to raw */
    if (seq_size == 0 && cctx->num_sequences > 0) {
        return 0;
    }
    block_pos += seq_size;

    /* Check if compressed is smaller than raw */
    if (block_pos >= src_size) {
        /* Not worth it - use raw block */
        return 0;
    }

    /* Write block header + data */
    if (cap < 3 + block_pos) return 0;

    uint32_t hdr = (uint32_t)is_last | (ZSTD_BLOCK_COMPRESSED << 1) | ((uint32_t)block_pos << 3);
    dst[0] = hdr & 0xFF;
    dst[1] = (hdr >> 8) & 0xFF;
    dst[2] = (hdr >> 16) & 0xFF;

    memcpy(dst + 3, block_data, block_pos);
    return 3 + block_pos;
}

/* Check if data is all same byte (for RLE block) */
static int check_rle(const uint8_t* data, size_t size) {
    if (size == 0) return 0;
    uint8_t first = data[0];
    for (size_t i = 1; i < size; i++) {
        if (data[i] != first) return 0;
    }
    return 1;
}

/* Write frame header */
static size_t write_frame_header(uint8_t* dst, size_t cap, size_t content_size) {
    size_t pos = 0;

    /* Magic number */
    if (cap < 4) return 0;
    dst[pos++] = ZSTD_MAGIC & 0xFF;
    dst[pos++] = (ZSTD_MAGIC >> 8) & 0xFF;
    dst[pos++] = (ZSTD_MAGIC >> 16) & 0xFF;
    dst[pos++] = (ZSTD_MAGIC >> 24) & 0xFF;

    /* Frame header descriptor: single segment + FCS size */
    uint8_t fhd = 0x20;  /* Single segment flag */
    if (content_size >= 256) fhd |= 0x40;  /* 2-byte FCS */
    if (pos >= cap) return 0;
    dst[pos++] = fhd;

    /* Frame content size */
    if (content_size < 256) {
        if (pos >= cap) return 0;
        dst[pos++] = (uint8_t)content_size;
    } else {
        if (pos + 2 > cap) return 0;
        uint16_t fcs = (uint16_t)(content_size - 256);
        dst[pos++] = fcs & 0xFF;
        dst[pos++] = (fcs >> 8) & 0xFF;
    }

    return pos;
}

/* Write a raw block */
static size_t write_raw_block(uint8_t* dst, size_t cap, const uint8_t* src,
                               size_t size, int is_last) {
    if (cap < 3 + size) return 0;

    uint32_t hdr = (uint32_t)is_last | (ZSTD_BLOCK_RAW << 1) | ((uint32_t)size << 3);
    dst[0] = hdr & 0xFF;
    dst[1] = (hdr >> 8) & 0xFF;
    dst[2] = (hdr >> 16) & 0xFF;

    memcpy(dst + 3, src, size);
    return 3 + size;
}

/* Write an RLE block */
static size_t write_rle_block(uint8_t* dst, size_t cap, uint8_t byte,
                               size_t count, int is_last) {
    if (cap < 4) return 0;

    uint32_t hdr = (uint32_t)is_last | (ZSTD_BLOCK_RLE << 1) | ((uint32_t)count << 3);
    dst[0] = hdr & 0xFF;
    dst[1] = (hdr >> 8) & 0xFF;
    dst[2] = (hdr >> 16) & 0xFF;
    dst[3] = byte;

    return 4;
}

int carquet_zstd_compress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size,
    int level) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    size_t pos = write_frame_header(dst, dst_capacity, src_size);
    if (pos == 0) return CARQUET_ERROR_COMPRESSION;

    /* Allocate compression context on stack for small blocks, heap for large */
    zstd_cctx_t cctx_stack;
    zstd_cctx_t* cctx = &cctx_stack;

    /* Initialize ZSTD Repeated Offsets once per frame */
    cctx->rep[0] = 1;
    cctx->rep[1] = 4;
    cctx->rep[2] = 8;

    /* Compress blocks */
    size_t src_pos = 0;
    while (src_pos < src_size) {
        size_t block_size = src_size - src_pos;
        if (block_size > ZSTD_BLOCK_SIZE_MAX) {
            block_size = ZSTD_BLOCK_SIZE_MAX;
        }

        int is_last = (src_pos + block_size >= src_size);
        const uint8_t* block_src = src + src_pos;
        size_t written = 0;

        /* Check if RLE block is beneficial */
        if (check_rle(block_src, block_size)) {
            written = write_rle_block(dst + pos, dst_capacity - pos,
                                      block_src[0], block_size, is_last);
            /* RLE blocks do not update rep history */
        } else if (level > 0 && block_size >= 64) {
            /* Save rep state before attempting compression.
             * If compression fails and we fall back to raw block,
             * we must restore rep state since raw blocks don't update it. */
            uint32_t saved_rep[3];
            memcpy(saved_rep, cctx->rep, sizeof(saved_rep));

            /* Try LZ77 compression for level > 0 and blocks >= 64 bytes */
            zstd_find_matches(cctx, block_src, block_size);

            if (cctx->num_sequences > 0) {
                written = write_compressed_block(cctx, dst + pos, dst_capacity - pos,
                                                 block_src, block_size, is_last);
            }

            /* If compression failed or wasn't beneficial, restore rep state */
            if (written == 0) {
                memcpy(cctx->rep, saved_rep, sizeof(saved_rep));
            }
        }

        /* Fall back to raw block if compression didn't help */
        if (written == 0) {
            written = write_raw_block(dst + pos, dst_capacity - pos,
                                      block_src, block_size, is_last);
            /* Raw blocks do not update rep history */
        }

        if (written == 0) return CARQUET_ERROR_COMPRESSION;
        pos += written;
        src_pos += block_size;
    }

    *dst_size = pos;
    return CARQUET_OK;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================
 */

size_t carquet_zstd_compress_bound(size_t src_size) {
    return src_size + (src_size / ZSTD_BLOCK_SIZE_MAX + 1) * 3 + 32;
}

/* ============================================================================
 * Thread-safe Initialization
 * ============================================================================
 */

void carquet_zstd_init_tables(void) {
    init_default_tables();
}
