/**
 * @file dispatch.c
 * @brief SIMD function dispatch
 *
 * This file provides runtime dispatch for SIMD-optimized functions based on
 * detected CPU features. Functions are selected at initialization time and
 * stored in function pointer tables for efficient runtime access.
 */

#include <carquet/carquet.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ============================================================================
 * Function Pointer Types
 * ============================================================================
 */

typedef void (*prefix_sum_i32_fn)(int32_t* values, int64_t count, int32_t initial);
typedef void (*prefix_sum_i64_fn)(int64_t* values, int64_t count, int64_t initial);

typedef void (*gather_i32_fn)(const int32_t* dict, const uint32_t* indices,
                               int64_t count, int32_t* output);
typedef void (*gather_i64_fn)(const int64_t* dict, const uint32_t* indices,
                               int64_t count, int64_t* output);
typedef void (*gather_float_fn)(const float* dict, const uint32_t* indices,
                                 int64_t count, float* output);
typedef void (*gather_double_fn)(const double* dict, const uint32_t* indices,
                                  int64_t count, double* output);

typedef void (*byte_split_encode_float_fn)(const float* values, int64_t count,
                                            uint8_t* output);
typedef void (*byte_split_decode_float_fn)(const uint8_t* data, int64_t count,
                                            float* values);
typedef void (*byte_split_encode_double_fn)(const double* values, int64_t count,
                                             uint8_t* output);
typedef void (*byte_split_decode_double_fn)(const uint8_t* data, int64_t count,
                                             double* values);

typedef void (*memset_fn)(void* dest, uint8_t value, size_t n);
typedef void (*memcpy_fn)(void* dest, const void* src, size_t n);

typedef void (*unpack_bools_fn)(const uint8_t* input, uint8_t* output, int64_t count);
typedef void (*pack_bools_fn)(const uint8_t* input, uint8_t* output, int64_t count);

typedef int64_t (*find_run_length_i32_fn)(const int32_t* values, int64_t count);

typedef uint32_t (*crc32c_fn)(uint32_t crc, const uint8_t* data, size_t len);

/* ============================================================================
 * Scalar Fallback Implementations
 * ============================================================================
 */

static void scalar_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial) {
    int32_t sum = initial;
    for (int64_t i = 0; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

static void scalar_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial) {
    int64_t sum = initial;
    for (int64_t i = 0; i < count; i++) {
        sum += values[i];
        values[i] = sum;
    }
}

static void scalar_gather_i32(const int32_t* dict, const uint32_t* indices,
                               int64_t count, int32_t* output) {
    for (int64_t i = 0; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

static void scalar_gather_i64(const int64_t* dict, const uint32_t* indices,
                               int64_t count, int64_t* output) {
    for (int64_t i = 0; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

static void scalar_gather_float(const float* dict, const uint32_t* indices,
                                 int64_t count, float* output) {
    for (int64_t i = 0; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

static void scalar_gather_double(const double* dict, const uint32_t* indices,
                                  int64_t count, double* output) {
    for (int64_t i = 0; i < count; i++) {
        output[i] = dict[indices[i]];
    }
}

static void scalar_byte_split_encode_float(const float* values, int64_t count,
                                            uint8_t* output) {
    const uint8_t* src = (const uint8_t*)values;
    for (int64_t i = 0; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            output[b * count + i] = src[i * 4 + b];
        }
    }
}

static void scalar_byte_split_decode_float(const uint8_t* data, int64_t count,
                                            float* values) {
    uint8_t* dst = (uint8_t*)values;
    for (int64_t i = 0; i < count; i++) {
        for (int b = 0; b < 4; b++) {
            dst[i * 4 + b] = data[b * count + i];
        }
    }
}

static void scalar_byte_split_encode_double(const double* values, int64_t count,
                                             uint8_t* output) {
    const uint8_t* src = (const uint8_t*)values;
    for (int64_t i = 0; i < count; i++) {
        for (int b = 0; b < 8; b++) {
            output[b * count + i] = src[i * 8 + b];
        }
    }
}

static void scalar_byte_split_decode_double(const uint8_t* data, int64_t count,
                                             double* values) {
    uint8_t* dst = (uint8_t*)values;
    for (int64_t i = 0; i < count; i++) {
        for (int b = 0; b < 8; b++) {
            dst[i * 8 + b] = data[b * count + i];
        }
    }
}

static void scalar_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        int byte_idx = (int)(i / 8);
        int bit_idx = (int)(i % 8);
        output[i] = (input[byte_idx] >> bit_idx) & 1;
    }
}

static void scalar_pack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    for (int64_t i = 0; i < count; i += 8) {
        uint8_t byte = 0;
        for (int64_t j = 0; j < 8 && i + j < count; j++) {
            if (input[i + j]) {
                byte |= (1 << j);
            }
        }
        output[i / 8] = byte;
    }
}

static int64_t scalar_find_run_length_i32(const int32_t* values, int64_t count) {
    if (count == 0) return 0;
    int32_t first = values[0];
    for (int64_t i = 1; i < count; i++) {
        if (values[i] != first) return i;
    }
    return count;
}

/* CRC32C using software table - Castagnoli polynomial */
static const uint32_t crc32c_table[256] = {
    0x00000000, 0xF26B8303, 0xE13B70F7, 0x1350F3F4, 0xC79A971F, 0x35F1141C, 0x26A1E7E8, 0xD4CA64EB,
    0x8AD958CF, 0x78B2DBCC, 0x6BE22838, 0x9989AB3B, 0x4D43CFD0, 0xBF284CD3, 0xAC78BF27, 0x5E133C24,
    0x105EC76F, 0xE235446C, 0xF165B798, 0x030E349B, 0xD7C45070, 0x25AFD373, 0x36FF2087, 0xC494A384,
    0x9A879FA0, 0x68EC1CA3, 0x7BBCEF57, 0x89D76C54, 0x5D1D08BF, 0xAF768BBC, 0xBC267848, 0x4E4DFB4B,
    0x20BD8EDE, 0xD2D60DDD, 0xC186FE29, 0x33ED7D2A, 0xE72719C1, 0x154C9AC2, 0x061C6936, 0xF477EA35,
    0xAA64D611, 0x580F5512, 0x4B5FA6E6, 0xB93425E5, 0x6DFE410E, 0x9F95C20D, 0x8CC531F9, 0x7EAEB2FA,
    0x30E349B1, 0xC288CAB2, 0xD1D83946, 0x23B3BA45, 0xF779DEAE, 0x05125DAD, 0x1642AE59, 0xE4292D5A,
    0xBA3A117E, 0x4851927D, 0x5B016189, 0xA96AE28A, 0x7DA08661, 0x8FCB0562, 0x9C9BF696, 0x6EF07595,
    0x417B1DBC, 0xB3109EBF, 0xA0406D4B, 0x522BEE48, 0x86E18AA3, 0x748A09A0, 0x67DAFA54, 0x95B17957,
    0xCBA24573, 0x39C9C670, 0x2A993584, 0xD8F2B687, 0x0C38D26C, 0xFE53516F, 0xED03A29B, 0x1F682198,
    0x5125DAD3, 0xA34E59D0, 0xB01EAA24, 0x42752927, 0x96BF4DCC, 0x64D4CECF, 0x77843D3B, 0x85EFBE38,
    0xDBFC821C, 0x2997011F, 0x3AC7F2EB, 0xC8AC71E8, 0x1C661503, 0xEE0D9600, 0xFD5D65F4, 0x0F36E6F7,
    0x61C69362, 0x93AD1061, 0x80FDE395, 0x72966096, 0xA65C047D, 0x5437877E, 0x4767748A, 0xB50CF789,
    0xEB1FCBAD, 0x197448AE, 0x0A24BB5A, 0xF84F3859, 0x2C855CB2, 0xDEEEDFB1, 0xCDBE2C45, 0x3FD5AF46,
    0x7198540D, 0x83F3D70E, 0x90A324FA, 0x62C8A7F9, 0xB602C312, 0x44694011, 0x5739B3E5, 0xA55230E6,
    0xFB410CC2, 0x092A8FC1, 0x1A7A7C35, 0xE811FF36, 0x3CDB9BDD, 0xCEB018DE, 0xDDE0EB2A, 0x2F8B6829,
    0x82F63B78, 0x709DB87B, 0x63CD4B8F, 0x91A6C88C, 0x456CAC67, 0xB7072F64, 0xA457DC90, 0x563C5F93,
    0x082F63B7, 0xFA44E0B4, 0xE9141340, 0x1B7F9043, 0xCFB5F4A8, 0x3DDE77AB, 0x2E8E845F, 0xDCE5075C,
    0x92A8FC17, 0x60C37F14, 0x73938CE0, 0x81F80FE3, 0x55326B08, 0xA759E80B, 0xB4091BFF, 0x466298FC,
    0x1871A4D8, 0xEA1A27DB, 0xF94AD42F, 0x0B21572C, 0xDFEB33C7, 0x2D80B0C4, 0x3ED04330, 0xCCBBC033,
    0xA24BB5A6, 0x502036A5, 0x4370C551, 0xB11B4652, 0x65D122B9, 0x97BAA1BA, 0x84EA524E, 0x7681D14D,
    0x2892ED69, 0xDAF96E6A, 0xC9A99D9E, 0x3BC21E9D, 0xEF087A76, 0x1D63F975, 0x0E330A81, 0xFC588982,
    0xB21572C9, 0x407EF1CA, 0x532E023E, 0xA145813D, 0x758FE5D6, 0x87E466D5, 0x94B49521, 0x66DF1622,
    0x38CC2A06, 0xCAA7A905, 0xD9F75AF1, 0x2B9CD9F2, 0xFF56BD19, 0x0D3D3E1A, 0x1E6DCDEE, 0xEC064EED,
    0xC38D26C4, 0x31E6A5C7, 0x22B65633, 0xD0DDD530, 0x0417B1DB, 0xF67C32D8, 0xE52CC12C, 0x1747422F,
    0x49547E0B, 0xBB3FFD08, 0xA86F0EFC, 0x5A048DFF, 0x8ECEE914, 0x7CA56A17, 0x6FF599E3, 0x9D9E1AE0,
    0xD3D3E1AB, 0x21B862A8, 0x32E8915C, 0xC083125F, 0x144976B4, 0xE622F5B7, 0xF5720643, 0x07198540,
    0x590AB964, 0xAB613A67, 0xB831C993, 0x4A5A4A90, 0x9E902E7B, 0x6CFBAD78, 0x7FAB5E8C, 0x8DC0DD8F,
    0xE330A81A, 0x115B2B19, 0x020BD8ED, 0xF0605BEE, 0x24AA3F05, 0xD6C1BC06, 0xC5914FF2, 0x37FACCF1,
    0x69E9F0D5, 0x9B8273D6, 0x88D28022, 0x7AB90321, 0xAE7367CA, 0x5C18E4C9, 0x4F48173D, 0xBD23943E,
    0xF36E6F75, 0x0105EC76, 0x12551F82, 0xE03E9C81, 0x34F4F86A, 0xC69F7B69, 0xD5CF889D, 0x27A40B9E,
    0x79B737BA, 0x8BDCB4B9, 0x988C474D, 0x6AE7C44E, 0xBE2DA0A5, 0x4C4623A6, 0x5F16D052, 0xAD7D5351
};

static uint32_t scalar_crc32c(uint32_t crc, const uint8_t* data, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc = crc32c_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

/* ============================================================================
 * External SIMD Function Declarations
 * ============================================================================
 */

/* Use CMake defines instead of compiler intrinsic macros, since dispatch.c
 * is not compiled with -msse4.2/-mavx2/-mavx512f flags */
#if defined(CARQUET_ARCH_X86)

#ifdef CARQUET_ENABLE_SSE
extern void carquet_sse_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial);
extern void carquet_sse_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial);
extern void carquet_sse_gather_i32(const int32_t* dict, const uint32_t* indices,
                                    int64_t count, int32_t* output);
extern void carquet_sse_gather_float(const float* dict, const uint32_t* indices,
                                      int64_t count, float* output);
extern void carquet_sse_byte_stream_split_encode_float(const float* values, int64_t count,
                                                        uint8_t* output);
extern void carquet_sse_byte_stream_split_decode_float(const uint8_t* data, int64_t count,
                                                        float* values);
extern void carquet_sse_byte_stream_split_encode_double(const double* values, int64_t count,
                                                         uint8_t* output);
extern void carquet_sse_byte_stream_split_decode_double(const uint8_t* data, int64_t count,
                                                         double* values);
extern void carquet_sse_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count);
extern void carquet_sse_pack_bools(const uint8_t* input, uint8_t* output, int64_t count);
extern uint32_t carquet_sse_crc32c(uint32_t crc, const uint8_t* data, size_t len);
#endif

#ifdef CARQUET_ENABLE_AVX2
extern void carquet_avx2_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial);
extern void carquet_avx2_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial);
extern void carquet_avx2_gather_i32(const int32_t* dict, const uint32_t* indices,
                                     int64_t count, int32_t* output);
extern void carquet_avx2_gather_i64(const int64_t* dict, const uint32_t* indices,
                                     int64_t count, int64_t* output);
extern void carquet_avx2_gather_float(const float* dict, const uint32_t* indices,
                                       int64_t count, float* output);
extern void carquet_avx2_gather_double(const double* dict, const uint32_t* indices,
                                        int64_t count, double* output);
extern void carquet_avx2_byte_stream_split_encode_float(const float* values, int64_t count,
                                                         uint8_t* output);
extern void carquet_avx2_byte_stream_split_decode_float(const uint8_t* data, int64_t count,
                                                         float* values);
extern void carquet_avx2_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count);
extern void carquet_avx2_pack_bools(const uint8_t* input, uint8_t* output, int64_t count);
extern int64_t carquet_avx2_find_run_length_i32(const int32_t* values, int64_t count);
#endif

#ifdef CARQUET_ENABLE_AVX512
extern void carquet_avx512_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial);
extern void carquet_avx512_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial);
extern void carquet_avx512_gather_i32(const int32_t* dict, const uint32_t* indices,
                                       int64_t count, int32_t* output);
extern void carquet_avx512_gather_i64(const int64_t* dict, const uint32_t* indices,
                                       int64_t count, int64_t* output);
extern void carquet_avx512_gather_float(const float* dict, const uint32_t* indices,
                                         int64_t count, float* output);
extern void carquet_avx512_gather_double(const double* dict, const uint32_t* indices,
                                          int64_t count, double* output);
extern void carquet_avx512_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count);
extern void carquet_avx512_pack_bools(const uint8_t* input, uint8_t* output, int64_t count);
extern int64_t carquet_avx512_find_run_length_i32(const int32_t* values, int64_t count);
#endif

#endif /* CARQUET_ARCH_X86 */

#if defined(__aarch64__)

/* NEON declarations - always available on AArch64 */
#ifdef __ARM_NEON
extern void carquet_neon_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial);
extern void carquet_neon_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial);
extern void carquet_neon_gather_i32(const int32_t* dict, const uint32_t* indices,
                                     int64_t count, int32_t* output);
extern void carquet_neon_gather_i64(const int64_t* dict, const uint32_t* indices,
                                     int64_t count, int64_t* output);
extern void carquet_neon_gather_float(const float* dict, const uint32_t* indices,
                                       int64_t count, float* output);
extern void carquet_neon_gather_double(const double* dict, const uint32_t* indices,
                                        int64_t count, double* output);
extern void carquet_neon_byte_stream_split_encode_float(const float* values, int64_t count,
                                                         uint8_t* output);
extern void carquet_neon_byte_stream_split_decode_float(const uint8_t* data, int64_t count,
                                                         float* values);
extern void carquet_neon_byte_stream_split_encode_double(const double* values, int64_t count,
                                                          uint8_t* output);
extern void carquet_neon_byte_stream_split_decode_double(const uint8_t* data, int64_t count,
                                                          double* values);
extern void carquet_neon_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count);
extern void carquet_neon_pack_bools(const uint8_t* input, uint8_t* output, int64_t count);
extern int64_t carquet_neon_find_run_length_i32(const int32_t* values, int64_t count);
extern uint32_t carquet_neon_crc32c(uint32_t crc, const uint8_t* data, size_t len);
#endif

#ifdef __ARM_FEATURE_SVE
extern void carquet_sve_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial);
extern void carquet_sve_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial);
extern void carquet_sve_gather_i32(const int32_t* dict, const uint32_t* indices,
                                    int64_t count, int32_t* output);
extern void carquet_sve_gather_i64(const int64_t* dict, const uint32_t* indices,
                                    int64_t count, int64_t* output);
extern void carquet_sve_gather_float(const float* dict, const uint32_t* indices,
                                      int64_t count, float* output);
extern void carquet_sve_gather_double(const double* dict, const uint32_t* indices,
                                       int64_t count, double* output);
#endif

#endif /* AArch64 */

/* ============================================================================
 * Dispatch Table
 * ============================================================================
 */

typedef struct {
    prefix_sum_i32_fn prefix_sum_i32;
    prefix_sum_i64_fn prefix_sum_i64;
    gather_i32_fn gather_i32;
    gather_i64_fn gather_i64;
    gather_float_fn gather_float;
    gather_double_fn gather_double;
    byte_split_encode_float_fn byte_split_encode_float;
    byte_split_decode_float_fn byte_split_decode_float;
    byte_split_encode_double_fn byte_split_encode_double;
    byte_split_decode_double_fn byte_split_decode_double;
    unpack_bools_fn unpack_bools;
    pack_bools_fn pack_bools;
    find_run_length_i32_fn find_run_length_i32;
    crc32c_fn crc32c;
} carquet_simd_dispatch_t;

static carquet_simd_dispatch_t g_dispatch = {0};
static int g_dispatch_initialized = 0;

/* ============================================================================
 * Dispatch Initialization
 * ============================================================================
 */

void carquet_simd_dispatch_init(void) {
    if (g_dispatch_initialized) {
        return;
    }

    const carquet_cpu_info_t* cpu = carquet_get_cpu_info();
    (void)cpu;  /* May be unused on some platforms */

    /* Start with scalar fallbacks */
    g_dispatch.prefix_sum_i32 = scalar_prefix_sum_i32;
    g_dispatch.prefix_sum_i64 = scalar_prefix_sum_i64;
    g_dispatch.gather_i32 = scalar_gather_i32;
    g_dispatch.gather_i64 = scalar_gather_i64;
    g_dispatch.gather_float = scalar_gather_float;
    g_dispatch.gather_double = scalar_gather_double;
    g_dispatch.byte_split_encode_float = scalar_byte_split_encode_float;
    g_dispatch.byte_split_decode_float = scalar_byte_split_decode_float;
    g_dispatch.byte_split_encode_double = scalar_byte_split_encode_double;
    g_dispatch.byte_split_decode_double = scalar_byte_split_decode_double;
    g_dispatch.unpack_bools = scalar_unpack_bools;
    g_dispatch.pack_bools = scalar_pack_bools;
    g_dispatch.find_run_length_i32 = scalar_find_run_length_i32;
    g_dispatch.crc32c = scalar_crc32c;

#if defined(CARQUET_ARCH_X86)

#ifdef CARQUET_ENABLE_SSE
    if (cpu->has_sse42) {
        g_dispatch.prefix_sum_i32 = carquet_sse_prefix_sum_i32;
        g_dispatch.prefix_sum_i64 = carquet_sse_prefix_sum_i64;
        g_dispatch.gather_i32 = carquet_sse_gather_i32;
        g_dispatch.gather_float = carquet_sse_gather_float;
        g_dispatch.byte_split_encode_float = carquet_sse_byte_stream_split_encode_float;
        g_dispatch.byte_split_decode_float = carquet_sse_byte_stream_split_decode_float;
        g_dispatch.byte_split_encode_double = carquet_sse_byte_stream_split_encode_double;
        g_dispatch.byte_split_decode_double = carquet_sse_byte_stream_split_decode_double;
        g_dispatch.unpack_bools = carquet_sse_unpack_bools;
        g_dispatch.pack_bools = carquet_sse_pack_bools;
        g_dispatch.crc32c = carquet_sse_crc32c;
    }
#endif

#ifdef CARQUET_ENABLE_AVX2
    if (cpu->has_avx2) {
        g_dispatch.prefix_sum_i32 = carquet_avx2_prefix_sum_i32;
        g_dispatch.prefix_sum_i64 = carquet_avx2_prefix_sum_i64;
        g_dispatch.gather_i32 = carquet_avx2_gather_i32;
        g_dispatch.gather_i64 = carquet_avx2_gather_i64;
        g_dispatch.gather_float = carquet_avx2_gather_float;
        g_dispatch.gather_double = carquet_avx2_gather_double;
        g_dispatch.byte_split_encode_float = carquet_avx2_byte_stream_split_encode_float;
        g_dispatch.byte_split_decode_float = carquet_avx2_byte_stream_split_decode_float;
        g_dispatch.unpack_bools = carquet_avx2_unpack_bools;
        g_dispatch.pack_bools = carquet_avx2_pack_bools;
        g_dispatch.find_run_length_i32 = carquet_avx2_find_run_length_i32;
    }
#endif

#ifdef CARQUET_ENABLE_AVX512
    if (cpu->has_avx512f) {
        g_dispatch.prefix_sum_i32 = carquet_avx512_prefix_sum_i32;
        g_dispatch.prefix_sum_i64 = carquet_avx512_prefix_sum_i64;
        g_dispatch.gather_i32 = carquet_avx512_gather_i32;
        g_dispatch.gather_i64 = carquet_avx512_gather_i64;
        g_dispatch.gather_float = carquet_avx512_gather_float;
        g_dispatch.gather_double = carquet_avx512_gather_double;
        g_dispatch.unpack_bools = carquet_avx512_unpack_bools;
        g_dispatch.pack_bools = carquet_avx512_pack_bools;
        g_dispatch.find_run_length_i32 = carquet_avx512_find_run_length_i32;
    }
#endif

#endif /* CARQUET_ARCH_X86 */

#if defined(__aarch64__)

    /* NEON is always available on AArch64 - register NEON functions first */
#ifdef __ARM_NEON
    /* NEON optimized functions - always enabled on AArch64 */
    g_dispatch.prefix_sum_i32 = carquet_neon_prefix_sum_i32;
    g_dispatch.prefix_sum_i64 = carquet_neon_prefix_sum_i64;
    g_dispatch.gather_i32 = carquet_neon_gather_i32;
    g_dispatch.gather_i64 = carquet_neon_gather_i64;
    g_dispatch.gather_float = carquet_neon_gather_float;
    g_dispatch.gather_double = carquet_neon_gather_double;
    g_dispatch.byte_split_encode_float = carquet_neon_byte_stream_split_encode_float;
    g_dispatch.byte_split_decode_float = carquet_neon_byte_stream_split_decode_float;
    g_dispatch.byte_split_encode_double = carquet_neon_byte_stream_split_encode_double;
    g_dispatch.byte_split_decode_double = carquet_neon_byte_stream_split_decode_double;
    g_dispatch.unpack_bools = carquet_neon_unpack_bools;
    g_dispatch.pack_bools = carquet_neon_pack_bools;
    g_dispatch.find_run_length_i32 = carquet_neon_find_run_length_i32;
    g_dispatch.crc32c = carquet_neon_crc32c;
#endif

    /* SVE overrides NEON if available (better performance on supporting hardware) */
#ifdef __ARM_FEATURE_SVE
    if (cpu->has_sve) {
        g_dispatch.prefix_sum_i32 = carquet_sve_prefix_sum_i32;
        g_dispatch.prefix_sum_i64 = carquet_sve_prefix_sum_i64;
        g_dispatch.gather_i32 = carquet_sve_gather_i32;
        g_dispatch.gather_i64 = carquet_sve_gather_i64;
        g_dispatch.gather_float = carquet_sve_gather_float;
        g_dispatch.gather_double = carquet_sve_gather_double;
    }
#endif

#endif /* AArch64 */

    g_dispatch_initialized = 1;
}

/* ============================================================================
 * Public Dispatch Functions
 * ============================================================================
 */

void carquet_dispatch_prefix_sum_i32(int32_t* values, int64_t count, int32_t initial) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.prefix_sum_i32(values, count, initial);
}

void carquet_dispatch_prefix_sum_i64(int64_t* values, int64_t count, int64_t initial) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.prefix_sum_i64(values, count, initial);
}

void carquet_dispatch_gather_i32(const int32_t* dict, const uint32_t* indices,
                                  int64_t count, int32_t* output) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.gather_i32(dict, indices, count, output);
}

void carquet_dispatch_gather_i64(const int64_t* dict, const uint32_t* indices,
                                  int64_t count, int64_t* output) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.gather_i64(dict, indices, count, output);
}

void carquet_dispatch_gather_float(const float* dict, const uint32_t* indices,
                                    int64_t count, float* output) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.gather_float(dict, indices, count, output);
}

void carquet_dispatch_gather_double(const double* dict, const uint32_t* indices,
                                     int64_t count, double* output) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.gather_double(dict, indices, count, output);
}

void carquet_dispatch_byte_split_encode_float(const float* values, int64_t count,
                                               uint8_t* output) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.byte_split_encode_float(values, count, output);
}

void carquet_dispatch_byte_split_decode_float(const uint8_t* data, int64_t count,
                                               float* values) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.byte_split_decode_float(data, count, values);
}

void carquet_dispatch_byte_split_encode_double(const double* values, int64_t count,
                                                uint8_t* output) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.byte_split_encode_double(values, count, output);
}

void carquet_dispatch_byte_split_decode_double(const uint8_t* data, int64_t count,
                                                double* values) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.byte_split_decode_double(data, count, values);
}

void carquet_dispatch_unpack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.unpack_bools(input, output, count);
}

void carquet_dispatch_pack_bools(const uint8_t* input, uint8_t* output, int64_t count) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    g_dispatch.pack_bools(input, output, count);
}

int64_t carquet_dispatch_find_run_length_i32(const int32_t* values, int64_t count) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    return g_dispatch.find_run_length_i32(values, count);
}

uint32_t carquet_dispatch_crc32c(uint32_t crc, const uint8_t* data, size_t len) {
    if (!g_dispatch_initialized) carquet_simd_dispatch_init();
    return g_dispatch.crc32c(crc, data, len);
}
