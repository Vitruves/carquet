/**
 * @file zstd.c
 * @brief ZSTD compression/decompression using libzstd
 *
 * Uses streaming context for better performance on repeated decompressions.
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <zstd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Thread-local storage compatibility:
 * - __thread requires macOS 10.7+ / iOS 9.0+
 * - For older targets, use pthread-based TLS
 */
#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#if defined(MAC_OS_X_VERSION_MIN_REQUIRED) && MAC_OS_X_VERSION_MIN_REQUIRED < 1070
#define CARQUET_USE_PTHREAD_TLS 1
#endif
#endif

/* Thread-local decompression contexts for parallel column reading */
#ifdef _OPENMP

#if defined(CARQUET_USE_PTHREAD_TLS)
/* Use pthread TLS for older macOS (< 10.7) */
#include <pthread.h>

static pthread_key_t tls_dctx_key;
static pthread_once_t tls_dctx_once = PTHREAD_ONCE_INIT;

static void destroy_dctx(void* ctx) {
    if (ctx) {
        ZSTD_freeDCtx((ZSTD_DCtx*)ctx);
    }
}

static void init_tls_key(void) {
    pthread_key_create(&tls_dctx_key, destroy_dctx);
}

static ZSTD_DCtx* get_dctx(void) {
    pthread_once(&tls_dctx_once, init_tls_key);
    ZSTD_DCtx* dctx = (ZSTD_DCtx*)pthread_getspecific(tls_dctx_key);
    if (!dctx) {
        dctx = ZSTD_createDCtx();
        if (dctx) {
            pthread_setspecific(tls_dctx_key, dctx);
        }
    }
    return dctx;
}

#else
/* Use thread-local storage for modern systems */
#ifdef _MSC_VER
static __declspec(thread) ZSTD_DCtx* tls_dctx = NULL;
#else
static __thread ZSTD_DCtx* tls_dctx = NULL;
#endif

static ZSTD_DCtx* get_dctx(void) {
    if (!tls_dctx) {
        tls_dctx = ZSTD_createDCtx();
    }
    return tls_dctx;
}
#endif /* CARQUET_USE_PTHREAD_TLS */

#else
/* No OpenMP - use global context */
static ZSTD_DCtx* global_dctx = NULL;

static ZSTD_DCtx* get_dctx(void) {
    if (!global_dctx) {
        global_dctx = ZSTD_createDCtx();
    }
    return global_dctx;
}
#endif /* _OPENMP */

int carquet_zstd_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Use streaming context for better buffer reuse */
    ZSTD_DCtx* dctx = get_dctx();
    if (!dctx) {
        /* Fallback to simple API */
        size_t result = ZSTD_decompress(dst, dst_capacity, src, src_size);
        if (ZSTD_isError(result)) {
            return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
        }
        *dst_size = result;
        return CARQUET_OK;
    }

    size_t result = ZSTD_decompressDCtx(dctx, dst, dst_capacity, src, src_size);
    if (ZSTD_isError(result)) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    *dst_size = result;
    return CARQUET_OK;
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

    if (level < 1) level = 1;
    if (level > ZSTD_maxCLevel()) level = ZSTD_maxCLevel();

    size_t result = ZSTD_compress(dst, dst_capacity, src, src_size, level);
    if (ZSTD_isError(result)) {
        return CARQUET_ERROR_COMPRESSION;
    }

    *dst_size = result;
    return CARQUET_OK;
}

size_t carquet_zstd_compress_bound(size_t src_size) {
    return ZSTD_compressBound(src_size);
}

void carquet_zstd_init_tables(void) {
    /* No-op - libzstd handles initialization internally */
}
