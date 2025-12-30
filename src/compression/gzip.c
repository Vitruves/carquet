/**
 * @file gzip.c
 * @brief DEFLATE compression/decompression using zlib
 *
 * Parquet uses raw DEFLATE (no gzip/zlib headers).
 */

#include <carquet/error.h>
#include <stdint.h>
#include <stddef.h>
#include <zlib.h>

int carquet_gzip_decompress(
    const uint8_t* src,
    size_t src_size,
    uint8_t* dst,
    size_t dst_capacity,
    size_t* dst_size) {

    if (!src || !dst || !dst_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    z_stream strm = {0};
    strm.next_in = (Bytef*)src;
    strm.avail_in = (uInt)src_size;
    strm.next_out = (Bytef*)dst;
    strm.avail_out = (uInt)dst_capacity;

    /* 15 + 16 = gzip format (RFC 1952) */
    if (inflateInit2(&strm, 15 + 16) != Z_OK) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    int ret = inflate(&strm, Z_FINISH);
    size_t output_size = strm.total_out;
    inflateEnd(&strm);

    if (ret != Z_STREAM_END) {
        return CARQUET_ERROR_INVALID_COMPRESSED_DATA;
    }

    *dst_size = output_size;
    return CARQUET_OK;
}

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

    if (level < 1) level = 1;
    if (level > 9) level = 9;

    z_stream strm = {0};
    strm.next_in = (Bytef*)src;
    strm.avail_in = (uInt)src_size;
    strm.next_out = (Bytef*)dst;
    strm.avail_out = (uInt)dst_capacity;

    /* 15 + 16 = gzip format (RFC 1952) */
    if (deflateInit2(&strm, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        return CARQUET_ERROR_COMPRESSION;
    }

    int ret = deflate(&strm, Z_FINISH);
    size_t output_size = strm.total_out;
    deflateEnd(&strm);

    if (ret != Z_STREAM_END) {
        return CARQUET_ERROR_COMPRESSION;
    }

    *dst_size = output_size;
    return CARQUET_OK;
}

size_t carquet_gzip_compress_bound(size_t src_size) {
    /* compressBound is for zlib format; gzip adds ~18 bytes header/trailer */
    return compressBound((uLong)src_size) + 18;
}

void carquet_gzip_init_tables(void) {
    /* No-op - zlib handles initialization internally */
}
