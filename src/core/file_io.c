/**
 * @file file_io.c
 * @brief UTF-8 path support for Windows file APIs (see file_io.h).
 */

#include "file_io.h"

#ifdef _WIN32

#include "allocator.h"
#include <windows.h>

wchar_t* carquet_path_to_wide(const char* path) {
    if (!path) return NULL;

    /* Pure ASCII reads the same in every ANSI code page: keep the narrow API. */
    const unsigned char* p = (const unsigned char*)path;
    while (*p && *p < 0x80) p++;
    if (!*p) return NULL;

    /* MB_ERR_INVALID_CHARS: a path that is not well-formed UTF-8 is most
     * likely ANSI-encoded; let the caller fall back to the narrow API. */
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    if (n <= 0) return NULL;

    wchar_t* wide = (wchar_t*)carquet_mem_malloc((size_t)n * sizeof(wchar_t));
    if (!wide) return NULL;
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide, n) != n) {
        carquet_mem_free(wide);
        return NULL;
    }
    return wide;
}

FILE* carquet_fopen(const char* path, const char* mode) {
    wchar_t* wpath = carquet_path_to_wide(path);
    if (!wpath) return fopen(path, mode);

    wchar_t wmode[16];
    size_t i = 0;
    for (; mode[i] && i + 1 < sizeof(wmode) / sizeof(wmode[0]); i++) {
        wmode[i] = (wchar_t)(unsigned char)mode[i];
    }
    wmode[i] = L'\0';

    FILE* f = _wfopen(wpath, wmode);
    carquet_mem_free(wpath);
    return f;
}

int carquet_remove(const char* path) {
    wchar_t* wpath = carquet_path_to_wide(path);
    if (!wpath) return remove(path);
    int rc = _wremove(wpath);
    carquet_mem_free(wpath);
    return rc;
}

#else

/* ISO C forbids an empty translation unit. */
typedef int carquet_file_io_unused_t;

#endif
