/**
 * @file test_footer_alloc.c
 * @brief Footer parsing reports a failed allocation
 */

#include <carquet/carquet.h>
#include "thrift/parquet_types.h"
#include "core/arena.h"
#include "test_helpers.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ALLOC_FAIL_ABOVE ((size_t)256 * 1024)

static void* cap_malloc(size_t size, void* ctx) {
    (void)ctx;
    return size > ALLOC_FAIL_ABOVE ? NULL : malloc(size);
}
static void* cap_realloc(void* ptr, size_t size, void* ctx) {
    (void)ctx;
    return size > ALLOC_FAIL_ABOVE ? NULL : realloc(ptr, size);
}
static void cap_free(void* ptr, void* ctx) {
    (void)ctx;
    free(ptr);
}

static int test_footer_alloc_failure(void) {
    enum { ELEMS = 10000 };
    if ((size_t)ELEMS * sizeof(parquet_schema_element_t) <= ALLOC_FAIL_ABOVE)
        TEST_FAIL("footer_alloc_failure", "schema array too small to exercise the path");

    /* FileMetaData { 2: list<SchemaElement> of ELEMS empty structs }. */
    size_t size = 4 + ELEMS + 1;
    uint8_t* footer = calloc(size, 1);
    if (!footer) TEST_FAIL("footer_alloc_failure", "test buffer");
    footer[0] = 0x29;                       /* field 2, LIST */
    footer[1] = 0xFC;                       /* long list of STRUCT */
    footer[2] = (uint8_t)((ELEMS & 0x7F) | 0x80);
    footer[3] = (uint8_t)(ELEMS >> 7);      /* varint(ELEMS) */

    carquet_allocator_t capped = { cap_malloc, cap_realloc, cap_free, NULL };
    carquet_set_allocator(&capped);

    carquet_arena_t arena;
    carquet_status_t st = carquet_arena_init(&arena);
    if (st == CARQUET_OK) {
        parquet_file_metadata_t md;
        carquet_error_t err = CARQUET_ERROR_INIT;
        st = parquet_parse_file_metadata(footer, size, &arena, &md, &err);
        carquet_arena_destroy(&arena);
    } else {
        st = CARQUET_OK;   /* arena itself did not fit: reported below */
    }
    carquet_set_allocator(NULL);
    free(footer);

    if (st != CARQUET_ERROR_OUT_OF_MEMORY)
        TEST_FAIL("footer_alloc_failure", "allocation failure not reported as out of memory");

    TEST_PASS("footer_alloc_failure");
    return 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int failures = 0;
    failures += test_footer_alloc_failure();
    if (failures) {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
    printf("All tests passed\n");
    return 0;
}
