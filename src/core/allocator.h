/**
 * @file allocator.h
 * @brief Internal allocation wrappers that route through the global allocator.
 *
 * Every heap allocation in the library goes through these wrappers so that a
 * custom allocator installed via carquet_set_allocator() is actually used.
 *
 * The public contract requires carquet_set_allocator() to be called before
 * any allocation and before concurrent use, so the active allocator is fixed
 * for the lifetime of all allocations: a block allocated through these
 * wrappers is always freed through them with the same allocator. Never mix
 * these with libc malloc/free for the same pointer.
 */
#ifndef CARQUET_CORE_ALLOCATOR_H
#define CARQUET_CORE_ALLOCATOR_H

#include <stddef.h>

/** Allocate @p size bytes (size 0 yields a unique freeable pointer or NULL). */
void* carquet_mem_malloc(size_t size);

/** Allocate @p nmemb * @p size zeroed bytes, with overflow check. */
void* carquet_mem_calloc(size_t nmemb, size_t size);

/** Resize @p ptr to @p size bytes (ptr may be NULL => malloc). */
void* carquet_mem_realloc(void* ptr, size_t size);

/** Free @p ptr (NULL is a no-op). */
void carquet_mem_free(void* ptr);

/* ----------------------------------------------------------------------------
 * Large-block reuse
 *
 * Code that knows the size of a large block it is done with (a buffer's
 * capacity) can hand it to a small process-wide pool instead of freeing it,
 * and code about to allocate a large block can take one back. With glibc,
 * every freed block from 128 KB up goes back to the kernel and the next one
 * costs a page fault per 4 KB page; a scan or a write that allocates its
 * buffers anew each time (a new reader / writer per file) spent a fifth of
 * its time there. The pool only ever holds ordinary blocks of the current
 * allocator, so a block taken from it may be freed with carquet_mem_free
 * like any other. Bounded (CARQUET_MEM_POOL_MAX_BYTES), emptied when the
 * allocator changes and by carquet_cleanup(), and blocks unused for a few
 * seconds are freed on the next pool call.
 * ------------------------------------------------------------------------- */

#define CARQUET_MEM_POOL_MIN_BLOCK ((size_t)256 * 1024)

/* A pooled block of at least `size` bytes (at most twice that), or NULL if
 * the pool has none; *capacity receives its size. */
void* carquet_mem_pool_try_take(size_t size, size_t* capacity);

/* Done with `ptr`, a block of `capacity` bytes from carquet_mem_malloc /
 * realloc / the pool: pooled when it fits, freed otherwise. */
void carquet_mem_pool_give(void* ptr, size_t capacity);

/* Free every pooled block. */
void carquet_mem_pool_flush(void);

#endif /* CARQUET_CORE_ALLOCATOR_H */
