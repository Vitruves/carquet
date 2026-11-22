/**
 * @file allocator.c
 * @brief Global memory allocator accessor.
 *
 * Stores the process-wide allocator configuration. The default is the C
 * standard library allocator. carquet_set_allocator() must be called before
 * any concurrent use (it is documented as not thread-safe).
 *
 * This provides the public allocator accessors declared in carquet.h and the
 * internal carquet_mem_* wrappers (see allocator.h) that the rest of the
 * library uses for every heap allocation, so a custom allocator is honored.
 */

#include <carquet/carquet.h>
#include "allocator.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

static void* default_malloc(size_t size, void* ctx) {
    (void)ctx;
    return malloc(size);
}

static void* default_realloc(void* ptr, size_t size, void* ctx) {
    (void)ctx;
    return realloc(ptr, size);
}

static void default_free(void* ptr, void* ctx) {
    (void)ctx;
    free(ptr);
}

static const carquet_allocator_t g_default_allocator = {
    default_malloc,
    default_realloc,
    default_free,
    NULL
};

static carquet_allocator_t g_allocator = {
    default_malloc,
    default_realloc,
    default_free,
    NULL
};

void carquet_set_allocator(const carquet_allocator_t* allocator) {
    /* Pooled blocks belong to the allocator being replaced. */
    carquet_mem_pool_flush();
    if (allocator == NULL ||
        allocator->malloc == NULL ||
        allocator->realloc == NULL ||
        allocator->free == NULL) {
        /* NULL or incomplete allocator resets to the libc default. */
        g_allocator = g_default_allocator;
        return;
    }
    g_allocator = *allocator;
}

const carquet_allocator_t* carquet_get_allocator(void) {
    return &g_allocator;
}

void carquet_free(void* ptr) {
    carquet_mem_free(ptr);
}

/* ============================================================================
 * Internal wrappers (see allocator.h)
 * ============================================================================
 */

void* carquet_mem_malloc(size_t size) {
    return g_allocator.malloc(size, g_allocator.ctx);
}

void* carquet_mem_calloc(size_t nmemb, size_t size) {
    size_t total;
    if (nmemb != 0 && size > SIZE_MAX / nmemb) {
        return NULL;  /* multiplication would overflow */
    }
    total = nmemb * size;
    void* p = g_allocator.malloc(total, g_allocator.ctx);
    if (p && total) {
        memset(p, 0, total);
    }
    return p;
}

void* carquet_mem_realloc(void* ptr, size_t size) {
    return g_allocator.realloc(ptr, size, g_allocator.ctx);
}

void carquet_mem_free(void* ptr) {
    if (ptr) {
        g_allocator.free(ptr, g_allocator.ctx);
    }
}

/* ============================================================================
 * Large-block pool (see allocator.h)
 * ============================================================================
 */

#define CARQUET_MEM_POOL_MAX_BYTES ((size_t)256 * 1024 * 1024)
#define CARQUET_MEM_POOL_SLOTS 64
#define CARQUET_MEM_POOL_IDLE_SECONDS 10

typedef struct {
    void* ptr;
    size_t size;
    time_t stamp;
} pool_block_t;

static pool_block_t g_pool[CARQUET_MEM_POOL_SLOTS];
static int g_pool_count;
static size_t g_pool_bytes;

#if defined(__GNUC__) || defined(__clang__)
static volatile int g_pool_lock;
#define POOL_HAVE_LOCK 1
static void pool_lock(void) {
    while (__atomic_exchange_n(&g_pool_lock, 1, __ATOMIC_ACQUIRE)) { /* spin */ }
}
static void pool_unlock(void) { __atomic_store_n(&g_pool_lock, 0, __ATOMIC_RELEASE); }
#elif defined(_MSC_VER)
#include <intrin.h>
static volatile long g_pool_lock;
#define POOL_HAVE_LOCK 1
static void pool_lock(void) {
    while (_InterlockedExchange(&g_pool_lock, 1)) { /* spin */ }
}
static void pool_unlock(void) { _InterlockedExchange(&g_pool_lock, 0); }
#endif

#ifdef POOL_HAVE_LOCK
/* Move blocks idle for too long into out[] (freed by the caller, outside
 * the lock). Called with the lock held. */
static int pool_take_stale(time_t now, void** out) {
    int n = 0;
    for (int i = 0; i < g_pool_count;) {
        if (now - g_pool[i].stamp > CARQUET_MEM_POOL_IDLE_SECONDS) {
            out[n++] = g_pool[i].ptr;
            g_pool_bytes -= g_pool[i].size;
            g_pool[i] = g_pool[--g_pool_count];
        } else {
            i++;
        }
    }
    return n;
}

static void free_all(void** blocks, int n) {
    for (int i = 0; i < n; i++) carquet_mem_free(blocks[i]);
}
#endif

void* carquet_mem_pool_try_take(size_t size, size_t* capacity) {
#ifdef POOL_HAVE_LOCK
    if (size < CARQUET_MEM_POOL_MIN_BLOCK || g_pool_count == 0) return NULL;
    void* stale[CARQUET_MEM_POOL_SLOTS];
    time_t now = time(NULL);
    void* found = NULL;
    pool_lock();
    int nstale = pool_take_stale(now, stale);
    int best = -1;
    for (int i = 0; i < g_pool_count; i++) {
        size_t s = g_pool[i].size;
        if (s >= size && s / 2 <= size && (best < 0 || s < g_pool[best].size)) best = i;
    }
    if (best >= 0) {
        found = g_pool[best].ptr;
        *capacity = g_pool[best].size;
        g_pool_bytes -= g_pool[best].size;
        g_pool[best] = g_pool[--g_pool_count];
    }
    pool_unlock();
    free_all(stale, nstale);
    return found;
#else
    (void)size; (void)capacity;
    return NULL;
#endif
}

void carquet_mem_pool_give(void* ptr, size_t capacity) {
    if (!ptr) return;
#ifdef POOL_HAVE_LOCK
    if (capacity >= CARQUET_MEM_POOL_MIN_BLOCK && capacity <= CARQUET_MEM_POOL_MAX_BYTES) {
        void* stale[CARQUET_MEM_POOL_SLOTS];
        time_t now = time(NULL);
        bool kept = false;
        pool_lock();
        int nstale = pool_take_stale(now, stale);
        if (g_pool_count < CARQUET_MEM_POOL_SLOTS &&
            g_pool_bytes + capacity <= CARQUET_MEM_POOL_MAX_BYTES) {
            g_pool[g_pool_count].ptr = ptr;
            g_pool[g_pool_count].size = capacity;
            g_pool[g_pool_count].stamp = now;
            g_pool_count++;
            g_pool_bytes += capacity;
            kept = true;
        }
        pool_unlock();
        free_all(stale, nstale);
        if (kept) return;
    }
#else
    (void)capacity;
#endif
    carquet_mem_free(ptr);
}

void* carquet_mem_pool_alloc(size_t size, size_t* capacity) {
    void* p = carquet_mem_pool_try_take(size, capacity);
    if (p) return p;
    p = carquet_mem_malloc(size);
    *capacity = p ? size : 0;
    return p;
}

void carquet_mem_pool_flush(void) {
#ifdef POOL_HAVE_LOCK
    void* blocks[CARQUET_MEM_POOL_SLOTS];
    pool_lock();
    int n = g_pool_count;
    for (int i = 0; i < n; i++) blocks[i] = g_pool[i].ptr;
    g_pool_count = 0;
    g_pool_bytes = 0;
    pool_unlock();
    free_all(blocks, n);
#endif
}
