/**
 * @file parallel_copy.c
 * @brief Multi-threaded copy of large input batches into writer buffers.
 *
 * carquet_writer_write_batch() must copy the caller's values (the caller may
 * reuse its buffer as soon as the call returns), and the copy runs on the
 * caller's thread while the encode pool works on earlier row groups. One
 * thread copies at a few GB/s; a 10M-row, 200 MB write spent a third of its
 * time there. Large copies are cut into pieces run by a small dedicated pool
 * (not the encode pool, whose queued encode tasks the copy would wait
 * behind) and the calling thread. One copy at a time uses the pool; another
 * writer copying meanwhile copies serially rather than wait.
 */

#include "parallel_copy.h"
#include "reader/worker_pool.h"

#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define PC_MIN_BYTES ((size_t)4 << 20)     /* below: one memcpy */
#define PC_PIECE_MIN ((size_t)1 << 20)
#define PC_MAX_HELPERS 4

#if defined(__GNUC__) || defined(__clang__)
static volatile int g_pc_busy;
#define PC_HAVE_ATOMICS 1
static int pc_try_acquire(void) { return __atomic_exchange_n(&g_pc_busy, 1, __ATOMIC_ACQUIRE) == 0; }
static void pc_acquire(void) { while (!pc_try_acquire()) { /* spin */ } }
static void pc_release(void) { __atomic_store_n(&g_pc_busy, 0, __ATOMIC_RELEASE); }
#elif defined(_MSC_VER)
#include <intrin.h>
static volatile long g_pc_busy;
#define PC_HAVE_ATOMICS 1
static int pc_try_acquire(void) { return _InterlockedExchange(&g_pc_busy, 1) == 0; }
static void pc_acquire(void) { while (!pc_try_acquire()) { /* spin */ } }
static void pc_release(void) { _InterlockedExchange(&g_pc_busy, 0); }
#endif

#ifdef PC_HAVE_ATOMICS
/* Owned by whoever holds g_pc_busy. */
static carquet_worker_pool_t* g_pc_pool;
static int g_pc_helpers = -1;   /* -1: not decided yet; 0: no pool */

typedef struct {
    uint8_t* dst;
    const uint8_t* src;
    size_t n;
} pc_piece_t;

static void pc_run(void* arg) {
    pc_piece_t* p = (pc_piece_t*)arg;
    memcpy(p->dst, p->src, p->n);
}

static int pc_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}
#endif

void carquet_parallel_copy(void* dst, const void* src, size_t n) {
#ifdef PC_HAVE_ATOMICS
    if (n >= PC_MIN_BYTES && pc_try_acquire()) {
        if (g_pc_helpers < 0) {
            int helpers = pc_cpu_count() / 2;
            if (helpers > PC_MAX_HELPERS) helpers = PC_MAX_HELPERS;
            g_pc_pool = helpers > 0 ? carquet_worker_pool_create(helpers) : NULL;
            g_pc_helpers = g_pc_pool ? helpers : 0;
        }
        if (g_pc_helpers > 0) {
            size_t parts = n / PC_PIECE_MIN;
            if (parts > (size_t)g_pc_helpers + 1) parts = (size_t)g_pc_helpers + 1;
            size_t per = (n / parts + 63) & ~(size_t)63;
            pc_piece_t pieces[PC_MAX_HELPERS + 1];
            size_t np = 0;
            for (size_t off = 0; off < n; off += per) {
                pieces[np].dst = (uint8_t*)dst + off;
                pieces[np].src = (const uint8_t*)src + off;
                pieces[np].n = n - off < per ? n - off : per;
                np++;
            }
            for (size_t i = 1; i < np; i++) {
                carquet_worker_pool_submit(g_pc_pool, pc_run, &pieces[i]);
            }
            pc_run(&pieces[0]);
            carquet_worker_pool_wait(g_pc_pool);
            pc_release();
            return;
        }
        pc_release();
    }
#endif
    if (n) memcpy(dst, src, n);
}

void carquet_parallel_copy_shutdown(void) {
#ifdef PC_HAVE_ATOMICS
    pc_acquire();
    if (g_pc_pool) carquet_worker_pool_destroy(g_pc_pool);
    g_pc_pool = NULL;
    g_pc_helpers = -1;
    pc_release();
#endif
}
