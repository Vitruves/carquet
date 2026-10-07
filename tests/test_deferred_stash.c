/**
 * @file test_deferred_stash.c
 * @brief Regression: the deferred-encode stash must hold the dense
 *        (non-null) values, not one slot per logical row.
 *
 * When a row group can be finalized in parallel (OpenMP build, >1 column,
 * no page index) every eligible column writer switches to `defer_encode`:
 * write_batch() stashes its input and the whole row group is replayed
 * through encode_batch_eager() later, from inside the OpenMP region.
 *
 * write_batch()'s `values` array is dense — for an OPTIONAL column it holds
 * only the non-null entries, packed, while `def_levels` has one entry per
 * logical row. stash_deferred_batch() nevertheless copied
 * `num_values * stride` bytes, i.e. one slot per logical row. Two bugs
 * followed:
 *
 *   1. Out-of-bounds read of `(num_values - non_null) * stride` bytes past
 *      the caller's array. For a sparse column this is unbounded in practice
 *      (a 131072-row batch of 4-byte values over-read 512 KB) and segfaulted
 *      when it ran off the end of the heap.
 *   2. Silent value corruption. drain_deferred() replays the stash through
 *      encode_batch_eager(), which walks it with a *dense* cursor, so the
 *      null-sized gaps the stash left between batches shifted every value
 *      after the first batch.
 *
 * Reproduction recipe: OPTIONAL fixed-width columns + a codec (compression
 * is required for defer eligibility) + dictionary off + page index off +
 * several write_batch() calls landing in one row group, with the dense value
 * arrays allocated to their exact size so the over-read is a real heap
 * overflow under ASan.
 *
 * Without OpenMP the deferred path does not engage and these cases simply
 * exercise the eager sparse path; they pass either way, and the CI matrix
 * covers both.
 */

#include <carquet/carquet.h>
#include "test_helpers.h"

#include <stdint.h>

#define BATCHES         4
#define ROWS_PER_BATCH  50000
#define ROWS            (BATCHES * ROWS_PER_BATCH)

/* Deliberately coprime densities so a shift of one column's dense cursor
 * cannot accidentally still line up with the other's. */
#define I32_PRESENT(i)  ((i) % 64 == 0)
#define F64_PRESENT(i)  ((i) % 97 == 0)

static int32_t i32_value(int64_t i) { return (int32_t)(i * 3 + 1); }
static double  f64_value(int64_t i) { return (double)i * 0.25 + 1.0; }

/* Schema: one REQUIRED column (a second column is needed for parallel
 * finalize) plus the two sparse OPTIONAL columns under test. */
static carquet_schema_t* make_schema(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = carquet_schema_create(&err);
    if (!s) return NULL;
    if (carquet_schema_add_column(s, "id", CARQUET_PHYSICAL_INT64, NULL,
                                  CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(s, "sparse_i32", CARQUET_PHYSICAL_INT32, NULL,
                                  CARQUET_REPETITION_OPTIONAL, 0, 0) != CARQUET_OK ||
        carquet_schema_add_column(s, "sparse_f64", CARQUET_PHYSICAL_DOUBLE, NULL,
                                  CARQUET_REPETITION_OPTIONAL, 0, 0) != CARQUET_OK) {
        carquet_schema_free(s);
        return NULL;
    }
    return s;
}

/* Options that make the columns defer-eligible: a codec is required, a
 * dictionary disqualifies the column, and the page index disables parallel
 * finalize for the whole row group. row_group_size is left at its 128 MB
 * default so every batch below lands in a single row group and is stashed
 * together — which is what exercises the multi-batch stash layout. */
static void init_options(carquet_writer_options_t* wo) {
    carquet_writer_options_init(wo);
    wo->compression = CARQUET_COMPRESSION_SNAPPY;
    wo->dictionary_encoding = CARQUET_ENCODING_PLAIN;
    wo->write_page_index = false;
}

static int test_deferred_stash_multi_batch(void) {
    const char* msg = NULL;
    char path[512];
    carquet_test_temp_path(path, sizeof(path), "deferred_stash");

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = make_schema();
    if (!s) TEST_FAIL("deferred_stash_multi_batch", "schema create failed");

    carquet_writer_options_t wo;
    init_options(&wo);

    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("deferred_stash_multi_batch", "writer create failed"); }

    int64_t* ids = (int64_t*)malloc(sizeof(int64_t) * ROWS_PER_BATCH);
    int16_t* def_i32 = (int16_t*)malloc(sizeof(int16_t) * ROWS_PER_BATCH);
    int16_t* def_f64 = (int16_t*)malloc(sizeof(int16_t) * ROWS_PER_BATCH);
    if (!ids || !def_i32 || !def_f64) { msg = "alloc failed"; goto done; }

    for (int b = 0; b < BATCHES; b++) {
        int64_t base = (int64_t)b * ROWS_PER_BATCH;

        int64_t n_i32 = 0, n_f64 = 0;
        for (int64_t k = 0; k < ROWS_PER_BATCH; k++) {
            int64_t i = base + k;
            ids[k] = i;
            def_i32[k] = I32_PRESENT(i) ? 1 : 0;
            def_f64[k] = F64_PRESENT(i) ? 1 : 0;
            if (def_i32[k]) n_i32++;
            if (def_f64[k]) n_f64++;
        }

        /* Exact-sized dense arrays: anything the writer reads past the last
         * present value is a genuine heap-buffer-overflow, not slack. */
        int32_t* vals_i32 = (int32_t*)malloc(sizeof(int32_t) * (size_t)n_i32);
        double*  vals_f64 = (double*)malloc(sizeof(double) * (size_t)n_f64);
        if (!vals_i32 || !vals_f64) {
            free(vals_i32); free(vals_f64);
            msg = "alloc failed"; goto done;
        }
        int64_t vi = 0, vf = 0;
        for (int64_t k = 0; k < ROWS_PER_BATCH; k++) {
            int64_t i = base + k;
            if (def_i32[k]) vals_i32[vi++] = i32_value(i);
            if (def_f64[k]) vals_f64[vf++] = f64_value(i);
        }

        carquet_status_t st =
            carquet_writer_write_batch(w, 0, ids, ROWS_PER_BATCH, NULL, NULL);
        if (st == CARQUET_OK)
            st = carquet_writer_write_batch(w, 1, vals_i32, ROWS_PER_BATCH, def_i32, NULL);
        if (st == CARQUET_OK)
            st = carquet_writer_write_batch(w, 2, vals_f64, ROWS_PER_BATCH, def_f64, NULL);

        free(vals_i32);
        free(vals_f64);
        if (st != CARQUET_OK) { msg = "write_batch failed"; goto done; }
    }

    if (carquet_writer_close(w) != CARQUET_OK) {
        w = NULL; msg = "close failed"; goto done;
    }
    w = NULL;

    {
        carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
        if (!r) { msg = "reader open failed"; goto done; }

        int16_t* outdef = (int16_t*)malloc(sizeof(int16_t) * ROWS);
        int32_t* out_i32 = (int32_t*)malloc(sizeof(int32_t) * ROWS);
        double*  out_f64 = (double*)malloc(sizeof(double) * ROWS);
        if (!outdef || !out_i32 || !out_f64) {
            free(outdef); free(out_i32); free(out_f64);
            carquet_reader_close(r);
            msg = "alloc failed"; goto done;
        }

        /* A single row group is expected: every batch was stashed together. */
        if (carquet_reader_num_row_groups(r) != 1) msg = "expected one row group";

        if (!msg) {
            carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 1, &err);
            int64_t got = c ? carquet_column_read_batch(c, out_i32, ROWS, outdef, NULL) : -1;
            if (got != ROWS) {
                msg = "sparse_i32 short read";
            } else {
                int64_t vi = 0;
                for (int64_t i = 0; i < ROWS && !msg; i++) {
                    if (outdef[i] != (I32_PRESENT(i) ? 1 : 0)) msg = "sparse_i32 def level mismatch";
                    else if (outdef[i] && out_i32[vi++] != i32_value(i)) msg = "sparse_i32 value mismatch";
                }
            }
            carquet_column_reader_free(c);
        }

        if (!msg) {
            carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 2, &err);
            int64_t got = c ? carquet_column_read_batch(c, out_f64, ROWS, outdef, NULL) : -1;
            if (got != ROWS) {
                msg = "sparse_f64 short read";
            } else {
                int64_t vf = 0;
                for (int64_t i = 0; i < ROWS && !msg; i++) {
                    if (outdef[i] != (F64_PRESENT(i) ? 1 : 0)) msg = "sparse_f64 def level mismatch";
                    else if (outdef[i] && out_f64[vf++] != f64_value(i)) msg = "sparse_f64 value mismatch";
                }
            }
            carquet_column_reader_free(c);
        }

        free(outdef); free(out_i32); free(out_f64);
        carquet_reader_close(r);
    }

done:
    if (w) carquet_writer_close(w);
    free(ids); free(def_i32); free(def_f64);
    carquet_schema_free(s);
    carquet_test_cleanup(path);
    if (msg) TEST_FAIL("deferred_stash_multi_batch", msg);
    TEST_PASS("deferred_stash_multi_batch");
    return 0;
}

/* Boundary batches: an all-null batch (dense array empty), a single-value
 * batch, and a fully-present batch, stashed back to back in one row group. */
static int test_deferred_stash_boundaries(void) {
#define BOUND_ROWS 128
    const char* msg = NULL;
    char path[512];
    carquet_test_temp_path(path, sizeof(path), "deferred_stash_bounds");

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = make_schema();
    if (!s) TEST_FAIL("deferred_stash_boundaries", "schema create failed");

    carquet_writer_options_t wo;
    init_options(&wo);

    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("deferred_stash_boundaries", "writer create failed"); }

    static int64_t ids[BOUND_ROWS];
    static int16_t def[BOUND_ROWS];
    static double  f64_all[BOUND_ROWS];

    /* present_count: 0 (all null), 1 (single value), BOUND_ROWS (all present) */
    const int64_t present_counts[3] = { 0, 1, BOUND_ROWS };

    for (int b = 0; b < 3 && !msg; b++) {
        int64_t base = (int64_t)b * BOUND_ROWS;
        int64_t want = present_counts[b];
        int64_t n = 0;
        for (int64_t k = 0; k < BOUND_ROWS; k++) {
            ids[k] = base + k;
            def[k] = (k < want) ? 1 : 0;
            if (def[k]) n++;
            f64_all[k] = f64_value(base + k);
        }

        /* malloc(0) may return NULL, and `values` is nonnull per the API
         * contract, so hand the empty case a valid 1-byte allocation. */
        int32_t* vals = (int32_t*)malloc(n ? sizeof(int32_t) * (size_t)n : 1);
        if (!vals) { msg = "alloc failed"; break; }
        for (int64_t k = 0; k < n; k++) vals[k] = i32_value(base + k);

        carquet_status_t st =
            carquet_writer_write_batch(w, 0, ids, BOUND_ROWS, NULL, NULL);
        if (st == CARQUET_OK)
            st = carquet_writer_write_batch(w, 1, vals, BOUND_ROWS, def, NULL);
        if (st == CARQUET_OK)
            st = carquet_writer_write_batch(w, 2, f64_all, BOUND_ROWS, NULL, NULL);
        free(vals);
        if (st != CARQUET_OK) msg = "write_batch failed";
    }

    if (!msg && carquet_writer_close(w) != CARQUET_OK) msg = "close failed";
    else if (msg) carquet_writer_close(w);

    if (!msg) {
        carquet_reader_t* r = carquet_reader_open(path, NULL, &err);
        if (!r) msg = "reader open failed";
        else {
            static int16_t outdef[BOUND_ROWS * 3];
            static int32_t out[BOUND_ROWS * 3];
            carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 1, &err);
            int64_t got = c ? carquet_column_read_batch(c, out, BOUND_ROWS * 3, outdef, NULL) : -1;
            if (got != BOUND_ROWS * 3) {
                msg = "short read";
            } else {
                int64_t vi = 0;
                for (int64_t i = 0; i < BOUND_ROWS * 3 && !msg; i++) {
                    int b = (int)(i / BOUND_ROWS);
                    int64_t k = i % BOUND_ROWS;
                    int16_t want_def = (k < present_counts[b]) ? 1 : 0;
                    if (outdef[i] != want_def) msg = "def level mismatch";
                    else if (want_def && out[vi++] != i32_value(i)) msg = "value mismatch";
                }
            }
            carquet_column_reader_free(c);
            carquet_reader_close(r);
        }
    }

    carquet_schema_free(s);
    carquet_test_cleanup(path);
    if (msg) TEST_FAIL("deferred_stash_boundaries", msg);
    TEST_PASS("deferred_stash_boundaries");
    return 0;
#undef BOUND_ROWS
}

/* Rejection path: an out-of-range column index must be refused, not stashed.
 * (`values` is nonnull per the API contract, so a NULL values pointer is not
 * a testable rejection — it is undefined behaviour at the call site.) */
static int test_deferred_stash_rejects_bad_column(void) {
    char path[512];
    carquet_test_temp_path(path, sizeof(path), "deferred_stash_badcol");

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* s = make_schema();
    if (!s) TEST_FAIL("deferred_stash_rejects_bad_column", "schema create failed");

    carquet_writer_options_t wo;
    init_options(&wo);

    carquet_writer_t* w = carquet_writer_create(path, s, &wo, &err);
    if (!w) { carquet_schema_free(s); TEST_FAIL("deferred_stash_rejects_bad_column", "writer create failed"); }

    int32_t vals[2] = { 7, 9 };
    int16_t def[4] = { 1, 0, 1, 0 };
    carquet_status_t too_high = carquet_writer_write_batch(w, 3, vals, 4, def, NULL);
    carquet_status_t negative = carquet_writer_write_batch(w, -1, vals, 4, def, NULL);

    carquet_writer_close(w);
    carquet_schema_free(s);
    carquet_test_cleanup(path);

    if (too_high != CARQUET_ERROR_INVALID_ARGUMENT ||
        negative != CARQUET_ERROR_INVALID_ARGUMENT)
        TEST_FAIL("deferred_stash_rejects_bad_column", "out-of-range column not rejected");
    TEST_PASS("deferred_stash_rejects_bad_column");
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_deferred_stash_multi_batch();
    failures += test_deferred_stash_boundaries();
    failures += test_deferred_stash_rejects_bad_column();
    if (failures) { printf("\n%d test(s) FAILED\n", failures); return 1; }
    printf("\nAll deferred stash tests passed\n");
    return 0;
}
