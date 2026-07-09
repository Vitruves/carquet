# Reading Files

## Choose the Reader API

Use the batch reader unless you specifically need raw Parquet level streams.

| Need | API |
| --- | --- |
| Project a subset of columns, iterate in row batches, parallelize work | `carquet_batch_reader_t` |
| Read one column from one row group, inspect `def_levels` / `rep_levels`, or skip values manually | `carquet_column_reader_t` |
| Only inspect footer metadata | `carquet_get_file_info()`, `carquet_validate_file()` |

## Open a Reader

```c
#include <carquet/carquet.h>

carquet_error_t err = CARQUET_ERROR_INIT;

carquet_reader_options_t opts;
carquet_reader_options_init(&opts);
opts.use_mmap = true;          /* Good default for local files */
opts.verify_checksums = true;  /* Keep enabled unless you trust the source */
opts.num_threads = 0;          /* Auto */

carquet_reader_t* reader = carquet_reader_open("data.parquet", &opts, &err);
if (!reader) {
    char buf[512];
    carquet_error_format(&err, buf, sizeof(buf));
    fprintf(stderr, "%s\n", buf);
    return 1;
}
```

Other entry points:

- `carquet_reader_open_file(FILE*, ...)`: caller keeps ownership of the `FILE*`
- `carquet_reader_open_buffer(const void* buffer, size_t size, ...)`: buffer must outlive the reader

Useful metadata calls right after open:

- `carquet_reader_schema()`
- `carquet_reader_num_rows()`
- `carquet_reader_num_row_groups()`
- `carquet_reader_num_columns()`
- `carquet_reader_is_mmap()`

## Batch Reader Workflow

This is the default path for scans and analytics.

```c
carquet_batch_reader_config_t cfg;
carquet_batch_reader_config_init(&cfg);

const char* cols[] = {"id", "price"};
cfg.column_names = cols;
cfg.num_column_names = 2;
cfg.batch_size = 65536;
cfg.num_threads = 0;

carquet_batch_reader_t* br = carquet_batch_reader_create(reader, &cfg, &err);
if (!br) {
    fprintf(stderr, "%s\n", err.message);
    carquet_reader_close(reader);
    return 1;
}

carquet_row_batch_t* batch = NULL;
while (carquet_batch_reader_next(br, &batch) == CARQUET_OK && batch) {
    const void* id_data;
    const uint8_t* id_nulls;
    int64_t id_count;

    carquet_row_batch_column(batch, 0, &id_data, &id_nulls, &id_count);
    const int64_t* ids = id_data;

    carquet_row_batch_free(batch);
    batch = NULL;
}

carquet_batch_reader_free(br);
carquet_reader_close(reader);
```

Notes:

- `column_indices` takes precedence over `column_names`.
- `row_group_filter` lets you skip entire row groups before any data pages are read. Use it with `carquet_reader_row_group_matches()` or `carquet_reader_column_statistics()`.
- `carquet_row_batch_num_columns()` is the number of projected columns, not the file-wide total.
- For nullable columns, the null bitmap uses `1` for present and `0` for null:

```c
bool is_null = null_bitmap && !(null_bitmap[i / 8] & (1u << (i % 8)));
```

- `BYTE_ARRAY` data comes back as `const carquet_byte_array_t*`.
- `FIXED_LEN_BYTE_ARRAY` data comes back as tightly packed bytes. Use the schema's type length to stride the buffer correctly.

## Column Reader Workflow

Use a column reader when you need explicit control over row groups or Parquet levels.

```c
carquet_column_reader_t* col = carquet_reader_get_column(reader, 0, 2, &err);
if (!col) {
    fprintf(stderr, "%s\n", err.message);
    carquet_reader_close(reader);
    return 1;
}

int32_t values[1024];
int16_t def_levels[1024];
int64_t n;

while ((n = carquet_column_read_batch(col, values, 1024, def_levels, NULL)) > 0) {
    /* values contains only materialized values for this physical type */
}

carquet_column_reader_free(col);
```

`carquet_column_read_batch()` returns `-1` for any failure and cannot distinguish a page-read error that truncates a batch after some values were already read from a clean short read at end-of-column — the `n > 0` loop above treats both as "keep going / done". When you need to tell those apart (e.g. so incomplete data is never processed as complete), use `carquet_column_read_batch_ex()`, which reports a distinct status code and message through a `carquet_error_t` out-parameter:

```c
carquet_error_t err = CARQUET_ERROR_INIT;
int64_t n = carquet_column_read_batch_ex(col, values, 1024, def_levels, NULL, &err);
if (n < 0) {
    /* hard failure, nothing read — err.code / err.message say why */
} else if (carquet_error_is_set(&err)) {
    /* n values are valid, but a read error truncated the batch:
       do NOT treat n as a clean end-of-column result */
} else {
    /* clean read of n values (n < requested ⇒ end of column) */
}
```

The error out-parameter is cleared on entry, so `err.code == CARQUET_OK` reliably means "no failure", independent of the count. Passing `NULL` for the error makes it behave exactly like `carquet_column_read_batch()`.

Also available:

- `carquet_column_skip()`
- `carquet_column_has_next()`
- `carquet_column_remaining()`

## Reading Nested Data (lists, structs, maps)

Carquet reads nested schemas — structs (groups), `LIST`, and `MAP` — at the **physical leaf-column level**, which is how Parquet stores nested data on disk. It does not currently reassemble leaves into materialized nested value objects for you (that is, there is no `List<Struct<…>>`-shaped result type); instead you read each leaf column plus its repetition and definition levels and reconstruct the structure yourself. This is a deliberate boundary: the format-level support is complete (nested files written by carquet are read back as native nested types by Arrow/PyArrow and DuckDB), but high-level nested materialization is tracked as future work in `TODO.md`.

What this means in practice:

- A nested field expands to one column chunk **per leaf**. For example a PyArrow `struct<a, b>` is two leaf columns (`a`, `b`); a `map<string,int32>` is two leaves (`key`, `value`); a `list<int32>` is one leaf (`element`). `carquet_reader_num_columns()` therefore counts leaves, not top-level fields.
- Read each leaf with the column-reader workflow above, requesting **definition and repetition levels**. Definition levels tell you which entries are null / which optional ancestors are present; repetition levels tell you where each list/row begins.
- Use the reconstruction helpers to recover structure from the levels:
  - `carquet_count_rows(rep_levels, n, ...)` — count logical (top-level) rows from repetition levels.
  - `carquet_list_offsets(rep_levels, n, list_level, offsets, max)` — compute list boundaries (offsets) for a given repetition level.

```c
/* list<int32> "tags": one leaf, max_def = 3, max_rep = 1 */
carquet_column_reader_t* col = carquet_reader_get_column(reader, 0, leaf_index, &err);

int32_t values[1024];
int16_t def_levels[1024];
int16_t rep_levels[1024];
int64_t n = carquet_column_read_batch(col, values, 1024, def_levels, rep_levels);

int64_t offsets[256];
int64_t num_lists = carquet_list_offsets(rep_levels, n, 1, offsets, 256);
/* offsets[i]..offsets[i+1] delimits the i-th list; def_level < max_def marks
 * null lists / null elements per the Parquet level rules. */

carquet_column_reader_free(col);
```

If you need values presented as native nested types rather than leaves + levels, that reassembly currently lives in the calling application (or a higher-level binding). See `tests/test_nested.c` for end-to-end reconstruction examples.

## Predicate Pushdown and Cheap Inspection

Use row-group statistics before you start reading payload pages:

- `carquet_reader_column_statistics()`: get min/max/null counts for one row group + column
- `carquet_reader_row_group_matches()`: ask whether one row group might satisfy a predicate
- `carquet_reader_filter_row_groups()`: collect all candidate row groups at once

Use footer-only helpers when you do not need to build a full reader:

- `carquet_get_file_info()`
- `carquet_validate_file()`

### Column Statistics

`carquet_reader_column_statistics()` returns aggregated stats for a single (row group, column). The `min_value` / `max_value` fields are raw bytes — interpret them according to the column's physical type. For `FLOAT16` columns the two bytes are the little-endian IEEE half representation of the numeric min/max (not a lexicographic bound). `INT96` and `GEOMETRY`/`GEOGRAPHY` columns report no min/max (`has_min_max == false`). Legacy files whose V1 pages use the deprecated `BIT_PACKED` level encoding are decoded transparently.

For `GEOMETRY`/`GEOGRAPHY` columns, call `carquet_reader_geospatial_statistics()` to get the coordinate bounding box (`xmin/xmax/ymin/ymax`, plus `z`/`m` when present) and the set of ISO-WKB geometry type codes. It returns `CARQUET_ERROR_INVALID_METADATA` for columns that carry no geospatial statistics (not an error). The `carquet stat` CLI prints this automatically.

```c
carquet_column_statistics_t s;
if (carquet_reader_column_statistics(reader, /*row_group=*/0, /*column=*/0, &s)
        == CARQUET_OK && s.has_min_max) {
    /* Fixed-width numeric types: cast directly. */
    int64_t min_id = *(const int64_t*)s.min_value;
    int64_t max_id = *(const int64_t*)s.max_value;
    printf("id [%lld, %lld] nulls=%lld\n",
           (long long)min_id, (long long)max_id, (long long)s.null_count);
}
```

For `BYTE_ARRAY` columns the payload is the bytes themselves with length in `min_value_size` / `max_value_size` — **not** a `carquet_byte_array_t` struct:

```c
if (s.has_min_max) {
    printf("min=%.*s max=%.*s\n",
           s.min_value_size, (const char*)s.min_value,
           s.max_value_size, (const char*)s.max_value);
}
```

Long byte-array max values written by carquet are truncated at 32 bytes and incremented, so the stored bound is an upper bound but not necessarily an exact value present in the column.

To prune row groups in bulk, pass a typed value to `carquet_reader_filter_row_groups()`:

```c
int32_t threshold = 5000;
int32_t matches[64];
int32_t n = carquet_reader_filter_row_groups(
    reader, /*column=*/0, CARQUET_COMPARE_GT,
    &threshold, sizeof(threshold), matches, 64);
```

## Filtering Rows With a Page Filter

The batch reader can attach a page-level filter that skips data pages whose column-index statistics prove no value can satisfy the predicate. Only the matching pages are decompressed and decoded; pruned pages are never read from disk past their offset-index entry. This applies to both the sequential reader and the parallel pipeline.

```c
carquet_batch_reader_t* br = carquet_batch_reader_create(reader, &cfg, &err);

int32_t lo = 1000;
int32_t hi = 2000;
carquet_filter_clause_t clause = {
    .column_index = 3,                 /* file column to filter on */
    .op = CARQUET_FILTER_RANGE,
    .has_lo = true, .lo = &lo, .lo_size = sizeof(lo),
    .has_hi = true, .hi = &hi, .hi_size = sizeof(hi),
};
carquet_batch_reader_set_page_filter(br, &clause, 1);
```

Each clause is a `(column_index, op, value)` triple; multiple clauses are AND'd together and may reference different columns (whether or not those columns are part of the projection — a predicate column that is not projected is read **only** via its column + offset index, never decompressed). Supported ops cover `EQ`, `NE`, `LT`, `LE`, `GT`, `GE`, `RANGE`, `IN`, `IS_NULL`, and `IS_NOT_NULL`.

```c
int64_t in_values[] = {10, 20, 30};
carquet_filter_clause_t in_clause = {
    .column_index = 0,
    .op = CARQUET_FILTER_IN,
    .values = in_values,
    .value_count = 3,
};
```

Pass `clauses = NULL, count = 0` to clear an active filter.

After reading, `carquet_batch_reader_rows_skipped()` reports the running count of rows the filter pruned — useful for confirming selectivity:

```c
int64_t skipped = carquet_batch_reader_rows_skipped(br);
```

**Requirements and semantics**

- The file must have been written with `write_page_index = true` for every column the filter references. Filtering a column without a page index returns `CARQUET_ERROR_PAGE_INDEX_REQUIRED` on the next `carquet_batch_reader_next()` call.
- `INT96` columns have no defined sort order per the Parquet spec and are rejected with `CARQUET_ERROR_INVALID_ARGUMENT` at filter-set time.
- For BYTE_ARRAY columns with truncated min/max stats, the filter is conservative: a page may be kept even when no row in it actually matches. Filtering is page-granular by design — rows inside a matching page that fail the predicate are still returned and the caller must apply the exact predicate to the batch if needed.
- Floats whose predicate value is NaN match nothing under ordered ops (`EQ`, `LT`, …), mirroring Arrow semantics.
- The clauses array and any pointers it references must remain valid until the next `set_page_filter()` call or until the batch reader is freed; no copy is made.

## Metadata, Bloom Filters, and Page Indexes

Carquet exposes the optional metadata structures that many readers hide:

- Key-value footer metadata: `carquet_reader_num_metadata()`, `carquet_reader_get_metadata()`, `carquet_reader_find_metadata()`
- Bloom filters: `carquet_reader_get_bloom_filter()` plus `carquet_bloom_filter_check_*()`
- Page indexes: `carquet_reader_get_column_index()`, `carquet_reader_get_offset_index()`
- Column chunk metadata: `carquet_reader_column_chunk_metadata()`

Use these when you need diagnostics, custom pruning, or interoperability checks. For a compact end-to-end example, see [`examples/advanced_features.c`](../examples/advanced_features.c).

## Lifetime and Ownership

- Close column readers with `carquet_column_reader_free()`.
- Free each row batch with `carquet_row_batch_free()` before asking for the next one.
- Pointers returned from batch APIs belong to the batch. Do not keep them after freeing the batch.
- Pointers returned from reader metadata APIs belong to the reader. Do not keep them after closing the reader.
