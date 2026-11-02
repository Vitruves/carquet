# carquet for Python

Python bindings for [carquet](https://github.com/Vitruves/carquet), a pure C11 Apache Parquet reader and writer. The wheel bundles the C library (zstd, zlib, LZ4, Snappy and Brotli compiled in) and talks to it through `ctypes`; data crosses the boundary through the [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html), so decoding, decompression, encoding and I/O all run in C, without the GIL, and PyArrow receives the buffers directly.

```bash
pip install "carquet[arrow]"   # with pyarrow: tables, DataFrames, streaming
pip install carquet            # library only: metadata, schema, statistics, validation
```

Requires Python 3.9+. PyArrow (>= 14) is imported lazily by the functions that produce or consume Arrow data; metadata, schema and statistics access, `info()`, `validate()` and `cpu_info()` work without it.

## Quick start

```python
import carquet as cq

table = cq.read_table("data.parquet")                                   # pyarrow.Table
table = cq.read_table("data.parquet", columns=["id", "price"], filters=[("price", ">", 100)])
df = cq.read_pandas("data.parquet")                                     # pandas.DataFrame (index restored)

cq.write_table(table, "out.parquet", compression="zstd")                # also: DataFrame, polars, dict, batches...

with cq.ParquetFile("data.parquet") as pf:
    pf.schema                                   # pyarrow.Schema
    pf.parquet_schema                           # physical / logical types, levels
    pf.metadata.row_group(0).column("price").statistics   # Statistics(min=..., max=..., null_count=...)
    for batch in pf.iter_batches(batch_size=50_000, columns=["id"]):
        ...
```

## Reading

### `read_table`

```python
cq.read_table(source, *, columns=None, filters=None, row_groups=None, use_threads=True,
              memory_map=True, verify_checksums=True, restore_arrow_schema=True) -> pyarrow.Table
```

`source` is a path, a bytes-like object (`bytes`, `bytearray`, `memoryview`, `pyarrow.Buffer`), a binary file object (`io.BytesIO`, an open file), a directory (every `*.parquet` file under it, sorted; hidden and `_`-prefixed entries skipped) or a list of any of these (concatenated). Buffers are read in place, without a copy.

- `columns`: top-level fields to read, in the order wanted. Only the projected columns are read and decoded. A dotted path (`"address.city"`) selects its top-level field.
- `filters`: a row predicate, see [Filters](#filters). The result is exact; row groups and pages that statistics rule out are not read.
- `row_groups`: indices of the row groups to read, in that order.
- `use_threads`: decode columns in parallel.
- `memory_map`: map the file instead of buffered reads.
- `restore_arrow_schema`: apply the Arrow types stored in the file's `ARROW:schema` entry (time zones, dictionary / categorical, large types, list child names, field metadata, pandas metadata), as `pyarrow.parquet` does.

`cq.read_pandas(source, **kwargs)` is `read_table(...).to_pandas()`. For polars, `pl.from_arrow(cq.read_table(...))` is zero-copy.

### `ParquetFile`

```python
cq.ParquetFile(source, *, memory_map=True, verify_checksums=True, num_threads=0,
               buffer_size=None, restore_arrow_schema=True)
```

An open file; use it as a context manager or call `close()`. Not thread-safe: open one per thread (separate `ParquetFile` objects can be used concurrently, since the GIL is released).

| Member | Returns |
|---|---|
| `schema` / `schema_arrow` | `pyarrow.Schema` of what `read()` returns |
| `parquet_schema` | `ParquetSchema` (see [Schema](#schema)) |
| `metadata` | `FileMetaData` (row groups, column chunks, statistics) |
| `key_value_metadata` | footer key/value metadata as a `dict` |
| `num_rows`, `num_row_groups`, `num_columns` | counts (`num_columns` counts leaf columns) |
| `read(columns=None, *, filters=None, row_groups=None, use_threads=True)` | `pyarrow.Table` |
| `read_row_group(i, columns=None)`, `read_row_groups(indices, columns=None)` | `pyarrow.Table` |
| `iter_batches(batch_size=65536, columns=None, *, filters=None, row_groups=None)` | iterator of `pyarrow.RecordBatch` (never spanning row groups) |
| `iter_row_groups(columns=None)` | iterator of `pyarrow.Table`, one per row group |
| `slice(offset, length=None, columns=None)`, `head(n=5, columns=None)` | `pyarrow.Table` of a row window; row groups outside it are not read, leading pages are skipped through the offset index when present |
| `count_rows(filters=None)` | row count, or the number of matching rows (only the filter columns are read) |
| `filter_row_groups(filters)` | indices of the row groups that statistics cannot rule out |
| `row_group(i)` | `RowGroupMetaData` |
| `bloom_filter(column, row_group=0)` | `BloomFilter` or `None` |
| `column_index(column, row_group=0)`, `offset_index(column, row_group=0)` | `ColumnIndex` / `OffsetIndex` or `None` |

`cq.iter_batches(source, batch_size, **kwargs)` streams a file (or several) and closes it at the end. Closing a `ParquetFile` stops its iterators.

### Filters

`filters` uses the `pyarrow.parquet` DNF form: a list of `(column, op, value)` tuples is an AND, a list of such lists is an OR of ANDs. A `pyarrow.compute.Expression` is also accepted (exact filtering only, no pushdown).

```python
cq.read_table(path, filters=[("year", ">=", 2020), ("country", "in", ["FR", "DE"])])
cq.read_table(path, filters=[[("id", "<", 10)], [("status", "is null")]])        # OR
cq.read_table(path, filters=pc.field("price") * pc.field("qty") > 1000)
```

Ops: `==` (`=`), `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `is null`, `is not null` (the last two take no value; `("x", "==", None)` means `is null`). Values are converted with the column's logical type: `datetime` / `date` / `numpy.datetime64` / `pandas.Timestamp` for timestamps and dates, `Decimal` for decimals, `str` for strings, `uuid.UUID` for UUIDs.

Pushdown never drops a matching row: a value that is inexact in the column's type (a float32 approximation, a timestamp finer than the column unit, a datetime compared with a date), out of its range, NaN or None, and `!=` on float columns (statistics ignore NaN) are only applied by the exact filter. Filtering happens in three stages: row groups whose min/max/null-count statistics rule the predicate out are skipped; for a single AND of flat columns in a file written with a page index, pages are pruned the same way inside the batch reader; the surviving rows are then filtered exactly with `pyarrow.compute`. Filter columns that are not in `columns` are read and dropped.

## Writing

### `write_table`

```python
cq.write_table(data, where, *, row_group_size=None, **options) -> None
```

`data` is a `pyarrow.Table`, `RecordBatch` or `RecordBatchReader`, a pandas DataFrame, any object implementing the Arrow PyCapsule stream interface (polars, DuckDB, ...), a `dict` of columns, or a list / iterator of RecordBatches. `where` is a path or a writable binary file object. If writing fails, a file created by the call is removed.

| Option | Meaning |
|---|---|
| `compression` | `"zstd"` (default), `"snappy"`, `"gzip"`, `"brotli"`, `"lz4"` (LZ4_RAW, what PyArrow/DuckDB/Spark write), `"lz4_hadoop"` (legacy framed LZ4), `"none"`; or `{column: codec}` (unlisted columns are uncompressed) |
| `compression_level` | an int (zstd 1-22, gzip 1-9, brotli 0-11) or `{column: level}`; `None` = codec default |
| `row_group_size` | rows per row group (default 1 048 576) |
| `data_page_size` | target page size in bytes (default 1 MiB), or `{column: bytes}` |
| `max_rows_per_page` | rows per data page |
| `use_dictionary` | `True` or a list of columns to dictionary-encode; the writer falls back to PLAIN for a chunk whose dictionary does not pay off. Default: carquet's own choice (PLAIN, BYTE_STREAM_SPLIT for floats) |
| `column_encoding` | one encoding for every column or `{column: encoding}`: `"plain"`, `"dictionary"`, `"delta_binary_packed"` (`"delta"`), `"delta_length_byte_array"`, `"delta_byte_array"`, `"byte_stream_split"`, `"rle"` |
| `use_byte_stream_split` | `True` (float columns) or a list of columns |
| `write_statistics` | `True` (default), `False`, or a list of columns |
| `max_statistics_size` | maximum stored size of binary min/max values (bytes, default 32) |
| `write_page_index` | write the column / offset indexes (page-level pruning) |
| `bloom_filters` | `True`, a list of columns, or `{column: {"ndv": n, "fpp": p}}` |
| `sorting_columns` | declared sort order: names, `(name, "descending")` tuples or `pyarrow.parquet.SortingColumn` objects (recorded, not verified) |
| `coerce_timestamps` / `allow_truncated_timestamps` | rescale every timestamp to `"ms"` / `"us"` / `"ns"`; truncation raises unless allowed |
| `data_page_version` | `"1.0"` (default) or `"2.0"` |
| `version` | footer format version: `"2.6"` (default) or `"1.0"` for very old readers |
| `write_page_checksum` (alias `write_crc`) | page CRC32 checksums (default on) |
| `write_batch_size`, `dictionary_pagesize_limit` | internal value-batch size, dictionary page size limit |
| `store_schema` | embed the Arrow schema (`ARROW:schema`) and the schema metadata (e.g. pandas) so readers restore exact Arrow types (default on) |
| `metadata` | extra `{key: value}` footer metadata |
| `append` | add the rows as new row groups to an existing file |

Brotli is compiled into the published wheels. A library built without it raises `CarquetError` with `status == StatusCode.UNSUPPORTED_CODEC`.

### `ParquetWriter`

```python
with cq.ParquetWriter("out.parquet", schema, compression="zstd", row_group_size=500_000) as w:
    for batch in source:
        w.write(batch)                 # Table, RecordBatch, reader, DataFrame, dict...
    w.add_metadata("origin", "etl-v2")
```

Takes the same options as `write_table`. Rows are packed into row groups of `row_group_size` rows however they arrive (`flush()` ends the current row group early). Data whose types differ from the writer schema is cast when PyArrow can do it safely, otherwise `ValueError`. On an exception inside the `with` block the file is discarded (`abort()`); `num_rows` counts the rows written so far.

`cq.ParquetWriter(path, append=True)` opens an existing file and adds row groups after the existing ones; the schema defaults to the file's.

## Metadata

`cq.read_metadata(source)` (or `ParquetFile.metadata`) returns a `FileMetaData`; everything is copied out of the footer, so these objects stay valid after the file is closed.

```python
md = cq.read_metadata("data.parquet")
md.num_rows, md.num_row_groups, md.num_columns, md.created_by, md.format_version, md.file_size
md.key_value_metadata                      # {"ARROW:schema": ..., ...}
rg = md.row_group(0)                       # RowGroupMetaData: num_rows, total_byte_size, total_compressed_size
col = rg.column("price")                   # by leaf index, dotted path or unique leaf name
col.compression, col.encodings, col.total_compressed_size, col.has_dictionary_page, col.has_bloom_filter
col.statistics.min, col.statistics.max, col.statistics.null_count, col.statistics.distinct_count
md.to_dict()
```

Statistics values are decoded with the column's logical type (`str`, `datetime`, `date`, `time`, `Decimal`, `uuid.UUID`, ...); the stored bytes are in `min_raw` / `max_raw`. GEOMETRY / GEOGRAPHY chunks also carry `geospatial_statistics` (bounding box, Z / M ranges, geometry types).

Page indexes and bloom filters:

```python
with cq.ParquetFile("data.parquet") as pf:
    ci = pf.column_index("price", row_group=0)      # per-page min / max / null_count
    oi = pf.offset_index("price", row_group=0)      # per-page offset / size / first row
    bf = pf.bloom_filter("user_id", row_group=0)
    if bf is not None and 12345 not in bf:          # False = definitely absent
        ...
```

## Schema

`ParquetFile.parquet_schema` / `cq.read_parquet_schema(source)` describe the Parquet side of the schema (no PyArrow needed). A `ParquetSchema` is a sequence of `ColumnSchema`, one per leaf column:

```python
s = cq.read_parquet_schema("data.parquet")
len(s), s.names, s.fields                  # leaf count, dotted leaf paths, top-level field names
c = s["amount"]                            # by index, dotted path or unique leaf name
c.physical_type, c.logical_type, c.repetition, c.max_definition_level, c.max_repetition_level
c.logical_type.precision, c.logical_type.scale, str(c.logical_type)    # 'DECIMAL(12, 2)'
s.elements                                 # every node (groups and leaves) in pre-order
print(s)                                   # indented tree
```

`cq.read_schema(source)` returns the Arrow schema.

## Type fidelity

What `write_table` writes, `read_table` reads back identically (`table.equals`), and `pyarrow.parquet.read_table` agrees. Files written by PyArrow, DuckDB, Spark, polars and others read with the same types `pyarrow.parquet` produces:

- DECIMAL columns (fixed-size binary, INT32 or INT64 storage) come back as `decimal128` / `decimal256`, INT96 timestamps as `timestamp[ns]` (vectorised NumPy conversions, no per-value Python work).
- The `ARROW:schema` entry restores time zones, `dictionary` (pandas categorical), `large_string` / `large_binary` / `large_list`, `int8` / `uint16`, `date64`, durations, list child names and field metadata; the file's other key/value metadata becomes the schema metadata (so `to_pandas()` restores the index).
- `write_table` stores the Arrow schema with PyArrow's own serializer, so nested schemas round-trip exactly too.
- Types without a direct Parquet mapping are written as their storage type and restored from `ARROW:schema`, as PyArrow does: `duration` as INT64, `string_view` / `binary_view` as strings / binary, `null` as an all-null INT32. `fixed_size_list` round-trips (PyArrow itself cannot read a `fixed_size_list` column that contains null lists).

Dictionary columns come back with one dictionary per row group, as with PyArrow.

## Performance

- Every C call releases the GIL: separate `ParquetFile` objects read in parallel from Python threads.
- Within a read, columns are decoded in parallel (OpenMP on Linux wheels; a persistent worker pool on macOS / Windows wheels), and batches are exported to Arrow in parallel per column.
- Column projection, row-group selection, statistics / page-index pruning and row windows (`slice`, `head`) are pushed down to C: unselected data is never decompressed.
- Memory-mapped files and in-memory buffers are read in place.
- Writing hands each RecordBatch to C without a Python-side copy; pages are encoded in parallel and row groups written on a background thread.

`cq.cpu_info()` lists the SIMD features the library dispatches to.

## Errors

Every failing C call raises `carquet.CarquetError`. It carries `code` (the numeric `carquet_status_t`), `status` (a `carquet.StatusCode` `IntEnum`, e.g. `StatusCode.FILE_NOT_FOUND`), `message` (the library's detailed message, never empty) and, when the library sets them, `offset`, `column_index` and `row_group_index`. Invalid arguments raise the usual `TypeError` / `ValueError` / `KeyError` (unknown column) / `IndexError` (row group out of range).

```python
try:
    cq.info("missing.parquet")
except cq.CarquetError as e:
    print(e.status.name, e.code, e.message)
```

## Other helpers

- `cq.info(path)`: `FileInfo` dataclass (`num_rows`, `num_row_groups`, `num_columns`, `file_size`, `version`, `created_by`) from the footer only.
- `cq.validate(path)`: structural validation (magic bytes, footer, metadata); raises `CarquetError`.
- `cq.__version__`, `cq.lib_version()`: package version and the loaded library's `carquet_version()`, both from `CARQUET_VERSION_STRING` in `include/carquet/carquet.h`.
- `cq.cpu_info()`: runtime SIMD detection (`has_neon`, `has_avx2`, `has_avx512f`, ...).

## Changes from 0.7.2

- `Reader` / `open_reader()` are now `ParquetFile` (the old names remain as aliases). `read_row_group()` returns a `pyarrow.Table` instead of a `RecordBatch`; `iter_batches()` yields batches of `batch_size` rows (65 536 by default) instead of one per row group; `iter_row_groups()` gives the old behaviour as Tables.
- `use_mmap=` is spelled `memory_map=` (still accepted); `schema()` is `read_schema()` (still available).
- `columns=` now projects in C (only the selected columns are read) instead of selecting after a full read.
- `write_table(..., store_schema=True)` (the default) used to write an `ARROW:schema` entry with DECIMAL precision 0 that made PyArrow reject the file; files with decimal columns written by 0.7.2 can be read by carquet, or rewritten.

## Interoperability

Files written by carquet are standard Parquet (format version 2 footer, optional DATA_PAGE_V2, page index, bloom filters), readable by every mainstream implementation:

```python
import pyarrow.parquet as pq
pq.read_table("out.parquet")

import duckdb
duckdb.sql("SELECT count(*) FROM 'out.parquet'")

import polars as pl
pl.read_parquet("out.parquet")
```

## How the library is bundled

The wheel contains the pure-Python package and one shared library placed next to `__init__.py`: `carquet/libcarquet.so` (Linux), `carquet/libcarquet.dylib` (macOS) or `carquet/carquet.dll` (Windows). It is built from the repository's own `CMakeLists.txt` with `CARQUET_BUILD_SHARED=ON`, `CARQUET_BUILD_CLI=OFF`, `CARQUET_BUILD_DEV=OFF` and `CARQUET_BUNDLE_DEPS=ON`, so zstd, zlib, LZ4 and Brotli are fetched at build time and linked statically into that single file (Snappy and LZ4_RAW are implemented inside carquet). The result depends only on the C runtime and system libraries. Linux wheels keep OpenMP; `auditwheel` grafts `libgomp` into the wheel. The macOS and Windows wheels are built without OpenMP (no `libomp` / `vcomp` dependency) and parallelise reads with the library's own worker pool.

The loader (`carquet/_ffi.py`) looks for the library next to the package, then in `lib/`, `lib64/` and `bin/` under it (layouts a plain `cmake --install` produces), then on the system path. Set the `CARQUET_LIBRARY` environment variable to an explicit file to override.

## Building the wheel locally

From the repository root, with CMake >= 3.16, a C11 compiler and git (for the codec sources) available:

```bash
pip wheel ./python -w dist/          # or: python -m build python/ --outdir dist/
pip install dist/carquet-*.whl
pytest python/tests                  # needs pytest + pyarrow (pandas / polars tests skip when absent)
```

`pip install -e ./python` is not supported (the shared library must be installed into the package); use a regular install. For development without reinstalling, build the shared library once (`cmake -B build-py -DCMAKE_BUILD_TYPE=Release -DCARQUET_BUILD_SHARED=ON -DCARQUET_BUNDLE_DEPS=ON && cmake --build build-py`) and run `CARQUET_LIBRARY=build-py/libcarquet.dylib PYTHONPATH=python/src pytest python/tests`. If your shell exports `CC`, `CFLAGS`, `LDFLAGS` or `SDKROOT` for another toolchain, clear them for the build (`env -u CC -u CFLAGS -u CPPFLAGS -u LDFLAGS -u SDKROOT pip wheel ./python -w dist/`).

The build is driven by `python/CMakeLists.txt`, a thin wrapper that adds the repository root as a subdirectory and installs only the runtime library (CMake component `carquet_python`) into the package directory. `pyproject.toml` (scikit-build-core) does the rest.

### Source distribution

The sdist root is `python/`, so the C tree one level up cannot be referenced as `..` once unpacked. `tool.scikit-build.sdist.force-include` therefore copies `CMakeLists.txt`, `LICENSE`, `include/`, `src/` and `cmake/` into `native/` inside the sdist, `python/CMakeLists.txt` prefers `native/` when it exists, and a `[[tool.scikit-build.overrides]]` entry (`if.from-sdist = true`) points the version regex at `native/include/carquet/carquet.h`. `python -m build --sdist python/` produces it; `pip install carquet-<version>.tar.gz` compiles it like the git checkout.

### Release wheels

`.github/workflows/wheels.yml` builds wheels with [cibuildwheel](https://cibuildwheel.pypa.io/) for CPython 3.9 - 3.13 on Linux (x86_64, aarch64; manylinux_2_28), macOS (arm64, x86_64) and Windows (AMD64), runs this test suite against each wheel, builds the sdist and publishes everything to PyPI via trusted publishing when a `v*` tag is pushed.

## Supported platforms

| Platform | Wheels | Notes |
|---|---|---|
| Linux x86_64 / aarch64 (glibc 2.28+) | yes | SSE/AVX2/AVX-512 or NEON dispatched at runtime; OpenMP |
| macOS 11+ arm64, macOS 10.13+ x86_64 | yes | worker-pool parallelism, no OpenMP |
| Windows x64 | yes | MSVC build, worker-pool parallelism; Windows loader path is covered by CI only |
| musl, 32-bit, PyPy | source build only | no PyArrow wheels exist there to test against |

## Limitations

- No Parquet modular encryption (not implemented in the C library).
- Structs and lists nested more than one level deep are assembled one row group at a time (`carquet_reader_read_arrow_fields`); `iter_batches()` slices them afterwards, so peak memory is one row group of those columns. Flat columns and single-level `list` / `map` columns stream at `batch_size`.
- Page-level pruning applies to a single AND of flat columns; OR-filters and nested columns use row-group statistics only (results are exact either way).
- A dotted path in `columns` selects its whole top-level field.
- Unions and run-end-encoded Arrow arrays are rejected on write (no Parquet equivalent); dictionary-encoded Arrow columns are written as their values (and restored as dictionaries through `ARROW:schema`).
- `created_by` is always the library's own string (`Carquet`); the page index records `boundary_order` as `UNORDERED`.
