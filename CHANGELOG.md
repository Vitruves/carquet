# Changelog

## v0.5.1

Bug-fix + performance release. No public API/ABI change; default output bytes unchanged.

### Bug Fixes

- **Windows large-file reads (>2 GiB)**: `src/reader/file_reader.c` and `src/reader/page_reader.c` used plain `fseek`/`ftell`, which silently truncate on 64-bit Windows where `long` is 32-bit. All I/O now routes through `carquet_fseek64`/`carquet_ftell64` (`src/core/compat.h`) dispatching to `_fseeki64`/`_ftelli64` on Windows, `fseeko`/`ftello` on POSIX. Tests migrated too so Windows CI exercises the wrappers.

### Performance

- **Parallel encode + compress**: non-dictionary, compressed, fixed-stride columns now stash raw input during `write_batch` and replay the unchanged eager encode path inside the OpenMP per-column finalize, so encode and compression run concurrently across columns. Output byte-identical to the eager path; self-gated per column and per row group, no cost when not applicable.
- **NEON dispatch tuned**: dropped hand-written NEON paths for `byte_stream_split` float and i64 min/max + copy-min/max — measured slower than the auto-vectorized scalar on Apple M3. Double `byte_stream_split` and i32/float/double min/max keep their NEON paths. Bit-identical output.

## v0.5.0

Closes several Arrow-interoperability and Parquet-conformance gaps: dictionary writing, writer encoding breadth (now symmetric — readable by carquet itself), the `INTERVAL` logical type, `sorting_columns` metadata, per-column bloom configuration, a row-count page-flush knob, INT96 writing, opt-in Data Page V2 writing, opt-in `ARROW:schema` footer metadata, correct FLOAT16 statistics ordering, deprecated BIT_PACKED level decoding, GEOMETRY/GEOGRAPHY GeospatialStatistics, and the Arrow-writer parity knobs (TIMESTAMP coercion/truncation, `write_batch_size`). No new dependencies; default output bytes are unchanged (every addition is opt-in, a previously-unsupported type, or a read-path/spec-correctness fix). It also lands an API/correctness audit: previously-unlinkable public functions are implemented, a configured custom allocator is now actually honored library-wide, row-group predicate pushdown is type-correct, and x86 SIMD feature dispatch is OS-state-gated and race-free.

### Compatibility

**Not binary-compatible with v0.4.4 — recompile against the new headers; do not merely relink.** Source compatibility is preserved (existing code compiles unchanged: every new option-struct field defaults to the previous behavior, and the `created_by` array decays to `char*` where the old `const char*` was read), but the ABI changed:

- **Enlarged caller-allocated public structs.** `carquet_writer_options_t` gained `write_arrow_schema`, `data_page_version`, `max_rows_per_page`, `coerce_timestamps`, `coerce_timestamp_unit`, `allow_timestamp_truncation`, and `write_batch_size`; `carquet_batch_reader_config_t` and the bloom-filter options also grew. These structs are allocated **by value** by callers, so their `sizeof` changing means a v0.4.4 binary relinked against v0.4.5 (or vice versa) without recompiling reads/writes them at the wrong offsets. This affects real existing consumers and is the primary reason this is an ABI break.
- **New enumerator** `CARQUET_LOGICAL_INTERVAL` added to `carquet_logical_type_id_t`.
- **`carquet_file_info_t` layout change**: `created_by` changed from `const char*` to inline `char created_by[CARQUET_CREATED_BY_MAX]` (256), changing the struct's size. Listed for completeness only: its sole consumer, `carquet_get_file_info()`, had no definition before v0.4.5 (it never linked), so no existing binary or even compilable program can depend on the old layout — this part breaks nothing in practice.

### Bug Fixes

- **Implemented missing public API functions**: `carquet_reader_open_file()`, `carquet_get_file_info()`, `carquet_validate_file()`, `carquet_set_allocator()`, and `carquet_get_allocator()` were declared in `carquet.h` but had no definitions, so any program calling them failed at link time. They are now implemented. `carquet_reader_open_file()` reads from a caller-owned `FILE*` (carquet does not take ownership or close it). `carquet_validate_file()` does structural validation **and** streams every row group/column/page with checksum verification enabled, so CRC, decompression, and decode errors are all surfaced rather than a footer-only check.
- **Custom allocator is now actually used**: `carquet_set_allocator()` previously stored the allocator but the library kept allocating through libc, silently ignoring it. Every heap allocation in the library now routes through the configured allocator via internal wrappers, and `carquet_get_allocator()` returns the active one (libc by default). Scratch memory owned by zlib/zstd is unaffected. Verified ASan/UBSan-clean and allocation-balanced across the full suite.
- **Predicate-pushdown comparisons fixed**: row-group statistics filtering read `BOOLEAN` stats with the 4-byte `INT32` comparator (a 3-byte out-of-bounds read of the 1-byte stat), compared unsigned (`UINT32`/`UINT64` via logical or legacy `ConvertedType`) columns with signed ordering (false negatives for large values), and ordered `FLOAT16` stats byte-lexicographically instead of numerically. Comparisons are now type-correct, unsigned-aware, FLOAT16-numeric, and read stat bytes via memcpy (no unaligned access); a stat whose width does not match the column type is treated as indeterminate so the row group is conservatively kept.
- **x86 SIMD dispatch hardened**: CPU feature detection now checks `OSXSAVE` + `XGETBV`/`XCR0` before selecting AVX/AVX2 (the OS must have enabled YMM state) and additionally requires opmask/ZMM state plus `AVX512BW`+`AVX512VL` before selecting AVX-512 paths (the AVX-512 objects are compiled with BW/VL), preventing `#UD`/`#GP` on machines that advertise but do not fully enable these. The lazy dispatch-table initialization is now acquire/release atomic, closing a first-use data race.
- **Removed a latent bit-unpack buffer-overflow landmine**: the unused `carquet_neon_bitunpack8_32` was documented and named as an 8-value unpacker but dispatched to 32- and 16-value kernels for bit widths 1 and 2. It had no callers (NEON never wired the bit-unpack table), so it could not corrupt memory as shipped, but wiring it like the x86 path would have written 32/16 values into 8-element buffers. It and the now-orphaned pure-scalar `carquet_neon_bitunpack16_2bit` and `carquet_avx2_bitunpack64_1bit` dead kernels were removed.

### Performance

- **Wide SIMD bit-unpacking**: bit-unpacking — used by DELTA decode (`carquet_bitunpack_32`) and by RLE definition/repetition-level decode (`carquet_rle_decode_levels`) — now processes 16 or 32 values per SIMD call for the common bit widths 1/4/8/16 instead of 8 at a time, via a new verified dispatch tier that finally wires the previously dead-but-correct wider kernels (SSE 32×1-bit; AVX2 16×4/8-bit; AVX-512 32×4/8-bit and 16×16-bit; NEON 32×1-bit). Other widths and truncated / output-capped bit-packed runs fall back to the unchanged 8-at-a-time path, so output is bit-identical. A new `test_bitunpack_wide` validates `carquet_bitunpack_32` and the RLE level/all decoders against an independent scalar oracle across every bit width and chunk-boundary count (ASan+UBSan-clean).

### New Features

- **Dictionary page writing (opt-in)**: The writer can emit a real `DICTIONARY_PAGE` (PLAIN-encoded entries) followed by `RLE_DICTIONARY` data pages, with the column chunk's `dictionary_page_offset` set. Enable it per column with `carquet_writer_set_column_encoding(writer, col, CARQUET_ENCODING_RLE_DICTIONARY)`. A column whose dictionary would exceed `dictionary_page_size`, or whose values are effectively all-unique, transparently **falls back to PLAIN** (with an early-abort so high-cardinality columns don't pay to build a dictionary they won't use), and its `ColumnMetaData.encodings` then advertises only the encodings actually used (no spurious `RLE_DICTIONARY`). It is opt-in rather than default because defaulting to it regressed read throughput substantially (notably the zero-copy uncompressed read path) for a size win that is negligible under zstd; the default policy is unchanged from v0.4.4 (PLAIN, with automatic `BYTE_STREAM_SPLIT` for FLOAT/DOUBLE when compression is enabled).
- **Broader per-column writer encodings**: `carquet_writer_set_column_encoding()` now accepts `DELTA_BINARY_PACKED` (`INT32`/`INT64`), `DELTA_LENGTH_BYTE_ARRAY` (`BYTE_ARRAY`), `DELTA_BYTE_ARRAY` (`BYTE_ARRAY` and `FIXED_LEN_BYTE_ARRAY`, per the spec), and `BYTE_STREAM_SPLIT` for `FLOAT`/`DOUBLE`/`INT32`/`INT64`/`FIXED_LEN_BYTE_ARRAY`. This relaxes the stricter-than-necessary rejection added in v0.4.2 to exactly the set the writer can correctly emit.
- **Reader decodes the full opt-in encoding set (symmetry fix)**: the page reader now decodes `DELTA_BINARY_PACKED`, `DELTA_LENGTH_BYTE_ARRAY`, `DELTA_BYTE_ARRAY` (incl. FLBA), and `BYTE_STREAM_SPLIT` for `INT32`/`INT64`/`FIXED_LEN_BYTE_ARRAY` in both V1 and V2 data pages. Previously these could be written but not read back by carquet itself; `DELTA_BYTE_ARRAY` reconstruction reuses the page-retain lifetime so byte-array values stay valid across a multi-page batch (ASAN-clean). A new `test_encoding_roundtrip` test asserts carquet-reads-carquet exactness for every encoding, including a nullable column.
- **`INTERVAL` logical type**: schema/metadata support for the `INTERVAL` annotation (legacy `ConvertedType=INTERVAL`, `FIXED_LEN_BYTE_ARRAY` of length 12); written and read back, and recognized by PyArrow.
- **`sorting_columns` metadata**: The declared sort order of row groups is now written into and read back from row-group metadata instead of being skipped. The order is recorded on every row group (matching PyArrow); the writer records the declaration only and does not sort or verify the data.
- **Per-column bloom NDV/FPP** and **`max_rows_per_page`**: bloom filters can be sized per column; data pages can be flushed by row count in addition to byte size.
- **INT96 writing**: the deprecated `INT96` physical type can now be written (PLAIN, the only valid INT96 encoding). No min/max statistics are produced, matching parquet-cpp's undefined INT96 sort order; PyArrow reads it back as `timestamp[ns]`.
- **Data Page V2 writing (opt-in)**: `carquet_writer_options_t.data_page_version = 2` writes `DATA_PAGE_V2` — repetition/definition levels stored uncompressed and outside the compressed value region, byte lengths carried in `DataPageHeaderV2` (no inline 4-byte level prefix), `num_rows`/`num_nulls` tracked per page. Any value other than 2 keeps the V1 path, which is byte-for-byte unchanged (the level-prefix change and row counting are guarded behind the V2 flag, zero hot-path cost).
- **`ARROW:schema` footer metadata (opt-in)**: with `carquet_writer_options_t.write_arrow_schema = true`, the original Arrow schema is embedded as a base64-encoded encapsulated Arrow IPC Schema message under the `ARROW:schema` key, so Arrow/PyArrow recover Arrow-specific type information losslessly. The Arrow IPC FlatBuffer and base64 are produced by a small hand-written builder (`src/writer/arrow_schema.c`) — no flatbuffers/Arrow dependency. Emitted only for flat schemas and only when the user has not already set that key; nested schemas are left without it rather than written inconsistently.
- **FLOAT16 statistics ordering**: min/max for the `FLOAT16` logical type (`FIXED_LEN_BYTE_ARRAY(2)`) are now computed by the represented floating-point value with NaNs skipped, and a zero min/max is normalized to `-0.0`/`+0.0` (per spec), instead of the incorrect byte-lexicographic ordering. Applied at both the page and column-merge level.
- **Deprecated BIT_PACKED level decoding (read)**: the reader now decodes legacy Data Page V1 files whose definition/repetition levels use the deprecated `BIT_PACKED` encoding (MSB-first, no length prefix), dispatching on the page header's level-encoding fields instead of assuming RLE. New `carquet_decode_bitpacked_levels()` in `core/bitpack`.
- **GeospatialStatistics for GEOMETRY/GEOGRAPHY**: the writer now computes and emits `ColumnMetaData.geospatial_statistics` (field 17): a coordinate bounding box (`xmin/xmax/ymin/ymax`, plus `z`/`m` when present) and the set of ISO-WKB geometry type codes, parsed from the column's WKB values (Point/LineString/Polygon/Multi*/GeometryCollection, both endiannesses, XY/XYZ/XYM/XYZM, EWKB-flag tolerant). New `core/geo_wkb` walker; thrift encode + decode added. Min/max remain suppressed for these types as before.
- **TIMESTAMP coercion**: `carquet_writer_options_t.coerce_timestamps` rescales every `TIMESTAMP` column to `coerce_timestamp_unit` on write and emits the metadata (modern + legacy `ConvertedType`) at that unit, mirroring PyArrow's `coerce_timestamps`; `allow_timestamp_truncation` gates lossy finer→coarser conversion (mirrors `allow_truncated_timestamps`). Off by default.
- **`write_batch_size`**: `carquet_writer_options_t.write_batch_size` caps the internal value-batch size used during column writing (mirrors PyArrow's `write_batch_size`); 0 keeps the automatic page-size-derived heuristic.
- A new `test_writer_extensions` test covers all of the above, asserting on-disk `DATA_PAGE_V2` headers, FLOAT16 numeric stats, the BIT_PACKED spec worked-example, footer GeospatialStatistics, TIMESTAMP coercion (incl. the disallowed-truncation error path) and `write_batch_size` correctness, and Arrow/DuckDB round-trip.

### API

- Added `carquet_sorting_column_t` and `carquet_writer_set_sorting_columns()` to declare row-group sort order.
- Added `carquet_writer_set_column_bloom_filter_options()` (per-column NDV + FPP), the `CARQUET_LOGICAL_INTERVAL` logical type, and the `max_rows_per_page` writer option (0 = unlimited, default; zero hot-path cost when unset). The existing `carquet_writer_set_column_bloom_filter()` and default behavior are unchanged.
- Added `carquet_writer_options_t.write_arrow_schema` (bool, default false) and `carquet_writer_options_t.data_page_version` (int32, default 1). Both default to the pre-existing behavior, so existing code and default output are unaffected.
- No public API change for FLOAT16 stats or BIT_PACKED level reading — these are automatic (FLOAT16 statistics are written whenever the logical type is used; BIT_PACKED is a read-path addition).
- Added `carquet_reader_geospatial_statistics()` and `carquet_geospatial_statistics_t` to read back the bounding box + ISO-WKB type codes a GEOMETRY/GEOGRAPHY chunk carries. The `carquet stat` CLI now prints these (`bbox x[..] y[..] types[..]`). Verified interoperable: DuckDB 1.4 reads carquet GEOMETRY columns as native `GEOMETRY` (correct WKT + matching extent), and Arrow/DuckDB read the FLOAT16, INT96, and Data Page V2 files with correct types and values.
- Added `carquet_writer_options_t.coerce_timestamps`, `coerce_timestamp_unit`, `allow_timestamp_truncation`, and `write_batch_size`. All default to the pre-existing behavior (no coercion, automatic batching), so existing code and default output are unaffected.
- No change to default on-disk encoding: it remains v0.4.4's policy (PLAIN, with automatic `BYTE_STREAM_SPLIT` for FLOAT/DOUBLE when a compression codec is set). Dictionary, `DELTA_*`, and the widened `BYTE_STREAM_SPLIT` are all opt-in via `carquet_writer_set_column_encoding()`. Default output bytes are unchanged from v0.4.4.
- Implemented the five previously-unlinkable public functions (`carquet_reader_open_file`, `carquet_get_file_info`, `carquet_validate_file`, `carquet_set_allocator`, `carquet_get_allocator`); no signature changes. `carquet_file_info_t.created_by` changed from `const char*` to an inline `char created_by[CARQUET_CREATED_BY_MAX]` (256) so the creator string is owned by the caller's struct with no separate free and no dangling lifetime; added the `CARQUET_CREATED_BY_MAX` macro. This is a struct-layout change but has no practical compatibility impact (see Compatibility — `carquet_get_file_info()` never linked before). `carquet_set_allocator(NULL)` (or an allocator with any NULL hook) resets to the libc default.
- Clarified the batch-reader lifetime contract in the public headers: a batch from `carquet_batch_reader_next()` — and every data, null-bitmap, and dictionary pointer obtained from it — is owned by the batch reader and invalidated by the next `next()` call on that reader or by freeing it; copy out anything you need to keep across batches. This documents existing pooled-buffer behavior accurately; there is no code or behavior change.

### Internal

- Fixed pre-existing undefined behavior in the test suite surfaced under UBSan: unaligned `int64_t`/`int32_t`/`double` loads of zero-copy mmap batch views in `test_mmap` (now read via memcpy helpers) and a signed-integer-overflow in a `test_writer_extensions` LCG data generator (now computed in `uint32_t`). Library behavior is unaffected; the full suite is ASan+UBSan-clean with `halt_on_error=1`.

## v0.4.4

### Bug Fixes

- **Fixed page checksum interoperability**: Page CRCs now use standard IEEE CRC32 as required by the Parquet spec, so files validate with readers that enable page checksum verification.
- **Fixed LZ4 writer metadata**: `CARQUET_COMPRESSION_LZ4` writer requests are normalized to the interoperable Parquet `LZ4_RAW` codec because carquet writes raw LZ4 blocks, not the deprecated framed LZ4 format.
- **Fixed Bloom filter block selection**: Split block Bloom filters now use the Parquet-specified block selection formula.
- **Tightened schema validation**: Primitive logical type annotations are now validated against their allowed physical types and parameters before being added to a schema.
- **Added newer logical type metadata**: The writer/parser now supports Parquet `VARIANT`, `GEOMETRY`, and `GEOGRAPHY` logical annotations, including geography edge interpolation metadata.
- **Added column order metadata and stricter statistics semantics**: File metadata now emits `column_orders`, unsigned integer statistics use unsigned ordering, floating-point statistics ignore NaN values, and undefined-order logical types suppress min/max statistics.

### Performance

- **Hardware-accelerated page checksums**: The IEEE CRC32 used for page checksums now routes through zlib's `crc32`, which ships PCLMULQDQ-folding on x86 and FEAT_CRC32 on ARMv8 instead of a scalar slicing-by-8 loop. CRC is computed for every page on both write and read, so this lifts the whole pipeline — most dramatically for fast codecs where the checksum dominated (e.g. `large/snappy` read ~160ms → ~22ms, write ~824ms → ~231ms on Apple M3). Removes the now-unused CRC32C SIMD code paths.
- **Re-fused statistics collection**: Fixed-width PLAIN encoding again computes column min/max in the same SIMD pass that copies values into the page buffer, instead of a separate scalar pass, halving the memory traffic of the write hot path. Float/double min/max use the SIMD path with a NaN-skipping rescan only when NaNs are present.

### API

- Public API added `CARQUET_LOGICAL_VARIANT`, `CARQUET_LOGICAL_GEOMETRY`, `CARQUET_LOGICAL_GEOGRAPHY`, geospatial edge algorithm constants, and `carquet_schema_add_variant()`. Existing `CARQUET_COMPRESSION_LZ4` writer usage is accepted, but emitted metadata now uses `LZ4_RAW`.

## v0.4.3

### Bug Fixes

- **Restored Parquet logical type backward compatibility**: Writer metadata now includes the legacy `ConvertedType` annotations required by older readers when a matching modern `LogicalType` is present, including string, date/time, timestamp, decimal, integer, JSON/BSON, enum, list, and map annotations. Decimal scale/precision are also mirrored into the legacy schema fields, and old files that only contain `ConvertedType` are normalized to carquet logical types when read.

### API

- No public API changes.

## v0.4.2

### Bug Fixes

- **Statistics are now actually written to Parquet files**: `write_statistics = true` (the default) was silently being ignored — page-level min/max/null counts were never propagated up into the column chunk metadata, so files contained no statistics regardless of the option. Column stats now flow correctly from page → column → row group → file metadata. Existing files written with older versions are unaffected; only the writer was broken.
- **Fixed `carquet stat` CLI segfault on string columns**: The `stat` subcommand crashed when displaying min/max for `BYTE_ARRAY` (string) columns. The bug was latent until statistics actually started being written.
- **Fixed nullable value decoding contract**: Optional columns now consistently use a packed non-null value stream plus definition levels, including `BYTE_ARRAY`, dictionary pages, partial page reads, and generated reader code.
- **Fixed null bitmap polarity**: Batch reader null bitmaps now consistently follow the documented convention: bit set means the value is present.
- **Fixed compressed mmap batch-reader edge cases**: The pipeline no longer drops null information for optional/repeated columns, and intra-column split tasks no longer share mutable column-reader scratch buffers.
- **Rejected unsupported writer encodings**: Per-column encoding overrides now fail fast for encodings the writer cannot actually emit, avoiding mismatched metadata and payloads.

### New Features

- **Statistics for all primitive types**: Min/max/null-count are now tracked for `BOOLEAN`, `BYTE_ARRAY`, and `FIXED_LEN_BYTE_ARRAY` in addition to the numeric types. Byte-array min/max use Parquet-spec lexicographic ordering with min/max truncation at 32 bytes for long values (max is incremented so the stored bound remains a valid upper bound).
- **New CLI commands**:
  - `carquet cat [-n LIMIT] [-s OFFSET] [-c COLS] <file>` — print rows with arbitrary slicing and column filtering. Fills the gap between `head`/`tail` (anchored to start/end) and `sample` (random).
  - `carquet export [--format csv] [-n LIMIT] [-s OFFSET] [-c COLS] <file>` — write rows to stdout as RFC 4180 CSV with the same slicing/filter options. Useful for piping into shell tools or other CSV consumers.
- **Cleaner `carquet stat` output**: columns are now auto-sized to their actual content width instead of fixed 20/30-character cells, so short numeric stats no longer create huge empty gaps and long string min/max are no longer truncated.

## v0.4.1

- Hardened page reader bounds checks for mmap and fread paths, including page payload spans, page sizes, and offset arithmetic.
- Fixed compressed Data Page V2 handling in the fread reader path by decompressing only the compressed data section while preserving uncompressed level bytes.
- Improved malformed-input resistance with checked allocation/growth arithmetic in core buffer and arena helpers.
- Tightened batch-reader coalesced reads so unsupported or suspicious page layouts fall back to the standard page reader.
- Made page writer value appends transactional on encode failure, preventing partial page state from leaking into later writes.

## v0.4.0

### New Features

- **CLI tool (`carquet`)**: Ships a built-in command-line tool for inspecting Parquet files and generating reader code. Built by default (`CARQUET_BUILD_CLI=ON`), installed globally with `make install`.
  - `schema` — print file schema
  - `info` — print detailed file metadata
  - `head` / `tail` — print first/last N rows
  - `count` — print total row count
  - `columns` — list column names (one per line)
  - `stat` — print column statistics
  - `validate` — verify file integrity
  - `sample` — print N random rows
  - `codegen` — generate C reader code from a Parquet file's schema
  - All subcommands support `-h` / `--help`.

- **Code generation (`carquet codegen`)**: Reads a real Parquet file's schema and generates a complete, compilable C program tailored to that schema.
  - `-f` / `--file` — input Parquet file to inspect (generates a placeholder path if omitted)
  - `-o` / `--output` — output source file (prints compile command on stderr)
  - `-b` / `--batch-size` — batch size in generated code
  - `-c` / `--columns` — comma-separated column filter
  - `--mmap` — generate memory-mapped I/O reader
  - `--skeleton` — generate empty `process_batch` body for custom logic
  - Auto-detects compiler (respects `$CC`), carquet include/lib paths, and link dependencies
  - Embeds the source Parquet file as default input so the generated binary works without arguments
  - Generated code compiles with zero warnings

- **Versioned manual in `docs/`**: Added focused in-repo documentation for the main workflows and API concepts.
  - `docs/README.md` — manual index and API surface guide
  - `docs/reading.md` — reader setup, batch scans, column reads, filtering, metadata inspection
  - `docs/writing.md` — schema creation, required/nullable writes, row groups, buffer writer
  - `docs/nested-data.md` — groups, lists, maps, definition levels, repetition levels
  - `docs/performance.md` — mmap, zero-copy, dictionary-preserving reads, prebuffering, tuning
  - `docs/error-handling.md` — status codes, rich error context, type mapping, and level/null conventions

- **Row group predicate pushdown in batch reader**: Added `row_group_filter` callback to `carquet_batch_reader_config_t` for zero-I/O elimination of non-matching row groups using column statistics.
- **I/O coalescing**: Added `carquet_reader_prebuffer()` to pre-read multiple column chunks in a single coalesced read.
- **Speculative footer read**: File open reads up to 64KB from the end in a single I/O call, reducing the open path from 3 I/O calls to 2 for most files.
- **Data Page V2 decoding**: Page reader support for Parquet Data Page V2.
- **Write-path profiling target**: Added the `profile_write` binary for dedicated write-path profiling.

### New APIs

- **Bloom filter query**: Read bloom filters from Parquet files and check value membership — enables column-chunk-level predicate pushdown.
  `carquet_reader_get_bloom_filter()`, `carquet_bloom_filter_check_i32/i64/float/double/bytes()`, `carquet_bloom_filter_size()`, `carquet_bloom_filter_destroy()`
- **Page index read**: Access per-page min/max statistics (column index) and page file locations (offset index) — enables page-level predicate pushdown, skipping individual pages within a column chunk.
  `carquet_reader_get_column_index()`, `carquet_column_index_num_pages()`, `carquet_column_index_get_page_stats()`, `carquet_reader_get_offset_index()`, `carquet_offset_index_get_page_location()`
- **Key-value metadata**: Read and write arbitrary string key-value pairs in the Parquet footer (used by Pandas, Arrow, Spark for schema annotations).
  `carquet_reader_num_metadata()`, `carquet_reader_get_metadata()`, `carquet_reader_find_metadata()`, `carquet_writer_add_metadata()`
- **Column chunk metadata**: Inspect per-column-per-row-group details: codec, encoding, sizes, and which optional features (bloom filter, page index) are present.
  `carquet_reader_column_chunk_metadata()`
- **Per-column writer options**: Override global encoding, compression, statistics, and bloom filter settings on a per-column basis.
  `carquet_writer_set_column_encoding()`, `carquet_writer_set_column_compression()`, `carquet_writer_set_column_statistics()`, `carquet_writer_set_column_bloom_filter()`
- **Buffer writer**: Write Parquet data to an in-memory buffer instead of a file — useful for network protocols, embedding, and testing.
  `carquet_writer_create_buffer()`, `carquet_writer_get_buffer()`

### Performance

- **Multi-row-group pipeline decompression**: Persistent worker pool with pipeline ring buffer for parallel bulk-reads. On 10M-row / 10-RG / 3-column benchmark (Apple M3): snappy 16ms (was 40ms), zstd 25ms (was 44ms), lz4 12ms (was 26ms).
- **ZSTD thread safety**: Per-thread `ZSTD_DCtx`/`ZSTD_CCtx` cache via `pthread_key_create` on all POSIX builds.
- **ARM NEON byte-stream-split**: Widened AArch64 double encode/decode hot loop to 4 doubles at a time.
- **Cheaper page-load peeks**: Zero-length batch reads share the page-loader helper.

### Bug Fixes

- **Fixed BYTE_ARRAY nullable column reads**: Writer now encodes all `num_values` entries for BYTE_ARRAY columns (including zero-length entries for nulls) so values stay aligned with definition levels. Reader's PLAIN decoder and dictionary lookup paths updated to match.
- **Fixed dictionary-encoded nullable columns**: Dictionary index decoding and lookup now process `num_values` entries instead of `non_null_count`.

### Internal

- CLI sources in `src/cli/`: `main.c`, `commands.c`, `codegen.c`, `codegen_read.c`, `codegen_write.c` (stub).
- Build option `CARQUET_BUILD_CLI` (default ON), installs `carquet` binary alongside the library.
- Batch reader pipeline serves pre-read data via zero-copy views from ring buffer slots.
- Worker pool queue capacity increased to 512 for cross-RG bulk-read task submission.
- Page reader fread path uses `prebuf_read_at()` helper for prebuffer cache.
- Windows compatibility: `_getcwd`, `_access`, `_fullpath`, `gmtime_s` behind `#ifdef _WIN32`.

## v0.3.1

- Snappy compression updates and fuzzing improvements.

## v0.3.1_2

- Minor build fix.

## v0.3.1_1

- Minor build fix.

## v0.3.0_6

- Windows build fixes.
