# Changelog

## v0.6.1

Security hardening release from a full library audit, plus one small, focused API addition. No breaking API/ABI change; existing symbols and default output bytes are unchanged.

### Packaging & Build

- **Prebuilt binaries on every release**: a new `.github/workflows/release.yml` attaches self-contained static libraries to each `vX.Y.Z` GitHub Release for macOS arm64, Linux x86-64/arm64, and Windows x64 — no source build required. Each holds `include/carquet/`, `lib/libcarquet.a` with **zstd/zlib/lz4 bundled in**, and the CMake package config; consume via `find_package(carquet)` or `-Ilib/include -Llib -lcarquet`. Built on the oldest runner (low glibc floor) and without LTO/OpenMP so it links from any consumer toolchain.
- **New `CARQUET_BUNDLE_DEPS` CMake option** (default `OFF`): forces fetch + static-bundle of zstd/zlib/lz4 for a portable self-contained archive; the normal build still prefers system libraries.
- **`find_package(carquet)` now works from an install tree**: `carquetConfig.cmake` is now installed (previously only `carquetTargets.cmake` shipped), static builds export their bundled compression archives in the install link interface (previously dropped → undefined zstd/zlib/lz4 symbols), and the config resolves `Threads`/`OpenMP` via `find_dependency`.
- **`xmake.lua` build alongside CMake**: carquet can now also be built with [xmake](https://xmake.io), mirroring the same options and test wiring; CMake remains the reference build.

### API

- **New `carquet_column_read_batch_ex()` for detailed read-error reporting** (small, additive, ABI-compatible): `carquet_column_read_batch()` collapsed four distinct failure conditions (invalid `max_values`, unknown physical type, scratch-buffer allocation failure, and page-read failure) onto the single return value `-1`, and could not signal a page-read failure that truncated a batch after some values had already been read — it returned the partial count, indistinguishable from a clean end-of-column short read. The new variant takes a trailing `carquet_error_t*` out-parameter that carries a distinct status code (and message) per failure and, on a mid-batch truncation, reports the error while still returning the salvaged count so the caller can both use the partial data and detect that it is incomplete. `carquet_column_read_batch()` is now a thin wrapper over the new function with identical behavior — no existing call site changes and no ABI break.

### Security / Bug Fixes

- **Reject columns whose chunk-metadata physical type disagrees with the schema** (heap buffer overflow, found by fuzzing): the value width is derived from the schema on the batch-reader path but from the column-chunk metadata on the page-reader path. A crafted file that declared, say, `INT32` in the schema (4 bytes/value) but `BYTE_ARRAY` in the chunk metadata (`sizeof(carquet_byte_array_t)`/value) made the page decode write past the batch reader's output buffer. The column reader now validates the two types match at construction and rejects the file with `CARQUET_ERROR_INVALID_METADATA` when they do not.
- **Reject malformed dictionary pages that could read out of bounds**: a crafted BYTE_ARRAY dictionary with an oversized entry-length prefix could slip past a size check and expose out-of-bounds bytes when the value was read. Such pages are now rejected.
- **Reject truncated PLAIN pages on the memory-mapped read path**: a crafted page header could claim more values than its payload actually held, causing an out-of-bounds read when the values were copied out (found by fuzzing). The zero-copy read path now validates the payload size before use, matching the buffered path.
- **Guard against denial-of-service from crafted footers**: a malformed schema declaring a huge child count could stall the reader in a near-endless loop; oversized footer-length fields on very large files could mis-parse. Both are now bounded and rejected cleanly.
- **Guard length-array allocations in DELTA byte-array decoders**: value counts from the page header are now range-checked before allocating, preventing a size overflow on 32-bit platforms.
- **Clear stale error on memory-map fallback**: when a file opened successfully via the buffered fallback after a memory-map attempt failed, a leftover error was left set on the caller's error object. The error is now cleared on success.
- **Avoid unaligned float/double reads in BYTE_STREAM_SPLIT for FIXED_LEN_BYTE_ARRAY**: the 4-/8-byte BYTE_STREAM_SPLIT fast paths reinterpreted the value buffer as `float*`/`double*`. For FLBA(4)/FLBA(8) that buffer carries no alignment guarantee, so the cast was undefined behavior and could `SIGBUS` on strict-alignment targets. The fast path is now taken only when the buffer is naturally aligned; otherwise the generic byte transpose runs. Output bytes are identical (the aligned FLOAT/DOUBLE paths are unaffected).
- **Bounds-check the BYTE_ARRAY dictionary fallback scan** (defense-in-depth): the unreachable fallback that scans a dictionary entry-by-entry computed `4 + len` in 32-bit arithmetic with no per-entry bound, so a wrapped length could have walked past the dictionary buffer. The scan is now `size_t`-widened and bounded against the dictionary size. This path is not reachable by current readers (the offset table is always built); the guard prevents future misuse.

### Interoperability

- **Read DELTA_BINARY_PACKED pages from any conformant writer**: the decoder previously accepted only the 128-values-per-block / 4-mini-block layout that carquet itself writes and rejected other spec-valid layouts (e.g. block sizes of 256 or 512). It now accepts any block size that follows the Parquet spec, so DELTA-encoded columns written by Arrow, parquet-mr and others read correctly.
- **Keep predicate pushdown enabled for empty-string min/max stats**: presence of a column's `min_value`/`max_value` is now tracked from the Thrift field itself rather than inferred from its length, so a foreign file whose BYTE_ARRAY/STRING minimum is the empty string is no longer treated as having no statistics. Previously such a column silently disabled row-group pruning (a missed optimization, never a wrong result). Reader-side only; carquet's own writer output is unchanged.

### Performance

- **Faster decode of wide bit-packed values (~1.8–2.0× on Apple Silicon)**: the general bit-unpacking path for widths of 9–32 bits — used to decode dictionary indices for dictionaries with more than 256 entries, and wide RLE/DELTA runs — was rewritten from a byte-at-a-time loop into a branchless load-shift-mask. Output is byte-for-byte identical.

### Robustness

- **Serialize lazy SIMD dispatch initialization**: the runtime SIMD function table is now populated under a lock (matching the library's other lazy init), removing a data race when the first SIMD use happens concurrently across threads (e.g. parallel column reads).
- **Prevent SIMD instructions from leaking into portable code paths under LTO** (x86 builds): the AVX/AVX-512 translation units are compiled without LTO so ISA-specific code cannot be inlined into baseline paths that run before the runtime CPU check. `CARQUET_NATIVE_ARCH` is now documented as producing a host-only, non-portable binary.
- **Remove a data race on the parallel column-read error flag**: the OpenMP column-read loop shared a single `read_error` flag written by every thread. Each column now writes its own error slot, reduced after the parallel region, so there is no shared write (the previous torn-write was benign in practice but a genuine data race).

## v0.6.0

Source-compatible additive release. Closes the two remaining "small" Arrow-parity gaps. ABI break only in the strict sense that `carquet_writer_options_t` gains a trailing `int32_t` field; existing code that goes through `carquet_writer_options_init()` is unaffected and default output bytes are unchanged.

### New Features

- **Page-level filtering (predicate pushdown to data pages)**: new `carquet_batch_reader_set_page_filter(reader, clauses, count)` attaches a conjunction of clauses to the batch reader. Each clause is `(column_index, op, value)`; supported ops are `EQ` / `NE` / `LT` / `LE` / `GT` / `GE` / `RANGE` / `IN` / `IS_NULL` / `IS_NOT_NULL`. The filter is evaluated against the column index, and only pages whose min/max range could match are decompressed and decoded. Predicate columns do not need to be in the projection — when they are not, only their column + offset index is read, never the data pages. Clauses AND'd across different columns are supported, with per-row-group row range intersection. Both sequential and pipeline batch-reader paths are filter-aware: the pipeline pre-reads only matching pages into the slot buffers in parallel. Diagnostic `carquet_batch_reader_rows_skipped()` reports the running pruned-row count. `INT96` predicates are rejected (no defined sort order); a clause against a column written without `write_page_index` surfaces `CARQUET_ERROR_PAGE_INDEX_REQUIRED` on the next `next()` call. New error code `CARQUET_ERROR_PAGE_INDEX_REQUIRED`, new types `carquet_filter_op_t` / `carquet_filter_clause_t`, implementation in `src/reader/page_filter.{c,h}` with a new column reader `seek_to_data_page` primitive in `src/reader/page_reader.c`.
- **Append row groups to existing files**: new `carquet_writer_open_append(path, schema, options, error)` opens an existing Parquet file with read+write access, parses the trailing footer, validates that the supplied schema matches the file's leaf columns (count / name / physical type / repetition), and returns a writer positioned at the byte just before the existing footer. Subsequent `write_batch` / `new_row_group` calls add row groups; `close()` writes a fresh footer listing the existing row groups followed by the new ones. Existing bloom filters and page indexes are preserved (they sit between row group data and the original footer, which is the region the writer overwrites). Key-value metadata is carried over; further `add_metadata` calls accumulate. Schema mismatch surfaces as `CARQUET_ERROR_INVALID_SCHEMA` and leaves the file unmodified. This enables streaming ingestion, log accumulation, and periodic-snapshot workflows without read-then-rewrite.
- **CLI: row-predicate filtering via `--filter`**: `carquet cat`, `count`, `head`, and `export` accept `-p / --filter EXPR` to push a row predicate down to the page level. Grammar: `column OP value [AND column OP value]...`, with `OP` ∈ {`=`, `==`, `!=`, `<>`, `<`, `<=`, `>`, `>=`}, plus `column IS NULL` and `column IS NOT NULL`. Values are signed numbers, single-quoted strings (with `''` escape), or `TRUE`/`FALSE`; the literal is parsed against the column's physical type. Pages whose column-index min/max cannot match the predicate are skipped without decompression. `tail` and `sample` reject `--filter` with a helpful workaround hint (`cat -p ... | tail`, `cat -p ... | shuf | head`) since they would require materializing every matching row. When the file lacks a page index for a referenced column, the CLI surfaces a clear "re-write with write_page_index = true" message instead of returning zero rows. Internally the CLI builds a `carquet_filter_clause_t[]` and routes filtered reads through the batch reader; unfiltered reads keep their existing column-reader fast path so behavior on plain `cat`/`count` is byte-for-byte unchanged.
- **CLI: `info` now surfaces page-index and sort-order metadata**: when at least one column has a page index, `info` prints a per-column table with the page count and the column-index `boundary_order` (`UNORDERED` / `ASCENDING` / `DESCENDING`); when the file's row groups carry `sorting_columns`, `info` prints a sort-order section listing each declared sort column with its direction and NULLs placement. Both sections are omitted when the metadata is absent, so output on existing files without those structures is unchanged.
- **CLI: `INTERVAL` logical type displayed correctly**: schema/info/stat output now annotates `FIXED_LEN_BYTE_ARRAY(12)` columns carrying the `INTERVAL` annotation as `INTERVAL` instead of falling back to the bare physical type. Affects display only; on-disk format is unchanged.
- **Custom compression codec registration**: new public API `carquet_register_codec(codec, impl)` + `carquet_custom_codec_t` (compress / decompress / compress_bound + opaque `user_data`). A registered codec takes priority over the built-in for that slot in both the reader (`carquet_decompress_page`) and writer (`compress_data`), so users can swap in a hardware-accelerated implementation or fill the `LZO` / `BROTLI` slots that carquet has no built-in for. Passing `impl = NULL` unregisters; registering against `UNCOMPRESSED` is rejected (that path has a no-copy fast lane that must not be intercepted). Registrations are process-wide and not safe to mutate concurrently with reader/writer activity. Implementation: `src/compression/custom.{h,c}`.
- **Per-column page-size override**: new `carquet_writer_set_column_page_size(writer, column_index, bytes)` overrides `options.page_size` for one column. Useful when some columns benefit from smaller pages (finer page-index pruning) and others from larger pages (lower per-page header overhead). Mirrors the existing per-column compression/encoding override pattern.
- **Configurable BYTE_ARRAY statistics truncation cap**: new `carquet_writer_set_max_statistics_size(writer, bytes)` controls the byte cap for variable-length min/max stored in column statistics (default `32`, matching Arrow and the Parquet spec recommendation). The existing truncation logic — min stored as leading prefix, max stored as prefix-then-increment, all-`0xFF` max suppressed rather than emitted as an invalid bound — is unchanged; only the threshold becomes tunable. The `is_min/max_value_exact` flags continue to reflect whether the stored value equals the actual column min/max.
- **File format version control**: new `carquet_writer_options_t.file_format_version` (default `2`) writes the requested value into `FileMetaData.version` in the footer. Set to `1` for the rare older readers that reject version-2 files; any other value is treated as `2`. Independent of `data_page_version`.

### Bug Fixes

- **Nullable batched-write value mis-indexing**: when `carquet_writer_write_batch` was called on an OPTIONAL column with a batch large enough to be split into multiple page chunks (i.e. larger than `write_batch_size` or the page-size-derived heuristic), the eager encode path indexed the dense values array by the logical row offset instead of the cumulative non-null count, so chunks beyond the first read values from the wrong rows of the dense array. The values array is now advanced by the per-chunk non-null count as the spec requires. Fixes silent corruption of OPTIONAL columns whose writer batches were larger than one page worth of rows.
- **`codegen` aborting with "buffer overflow detected" in Release builds**: `cmd_codegen_read` resolved the input path with `realpath()` into a 1024-byte buffer, but `realpath()` requires a buffer of at least `PATH_MAX` (4096 on Linux). On distros that enable `_FORTIFY_SOURCE` for optimized builds (e.g. Ubuntu's GCC at `-O2`), `__realpath_chk` aborts unconditionally because the buffer is smaller than `PATH_MAX`, so `carquet codegen -f file.parquet` died before producing output; unfortified Debug builds skip the check and worked. The buffer is now sized to `PATH_MAX`/`_MAX_PATH` (and the escaped-path buffer widened to match), in `src/cli/codegen_read.c`.
- **Page index offsets now absolute**: `OffsetIndex.PageLocation.offset` is now an absolute file offset per the Parquet spec. Previously, the writer recorded relative-within-column offsets because eager page flushes ran before the row-group writer set the column's absolute file position. The column writer now stores per-column relative offsets while flushing and applies the column file offset as a single shift at finalize time. Downstream readers that consume `OffsetIndex` (now including carquet's own page filter) get spec-compliant offsets.
- **Malformed Data Page V1 levels when a page spanned multiple value chunks**: for an OPTIONAL or repeated column whose page accumulated more than one `add_values` chunk before a byte-size-triggered flush (the common case with a small `page_size`), each chunk's definition/repetition levels were emitted with their own 4-byte RLE length prefix, so the page's level section became several concatenated `[len][rle]` runs while the page header advertised a single level stream. carquet's own reader tolerated this, but PyArrow (`Number of decoded rep / def levels do not match num_values in page header`) and DuckDB (`Out of buffer`) rejected such files. Levels are now stored as raw RLE with the single V1 length prefix written once at page assembly, so multi-chunk pages are spec-conformant. Output for single-chunk pages (the default for large pages) is byte-identical; row-count-split pages (`max_rows_per_page`) and Data Page V2 were already correct. Found by the new `zstd+multipage` interop variant.

### Performance

- **NEON BYTE_STREAM_SPLIT for double rewritten with structure load/stores**: the AArch64 transpose for `double` columns (the default encoding for floats under compression) now uses `vld4q_u16` / `vst4q_u16` instead of `vqtbl` table lookups. `LD4`/`ST4` de-interleave at the load-store unit and far outpace the two-register `vqtbl2q` gather on Apple Silicon. Measured on M3: double **decode +47–69%** (read path) and encode well above the old table path, byte-exact identical output. `float` BSS stays on the auto-vectorized scalar path — benchmarked faster than both `vqtbl` and `vld4` there, where the latter regresses small in-cache decode. Implementation in `src/simd/arm/neon_ops.c`, wiring in `src/simd/dispatch.c`.

### Testing

- **Fuzzing seeded from the Apache `parquet-testing` corpus**: the fuzz harness (`fuzz/fetch_corpus.sh` + `run_fuzzer.py`) now pulls Apache's official `parquet-testing` data files — real-world files spanning varied types, encodings, codecs, nested/nullable, multi-row-group and page-indexed layouts, plus the intentionally malformed inputs under `bad_data/` — and uses them as seeds for the reader and page-filter targets. Fuzzing carquet directly against the canonical Parquet reference corpus (rather than only self-generated seeds) is independent confirmation that carquet correctly parses spec-conformant files and rejects malformed ones. New targets `fuzz_append` and `fuzz_page_filter` extend coverage to the append and page-filter paths.

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
