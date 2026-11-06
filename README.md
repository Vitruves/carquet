<h1 align="center">
  <img src="res/img/carquet_banner.png" alt="Carquet" width="840" />
</h1>

<p align="center">
  A fast, pure C library for reading and writing Apache Parquet files.
</p>

<p align="center">
  <a href="https://vitruves.github.io/carquet/"><img src="https://img.shields.io/badge/Website-vitruves.github.io%2Fcarquet-2a78d6?style=for-the-badge" alt="Website" /></a>
</p>

<p align="center">
  <a href="https://github.com/Vitruves/carquet/actions/workflows/cpp.yml"><img src="https://github.com/Vitruves/carquet/actions/workflows/cpp.yml/badge.svg" alt="Build" /></a>
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue" alt="Platform" />
  <img src="https://img.shields.io/badge/C-C11-blue" alt="C Standard" />
  <a href="https://github.com/Vitruves/carquet/releases"><img src="https://img.shields.io/github/v/release/Vitruves/carquet?color=blue" alt="Release" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License" /></a>
  <a href="https://github.com/Vitruves/carquet/graphs/traffic"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Vitruves/carquet/traffic/clones.json" alt="Clones" /></a>
  <br/>
  <a href="https://pypi.org/project/carquet/"><img src="https://img.shields.io/pypi/v/carquet?color=blue&label=pypi" alt="PyPI" /></a>
  <a href="https://pypi.org/project/carquet/"><img src="https://img.shields.io/pypi/pyversions/carquet" alt="Python versions" /></a>
  <a href="https://pepy.tech/projects/carquet"><img src="https://img.shields.io/pepy/dt/carquet?color=blue&label=pypi%20downloads" alt="PyPI downloads" /></a>
  <img src="https://img.shields.io/badge/SIMD-SSE4.2%20%C2%B7%20AVX2%20%C2%B7%20AVX--512%20%C2%B7%20NEON%20%C2%B7%20SVE-eb6834" alt="SIMD" />
</p>

<p align="center">
  <a href="#highlights">Highlights</a> &nbsp;·&nbsp;
  <a href="#benchmarks">Benchmarks</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#build-configurations">Build configurations</a> &nbsp;·&nbsp;
  <a href="#c-api">C API</a> &nbsp;·&nbsp;
  <a href="#python-api">Python API</a> &nbsp;·&nbsp;
  <a href="#cli-tool">CLI</a> &nbsp;·&nbsp;
  <a href="#parquet-feature-support">Features</a> &nbsp;·&nbsp;
  <a href="docs/README.md">Manual</a> &nbsp;·&nbsp;
  <a href="CHANGELOG.md">Changelog</a>
</p>

<p align="center">
  <b>English</b> &nbsp;·&nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp;·&nbsp;
  <a href="README.ja.md">日本語</a> &nbsp;·&nbsp;
  <a href="README.ko.md">한국어</a> &nbsp;·&nbsp;
  <a href="README.de.md">Deutsch</a> &nbsp;·&nbsp;
  <a href="README.fr.md">Français</a> &nbsp;·&nbsp;
  <a href="README.pt-BR.md">Português (Brasil)</a>
</p>

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### Highlights

<div align="center">

| | |
|---|---|
| <img src="res/img/icons/fast.svg" width="22" height="22" align="absmiddle" alt="" />&nbsp; **Fast** | Writes **2.3–5.9x faster** than Arrow C++ and reads compressed data **1.5–2.1x faster** from the very same file (Apple M3, 10M rows). Uncompressed reads are served straight from the mmap in **0.30 ms**. |
| <img src="res/img/icons/simd.svg" width="22" height="22" align="absmiddle" alt="" />&nbsp; **SIMD** | 30+ hand-written vector kernels for **SSE4.2, AVX, AVX2, AVX-512, NEON and SVE**: bit-unpacking and RLE levels, dictionary gathers, byte-stream split, min/max statistics, null bitmaps, Snappy / LZ4 match copies. One portable binary picks the widest kernel at runtime, with a scalar fallback. |
| <img src="res/img/icons/small.svg" width="22" height="22" align="absmiddle" alt="" />&nbsp; **Small** | Pure C11, a **~500 KB** library (Arrow C++: 50 MB+). Four compression dependencies (zstd, zlib, lz4, brotli), fetched automatically when missing. Builds in under a minute with a single `make` or `xmake`. |
| <img src="res/img/icons/complete.svg" width="22" height="22" align="absmiddle" alt="" />&nbsp; **Complete** | All physical types and encodings, Snappy / GZIP / LZ4 / ZSTD / Brotli, V1 + V2 pages, nested schemas at any depth, bloom filters, page indexes, statistics-based pruning, append. [Full matrix ↓](#parquet-feature-support) |
| <img src="res/img/icons/interop.svg" width="22" height="22" align="absmiddle" alt="" />&nbsp; **Interoperable** | Files read by PyArrow, DuckDB and Spark out of the box. Zero-dependency [Arrow C Data Interface](docs/reading.md#export-to-arrow-c-data-interface) bridge (struct / list / map at any depth) and a full [Python API](#python-api) on top of PyArrow. |
| <img src="res/img/icons/tooling.svg" width="22" height="22" align="absmiddle" alt="" />&nbsp; **Tooling** | `carquet` CLI to inspect, filter, export and validate files, plus `codegen` to emit a ready-to-compile C reader. A ready-made [agent skill](#ai-coding-agents) for Claude Code and Codex. |

</div>

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### Benchmarks

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="res/img/bench_10m_m3_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="res/img/bench_10m_m3_light.png">
    <img src="res/img/bench_10m_m3_light.png" alt="Write and read time for 10 million rows on Apple M3: carquet vs Arrow C++ and Python Parquet libraries, per codec" width="900">
  </picture>
</p>

<p align="center"><b><a href="https://vitruves.github.io/carquet/">Explore the interactive benchmark on the website →</a></b><br/><sub>switch between read and write, and between 100K, 1M and 10M rows</sub></p>

Speed-up of carquet over Arrow C++ at 10M rows (higher = carquet faster):

<div align="center">

| | ARM · Apple M3 | | x86 · Windows (Haswell, MSVC) | | x86 · Linux (Xeon D-1531) | |
|---|---:|---:|---:|---:|---:|---:|
| **Codec** | **Write** | **Read** | **Write** | **Read** | **Write** | **Read** |
| snappy | **4.74x** | **1.89x** | **3.39x** | **1.17x** | **1.55x** | **1.25x** |
| zstd | **5.91x** | **1.77x** | **3.83x** | **1.13x** | **1.31x** | **1.04x** |
| lz4 | **4.61x** | **1.89x** | **2.96x** | **1.05x** | **1.02x** | 0.83x |
| none | **2.30x** | **46.3x**\* | **2.10x** | **62.0x**\* | **1.13x** | **40.6x**\* |

</div>

<sub>\* mmap zero-copy path. Apple M3: carquet 0.7.3 vs Arrow C++ 26.0.1 (2026-11-06). Windows: carquet 0.7.3 vs Arrow C++ 24.0.0 (2026-11-02). Linux: carquet 0.4.4 vs Arrow C++ 23.0.1, before the 0.7.1 writer parallelism, not re-measured since.</sub>

Compressed reads decode every value, and on Linux and macOS both libraries link the same system lz4 / zstd, so codec speed is identical. The fairest comparison is the **same-file cross-read** (both libraries read the exact same file): carquet is **1.5–2.1x faster** on M3, 1.6–2.6x on Linux and 1.04–1.30x on the Windows VM, whose cold reads are bound by the virtual disk.

<details>
<summary><b>Benchmark methodology</b></summary>

<br/>

All benchmarks use identical data (deterministic LCG PRNG), identical Parquet settings (no dictionary, BYTE_STREAM_SPLIT for floats, page checksums, mmap reads), trimmed median of 11–51 iterations, with the OS page cache purged between write and read phases and a cooldown between configurations. Schema: 3 columns (INT64, DOUBLE, INT32). Compared against the Arrow C++ low-level Parquet reader (bypassing Arrow Table materialization) and PyArrow.

The page cache is purged per file: `posix_fadvise(DONTNEED)` on Linux; on macOS and Windows the file is copied without caching (`F_NOCACHE`, `FILE_FLAG_NO_BUFFERING`) and the copy replaces it (`benchmark/bench_purge_win32.h`). On Windows, carquet links the zstd / lz4 it builds itself and Arrow C++ the conda-forge ones, so codec speed can differ slightly between the two.

The **same-file cross-read** benchmark is the fairest comparison: both libraries read the exact same Parquet file (written by one, read by both). This eliminates differences in page sizes, encoding choices, and row group layout.

**Uncompressed reads** marked with \* use Carquet's **mmap zero-copy path**: for PLAIN-encoded, uncompressed, fixed-size, required columns, the batch reader returns pointers directly into the memory-mapped file with no memcpy. Arrow always materializes into its own buffers. **The compressed read numbers are the most representative measure of end-to-end read throughput.**

</details>

<details>
<summary><b>Full x86 results</b> (Intel Xeon D-1531, Linux)</summary>

<br/>

*12 threads @ 2.2GHz, 32GB RAM, Ubuntu 24.04, ZSTD level 1*

##### 10M rows vs Arrow C++

<div align="center">

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio | Size |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | **1557ms** | 1766ms | **1.13x** | **1.25ms** | 50.8ms | **40.6x**\* | 190.7MB |
| snappy | **1002ms** | 1549ms | **1.55x** | **78ms** | 97.8ms | **1.25x** | 125.1MB |
| zstd | **1311ms** | 1714ms | **1.31x** | **76.8ms** | 80.2ms | **1.04x** | 95.3MB |
| lz4 | **1521ms** | 1554ms | **1.02x** | 59.1ms | **49.0ms** | 0.83x | 122.9MB |

</div>

<br/>

##### 1M rows vs Arrow C++

<div align="center">

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **180ms** | 196ms | **1.09x** | **0.22ms** | 6.2ms | **28x**\* |
| snappy | **141ms** | 148ms | **1.05x** | **8.1ms** | 11.6ms | **1.44x** |
| zstd | **131ms** | 185ms | **1.41x** | 10.3ms | **9.1ms** | 0.88x |
| lz4 | **143ms** | 149ms | **1.04x** | 8.5ms | **6.1ms** | 0.72x |

</div>

<br/>

##### 100K rows vs Arrow C++

<div align="center">

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **14.1ms** | 18.4ms | **1.30x** | **0.11ms** | 2.18ms | **19.8x**\* |
| snappy | **10.1ms** | 10.6ms | **1.05x** | **1.27ms** | 5.97ms | **4.70x** |
| zstd | **8.7ms** | 14.1ms | **1.62x** | **1.58ms** | 3.88ms | **2.46x** |
| lz4 | **9.6ms** | 11.0ms | **1.14x** | **0.77ms** | 2.78ms | **3.61x** |

</div>

<br/>

##### Same-file cross-read (10M rows)

Both libraries read the **same** Parquet file, the fairest apples-to-apples comparison.

<div align="center">

| Codec | Writer | Carquet Read | Arrow C++ Read | Ratio |
|-------|--------|-------------|----------------|-------|
| none | Carquet | **0.99ms** | 73.6ms | **74x**\* |
| none | Arrow | **7.6ms** | 51.2ms | **6.8x**\* |
| snappy | Carquet | **41.0ms** | 107ms | **2.61x** |
| snappy | Arrow | **43.4ms** | 101ms | **2.33x** |
| zstd | Carquet | **46.1ms** | 88.4ms | **1.92x** |
| zstd | Arrow | **49.1ms** | 79.5ms | **1.62x** |
| lz4 | Carquet | **34.8ms** | 74.8ms | **2.15x** |
| lz4 | Arrow | **27.4ms** | 52.0ms | **1.90x** |

</div>

<br/>

##### 10M rows vs PyArrow

<div align="center">

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **1557ms** | 1806ms | **1.16x** | **1.25ms** | 213ms | **170x**\* |
| snappy | **1002ms** | 1649ms | **1.65x** | **78ms** | 384ms | **4.91x** |
| zstd | **1311ms** | 1796ms | **1.37x** | **76.8ms** | 369ms | **4.81x** |
| lz4 | **1521ms** | 1676ms | **1.10x** | **59.1ms** | 281ms | **4.76x** |

</div>

\* Zero-copy mmap path

</details>

<details>
<summary><b>Full ARM results</b> (Apple M3, macOS)</summary>

<br/>

*Carquet 0.7.3, MacBook Air M3, 16GB RAM, macOS 26.5, Arrow C++ 26.0.1, PyArrow 26.0.1, fastparquet 2026.9.0, ZSTD level 1, measured on 2026-11-06. Raw report: [`benchmark/bench_0.7.3_darwin_arm64_20261106.json`](benchmark/bench_0.7.3_darwin_arm64_20261106.json).*

##### 10M rows vs Arrow C++

<div align="center">

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio | Size |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **48.8ms** | 112ms | **2.30x** | **0.30ms** | 13.9ms | **46.3x**\* | 200.0MB |
| snappy | **57.2ms** | 271ms | **4.74x** | **12.6ms** | 23.8ms | **1.89x** | 131.2MB |
| zstd | **59.7ms** | 353ms | **5.91x** | **19.8ms** | 35.0ms | **1.77x** | 99.9MB |
| lz4 | **52.4ms** | 241ms | **4.61x** | **9.60ms** | 18.2ms | **1.89x** | 128.7MB |

</div>

<br/>

##### 1M rows vs Arrow C++

<div align="center">

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **6.75ms** | 13.2ms | **1.95x** | **0.07ms** | 1.72ms | **24.6x**\* |
| snappy | **7.02ms** | 24.0ms | **3.42x** | **1.58ms** | 2.49ms | **1.58x** |
| zstd | **7.97ms** | 33.6ms | **4.22x** | **2.16ms** | 3.03ms | **1.40x** |
| lz4 | **6.54ms** | 25.2ms | **3.85x** | **0.97ms** | 1.96ms | **2.02x** |

</div>

<br/>

##### 100K rows vs Arrow C++

<div align="center">

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **1.08ms** | 1.75ms | **1.62x** | **0.02ms** | 0.24ms | **12.0x**\* |
| snappy | **1.80ms** | 2.52ms | **1.40x** | **0.36ms** | 0.90ms | **2.50x** |
| zstd | **1.81ms** | 3.60ms | **1.99x** | **0.63ms** | 1.30ms | **2.06x** |
| lz4 | **1.67ms** | 2.57ms | **1.54x** | **0.29ms** | 0.61ms | **2.10x** |

</div>

<br/>

##### 10M rows vs PyArrow

<div align="center">

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **48.8ms** | 170ms | **3.49x** | **0.30ms** | 26.8ms | **89.2x**\* |
| snappy | **57.2ms** | 294ms | **5.14x** | **12.6ms** | 38.4ms | **3.05x** |
| zstd | **59.7ms** | 423ms | **7.09x** | **19.8ms** | 42.5ms | **2.15x** |
| lz4 | **52.4ms** | 283ms | **5.41x** | **9.60ms** | 30.2ms | **3.15x** |

</div>

<br/>

##### 1M rows vs PyArrow

<div align="center">

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **6.75ms** | 17.0ms | **2.52x** | **0.07ms** | 2.73ms | **39.0x**\* |
| snappy | **7.02ms** | 27.9ms | **3.97x** | **1.58ms** | 3.66ms | **2.32x** |
| zstd | **7.97ms** | 38.8ms | **4.87x** | **2.16ms** | 3.96ms | **1.83x** |
| lz4 | **6.54ms** | 29.2ms | **4.46x** | **0.97ms** | 2.98ms | **3.07x** |

</div>

<br/>

##### 100K rows vs PyArrow

<div align="center">

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **1.08ms** | 1.99ms | **1.84x** | **0.02ms** | 0.24ms | **12.0x**\* |
| snappy | **1.80ms** | 2.82ms | **1.57x** | **0.36ms** | 0.66ms | **1.83x** |
| zstd | **1.81ms** | 4.17ms | **2.30x** | **0.63ms** | 0.72ms | **1.14x** |
| lz4 | **1.67ms** | 2.93ms | **1.75x** | **0.29ms** | 0.43ms | **1.48x** |

</div>

<br/>

##### 10M rows vs fastparquet

<div align="center">

| Codec | Carquet Write | fastparquet Write | W ratio | Carquet Read | fastparquet Read | R ratio |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **48.8ms** | 65.4ms | **1.34x** | **0.30ms** | 49.0ms | **163.5x**\* |
| snappy | **57.2ms** | 204ms | **3.57x** | **12.6ms** | 110ms | **8.78x** |
| zstd | **59.7ms** | 331ms | **5.55x** | **19.8ms** | 185ms | **9.39x** |
| lz4 | **52.4ms** | 225ms | **4.30x** | **9.60ms** | 55.3ms | **5.76x** |

</div>

<br/>

##### 1M rows vs fastparquet

<div align="center">

| Codec | Carquet Write | fastparquet Write | W ratio | Carquet Read | fastparquet Read | R ratio |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **6.75ms** | 7.45ms | **1.10x** | **0.07ms** | 3.00ms | **42.9x**\* |
| snappy | **7.02ms** | 22.1ms | **3.16x** | **1.58ms** | 10.1ms | **6.39x** |
| zstd | **7.97ms** | 32.6ms | **4.09x** | **2.16ms** | 18.4ms | **8.54x** |
| lz4 | **6.54ms** | 21.6ms | **3.30x** | **0.97ms** | 5.06ms | **5.22x** |

</div>

<br/>

##### 100K rows vs fastparquet

<div align="center">

| Codec | Carquet Write | fastparquet Write | W ratio | Carquet Read | fastparquet Read | R ratio |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | 1.08ms | **1.04ms** | 0.96x | **0.02ms** | 0.52ms | **26.0x**\* |
| snappy | **1.80ms** | 2.31ms | **1.28x** | **0.36ms** | 1.20ms | **3.33x** |
| zstd | **1.81ms** | 3.45ms | **1.91x** | **0.63ms** | 2.05ms | **3.25x** |
| lz4 | **1.67ms** | 2.25ms | **1.35x** | **0.29ms** | 0.73ms | **2.52x** |

</div>

<br/>

##### Same-file cross-read (10M rows)

Both libraries read the **same** Parquet file, the fairest apples-to-apples comparison.

<div align="center">

| Codec | Writer | Carquet Read | Arrow C++ Read | Ratio |
|-------|--------|-------------|----------------|-------|
| none | Carquet | **0.43ms** | 16.9ms | **39.2x**\* |
| none | Arrow | **0.90ms** | 14.8ms | **16.5x**\* |
| snappy | Carquet | **13.6ms** | 24.5ms | **1.80x** |
| snappy | Arrow | **11.4ms** | 22.9ms | **2.01x** |
| zstd | Carquet | **19.4ms** | 28.6ms | **1.48x** |
| zstd | Arrow | **18.7ms** | 28.7ms | **1.54x** |
| lz4 | Carquet | **9.39ms** | 19.1ms | **2.04x** |
| lz4 | Arrow | **8.97ms** | 18.8ms | **2.09x** |

</div>

\* Zero-copy mmap path. fastparquet reads through pandas and cannot write BYTE_STREAM_SPLIT or page checksums, so its rows are less directly comparable.

</details>

<details>
<summary><b>Full Windows results</b> (Intel Haswell VM, MSVC)</summary>

<br/>

*Carquet 0.7.3 built with MSVC 19.44 (x64 Release, OpenMP), OpenStack VM with 4 vCPU (Intel Haswell, 3.1 GHz), 14.6GB RAM, virtual SAS disk, Windows Server 2025, Arrow C++ 24.0.0 (conda-forge), ZSTD level 1, measured on 2026-11-02. Cold reads go to the virtual disk, which narrows the read ratios; writes and the zero-copy path are CPU-bound.*

##### 10M rows vs Arrow C++

<div align="center">

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio | Size |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | **182ms** | 382ms | **2.10x** | **2.04ms** | 126.5ms | **62.0x**\* | 190.7MB |
| snappy | **175ms** | 594ms | **3.39x** | **115.6ms** | 134.9ms | **1.17x** | 125.1MB |
| zstd | **242ms** | 927ms | **3.83x** | **140.7ms** | 159.2ms | **1.13x** | 95.3MB |
| lz4 | **178ms** | 527ms | **2.96x** | **95.5ms** | 99.9ms | **1.05x** | 122.9MB |

</div>

<br/>

##### 1M rows vs Arrow C++

<div align="center">

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **15.6ms** | 37.6ms | **2.41x** | **0.44ms** | 13.72ms | **31.2x**\* |
| snappy | **25.4ms** | 61.2ms | **2.41x** | **14.01ms** | 16.31ms | **1.16x** |
| zstd | **30.4ms** | 94.3ms | **3.10x** | **17.09ms** | 18.38ms | **1.08x** |
| lz4 | **26.4ms** | 53.8ms | **2.04x** | 13.07ms | **12.89ms** | 0.99x |

</div>

<br/>

##### 100K rows vs Arrow C++

<div align="center">

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **3.53ms** | 3.95ms | **1.12x** | **0.16ms** | 2.13ms | **13.3x**\* |
| snappy | **5.60ms** | 6.23ms | **1.11x** | **2.79ms** | 3.92ms | **1.41x** |
| zstd | **7.11ms** | 9.79ms | **1.38x** | **2.86ms** | 4.36ms | **1.52x** |
| lz4 | **5.75ms** | 5.82ms | **1.01x** | 2.22ms | **2.15ms** | 0.97x |

</div>

<br/>

##### Same-file cross-read (10M rows)

<div align="center">

| Codec | Writer | Carquet Read | Arrow C++ Read | Ratio |
|-------|--------|-------------|----------------|-------|
| none | Carquet | **2.65ms** | 116.9ms | **44.1x**\* |
| none | Arrow | **5.29ms** | 126.6ms | **23.9x**\* |
| snappy | Carquet | **115.2ms** | 135.0ms | **1.17x** |
| snappy | Arrow | **112.7ms** | 146.2ms | **1.30x** |
| zstd | Carquet | **137.7ms** | 150.5ms | **1.09x** |
| zstd | Arrow | **134.9ms** | 155.3ms | **1.15x** |
| lz4 | Carquet | **92.1ms** | 95.6ms | **1.04x** |
| lz4 | Arrow | **89.7ms** | 98.1ms | **1.09x** |

</div>

\* Zero-copy mmap path

</details>

<details>
<summary><b>Running the benchmarks yourself</b></summary>

<br/>

```bash
make benchmark                                    # Release build of the benchmark programs into build/
./build/benchmark_carquet                         # carquet alone
python3 benchmark/run_benchmark.py --skip-xlarge  # full comparison with PyArrow (+ Arrow C++ if built)
```

<div align="center">

| `run_benchmark.py` flag / variable | Effect |
|---|---|
| `--quick` | Only the 10M-row configs (what the chart shows) |
| `--skip-xlarge` | Skip the 100M-row configs (~2 GB per codec, 30+ minutes) |
| `--fastparquet` | Also compare fastparquet (`pip install fastparquet pandas`) |
| `--no-pyarrow` / `--no-arrow-cpp` | Skip PyArrow / Arrow C++ |
| `--iterations N` | Fixed iteration count (default: auto, 11–51) |
| `--build-dir DIR` / `--temp-dir DIR` | Where the binaries live / where test files are written |
| `CARQUET_BENCH_ZSTD_LEVEL=3` | ZSTD level (default 1) |

</div>

Always benchmark a **Release** build: `make benchmark` reconfigures `build/` as Release; a leftover Debug `build/` makes carquet look several times slower. `make native` + `-DCARQUET_BUILD_BENCHMARKS=ON` gives the host-tuned numbers.

fastparquet uses the same data, row group size, no dictionary and zstd level, and LZ4_RAW for `lz4`. It cannot write BYTE_STREAM_SPLIT or page checksums, and it reads through pandas, so its numbers are less directly comparable than Arrow's.

The Arrow C++ comparison needs its own target (Arrow C++ found on the system, or pointed at with `CARQUET_ARROW_CPP_ROOT`):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCARQUET_NATIVE_ARCH=ON \
  -DCARQUET_BUILD_BENCHMARKS=ON -DCARQUET_BUILD_ARROW_CPP_BENCHMARK=ON \
  [-DCARQUET_ARROW_CPP_ROOT=/path/to/arrow-prefix]
cmake --build build -j
```

It uses the low-level `parquet::ParquetFileReader` API (no Arrow Table materialization) with parallel row group readers.

On Windows (Developer PowerShell for VS), Arrow C++ can come from conda-forge (`micromamba create -p C:\arrow -c conda-forge libarrow libparquet`); its DLLs must be on `PATH` when the benchmark runs:

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCARQUET_BUILD_BENCHMARKS=ON `
  -DCARQUET_BUILD_ARROW_CPP_BENCHMARK=ON -DCARQUET_ARROW_CPP_ROOT=C:/arrow/Library
cmake --build build
$env:PATH = "C:\arrow\Library\bin;$env:PATH"
python benchmark\run_benchmark.py --build-dir build --skip-xlarge --no-pyarrow
```

Keep the build and `--temp-dir` folders out of real-time antivirus scanning: Windows Defender scans every file the benchmark writes.

The README chart is drawn from the JSON a run leaves in `benchmark/`, in a light and a dark variant that GitHub picks from the reader's theme:

```bash
python3 benchmark/plot_readme_chart.py benchmark/bench_<version>_<os>_<arch>_<date>.json   # writes res/img/bench_10m_m3_{light,dark}.png
```

</details>

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### Thirty seconds of carquet

```c
#include <carquet/carquet.h>

carquet_error_t err = CARQUET_ERROR_INIT;
carquet_schema_t* schema = carquet_schema_create(&err);
carquet_schema_add_column(schema, "id", CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);

carquet_writer_t* w = carquet_writer_create("ids.parquet", schema, NULL, &err);
int64_t ids[] = {1, 2, 3};
carquet_writer_write_batch(w, 0, ids, 3, NULL, NULL);
carquet_writer_close(w);

carquet_reader_t* r = carquet_reader_open("ids.parquet", NULL, &err);
printf("%lld rows\n", (long long)carquet_reader_num_rows(r));   /* 3 rows */
carquet_reader_close(r);
carquet_schema_free(schema);
```

> **Where carquet fits.** Arrow stays the right choice when you need its ecosystem (compute kernels, datasets, Flight, encryption). carquet is for C and C++ programs, embedded systems and language runtimes that only need to read and write Parquet quickly and correctly.

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### Installation

**Requirements:** a C11 compiler (GCC 4.9+, Clang 3.4+, MSVC 2015+) and CMake 3.16+ (or [xmake](#from-source-with-xmake)). zstd, zlib, lz4 and brotli are used from the system when present and fetched otherwise. OpenMP is optional (parallel reads and writes).

<br/>

#### From source, with `make`

The simplest route. `make` is a thin wrapper that calls CMake with the right flags, there is nothing else to configure:

```bash
git clone https://github.com/Vitruves/carquet.git && cd carquet
make                  # optimized Release build into build/
sudo make install     # library, headers, CLI, pkg-config and CMake package files
```

Then link with `-lcarquet`, `pkg-config --cflags --libs carquet`, or `find_package(carquet)` in CMake.

<div align="center">

| Command | What it does |
|---|---|
| `make` | Release static library + CLI (default) |
| `make native` | Same, tuned for this CPU (`-march=native`) |
| `make shared` | Shared library instead of static |
| `make debug` / `make test` | Debug build / build and run the test suite |
| `make install PREFIX=$HOME/.local` | Install to another prefix (default `/usr/local`) |

</div>

All targets and variables: [docs/building.md](docs/building.md#make-targets).

<br/>

#### From source, with `xmake`

[xmake](https://xmake.io) needs nothing else installed: it downloads pinned zstd 1.5.6, zlib 1.3.1, lz4 1.10.0 and brotli 1.1.0 and links them statically, so the `carquet` CLI has no runtime dependency.

```bash
git clone https://github.com/Vitruves/carquet.git && cd carquet
xmake                         # Release static library + CLI into build/<plat>/<arch>/release/
xmake install -o /usr/local   # headers, library and CLI (sudo if the prefix needs it)
```

A static install contains only `libcarquet.a`, so link the codecs yourself (`-lcarquet -lzstd -lz -llz4`, plus `-lbrotlienc -lbrotlidec -lbrotlicommon` unless built with `--brotli=n`). Configure with `xmake f --shared=y` instead for a self-contained shared library with the codecs built in.

<div align="center">

| `xmake f …` then `xmake` | What it does |
|---|---|
| `--shared=y` | Shared library with the codecs built in |
| `--native_arch=y` | Tune for this CPU (`-march=native`) |
| `--brotli=n` | Drop the BROTLI codec and its dependency |
| `--openmp=n` | Build without OpenMP |
| `-m debug --tests=y` | Debug build with tests (`xmake test`) |

</div>

All options: [docs/building.md](docs/building.md#xmake).

<br/>

#### In your CMake project

```cmake
include(FetchContent)
FetchContent_Declare(carquet GIT_REPOSITORY https://github.com/Vitruves/carquet.git GIT_TAG main)
FetchContent_MakeAvailable(carquet)
target_link_libraries(myapp PRIVATE carquet)
```

Or, after `make install`: `find_package(carquet REQUIRED)` and link `carquet::carquet`.

Set options before `FetchContent_MakeAvailable` (`set(CARQUET_BUILD_CLI OFF CACHE BOOL "" FORCE)`), or pass them as `-D` flags when building carquet on its own:

<div align="center">

| Option | Default | |
|---|:---:|---|
| `CARQUET_BUILD_SHARED` | OFF | Shared instead of static library |
| `CARQUET_BUILD_CLI` | ON | `carquet` CLI (turn OFF when embedding) |
| `CARQUET_NATIVE_ARCH` | OFF | `-march=native`, host-only binary |
| `CARQUET_ENABLE_BROTLI` | ON | OFF drops the libbrotli dependency |
| `CARQUET_BUNDLE_DEPS` | OFF | Always fetch and statically bundle the codecs |
| `CARQUET_ZSTD_TARGET` (and `_ZLIB_`, `_LZ4_`, `_BROTLI_`) | empty | Link your project's existing codec target |

</div>

All options: [docs/building.md](docs/building.md#cmake-options).

<br/>

#### Python

```bash
pip install carquet
```

Prebuilt wheels bundle the library for Linux, macOS and Windows; see the [Python API](#python-api).

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### Build configurations

All `make` targets and variables, every CMake and xmake option, Windows toolchains (MSVC, clang-cl, MinGW-w64), linking your own zstd / zlib / lz4 / brotli, tests, sanitizers and the Doxygen reference are documented in **[docs/building.md](docs/building.md)**.

<a id="ai-coding-agents"></a>

<details>
<summary><b>AI coding agents</b> (a ready-made skill for Claude Code and Codex)</summary>

<br/>

[`.agents/skills/carquet/`](.agents/skills/carquet/) is an [Agent Skill](https://agentskills.io) that teaches a coding agent how to integrate carquet into your project: build wiring (CMake, FetchContent, pkg-config, xmake), which API to use for what, complete recipes, the level / null-bitmap / lifetime rules, type mappings and the usual pitfalls with their fixes. Copy or link it into your repository and the agent picks it up automatically:

```bash
mkdir -p .claude/skills && cp -r path/to/carquet/.agents/skills/carquet .claude/skills/   # Claude Code
mkdir -p .agents/skills && cp -r path/to/carquet/.agents/skills/carquet .agents/skills/   # Codex CLI
ln -s ../../third_party/carquet/.agents/skills/carquet .claude/skills/carquet               # vendored: stays in sync
```

The same files also work as a personal skill (`~/.claude/skills/`, `~/.codex/skills/`).

</details>

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### C API

A Write and a Read example below; the [manual in `docs/`](docs/README.md) covers everything else.

<br/>

#### Write a Parquet File

```c
#include <carquet/carquet.h>

int main(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    // Define schema
    carquet_schema_t* schema = carquet_schema_create(&err);
    carquet_schema_add_column(schema, "id",    CARQUET_PHYSICAL_INT64,  NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);

    // Configure writer
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_ZSTD;

    // Write
    carquet_writer_t* w = carquet_writer_create("output.parquet", schema, &opts, &err);

    int64_t ids[]    = {1, 2, 3, 4, 5};
    double values[]  = {1.1, 2.2, 3.3, 4.4, 5.5};
    carquet_writer_write_batch(w, 0, ids, 5, NULL, NULL);
    carquet_writer_write_batch(w, 1, values, 5, NULL, NULL);
    carquet_writer_close(w);

    carquet_schema_free(schema);
    return 0;
}
```

<br/>

#### Read a Parquet File

```c
#include <carquet/carquet.h>
#include <stdio.h>

int main(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    // Open with mmap for best read performance
    carquet_reader_options_t opts;
    carquet_reader_options_init(&opts);
    opts.use_mmap = true;

    carquet_reader_t* r = carquet_reader_open("output.parquet", &opts, &err);
    if (!r) { printf("Error: %s\n", err.message); return 1; }

    printf("Rows: %lld, Columns: %d\n",
           (long long)carquet_reader_num_rows(r),
           carquet_reader_num_columns(r));

    // Batch reader for efficient iteration
    carquet_batch_reader_config_t cfg;
    carquet_batch_reader_config_init(&cfg);
    cfg.batch_size = 65536;

    carquet_batch_reader_t* br = carquet_batch_reader_create(r, &cfg, &err);
    carquet_row_batch_t* batch = NULL;

    while (carquet_batch_reader_next(br, &batch) == CARQUET_OK && batch) {
        const void* data;
        const uint8_t* nulls;
        int64_t n;
        carquet_row_batch_column(batch, 0, &data, &nulls, &n);
        const int64_t* ids = (const int64_t*)data;
        // process ids[0..n-1] ...
        carquet_row_batch_free(batch);
        batch = NULL;
    }

    carquet_batch_reader_free(br);
    carquet_reader_close(r);
    return 0;
}
```

<br/>

#### More Recipes

Everything beyond flat read/write lives in the manual; each page links to a runnable example:

<div align="center">

| You want to… | See |
|---|---|
| Nullable columns, row groups, buffer output | [`docs/writing.md`](docs/writing.md) |
| Lists, maps, groups, definition/repetition levels | [`docs/nested-data.md`](docs/nested-data.md) |
| Column projection, statistics, metadata inspection | [`docs/reading.md`](docs/reading.md) |
| Predicate pushdown, page-level filtering | [`docs/reading.md`](docs/reading.md) |
| Append row groups to an existing file | [`docs/writing.md`](docs/writing.md#append-to-an-existing-file) |
| Compression, custom codecs, writer tuning | [`docs/writing.md`](docs/writing.md), [`docs/performance.md`](docs/performance.md) |
| mmap, zero-copy, prebuffering, I/O coalescing | [`docs/performance.md`](docs/performance.md) |
| Error codes and recovery hints | [`docs/error-handling.md`](docs/error-handling.md) |

</div>

<br/>

#### API Reference

The full API is in [`include/carquet/carquet.h`](include/carquet/carquet.h). Key types:

<div align="center">

| Type | Purpose |
|------|---------|
| `carquet_reader_t` | File reader (open from path, `FILE*`, or memory buffer) |
| `carquet_writer_t` | File writer |
| `carquet_batch_reader_t` | High-level batch iteration |
| `carquet_schema_t` | Schema definition and introspection |
| `carquet_error_t` | Rich error info (code, message, source location, recovery hint) |

</div>

The [manual](docs/README.md) explains which surface to use when; the source layout and architecture are documented in [CONTRIBUTING.md](CONTRIBUTING.md). A browsable HTML reference can be generated from the headers with Doxygen, see [docs/building.md](docs/building.md#tests-sanitizers-and-api-docs).

<details>
<summary><b>Example application: MetricStore</b> (<code>mocklib/</code>)</summary>

<br/>

[`mocklib/`](mocklib/) is **MetricStore**, a complete, self-contained example application built on top of carquet, useful both as a reference for real-world usage and as an end-to-end integration test of the public API.

It models a time-series telemetry store (ingest events, then introspect and query them) and, in doing so, exercises **117 of carquet's 143 public functions (~82 %)**:

- **Write side**: schema construction with logical types, per-column encoding / compression / bloom tuning, page indexes and statistics
- **Read side**: column projection, predicate pushdown, page-level filtering, bloom membership, the Arrow C Data Interface bridge, nested `LIST` reconstruction, buffer/append I/O and metadata introspection

It links carquet the way any downstream project would (`find_package(carquet)` / `add_subdirectory`), ships a self-checking round-trip test, and its `write_sample` binary emits a file for external inspection. See [`mocklib/README.md`](mocklib/README.md).

</details>

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### Python API

`carquet` on PyPI is not a wrapper around the C calls above: it is a **Pythonic, `pyarrow.parquet`-style API** (`read_table`, `ParquetFile`, `ParquetWriter`, DNF `filters`) built on the same engine. The wheel bundles the C library with every codec compiled in; data crosses into Python through the Arrow C Data Interface, so decoding, decompression and I/O run in C without the GIL and PyArrow receives the buffers directly.

```bash
pip install "carquet[arrow]"   # with PyArrow: tables, DataFrames, streaming
pip install carquet            # metadata, schema, statistics and validation only
```

```python
import carquet as cq

table = cq.read_table("data.parquet", columns=["id", "price"], filters=[("price", ">", 100)])
df    = cq.read_pandas("data.parquet")                          # pandas.DataFrame, index restored
cq.write_table(table, "out.parquet", compression="zstd")        # also DataFrame, polars, dict, batches

with cq.ParquetFile("data.parquet") as pf:
    for batch in pf.iter_batches(batch_size=50_000, columns=["id"]):
        ...
    pf.metadata.row_group(0).column("price").statistics       # min / max / null_count
```

<div align="center">

| You want to… | Use |
|---|---|
| Read a file, a directory, bytes or a file object | `cq.read_table`, `cq.read_pandas` |
| Stream large files | `ParquetFile.iter_batches`, `cq.iter_batches` |
| Prune with statistics, page indexes, bloom filters | `filters=` (DNF or `pyarrow.compute` expression), `ParquetFile.bloom_filter` |
| Write tables, DataFrames, polars, DuckDB results | `cq.write_table`, `cq.ParquetWriter` (incremental, `append=True`) |
| Inspect without PyArrow | `cq.read_metadata`, `cq.read_parquet_schema`, `cq.info`, `cq.validate` |

</div>

Types round-trip exactly with `pyarrow.parquet` (time zones, categoricals, decimals, large types, nested data). The full reference is in [`python/README.md`](python/README.md).

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### CLI Tool

Carquet ships with a command-line tool for inspecting Parquet files and generating C reader code. It is built and installed by default alongside the library.

```
Commands:
  schema     Print file schema
  info       Print detailed file metadata
  head       Print first N rows
  tail       Print last N rows
  cat        Print rows with slicing/column/row filtering
  count      Print total row count
  columns    List column names (one per line)
  stat       Print column statistics
  validate   Verify file integrity
  sample     Print N random rows
  export     Write rows to stdout as CSV
  codegen    Generate C reader code
```

Typical usage:

```bash
carquet schema data.parquet
carquet head -n 20 data.parquet
carquet stat data.parquet
carquet validate data.parquet
```

<br/>

#### Row Filtering

`cat`, `count`, `head`, `tail`, `sample`, and `export` accept `-p / --filter EXPR` to push a row predicate down to the page level: only pages whose column-index min/max can match the predicate are decompressed.

```bash
carquet cat -p "price > 100 AND status = 'active'" data.parquet
carquet count --filter "id >= 1000" data.parquet
carquet export --filter "ts IS NOT NULL" -c id,ts data.parquet
```

The grammar is `column OP value [AND column OP value]...` with `OP` ∈ {`=`, `==`, `!=`, `<>`, `<`, `<=`, `>`, `>=`}, plus `column IS NULL` / `column IS NOT NULL`.

> Filtering requires the file to have a page index (`write_page_index = true`).

<br/>

#### Code Generation

Generate a complete, compilable C reader from any Parquet file's schema:

```bash
carquet codegen -f data.parquet -o reader.c
# Generated: reader.c
# Compile:   clang -o reader reader.c -I.../include -L.../build -lcarquet ...

./reader                    # reads data.parquet (embedded as default)
./reader other.parquet      # override with a different file
```

<details>
<summary><b>codegen options</b></summary>

<br/>

<div align="center">

| Flag | Description |
|------|-------------|
| `-f`, `--file FILE` | Parquet file to inspect the schema from |
| `-o`, `--output FILE` | Output source file (default: stdout) |
| `--mmap` | Use memory-mapped I/O in the generated code |
| `--skeleton` | Generate an empty `process_batch` for custom logic |
| `-c`, `--columns COLS` | Comma-separated column filter |
| `-b`, `--batch-size N` | Batch size (default: 1024) |

</div>

</details>

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### Parquet Feature Support

What the library implements today, by area.

<br/>

#### Format

<div align="center">

| Feature | Status |
|---------|--------|
| Physical types | ✓  All 8 (BOOLEAN through FIXED_LEN_BYTE_ARRAY) |
| Logical types | ✓  STRING, DATE, TIME, TIMESTAMP, DECIMAL, UUID, JSON, INTERVAL, FLOAT16, VARIANT, GEOMETRY, GEOGRAPHY |
| Encodings | ✓  PLAIN, RLE, DICTIONARY, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY, BYTE_STREAM_SPLIT (read + write) |
| Data Page versions | ✓  V1 (default) and V2 (read + write) |
| Compression | ✓  UNCOMPRESSED, SNAPPY, GZIP, LZ4 (Hadoop-framed, codec 5), LZ4_RAW (codec 7), ZSTD, BROTLI (`CARQUET_ENABLE_BROTLI`, default ON); LZO via `carquet_register_codec()` |
| Nested schemas | ✓  Groups, lists, maps with definition/repetition levels; single-level LIST/MAP auto-shredding on write (`carquet_writer_write_list_column`) and List reconstruction on read; nested `ARROW:schema` emission |
| Encryption | ✗ Not supported; an encrypted-footer file (`PARE`) is reported as `CARQUET_ERROR_NOT_IMPLEMENTED` with an explicit message |

</div>

<details>
<summary><b>Indexing &amp; pruning</b></summary>

<br/>

<div align="center">

| Feature | Status |
|---------|--------|
| Bloom filters | ✓  Read, write, and query (`carquet_bloom_filter_check_*`); consulted automatically for row-group pruning |
| Page indexes | ✓  Column index + offset index (read + write + per-page stats access) |
| Statistics | ✓  Min/max/null count per column chunk; exact `distinct_count` for dictionary columns; Parquet 2.9 SizeStatistics (unencoded byte-array bytes + level histograms) |
| Predicate pushdown | ✓  Automatic row-group pruning via statistics + bloom filters (no callback needed); page-level filtering via column index (`carquet_batch_reader_set_page_filter`) |
| Geospatial statistics | ✓  Per-column bounding box and geometry types for GEOMETRY/GEOGRAPHY (`carquet_reader_geospatial_statistics`) |
| Column projection | ✓  Read only selected columns |
| Row window | ✓  Read an `[offset, limit)` slice of rows, skipping pages entirely (`carquet_batch_reader_set_row_range`) |

</div>

</details>

<details>
<summary><b>I/O &amp; extensibility</b></summary>

<br/>

<div align="center">

| Feature | Status |
|---------|--------|
| Append | ✓  Add row groups to an existing file (`carquet_writer_open_append`) |
| Custom codecs | ✓  Register a custom compress/decompress impl per codec slot (`carquet_register_codec`) |
| Key-value metadata | ✓  Read and write arbitrary footer metadata |
| Per-field metadata | ✓  Arrow `Field.custom_metadata` (variable labels/descriptions) via `ARROW:schema` (read + write) |
| Per-column options | ✓  Per-column encoding, compression, statistics, bloom filter |
| Buffer writer | ✓  Write Parquet to an in-memory buffer |
| CRC32 | ✓  Page-level verification (HW-accelerated on ARM) |
| Memory-mapped I/O | ✓  Zero-copy reads for uncompressed PLAIN data |
| Dictionary-preserving reads | ✓  Return RLE indices + dictionary instead of materialized values (`preserve_dictionaries`, `carquet_row_batch_column_dictionary`) |
| I/O coalescing | ✓  Pre-buffer multi-column reads in a single I/O |
| Speculative footer | ✓  Single-I/O file open for most files |
| OpenMP parallel reads | ✓  When available |
| Parallel writes | ✓  Page-level parallel encode/compress across columns (OpenMP) + background row-group I/O (`async_io`) |

</div>

</details>

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### Interoperability

Carquet files are fully compatible with PyArrow, DuckDB, Spark, and any Parquet reader.

```python
import pyarrow.parquet as pq
table = pq.read_table("carquet_output.parquet")  # just works
```

```sql
-- DuckDB
SELECT * FROM read_parquet('carquet_output.parquet');
```

<details>
<summary><b>Interop test suite</b></summary>

<br/>

Bidirectional round-trips (C writes / Python reads, and back):

```bash
cmake -B build -DCARQUET_BUILD_INTEROP=ON && cmake --build build
python3 interop/run_interop.py
```

Large multi-row-group Snappy files (many row groups, wide `BYTE_ARRAY` columns, page offsets past 2 GiB) have their own generator, which builds the file with PyArrow and checks that carquet decodes every page:

```bash
CARQUET_BIN=build/carquet python3 interop/snappy_multi_rowgroup.py --rows 150000 --row-groups 92
```

</details>

<br/>
<br/>

<img src="res/img/divider.svg" width="100%" alt="" />

### Contributing, Citing, License

- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md) covers setup, code style and the per-pull-request gate (tests, sanitizers, fuzzing, interop, benchmarks). Release notes are in [CHANGELOG.md](CHANGELOG.md).
- **Citing**: a [`CITATION.cff`](CITATION.cff) is included; GitHub's "Cite this repository" button formats it as BibTeX or APA.
- **License**: MIT, see [LICENSE](LICENSE).
