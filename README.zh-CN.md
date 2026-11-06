<h1 align="center">
  <img src="res/img/carquet_banner.png" alt="Carquet" width="840" />
</h1>

<p align="center">
  一个快速、纯 C 实现的 Apache Parquet 文件读写库。
</p>

<p align="center">
  <a href="https://vitruves.github.io/carquet/"><img src="https://img.shields.io/badge/Website-vitruves.github.io%2Fcarquet-2a78d6?style=for-the-badge" alt="官方网站" /></a>
</p>

<p align="center">
  <a href="https://github.com/Vitruves/carquet/actions/workflows/cpp.yml"><img src="https://github.com/Vitruves/carquet/actions/workflows/cpp.yml/badge.svg" alt="构建" /></a>
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue" alt="平台" />
  <img src="https://img.shields.io/badge/C-C11-blue" alt="C 标准" />
  <a href="https://github.com/Vitruves/carquet/releases"><img src="https://img.shields.io/github/v/release/Vitruves/carquet?color=blue" alt="发布版本" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="许可证" /></a>
  <a href="https://github.com/Vitruves/carquet/graphs/traffic"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Vitruves/carquet/traffic/clones.json" alt="克隆数" /></a>
  <br/>
  <a href="https://pypi.org/project/carquet/"><img src="https://img.shields.io/pypi/v/carquet?color=blue&label=pypi" alt="PyPI" /></a>
  <a href="https://pypi.org/project/carquet/"><img src="https://img.shields.io/pypi/pyversions/carquet" alt="Python 版本" /></a>
  <a href="https://pepy.tech/projects/carquet"><img src="https://img.shields.io/pepy/dt/carquet?color=blue&label=pypi%20downloads" alt="PyPI 下载量" /></a>
  <img src="https://img.shields.io/badge/SIMD-SSE4.2%20%C2%B7%20AVX2%20%C2%B7%20AVX--512%20%C2%B7%20NEON%20%C2%B7%20SVE-eb6834" alt="SIMD" />
</p>

<p align="center">
  <a href="#亮点">亮点</a> &nbsp;·&nbsp;
  <a href="#基准测试">基准测试</a> &nbsp;·&nbsp;
  <a href="#安装">安装</a> &nbsp;·&nbsp;
  <a href="#构建配置">构建配置</a> &nbsp;·&nbsp;
  <a href="#c-api">C API</a> &nbsp;·&nbsp;
  <a href="#python-api">Python API</a> &nbsp;·&nbsp;
  <a href="#cli-工具">CLI</a> &nbsp;·&nbsp;
  <a href="#parquet-功能支持">功能支持</a> &nbsp;·&nbsp;
  <a href="docs/README.md">手册</a> &nbsp;·&nbsp;
  <a href="CHANGELOG.md">更新日志</a>
</p>

<p align="center">
  <a href="README.md">English</a> &nbsp;·&nbsp;
  <b>简体中文</b> &nbsp;·&nbsp;
  <a href="README.ja.md">日本語</a> &nbsp;·&nbsp;
  <a href="README.ko.md">한국어</a> &nbsp;·&nbsp;
  <a href="README.de.md">Deutsch</a> &nbsp;·&nbsp;
  <a href="README.fr.md">Français</a> &nbsp;·&nbsp;
  <a href="README.pt-BR.md">Português (Brasil)</a>
</p>

<br/>

---

<!-- ────────────────────────────  亮点  ──────────────────────────── -->

## 亮点

<div align="center">

| | |
|---|---|
| ⚡ **快速** | 写入比 Arrow C++ **快 2.3–5.9 倍**，读取同一个文件的压缩数据**快 1.5–2.1 倍**（Apple M3，1000 万行）。未压缩数据直接从 mmap 提供，仅需 **0.30 ms**。支持 SSE4.2 / AVX2 / AVX-512 / NEON / SVE 的 SIMD 优化，运行时分派。 |
| 🪶 **小巧** | 纯 C11，库仅 **约 500 KB**（Arrow C++：50 MB 以上）。四个压缩依赖（zstd、zlib、lz4、brotli），缺失时自动获取。一条 `make` 或 `xmake` 命令，一分钟内完成构建。 |
| 📦 **完整** | 所有物理类型和编码，Snappy / GZIP / LZ4 / ZSTD / Brotli，V1 + V2 页，任意深度的嵌套模式，布隆过滤器，页索引，基于统计信息的剪枝，追加写入。[完整矩阵 ↓](#parquet-功能支持) |
| 🔗 **可互操作** | 生成的文件开箱即可被 PyArrow、DuckDB 和 Spark 读取。零依赖的 [Arrow C Data Interface](docs/reading.md#export-to-arrow-c-data-interface) 桥接（任意深度的 struct / list / map），以及构建于 PyArrow 之上的完整 [Python API](#python-api)。 |
| 🛠 **工具** | `carquet` CLI 用于检视、过滤、导出和校验文件，另有 `codegen` 可生成开箱即可编译的 C 读取器。还为 Claude Code 和 Codex 提供了现成的[智能体技能](#ai-coding-agents)。 |

</div>

<br/>

---

<!-- ────────────────────────────  基准测试  ──────────────────────────── -->

## 基准测试

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="res/img/bench_10m_m3_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="res/img/bench_10m_m3_light.png">
    <img src="res/img/bench_10m_m3_light.png" alt="Apple M3 上 1000 万行的写入和读取耗时：carquet 与 Arrow C++ 及各 Python Parquet 库按编解码器对比" width="900">
  </picture>
</p>

<p align="center"><b><a href="https://vitruves.github.io/carquet/">在网站上浏览交互式基准测试 →</a></b><br/><sub>可在读取与写入之间切换，也可在 10 万、100 万和 1000 万行之间切换</sub></p>

1000 万行下 carquet 相对 Arrow C++ 的加速比（越高 = carquet 越快）：

<div align="center">

| | ARM · Apple M3 | | x86 · Windows（Haswell，MSVC） | | x86 · Linux（Xeon D-1531） | |
|---|---:|---:|---:|---:|---:|---:|
| **编解码器** | **写入** | **读取** | **写入** | **读取** | **写入** | **读取** |
| snappy | **4.74x** | **1.89x** | **3.39x** | **1.17x** | **1.55x** | **1.25x** |
| zstd | **5.91x** | **1.77x** | **3.83x** | **1.13x** | **1.31x** | **1.04x** |
| lz4 | **4.61x** | **1.89x** | **2.96x** | **1.05x** | **1.02x** | 0.83x |
| none | **2.30x** | **46.3x**\* | **2.10x** | **62.0x**\* | **1.13x** | **40.6x**\* |

</div>

<sub>\* mmap 零拷贝路径。Apple M3：carquet 0.7.3 对比 Arrow C++ 26.0.1（2026-11-06）。Windows：carquet 0.7.3 对比 Arrow C++ 24.0.0（2026-11-02）。Linux：carquet 0.4.4 对比 Arrow C++ 23.0.1，早于 0.7.1 引入的写入器并行化，此后未重新测量。</sub>

压缩读取会解码每一个值，而且在 Linux 和 macOS 上两个库链接的是同一份系统 lz4 / zstd，因此编解码器速度完全相同。最公平的对比是**同文件交叉读取**（两个库读取完全相同的文件）：carquet 在 M3 上**快 1.5–2.1 倍**，在 Linux 上快 1.6–2.6 倍，在 Windows 虚拟机上快 1.04–1.30 倍，后者的冷读取受限于虚拟磁盘。

<details>
<summary><b>基准测试方法</b></summary>

<br/>

所有基准测试使用相同的数据（确定性 LCG 伪随机数生成器）、相同的 Parquet 设置（无字典、浮点数使用 BYTE_STREAM_SPLIT、页校验和、mmap 读取），取 11–51 次迭代的截尾中位数，在写入和读取阶段之间清空操作系统页缓存，并在各配置之间留有冷却时间。模式：3 列（INT64、DOUBLE、INT32）。对比对象为 Arrow C++ 底层 Parquet 读取器（绕过 Arrow Table 物化）和 PyArrow。

页缓存按文件清空：Linux 上使用 `posix_fadvise(DONTNEED)`；macOS 和 Windows 上则以不经缓存的方式复制文件（`F_NOCACHE`、`FILE_FLAG_NO_BUFFERING`），再用副本替换原文件（`benchmark/bench_purge_win32.h`）。在 Windows 上，carquet 链接的是它自行构建的 zstd / lz4，而 Arrow C++ 链接的是 conda-forge 提供的版本，因此两者的编解码器速度可能略有差异。

**同文件交叉读取**基准测试是最公平的对比：两个库读取完全相同的 Parquet 文件（由其中一个写入，两者都读取）。这消除了页大小、编码选择和行组布局上的差异。

标有 \* 的**未压缩读取**使用 Carquet 的 **mmap 零拷贝路径**：对于 PLAIN 编码、未压缩、定长、required 的列，批量读取器直接返回指向内存映射文件的指针，没有 memcpy。Arrow 总是物化到自己的缓冲区中。**压缩读取数据是端到端读取吞吐量最具代表性的衡量指标。**

</details>

<details>
<summary><b>完整 x86 结果</b>（Intel Xeon D-1531，Linux）</summary>

<br/>

*12 线程 @ 2.2GHz，32GB 内存，Ubuntu 24.04，ZSTD 级别 1*

#### 1000 万行 vs Arrow C++

<div align="center">

| 编解码器 | Carquet 写入 | Arrow C++ 写入 | 写入比 | Carquet 读取 | Arrow C++ 读取 | 读取比 | 大小 |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | **1557ms** | 1766ms | **1.13x** | **1.25ms** | 50.8ms | **40.6x**\* | 190.7MB |
| snappy | **1002ms** | 1549ms | **1.55x** | **78ms** | 97.8ms | **1.25x** | 125.1MB |
| zstd | **1311ms** | 1714ms | **1.31x** | **76.8ms** | 80.2ms | **1.04x** | 95.3MB |
| lz4 | **1521ms** | 1554ms | **1.02x** | 59.1ms | **49.0ms** | 0.83x | 122.9MB |

</div>

<br/>

#### 100 万行 vs Arrow C++

<div align="center">

| 编解码器 | Carquet 写入 | Arrow C++ 写入 | 写入比 | Carquet 读取 | Arrow C++ 读取 | 读取比 |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **180ms** | 196ms | **1.09x** | **0.22ms** | 6.2ms | **28x**\* |
| snappy | **141ms** | 148ms | **1.05x** | **8.1ms** | 11.6ms | **1.44x** |
| zstd | **131ms** | 185ms | **1.41x** | 10.3ms | **9.1ms** | 0.88x |
| lz4 | **143ms** | 149ms | **1.04x** | 8.5ms | **6.1ms** | 0.72x |

</div>

<br/>

#### 10 万行 vs Arrow C++

<div align="center">

| 编解码器 | Carquet 写入 | Arrow C++ 写入 | 写入比 | Carquet 读取 | Arrow C++ 读取 | 读取比 |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **14.1ms** | 18.4ms | **1.30x** | **0.11ms** | 2.18ms | **19.8x**\* |
| snappy | **10.1ms** | 10.6ms | **1.05x** | **1.27ms** | 5.97ms | **4.70x** |
| zstd | **8.7ms** | 14.1ms | **1.62x** | **1.58ms** | 3.88ms | **2.46x** |
| lz4 | **9.6ms** | 11.0ms | **1.14x** | **0.77ms** | 2.78ms | **3.61x** |

</div>

<br/>

#### 同文件交叉读取（1000 万行）

两个库读取**同一个** Parquet 文件，这是最公平的同等条件对比。

<div align="center">

| 编解码器 | 写入方 | Carquet 读取 | Arrow C++ 读取 | 比值 |
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

#### 1000 万行 vs PyArrow

<div align="center">

| 编解码器 | Carquet 写入 | PyArrow 写入 | 写入比 | Carquet 读取 | PyArrow 读取 | 读取比 |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **1557ms** | 1806ms | **1.16x** | **1.25ms** | 213ms | **170x**\* |
| snappy | **1002ms** | 1649ms | **1.65x** | **78ms** | 384ms | **4.91x** |
| zstd | **1311ms** | 1796ms | **1.37x** | **76.8ms** | 369ms | **4.81x** |
| lz4 | **1521ms** | 1676ms | **1.10x** | **59.1ms** | 281ms | **4.76x** |

</div>

\* 零拷贝 mmap 路径

</details>

<details>
<summary><b>完整 ARM 结果</b>（Apple M3，macOS）</summary>

<br/>

*Carquet 0.7.3，MacBook Air M3，16GB 内存，macOS 26.5，Arrow C++ 26.0.1，PyArrow 26.0.1，fastparquet 2026.9.0，ZSTD 级别 1，测量于 2026-11-06。原始报告：[`benchmark/bench_0.7.3_darwin_arm64_20261106.json`](benchmark/bench_0.7.3_darwin_arm64_20261106.json)。*

#### 1000 万行 vs Arrow C++

<div align="center">

| 编解码器 | Carquet 写入 | Arrow C++ 写入 | 写入比 | Carquet 读取 | Arrow C++ 读取 | 读取比 | 大小 |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **48.8ms** | 112ms | **2.30x** | **0.30ms** | 13.9ms | **46.3x**\* | 200.0MB |
| snappy | **57.2ms** | 271ms | **4.74x** | **12.6ms** | 23.8ms | **1.89x** | 131.2MB |
| zstd | **59.7ms** | 353ms | **5.91x** | **19.8ms** | 35.0ms | **1.77x** | 99.9MB |
| lz4 | **52.4ms** | 241ms | **4.61x** | **9.60ms** | 18.2ms | **1.89x** | 128.7MB |

</div>

<br/>

#### 100 万行 vs Arrow C++

<div align="center">

| 编解码器 | Carquet 写入 | Arrow C++ 写入 | 写入比 | Carquet 读取 | Arrow C++ 读取 | 读取比 |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **6.75ms** | 13.2ms | **1.95x** | **0.07ms** | 1.72ms | **24.6x**\* |
| snappy | **7.02ms** | 24.0ms | **3.42x** | **1.58ms** | 2.49ms | **1.58x** |
| zstd | **7.97ms** | 33.6ms | **4.22x** | **2.16ms** | 3.03ms | **1.40x** |
| lz4 | **6.54ms** | 25.2ms | **3.85x** | **0.97ms** | 1.96ms | **2.02x** |

</div>

<br/>

#### 10 万行 vs Arrow C++

<div align="center">

| 编解码器 | Carquet 写入 | Arrow C++ 写入 | 写入比 | Carquet 读取 | Arrow C++ 读取 | 读取比 |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **1.08ms** | 1.75ms | **1.62x** | **0.02ms** | 0.24ms | **12.0x**\* |
| snappy | **1.80ms** | 2.52ms | **1.40x** | **0.36ms** | 0.90ms | **2.50x** |
| zstd | **1.81ms** | 3.60ms | **1.99x** | **0.63ms** | 1.30ms | **2.06x** |
| lz4 | **1.67ms** | 2.57ms | **1.54x** | **0.29ms** | 0.61ms | **2.10x** |

</div>

<br/>

#### 1000 万行 vs PyArrow

<div align="center">

| 编解码器 | Carquet 写入 | PyArrow 写入 | 写入比 | Carquet 读取 | PyArrow 读取 | 读取比 |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **48.8ms** | 170ms | **3.49x** | **0.30ms** | 26.8ms | **89.2x**\* |
| snappy | **57.2ms** | 294ms | **5.14x** | **12.6ms** | 38.4ms | **3.05x** |
| zstd | **59.7ms** | 423ms | **7.09x** | **19.8ms** | 42.5ms | **2.15x** |
| lz4 | **52.4ms** | 283ms | **5.41x** | **9.60ms** | 30.2ms | **3.15x** |

</div>

<br/>

#### 100 万行 vs PyArrow

<div align="center">

| 编解码器 | Carquet 写入 | PyArrow 写入 | 写入比 | Carquet 读取 | PyArrow 读取 | 读取比 |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **6.75ms** | 17.0ms | **2.52x** | **0.07ms** | 2.73ms | **39.0x**\* |
| snappy | **7.02ms** | 27.9ms | **3.97x** | **1.58ms** | 3.66ms | **2.32x** |
| zstd | **7.97ms** | 38.8ms | **4.87x** | **2.16ms** | 3.96ms | **1.83x** |
| lz4 | **6.54ms** | 29.2ms | **4.46x** | **0.97ms** | 2.98ms | **3.07x** |

</div>

<br/>

#### 10 万行 vs PyArrow

<div align="center">

| 编解码器 | Carquet 写入 | PyArrow 写入 | 写入比 | Carquet 读取 | PyArrow 读取 | 读取比 |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **1.08ms** | 1.99ms | **1.84x** | **0.02ms** | 0.24ms | **12.0x**\* |
| snappy | **1.80ms** | 2.82ms | **1.57x** | **0.36ms** | 0.66ms | **1.83x** |
| zstd | **1.81ms** | 4.17ms | **2.30x** | **0.63ms** | 0.72ms | **1.14x** |
| lz4 | **1.67ms** | 2.93ms | **1.75x** | **0.29ms** | 0.43ms | **1.48x** |

</div>

<br/>

#### 1000 万行 vs fastparquet

<div align="center">

| 编解码器 | Carquet 写入 | fastparquet 写入 | 写入比 | Carquet 读取 | fastparquet 读取 | 读取比 |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **48.8ms** | 65.4ms | **1.34x** | **0.30ms** | 49.0ms | **163.5x**\* |
| snappy | **57.2ms** | 204ms | **3.57x** | **12.6ms** | 110ms | **8.78x** |
| zstd | **59.7ms** | 331ms | **5.55x** | **19.8ms** | 185ms | **9.39x** |
| lz4 | **52.4ms** | 225ms | **4.30x** | **9.60ms** | 55.3ms | **5.76x** |

</div>

<br/>

#### 100 万行 vs fastparquet

<div align="center">

| 编解码器 | Carquet 写入 | fastparquet 写入 | 写入比 | Carquet 读取 | fastparquet 读取 | 读取比 |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | **6.75ms** | 7.45ms | **1.10x** | **0.07ms** | 3.00ms | **42.9x**\* |
| snappy | **7.02ms** | 22.1ms | **3.16x** | **1.58ms** | 10.1ms | **6.39x** |
| zstd | **7.97ms** | 32.6ms | **4.09x** | **2.16ms** | 18.4ms | **8.54x** |
| lz4 | **6.54ms** | 21.6ms | **3.30x** | **0.97ms** | 5.06ms | **5.22x** |

</div>

<br/>

#### 10 万行 vs fastparquet

<div align="center">

| 编解码器 | Carquet 写入 | fastparquet 写入 | 写入比 | Carquet 读取 | fastparquet 读取 | 读取比 |
|-------|--------------|---------|---------|-------------|---------|---------|
| none | 1.08ms | **1.04ms** | 0.96x | **0.02ms** | 0.52ms | **26.0x**\* |
| snappy | **1.80ms** | 2.31ms | **1.28x** | **0.36ms** | 1.20ms | **3.33x** |
| zstd | **1.81ms** | 3.45ms | **1.91x** | **0.63ms** | 2.05ms | **3.25x** |
| lz4 | **1.67ms** | 2.25ms | **1.35x** | **0.29ms** | 0.73ms | **2.52x** |

</div>

<br/>

#### 同文件交叉读取（1000 万行）

两个库读取**同一个** Parquet 文件，这是最公平的同等条件对比。

<div align="center">

| 编解码器 | 写入方 | Carquet 读取 | Arrow C++ 读取 | 比值 |
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

\* 零拷贝 mmap 路径。fastparquet 通过 pandas 读取，且无法写入 BYTE_STREAM_SPLIT 或页校验和，因此其结果的可比性较低。

</details>

<details>
<summary><b>完整 Windows 结果</b>（Intel Haswell 虚拟机，MSVC）</summary>

<br/>

*Carquet 0.7.3 使用 MSVC 19.44 构建（x64 Release，OpenMP），OpenStack 虚拟机，4 个 vCPU（Intel Haswell，3.1 GHz），14.6GB 内存，虚拟 SAS 磁盘，Windows Server 2025，Arrow C++ 24.0.0（conda-forge），ZSTD 级别 1，测量于 2026-11-02。冷读取需要访问虚拟磁盘，这缩小了读取比；写入和零拷贝路径则受 CPU 限制。*

#### 1000 万行 vs Arrow C++

<div align="center">

| 编解码器 | Carquet 写入 | Arrow C++ 写入 | 写入比 | Carquet 读取 | Arrow C++ 读取 | 读取比 | 大小 |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | **182ms** | 382ms | **2.10x** | **2.04ms** | 126.5ms | **62.0x**\* | 190.7MB |
| snappy | **175ms** | 594ms | **3.39x** | **115.6ms** | 134.9ms | **1.17x** | 125.1MB |
| zstd | **242ms** | 927ms | **3.83x** | **140.7ms** | 159.2ms | **1.13x** | 95.3MB |
| lz4 | **178ms** | 527ms | **2.96x** | **95.5ms** | 99.9ms | **1.05x** | 122.9MB |

</div>

<br/>

#### 100 万行 vs Arrow C++

<div align="center">

| 编解码器 | Carquet 写入 | Arrow C++ 写入 | 写入比 | Carquet 读取 | Arrow C++ 读取 | 读取比 |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **15.6ms** | 37.6ms | **2.41x** | **0.44ms** | 13.72ms | **31.2x**\* |
| snappy | **25.4ms** | 61.2ms | **2.41x** | **14.01ms** | 16.31ms | **1.16x** |
| zstd | **30.4ms** | 94.3ms | **3.10x** | **17.09ms** | 18.38ms | **1.08x** |
| lz4 | **26.4ms** | 53.8ms | **2.04x** | 13.07ms | **12.89ms** | 0.99x |

</div>

<br/>

#### 10 万行 vs Arrow C++

<div align="center">

| 编解码器 | Carquet 写入 | Arrow C++ 写入 | 写入比 | Carquet 读取 | Arrow C++ 读取 | 读取比 |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **3.53ms** | 3.95ms | **1.12x** | **0.16ms** | 2.13ms | **13.3x**\* |
| snappy | **5.60ms** | 6.23ms | **1.11x** | **2.79ms** | 3.92ms | **1.41x** |
| zstd | **7.11ms** | 9.79ms | **1.38x** | **2.86ms** | 4.36ms | **1.52x** |
| lz4 | **5.75ms** | 5.82ms | **1.01x** | 2.22ms | **2.15ms** | 0.97x |

</div>

<br/>

#### 同文件交叉读取（1000 万行）

<div align="center">

| 编解码器 | 写入方 | Carquet 读取 | Arrow C++ 读取 | 比值 |
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

\* 零拷贝 mmap 路径

</details>

<details>
<summary><b>自行运行基准测试</b></summary>

<br/>

```bash
make benchmark                                    # Release build of the benchmark programs into build/
./build/benchmark_carquet                         # carquet alone
python3 benchmark/run_benchmark.py --skip-xlarge  # full comparison with PyArrow (+ Arrow C++ if built)
```

<div align="center">

| `run_benchmark.py` 标志 / 变量 | 作用 |
|---|---|
| `--quick` | 只运行 1000 万行的配置（即图表所展示的） |
| `--skip-xlarge` | 跳过 1 亿行的配置（每种编解码器约 2 GB，耗时 30 分钟以上） |
| `--fastparquet` | 同时对比 fastparquet（`pip install fastparquet pandas`） |
| `--no-pyarrow` / `--no-arrow-cpp` | 跳过 PyArrow / Arrow C++ |
| `--iterations N` | 固定迭代次数（默认：自动，11–51） |
| `--build-dir DIR` / `--temp-dir DIR` | 可执行文件所在目录 / 测试文件写入目录 |
| `CARQUET_BENCH_ZSTD_LEVEL=3` | ZSTD 级别（默认 1） |

</div>

务必对 **Release** 构建进行基准测试：`make benchmark` 会把 `build/` 重新配置为 Release；残留的 Debug `build/` 会让 carquet 看起来慢好几倍。`make native` + `-DCARQUET_BUILD_BENCHMARKS=ON` 可得到针对本机调优的数据。

fastparquet 使用相同的数据、行组大小、无字典和 zstd 级别，`lz4` 则使用 LZ4_RAW。它无法写入 BYTE_STREAM_SPLIT 或页校验和，并且通过 pandas 读取，因此其结果的可比性不如 Arrow。

Arrow C++ 对比需要单独的目标（系统中能找到 Arrow C++，或用 `CARQUET_ARROW_CPP_ROOT` 指定其位置）：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCARQUET_NATIVE_ARCH=ON \
  -DCARQUET_BUILD_BENCHMARKS=ON -DCARQUET_BUILD_ARROW_CPP_BENCHMARK=ON \
  [-DCARQUET_ARROW_CPP_ROOT=/path/to/arrow-prefix]
cmake --build build -j
```

它使用底层 `parquet::ParquetFileReader` API（不做 Arrow Table 物化），并采用并行行组读取器。

在 Windows 上（Developer PowerShell for VS），Arrow C++ 可以来自 conda-forge（`micromamba create -p C:\arrow -c conda-forge libarrow libparquet`）；运行基准测试时其 DLL 必须位于 `PATH` 中：

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCARQUET_BUILD_BENCHMARKS=ON `
  -DCARQUET_BUILD_ARROW_CPP_BENCHMARK=ON -DCARQUET_ARROW_CPP_ROOT=C:/arrow/Library
cmake --build build
$env:PATH = "C:\arrow\Library\bin;$env:PATH"
python benchmark\run_benchmark.py --build-dir build --skip-xlarge --no-pyarrow
```

请将构建目录和 `--temp-dir` 目录排除在杀毒软件实时扫描之外：Windows Defender 会扫描基准测试写入的每一个文件。

README 中的图表根据运行后留在 `benchmark/` 中的 JSON 绘制，分为浅色和深色两个版本，GitHub 会根据读者的主题选择：

```bash
python3 benchmark/plot_readme_chart.py benchmark/bench_<version>_<os>_<arch>_<date>.json   # writes res/img/bench_10m_m3_{light,dark}.png
```

</details>

<br/>

---

<!-- ────────────────────────────  速览  ──────────────────────────── -->

## 三十秒上手 carquet

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

> **carquet 的定位。** 当你需要 Arrow 的生态系统（计算内核、数据集、Flight、加密）时，Arrow 仍然是正确的选择。carquet 面向那些只需要快速、正确地读写 Parquet 的 C 和 C++ 程序、嵌入式系统和语言运行时。

<br/>

---

<!-- ────────────────────────────  安装  ──────────────────────────── -->

## 安装

本节安装的是 **C 库、其头文件以及 `carquet` CLI**。Python 用户请使用单独的 [Python API](#python-api)（`pip install carquet`）。

**环境要求：** C11 编译器（GCC 4.9+、Clang 3.4+、MSVC 2015+）以及 CMake 3.16+（或 [xmake](#使用-xmake-从源码构建)）。zstd、zlib、lz4 和 brotli 若系统中已有则直接使用，否则自动获取。OpenMP 可选（用于并行读取和写入）。

<br/>

### 使用 `make` 从源码构建

最简单的方式。`make` 是一层薄封装，会以正确的标志调用 CMake，无需其他任何配置：

```bash
git clone https://github.com/Vitruves/carquet.git && cd carquet
make                  # optimized Release build into build/
sudo make install     # library, headers, CLI, pkg-config and CMake package files
```

之后使用 `-lcarquet`、`pkg-config --cflags --libs carquet` 链接，或在 CMake 中使用 `find_package(carquet)`。

<br/>

### 使用 `xmake` 从源码构建

[xmake](https://xmake.io) 无需预先安装其他任何东西：它会下载固定版本的 zstd 1.5.6、zlib 1.3.1、lz4 1.10.0 和 brotli 1.1.0 并静态链接，因此 `carquet` CLI 没有任何运行时依赖。

```bash
git clone https://github.com/Vitruves/carquet.git && cd carquet
xmake                         # Release static library + CLI into build/<plat>/<arch>/release/
xmake install -o /usr/local   # headers, library and CLI (sudo if the prefix needs it)
```

静态安装只包含 `libcarquet.a`，因此需要自行链接编解码器（`-lcarquet -lzstd -lz -llz4`，若未使用 `--brotli=n` 构建，还需 `-lbrotlienc -lbrotlidec -lbrotlicommon`）。若改用 `xmake f --shared=y` 配置，则会得到一个内置编解码器的自包含共享库。

<br/>

### 在你的 CMake 项目中使用

```cmake
include(FetchContent)
FetchContent_Declare(carquet GIT_REPOSITORY https://github.com/Vitruves/carquet.git GIT_TAG main)
FetchContent_MakeAvailable(carquet)
target_link_libraries(myapp PRIVATE carquet)
```

或者在 `make install` 之后：`find_package(carquet REQUIRED)` 并链接 `carquet::carquet`。

<br/>

---

<!-- ────────────────────────────  构建配置  ──────────────────────────── -->

## 构建配置

### `make` 目标

每个目标都会配置 `build/` 并并行构建。`make help` 会打印此列表。每个目标都有对应的 [xmake 等价命令](#xmake)。

<div align="center">

| 命令 | 构建类型 | 你将得到 |
|---|---|---|
| `make` / `make release` | Release | 静态库 + `carquet` CLI（默认） |
| `make native` | Release | 同上，但使用 `-march=native` 针对本机 CPU 调优（生成的二进制无法在较旧的 CPU 上运行） |
| `make shared` | Release | 共享库（`.so` / `.dylib` / `.dll`）而非静态库 |
| `make benchmark` | Release | 基准测试程序（`benchmark_carquet`、性能分析目标） |
| `make debug` | Debug | 带断言、无优化的库 + CLI |
| `make test` | Debug | 测试套件，构建后用 `ctest` 运行 |
| `make dev` | Debug | 全部内容：测试、示例、基准测试、互操作程序 |
| `make fuzz` | Debug | libFuzzer 目标（请使用 Clang 工具链） |
| `make install` | Release | 构建 `release`，然后安装到 `PREFIX` |
| `make clean` | | 删除构建目录 |

</div>

<div align="center">

| 变量 | 默认值 | 示例 |
|---|---|---|
| `PREFIX` | CMake 默认值（Unix 上为 `/usr/local`） | `make install PREFIX=$HOME/.local` |
| `BUILD` | `build` | `make test BUILD=build-debug`（并存多个配置） |
| `JOBS` | CPU 核心数 | `make JOBS=4` |
| `CMAKE` | `cmake` | `make CMAKE=/opt/cmake/bin/cmake` |

</div>

默认情况下所有目标共用 `build/`，因此在 `make benchmark` 之后运行 `make test` 会把它重新配置为 Debug。使用 `BUILD=` 可将 Release 和 Debug 构建树分开。

<br/>

### CMake 选项

对于 `make` 目标未覆盖的情况，请直接调用 CMake，并以 `-DOPTION=ON|OFF` 传入选项：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCARQUET_BUILD_SHARED=ON -DCARQUET_ENABLE_BROTLI=OFF
cmake --build build -j
sudo cmake --install build --prefix /usr/local
```

<div align="center">

| 构建内容 | 默认值 | |
|---|:---:|---|
| `CARQUET_BUILD_CLI` | ON | `carquet` 命令行工具 |
| `CARQUET_BUILD_SHARED` | OFF | 共享库而非静态库 |
| `CARQUET_BUILD_DEV` | OFF | 一次性开启测试、示例、基准测试和互操作 |
| `CARQUET_BUILD_TESTS` | = DEV | 测试套件（`ctest`） |
| `CARQUET_BUILD_EXAMPLES` | = DEV | [`examples/`](examples/) 中的程序 |
| `CARQUET_BUILD_BENCHMARKS` | = DEV | 基准测试和性能分析程序 |
| `CARQUET_BUILD_ARROW_CPP_BENCHMARK` | OFF | Arrow C++ 对比基准测试（需要 Arrow C++） |
| `CARQUET_BUILD_INTEROP` | = DEV | C / Python 往返测试程序 |
| `CARQUET_BUILD_FUZZ` | OFF | 模糊测试目标 |
| `CARQUET_BUILD_DOCS` | OFF | `docs` 目标，将 Doxygen HTML 输出到 `build/docs/html/` |

| 依赖 | 默认值 | |
|---|:---:|---|
| `CARQUET_ENABLE_BROTLI` | ON | BROTLI 编解码器；OFF 则去掉 libbrotli 依赖 |
| `CARQUET_BUNDLE_DEPS` | OFF | 始终获取并静态打包 zstd / zlib / lz4 / brotli（可移植的发布产物、Python wheel） |
| `CARQUET_ZSTD_TARGET`、`_ZLIB_`、`_LZ4_`、`_BROTLI_TARGET` | 空 | 链接已有的 CMake 目标，而不是查找或获取（见下文） |
| `CARQUET_ARROW_CPP_ROOT` | 空 | 对比基准测试所用的 Arrow C++ 安装前缀 |
| `CMAKE_DISABLE_FIND_PACKAGE_OpenMP` | OFF | ON 则不使用 OpenMP 构建（运行时不需要 `libgomp` / `vcomp140.dll`） |

| CPU | 默认值 | |
|---|:---:|---|
| `CARQUET_NATIVE_ARCH` | OFF | `-march=native`；最快，但二进制仅限本机运行 |
| `CARQUET_ENABLE_SSE`、`_AVX`、`_AVX2`、`_AVX512` | ON | x86 SIMD 内核，运行时根据 CPU 支持情况选择 |
| `CARQUET_ENABLE_NEON` | ON | ARM NEON 内核 |
| `CARQUET_ENABLE_SVE` | OFF | ARM SVE 内核（实验性） |
| `CARQUET_MSVC_HARDENING` | ON | MSVC：`/guard:cf` 和 `/CETCOMPAT` |

</div>

SIMD 内核会被编译进库并在运行时选择，同时提供标量回退实现，因此默认构建可在其架构的任意 CPU 上运行。只有 `CARQUET_NATIVE_ARCH` 会把二进制绑定到构建机器。

<br/>

### xmake

`xmake.lua` 是 CMake 的完整替代方案，选项和默认值相同。先用 `xmake f` 配置一次，然后用 `xmake` 构建；配置会被记住，直到下一次 `xmake f`。

<div align="center">

| `make` 目标 | xmake 等价命令 |
|---|---|
| `make` | `xmake` |
| `make native` | `xmake f --native_arch=y && xmake` |
| `make shared` | `xmake f --shared=y && xmake` |
| `make benchmark` | `xmake f --benchmarks=y && xmake` |
| `make debug` | `xmake f -m debug && xmake` |
| `make test` | `xmake f -m debug --tests=y && xmake && xmake test` |
| `make dev` | `xmake f -m debug --dev=y && xmake` |
| `make fuzz` | `xmake f -m debug --fuzz=y --toolchain=clang && xmake` |
| `make clean` | `xmake clean`（或 `xmake f -c` 同时重置配置） |
| `cmake --build build --target docs` | `xmake docs`（无需配置标志） |

</div>

选项与 CMake 的一一对应，去掉 `CARQUET_` 前缀并改为小写，写作 `--option=y|n`。可以自由组合：`xmake f -m release --shared=y --brotli=n --avx512=n`。

<div align="center">

| xmake 选项 | 默认值 | 说明 |
|--------------|---------|-------------|
| `--dev` | n | 构建测试、示例、基准测试和互操作 |
| `--tests` / `--examples` / `--benchmarks` / `--interop` | n | 单独构建某一组 |
| `--arrow_cpp_benchmark` | n | Arrow C++ 对比基准测试（若可用，使用 xmake 仓库中的 Arrow） |
| `--cli` | y | 构建 `carquet` CLI 工具 |
| `--shared` | n | 构建共享库而非静态库 |
| `--openmp` | y | OpenMP 并行列读取（不可用时自动禁用） |
| `--native_arch` | n | `-march=native` 以获得最高性能（仅限本机运行的二进制） |
| `--sse` / `--avx` / `--avx2` / `--avx512` / `--neon` | y | SIMD 指令集（自动检测） |
| `--sve` | n | ARM SVE（实验性） |
| `--brotli` | y | BROTLI 编解码器（来自 xmake 仓库的 libbrotli） |
| `--fuzz` | n | 构建模糊测试目标（使用 `--toolchain=clang`）；然后可用 `--fuzz_engine=libFuzzer\|AFL`、`--fuzz_sanitizers=y\|n` |

</div>

<details>
<summary><b>Windows</b>（MSVC、clang-cl、MinGW-w64）</summary>

<br/>

支持 MSVC（x64 和 Win32）、clang-cl 和 MinGW-w64 GCC，静态或共享均可。`CMakePresets.json` 提供了 `msvc`、`msvc-shared`、`clang-cl` 和 `mingw` 预设：

```powershell
# from a Developer PowerShell for VS (or with MinGW-w64 gcc + ninja on PATH for "mingw")
cmake --preset msvc
cmake --build --preset msvc
ctest --preset msvc
```

- 手动配置时，除了 C 编译器，还要设置 C++ 编译器（例如 `CC=clang-cl CXX=clang-cl`）：获取的 zstd 项目会启用 C++，否则 CMake 会选中不匹配的编译器驱动。
- 使用 `-DCARQUET_BUILD_SHARED=ON` 时会得到 `carquet.dll` 以及 `carquet.lib` 导入库（MinGW 下为 `libcarquet.dll` / `libcarquet.dll.a`）。使用方必须定义 `CARQUET_BUILD_SHARED`；`find_package(carquet)` 和 `carquet.pc` 会自动添加。请将 DLL 放在可执行文件旁边或 `PATH` 中。
- carquet 交给你的内存（例如 `carquet_writer_get_buffer()` 的输出）请用 `carquet_free()` 而非 `free()` 释放。DLL 和你的程序可能使用不同的 C 运行时堆（Debug 与 Release CRT、`/MT`、MinGW 与 MSVC）。
- 文件路径可以是 UTF-8（非 ASCII 文件名可用）。`carquet` CLI 接受 Unicode 参数，并向控制台输出 UTF-8。
- 启用 OpenMP 的 MinGW 构建在运行时需要 `libgomp-1.dll` 和 `libwinpthread-1.dll`。MSVC 构建需要 `vcomp140.dll`（来自 Visual C++ Redistributable）。传入 `-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON` 可得到没有该依赖的构建。
- Windows Defender 的机器学习启发式检测可能会将 `test_encoding_roundtrip.exe` 和 `test_fuzz_regressions.exe` 误判为 `Trojan:Win64/CobaltStrike.RDH!MTB` 并隔离（这是对其内嵌二进制测试数据的误报；ctest 随后会将它们报告为 "Not Run"）。将构建目录排除在扫描之外即可运行完整测试套件。

</details>

<details>
<summary><b>使用你自己的 zstd / zlib / lz4 / brotli</b>（将 carquet 嵌入更大的 CMake 项目）</summary>

<br/>

当 carquet 通过 `add_subdirectory()` 或 `FetchContent` 引入时，它按以下顺序解析每个压缩库：

1. 由 `CARQUET_ZSTD_TARGET`、`CARQUET_ZLIB_TARGET`、`CARQUET_LZ4_TARGET` 或 `CARQUET_BROTLI_TARGET` 指定的目标（若已设置）。
2. 系统已安装的副本（`CARQUET_BUNDLE_DEPS` 为 ON 时跳过）。
3. 它自行获取并构建的副本。如果你的项目已经以常用目标名（`libzstd_static`、`zlibstatic`、`lz4_static`）构建了该库，carquet 会复用该目标，而不是再获取一份。

因此，自带这些库且系统中没有副本的项目无需任何配置。若要显式选择，请指定目标名：

```cmake
set(CARQUET_ZLIB_TARGET zlibstatic)
add_subdirectory(carquet)
```

指定的目标会原样链接，不做查找也不获取；如果给出的名字不是一个目标，则会产生配置错误，而不是静默回退。该目标可以是导入的或在源码树内构建的、静态的或共享的，并且必须以使用要求（usage requirement）的形式携带其头文件目录。

</details>

<details>
<summary><b>测试、Sanitizer 与 API 文档</b></summary>

<br/>

```bash
make test                                    # Debug build + ctest
make dev && ctest --test-dir build           # everything, including examples and interop programs
cmake -B build -DCARQUET_BUILD_DOCS=ON && cmake --build build --target docs   # Doxygen → build/docs/html/
```

[CONTRIBUTING.md](CONTRIBUTING.md) 描述了每个 Pull Request 必须通过的检查关卡（测试、Sanitizer、模糊测试、互操作、基准测试）。

</details>

<a id="ai-coding-agents"></a>

<details>
<summary><b>AI 编程助手</b>（为 Claude Code 和 Codex 准备的现成技能）</summary>

<br/>

[`.agents/skills/carquet/`](.agents/skills/carquet/) 是一个 [Agent Skill](https://agentskills.io)，用于教会编程助手如何将 carquet 集成到你的项目中：构建接线（CMake、FetchContent、pkg-config、xmake）、何种任务用何种 API、完整的示例配方、层级 / 空值位图 / 生命周期规则、类型映射以及常见陷阱及其修复方法。把它复制或链接到你的仓库中，助手会自动识别并使用：

```bash
mkdir -p .claude/skills && cp -r path/to/carquet/.agents/skills/carquet .claude/skills/   # Claude Code
mkdir -p .agents/skills && cp -r path/to/carquet/.agents/skills/carquet .agents/skills/   # Codex CLI
ln -s ../../third_party/carquet/.agents/skills/carquet .claude/skills/carquet               # vendored: stays in sync
```

这些文件同样可以作为个人技能使用（`~/.claude/skills/`、`~/.codex/skills/`）。

</details>

<br/>

---

<!-- ────────────────────────────  C API  ──────────────────────────── -->

## C API

下面是一个写入示例和一个读取示例；其余内容见 [`docs/` 中的手册](docs/README.md)。

<br/>

### 写入 Parquet 文件

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

### 读取 Parquet 文件

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

### 更多示例

平面读写之外的所有内容都在手册中；每一页都链接到一个可运行的示例：

<div align="center">

| 你想要… | 参见 |
|---|---|
| 可空列、行组、缓冲区输出 | [`docs/writing.md`](docs/writing.md) |
| 列表、映射、组、定义/重复层级 | [`docs/nested-data.md`](docs/nested-data.md) |
| 列投影、统计信息、元数据检视 | [`docs/reading.md`](docs/reading.md) |
| 谓词下推、页级过滤 | [`docs/reading.md`](docs/reading.md) |
| 向现有文件追加行组 | [`docs/writing.md`](docs/writing.md#append-to-an-existing-file) |
| 压缩、自定义编解码器、写入器调优 | [`docs/writing.md`](docs/writing.md)、[`docs/performance.md`](docs/performance.md) |
| mmap、零拷贝、预缓冲、I/O 合并 | [`docs/performance.md`](docs/performance.md) |
| 错误码与恢复提示 | [`docs/error-handling.md`](docs/error-handling.md) |

</div>

<br/>

### API 参考

完整 API 位于 [`include/carquet/carquet.h`](include/carquet/carquet.h)。主要类型：

<div align="center">

| 类型 | 用途 |
|------|---------|
| `carquet_reader_t` | 文件读取器（可从路径、`FILE*` 或内存缓冲区打开） |
| `carquet_writer_t` | 文件写入器 |
| `carquet_batch_reader_t` | 高层批量迭代 |
| `carquet_schema_t` | 模式定义与自省 |
| `carquet_error_t` | 丰富的错误信息（错误码、消息、源码位置、恢复提示） |

</div>

[手册](docs/README.md)说明了何时使用哪一层接口；源码布局和架构记录在 [CONTRIBUTING.md](CONTRIBUTING.md) 中。可以用 Doxygen 从头文件生成可浏览的 HTML 参考文档，见[构建配置](#构建配置)下的 *测试、Sanitizer 与 API 文档*。

<details>
<summary><b>示例应用：MetricStore</b>（<code>mocklib/</code>）</summary>

<br/>

[`mocklib/`](mocklib/) 是 **MetricStore**，一个构建在 carquet 之上的完整、自包含的示例应用，既可作为真实使用场景的参考，也可作为公共 API 的端到端集成测试。

它模拟了一个时间序列遥测存储（摄入事件，然后进行检视和查询），在此过程中调用了 **carquet 的 143 个公共函数中的 117 个（约 82 %）**：

- **写入侧**：带逻辑类型的模式构建、每列的编码 / 压缩 / 布隆过滤器调优、页索引和统计信息
- **读取侧**：列投影、谓词下推、页级过滤、布隆过滤器成员查询、Arrow C Data Interface 桥接、嵌套 `LIST` 重建、缓冲区/追加 I/O 以及元数据自省

它像任何下游项目一样链接 carquet（`find_package(carquet)` / `add_subdirectory`），附带一个自检的往返测试，其 `write_sample` 可执行文件会生成一个文件供外部检视。见 [`mocklib/README.md`](mocklib/README.md)。

</details>

<br/>

---

<!-- ────────────────────────────  PYTHON API  ──────────────────────────── -->

## Python API

PyPI 上的 `carquet` 并不是对上述 C 调用的简单封装：它是一套构建于同一引擎之上的 **Python 风格、类 `pyarrow.parquet` 的 API**（`read_table`、`ParquetFile`、`ParquetWriter`、DNF `filters`）。wheel 内置了编译进全部编解码器的 C 库；数据通过 Arrow C Data Interface 传入 Python，因此解码、解压和 I/O 都在 C 中运行且不持有 GIL，PyArrow 直接接收这些缓冲区。

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

| 你想要… | 使用 |
|---|---|
| 读取文件、目录、字节串或文件对象 | `cq.read_table`、`cq.read_pandas` |
| 流式处理大文件 | `ParquetFile.iter_batches`、`cq.iter_batches` |
| 利用统计信息、页索引、布隆过滤器剪枝 | `filters=`（DNF 或 `pyarrow.compute` 表达式）、`ParquetFile.bloom_filter` |
| 写入表、DataFrame、polars、DuckDB 结果 | `cq.write_table`、`cq.ParquetWriter`（增量写入，`append=True`） |
| 不依赖 PyArrow 进行检视 | `cq.read_metadata`、`cq.read_parquet_schema`、`cq.info`、`cq.validate` |

</div>

类型与 `pyarrow.parquet` 之间可精确往返（时区、分类类型、十进制数、large 类型、嵌套数据）。完整参考见 [`python/README.md`](python/README.md)。

<br/>

---

<!-- ────────────────────────────  CLI 工具  ──────────────────────────── -->

## CLI 工具

Carquet 附带一个命令行工具，用于检视 Parquet 文件和生成 C 读取器代码。它默认与库一起构建和安装。

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

典型用法：

```bash
carquet schema data.parquet
carquet head -n 20 data.parquet
carquet stat data.parquet
carquet validate data.parquet
```

<br/>

### 行过滤

`cat`、`count`、`head`、`tail`、`sample` 和 `export` 接受 `-p / --filter EXPR`，将行谓词下推到页级别：只有列索引 min/max 可能匹配该谓词的页才会被解压。

```bash
carquet cat -p "price > 100 AND status = 'active'" data.parquet
carquet count --filter "id >= 1000" data.parquet
carquet export --filter "ts IS NOT NULL" -c id,ts data.parquet
```

语法为 `column OP value [AND column OP value]...`，其中 `OP` ∈ {`=`, `==`, `!=`, `<>`, `<`, `<=`, `>`, `>=`}，另有 `column IS NULL` / `column IS NOT NULL`。

> 过滤要求文件带有页索引（`write_page_index = true`）。

<br/>

### 代码生成

从任意 Parquet 文件的模式生成一个完整、可编译的 C 读取器：

```bash
carquet codegen -f data.parquet -o reader.c
# Generated: reader.c
# Compile:   clang -o reader reader.c -I.../include -L.../build -lcarquet ...

./reader                    # reads data.parquet (embedded as default)
./reader other.parquet      # override with a different file
```

<details>
<summary><b>codegen 选项</b></summary>

<br/>

<div align="center">

| 标志 | 说明 |
|------|-------------|
| `-f`, `--file FILE` | 用于读取模式的 Parquet 文件 |
| `-o`, `--output FILE` | 输出源文件（默认：stdout） |
| `--mmap` | 在生成的代码中使用内存映射 I/O |
| `--skeleton` | 生成空的 `process_batch` 以便填入自定义逻辑 |
| `-c`, `--columns COLS` | 逗号分隔的列过滤器 |
| `-b`, `--batch-size N` | 批大小（默认：1024） |

</div>

</details>

<br/>

---

<!-- ────────────────────────────  功能支持  ──────────────────────────── -->

## Parquet 功能支持

本库目前实现的功能，按领域列出。

<br/>

### 格式

<div align="center">

| 功能 | 状态 |
|---------|--------|
| 物理类型 | ✓  全部 8 种（从 BOOLEAN 到 FIXED_LEN_BYTE_ARRAY） |
| 逻辑类型 | ✓  STRING, DATE, TIME, TIMESTAMP, DECIMAL, UUID, JSON, INTERVAL, FLOAT16, VARIANT, GEOMETRY, GEOGRAPHY |
| 编码 | ✓  PLAIN, RLE, DICTIONARY, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY, BYTE_STREAM_SPLIT（读 + 写） |
| 数据页版本 | ✓  V1（默认）和 V2（读 + 写） |
| 压缩 | ✓  UNCOMPRESSED、SNAPPY、GZIP、LZ4（Hadoop 帧格式，编解码器 5）、LZ4_RAW（编解码器 7）、ZSTD、BROTLI（`CARQUET_ENABLE_BROTLI`，默认 ON）；LZO 可通过 `carquet_register_codec()` 注册 |
| 嵌套模式 | ✓  带定义/重复层级的组、列表、映射；写入时单层 LIST/MAP 自动拆分（`carquet_writer_write_list_column`），读取时重建 List；生成嵌套的 `ARROW:schema` |
| 加密 | ✗ 不支持；加密页脚文件（`PARE`）会以 `CARQUET_ERROR_NOT_IMPLEMENTED` 报告并附带明确的错误消息 |

</div>

<details>
<summary><b>索引与剪枝</b></summary>

<br/>

<div align="center">

| 功能 | 状态 |
|---------|--------|
| 布隆过滤器 | ✓  读取、写入和查询（`carquet_bloom_filter_check_*`）；自动用于行组剪枝 |
| 页索引 | ✓  列索引 + 偏移索引（读 + 写 + 按页访问统计信息） |
| 统计信息 | ✓  每个列块的 min/max/空值计数；字典列的精确 `distinct_count`；Parquet 2.9 SizeStatistics（未编码的字节数组字节数 + 层级直方图） |
| 谓词下推 | ✓  通过统计信息 + 布隆过滤器自动进行行组剪枝（无需回调）；通过列索引进行页级过滤（`carquet_batch_reader_set_page_filter`） |
| 地理空间统计信息 | ✓  GEOMETRY/GEOGRAPHY 的每列边界框和几何类型（`carquet_reader_geospatial_statistics`） |
| 列投影 | ✓  只读取选定的列 |
| 行窗口 | ✓  读取 `[offset, limit)` 范围的行，整页跳过（`carquet_batch_reader_set_row_range`） |

</div>

</details>

<details>
<summary><b>I/O 与可扩展性</b></summary>

<br/>

<div align="center">

| 功能 | 状态 |
|---------|--------|
| 追加 | ✓  向现有文件添加行组（`carquet_writer_open_append`） |
| 自定义编解码器 | ✓  为每个编解码器槽位注册自定义的压缩/解压实现（`carquet_register_codec`） |
| 键值元数据 | ✓  读写任意页脚元数据 |
| 字段级元数据 | ✓  通过 `ARROW:schema` 支持 Arrow `Field.custom_metadata`（变量标签/描述）（读 + 写） |
| 列级选项 | ✓  每列独立的编码、压缩、统计信息、布隆过滤器 |
| 缓冲区写入器 | ✓  将 Parquet 写入内存缓冲区 |
| CRC32 | ✓  页级校验（ARM 上硬件加速） |
| 内存映射 I/O | ✓  未压缩 PLAIN 数据的零拷贝读取 |
| 保留字典的读取 | ✓  返回 RLE 索引 + 字典而非物化后的值（`preserve_dictionaries`、`carquet_row_batch_column_dictionary`） |
| I/O 合并 | ✓  单次 I/O 预缓冲多列读取 |
| 推测性页脚读取 | ✓  大多数文件只需一次 I/O 即可打开 |
| OpenMP 并行读取 | ✓  可用时启用 |
| 并行写入 | ✓  跨列的页级并行编码/压缩（OpenMP）+ 后台行组 I/O（`async_io`） |

</div>

</details>

<br/>

---

<!-- ────────────────────────────  互操作性  ──────────────────────────── -->

## 互操作性

Carquet 生成的文件与 PyArrow、DuckDB、Spark 以及任何 Parquet 读取器完全兼容。

```python
import pyarrow.parquet as pq
table = pq.read_table("carquet_output.parquet")  # just works
```

```sql
-- DuckDB
SELECT * FROM read_parquet('carquet_output.parquet');
```

<details>
<summary><b>互操作测试套件</b></summary>

<br/>

双向往返测试（C 写入 / Python 读取，以及反向）：

```bash
cmake -B build -DCARQUET_BUILD_INTEROP=ON && cmake --build build
python3 interop/run_interop.py
```

大型多行组 Snappy 文件（许多行组、宽 `BYTE_ARRAY` 列、超过 2 GiB 的页偏移）有自己的生成器，它用 PyArrow 构建文件并检查 carquet 能解码每一页：

```bash
CARQUET_BIN=build/carquet python3 interop/snappy_multi_rowgroup.py --rows 150000 --row-groups 92
```

</details>

<br/>

---

<!-- ────────────────────────────  贡献与许可证  ──────────────────────────── -->

## 贡献、引用与许可证

- **贡献**：[CONTRIBUTING.md](CONTRIBUTING.md) 涵盖环境搭建、代码风格以及每个 Pull Request 必须通过的检查关卡（测试、Sanitizer、模糊测试、互操作、基准测试）。发布说明见 [CHANGELOG.md](CHANGELOG.md)。
- **引用**：仓库包含 [`CITATION.cff`](CITATION.cff)；GitHub 的 "Cite this repository" 按钮可将其格式化为 BibTeX 或 APA。
- **许可证**：MIT，见 [LICENSE](LICENSE)。
