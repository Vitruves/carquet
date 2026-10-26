# carquet：一个 500 KB 的纯 C Parquet 读写库，读写速度超过 Arrow C++

> 项目地址：https://github.com/Vitruves/carquet ｜ 中文 README：https://github.com/Vitruves/carquet/blob/main/README.zh-CN.md ｜ 协议：MIT

## 为什么还要写一个 Parquet 库

Apache Parquet 已经是列式存储的事实标准，但在 C/C++ 里读写它，几乎只有一条路：Apache Arrow C++。Arrow 很强大，代价也很明显，库体积 50 MB 以上，依赖链很长，完整构建要几十分钟，嵌入式设备、小工具、其他语言的运行时经常只是想"把这个文件读出来"而已。

carquet 就是为这种场景写的：

- **纯 C11**，一个头文件 `carquet/carquet.h`，Release 构建约 500 KB；
- 只依赖 zstd、zlib、lz4、brotli 四个压缩库，CMake 找不到时自动下载编译，Brotli 还可以关掉；
- 读写路径针对 x86（SSE4.2 / AVX2 / AVX-512）和 ARM（NEON / SVE）做了 SIMD 优化，运行时自动检测；
- 写出的文件 PyArrow、DuckDB、Spark 都能直接读，反过来也一样；
- 通过 Arrow C Data Interface 与 Arrow 生态零依赖互通，支持任意深度的嵌套类型。

## 性能

下面是在 Apple M3 上对 1000 万行（INT64 + DOUBLE + INT32 三列）的测试，取 11 次运行的截尾中位数，读写之间清空页缓存：

![carquet vs Arrow C++ vs PyArrow，1000 万行](https://raw.githubusercontent.com/Vitruves/carquet/main/res/img/bench_10m_m3_light.png)

| 编解码器 | carquet 写入 | Arrow C++ 写入 | carquet 读取 | Arrow C++ 读取 |
|---|---|---|---|---|
| none | 74.7 ms | 137 ms | **0.36 ms** | 14.0 ms |
| snappy | **52.7 ms** | 244 ms | **12.4 ms** | 20.6 ms |
| zstd | **60.4 ms** | 342 ms | **19.8 ms** | 26.9 ms |
| lz4 | **57.2 ms** | 256 ms | **9.2 ms** | 16.2 ms |

几点说明：

- 未压缩数据读取只要 0.36 ms，是因为 carquet 走的是 **mmap 零拷贝路径**：对 PLAIN 编码、未压缩、定长、非空的列，批量读取器直接返回指向内存映射文件的指针，一个字节都不复制；Arrow 总是会物化到自己的缓冲区。
- 压缩数据的读取没有任何捷径，每个值都要解压和解码，而且两边用的是同一个系统 lz4 / zstd 库，所以差距来自 carquet 自己的流水线：按页切分的并行解压任务、`MAP_SHARED` 映射加 `madvise`、NEON 直接转置 BYTE_STREAM_SPLIT。
- 写入快 1.8–5.7 倍，主要靠**行组级流水线编码**：行组在一个常驻线程池上编码，调用方填充下一个行组时上一个还在压缩，页面直接从编码缓冲区写入文件，不再做整体拷贝。

最公平的对比是"同一个文件两边读"（cross-read）：carquet 读 Arrow 写出的 snappy 文件 14.3 ms，Arrow 自己读 21.8 ms。所有脚本都在仓库里（`benchmark/run_benchmark.py`），数据用确定性 PRNG 生成，可以自己复现。

## 安装

### CMake

```bash
git clone https://github.com/Vitruves/carquet.git
cd carquet
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --install build          # 安装 libcarquet、头文件、carquet 命令行工具、pkg-config 和 CMake 包
```

在自己的项目里：

```cmake
find_package(carquet CONFIG REQUIRED)
target_link_libraries(app PRIVATE carquet)
```

也可以用 `FetchContent` 固定版本，或 `add_subdirectory` 直接内嵌源码；如果项目已经编译了自己的 zstd/zlib/lz4，设置 `CARQUET_ZSTD_TARGET` 等变量即可复用，carquet 不会再下载一份。xmake 用户有现成的 `xmake.lua`。

### Python

```bash
pip install carquet
```

```python
import carquet
table = carquet.read_table("data.parquet")                 # pyarrow.Table
carquet.write_table(table, "out.parquet", compression="zstd")
print(carquet.info("out.parquet").num_rows)
```

wheel 里打包了同一份 C 库（压缩库静态链入，没有外部依赖），通过 Arrow C Data Interface 把数据交给 PyArrow，Python 侧不做任何解码。

## C API 一瞥

写文件：定义 schema，按列写入，关闭。

```c
#include <carquet/carquet.h>

carquet_error_t err = CARQUET_ERROR_INIT;
carquet_schema_t* schema = carquet_schema_create(&err);
carquet_schema_add_column(schema, "id",    CARQUET_PHYSICAL_INT64,  NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);

carquet_writer_options_t opts;
carquet_writer_options_init(&opts);
opts.compression = CARQUET_COMPRESSION_ZSTD;

carquet_writer_t* w = carquet_writer_create("output.parquet", schema, &opts, &err);
int64_t ids[]   = {1, 2, 3, 4, 5};
double values[] = {1.1, 2.2, 3.3, 4.4, 5.5};
carquet_writer_write_batch(w, 0, ids, 5, NULL, NULL);
carquet_writer_write_batch(w, 1, values, 5, NULL, NULL);
carquet_writer_close(w);
carquet_schema_free(schema);
```

读文件：打开（建议开 mmap），用批量读取器逐批迭代。

```c
carquet_reader_options_t ro;
carquet_reader_options_init(&ro);
ro.use_mmap = true;
carquet_reader_t* r = carquet_reader_open("output.parquet", &ro, &err);

carquet_batch_reader_config_t cfg;
carquet_batch_reader_config_init(&cfg);           /* 默认每批 65536 行 */
carquet_batch_reader_t* br = carquet_batch_reader_create(r, &cfg, &err);

carquet_row_batch_t* batch = NULL;
while (carquet_batch_reader_next(br, &batch) == CARQUET_OK && batch) {
    const void* data; const uint8_t* valid; int64_t n;
    carquet_row_batch_column(batch, 0, &data, &valid, &n);
    const int64_t* ids = (const int64_t*)data;   /* 处理 ids[0..n) */
    carquet_row_batch_free(batch);
    batch = NULL;
}
carquet_batch_reader_free(br);
carquet_reader_close(r);
```

可空列的约定是 Parquet 原生的：写入时 `values` 只放非空值、`def_levels` 每行一个；批量读取器返回按行对齐的数据加一个有效位图。列表和映射有 `carquet_writer_write_list_column()` / `carquet_row_batch_column_list()` 这样的 Arrow 布局辅助函数，更深的嵌套直接走 Arrow 桥接（`carquet_writer_write_arrow()` / `carquet_reader_read_arrow()`）。

## 谓词下推与页级过滤

写文件时打开 `write_page_index`，读取时就能让批量读取器只解压可能命中的页：

```c
int64_t lo = 1000, hi = 2000;
carquet_filter_clause_t clause = {
    .column_index = 0, .op = CARQUET_FILTER_RANGE,
    .has_lo = true, .lo = &lo, .lo_size = sizeof lo,
    .has_hi = true, .hi = &hi, .hi_size = sizeof hi,
};
carquet_batch_reader_set_page_filter(br, &clause, 1);
```

装上过滤器后，行组级别会先用列统计信息和布隆过滤器剪枝，再按列索引决定页；谓词列不在投影里也没关系，它只会被读索引，不会被解压。

## 命令行工具

库自带的 `carquet` 命令可以不写代码地检查文件：

```bash
carquet schema data.parquet                     # schema
carquet info data.parquet                       # 行组、编码、压缩、大小
carquet stat data.parquet                       # 每列统计信息
carquet cat -p "price > 100 AND status = 'active'" data.parquet   # 带谓词的行输出
carquet validate data.parquet                   # 校验页 CRC 与结构
carquet codegen -f data.parquet -o reader.c     # 生成一个可编译的 C 读取程序
```

## 功能覆盖

- 全部 8 种物理类型；STRING、DATE、TIME、TIMESTAMP、DECIMAL、UUID、JSON、INTERVAL、FLOAT16、VARIANT、GEOMETRY、GEOGRAPHY 逻辑类型；
- PLAIN、RLE、DICTIONARY、DELTA_BINARY_PACKED、DELTA_LENGTH_BYTE_ARRAY、DELTA_BYTE_ARRAY、BYTE_STREAM_SPLIT 编码，Data Page V1 和 V2，读写都支持；
- Snappy、GZIP、LZ4（两种帧格式）、ZSTD、Brotli；
- 布隆过滤器、列索引 / 偏移索引、列统计（含 Parquet 2.9 的 SizeStatistics）、地理空间统计、排序列声明；
- 追加行组到已有文件、写到内存缓冲区、自定义压缩器注册、键值元数据与 Arrow 字段元数据；
- 目前不支持模块化加密（遇到加密文件会给出明确错误）。

## 对 AI 编程助手的支持

仓库里带了一个 Agent Skill（`.agents/skills/carquet/`），内容是构建接入方式、API 选型、完整示例、常见坑和类型对照表。把它复制到自己项目的 `.claude/skills/` 或 `.agents/skills/` 目录，Claude Code 或 Codex 就知道怎么正确地把 carquet 接进来，写可空列、嵌套列时不会踩错 definition level 的约定。

## 质量保证

每个版本在 Linux、macOS、Windows 上跑 42 组单元测试，加上 AddressSanitizer / UBSan、12 个 libFuzzer 目标、与 PyArrow 和 DuckDB 的双向互操作测试（含 59 个外部语料文件和 8 个畸形文件）。0.7.2 版本刚做完一轮全库审计，修复了 35 个问题，压缩读写分别提速 10–17 % 和 10–15 %。

## 总结

如果你的需求是在 C/C++ 程序、嵌入式环境或某个语言运行时里**快速、正确地读写 Parquet**，而不是整个 Arrow 生态，carquet 是一个很小的选择：一个头文件，几百 KB，一分钟构建，速度不输给大家伙。欢迎试用、提 issue、点个 star。

- GitHub：https://github.com/Vitruves/carquet
- 文档：https://github.com/Vitruves/carquet/tree/main/docs
- 更新日志：https://github.com/Vitruves/carquet/blob/main/CHANGELOG.md
