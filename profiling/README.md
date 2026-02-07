# Carquet Profiling Suite

Comprehensive profiling tools for identifying performance bottlenecks in Carquet read/write paths.

## Quick Start

```bash
# Build and run full profiling
./run_profile.sh full

# Quick statistics only
./run_profile.sh stat

# Generate flamegraph
./run_profile.sh flamegraph

# Run micro-benchmarks
./run_profile.sh micro
```

## Prerequisites

### Required
- `perf` - Linux performance profiler
  ```bash
  sudo apt install linux-tools-generic linux-tools-$(uname -r)
  ```

### Optional (for flamegraphs)
- [FlameGraph](https://github.com/brendangregg/FlameGraph)
  ```bash
  git clone https://github.com/brendangregg/FlameGraph ~/FlameGraph
  ```

## Profiling Binaries

### `profile_read`

Full read path profiler that exercises:
- Dictionary encoding with gather operations
- RLE level decoding
- Null bitmap construction
- Various compression codecs
- SIMD dispatch paths

```bash
# Basic usage
./profile_read -r 5000000 -i 10

# With dictionary encoding and 10% nulls
./profile_read -r 10000000 -d -n 1

# With ZSTD compression
./profile_read -r 5000000 -c 2

# Profile with perf
perf record -g ./profile_read -r 5000000 -i 5
perf report --hierarchy
```

Options:
| Flag | Description | Default |
|------|-------------|---------|
| `-r, --rows N` | Number of rows | 10000000 |
| `-b, --batch N` | Batch size | 262144 |
| `-i, --iterations N` | Test iterations | 10 |
| `-d, --dictionary` | Enable dictionary encoding | off |
| `-n, --nulls MODE` | Null ratio: 0=none, 1=10%, 2=30%, 3=50% | 0 |
| `-c, --compression N` | 0=none, 1=snappy, 2=zstd, 3=lz4 | 0 |
| `-v, --verbose` | Verbose output | off |

### `profile_micro`

Micro-benchmarks for isolated component profiling:

```bash
# All components
./profile_micro --component all

# RLE decoding only
./profile_micro --component rle --count 2000000 --iterations 500

# Dictionary gather
./profile_micro --component gather

# Null bitmap operations
./profile_micro --component null

# Dispatch overhead measurement
./profile_micro --component dispatch --iterations 1000000
```

Components:
- `rle` - RLE level decoding (single, batch, decode_levels API)
- `gather` - Dictionary gather operations (scalar vs SIMD)
- `null` - Null bitmap construction (count_non_nulls, build_bitmap)
- `compression` - LZ4/Snappy compress/decompress
- `dispatch` - SIMD dispatch function call overhead

## Profiling Modes

### `full` - Complete Analysis
Runs all profiling steps:
1. CPU statistics (cycles, cache, branches)
2. Call graph recording
3. Function-level report
4. Source annotations for hot functions
5. Flamegraph generation (if FlameGraph installed)

### `stat` - Quick Statistics
CPU hardware counter statistics:
- Instructions per cycle (IPC)
- Cache hit/miss rates
- Branch prediction accuracy

```bash
./run_profile.sh stat --rows 5000000
```

### `flamegraph` - Visual Call Graph
Interactive SVG flamegraph for identifying hot paths:

```bash
./run_profile.sh flamegraph --rows 10000000
```

Open `output/carquet_*_flamegraph.svg` in a browser.

### `compare` - Implementation Comparison
Compares scalar vs SIMD implementations:
- Gather operations (scalar vs dispatch)
- Null bitmap (scalar vs SIMD)
- Dispatch overhead measurement

```bash
./run_profile.sh compare
```

## Manual Profiling

### Basic perf Commands

```bash
# Record with call graph
perf record -g --call-graph dwarf ./profile_read -r 5000000

# View report interactively
perf report

# Hierarchical view
perf report --hierarchy

# Source annotation for specific function
perf annotate carquet_rle_decoder_get

# Statistics
perf stat -e cycles,instructions,cache-misses ./profile_read -r 1000000
```

### Advanced Analysis

```bash
# Branch prediction analysis
perf stat -e branch-misses,branch-instructions ./profile_micro --component rle

# Cache analysis
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
    ./profile_read -r 5000000

# Memory bandwidth
perf stat -e mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit \
    ./profile_read -r 5000000 2>/dev/null || echo "(Requires Intel CPU)"

# Record specific events
perf record -e cache-misses -g ./profile_read -r 5000000
```

### Flamegraph Generation

```bash
# Clone FlameGraph if not installed
git clone https://github.com/brendangregg/FlameGraph ~/FlameGraph

# Record
perf record -g --call-graph dwarf -F 999 ./profile_read -r 10000000 -i 5

# Generate
perf script | ~/FlameGraph/stackcollapse-perf.pl | ~/FlameGraph/flamegraph.pl > flame.svg
```

## Interpreting Results

### Key Metrics to Watch

1. **Instructions per Cycle (IPC)**
   - Good: > 1.5
   - Bad: < 0.5 (memory-bound)

2. **Cache Miss Rate**
   - L1 miss: < 5% is good
   - LLC miss: < 1% is good

3. **Branch Misprediction**
   - < 2% is good
   - > 5% indicates branchy code

### Common Bottlenecks

| Symptom | Cause | Solution |
|---------|-------|----------|
| Low IPC, high LLC misses | Random memory access | Add prefetching |
| High branch mispredictions | Unpredictable conditionals | Branchless algorithms |
| High function call overhead | Small functions called often | Inline or batch |
| Visible in flamegraph | Hot function | Optimize that function |

## Output Files

Profiling output is saved to `profiling/output/`:

| File | Description |
|------|-------------|
| `*_stat.txt` | CPU statistics |
| `*_perf.data` | Raw perf recording |
| `*_report.txt` | Hierarchical function report |
| `*_top_functions.txt` | Flat function list by samples |
| `*_flamegraph.svg` | Interactive flamegraph |
| `*_icicle.svg` | Reversed (icicle) flamegraph |
| `*_micro.txt` | Micro-benchmark results |
| `*_annotations/` | Source annotations for hot functions |

## PyArrow Comparison

The profiling tools are designed to produce comparable workloads to PyArrow:

```python
# Equivalent PyArrow benchmark
import pyarrow.parquet as pq
import time

start = time.time()
table = pq.read_table('/tmp/test.parquet')
elapsed = time.time() - start
print(f"{len(table) / elapsed / 1e6:.2f} M rows/sec")
```

Target: Carquet should achieve at least 50% of PyArrow read performance.

## Architecture-Specific Notes

### x86-64
- SSE4.2, AVX2, AVX-512 paths available
- Prefetching critical for dictionary gather
- Check dispatch overhead vs inline SIMD

### ARM64
- NEON paths typically faster than x86
- SVE available but experimental
- Less dispatch overhead due to simpler calling convention
