# Carquet Fuzzing

This directory contains fuzz targets for testing carquet with random/malformed inputs.

## Quick Start

```bash
# Build and run the reader fuzzer for 5 minutes
./run_fuzzer.sh reader -max_total_time=300

# Run all fuzzers
./run_fuzzer.sh all
```

## Requirements

- **Clang** with libFuzzer support (included in Clang 6.0+)
- Or **AFL++** for alternative fuzzing

## Fuzz Targets

| Target | Description | Attack Surface |
|--------|-------------|----------------|
| `fuzz_reader` | Full Parquet file reader | File format, metadata, all decoders |
| `fuzz_compression` | Compression decoders | Snappy, LZ4, GZIP, ZSTD |
| `fuzz_encodings` | Encoding decoders | RLE, Delta, Plain |
| `fuzz_thrift` | Thrift protocol decoder | Metadata parsing |

## Building Manually

### With libFuzzer (recommended)

```bash
# From project root
cmake -B build-fuzz \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCARQUET_BUILD_FUZZ=ON \
    -DCARQUET_BUILD_TESTS=OFF

cmake --build build-fuzz
```

### With AFL++

```bash
CC=afl-clang-fast cmake -B build-afl \
    -DCARQUET_BUILD_FUZZ=ON \
    -DCARQUET_FUZZ_ENGINE=AFL \
    -DCMAKE_BUILD_TYPE=Debug

cmake --build build-afl

# Run with AFL++
afl-fuzz -i corpus_reader -o findings -- ./build-afl/fuzz/fuzz_reader_afl @@
```

## Running Fuzzers

### Basic Usage

```bash
# Run for a specific time
./build-fuzz/fuzz/fuzz_reader corpus_reader -max_total_time=3600

# Run with multiple jobs
./build-fuzz/fuzz/fuzz_reader corpus_reader -jobs=4 -workers=4

# Run until crash found
./build-fuzz/fuzz/fuzz_reader corpus_reader
```

### Useful libFuzzer Options

| Option | Description |
|--------|-------------|
| `-max_total_time=N` | Run for N seconds |
| `-jobs=N` | Run N fuzzing jobs in parallel |
| `-workers=N` | Use N worker processes |
| `-dict=file` | Use dictionary file for mutations |
| `-max_len=N` | Maximum input size |
| `-only_ascii=1` | Only use ASCII characters |

## Seed Corpus

For better fuzzing, add valid Parquet files to the corpus directories:

```bash
# Copy some valid test files
cp ../build/test_files/*.parquet corpus_reader/
```

## Crash Analysis

When a crash is found:

1. The crashing input is saved to `crash-<hash>`
2. Reproduce with: `./fuzz_reader crash-<hash>`
3. Get stack trace: `ASAN_OPTIONS=symbolize=1 ./fuzz_reader crash-<hash>`

## Coverage

To see code coverage:

```bash
# Build with coverage
cmake -B build-cov \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
    -DCARQUET_BUILD_FUZZ=ON

cmake --build build-cov

# Run fuzzer
./build-cov/fuzz/fuzz_reader corpus_reader -max_total_time=60

# Generate coverage report
llvm-profdata merge -sparse default.profraw -o default.profdata
llvm-cov show ./build-cov/fuzz/fuzz_reader -instr-profile=default.profdata
```
