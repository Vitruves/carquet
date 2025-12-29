#!/bin/bash
#
# Run Carquet vs PyArrow benchmark comparison
#
# Usage:
#   ./run_benchmark.sh                    # Uses python3 from PATH
#   PYTHON=/path/to/python ./run_benchmark.sh  # Uses specified Python
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"

# Find Python: use PYTHON env var if set, otherwise find python3
if [ -n "${PYTHON}" ]; then
    # User specified Python
    if [ ! -x "${PYTHON}" ]; then
        echo "Error: Specified PYTHON='${PYTHON}' is not executable"
        exit 1
    fi
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "Error: No Python interpreter found"
    echo "Please install Python 3 or set PYTHON environment variable"
    exit 1
fi

# Verify Python has required modules
if ! "${PYTHON}" -c "import pyarrow.parquet" 2>/dev/null; then
    echo "Error: PyArrow not found in Python environment"
    echo "Using: ${PYTHON}"
    echo ""
    echo "Install with: ${PYTHON} -m pip install pyarrow"
    exit 1
fi

echo "Using Python: ${PYTHON}"
echo ""
echo "========================================"
echo "  Carquet vs PyArrow Benchmark"
echo "========================================"
echo ""

# Build benchmark if needed
if [ ! -f "${BUILD_DIR}/benchmark_carquet" ]; then
    echo "Building Carquet benchmark..."
    cd "${BUILD_DIR}"
    make benchmark_carquet 2>/dev/null || cmake --build . --target benchmark_carquet
    cd "${SCRIPT_DIR}"
fi

# Create results file
RESULTS_FILE="${SCRIPT_DIR}/results.csv"
echo "library,dataset,compression,rows,write_ms,read_ms,file_bytes" > "${RESULTS_FILE}"

echo ""
echo "Running Carquet benchmark..."
echo "----------------------------------------"
"${BUILD_DIR}/benchmark_carquet" 2>&1 | tee /tmp/carquet_output.txt
grep "^CSV:" /tmp/carquet_output.txt | sed 's/^CSV://' >> "${RESULTS_FILE}"

echo ""
echo "Running PyArrow benchmark..."
echo "----------------------------------------"
"${PYTHON}" "${SCRIPT_DIR}/benchmark_pyarrow.py" 2>&1 | tee /tmp/pyarrow_output.txt
grep "^CSV:" /tmp/pyarrow_output.txt | sed 's/^CSV://' >> "${RESULTS_FILE}"

echo ""
echo "========================================"
echo "  Results Summary"
echo "========================================"
echo ""

# Parse and display comparison
"${PYTHON}" << 'PYTHON_SCRIPT'
import csv

results = {}
with open('/tmp/carquet_output.txt') as f:
    for line in f:
        if line.startswith('CSV:'):
            parts = line.strip()[4:].split(',')
            key = (parts[1], parts[2])  # (dataset, compression)
            results[('carquet', key)] = {
                'write_ms': float(parts[4]),
                'read_ms': float(parts[5]),
                'file_bytes': int(parts[6])
            }

with open('/tmp/pyarrow_output.txt') as f:
    for line in f:
        if line.startswith('CSV:'):
            parts = line.strip()[4:].split(',')
            key = (parts[1], parts[2])
            results[('pyarrow', key)] = {
                'write_ms': float(parts[4]),
                'read_ms': float(parts[5]),
                'file_bytes': int(parts[6])
            }

print(f"{'Dataset':<10} {'Compress':<8} {'Metric':<8} {'Carquet':>12} {'PyArrow':>12} {'Speedup':>10}")
print("-" * 70)

for dataset in ['small', 'medium', 'large']:
    for compression in ['none', 'snappy', 'zstd']:
        key = (dataset, compression)
        c = results.get(('carquet', key), {})
        p = results.get(('pyarrow', key), {})

        if c and p:
            # Write comparison
            c_write = c['write_ms']
            p_write = p['write_ms']
            speedup_write = p_write / c_write if c_write > 0 else 0
            print(f"{dataset:<10} {compression:<8} {'Write':<8} {c_write:>10.2f}ms {p_write:>10.2f}ms {speedup_write:>9.2f}x")

            # Read comparison
            c_read = c['read_ms']
            p_read = p['read_ms']
            speedup_read = p_read / c_read if c_read > 0 else 0
            print(f"{'':<10} {'':<8} {'Read':<8} {c_read:>10.2f}ms {p_read:>10.2f}ms {speedup_read:>9.2f}x")

            # File size comparison
            c_size = c['file_bytes'] / (1024*1024)
            p_size = p['file_bytes'] / (1024*1024)
            ratio = c_size / p_size if p_size > 0 else 0
            print(f"{'':<10} {'':<8} {'Size':<8} {c_size:>10.2f}MB {p_size:>10.2f}MB {ratio:>9.2f}x")
            print()

print("\nSpeedup > 1.0 means Carquet is faster")
print("Size ratio < 1.0 means Carquet produces smaller files")
PYTHON_SCRIPT

echo ""
echo "Full results saved to: ${RESULTS_FILE}"
