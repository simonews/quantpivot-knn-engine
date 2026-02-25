# QuantPivot KNN Engine – High-Performance Nearest Neighbor Search

## Overview

High-performance implementation of the K-Nearest Neighbors algorithm optimized through pivot-based pruning, sparse quantization and low-level SIMD vectorization (SSE/AVX), achieving up to **14x speedup** over a sequential C baseline.

The project combines algorithmic pruning techniques with hardware-level optimizations and multi-threaded execution to significantly reduce distance computations and overall runtime.

---

## Key Features

- **Pivot-based pruning** – Reduces distance computations by 70–90% using the triangle inequality.
- **Sparse quantization** – Binary vector representation for fast approximate filtering before exact refinement.
- **SIMD vectorization (x86 Assembly)** – Hand-optimized implementations using:
  - SSE (32-bit, float – XMM registers)
  - AVX (64-bit, double – YMM registers)
- **Multi-threading (OpenMP)** – Parallel execution with dynamic scheduling and thread-private buffers.
- **Residual handling** – Correct support for arbitrary vector dimensions (including non-multiples of 4).

---

## Performance

Benchmark configuration: 2000 query vectors · 2000 dataset points · 256 dimensions

| Version          | Architecture | SIMD | Threads | FIT Time | PREDICT Time | Total    | Speedup   |
|------------------|--------------|------|---------|----------|--------------|----------|-----------|
| C Sequential     | 64-bit       | No   | 1       | ~0.15s   | ~0.25s       | ~0.40s   | 1.0x      |
| OpenMP           | 64-bit       | No   | 4       | 0.029s   | 0.056s       | 0.085s   | 4.7x      |
| AVX              | 64-bit       | AVX  | 1       | 0.012s   | 0.020s       | 0.032s   | 12.5x     |
| **AVX + OpenMP** | **64-bit**   | **AVX** | **4** | **0.004s** | **0.007s** | **0.011s** | **~14x** |

---

## Repository Structure
```
.
├── ProgettoGruppo11/
│   └── src/
│       ├── 32bit/      # 32-bit SSE implementation (float)
│       ├── 64bit/      # 64-bit AVX sequential implementation (double)
│       └── 64omp/      # 64-bit AVX + OpenMP implementation
├── docs/               # Technical documentation and report
├── .gitignore
└── test.py
```

---

## Requirements

| Dependency    | Minimum Version | Notes                                   |
|---------------|-----------------|-----------------------------------------|
| GCC           | 7.0+            | With OpenMP support                     |
| NASM          | 2.13+           | Assembler                               |
| CPU           | Sandy Bridge+   | x86-64 with AVX support                 |
| OS            | Ubuntu 20.04+   | Linux only                              |

---

## Build and Execution

### Recommended Version – AVX + OpenMP (64-bit)
```bash
cd src/64omp
make clean && make

# Run with default number of threads
./main64omp

# Run with specific number of threads
OMP_NUM_THREADS=4 ./main64omp
```

### Other Versions
```bash
# 32-bit SSE
cd src/32bit && make && ./main32

# 64-bit AVX (sequential)
cd src/64bit && make && ./main64
```

---

## Algorithm Design

### Phase 1 – FIT (Index Construction)

1. Selection of `h` pivot points from the dataset
2. Quantization of all vectors into sparse binary representation
3. Pre-computation of approximate distances from each point to pivots

**Complexity:** `O(N × h × D)`

### Phase 2 – PREDICT (Query Search)

1. Quantization of the query vector
2. Query-to-pivot distance computation
3. Candidate pruning via triangle inequality
4. Exact Euclidean distance refinement on filtered candidates
5. Sorting and return of top-K neighbors

**Complexity:** `O(nq × N × h)`, with pruning rate between 70–90%

---

## SIMD Implementation Details

### Vectorized Euclidean Distance

Both versions process **4 elements per iteration**:

- **32-bit SSE:** 4 floats using 128-bit XMM registers
- **64-bit AVX:** 4 doubles using 256-bit YMM registers

### Residual Element Handling

For dimensions not divisible by 4:
- Main loop processes `D / 4` iterations
- Residual loop processes `D mod 4` elements
- Scalar accumulation ensures numerical correctness

**Example (D = 258):**
- 64 vectorized iterations → 256 elements
- 2 scalar iterations → 2 elements
- Total: 258 elements (validated with error < 1e-15)

---

## Thread Safety (OpenMP Version)

Thread safety is guaranteed through:
- Thread-private working buffers
- Disjoint memory writes
- Independent SIMD register usage per core
- Dynamic scheduling to balance irregular pruning workloads

---

## Data Format

Binary `.ds2` format:
```
[4 bytes]           Number of rows (N)
[4 bytes]           Number of columns (D)
[N × D × sizeof(T)] Matrix data (row-major order)
```

- **32-bit version:** `float` (4 bytes per element)
- **64-bit version:** `double` (8 bytes per element)

---

## Configuration Parameters

Defined in `main.c`:
```c
int h = 20;   // Number of pivots
int k = 8;    // Number of nearest neighbors
int x = 2;    // Sparsity parameter for quantization
```

Recommended pivot values by dataset size:

| Dataset Size | RAM Usage     | Runtime (4 threads) | Recommendation                   |
|--------------|---------------|---------------------|----------------------------------|
| N < 10K      | < 100 MB      | < 1s                | Sequential sufficient            |
| 10K – 100K   | 100 MB – 1 GB | 1–30s               | OpenMP recommended               |
| 100K – 500K  | 1–5 GB        | 30s – 5min          | Increase pivot count             |
| N > 500K     | > 5 GB        | > 5min              | Consider approximate methods (LSH/HNSW) |

---

## Correctness Verification

Automatic validation tests are included:
```bash
./main64omp
```

Expected output:
```
[TEST] Verifying euclidean_distance_asm...
      C Distance:   6.408924176
      ASM Distance: 6.408924176
      Difference:   1.78e-15
      TEST PASSED
```

All versions are verified for:
- Correctness for `D mod 4 ∈ {0, 1, 2, 3}`
- Numerical stability (< 1e-6 for float, < 1e-15 for double)
- Thread safety under concurrent execution

---

## Documentation

A detailed technical report (ITA) describing algorithm design, optimization strategies and benchmarking methodology is available at:
```
docs/quantpivot-report.pdf
```

---

## Authors

- **Andrea Attadia**
- **Vito Simone Goffredo**
- **Christian Iuele**

Developed as part of the *Advanced Architectures of Processing and Programming Systems* course.
