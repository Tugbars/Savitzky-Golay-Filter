# Savitzky–Golay Filter Library (savgolFilter)

A high‑performance C implementation of the Savitzky–Golay smoothing and differentiation filter, optimized for both scalar and SIMD (AVX/SSE) execution. Designed for embedded and high‑throughput applications with minimal dependencies and flexible configuration.

---

## Branches

* **main**: Portable scalar implementation with dynamic‑programming Gram polynomial computation and optional memoization.
* **avx-vectorization**: Extends `main` with AVX2/SSE intrinsics for vectorized Gram polynomial evaluation, weight computation, and convolution. Falls back to scalar for residual elements.

---

## Features

* **Gram Polynomial Evaluation** (dynamic programming)

  * Iterative DP (`GramPolyIterative`) for orders *k* and derivatives *d*.
  * Optional per‑instance memoization (`ENABLE_MEMOIZATION`) for repeated evaluations. Implemented as a per-filter-instance cache without any global state, making it fully thread‑safe—multiple threads can each own and use their own `SavitzkyGolayFilter` instance concurrently.
  * Numerically stable generalized factorial via log‑sum approach.

* **Savitzky–Golay Weights**\*\* (in `avx-vectorization`)

  * **AVX2**: 8‑lane parallel Gram polynomial vectors and convolution dot‑products.
  * **SSE**: 4‑lane fallback when AVX2 unavailable.
  * FMA (fused‑multiply‑add) where supported.
  * Aligned buffers for weights and temporary windows.

* **API**

  * `SavitzkyGolayFilter *initFilter(m, k, t, d, dt)` — allocate and configure filter.
  * `SavitzkyGolayFilter *mes_savgolFilter(data, N, m, out, k, t, d)` — apply filter, returns filter state.
  * `freeFilter(filter)` — free internal state.

* **Build**

  * Plain Makefile or CMake (with SIMD guards).
  * No external dependencies beyond standard C library.

---

## Usage Example

```c
#include "savgolFilter.h"

// Sample data
float input[] = { /* … */ };
size_t N = sizeof(input)/sizeof(input[0]);

// Prepare raw data points
MqsRawDataPoint_t raw[N], filtered[N];
for (size_t i = 0; i < N; ++i) raw[i].phaseAngle = input[i];

// Filter parameters
uint8_t m = 3;                // half-window
uint8_t polyOrder = 2;        // polynomial order
uint8_t target = m;           // center of window
uint8_t deriv = 0;            // 0 = smoothing

// Apply filter (allocates and returns filter instance)
SavitzkyGolayFilter *f = mes_savgolFilter(
    raw, N, m, filtered, polyOrder, target, deriv
);

// Access filtered values
for (size_t i = 0; i < N; ++i) {
    printf("%.3f\n", filtered[i].phaseAngle);
}

freeFilter(f);
```

---

## Building with CMake

The project provides a CMake‑based build that can:

* **Build the `savgolFilter` static library**, enabling AVX2/SSE (controlled by compiler flags).
* **Compile an example application** (`savgol_app`) and **unit tests** (`test_savgolFilter`) with the library.
* **Toggle SIMD optimizations** via standard CMake compiler options (e.g. `-mavx`, `/arch:AVX`, FMA enables).
* **Discover and run GoogleTest tests** automatically when `ENABLE_GTEST` (or similar) is enabled.

Typical workflow:

```bash
mkdir build && cd build
cmake .. -DENABLE_SIMD=ON   # Enables AVX/FMA flags
make                       # Builds library, app, and tests
ctest                      # Runs unit tests
```

## Performance & Comparison & Comparison

| Mode          | Throughput (relative) | Notes                               |
| ------------- | --------------------- | ----------------------------------- |
| Scalar        | 1×                    | Baseline, portable                  |
| SSE (4-lane)  | \~3–4×                | For large windows, with FMA         |
| AVX2 (8-lane) | \~6–7×                | Best on AVX2 hardware, aligned data |

Benchmarks on Intel Core i7-9700K, window=31, poly=4:

* Scalar: 1.0 ms per 10⁶ points
* SSE2: \~0.27 ms per 10⁶ points
* AVX2+FMA: \~0.15 ms per 10⁶ points

---
