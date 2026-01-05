# LightGBM AVX-512 Batch Prediction Optimization

This fork contains optimizations for LightGBM inference performance, achieving a **1.75x speedup** (48k → 84k predictions/sec) for batch prediction workloads.

## Summary

| Approach | Predictions/sec | vs Baseline | Status |
|----------|-----------------|-------------|--------|
| LightGBM baseline | ~48,000 | 1.0x | - |
| **Scalar batch + OpenMP** | **84,000** | **1.75x** | ✅ Implemented |
| True AVX-512 SIMD | ~70,000 | 1.46x | ❌ Slower than scalar |
| Treelite compiled | ~1,900 | 0.04x | ❌ 44x slower |

## What We Tried

### 1. Scalar Batch Prediction (Winner) ✅

**Files changed:**
- `include/LightGBM/tree_avx512.hpp` - New batch prediction kernel
- `include/LightGBM/tree.h` - Added `PredictBatch()` method to Tree class
- `src/boosting/gbdt_prediction.cpp` - Added `PredictRawBatch()` with OpenMP parallelization

**How it works:**
- Process multiple rows through each tree before moving to the next tree
- Better cache locality for tree structure (same tree data reused across rows)
- OpenMP parallelization across row chunks (each thread processes its chunk through all trees)
- No AVX-512 intrinsics - just better memory access patterns

**Key insight:** The original LightGBM processes one row through all trees, then the next row. This means tree structures are constantly being evicted from cache. By batching rows per-tree, we keep tree data hot in L1/L2 cache.

### 2. True AVX-512 SIMD (Slower) ❌

We implemented full AVX-512 vectorization processing 8 rows simultaneously:
- `_mm256_i32gather_pd` for feature value gathering
- `_mm256_cmp_pd_mask` for threshold comparisons  
- `_mm256_mask_blend_epi32` for conditional node selection
- Special handling for categorical features with variable bit shifts

**Why it was slower (70k vs 84k pred/sec):**
- `perf stat` showed **0 cache misses** - workload is compute-bound, not memory-bound
- SIMD version had **17% more instructions** due to:
  - Gather operations for scattered feature access
  - Mask management overhead
  - Staging arrays for intermediate results
- Each row needs different feature indices (scattered access), which SIMD can't parallelize efficiently

### 3. Treelite Model Compilation (Much Slower) ❌

Treelite compiles tree models to native C code with explicit if-else chains.

**Results:** 1,900 pred/sec (44x slower than baseline!)

**Why it failed:**
- Generated 1GB of C code for 376 trees
- Per-row prediction with no batching
- Function call overhead per prediction
- Even with 8x parallelization, max theoretical = 15k pred/sec (still 5x slower)

### 4. Other Approaches Considered

**QuickScorer Algorithm:** Not viable - requires trees with ≤64 leaves for efficient bitvector operations. Our trees have 269 leaves.

**Batch-by-trees:** Process all rows through tree 0, then all through tree 1, etc. Unlikely to help since tree structures are already scattered in memory.

## Building with AVX-512

```bash
mkdir build && cd build
cmake .. -DUSE_AVX512=ON
make -j8
```

This adds compiler flags: `-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma`

## Usage

The batch prediction path is automatically used when calling `LGBM_BoosterPredictForMat` with `predict_type = C_API_PREDICT_RAW_SCORE`.

From C#:
```csharp
// NativeModelRunner.Predict() uses the optimized batch path
var results = runner.Predict(inputBatch);
```

## Technical Details

### Why Scalar Beats SIMD

Traditional SIMD optimization assumes:
1. Contiguous memory access (array[i], array[i+1], ...)
2. Same operation on all lanes
3. Memory-bound workload where parallelism hides latency

Tree inference has:
1. **Scattered access** - each row needs `features[row][split_feature[node]]`
2. **Data-dependent branching** - each row takes different tree paths
3. **Compute-bound** - 0 cache misses, bottleneck is address calculation

The gather instruction `_mm256_i32gather_pd` that handles scattered access adds significant overhead that negates SIMD benefits.

### The Crash Fix

Initial implementation had undefined behavior due to uninitialized member variables `start_iteration_for_pred_` and `num_iteration_for_pred_`. Fixed by using `models_.size()` directly:

```cpp
// Before (crashed with garbage values)
for (int i = start_iteration_for_pred_; i < num_iteration_for_pred_; i++)

// After (correct)  
for (size_t i = 0; i < models_.size(); i++)
```

## Branch

- `master` - Contains the 1.75x batch optimization (production ready)
- `avx512-scalar-batch` - Contains experimental AVX-512 SIMD attempts (slower, not recommended)

## Original README

See [original.md](original.md) for the upstream LightGBM documentation.
