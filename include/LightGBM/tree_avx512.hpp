/*!
 * Copyright (c) 2024. AVX-512 batch prediction for LightGBM trees.
 * Licensed under the MIT License.
 */
#ifndef LIGHTGBM_TREE_AVX512_HPP_
#define LIGHTGBM_TREE_AVX512_HPP_

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include <cmath>
#include <cstdint>
#include <algorithm>

namespace LightGBM {

// Batch size for AVX-512 processing (8 doubles fit in 512-bit register)
constexpr int kAVX512BatchSize = 8;

/*!
 * \brief Process multiple rows through a single tree simultaneously.
 *        This improves cache locality by keeping tree data hot while
 *        processing multiple samples.
 * 
 * \param num_leaves Number of leaves in the tree
 * \param num_cat Number of categorical features (0 for numerical-only)
 * \param is_linear Whether tree has linear models at leaves
 * \param decision_type Array of decision types for each node
 * \param split_feature Array of split feature indices for each node
 * \param threshold Array of thresholds for each node
 * \param left_child Array of left child indices for each node
 * \param right_child Array of right child indices for each node
 * \param leaf_value Array of leaf output values
 * \param cat_boundaries Categorical feature boundaries (can be nullptr)
 * \param cat_threshold Categorical threshold bitsets (can be nullptr)
 * \param row_features Array of pointers to feature arrays for each row
 * \param output Output array for predictions
 * \param num_rows Number of rows to process
 */
inline void PredictTreeBatchAVX512(
    int num_leaves,
    int num_cat,
    bool is_linear,
    const int8_t* decision_type,
    const int* split_feature,
    const double* threshold,
    const int* left_child,
    const int* right_child,
    const double* leaf_value,
    const int* cat_boundaries,
    const uint32_t* cat_threshold,
    const double** row_features,
    double* output,
    int num_rows) {
  
  // Handle degenerate case: single leaf
  if (num_leaves <= 1) {
    double val = leaf_value[0];
#ifdef __AVX512F__
    __m512d val_vec = _mm512_set1_pd(val);
    int i = 0;
    for (; i + kAVX512BatchSize <= num_rows; i += kAVX512BatchSize) {
      _mm512_storeu_pd(&output[i], val_vec);
    }
    // Handle remainder
    for (; i < num_rows; ++i) {
      output[i] = val;
    }
#else
    for (int i = 0; i < num_rows; ++i) {
      output[i] = val;
    }
#endif
    return;
  }
  
  // Skip linear trees - fall back to per-row prediction (caller handles this)
  if (is_linear) {
    return;
  }
  
  // Constants for decision type masks
  constexpr int8_t kCategoricalMask = 1;
  constexpr int8_t kDefaultLeftMask = 2;
  
  // Process rows in batches of 8 for better cache utilization
  for (int row_start = 0; row_start < num_rows; row_start += kAVX512BatchSize) {
    int batch_size = std::min(kAVX512BatchSize, num_rows - row_start);
    
    // Track which node each sample is at
    alignas(64) int nodes[kAVX512BatchSize] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // Maximum iterations = maximum tree depth (safety limit)
    constexpr int kMaxIterations = 64;
    
    for (int iter = 0; iter < kMaxIterations; ++iter) {
      bool any_active = false;
      
      // Process each sample in the batch
      for (int i = 0; i < batch_size; ++i) {
        int node = nodes[i];
        
        if (node >= 0) {  // Still at internal node (not a leaf)
          any_active = true;
          
          int split_feat = split_feature[node];
          double fval = row_features[row_start + i][split_feat];
          int8_t dtype = decision_type[node];
          
          // Handle missing values (NaN)
          uint8_t missing_type = (dtype >> 2) & 3;
          
          if (std::isnan(fval)) {
            // NaN handling
            if (missing_type == 0) {  // MissingType::None
              fval = 0.0;
            } else if (missing_type == 2) {  // MissingType::NaN
              // Go to default direction
              if (dtype & kDefaultLeftMask) {
                nodes[i] = left_child[node];
              } else {
                nodes[i] = right_child[node];
              }
              continue;
            }
          }
          
          // Check if zero should be treated as missing
          if (missing_type == 1 && fval >= -1e-35 && fval <= 1e-35) {  // MissingType::Zero
            if (dtype & kDefaultLeftMask) {
              nodes[i] = left_child[node];
            } else {
              nodes[i] = right_child[node];
            }
            continue;
          }
          
          // Check if categorical or numerical split
          if ((dtype & kCategoricalMask) && num_cat > 0) {
            // Categorical decision
            int int_fval = static_cast<int>(fval);
            if (std::isnan(fval) || int_fval < 0) {
              nodes[i] = right_child[node];
            } else {
              int cat_idx = static_cast<int>(threshold[node]);
              // Check if value is in the categorical bitset
              int start = cat_boundaries[cat_idx];
              int end = cat_boundaries[cat_idx + 1];
              int num_words = end - start;
              bool found = false;
              if (int_fval < num_words * 32) {
                int word_idx = int_fval / 32;
                int bit_idx = int_fval % 32;
                if (word_idx < num_words) {
                  found = (cat_threshold[start + word_idx] >> bit_idx) & 1;
                }
              }
              nodes[i] = found ? left_child[node] : right_child[node];
            }
          } else {
            // Numerical decision
            if (fval <= threshold[node]) {
              nodes[i] = left_child[node];
            } else {
              nodes[i] = right_child[node];
            }
          }
        }
      }
      
      if (!any_active) break;
    }
    
    // All samples reached leaves - extract values
    // Negative node index indicates leaf: leaf_index = ~node
#ifdef __AVX512F__
    if (batch_size == kAVX512BatchSize) {
      alignas(64) double results[kAVX512BatchSize];
      for (int i = 0; i < kAVX512BatchSize; ++i) {
        results[i] = leaf_value[~nodes[i]];
      }
      _mm512_storeu_pd(&output[row_start], _mm512_load_pd(results));
    } else {
      for (int i = 0; i < batch_size; ++i) {
        output[row_start + i] = leaf_value[~nodes[i]];
      }
    }
#else
    for (int i = 0; i < batch_size; ++i) {
      output[row_start + i] = leaf_value[~nodes[i]];
    }
#endif
  }
}

}  // namespace LightGBM

#endif  // LIGHTGBM_TREE_AVX512_HPP_
