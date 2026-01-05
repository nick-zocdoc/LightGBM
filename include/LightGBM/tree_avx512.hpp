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
#include <cstdio>
#include <algorithm>
#include <sys/stat.h>

namespace LightGBM {

// Batch size for AVX-512 processing (8 doubles fit in 512-bit register)
constexpr int kAVX512BatchSize = 8;

// Debug logging for tree traversal comparison
static bool g_debug_first_row = false;
static int g_tree_counter = 0;
static const char* g_debug_dir = "/tmp/lgbm_tree_paths";
static const char* g_debug_mode = "scalar";  // or "simd"

inline void EnableTreeDebug(const char* mode) {
  g_debug_first_row = true;
  g_tree_counter = 0;
  g_debug_mode = mode;
  mkdir(g_debug_dir, 0755);
  char subdir[256];
  snprintf(subdir, sizeof(subdir), "%s/%s", g_debug_dir, mode);
  mkdir(subdir, 0755);
}

inline void DisableTreeDebug() {
  g_debug_first_row = false;
}

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
  
  // Debug logging for first row
  FILE* dbg = nullptr;
  bool logging = g_debug_first_row && (g_tree_counter < 400);
  if (logging) {
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/%s/tree_%04d.txt", g_debug_dir, g_debug_mode, g_tree_counter);
    dbg = fopen(fname, "w");
    if (dbg) {
      fprintf(dbg, "Tree %d: num_leaves=%d num_cat=%d\n", g_tree_counter, num_leaves, num_cat);
    }
    g_tree_counter++;
  }
  
  // Handle degenerate case: single leaf
  if (num_leaves <= 1) {
    double val = leaf_value[0];
    if (dbg) {
      fprintf(dbg, "Single leaf tree, value=%.10f\n", val);
      fclose(dbg);
    }
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
    if (dbg) {
      fprintf(dbg, "Linear tree, skipping\n");
      fclose(dbg);
    }
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
          
          // Debug log for first row only
          if (dbg && row_start == 0 && i == 0) {
            fprintf(dbg, "  node=%d feat=%d fval=%.10f thresh=%.10f dtype=%d missing_type=%d",
                    node, split_feat, fval, threshold[node], dtype, missing_type);
          }
          
          if (std::isnan(fval)) {
            // NaN handling
            if (missing_type == 0) {  // MissingType::None
              fval = 0.0;
              if (dbg && row_start == 0 && i == 0) fprintf(dbg, " [NaN->0]");
            } else if (missing_type == 2) {  // MissingType::NaN
              // Go to default direction
              if (dtype & kDefaultLeftMask) {
                nodes[i] = left_child[node];
                if (dbg && row_start == 0 && i == 0) fprintf(dbg, " [NaN->left=%d]\n", nodes[i]);
              } else {
                nodes[i] = right_child[node];
                if (dbg && row_start == 0 && i == 0) fprintf(dbg, " [NaN->right=%d]\n", nodes[i]);
              }
              continue;
            }
          }
          
          // Check if zero should be treated as missing
          if (missing_type == 1 && fval >= -1e-35 && fval <= 1e-35) {  // MissingType::Zero
            if (dtype & kDefaultLeftMask) {
              nodes[i] = left_child[node];
              if (dbg && row_start == 0 && i == 0) fprintf(dbg, " [Zero->left=%d]\n", nodes[i]);
            } else {
              nodes[i] = right_child[node];
              if (dbg && row_start == 0 && i == 0) fprintf(dbg, " [Zero->right=%d]\n", nodes[i]);
            }
            continue;
          }
          
          // Check if categorical or numerical split
          if ((dtype & kCategoricalMask) && num_cat > 0) {
            // Categorical decision
            int int_fval = static_cast<int>(fval);
            if (std::isnan(fval) || int_fval < 0) {
              nodes[i] = right_child[node];
              if (dbg && row_start == 0 && i == 0) fprintf(dbg, " [Cat:invalid->right=%d]\n", nodes[i]);
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
              if (dbg && row_start == 0 && i == 0) {
                fprintf(dbg, " [Cat:val=%d cat_idx=%d found=%d -> %s=%d]\n",
                        int_fval, cat_idx, found, found ? "left" : "right", nodes[i]);
              }
            }
          } else {
            // Numerical decision
            bool go_left = (fval <= threshold[node]);
            if (go_left) {
              nodes[i] = left_child[node];
            } else {
              nodes[i] = right_child[node];
            }
            if (dbg && row_start == 0 && i == 0) {
              fprintf(dbg, " [Num: %.10f %s %.10f -> %s=%d]\n",
                      fval, go_left ? "<=" : ">", threshold[node],
                      go_left ? "left" : "right", nodes[i]);
            }
          }
        }
      }
      
      if (!any_active) break;
    }
    
    // All samples reached leaves - extract values
    // Negative node index indicates leaf: leaf_index = ~node
    if (dbg && row_start == 0) {
      int leaf_idx = ~nodes[0];
      fprintf(dbg, "LEAF: index=%d value=%.10f\n", leaf_idx, leaf_value[leaf_idx]);
      fclose(dbg);
      dbg = nullptr;
    }
    
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

#ifdef __AVX512F__
//=============================================================================
// TRUE AVX-512 SIMD TREE TRAVERSAL
// Processes 8 rows in parallel using vectorized comparisons
//=============================================================================

/*!
 * \brief SIMD tree prediction - processes 8 rows in parallel
 *        Uses AVX-512 for vectorized comparisons and child selection
 */
inline void PredictTreeBatchAVX512_SIMD(
    int num_leaves,
    int num_cat,
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
  
  // Debug logging for first row
  FILE* dbg = nullptr;
  bool logging = g_debug_first_row && (g_tree_counter < 400);
  if (logging) {
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/%s/tree_%04d.txt", g_debug_dir, g_debug_mode, g_tree_counter);
    dbg = fopen(fname, "w");
    if (dbg) {
      fprintf(dbg, "Tree %d: num_leaves=%d num_cat=%d\n", g_tree_counter, num_leaves, num_cat);
    }
    g_tree_counter++;
  }
  
  // Handle degenerate case
  if (num_leaves <= 1) {
    if (dbg) {
      fprintf(dbg, "Single leaf tree, value=%.10f\n", leaf_value[0]);
      fclose(dbg);
    }
    __m512d val_vec = _mm512_set1_pd(leaf_value[0]);
    int i = 0;
    for (; i + 8 <= num_rows; i += 8) {
      _mm512_storeu_pd(&output[i], val_vec);
    }
    for (; i < num_rows; ++i) {
      output[i] = leaf_value[0];
    }
    return;
  }
  
  constexpr int8_t kCategoricalMask = 1;
  constexpr int8_t kDefaultLeftMask = 2;
  constexpr int kMaxDepth = 64;
  
  // Process in batches of 8
  for (int row_start = 0; row_start < num_rows; row_start += 8) {
    int batch_size = std::min(8, num_rows - row_start);
    __mmask8 active = static_cast<__mmask8>((1 << batch_size) - 1);
    
    // All samples start at node 0
    alignas(32) int32_t nodes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    for (int depth = 0; depth < kMaxDepth && active != 0; ++depth) {
      // For each active lane, get current node's split feature and threshold
      alignas(64) double fvals[8] = {0};
      alignas(32) int32_t split_feats[8] = {0};
      alignas(32) int32_t dtypes[8] = {0};
      alignas(64) double thresholds[8] = {0};
      alignas(32) int32_t left_children[8] = {0};
      alignas(32) int32_t right_children[8] = {0};
      
      // Gather data for each lane (scalar - can't vectorize double-indirect)
      for (int i = 0; i < batch_size; ++i) {
        if ((active >> i) & 1) {
          int node = nodes[i];
          split_feats[i] = split_feature[node];
          fvals[i] = row_features[row_start + i][split_feats[i]];
          dtypes[i] = decision_type[node];
          thresholds[i] = threshold[node];
          left_children[i] = left_child[node];
          right_children[i] = right_child[node];
        }
      }
      
      uint8_t missing_type_0 = (dtypes[0] >> 2) & 3;
      if (dbg && row_start == 0) {
        fprintf(dbg, "  node=%d feat=%d fval=%.10f thresh=%.10f dtype=%d missing_type=%d",
                nodes[0], split_feats[0], fvals[0], thresholds[0], dtypes[0], missing_type_0);
      }
      
      // Load into SIMD registers
      __m512d fvals_vec = _mm512_load_pd(fvals);
      __m512d thresh_vec = _mm512_load_pd(thresholds);
      __m256i left_vec = _mm256_load_si256((__m256i*)left_children);
      __m256i right_vec = _mm256_load_si256((__m256i*)right_children);
      __m256i dtypes_vec = _mm256_load_si256((__m256i*)dtypes);
      
      // Check for NaN (unordered comparison)
      __mmask8 is_nan = _mm512_cmp_pd_mask(fvals_vec, fvals_vec, _CMP_UNORD_Q) & active;
      
      // Determine categorical vs numerical
      __m256i cat_test = _mm256_and_si256(dtypes_vec, _mm256_set1_epi32(kCategoricalMask));
      __mmask8 is_categorical = _mm256_cmpneq_epi32_mask(cat_test, _mm256_setzero_si256()) & active;
      __mmask8 is_numerical = active & ~is_categorical & ~is_nan;
      
      // Initialize go_left mask
      __mmask8 go_left = 0;
      
      // === NUMERICAL PATH (SIMD) ===
      if (is_numerical != 0) {
        // fval <= threshold → go left
        go_left |= _mm512_cmp_pd_mask(fvals_vec, thresh_vec, _CMP_LE_OQ) & is_numerical;
      }
      
      // === CATEGORICAL PATH (scalar for now) ===
      if (is_categorical != 0 && num_cat > 0) {
        for (int i = 0; i < batch_size; ++i) {
          if ((is_categorical >> i) & 1) {
            double fval = fvals[i];
            int int_fval = static_cast<int>(fval);
            if (std::isnan(fval) || int_fval < 0) {
              // go right (don't set go_left)
              if (dbg && row_start == 0 && i == 0) fprintf(dbg, " [Cat:invalid->right]");
            } else {
              int cat_idx = static_cast<int>(thresholds[i]);
              int start = cat_boundaries[cat_idx];
              int end = cat_boundaries[cat_idx + 1];
              int num_words = end - start;
              bool found = false;
              if (int_fval < num_words * 32) {
                int word_idx = int_fval / 32;
                int bit_idx = int_fval % 32;
                if (word_idx < num_words) {
                  found = (cat_threshold[start + word_idx] >> bit_idx) & 1;
                  if (found) go_left |= (1 << i);
                }
              }
              if (dbg && row_start == 0 && i == 0) {
                fprintf(dbg, " [Cat:val=%d cat_idx=%d found=%d -> %s]",
                        int_fval, cat_idx, found, found ? "left" : "right");
              }
            }
          }
        }
      }
      
      // === NaN PATH (scalar) ===
      if (is_nan != 0) {
        for (int i = 0; i < batch_size; ++i) {
          if ((is_nan >> i) & 1) {
            int8_t dtype = static_cast<int8_t>(dtypes[i]);
            if (dtype & kDefaultLeftMask) {
              go_left |= (1 << i);
              if (dbg && row_start == 0 && i == 0) fprintf(dbg, " [NaN->left]");
            } else {
              if (dbg && row_start == 0 && i == 0) fprintf(dbg, " [NaN->right]");
            }
          }
        }
      }
      
      // Log numerical decision for lane 0
      if (dbg && row_start == 0 && ((is_numerical >> 0) & 1)) {
        bool went_left = (go_left >> 0) & 1;
        fprintf(dbg, " [Num: %.10f %s %.10f -> %s]",
                fvals[0], went_left ? "<=" : ">", thresholds[0],
                went_left ? "left" : "right");
      }
      
      // Select next node: left if go_left, right otherwise
      // Only update ACTIVE lanes - inactive lanes keep their leaf values
      __m256i old_nodes = _mm256_load_si256((__m256i*)nodes);
      __m256i new_nodes = _mm256_mask_blend_epi32(go_left, right_vec, left_vec);
      // Merge: active lanes get new_nodes, inactive lanes keep old_nodes
      __m256i merged_nodes = _mm256_mask_blend_epi32(active, old_nodes, new_nodes);
      _mm256_store_si256((__m256i*)nodes, merged_nodes);
      
      if (dbg && row_start == 0) {
        fprintf(dbg, " -> node=%d\n", nodes[0]);
      }
      
      // Update active mask: node >= 0 means still at internal node
      active = 0;
      for (int i = 0; i < batch_size; ++i) {
        if (nodes[i] >= 0) {
          active |= (1 << i);
        }
      }
    }
    
    // Log final result for lane 0
    if (dbg && row_start == 0) {
      int leaf_idx = ~nodes[0];
      fprintf(dbg, "LEAF: index=%d value=%.10f\n", leaf_idx, leaf_value[leaf_idx]);
      fclose(dbg);
      dbg = nullptr;
    }
    
    // All samples reached leaves: leaf_index = ~node
    for (int i = 0; i < batch_size; ++i) {
      output[row_start + i] = leaf_value[~nodes[i]];
    }
  }
}

#endif  // __AVX512F__

}  // namespace LightGBM

#endif  // LIGHTGBM_TREE_AVX512_HPP_
