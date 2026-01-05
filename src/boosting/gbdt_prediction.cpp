/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/tree.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <unordered_map>
#include <vector>
#include <cstring>

#include "gbdt.h"

namespace LightGBM {

void GBDT::PredictRaw(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  int early_stop_round_counter = 0;
  // set zero
  std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);
  const int end_iteration_for_pred = start_iteration_for_pred_ + num_iteration_for_pred_;
  for (int i = start_iteration_for_pred_; i < end_iteration_for_pred; ++i) {
    // predict all the trees for one iteration
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      output[k] += models_[i * num_tree_per_iteration_ + k]->Predict(features);
    }
    // check early stopping
    ++early_stop_round_counter;
    if (early_stop->round_period == early_stop_round_counter) {
      if (early_stop->callback_function(output, num_tree_per_iteration_)) {
        return;
      }
      early_stop_round_counter = 0;
    }
  }
}

void GBDT::PredictRawByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  int early_stop_round_counter = 0;
  // set zero
  std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);
  const int end_iteration_for_pred = start_iteration_for_pred_ + num_iteration_for_pred_;
  for (int i = start_iteration_for_pred_; i < end_iteration_for_pred; ++i) {
    // predict all the trees for one iteration
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      output[k] += models_[i * num_tree_per_iteration_ + k]->PredictByMap(features);
    }
    // check early stopping
    ++early_stop_round_counter;
    if (early_stop->round_period == early_stop_round_counter) {
      if (early_stop->callback_function(output, num_tree_per_iteration_)) {
        return;
      }
      early_stop_round_counter = 0;
    }
  }
}

void GBDT::PredictRawBatch(const double* features, int nrow, int ncol, double* output) const {
  // Initialize output to zero
  std::memset(output, 0, sizeof(double) * nrow);
  
  // Use all trees (don't rely on member variables which may not be initialized for batch path)
  const int num_trees = static_cast<int>(models_.size());
  if (num_trees == 0) return;
  
  const int num_threads = OMP_NUM_THREADS();
  
  // Each thread processes a chunk of rows through ALL trees
  // This keeps tree data cache-hot while processing batches of rows
  #pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
    const int chunk_size = (nrow + num_threads - 1) / num_threads;
    const int start_row = tid * chunk_size;
    const int end_row = std::min(start_row + chunk_size, nrow);
    const int my_nrow = end_row - start_row;
    
    if (my_nrow > 0) {
      // Create row pointers for this thread's chunk
      std::vector<const double*> row_ptrs(my_nrow);
      for (int i = 0; i < my_nrow; ++i) {
        row_ptrs[i] = features + (start_row + i) * ncol;
      }
      
      // Temporary buffer for per-tree results
      std::vector<double> tree_results(my_nrow);
      
      // Process all trees, using batch prediction for each
      for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        // Batch predict this tree for all rows in this thread's chunk
        models_[tree_idx]->PredictBatch(
            row_ptrs.data(), tree_results.data(), my_nrow);
        
        // Accumulate results
        for (int r = 0; r < my_nrow; ++r) {
          output[start_row + r] += tree_results[r];
        }
      }
    }
  }
}

void GBDT::Predict(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  PredictRaw(features, output, early_stop);
  if (average_output_) {
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      output[k] /= num_iteration_for_pred_;
    }
  }
  if (objective_function_ != nullptr) {
    objective_function_->ConvertOutput(output, output);
  }
}

void GBDT::PredictByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  PredictRawByMap(features, output, early_stop);
  if (average_output_) {
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      output[k] /= num_iteration_for_pred_;
    }
  }
  if (objective_function_ != nullptr) {
    objective_function_->ConvertOutput(output, output);
  }
}

void GBDT::PredictLeafIndex(const double* features, double* output) const {
  int start_tree = start_iteration_for_pred_ * num_tree_per_iteration_;
  int num_trees = num_iteration_for_pred_ * num_tree_per_iteration_;
  const auto* models_ptr = models_.data() + start_tree;
  for (int i = 0; i < num_trees; ++i) {
    output[i] = models_ptr[i]->PredictLeafIndex(features);
  }
}

void GBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const {
  int start_tree = start_iteration_for_pred_ * num_tree_per_iteration_;
  int num_trees = num_iteration_for_pred_ * num_tree_per_iteration_;
  const auto* models_ptr = models_.data() + start_tree;
  for (int i = 0; i < num_trees; ++i) {
    output[i] = models_ptr[i]->PredictLeafIndexByMap(features);
  }
}

}  // namespace LightGBM
