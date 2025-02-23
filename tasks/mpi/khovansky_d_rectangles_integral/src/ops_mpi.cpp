#include "mpi/khovansky_d_rectangles_integral/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace khovansky_d_rectangles_integral_mpi {

bool RectanglesIntegralSeq::PreProcessingImpl() {
  auto* ptr = reinterpret_cast<std::pair<double, double>*>(task_data->inputs[0]);
  bounds_.assign(ptr, ptr + task_data->inputs_count[0]);
  tolerance_ = *reinterpret_cast<double*>(task_data->inputs[1]);

  auto* func_ptr = reinterpret_cast<std::function<double(std::vector<double>)>*>(task_data->inputs[2]);
  if (func_ptr != nullptr) {
    function_ = *func_ptr;
  } else {
    return false;
  }

  return true;
}

bool RectanglesIntegralSeq::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  return task_data->inputs_count[0] > 0 && task_data->inputs.size() == 3 && task_data->outputs_count[0] == 1;
}

bool RectanglesIntegralSeq::RunImpl() {
  size_t dim = bounds_.size();
  std::vector<int> partitions(dim, 2);
  std::vector<double> step(dim);
  std::vector<double> variables(dim);

  double integral = 0;
  double prev_integral = 0;

  do {
    prev_integral = integral;
    integral = 0;

    for (size_t i = 0; i < dim; i++) {
      step[i] = (bounds_[i].second - bounds_[i].first) / partitions[i];
    }

    std::vector<int> indices(dim, 0);
    bool done = false;
    while (!done) {
      for (size_t i = 0; i < dim; i++) {
        variables[i] = bounds_[i].first + (indices[i] + 0.5) * step[i];
      }

      integral += function_(variables);

      size_t idx = 0;
      while (idx < dim) {
        indices[idx]++;
        if (indices[idx] < partitions[idx]) {
          break;
        }
        indices[idx] = 0;
        idx++;
      }
      if (idx == dim) {
        done = true;
      }
    }

    double volume = 1.0;
    for (size_t i = 0; i < dim; i++) {
      volume *= step[i];
      partitions[i] *= 2;
    }

    integral *= volume;
  } while (std::abs(integral - prev_integral) > tolerance_);

  computed_result_ = integral;
  return true;
}

bool RectanglesIntegralSeq::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = computed_result_;
  return true;
}

bool RectanglesIntegralMpi::PreProcessingImpl() {
  auto* ptr = reinterpret_cast<std::pair<double, double>*>(task_data->inputs[0]);
  bounds_.assign(ptr, ptr + task_data->inputs_count[0]);
  tolerance_ = *reinterpret_cast<double*>(task_data->inputs[1]);
  auto* func_ptr = reinterpret_cast<std::function<double(std::vector<double>)>*>(task_data->inputs[2]);
  if (func_ptr != nullptr) {
    function_ = *func_ptr;
  } else {
    return false;
  }
  return true;
}

bool RectanglesIntegralMpi::ValidationImpl() {
  return (task_data && task_data->inputs_count[0] > 0 && task_data->inputs.size() == 3 &&
           task_data->outputs_count[0] == 1);
}

bool RectanglesIntegralMpi::RunImpl() {
  size_t dim = bounds_.size();
  std::vector<int> partitions(dim, 2);
  std::vector<double> step(dim);
  std::vector<double> variables(dim);

  double integral = 0;
  double prev_integral = 0;

  do {
    prev_integral = integral;
    integral = 0;

    for (size_t i = 0; i < dim; i++) {
      step[i] = (bounds_[i].second - bounds_[i].first) / partitions[i];
    }

    std::vector<int> indices(dim, 0);
    bool done = false;
    while (!done) {
      for (size_t i = 0; i < dim; i++) {
        variables[i] = bounds_[i].first + (indices[i] + 0.5) * step[i];
      }
      integral += function_(variables);

      size_t idx = 0;
      while (idx < dim) {
        indices[idx]++;
        if (indices[idx] < partitions[idx]) {
          break;
        }
        indices[idx] = 0;
        idx++;
      }
      if (idx == dim) {
        done = true;
      }
    }

    double volume = 1.0;
    for (size_t i = 0; i < dim; i++) {
      volume *= step[i];
      partitions[i] *= 2;
    }

    integral *= volume;
  } while (std::abs(integral - prev_integral) > tolerance_);

  boost::mpi::reduce(world_, integral, computed_result_, std::plus<>(), 0);
  return true;
}

bool RectanglesIntegralMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    *reinterpret_cast<double*>(task_data->outputs[0]) = computed_result_;
  }
  return true;
}

}  // namespace khovansky_d_rectangles_integral_mpi