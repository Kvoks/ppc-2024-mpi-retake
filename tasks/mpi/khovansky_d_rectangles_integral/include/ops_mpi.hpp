#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include "core/task/include/task.hpp"

namespace khovansky_d_rectangles_integral_mpi {

class RectanglesIntegralSeq : public ppc::core::Task {
 public:
  explicit RectanglesIntegralSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
 private:
  double tolerance_;
  double computed_result_{};
  std::vector<std::pair<double, double>> bounds_;
  std::function<double(std::vector<double>)> function_;
};

class RectanglesIntegralMpi : public ppc::core::Task {
 public:
  explicit RectanglesIntegralMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)){}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
 private:
  boost::mpi::communicator world_;
  double tolerance_;
  double computed_result_{};
  std::vector<std::pair<double, double>> bounds_;
  std::function<double(std::vector<double>)> function_;
};

}  // namespace khovansky_d_rectangles_integral_mpi