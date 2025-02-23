#pragma once

#include <functional>
#include "core/task/include/task.hpp"

namespace khovansky_d_rectangles_integral_seq {

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

}  // namespace khovansky_d_rectangles_integral_seq