#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/khovansky_d_rectangles_integral/include/ops_seq.hpp"

TEST(khovansky_d_rectangles_integral_seq, test_integral_x) {
  std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}};
  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr = std::make_shared<std::function<double(std::vector<double>)>>(
      [](const std::vector<double> &args) -> double { return args[0]; });

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data_seq->inputs_count.emplace_back(bounds.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  task_data_seq->outputs_count.emplace_back(1);

  khovansky_d_rectangles_integral_seq::RectanglesIntegralSeq integral_task(task_data_seq);
  ASSERT_EQ(integral_task.ValidationImpl(), true);
  integral_task.PreProcessingImpl();
  integral_task.RunImpl();
  integral_task.PostProcessingImpl();

  ASSERT_NEAR(result, 0.5, tolerance);
}

TEST(khovansky_d_rectangles_integral_seq, test_multidimensional_integral_xy) {
  std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}};
  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr = std::make_shared<std::function<double(std::vector<double>)>>(
      [](const std::vector<double> &args) -> double { return args[0] * args[1]; });

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data_seq->inputs_count.emplace_back(bounds.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  task_data_seq->outputs_count.emplace_back(1);

  khovansky_d_rectangles_integral_seq::RectanglesIntegralSeq integral_task(task_data_seq);
  ASSERT_EQ(integral_task.ValidationImpl(), true);
  integral_task.PreProcessingImpl();
  integral_task.RunImpl();
  integral_task.PostProcessingImpl();

  ASSERT_NEAR(result, 0.25, tolerance);
}

TEST(khovansky_d_rectangles_integral_seq, test_multidimensional_integral_xyz) {
  std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr = std::make_shared<std::function<double(std::vector<double>)>>(
      [](const std::vector<double> &args) -> double { return args[0] * args[1] * args[2]; });

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data_seq->inputs_count.emplace_back(bounds.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  task_data_seq->outputs_count.emplace_back(1);

  khovansky_d_rectangles_integral_seq::RectanglesIntegralSeq integral_task(task_data_seq);
  ASSERT_EQ(integral_task.ValidationImpl(), true);
  integral_task.PreProcessingImpl();
  integral_task.RunImpl();
  integral_task.PostProcessingImpl();

  ASSERT_NEAR(result, 1.0 / 8, tolerance);
}

TEST(khovansky_d_rectangles_integral_seq, test_symmetric_function) {
  std::vector<std::pair<double, double>> bounds = {{-1.0, 1.0}};
  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr = std::make_shared<std::function<double(std::vector<double>)>>(
      [](const std::vector<double> &args) -> double { return args[0] * args[0]; });

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data_seq->inputs_count.emplace_back(bounds.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  task_data_seq->outputs_count.emplace_back(1);

  khovansky_d_rectangles_integral_seq::RectanglesIntegralSeq integral_task(task_data_seq);
  ASSERT_EQ(integral_task.ValidationImpl(), true);
  integral_task.PreProcessingImpl();
  integral_task.RunImpl();
  integral_task.PostProcessingImpl();

  ASSERT_NEAR(result, 2.0 / 3, tolerance);
}

TEST(khovansky_d_rectangles_integral_seq, test_integral_sum) {
  std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}};
  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr = std::make_shared<std::function<double(std::vector<double>)>>(
      [](const std::vector<double> &args) -> double { return args[0] + args[1]; });

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data_seq->inputs_count.emplace_back(bounds.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  task_data_seq->outputs_count.emplace_back(1);

  khovansky_d_rectangles_integral_seq::RectanglesIntegralSeq rectangles_integral(task_data_seq);
  ASSERT_EQ(rectangles_integral.ValidationImpl(), true);
  rectangles_integral.PreProcessingImpl();
  rectangles_integral.RunImpl();
  rectangles_integral.PostProcessingImpl();

  ASSERT_NEAR(result, 1.0, tolerance);
}