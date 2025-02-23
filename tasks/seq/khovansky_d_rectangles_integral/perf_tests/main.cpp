#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/khovansky_d_rectangles_integral/include/ops_seq.hpp"

TEST(khovansky_d_rectangles_integral_seq, test_pipeline_run) {
  constexpr int kNumRuns = 8000;

  std::vector<std::pair<double, double>> bounds = {
      {-100000000.0, 100000000.0}, {-100000000.0, 100000000.0}, {-100000000.0, 100000000.0}};

  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr =
      std::make_shared<std::function<double(std::vector<double>)>>([](const std::vector<double>& args) -> double {
        double product = 1.0;
        for (int i = 0; i < 9; ++i) {
          product *= args[0] * args[1] * args[2];
          product += std::sin(args[0] * args[1]) * std::cos(args[2] + 1);
        }
        return product;
      });

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data_seq->inputs_count.emplace_back(bounds.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  task_data_seq->outputs_count.emplace_back(1);

  auto test_task_sequential =
      std::make_shared<khovansky_d_rectangles_integral_seq::RectanglesIntegralSeq>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = kNumRuns;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_NEAR(result, 0.0, tolerance);
}

TEST(khovansky_d_rectangles_integral_seq, test_task_run) {
  constexpr int kNumRuns = 3000000;

  std::vector<std::pair<double, double>> bounds = {
    {-100000000.0, 100000000.0}, {-100000000.0, 100000000.0}, {-100000000.0, 100000000.0}};

  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr =
      std::make_shared<std::function<double(std::vector<double>)>>([](const std::vector<double>& args) -> double {
        double product = 1.0;
        for (int i = 0; i < 9; ++i) {
          product *= args[0] * args[1] * args[2];
          product += std::sin(args[0] * args[1]) * std::cos(args[2] + 1);
        }
        return product;
      });

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data_seq->inputs_count.emplace_back(bounds.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  task_data_seq->outputs_count.emplace_back(1);

  auto test_task_sequential =
      std::make_shared<khovansky_d_rectangles_integral_seq::RectanglesIntegralSeq>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = kNumRuns;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_NEAR(result, 0.0, tolerance);
}