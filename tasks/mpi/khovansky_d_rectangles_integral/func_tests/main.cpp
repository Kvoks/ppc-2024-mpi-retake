#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/khovansky_d_rectangles_integral/include/ops_mpi.hpp"

TEST(khovansky_d_rectangles_integral_mpi, test_integral_x) {
  boost::mpi::communicator world;
  std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}};
  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr = std::make_shared<std::function<double(std::vector<double>)>>(
      [](const std::vector<double> &args) -> double { return args[0]; });

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
    task_data_mpi->inputs_count.emplace_back(bounds.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  khovansky_d_rectangles_integral_mpi::RectanglesIntegralMpi rectangles_integral(task_data_mpi);
  ASSERT_EQ(rectangles_integral.ValidationImpl(), true);
  rectangles_integral.PreProcessingImpl();
  rectangles_integral.RunImpl();
  rectangles_integral.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_NEAR(result, 0.5, tolerance);
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_multidimensional_integral_xy) {
  boost::mpi::communicator world;
  std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}};
  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr = std::make_shared<std::function<double(std::vector<double>)>>(
      [](const std::vector<double> &args) -> double { return args[0] * args[1]; });

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
    task_data_mpi->inputs_count.emplace_back(bounds.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  khovansky_d_rectangles_integral_mpi::RectanglesIntegralMpi rectangles_integral(task_data_mpi);
  ASSERT_EQ(rectangles_integral.ValidationImpl(), true);
  rectangles_integral.PreProcessingImpl();
  rectangles_integral.RunImpl();
  rectangles_integral.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_NEAR(result, 0.25, tolerance);
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_multidimensional_integral_xyz) {
  boost::mpi::communicator world;
  std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr = std::make_shared<std::function<double(std::vector<double>)>>(
      [](const std::vector<double> &args) -> double { return args[0] * args[1] * args[2]; });

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
    task_data_mpi->inputs_count.emplace_back(bounds.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  khovansky_d_rectangles_integral_mpi::RectanglesIntegralMpi rectangles_integral(task_data_mpi);
  ASSERT_EQ(rectangles_integral.ValidationImpl(), true);
  rectangles_integral.PreProcessingImpl();
  rectangles_integral.RunImpl();
  rectangles_integral.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_NEAR(result, 1.0 / 8, tolerance);
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_integral_sum) {
  boost::mpi::communicator world;
  std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}};
  double tolerance = 1e-6;
  double result = 0.0;

  auto func_ptr = std::make_shared<std::function<double(std::vector<double>)>>(
      [](const std::vector<double> &args) -> double { return args[0] + args[1]; });

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
    task_data_mpi->inputs_count.emplace_back(bounds.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(func_ptr.get()));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  khovansky_d_rectangles_integral_mpi::RectanglesIntegralMpi rectangles_integral(task_data_mpi);
  ASSERT_EQ(rectangles_integral.ValidationImpl(), true);
  rectangles_integral.PreProcessingImpl();
  rectangles_integral.RunImpl();
  rectangles_integral.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_NEAR(result, 1.0, tolerance);
  }
}

TEST(khovansky_d_rectangles_integral_mpi, test_invalid_inputs) {
  boost::mpi::communicator world;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(0);
  }

  khovansky_d_rectangles_integral_mpi::RectanglesIntegralMpi rectangles_integral(task_data_mpi);
  ASSERT_EQ(rectangles_integral.ValidationImpl(), false);
}

TEST(khovansky_d_rectangles_integral_mpi, test_missing_inputs) {
  boost::mpi::communicator world;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(nullptr);
    task_data_mpi->inputs_count.emplace_back(1);
  }

  khovansky_d_rectangles_integral_mpi::RectanglesIntegralMpi rectangles_integral(task_data_mpi);
  ASSERT_EQ(rectangles_integral.ValidationImpl(), false);
}