#include "doctest/doctest.h"

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#include <Eigen/Core>
#include <mode/mode.hpp>

TEST_CASE("Lorenz")
{
  {
    using problem_type   = mode::initial_value_problem<float, Eigen::Vector3f>;
    using method_type    = mode::explicit_multi_stage_method<mode::butcher_tableau::runge_kutta_4<float>, problem_type>;

    const auto problem   = problem_type
    {
      0.0f,                                         /* t0 */
      Eigen::Vector3f(1.0f, 1.0f, 1.0f),            /* y0 */
      [&] (const float t, const Eigen::Vector3f& y) /* y' = f(t, y) */
      {
        constexpr auto sigma = 10.0f;
        constexpr auto rho   = 28.0f;
        constexpr auto beta  = 8.0f / 3.0f;
        return Eigen::Vector3f(sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]);
      }
    };

    auto iterator = mode::fixed_step_size_iterator<method_type>(problem, 0.01f /* h */);
    for (auto i = 0; i < 1000; ++i)
      ++iterator;
  }

  {
    using problem_type   = mode::initial_value_problem<float, Eigen::Vector3f>;
    using method_type    = mode::explicit_multi_step_method<mode::multi_step_tableau::adams_bashforth_3<float>, problem_type>;

    const auto problem   = problem_type
    {
      0.0f,                                         /* t0 */
      Eigen::Vector3f(1.0f, 1.0f, 1.0f),            /* y0 */
      [&] (const float t, const Eigen::Vector3f& y) /* y' = f(t, y) */
      {
        constexpr auto sigma = 10.0f;
        constexpr auto rho   = 28.0f;
        constexpr auto beta  = 8.0f / 3.0f;
        return Eigen::Vector3f(sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]);
      }
    };

    auto iterator = mode::fixed_step_size_iterator<method_type>(problem, 0.01f /* h */);
    for (auto i = 0; i < 1000; ++i)
      ++iterator;
  }
}