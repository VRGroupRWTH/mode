### MODE
MODE: A modern ordinary differential equation solver for C++ and CUDA.

### Requirements
- CMake 3.12+

### Getting Started
- Clone the repository.
- Run `bootstrap.[bat|sh]`. This will install Doctest and Eigen, and build the project under the `./build` directory.
- Optional: Run cmake on the `./build` directory.
  - Toggle `MODE_BUILD_TESTS` to build the tests.
  - Toggle `MODE_USE_EIGEN` to enable implicit method support.
  - Remember to generate or run `bootstrap.[bat|sh]` after changes. You can ignore the cmake developer errors as long as generation is successful.
- Alternative:
  - MODE is header-only, so you can copy the headers into your project rather than using the cmake build process.

### Example usage: Lorenz system
```cpp
#include <mode/mode.hpp>

using problem_type   = mode::initial_value_problem<float, Eigen::Vector3f>;
using method_type    = mode::explicit_multi_stage_method<mode::butcher_tableau::runge_kutta_4<float>, problem_type>;

std::int32_t main(std::int32_t argc, char** argv)
{
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

  return 0;
}
```
