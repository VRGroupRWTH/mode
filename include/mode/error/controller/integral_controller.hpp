#pragma once

#include <algorithm>
#include <cmath>
#include <complex>

#include <mode/cuda/decorators.hpp>
#include <mode/error/error_evaluation.hpp>
#include <mode/error/extended_result.hpp>
#include <mode/tableau/butcher_tableau.hpp>
#include <mode/quantity/quantity_operations.hpp>

namespace mode
{
template <typename type, typename tableau_type>
struct integral_controller
{
  // Reference: https://doi.org/10.1007/978-3-540-78862-1 Chapter: II.4, Section: Automatic Step Size Control, Equations: 4.11, 4.12, 4.13
  template <typename value_type>
  __device__ __host__ constexpr error_evaluation<type> evaluate(const value_type& value, const type step_size, const extended_result<value_type>& result)
  {
    type squared_sum(0);
    quantity_operations<value_type>::for_each([&] (const auto& p, const auto& r, const auto& e)
    {
      squared_sum += static_cast<type>(std::pow(std::abs(e) / (absolute_tolerance + relative_tolerance * std::max(std::abs(p), std::abs(r))), 2));
    }, value, result.value, result.error);

    type error   = std::sqrt(std::real(squared_sum) / quantity_operations<value_type>::size(value));
    type optimal = factor * std::pow (type(1) / error, ceschino_exponent);
    type limited = std::min (factor_maximum, std::max(factor_minimum, optimal));

    return {error <= type(1), step_size * limited};
  }
  
  const type            absolute_tolerance = type(1e-6);
  const type            relative_tolerance = type(1e-3);
  const type            factor             = type(0.8 );
  const type            factor_minimum     = type(1e-2);
  const type            factor_maximum     = type(1e+2);

  static constexpr type ceschino_exponent  = type(1) / (std::min(butcher_tableau::order_v<tableau_type>, butcher_tableau::extended_order_v<tableau_type>) + type(1));
};
}