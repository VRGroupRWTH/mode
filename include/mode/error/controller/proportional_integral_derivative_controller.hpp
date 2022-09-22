#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>

#include <mode/cuda/decorators.hpp>
#include <mode/cuda/function.hpp>
#include <mode/error/error_evaluation.hpp>
#include <mode/error/extended_result.hpp>
#include <mode/tableau/butcher_tableau.hpp>
#include <mode/quantity/quantity_operations.hpp>

namespace mode
{
template <typename type, typename tableau_type>
struct proportional_integral_derivative_controller
{
  // Reference: https://arxiv.org/pdf/2104.06836.pdf Section: 2.2 Error-Based Step Size Control, Equation: 2.6
  template <typename value_type>
  __device__ __host__ constexpr error_evaluation<type> evaluate(const value_type& value, const type step_size, const extended_result<value_type>& result)
  {
    type squared_sum(0);
    quantity_operations<value_type>::for_each([&] (const auto& p, const auto& r, const auto& e)
    {
      squared_sum += static_cast<type>(std::pow(std::abs(e) / (absolute_tolerance + relative_tolerance * std::max(std::abs(p), std::abs(r))), 2));
    }, value, result.value, result.error);

    error[0]     = type(1) / std::sqrt(std::real(squared_sum) / quantity_operations<value_type>::size(value));
    type optimal = std::pow(error[0], beta[0] / ceschino_exponent) * 
                   std::pow(error[1], beta[1] / ceschino_exponent) * 
                   std::pow(error[2], beta[2] / ceschino_exponent);
    type limited = limiter(optimal);

    const bool accept = limited >= accept_safety;
    if (accept)
      std::rotate(error.rbegin(), error.rbegin() + 1, error.rend());

    return {accept, step_size * limited};
  }
  
  const type                 absolute_tolerance = type(1e-6);
  const type                 relative_tolerance = type(1e-3);
  const type                 accept_safety      = type(0.81);
  const function<type(type)> limiter            = [ ] __device__ (type value) { return type(1) + std::atan(value - type(1)); };
  const std::array<type, 3>  beta               = { type(1)   , type(0)   , type(0)    };
  std::array<type, 3>        error              = { type(1e-3), type(1e-3), type(1e-3) };
  
  static constexpr type      ceschino_exponent  = type(1) / (std::min(butcher_tableau::order_v<tableau_type>, butcher_tableau::extended_order_v<tableau_type>) + type(1));
};
}