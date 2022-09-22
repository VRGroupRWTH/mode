#pragma once

#include <array>
#include <utility>

#include <mode/concepts/problem.hpp>
#include <mode/concepts/root_finder.hpp>
#include <mode/cuda/decorators.hpp>
#include <mode/error/extended_result.hpp>
#include <mode/root_finder/newton_root_finder.hpp>
#include <mode/tableau/butcher_tableau.hpp>
#include <mode/utility/constexpr_for.hpp>

namespace mode
{
template <typename tableau_type_, problem problem_type_, root_finder root_finder_ = newton_root_finder<problem_type_>>
class implicit_multi_stage_method
{
public:
  using tableau_type = tableau_type_;
  using problem_type = problem_type_;
  using value_type   = typename problem_type::function_type::result_type;

  __device__ __host__ static constexpr auto apply(const problem_type& problem, const typename problem_type::time_type step_size)
  {
    std::array<value_type, butcher_tableau::stages_v<tableau_type>> z;
    std::array<value_type, butcher_tableau::stages_v<tableau_type>> stages;
    constexpr_for<0, butcher_tableau::stages_v<tableau_type>, 1>([&problem, &step_size, &stages] (auto i)
    {
      std::get<i.value>(stages) = {};
    });

    if constexpr (butcher_tableau::is_extended_v<tableau_type>)
    {
      value_type higher, lower;
      constexpr_for<0, butcher_tableau::stages_v<tableau_type>, 1>([&stages, &higher, &lower] (auto i)
      {
        higher += std::get<i.value>(stages) * std::get<i.value>(butcher_tableau::b_v <tableau_type>);
        lower  += std::get<i.value>(stages) * std::get<i.value>(butcher_tableau::bs_v<tableau_type>);
      });
      return extended_result<value_type> {problem.value + higher * step_size, (higher - lower) * step_size};
    }
    else
    {
      value_type sum;
      constexpr_for<0, butcher_tableau::stages_v<tableau_type>, 1>([&stages, &sum] (auto i)
      {
        sum += std::get<i.value>(stages) * std::get<i.value>(butcher_tableau::b_v<tableau_type>);
      });
      return value_type(problem.value + sum * step_size);
    }
  }
};
}