#pragma once

#include <mode/concepts/problem.hpp>
#include <mode/concepts/root_finder.hpp>
#include <mode/cuda/decorators.hpp>
#include <mode/root_finder/newton_root_finder.hpp>
#include <mode/tableau/multi_step_tableau.hpp>
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

  __device__ __host__ static constexpr auto apply(const problem_type& problem, const typename problem_type::time_type step_size) {}
};
}