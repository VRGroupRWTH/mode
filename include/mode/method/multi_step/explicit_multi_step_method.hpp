#pragma once

#include <array>
#include <utility>

#include <mode/concepts/method.hpp>
#include <mode/concepts/problem.hpp>
#include <mode/cuda/decorators.hpp>
#include <mode/iterator/fixed_step_size_iterator.hpp>
#include <mode/method/multi_stage/explicit_multi_stage_method.hpp>
#include <mode/tableau/butcher/explicit/runge_kutta_4.hpp>
#include <mode/tableau/multi_step_tableau.hpp>
#include <mode/utility/constexpr_for.hpp>

namespace mode
{
template <
  typename tableau_type_       , 
  problem  problem_type_       , 
  method   initializer_method_ = explicit_multi_stage_method<butcher_tableau::runge_kutta_4<typename tableau_type_::type>, problem_type_>>
class explicit_multi_step_method
{
public:
  using tableau_type     = tableau_type_;
  using problem_type     = problem_type_;
  using value_type       = typename problem_type::function_type::result_type;
  using initializer_type = fixed_step_size_iterator<initializer_method_, butcher_tableau::is_extended_v<typename initializer_method_::tableau_type>>;

  __device__ __host__ constexpr auto apply(const problem_type& problem, const typename problem_type::time_type step_size)
  {
    if (!initialized_)
    {
      initialized_ = true;
      initializer_type initializer(problem, -step_size);
      for (auto iterator = history_.begin(); iterator != history_.end(); ++iterator)
      {
        ++initializer;
        *iterator = problem.function(initializer->time, initializer->value);
      }
    }

    std::rotate(history_.rbegin(), history_.rbegin() + 1, history_.rend());
    std::get<0>(history_) = problem.function(problem.time, problem.value);

    value_type sum;
    constexpr_for<0, multi_step_tableau::steps_v<tableau_type>, 1>([&] (const auto i)
    {
      sum += std::get<i.value + 1>(multi_step_tableau::b_v<tableau_type>) * std::get<i.value>(history_);
    });

    return value_type(problem.value + sum * step_size);
  }

private:
  bool                                                              initialized_ {false};
  std::array<value_type, multi_step_tableau::steps_v<tableau_type>> history_     {};
};
}