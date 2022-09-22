#pragma once

#include <concepts>

#include <mode/error/error_evaluation.hpp>
#include <mode/error/extended_result.hpp>

namespace mode
{
template <typename type>
concept error_controller = requires(type value)
{
  { 
    value.evaluate(
      typename type::problem_type           (), 
      typename type::problem_type::time_type(), 
      extended_result<typename type::problem_type::value_type>())
  } -> std::same_as<error_evaluation<typename type::problem_type::time_type>>;
};
}