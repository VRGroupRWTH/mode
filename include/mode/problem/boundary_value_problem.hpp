#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>

#include <mode/concepts/problem.hpp>
#include <mode/cuda/function.hpp>
#include <mode/utility/parameter_pack_expander.hpp>

namespace mode
{
template <typename time_type , typename value_type , std::size_t order = 1, std::size_t points = 2, typename       sequence = std::make_index_sequence<order>>
struct boundary_value_problem;
template <typename time_type_, typename value_type_, std::size_t order    , std::size_t points    , std::size_t... sequence>
struct boundary_value_problem<time_type_, value_type_, order, points, std::index_sequence<sequence...>>
{
  using time_type            = time_type_ ;
  using value_type           = value_type_;
  using time_container_type  = std::array<time_type, points>;
  using value_container_type = std::conditional_t<order == 1, std::array<value_type, points>, std::array<std::array<std::optional<value_type>, order>, points>>;
  using function_type        = function<value_type(time_type, const parameter_pack_expander<value_type, sequence>&...)>;

  // TODO: to_system_of_first_order()

  time_container_type  time     {};
  value_container_type value    {};
  function_type        function {};
};

// Satisfy the problem concept.
template <typename time_type , typename value_type , std::size_t order, std::size_t points, typename sequence>
struct is_problem<boundary_value_problem<time_type, value_type, order, points, sequence>> : std::true_type {};
}