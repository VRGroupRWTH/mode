#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <mode/concepts/problem.hpp>
#include <mode/cuda/decorators.hpp>
#include <mode/cuda/function.hpp>
#include <mode/quantity/jacobian.hpp>
#include <mode/utility/constexpr_for.hpp>
#include <mode/utility/parameter_pack_expander.hpp>

namespace mode
{
template <typename time_type , typename value_type , std::size_t order = 1, typename = std::make_index_sequence<order>>
struct initial_value_problem;
template <typename time_type_, typename value_type_, std::size_t order, std::size_t... sequence>
struct initial_value_problem<time_type_, value_type_, order, std::index_sequence<sequence...>>
{
  using time_type               = time_type_ ;
  using value_type              = value_type_;
  using value_container_type    = std::conditional_t<order == 1, value_type, std::array<value_type, order>>;
  using function_type           = function<value_type               (time_type, const parameter_pack_expander<value_type, sequence>&...)>;
  using jacobian_type           = function<jacobian_of_t<value_type>(time_type, const parameter_pack_expander<value_type, sequence>&...)>;

  using first_order_system_type = std::array<initial_value_problem<time_type, value_type, 1>, order>;

  template <std::size_t index, std::size_t sequence_index>
  __host__ __device__ constexpr const value_type&       select                  (const value_type& val, const first_order_system_type& system)
  {
    if constexpr (index == sequence_index)
      return val;
    else
      return std::get<sequence_index>(system).value;
  }
  __host__ __device__ constexpr first_order_system_type to_system_of_first_order() const
  {
    first_order_system_type result;
    constexpr_for<0, order, 1>([&] (const auto i)
    {
      std::get<i.value>(result) = 
      {
        time, 
        std::get<i.value>(value), 
        [=, this, &result] (time_type t, const value_type& v)
        {
          // Example resolution for order 3:
          // return function(t, v              , result[1].value, result[2].value);
          // return function(t, result[0].value, v              , result[2].value);
          // return function(t, result[0].value, result[1].value, v              );
          return function(t, select<i.value, sequence>(v, result)...);
        },
        [=, this, &result] (time_type t, const value_type& v)
        {
          // Example resolution for order 3:
          // return jacobian(t, v              , result[1].value, result[2].value);
          // return jacobian(t, result[0].value, v              , result[2].value);
          // return jacobian(t, result[0].value, result[1].value, v              );
          return jacobian(t, select<i.value, sequence>(v, result)...);
        }
      };
    });
    return result;
  }

  time_type            time     {};
  value_container_type value    {};
  function_type        function {};
  jacobian_type        jacobian {};
};

template <typename time_type, typename value_type, std::size_t order, typename sequence>
struct is_problem<initial_value_problem<time_type, value_type, order, sequence>> : std::true_type {};
}