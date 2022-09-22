#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct forward_euler
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<forward_euler<type>> = std::array {type(0.0)};
template <typename type>
__constant__ constexpr auto b_v<forward_euler<type>> = std::array {type(1.0)};
template <typename type>
__constant__ constexpr auto c_v<forward_euler<type>> = std::array {type(0.0)};
}