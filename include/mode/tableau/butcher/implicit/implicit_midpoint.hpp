#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct implicit_midpoint
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<implicit_midpoint<type>> = std::array {type(0.5)};
template <typename type>
__constant__ constexpr auto b_v<implicit_midpoint<type>> = std::array {type(1.0)};
template <typename type>
__constant__ constexpr auto c_v<implicit_midpoint<type>> = std::array {type(0.5)};
}