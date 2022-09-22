#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct midpoint
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<midpoint<type>> = std::array {type(0.5)};
template <typename type>
__constant__ constexpr auto b_v<midpoint<type>> = std::array {type(0.0), type(1.0)};
template <typename type>
__constant__ constexpr auto c_v<midpoint<type>> = std::array {type(0.0), type(0.5)};
}