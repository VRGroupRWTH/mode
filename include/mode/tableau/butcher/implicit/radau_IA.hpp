#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct radau_IA_3
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<radau_IA_3<type>> = std::array
{
  type(1.0 / 4.0), type(-1.0 /  4.0),
  type(1.0 / 4.0), type( 5.0 / 12.0),
};
template <typename type>
__constant__ constexpr auto b_v<radau_IA_3<type>> = std::array
{
  type(1.0 / 4.0), type(3.0 / 4.0)
};
template <typename type>
__constant__ constexpr auto c_v<radau_IA_3<type>> = std::array
{
  type(0.0), type(2.0 / 3.0)
};
}