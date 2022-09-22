#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct butcher_runge_kutta_6
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<butcher_runge_kutta_6<type>> = std::array
{
  type(0.25 ),
  type(0.125), type(0.125),
  type(0.0  ), type(0.0  ), type(0.5),
  type( 3.0 / 16.0), type(-3.0 / 8.0), type(3.0 / 8.0), type(  9.0 / 16.0),
  type(-3.0 /  7.0), type( 8.0 / 7.0), type(6.0 / 7.0), type(-12.0 /  7.0), type(8.0 / 7.0)
};
template <typename type>
__constant__ constexpr auto b_v<butcher_runge_kutta_6<type>> = std::array
{
  type(7.0 / 90.0), type(0.0), type(16.0 / 45.0), type(2.0 / 15.0), type(16.0 / 45.0), type(7.0 / 90.0)
};
template <typename type>
__constant__ constexpr auto c_v<butcher_runge_kutta_6<type>> = std::array
{
  type(0.0), type(0.25), type(0.25), type(0.5), type(0.75), type(1.0)
};
}