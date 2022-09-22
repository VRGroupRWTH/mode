#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct runge_kutta_4
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<runge_kutta_4<type>> = std::array
{
  type(0.5),
  type(0.0), type(0.5),
  type(0.0), type(0.0), type(1.0)
};
template <typename type>
__constant__ constexpr auto b_v<runge_kutta_4<type>> = std::array
{
  type(1.0 / 6.0), type(1.0 / 3.0), type(1.0 / 3.0), type(1.0 / 6.0)
};
template <typename type>
__constant__ constexpr auto c_v<runge_kutta_4<type>> = std::array
{
  type(0.0), type(0.5), type(0.5), type(1.0)
};
}