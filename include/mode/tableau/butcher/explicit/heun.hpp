#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct heun_2
{
  using type = type_;
};
template <typename type_ = double>
struct heun_3
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<heun_2<type>> = std::array 
{
  type(1.0)
};
template <typename type>
__constant__ constexpr auto b_v<heun_2<type>> = std::array 
{
  type(0.5), 
  type(0.5)
};
template <typename type>
__constant__ constexpr auto c_v<heun_2<type>> = std::array 
{
  type(0.0), 
  type(1.0)
};

template <typename type>
__constant__ constexpr auto a_v<heun_3<type>> = std::array 
{
  type(1.0 / 3.0),
  type(      0.0), type(2.0 / 3.0)
};
template <typename type>
__constant__ constexpr auto b_v<heun_3<type>> = std::array 
{
  type(0.25), 
  type(0.0 ), 
  type(0.75)
};
template <typename type>
__constant__ constexpr auto c_v<heun_3<type>> = std::array 
{
  type(      0.0), 
  type(1.0 / 3.0), 
  type(2.0 / 3.0)
};
}