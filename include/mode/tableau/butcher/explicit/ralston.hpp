#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct ralston_2
{
  using type = type_;
};
template <typename type_ = double>
struct ralston_3
{
  using type = type_;
};
template <typename type_ = double>
struct ralston_4
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<ralston_2<type>> = std::array
{
  type(2.0 / 3.0)
};
template <typename type>
__constant__ constexpr auto b_v<ralston_2<type>> = std::array
{
  type(0.25), 
  type(0.75)
};
template <typename type>
__constant__ constexpr auto c_v<ralston_2<type>> = std::array
{
  type(0.0), 
  type(2.0 / 3.0)
};

template <typename type>
__constant__ constexpr auto a_v<ralston_3<type>> = std::array
{
  type(0.5),
  type(0.0), type(0.75)
};
template <typename type>
__constant__ constexpr auto b_v<ralston_3<type>> = std::array
{
  type(2.0 / 9.0), 
  type(1.0 / 3.0), 
  type(4.0 / 9.0)
};
template <typename type>
__constant__ constexpr auto c_v<ralston_3<type>> = std::array
{
  type(0.0 ), 
  type(0.5 ), 
  type(0.75)
};

template <typename type>
__constant__ constexpr auto a_v<ralston_4<type>> = std::array
{
  type(0.4),
  type(0.29697761), type( 0.15875964),
  type(0.21810040), type(-3.05096516), type(3.83286476)
};
template <typename type>
__constant__ constexpr auto b_v<ralston_4<type>> = std::array
{
  type( 0.17476028), 
  type(-0.55148066), 
  type( 1.20553560), 
  type( 0.17118478)
};
template <typename type>
__constant__ constexpr auto c_v<ralston_4<type>> = std::array
{
  type(0.0), 
  type(0.4), 
  type(0.45573725), 
  type(1.0)
};
}