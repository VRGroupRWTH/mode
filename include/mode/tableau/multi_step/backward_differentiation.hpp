#pragma once

#include <mode/tableau/multi_step_tableau.hpp>

namespace mode::multi_step_tableau
{
template <typename type_ = double>
struct bdf_2
{
  using type = type_;
};
template <typename type_ = double>
struct bdf_3
{
  using type = type_;
};
template <typename type_ = double>
struct bdf_4
{
  using type = type_;
};
template <typename type_ = double>
struct bdf_5
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<bdf_2<type>> = std::array
{
  type(       1.0),
  type(-4.0 / 3.0),
  type( 1.0 / 3.0)
};
template <typename type>
__constant__ constexpr auto b_v<bdf_2<type>> = std::array
{
  type(2.0 / 3.0),
  type(      0.0),
  type(      0.0)
};

template <typename type>
__constant__ constexpr auto a_v<bdf_3<type>> = std::array
{
  type(         1.0),
  type(-18.0 / 11.0),
  type(  9.0 / 11.0),
  type(- 2.0 / 11.0)
};
template <typename type>
__constant__ constexpr auto b_v<bdf_3<type>> = std::array
{
  type(6.0 / 11.0),
  type(       0.0),
  type(       0.0),
  type(       0.0)
};

template <typename type>
__constant__ constexpr auto a_v<bdf_4<type>> = std::array
{
  type(         1.0),
  type(-48.0 / 25.0),
  type( 36.0 / 25.0),
  type(-16.0 / 25.0),
  type(  3.0 / 25.0)
};
template <typename type>
__constant__ constexpr auto b_v<bdf_4<type>> = std::array
{
  type(12.0 / 25.0),
  type(        0.0),
  type(        0.0),
  type(        0.0),
  type(        0.0)
};

template <typename type>
__constant__ constexpr auto a_v<bdf_5<type>> = std::array
{
  type(           1.0),
  type(-300.0 / 137.0),
  type( 300.0 / 137.0),
  type(-200.0 / 137.0),
  type(  75.0 / 137.0),
  type(- 12.0 / 137.0)
};
template <typename type>
__constant__ constexpr auto b_v<bdf_5<type>> = std::array
{
  type(60.0 / 137.0),
  type(         0.0),
  type(         0.0),
  type(         0.0),
  type(         0.0),
  type(         0.0)
};
}