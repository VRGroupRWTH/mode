#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct lobatto_IIID_2
{
  using type = type_;
};
template <typename type_ = double>
struct lobatto_IIID_4
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v <lobatto_IIID_2<type>> = std::array
{
  type( 0.5), type(0.5),
  type(-0.5), type(0.5),
};
template <typename type>
__constant__ constexpr auto b_v <lobatto_IIID_2<type>> = std::array
{
  type(0.5), type(0.5)
};
template <typename type>
__constant__ constexpr auto c_v <lobatto_IIID_2<type>> = std::array
{
  type(0.0), type(1.0)
};

template <typename type>
__constant__ constexpr auto a_v <lobatto_IIID_4<type>> = std::array
{
  type(1.0 /  6.0), type(       0.0), type(-1.0 / 6.0),
  type(1.0 / 12.0), type(5.0 / 12.0), type(       0.0),
  type(1.0 /  2.0), type(1.0 /  3.0), type( 1.0 / 6.0)
};
template <typename type>
__constant__ constexpr auto b_v <lobatto_IIID_4<type>> = std::array
{
  type(1.0 / 6.0), type(2.0 / 3.0), type(1.0 / 6.0)
};
template <typename type>
__constant__ constexpr auto c_v <lobatto_IIID_4<type>> = std::array
{
  type(0.0), type(0.5), type(1.0)
};
}