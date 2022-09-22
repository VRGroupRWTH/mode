#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct merson_4
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto        a_v             <merson_4<type>> = std::array
{
  type(1.0 / 3.0),
  type(1.0 / 6.0), type(1.0 / 6.0),
  type(1.0 / 8.0), type(      0.0), type( 3.0 / 8.0),
  type(      0.5), type(      0.0), type(-3.0 / 2.0), type(2.0)
};
template <typename type>
__constant__ constexpr auto        b_v             <merson_4<type>> = std::array
{
  type(1.0 /  6.0), type(0.0), type(       0.0), type(2.0 / 3.0), type(1.0 / 6.0)
};
template <typename type>
__constant__ constexpr auto        bs_v            <merson_4<type>> = std::array
{
  type(1.0 / 10.0), type(0.0), type(3.0 / 10.0), type(2.0 / 5.0), type(1.0 / 5.0)
};
template <typename type>
__constant__ constexpr auto        c_v             <merson_4<type>> = std::array
{
  type(0.0), type(1.0 / 3.0), type(1.0 / 3.0), type(0.5), type(1.0)
};

template <typename type>
__constant__ constexpr std::size_t order_v         <merson_4<type>> = 4;
template <typename type>
__constant__ constexpr std::size_t extended_order_v<merson_4<type>> = 3;
}