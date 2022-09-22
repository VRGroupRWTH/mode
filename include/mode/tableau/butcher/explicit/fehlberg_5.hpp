#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct fehlberg_5
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto        a_v             <fehlberg_5<type>> = std::array
{
  type(   1.0 /    4.0),
  type(   3.0 /   32.0), type(    9.0 /   32.0),
  type(1932.0 / 2197.0), type(-7200.0 / 2197.0), type( 7296.0 / 2197.0),
  type( 439.0 /  216.0), type(-            8.0), type( 3680.0 /  513.0), type(- 845.0 / 4104.0),
  type(-  8.0 /   27.0), type(             2.0), type(-3544.0 / 2565.0), type( 1859.0 / 4104.0), type(-11.0 / 40.0)
};
template <typename type>
__constant__ constexpr auto        b_v             <fehlberg_5<type>> = std::array
{
  type(16.0 / 135.0), type(0.0), type(6656.0 / 12825.0), type(28561.0 / 56430.0), type(-9.0 / 50.0), type(2.0 / 55.0)
};
template <typename type>
__constant__ constexpr auto        bs_v            <fehlberg_5<type>> = std::array
{
  type(25.0 / 216.0), type(0.0), type(1408.0 /  2565.0), type( 2197.0 /  4104.0), type(-1.0 /  5.0), type(0.0)
};
template <typename type>
__constant__ constexpr auto        c_v             <fehlberg_5<type>> = std::array
{
  type(0.0), type(1.0 / 4.0), type(3.0 / 8.0), type(12.0 / 13.0), type(1.0), type(0.5)
};

template <typename type>
__constant__ constexpr std::size_t order_v         <fehlberg_5<type>> = 5;
template <typename type>
__constant__ constexpr std::size_t extended_order_v<fehlberg_5<type>> = 4;
}