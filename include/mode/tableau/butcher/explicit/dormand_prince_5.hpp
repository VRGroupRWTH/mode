#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct dormand_prince_5
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto        a_v             <dormand_prince_5<type>> = std::array 
{
  type(    1.0 /    5.0),
  type(    3.0 /   40.0), type(     9.0 /   40.0),
  type(   44.0 /   45.0), type(-   56.0 /   15.0), type(   32.0 /    9.0),
  type(19372.0 / 6561.0), type(-25360.0 / 2187.0), type(64448.0 / 6561.0), type(-212.0 / 729.0),
  type( 9017.0 / 3168.0), type(-  355.0 /   33.0), type(46732.0 / 5247.0), type(  49.0 / 176.0), type(- 5103.0 / 18656.0),
  type(   35.0 /  384.0), type(              0.0), type(  500.0 / 1113.0), type( 125.0 / 192.0), type(- 2187.0 /  6784.0), type(11.0 / 84.0)
};
template <typename type>
__constant__ constexpr auto        b_v             <dormand_prince_5<type>> = std::array 
{
  type(  35.0 /   384.0), type(0.0), type( 500.0 /  1113.0), type(125.0 / 192.0), type(- 2187.0 /   6784.0), type( 11.0 /   84.0), type(0.0)
};
template <typename type>
__constant__ constexpr auto        bs_v            <dormand_prince_5<type>> = std::array 
{
  type(5179.0 / 57600.0), type(0.0), type(7571.0 / 16695.0), type(393.0 / 640.0), type(-92097.0 / 339200.0), type(187.0 / 2100.0), type(1.0 / 40.0)
};
template <typename type>
__constant__ constexpr auto        c_v             <dormand_prince_5<type>> = std::array 
{
  type(0.0), type(1.0 / 5.0), type(3.0 / 10.0), type(4.0 / 5.0), type(8.0 / 9.0), type(1.0), type(1.0)
};

template <typename type>
__constant__ constexpr std::size_t order_v         <dormand_prince_5<type>> = 5;
template <typename type>
__constant__ constexpr std::size_t extended_order_v<dormand_prince_5<type>> = 4;
}