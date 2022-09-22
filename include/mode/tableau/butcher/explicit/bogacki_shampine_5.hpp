#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct bogacki_shampine_5
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto        a_v             <bogacki_shampine_5<type>> = std::array
{
  type(     1.0 /      6.0),
  type(     2.0 /     27.0), type(     4.0 /    27.0),
  type(   183.0 /   1372.0), type(-  162.0 /   343.0), type(   1053.0 /     1372.0),
  type(    68.0 /    297.0), type(-    4.0 /    11.0), type(     42.0 /      143.0), type(  1960.0 /    3861.0),
  type(   597.0 /  22528.0), type(    81.0 /   352.0), type(  63099.0 /   585728.0), type( 58653.0 /  366080.0), type(  4617.0 / 20480.0),
  type(174197.0 / 959244.0), type(-30942.0 / 79937.0), type(8152137.0 / 19744439.0), type(666106.0 / 1039181.0), type(-29421.0 / 29068.0), type(482048.0 / 414219.0),
  type(   587.0 /   8064.0), type(               0.0), type(4440339.0 / 15491840.0), type( 24353.0 /  124800.0), type(   387.0 / 44800.0), type(  2152.0 /   5985.0), type(7267.0 / 94080.0)
};
template <typename type>
__constant__ constexpr auto        b_v             <bogacki_shampine_5<type>> = std::array
{
  type( 587.0 /  8064.0), type(0.0), type(4440339.0 / 15491840.0), type( 24353.0 /  124800.0), type(387.0 / 44800.0), type(2152.0 / 5985.0), type( 7267.0 /   94080.0), type(0.0)
};
template <typename type>
__constant__ constexpr auto        bs_v            <bogacki_shampine_5<type>> = std::array
{
  type(2479.0 / 34992.0), type(0.0), type(    123.0 /      416.0), type(612941.0 / 3411720.0), type( 43.0 /  1440.0), type(2272.0 / 6561.0), type(79937.0 / 1113912.0), type(3293.0 / 556956.0)
};
template <typename type>
__constant__ constexpr auto        c_v             <bogacki_shampine_5<type>> = std::array
{
  type(0.0), type(1.0 / 6.0), type(2.0 / 9.0), type(3.0 / 7.0), type(2.0 / 3.0), type(3.0 / 4.0), type(1.0), type(1.0)
};

template <typename type>
__constant__ constexpr std::size_t order_v         <bogacki_shampine_5<type>> = 5;
template <typename type>
__constant__ constexpr std::size_t extended_order_v<bogacki_shampine_5<type>> = 4;
}