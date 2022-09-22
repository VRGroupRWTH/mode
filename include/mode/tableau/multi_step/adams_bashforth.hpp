#pragma once

#include <mode/tableau/multi_step_tableau.hpp>

namespace mode::multi_step_tableau
{
template <typename type_ = double>
struct adams_bashforth_1
{
  using type = type_;
};
template <typename type_ = double>
struct adams_bashforth_2
{
  using type = type_;
};
template <typename type_ = double>
struct adams_bashforth_3
{
  using type = type_;
};
template <typename type_ = double>
struct adams_bashforth_4
{
  using type = type_;
};
template <typename type_ = double>
struct adams_bashforth_5
{
  using type = type_;
};
template <typename type_ = double>
struct adams_bashforth_6
{
  using type = type_;
};
template <typename type_ = double>
struct adams_bashforth_7
{
  using type = type_;
};
template <typename type_ = double>
struct adams_bashforth_8
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<adams_bashforth_1<type>> = std::array
{
  type(1.0),
  type(1.0)
};
template <typename type>
__constant__ constexpr auto b_v<adams_bashforth_1<type>> = std::array
{
  type(0.0),
  type(1.0)
};

template <typename type>
__constant__ constexpr auto a_v<adams_bashforth_2<type>> = std::array
{
  type(1.0), 
  type(1.0),
  type(0.0)
};
template <typename type>
__constant__ constexpr auto b_v<adams_bashforth_2<type>> = std::array
{
  type(       0.0),
  type( 3.0 / 2.0), 
  type(-1.0 / 2.0)
};

template <typename type>
__constant__ constexpr auto a_v<adams_bashforth_3<type>> = std::array
{
  type(1.0), 
  type(1.0),
  type(0.0),
  type(0.0)
};
template <typename type>
__constant__ constexpr auto b_v<adams_bashforth_3<type>> = std::array
{
  type(         0.0),
  type( 23.0 / 12.0), 
  type(- 4.0 /  3.0), 
  type(  5.0 / 12.0)
};

template <typename type>
__constant__ constexpr auto a_v<adams_bashforth_4<type>> = std::array
{
  type(1.0),
  type(1.0),
  type(0.0),
  type(0.0),
  type(0.0)
};
template <typename type>
__constant__ constexpr auto b_v<adams_bashforth_4<type>> = std::array
{
  type(         0.0),
  type( 55.0 / 24.0), 
  type(-59.0 / 24.0), 
  type( 37.0 / 24.0), 
  type(- 3.0 /  8.0)
};

template <typename type>
__constant__ constexpr auto a_v<adams_bashforth_5<type>> = std::array
{
  type(1.0),
  type(1.0),
  type(0.0),
  type(0.0),
  type(0.0),
  type(0.0)
};
template <typename type>
__constant__ constexpr auto b_v<adams_bashforth_5<type>> = std::array
{
  type(            0.0),
  type( 1901.0 / 720.0), 
  type(-1387.0 / 360.0), 
  type(  109.0 /  30.0), 
  type(- 637.0 / 360.0), 
  type(  251.0 / 720.0)
};

template <typename type>
__constant__ constexpr auto a_v<adams_bashforth_6<type>> = std::array
{
  type(1.0),
  type(1.0),
  type(0.0),
  type(0.0),
  type(0.0),
  type(0.0),
  type(0.0)
};
template <typename type>
__constant__ constexpr auto b_v<adams_bashforth_6<type>> = std::array
{
  type(             0.0),
  type( 4277.0 / 1440.0), 
  type(-2641.0 /  480.0), 
  type( 4991.0 /  720.0), 
  type(-3649.0 /  720.0), 
  type(  959.0 /  480.0), 
  type(-  95.0 /  288.0)
};

template <typename type>
__constant__ constexpr auto a_v<adams_bashforth_7<type>> = std::array
{
  type(1.0),
  type(1.0),
  type(0.0),
  type(0.0),
  type(0.0),
  type(0.0),
  type(0.0),
  type(0.0)
};
template <typename type>
__constant__ constexpr auto b_v<adams_bashforth_7<type>> = std::array
{
  type(                0.0),
  type( 198721.0 / 60480.0), 
  type(- 18637.0 /  2520.0), 
  type( 235183.0 / 20160.0), 
  type(- 10754.0 /   945.0), 
  type( 135713.0 / 20160.0), 
  type(-  5603.0 /  2520.0), 
  type(  19087.0 / 60480.0)
};

template <typename type>
__constant__ constexpr auto a_v<adams_bashforth_8<type>> = std::array
{
  type(1.0),
  type(1.0),
  type(0.0),
  type(0.0),
  type(0.0),
  type(0.0),
  type(0.0),
  type(0.0),
  type(0.0)
};
template <typename type>
__constant__ constexpr auto b_v<adams_bashforth_8<type>> = std::array
{
  type(                  0.0),
  type(   16083.0 /   4480.0), 
  type(-1152169.0 / 120960.0), 
  type(  242653.0 /  13440.0), 
  type(- 296053.0 /  13440.0), 
  type( 2102243.0 / 120960.0), 
  type(- 115747.0 /  13440.0), 
  type(   32863.0 /  13440.0), 
  type(-   5257.0 /  17280.0)
};
}