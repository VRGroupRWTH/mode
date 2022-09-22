#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct crank_nicolson
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto a_v<crank_nicolson<type>> = std::array
{
  type(0.0), type(0.0),
  type(0.5), type(0.5),
};
template <typename type>
__constant__ constexpr auto b_v<crank_nicolson<type>> = std::array
{
  type(0.5), type(0.5)
};
template <typename type>
__constant__ constexpr auto c_v<crank_nicolson<type>> = std::array
{
  type(0.0), type(1.0)
};
}