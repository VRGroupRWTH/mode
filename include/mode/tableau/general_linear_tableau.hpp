#pragma once

#include <array>
#include <cstddef>

#include <mode/cuda/decorators.hpp>

namespace mode::general_linear_tableau
{
template <typename type>
__constant__ constexpr auto        a_v      = std::array<type, 0> {};
template <typename type>
__constant__ constexpr auto        b_v      = std::array<type, 0> {};
template <typename type>
__constant__ constexpr auto        u_v      = std::array<type, 0> {};
template <typename type>
__constant__ constexpr auto        v_v      = std::array<type, 0> {};

template <typename type>
__constant__ constexpr std::size_t stages_v = 1;
template <typename type>
__constant__ constexpr std::size_t steps_v  = 1;

template <typename type>
__constant__ constexpr std::size_t order_v  = 1;
}