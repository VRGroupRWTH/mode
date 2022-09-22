#pragma once

#include <array>
#include <cstddef>

#include <mode/cuda/decorators.hpp>

namespace mode::multi_step_tableau
{
template <typename type>
__constant__ constexpr auto        a_v     = std::array<type, 1> {};
template <typename type>
__constant__ constexpr auto        b_v     = std::array<type, 1> {};

template <typename type>
__constant__ constexpr std::size_t steps_v = b_v<type>.size() - 1;

template <typename type>
__constant__ constexpr std::size_t order_v = 1;
}