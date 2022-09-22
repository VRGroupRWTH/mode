#pragma once

#include <array>
#include <cstddef>

#include <mode/cuda/decorators.hpp>

namespace mode::butcher_tableau
{
template <typename type>
__constant__ constexpr auto        a_v              = std::array<type, 0> {};
template <typename type>
__constant__ constexpr auto        b_v              = std::array<type, 0> {};
template <typename type>
__constant__ constexpr auto        bs_v             = std::array<type, 0> {};
template <typename type>
__constant__ constexpr auto        c_v              = std::array<type, 0> {};

template <typename type>
__constant__ constexpr std::size_t stages_v         = b_v <type>.size();

template <typename type>
__constant__ constexpr bool        is_extended_v    = bs_v<type>.size() > 0;

template <typename type>
__constant__ constexpr std::size_t order_v          = 1;
template <typename type>
__constant__ constexpr std::size_t extended_order_v = 1;
}