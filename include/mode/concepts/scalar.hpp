#pragma once

#include <type_traits>

#include <mode/concepts/complex.hpp>

namespace mode
{
template <typename type>
struct is_scalar : std::disjunction<std::is_arithmetic<type>, is_complex<type>>::type { };

template <typename type>
inline constexpr bool is_scalar_v = is_scalar<type>::value;

template <typename type>
concept scalar = is_scalar_v<type>;
}