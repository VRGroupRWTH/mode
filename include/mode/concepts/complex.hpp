#pragma once

#include <complex>
#include <type_traits>

namespace mode
{
template <typename type>
struct is_complex                     : std::false_type { };
template <typename type>
struct is_complex<std::complex<type>> : std::true_type  { };

template <typename type>
inline constexpr bool is_complex_v = is_complex<type>::value;

template <typename type>
concept complex = is_complex_v<type>;
}