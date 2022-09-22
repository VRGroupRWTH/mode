#pragma once

#include <type_traits>

namespace mode
{
template <typename type>
struct is_problem : std::false_type { };

template <typename type>
inline constexpr bool is_problem_v = is_problem<type>::value;

template <typename type>
concept problem = is_problem_v<type>;
}