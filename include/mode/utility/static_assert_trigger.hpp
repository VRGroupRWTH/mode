#pragma once

#include <type_traits>

namespace mode
{
template<typename type>
struct static_assert_trigger : std::false_type {};
}