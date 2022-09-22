#pragma once

namespace mode
{
template <typename type>
concept root_finder = requires(type value)
{
  value.apply(
    typename type::time_type    (),
    typename type::value_type   (),
    typename type::function_type(),
    typename type::jacobian_type(),
    typename type::time_type    (),
    typename type::time_type    ());
};
}