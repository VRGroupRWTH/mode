#pragma once

namespace mode
{
template <typename type>
concept method = requires(type value)
{
  value.apply(typename type::problem_type(), typename type::problem_type::time_type());
};
}