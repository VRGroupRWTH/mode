#pragma once

namespace mode
{
template <typename type>
struct error_evaluation
{
  bool accept;
  type next_step_size;
};
}