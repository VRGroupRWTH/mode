#pragma once

#ifdef _CUDACC
#include <nvfunctional>
#else
#include <functional>
#endif

namespace mode
{
template <typename type>
using function = 
#ifdef _CUDACC
  nvstd::function<type>;
#else
  std  ::function<type>;
#endif
}