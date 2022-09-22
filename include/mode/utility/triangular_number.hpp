#pragma once

namespace mode
{
template <auto rank>
constexpr auto triangular_number = rank * (rank + 1) / 2;
}