#pragma once

#include <mode/tableau/butcher_tableau.hpp>

namespace mode::butcher_tableau
{
template <typename type_ = double>
struct tsitouras_5
{
  using type = type_;
};

template <typename type>
__constant__ constexpr auto        a_v             <tsitouras_5<type>> = std::array
{
  type( 0.161               ),
  type(-0.008480655492356989), type(  0.335480655492357),
  type( 2.8971530571054935  ), type(- 6.359448489975075), type(4.3622954328695815),
  type( 5.325864828439257   ), type(-11.748883564062828), type(7.4955393428898365), type(-0.09249506636175525),
  type( 5.86145544294642    ), type(-12.92096931784711 ), type(8.159367898576159 ), type(-0.071584973281401  ), type(-0.028269050394068383),
  type( 0.09646076681806523 ), type(  0.01             ), type(0.4798896504144996), type( 1.379008574103742  ), type(-3.290069515436081   ), type(2.324710524099774)
};
template <typename type>
__constant__ constexpr auto        b_v             <tsitouras_5<type>> = std::array
{
  type(0.09646076681806523), type(0.01), type(0.4798896504144996), type(1.379008574103742), type(-3.290069515436081), type(2.324710524099774), type(0.0)
};
template <typename type>
__constant__ constexpr auto        bs_v            <tsitouras_5<type>> = std::array
{
  type(0.09468075576583923), type(0.009183565540343), type(0.4877705284247616), type(1.234297566930479), type(-2.707712349983526), type(1.866628418170587), type(-1.0 / 66.0)
};
template <typename type>
__constant__ constexpr auto        c_v             <tsitouras_5<type>> = std::array
{
  type(0.0), type(0.161), type(0.327), type(0.9), type(0.9800255409045097), type(1.0), type(1.0)
};

template <typename type>
__constant__ constexpr std::size_t order_v         <tsitouras_5<type>> = 5;
template <typename type>
__constant__ constexpr std::size_t extended_order_v<tsitouras_5<type>> = 4;
}