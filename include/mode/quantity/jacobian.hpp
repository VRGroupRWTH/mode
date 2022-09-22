#pragma once

#include <mode/concepts/scalar.hpp>

#ifdef MODE_USE_EIGEN
#include <Eigen/Core>
//#include <unsupported/Eigen/CXX11/Tensor>
#endif

namespace mode
{
template <typename type_>
class jacobian_of
{
  using type = type_;
};

#ifdef MODE_USE_EIGEN
template <typename type_, std::int32_t rows, std::int32_t options, std::int32_t max_rows, std::int32_t max_cols>
class jacobian_of<Eigen::Matrix<type_, rows, 1, options, max_rows, max_cols>>
{
  using type = Eigen::Matrix<type_, rows, rows, options, max_rows, max_cols>;
};
template <typename type_, std::int32_t cols, std::int32_t options, std::int32_t max_rows, std::int32_t max_cols>
class jacobian_of<Eigen::Matrix<type_, 1, cols, options, max_rows, max_cols>>
{
  using type = Eigen::Matrix<type_, cols, cols, options, max_rows, max_cols>;
};
#endif

template <typename type>
using jacobian_of_t = typename jacobian_of<type>::type;
}