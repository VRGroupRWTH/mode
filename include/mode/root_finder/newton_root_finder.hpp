#pragma once

#ifdef MODE_USE_EIGEN
#include <Eigen/Dense>
#endif

#include <mode/concepts/problem.hpp>
#include <mode/concepts/scalar.hpp>
#include <mode/cuda/decorators.hpp>
#include <mode/utility/static_assert_trigger.hpp>

namespace mode
{
template <problem problem_type>
class newton_root_finder
{
public:
  using time_type     = typename problem_type::time_type    ;
  using value_type    = typename problem_type::value_type   ;
  using function_type = typename problem_type::function_type;
  using jacobian_type = typename problem_type::jacobian_type;
  
  __device__ __host__ constexpr auto apply(
    const time_type&     time      ,
    const value_type&    value     , 
    const function_type& function  , 
    const jacobian_type& jacobian  ,
    const time_type&     step_size = time_type(0.001),
    const time_type&     epsilon   = time_type(0.001))
  {
    if constexpr (is_scalar_v<value_type>)
      return value - function(time, value) / jacobian(time, value);
    else
    {
#ifdef MODE_USE_EIGEN
      value_type result(value);

      auto dxdt         = function(time, result);
      auto dxdt_jacobi  = jacobian(time, result) * step_size - jacobian_type::Identity();
      auto lu           = Eigen::FullPivLU<value_type>(dxdt_jacobi);

      auto intermediate = lu.solve(step_size * dxdt).eval();
      result           -= intermediate;

      while (intermediate.norm() > epsilon)
      {
        dxdt            = function(time, result);
        //dxdt_jacobi     = jacobian(time, result) * step_size - jacobian_type::Identity();
        //lu              = Eigen::FullPivLU<value_type>(dxdt_jacobi);

        intermediate = lu.solve(value - result + step_size * dxdt).eval();
        result      -= intermediate;
      }

      return result;
#else
      //static_assert(static_assert_trigger<value_type>::value, "Eigen is necessary for the root finder to work on vector values.");
#endif
    }
  }
};
}