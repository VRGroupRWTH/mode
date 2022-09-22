#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>

#include <mode/concepts/error_controller.hpp>
#include <mode/concepts/method.hpp>
#include <mode/concepts/problem.hpp>
#include <mode/cuda/decorators.hpp>
#include <mode/error/controller/proportional_integral_controller.hpp>

namespace mode
{
template <
  method           method_type_           , 
  error_controller error_controller_type_ = proportional_integral_controller<std::remove_reference_t<typename method_type_::problem_type::time_type>, typename method_type_::tableau_type>>
class adaptive_step_size_iterator
{
public:
  using method_type           = method_type_;
  using error_controller_type = error_controller_type_;
  using problem_type          = typename method_type ::problem_type;
  using time_type             = typename problem_type::time_type;

  using iterator_category     = std::input_iterator_tag; // Single pass read forward.
  using difference_type       = std::ptrdiff_t;
  using value_type            = problem_type;
  using pointer               = value_type const*;
  using reference             = value_type const&;

  __device__ __host__ explicit adaptive_step_size_iterator(
    const problem_type&          problem           , 
    const time_type              initial_step_size = time_type(0.001), 
    const error_controller_type& error_controller  = error_controller_type(),
    const std::size_t            maximum_retries   = static_cast<std::size_t>(3))
  : method_          ()
  , problem_         (problem)
  , step_size_       (initial_step_size)
  , error_controller_(error_controller)
  , maximum_retries_ (maximum_retries)
  , retries_         (0)
  {
    
  }
  __device__ __host__ adaptive_step_size_iterator                            (const adaptive_step_size_iterator&  that) = default;
  __device__ __host__ adaptive_step_size_iterator                            (      adaptive_step_size_iterator&& temp) = default;
  __device__ __host__ virtual ~adaptive_step_size_iterator                   ()                                         = default;
  __device__ __host__ adaptive_step_size_iterator&           operator=       (const adaptive_step_size_iterator&  that) = default;
  __device__ __host__ adaptive_step_size_iterator&           operator=       (      adaptive_step_size_iterator&& temp) = default;

  __device__ __host__ constexpr reference                    operator*       () const noexcept
  {
    return  problem_;
  }
  __device__ __host__ constexpr pointer                      operator->      () const noexcept
  {
    return &problem_;
  }

  __device__ __host__ constexpr adaptive_step_size_iterator& operator++      ()
  {
    const auto result     = method_          .apply   (problem_      , step_size_);
    const auto evaluation = error_controller_.evaluate(problem_.value, step_size_, result);

    if (evaluation.accept || retries_ >= maximum_retries_)
    {
      problem_.value = result.value;
      problem_.time += step_size_;
      step_size_     = evaluation.next_step_size;
      retries_       = 0;
    }
    else
    {
      step_size_     = evaluation.next_step_size;
      retries_++;
      return operator++(); // Retry with the adapted step size.
    }
    return *this;
  }
  __device__ __host__ constexpr adaptive_step_size_iterator  operator++      (std::int32_t)
  {
    adaptive_step_size_iterator temp = *this;
    ++(*this);
    return temp;
  }

  __device__ __host__ friend constexpr bool                  operator==      (const adaptive_step_size_iterator& lhs, const adaptive_step_size_iterator& rhs) noexcept
  {
    return lhs.method_ == rhs.method_ && lhs.problem_ == rhs.problem_ && lhs.step_size_ == rhs.step_size_ && lhs.error_controller_ == rhs.error_controller_;
  }
  
  __device__ __host__ constexpr const method_type&           method          () const
  {
    return method_;
  }
  __device__ __host__ constexpr       method_type&           method          ()
  {
    return method_;
  }
  __device__ __host__ constexpr const problem_type&          problem         () const
  {
    return problem_;
  }
  __device__ __host__ constexpr       problem_type&          problem         ()
  {
    return problem_;
  }
  __device__ __host__ constexpr const time_type&             step_size       () const
  {
    return step_size_;
  }
  __device__ __host__ constexpr       time_type&             step_size       ()
  {
    return step_size_;
  }
  __device__ __host__ constexpr const error_controller_type& error_controller() const
  {
    return error_controller_;
  }
  __device__ __host__ constexpr       error_controller_type& error_controller()
  {
    return error_controller_;
  }
  __device__ __host__ constexpr const std::size_t&           maximum_retries () const
  {
    return maximum_retries_;
  }
  __device__ __host__ constexpr       std::size_t&           maximum_retries ()
  {
    return maximum_retries_;
  }

protected:
  method_type           method_          ;
  problem_type          problem_         ;
  time_type             step_size_       ;
  error_controller_type error_controller_;
  std::size_t           maximum_retries_ ;
  std::size_t           retries_         ;
};
}