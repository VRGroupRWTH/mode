#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>

#include <mode/concepts/method.hpp>
#include <mode/concepts/problem.hpp>
#include <mode/cuda/decorators.hpp>

namespace mode
{
template <method method_type_, bool convert_embedded_method = false>
class fixed_step_size_iterator
{
public:
  using method_type       = method_type_;
  using problem_type      = typename method_type ::problem_type;
  using time_type         = typename problem_type::time_type;

  using iterator_category = std::input_iterator_tag; // Single pass read forward.
  using difference_type   = std::ptrdiff_t;
  using value_type        = problem_type;
  using pointer           = value_type const*;
  using reference         = value_type const&;

  __device__ __host__ explicit fixed_step_size_iterator             (const problem_type& problem, const time_type step_size = time_type(0.001))
  : method_(), problem_(problem), step_size_(step_size)
  {
    
  }
  __device__ __host__ fixed_step_size_iterator                      (const fixed_step_size_iterator&  that) = default;
  __device__ __host__ fixed_step_size_iterator                      (      fixed_step_size_iterator&& temp) = default;
  __device__ __host__ virtual ~fixed_step_size_iterator             ()                                      = default;
  __device__ __host__ fixed_step_size_iterator&           operator= (const fixed_step_size_iterator&  that) = default;
  __device__ __host__ fixed_step_size_iterator&           operator= (      fixed_step_size_iterator&& temp) = default;

  __device__ __host__ constexpr reference                 operator* () const noexcept
  {
    return problem_;
  }
  __device__ __host__ constexpr pointer                   operator->() const noexcept
  {
    return &problem_;
  }

  __device__ __host__ constexpr fixed_step_size_iterator& operator++()
  {
    if constexpr (convert_embedded_method)
      problem_.value = method_.apply(problem_, step_size_).value;
    else
      problem_.value = method_.apply(problem_, step_size_);

    problem_.time += step_size_;

    return *this;
  }
  __device__ __host__ constexpr fixed_step_size_iterator  operator++(std::int32_t)
  {
    fixed_step_size_iterator temp = *this;
    ++(*this);
    return temp;
  }

  __device__ __host__ friend constexpr bool               operator==(const fixed_step_size_iterator& lhs, const fixed_step_size_iterator& rhs) noexcept
  {
    return lhs.method_ == rhs.method_ && lhs.problem_ == rhs.problem_ && lhs.step_size_ == rhs.step_size_;
  }

  __device__ __host__ constexpr const method_type&        method    () const
  {
    return method_;
  }
  __device__ __host__ constexpr       method_type&        method    ()
  {
    return method_;
  }
  __device__ __host__ constexpr const problem_type&       problem   () const
  {
    return problem_;
  }
  __device__ __host__ constexpr       problem_type&       problem   ()
  {
    return problem_;
  }
  __device__ __host__ constexpr const time_type&          step_size () const
  {
    return step_size_;
  }
  __device__ __host__ constexpr       time_type&          step_size ()
  {
    return step_size_;
  }

protected:
  method_type  method_   ;
  problem_type problem_  ;
  time_type    step_size_;
};
}