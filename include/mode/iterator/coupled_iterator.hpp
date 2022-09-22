#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <tuple>

#include <mode/cuda/decorators.hpp>
#include <mode/utility/constexpr_for.hpp>

namespace mode
{
template <typename method_type_, typename problems_type_, bool convert_embedded_method = false>
class coupled_iterator
{
public:
  using method_type       = method_type_;
  using problems_type     = problems_type_;
  using time_type         = typename problems_type::value_type::time_type;

  using iterator_category = std::input_iterator_tag; // Single pass read forward.
  using difference_type   = std::ptrdiff_t;
  using value_type        = problems_type;
  using pointer           = value_type const*;
  using reference         = value_type const&;

  __device__ __host__ explicit coupled_iterator                (const problems_type& problems, const time_type step_size = time_type(0.001))
  : method_(), problems_(problems), step_size_(step_size)
  {
    
  }
  __device__ __host__ coupled_iterator                         (const coupled_iterator&  that) = default;
  __device__ __host__ coupled_iterator                         (      coupled_iterator&& temp) = default;
  __device__ __host__ virtual ~coupled_iterator                ()                              = default;
  __device__ __host__ coupled_iterator&              operator= (const coupled_iterator&  that) = default;
  __device__ __host__ coupled_iterator&              operator= (      coupled_iterator&& temp) = default;

  __device__ __host__ constexpr reference            operator* () const noexcept
  {
    return problems_;
  }
  __device__ __host__ constexpr pointer              operator->() const noexcept
  {
    return &problems_;
  }

  __device__ __host__ constexpr coupled_iterator&    operator++()
  {
    constexpr_for<0, std::tuple_size_v<problems_type>, 1> ([&] (auto i)
    {
      if constexpr (convert_embedded_method)
        std::get<i.value>(problems_).value = method_.apply(std::get<i.value>(problems_), step_size_).value;
      else
        std::get<i.value>(problems_).value = method_.apply(std::get<i.value>(problems_), step_size_);

      std::get<i.value>(problems_).time += step_size_;
    });

    return *this;
  }
  __device__ __host__ constexpr coupled_iterator     operator++(std::int32_t)
  {
    coupled_iterator temp = *this;
    ++(*this);
    return temp;
  }

  __device__ __host__ friend constexpr bool          operator==(const coupled_iterator& lhs, const coupled_iterator& rhs) noexcept
  {
    return lhs.method_ == rhs.method_ && lhs.problems_ == rhs.problems_ && lhs.step_size_ == rhs.step_size_;
  }
  
  __device__ __host__ constexpr const method_type&   method    () const
  {
    return method_;
  }
  __device__ __host__ constexpr       method_type&   method    ()
  {
    return method_;
  }
  __device__ __host__ constexpr const problems_type& problems  () const
  {
    return problems_;
  }
  __device__ __host__ constexpr       problems_type& problems  ()
  {
    return problems_;
  }
  __device__ __host__ constexpr const time_type&     step_size () const
  {
    return step_size_;
  }
  __device__ __host__ constexpr       time_type&     step_size ()
  {
    return step_size_;
  }

protected:
  method_type   method_   ;
  problems_type problems_ ;
  time_type     step_size_;
};
}