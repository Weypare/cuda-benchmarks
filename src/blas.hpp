#pragma once
#include <memory>

#include <cublas_v2.h>
#include <tl/expected.hpp>

namespace cuda::blas
{
    using error = cublasStatus_t;
    template <class T>
    using result = tl::expected<T, error>;

    result<double> dot_no_handle(std::size_t n, const double *a, const double *b);
} // namespace cuda::blas
