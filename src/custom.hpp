#pragma once

#include <cuda_runtime.h>
#include <tl/expected.hpp>

#include "common.hpp"

namespace cuda::custom
{
    using error = cudaError_t;
    template <class T>
    using result = tl::expected<T, error>;

    result<double> dot(std::size_t n, const double *a, const double *b);

    result<void> scale(std::size_t n, double k, double *a);

    result<void> add(std::size_t n, const double *a, const double *b, double *c);
    result<void> add_zip_with(std::size_t n, const double *a, const double *b, double *c);

    result<void> multiply(std::size_t n, const double *a, const double *b, double *c);
    result<void> multiply_zip_with(std::size_t n, const double *a, const double *b, double *c);
} // namespace cuda::custom
