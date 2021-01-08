#pragma once
#include <memory>

#include <cublas_v2.h>
#include <tl/expected.hpp>

#include "utils.hpp"

namespace cuda::blas
{
    using error = cublasStatus_t;
    template <class T>
    using result = tl::expected<T, error>;

    namespace detail
    {
        struct cublas_handle_deleter {
            void operator()(cublasHandle_t ptr)
            {
                cublasDestroy(ptr);
            }
        };
    } // namespace detail

    struct cublas_handle : std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, detail::cublas_handle_deleter> {
        using Base = std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, detail::cublas_handle_deleter>;
        using Base::Base;
        UTILS_DEREF(cublasHandle_t);

        static result<cublas_handle> create()
        {
            cublasHandle_t handle;
            auto status = cublasCreate(&handle);
            if (status != CUBLAS_STATUS_SUCCESS)
                return tl::make_unexpected(status);
            return cublas_handle(handle);
        }
    };

    result<double> dot_no_handle(std::size_t n, const double *a, const double *b);

    result<double> dot(cublasHandle_t handle, std::size_t n, const double *a, const double *b);

    result<void> add(cublasHandle_t handle, std::size_t n, double *a, const double *b);

    result<void>
    matrix_add(cublasHandle_t handle, std::size_t m, std::size_t n, const double *A, const double *B, double *C);
} // namespace cuda::blas
