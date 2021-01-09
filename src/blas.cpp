#include "blas.hpp"

namespace cuda::blas
{
    result<double> dot_no_handle(std::size_t n, const double *a, const double *b)
    {
        auto handle_result = cublas_handle::create();
        if (!handle_result)
            return handle_result.error();
        auto &handle = handle_result.value();

        double result;
        auto status = cublasDdot(handle, n, a, 1, b, 1, &result);
        if (status != CUBLAS_STATUS_SUCCESS)
            return status;
        return result;
    }

    result<double> dot(cublasHandle_t handle, std::size_t n, const double *a, const double *b)
    {
        double result;
        auto status = cublasDdot(handle, n, a, 1, b, 1, &result);
        if (status != CUBLAS_STATUS_SUCCESS)
            return status;
        return result;
    }

    result<void> scale(cublasHandle_t handle, std::size_t n, double k, double *a)
    {
        auto status = cublasDscal(handle, n, &k, a, 1);
        if (status != CUBLAS_STATUS_SUCCESS)
            return tl::make_unexpected(status);
        return {};
    }

    result<void> kapb(cublasHandle_t handle, std::size_t n, double k, const double *a, double *b)
    {
        auto status = cublasDaxpy(handle, n, &k, a, 1, b, 1);
        if (status != CUBLAS_STATUS_SUCCESS)
            return tl::make_unexpected(status);
        return {};
    }

    result<void>
    matrix_add(cublasHandle_t handle, std::size_t m, std::size_t n, const double *A, const double *B, double *C)
    {
        double alpha = 1;
        auto status = cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, A, m, &alpha, B, m, C, m);
        if (status != CUBLAS_STATUS_SUCCESS)
            return tl::make_unexpected(status);
        return {};
    }

    result<void>
    matrix_multiply(cublasHandle_t handle, std::size_t m, std::size_t n, const double *A, const double *B, double *C)
    {
        double alpha = 1;
        double beta = 0;
        auto status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &alpha, A, m, B, m, &beta, C, m);
        if (status != CUBLAS_STATUS_SUCCESS)
            return tl::make_unexpected(status);
        return {};
    }
} // namespace cuda::blas
