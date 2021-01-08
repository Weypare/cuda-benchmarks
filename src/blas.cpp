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

    result<void> add(cublasHandle_t handle, std::size_t n, double *a, const double *b)
    {
        double alpha = 1;
        auto status = cublasDaxpy(handle, n, &alpha, b, 1, a, 1);
        if (status != CUBLAS_STATUS_SUCCESS)
            return tl::make_unexpected(status);
        return {};
    }
} // namespace cuda::blas
