#include <functional>

#include "custom.hpp"

namespace cuda::custom
{
    constexpr std::size_t THREADS_PER_BLOCK = 512;

    namespace kernel
    {
        __global__ void add(std::size_t n, const double *a, const double *b, double *c)
        {
            std::size_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (std::size_t idx = thread_id; idx < n; idx += stride) {
                c[idx] = a[idx] + b[idx];
            }
        }

        __global__ void multiply(std::size_t n, const double *a, const double *b, double *c)
        {
            std::size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (; idx < n; idx += stride) {
                c[idx] = a[idx] * b[idx];
            }
        }

        template <class Functor>
        __global__ void zip_with(std::size_t n, const double *a, const double *b, double *c)
        {
            std::size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;
            const auto functor = Functor{};

            for (; idx < n; idx += stride) {
                c[idx] = functor(a[idx], b[idx]);
            }
        }

        __global__ void dot(std::size_t n, const double *a, const double *b, double *buf)
        {
            std::size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            std::size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
            std::size_t stride = blockDim.x * gridDim.x;

            for (; idx < n; idx += stride) {
                buf[idx] = a[idx] * b[idx];
            }

            __syncthreads();

            int max_iter = 0;
            while (n > 1 << max_iter) {
                max_iter++;
            }
            int i = 0;
            while (i < max_iter) {
                std::size_t target_idx = tid * (2 << i);
                std::size_t source_idx = target_idx + (1 << i);
                if (source_idx < n) {
                    buf[target_idx] += buf[source_idx];
                }

                i++;
                __syncthreads();
            }
        }
    } // namespace kernel

    result<double> dot(std::size_t n, const double *a, const double *b)
    {
        auto malloc_result = cuda::malloc<double>(n);
        if (!malloc_result) {
            return malloc_result.error();
        }
        auto &buf = malloc_result.value();

        const auto blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel::dot<<<blocks, THREADS_PER_BLOCK>>>(n, a, b, buf);
        if (auto status = cuda::synchronize(); !status) {
            return status.error();
        }

        double result = 0;
        if (auto status = cuda::memcpy<double>(&result, buf, 1, cuda::memcpy_kind::D2H); !status) {
            return status.error();
        }

        return result;
    }

    result<void> add(std::size_t n, const double *a, const double *b, double *c)
    {
        const auto blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel::add<<<blocks, THREADS_PER_BLOCK>>>(n, a, b, c);
        if (auto status = cuda::synchronize(); !status) {
            return status;
        }
        return {};
    }
    result<void> add_zip_with(std::size_t n, const double *a, const double *b, double *c)
    {
        const auto blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        auto add = kernel::zip_with<std::plus<double>>;
        add<<<blocks, THREADS_PER_BLOCK>>>(n, a, b, c);
        if (auto status = cuda::synchronize(); !status) {
            return status;
        }
        return {};
    }

    result<void> multiply(std::size_t n, const double *a, const double *b, double *c)
    {
        const auto blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel::multiply<<<blocks, THREADS_PER_BLOCK>>>(n, a, b, c);
        if (auto status = cuda::synchronize(); !status) {
            return status;
        }
        return {};
    }
    result<void> multiply_zip_with(std::size_t n, const double *a, const double *b, double *c)
    {
        const auto blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        auto multiply = kernel::zip_with<std::multiplies<double>>;
        multiply<<<blocks, THREADS_PER_BLOCK>>>(n, a, b, c);
        if (auto status = cuda::synchronize(); !status) {
            return status;
        }
        return {};
    }
} // namespace cuda::custom
