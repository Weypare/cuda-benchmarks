#pragma once
#include <memory>

#include <cuda_runtime.h>
#include <tl/expected.hpp>

#include "utils.hpp"

namespace cuda
{
    using error = cudaError_t;
    template <class T>
    using result = tl::expected<T, error>;

    namespace detail
    {
        struct device_deleter {
            void operator()(void *ptr)
            {
                cudaFree(ptr);
            }
        };
    } // namespace detail

    template <class T>
    struct device_ptr : std::unique_ptr<T, detail::device_deleter> {
        using Base = std::unique_ptr<T, detail::device_deleter>;
        using Base::Base;
        UTILS_DEREF(T *);
    };
    template <class T>
    device_ptr(T *) -> device_ptr<T>;

    template <class T>
    inline result<device_ptr<T>> malloc(std::size_t n)
    {
        T *ptr;
        auto status = cudaMalloc(&ptr, sizeof(T) * n);
        if (status != cudaSuccess)
            return tl::make_unexpected(status);
        return device_ptr{ptr};
    }

    inline result<void> synchronize()
    {
        auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            return tl::make_unexpected(status);
        }
        return {};
    }

    enum class memcpy_kind {
        H2H = cudaMemcpyHostToHost,
        H2D = cudaMemcpyHostToDevice,
        D2H = cudaMemcpyDeviceToHost,
        D2D = cudaMemcpyDeviceToDevice
    };

    template <class T>
    inline result<void> memcpy(T *dst, const T *src, std::size_t n, memcpy_kind kind)
    {
        auto status = cudaMemcpy(dst, src, sizeof(T) * n, static_cast<cudaMemcpyKind>(kind));
        if (status != cudaSuccess)
            return tl::make_unexpected(status);
        return {};
    }

    template <class T>
    inline result<void> memset(T *dst, T value, std::size_t n)
    {
        auto status = cudaMemset(dst, value, sizeof(T) * n);
        if (status != cudaSuccess)
            return tl::make_unexpected(status);
        return {};
    }
} // namespace cuda
