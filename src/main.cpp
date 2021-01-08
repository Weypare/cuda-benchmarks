#include <stdexcept>

#include <benchmark/benchmark.h>

#include "common.hpp"
#include "blas.hpp"

static void BM_CuBLAS_ScalarProductNoHandle(benchmark::State &state)
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    if (!a || !b) {
        throw std::runtime_error{"Failed to allocate memory"};
    }

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    cuda::memset<double>(a_ptr, 2.0, n);
    cuda::memset<double>(b_ptr, 2.0, n);

    for (auto _ : state) {
        auto result = cuda::blas::dot_no_handle(n, a_ptr, b_ptr);
        if (!result) {
            throw std::runtime_error{"Failed to calculate vector sum"};
        }
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CuBLAS_ScalarProductNoHandle);

static void BM_CuBLAS_ScalarProduct(benchmark::State &state)
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    if (!a || !b) {
        throw std::runtime_error{"Failed to allocate memory"};
    }

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    cuda::memset<double>(a_ptr, 2.0, n);
    cuda::memset<double>(b_ptr, 2.0, n);

    auto handle_result = cuda::blas::cublas_handle::create();
    if (!handle_result) {
        throw std::runtime_error{"Failed to init cuBlas"};
    }
    auto &handle = handle_result.value();

    for (auto _ : state) {
        auto result = cuda::blas::dot(handle, n, a_ptr, b_ptr);
        if (!result) {
            throw std::runtime_error{"Failed to calculate vector sum"};
        }
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CuBLAS_ScalarProduct);

static void BM_CuBLAS_Add(benchmark::State &state)
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    if (!a || !b) {
        throw std::runtime_error{"Failed to allocate memory"};
    }

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    cuda::memset<double>(a_ptr, 2.0, n);
    cuda::memset<double>(b_ptr, 2.0, n);

    auto handle_result = cuda::blas::cublas_handle::create();
    if (!handle_result) {
        throw std::runtime_error{"Failed to init cuBlas"};
    }
    auto &handle = handle_result.value();

    for (auto _ : state) {
        auto result = cuda::blas::add(handle, n, a_ptr, b_ptr);
        if (!result) {
            throw std::runtime_error{"Failed to calculate vector sum"};
        }
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CuBLAS_Add);

BENCHMARK_MAIN();
