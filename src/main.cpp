#include <stdexcept>
#include <vector>

#include <benchmark/benchmark.h>

#include "blas.hpp"
#include "common.hpp"
#include "custom.hpp"

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

    std::vector<double> host(n, 1.0);
    if (!cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }
    if (!cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }

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

    std::vector<double> host(n, 1.0);
    if (!cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }
    if (!cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }

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

    std::vector<double> host(n, 1.0);
    if (!cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }
    if (!cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }

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

static void BM_CuBLAS_MatrixAdd(benchmark::State &state)
{
    constexpr auto n = 100;
    auto a = cuda::malloc<double>(n * n);
    auto b = cuda::malloc<double>(n * n);
    auto c = cuda::malloc<double>(n * n);
    if (!a || !b || !c) {
        throw std::runtime_error{"Failed to allocate memory"};
    }

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();
    auto &c_ptr = c.value();

    std::vector<double> host(n, 1.0);
    if (!cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }
    if (!cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }

    auto handle_result = cuda::blas::cublas_handle::create();
    if (!handle_result) {
        throw std::runtime_error{"Failed to init cuBlas"};
    }
    auto &handle = handle_result.value();

    for (auto _ : state) {
        auto result = cuda::blas::matrix_add(handle, n, n, a_ptr, b_ptr, c_ptr);
        if (!result) {
            throw std::runtime_error{"Failed to calculate matrix sum"};
        }
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CuBLAS_MatrixAdd);

static void BM_CuBLAS_MatrixMultiply(benchmark::State &state)
{
    constexpr auto n = 100;
    auto a = cuda::malloc<double>(n * n);
    auto b = cuda::malloc<double>(n * n);
    auto c = cuda::malloc<double>(n * n);
    if (!a || !b || !c) {
        throw std::runtime_error{"Failed to allocate memory"};
    }

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();
    auto &c_ptr = c.value();

    std::vector<double> host(n, 1.0);
    if (!cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }
    if (!cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }

    auto handle_result = cuda::blas::cublas_handle::create();
    if (!handle_result) {
        throw std::runtime_error{"Failed to init cuBlas"};
    }
    auto &handle = handle_result.value();

    for (auto _ : state) {
        auto result = cuda::blas::matrix_multiply(handle, n, n, a_ptr, b_ptr, c_ptr);
        if (!result) {
            throw std::runtime_error{"Failed to calculate matrix multiplication"};
        }
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CuBLAS_MatrixMultiply);

static void BM_Custom_ScalarProduct(benchmark::State &state)
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    if (!a || !b) {
        throw std::runtime_error{"Failed to allocate memory"};
    }

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    std::vector<double> host(n, 1.0);
    if (!cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }
    if (!cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }

    for (auto _ : state) {
        auto result = cuda::custom::dot(n, a_ptr, b_ptr);
        if (!result) {
            throw std::runtime_error{"Failed to calculate scalar product"};
        }
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Custom_ScalarProduct);

static void BM_Custom_Add(benchmark::State &state)
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    auto c = cuda::malloc<double>(n);
    if (!a || !b || !c) {
        throw std::runtime_error{"Failed to allocate memory"};
    }

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();
    auto &c_ptr = c.value();

    std::vector<double> host(n, 1.0);
    if (!cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }
    if (!cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }

    for (auto _ : state) {
        auto result = cuda::custom::add(n, a_ptr, b_ptr, c_ptr);
        if (!result) {
            throw std::runtime_error{"Failed to calculate scalar product"};
        }
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Custom_Add);

static void BM_Custom_Multiply(benchmark::State &state)
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    auto c = cuda::malloc<double>(n);
    if (!a || !b || !c) {
        throw std::runtime_error{"Failed to allocate memory"};
    }

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();
    auto &c_ptr = c.value();

    std::vector<double> host(n, 2.0);
    if (!cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }
    if (!cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D)) {
        throw std::runtime_error{"Failed to copy memory"};
    }

    for (auto _ : state) {
        auto result = cuda::custom::multiply(n, a_ptr, b_ptr, c_ptr);
        if (!result) {
            throw std::runtime_error{"Failed to calculate vector multiplication"};
        }
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Custom_Multiply);

BENCHMARK_MAIN();
