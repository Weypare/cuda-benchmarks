#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "blas.hpp"
#include "common.hpp"
#include "custom.hpp"

TEST_CASE("CuBLAS Scalar Product No Handle", "[cublas][vector][dot product]")
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    REQUIRE(a);
    REQUIRE(b);

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    std::vector<double> host(n, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D));
    REQUIRE(cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    auto result = cuda::blas::dot_no_handle(n, a_ptr, b_ptr);
    REQUIRE(result);

    auto value = result.value();
    REQUIRE(value == Approx(4000.0).epsilon(0.001));
}

TEST_CASE("CuBLAS Scalar Product", "[cublas][vector][dot product]")
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    REQUIRE(a);
    REQUIRE(b);

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    std::vector<double> host(n, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D));
    REQUIRE(cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    auto handle_result = cuda::blas::cublas_handle::create();
    REQUIRE(handle_result);
    auto &handle = handle_result.value();

    auto result = cuda::blas::dot(handle, n, a_ptr, b_ptr);
    REQUIRE(result);

    auto value = result.value();
    REQUIRE(value == Approx(4000.0).epsilon(0.001));
}

TEST_CASE("CuBLAS Vector addition", "[cublas][vector][addition]")
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    REQUIRE(a);
    REQUIRE(b);

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    std::vector<double> host(n, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    host = std::vector<double>(n, 3.0);
    REQUIRE(cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    auto handle_result = cuda::blas::cublas_handle::create();
    REQUIRE(handle_result);
    auto &handle = handle_result.value();

    auto result = cuda::blas::kapb(handle, n, 1, a_ptr, b_ptr);
    REQUIRE(result);

    REQUIRE(cuda::memcpy<double>(host.data(), b_ptr, n, cuda::memcpy_kind::D2H));
    REQUIRE_THAT(host, Catch::Matchers::Approx(std::vector<double>(n, 5.0)));
}

TEST_CASE("CuBLAS Vector scale", "[cublas][vector][scale]")
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    REQUIRE(a);

    auto &a_ptr = a.value();

    std::vector<double> host(n, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    auto handle_result = cuda::blas::cublas_handle::create();
    REQUIRE(handle_result);
    auto &handle = handle_result.value();

    auto result = cuda::blas::scale(handle, n, 4.0, a_ptr);
    REQUIRE(result);

    REQUIRE(cuda::memcpy<double>(host.data(), a_ptr, n, cuda::memcpy_kind::D2H));
    REQUIRE_THAT(host, Catch::Matchers::Approx(std::vector<double>(n, 8.0)));
}

TEST_CASE("CuBLAS kx + b", "[cublas][vector][kxpb]")
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    REQUIRE(a);
    REQUIRE(b);

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    std::vector<double> host(n, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    host = std::vector<double>(n, 3.0);
    REQUIRE(cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    auto handle_result = cuda::blas::cublas_handle::create();
    REQUIRE(handle_result);
    auto &handle = handle_result.value();

    auto result = cuda::blas::kapb(handle, n, 3, a_ptr, b_ptr);
    REQUIRE(result);

    REQUIRE(cuda::memcpy<double>(host.data(), b_ptr, n, cuda::memcpy_kind::D2H));
    REQUIRE_THAT(host, Catch::Matchers::Approx(std::vector<double>(n, 9.0)));
}

TEST_CASE("CuBLAS Matrix addition", "[cublas][matrix][addition]")
{
    constexpr auto n = 100;
    constexpr auto N = n * n;
    auto a = cuda::malloc<double>(N);
    auto b = cuda::malloc<double>(N);
    REQUIRE(a);
    REQUIRE(b);

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    std::vector<double> host(N, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), N, cuda::memcpy_kind::H2D));

    host = std::vector<double>(N, 3.0);
    REQUIRE(cuda::memcpy<double>(b_ptr, host.data(), N, cuda::memcpy_kind::H2D));

    auto handle_result = cuda::blas::cublas_handle::create();
    REQUIRE(handle_result);
    auto &handle = handle_result.value();

    auto result = cuda::blas::matrix_add(handle, n, n, a_ptr, b_ptr, a_ptr);
    REQUIRE(result);

    REQUIRE(cuda::memcpy<double>(host.data(), a_ptr, N, cuda::memcpy_kind::D2H));
    REQUIRE_THAT(host, Catch::Matchers::Approx(std::vector<double>(N, 5.0)));
}

TEST_CASE("CuBLAS Matrix multiplication", "[cublas][matrix][multiplication]")
{
    constexpr auto n = 100;
    constexpr auto N = n * n;
    auto a = cuda::malloc<double>(N);
    auto b = cuda::malloc<double>(N);
    REQUIRE(a);
    REQUIRE(b);

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    std::vector<double> host(N, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), N, cuda::memcpy_kind::H2D));

    host = std::vector<double>(N, 0.0);
    for (std::size_t i = 0; i < n; i++) {
        host[i * n + i] = 1.0;
    }

    REQUIRE(cuda::memcpy<double>(b_ptr, host.data(), N, cuda::memcpy_kind::H2D));

    auto handle_result = cuda::blas::cublas_handle::create();
    REQUIRE(handle_result);
    auto &handle = handle_result.value();

    auto result = cuda::blas::matrix_multiply(handle, n, n, a_ptr, b_ptr, a_ptr);
    REQUIRE(result);

    REQUIRE(cuda::memcpy<double>(host.data(), a_ptr, N, cuda::memcpy_kind::D2H));
    REQUIRE_THAT(host, Catch::Matchers::Approx(std::vector<double>(N, 2.0)));
}

TEST_CASE("Custom Scalar Product", "[custom][vector][dot product]")
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    REQUIRE(a);
    REQUIRE(b);

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    std::vector<double> host(n, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D));
    REQUIRE(cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    auto result = cuda::custom::dot(n, a_ptr, b_ptr);
    REQUIRE(result);

    auto value = result.value();
    REQUIRE(value == Approx(4000.0).epsilon(0.001));
}

TEST_CASE("Custom Vector addition", "[custom][vector][addition]")
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    REQUIRE(a);
    REQUIRE(b);

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    std::vector<double> host(n, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    host = std::vector<double>(n, 3.0);
    REQUIRE(cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    auto result = cuda::custom::kapb(n, 1, a_ptr, b_ptr, b_ptr);
    REQUIRE(result);

    REQUIRE(cuda::memcpy<double>(host.data(), b_ptr, n, cuda::memcpy_kind::D2H));
    REQUIRE_THAT(host, Catch::Matchers::Approx(std::vector<double>(n, 5.0)));
}

TEST_CASE("Custom Vector scale", "[custom][vector][scale]")
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    REQUIRE(a);

    auto &a_ptr = a.value();

    std::vector<double> host(n, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    auto result = cuda::custom::scale(n, 4.0, a_ptr);
    REQUIRE(result);

    REQUIRE(cuda::memcpy<double>(host.data(), a_ptr, n, cuda::memcpy_kind::D2H));
    REQUIRE_THAT(host, Catch::Matchers::Approx(std::vector<double>(n, 8.0)));
}

TEST_CASE("Custom kx + b", "[custom][vector][kxpb]")
{
    constexpr auto n = 1000;
    auto a = cuda::malloc<double>(n);
    auto b = cuda::malloc<double>(n);
    REQUIRE(a);
    REQUIRE(b);

    auto &a_ptr = a.value();
    auto &b_ptr = b.value();

    std::vector<double> host(n, 2.0);
    REQUIRE(cuda::memcpy<double>(a_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    host = std::vector<double>(n, 3.0);
    REQUIRE(cuda::memcpy<double>(b_ptr, host.data(), n, cuda::memcpy_kind::H2D));

    auto result = cuda::custom::kapb(n, 3, a_ptr, b_ptr, b_ptr);
    REQUIRE(result);

    REQUIRE(cuda::memcpy<double>(host.data(), b_ptr, n, cuda::memcpy_kind::D2H));
    REQUIRE_THAT(host, Catch::Matchers::Approx(std::vector<double>(n, 9.0)));
}
