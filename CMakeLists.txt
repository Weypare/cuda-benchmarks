cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(
    "CUDA Benchmarks"
    LANGUAGES CXX CUDA
)

include(cmake/CPM.cmake)

find_package(CUDAToolkit)
CPMFindPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    VERSION 1.5.2
    OPTIONS 
        "BENCHMARK_DOWNLOAD_DEPENDENCIES ON"
        "BENCHMARK_ENABLE_TESTING OFF"
)
CPMAddPackage(
    NAME expected
    GITHUB_REPOSITORY TartanLlama/expected
    VERSION 1.0.0
    DOWNLOAD_ONLY TRUE
)
CPMFindPackage(
    NAME Catch2
    GITHUB_REPOSITORY catchorg/Catch2
    VERSION 2.13.4
)

add_executable(
    bench 
    src/main.cpp
    src/utils.hpp
    src/common.hpp
    src/blas.cpp
    src/blas.hpp
    src/custom.cu
    src/custom.hpp
)
set_target_properties(bench PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
target_link_libraries(bench benchmark::benchmark CUDA::cudart CUDA::cublas)
target_include_directories(bench PRIVATE "${expected_SOURCE_DIR}/include")

add_executable(
    tests
    src/test.cpp
    src/utils.hpp
    src/common.hpp
    src/blas.cpp
    src/blas.hpp
    src/custom.cu
    src/custom.hpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2 CUDA::cudart CUDA::cublas)
set_target_properties(tests PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
target_include_directories(tests PRIVATE "${expected_SOURCE_DIR}/include")

include(CTest)
include("${Catch2_SOURCE_DIR}/contrib/Catch.cmake")
catch_discover_tests(tests)
