# CUDA benchmark

## Dependencies

- CUDA Runtime
- cuBlas
- [google/benchmark](https://github.com/google/benchmark)
- [TartanLlama/expected](https://github.com/TartanLlama/expected)
- [catchorg/Catch2](https://github.com/catchorg/Catch2)

## Bootstrapping

```bash
mkdir build && cd build
cmake ..
```

### Running benchmarks

```bash
make -C build
build/bench
```

### Running tests

```bash
make -C build
build/tests
```

Alternatively, via ctest:

```bash
make -C build
cd build
ctest
```
