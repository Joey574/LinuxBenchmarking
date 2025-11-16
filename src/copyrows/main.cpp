#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include <cblas.h>

#include "../../dependencies/Tensor.hpp"
#include "../../dependencies/Benchmarker.hpp"

__attribute__((used)) void VerifyCopyRow(const Tensor<float>& src, Tensor<float>& dest) {
    const size_t srcSize = src.Size();
    const size_t n = dest.Size() / srcSize;

    for (size_t i = 0; i < n; i++) {
        cblas_scopy(srcSize, src.Data(), 1, &dest.Data()[i*srcSize], 1);
    }
}

__attribute__((used)) void BlasCopyRow(const Tensor<float>& src, Tensor<float>& dest) {
    const size_t srcSize = src.Size();
    const size_t n = dest.Size() / srcSize;

    const float* __restrict srcData = src.Data();
    float* __restrict dstData = dest.Data();

    for (size_t i = 0; i < n; i++) {
        cblas_scopy(srcSize, srcData, 1, &dstData[i*srcSize], 1);
    }
}
__attribute__((used)) void StdCopyRow(const Tensor<float>& src, Tensor<float>& dest) {
    const size_t srcSize = src.Size();
    const size_t n = dest.Size() / srcSize;

    const float* __restrict srcData = src.Data();
    float* __restrict dstData = dest.Data();

    for (size_t i = 0; i < n; i++) {
        memcpy(&dstData[i*srcSize], srcData, srcSize*sizeof(float));
    }
}


int main() {
    Benchmarker::cAB verify = nullptr;
    Settings settings(1024, 1024, 4096);

    Benchmarker::RunBenchmark<1>("Blas", settings, &BlasCopyRow, verify);
    Benchmarker::RunBenchmark<1>("Std", settings, &StdCopyRow, verify);
}
