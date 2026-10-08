#include <immintrin.h>
#include <cblas.h>

#include "../../dependencies/Tensor.hpp"
#include "../../dependencies/Benchmarker.hpp"

__attribute__((used)) void VerifyLeakyReLU(const Tensor<float>& x, Tensor<float>& y) {
    const size_t n = x.Size();

    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        y.Data()[i] = x.Data()[i] > 0.0f ? x.Data()[i] : (x.Data()[i] * 0.1f);
    }
}

__attribute__((used)) void BasicLeakyReLU(const Tensor<float>& x, Tensor<float>& y) {
    const size_t n = x.Size();

    const float* __restrict xData = x.Data();
    float* __restrict yData = y.Data();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        yData[i] = xData[i] > 0.0f ? xData[i] : (xData[i] * 0.1f);
    }
}

int main() {
    Benchmarker::cAB verify = &VerifyLeakyReLU;
    Settings settings(4096, 2048, 65536);

    Benchmarker::RunBenchmark<2>("Basic", settings, &BasicLeakyReLU, verify);
}
