#include <cblas.h>
#include <omp.h>

#include "../../dependencies/Tensor.hpp"
#include "../../dependencies/Benchmarker.hpp"


__attribute__((used, optimize("no-tree-vectorize"), noinline))
void UnoptimizedaXpbY(float alpha, const Tensor<float>& x, float beta, Tensor<float>& y) {
    const size_t n = y.Size();

    const float* __restrict xData = x.Data();
    float* __restrict yData = y.Data();

    for (size_t i = 0; i < n; i++) {
        yData[i] = alpha * xData[i] + beta * yData[i];
    }
}
__attribute__((used, optimize("no-tree-vectorize"), noinline))
void ParallelaXpbY(float alpha, const Tensor<float>& x, float beta, Tensor<float>& y) {
    const size_t n = y.Size();

    const float* __restrict xData = x.Data();
    float* __restrict yData = y.Data();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        yData[i] = alpha * xData[i] + beta * yData[i];
    }
}
__attribute__((used, noinline))
void SimdaXpbY(float alpha, const Tensor<float>& x, float beta, Tensor<float>& y) {
    const size_t n = y.Size();

    const float* __restrict xData = x.Data();
    float* __restrict yData = y.Data();

    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        yData[i] = alpha * xData[i] + beta * yData[i];
    }
}
__attribute__((used, noinline))
void CustomSimdaXpbY(float alpha, const Tensor<float>& x, float beta, Tensor<float>& y) {
    const size_t n = y.Size();

    const float* __restrict xData = x.Data();
    float* __restrict yData = y.Data();

    const __m256 _alpha = _mm256_set1_ps(alpha);
    const __m256 _beta = _mm256_set1_ps(beta);

    size_t i = 0;
    for (; i+7 < n; i += 8) {
        __m256 _x = _mm256_load_ps(&xData[i]);
        __m256 _y = _mm256_load_ps(&yData[i]);

        _x = _mm256_mul_ps(_x, _alpha);
        _y = _mm256_mul_ps(_y, _beta);
        _y = _mm256_add_ps(_x, _y);

        _mm256_store_ps(&yData[i], _y);
    }

    for (; i < n; i++) {
        yData[i] = alpha * xData[i] + beta * yData[i];
    }
}

__attribute__((used, noinline))
void ParallelSimdaXpbY(float alpha, const Tensor<float>& x, float beta, Tensor<float>& y) {
    const size_t n = y.Size();

    const float* __restrict xData = x.Data();
    float* __restrict yData = y.Data();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        yData[i] = alpha * xData[i] + beta * yData[i];
    }
}
__attribute__((used))
void BlasaXpbY(float alpha, const Tensor<float>& x, float beta, Tensor<float>& y) {
    cblas_saxpby(x.Size(), alpha, x.Data(), 1, beta, y.Data(), 1);
}

double FlopsNeeded(size_t size) {
    return (double)size * 3.0;
}
int main() {
    Benchmarker::PinToCore();
    Benchmarker::aXpbY verify = &ParallelSimdaXpbY;

    constexpr unsigned long long low  = 1ULL << 18;
    constexpr unsigned long long high = 1ULL << 18;
    Settings settings(8192,  low, high);
    settings.FlopsNeeded = &FlopsNeeded;

    Benchmarker::RunBenchmark<0>("Serial", settings, &UnoptimizedaXpbY, verify);
    Benchmarker::RunBenchmark<0>("Parallel", settings, &ParallelaXpbY, verify);
    Benchmarker::RunBenchmark<0>("Simd", settings, &CustomSimdaXpbY, verify);
    Benchmarker::RunBenchmark<0>("ParallelSimd", settings, &ParallelSimdaXpbY, verify);
}
