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
void ParallelSimdaXpbY(float alpha, const Tensor<float>& x, float beta, Tensor<float>& y) {
    const size_t n = y.Size();

    const float* __restrict xData = x.Data();
    float* __restrict yData = y.Data();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        yData[i] = alpha * xData[i] + beta * yData[i];
    }
}

double FlopsNeeded(size_t size) {
    return (double)size * 3.0;
}
int main() {
    Benchmarker::aXpbY verify = &ParallelSimdaXpbY;
    
    constexpr unsigned long long low  = 1ULL << 14;
    constexpr unsigned long long high = 1ULL << 18;
    Settings settings(4096,  low, high);
    settings.FlopsNeeded = &FlopsNeeded;

    Benchmarker::RunBenchmark<0>("Serial", settings, &UnoptimizedaXpbY, verify);
    Benchmarker::RunBenchmark<0>("ParallelaXpbY", settings, &ParallelaXpbY, verify);
    Benchmarker::RunBenchmark<0>("SimdaXpbY", settings, &SimdaXpbY, verify);
    Benchmarker::RunBenchmark<0>("ParallelSimdaXpbY", settings, &ParallelSimdaXpbY, verify);
}
