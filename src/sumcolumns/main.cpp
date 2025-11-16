#include <cblas.h>
#include <omp.h>

#include "../../dependencies/Tensor.hpp"
#include "../../dependencies/Benchmarker.hpp"

__attribute__((used)) void VerifyColumnSum(const Tensor<float>& a, Tensor<float>& b) {
    const auto aDims = a.Dimensions();
    const size_t ar = aDims[0];
    const size_t ac = aDims[1];

    for (size_t r = 0; r < ar; r++) {

        #pragma omp simd
        for (size_t c = 0; c < ac; c++) {
            b.Data()[c] += a.Data()[r*ac+c];
        }
    }
}

__attribute__((used)) void BasicColumnSum(const Tensor<float>& a, Tensor<float>& b) {
    const auto aDims = a.Dimensions();
    const size_t ar = aDims[0];
    const size_t ac = aDims[1];

    for (size_t r = 0; r < ar; r++) {

        #pragma omp parallel for simd schedule(static)
        for (size_t c = 0; c < ac; c++) {
            b.Data()[c] += a.Data()[r*ac+c];
        }
    }
}
__attribute__((used)) void BlasColumnSum(const Tensor<float>& a, Tensor<float>& b) {
    const auto aDims = a.Dimensions();
    const size_t ar = aDims[0];
    const size_t ac = aDims[1];

    for (size_t i = 0; i < ac; i++) {
        b.Data()[i] += cblas_ssum(ar, &a.Data()[i], ar);
    }
}
template <size_t BLOCK_SIZE> __attribute__((used)) void BlockedColumnSum(const Tensor<float>& a, Tensor<float>& b) {
    const auto aDims = a.Dimensions();
    const size_t ar = aDims[0];
    const size_t ac = aDims[1];

    const float* __restrict aData = a.Data();
    float* __restrict bData = b.Data();

    for (size_t r = 0; r < ar; r += BLOCK_SIZE) {
        for (size_t c = 0; c < ac; c += BLOCK_SIZE) {
            const size_t rMax = std::min(r + BLOCK_SIZE, ar);
            const size_t cMax = std::min(c + BLOCK_SIZE, ac);

            for (size_t i = r; i < rMax; i++) {

                #pragma omp simd
                for (size_t j = c; j < cMax; j++) {
                    bData[j] += aData[i*ac+j];
                }
            }
        }
    }
}
__attribute__((used)) void BlasV2ColumnSum(const Tensor<float>& a, Tensor<float>& b) {
    const auto aDims = a.Dimensions();
    const size_t ar = aDims[0];
    const size_t ac = aDims[1];

    const float* __restrict aData = a.Data();
    float* __restrict bData = b.Data();

    for (size_t r = 0; r < ar; r++) {
        cblas_saxpy(ac, 1.0f, &aData[r*ac], 1, bData, 1);
    }
}

template <size_t NUM_THREADS> __attribute__((used)) void ParallelBlas(const Tensor<float>& a, Tensor<float>& b) {
    const auto aDims = a.Dimensions();
    const size_t ar = aDims[0];
    const size_t ac = aDims[1];

    Tensor<float> threadBuf(ac*NUM_THREADS);
    threadBuf.Zero();

    const float* __restrict aData = a.Data();
    float* __restrict bData = b.Data();
    float* __restrict tData = threadBuf.Data();

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        const int tid = omp_get_thread_num();
        float* myBuf = &tData[tid*ac];

        #pragma omp for schedule(static)
        for (size_t r = 0; r < ar; r++) {
            cblas_saxpy(ac, 1.0f, &aData[r*ac], 1, myBuf, 1);
        }

        // serialize into bData
        #pragma omp barrier
        #pragma omp single
        for (size_t i = 0; i < NUM_THREADS; i++) {
            cblas_saxpy(ac, 1.0f, &tData[i*ac], 1, bData, 1);
        }
    }
}


__attribute__((used)) void DispatchedColumnSum(const Tensor<float>& a, Tensor<float>& b) {
    const size_t size = a.Size();

    if (size >= 2048*1024) {
        ParallelBlas<16>(a, b);
    } else {
        BlasV2ColumnSum(a, b);
    }
}

int main() {
    Benchmarker::cAB verify = &BlasV2ColumnSum;
    Settings settings(512, 32, 4096);

    Benchmarker::RunBenchmark<0>("BlasV2", settings, &BlasV2ColumnSum, verify);
    Benchmarker::RunBenchmark<0>("ParallelBlas<16>", settings, &ParallelBlas<16>, verify);
    Benchmarker::RunBenchmark<0>("Dispatched", settings, &DispatchedColumnSum, verify);
}
