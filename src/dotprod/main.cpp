#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include <cblas.h>

#include "../../dependencies/Tensor.hpp"
#include "../../dependencies/Benchmarker.hpp"

__attribute__((used)) void VerifyDotProd(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const float* __restrict a_data = a.Data();
    const float* __restrict b_data = b.Data();
    float* __restrict c_data = c.Data();

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t ar = aDims[0];
    const size_t ac = aDims[1];
    const size_t br = bDims[0];
    const size_t bc = bDims[1];

    for (size_t i = 0; i < ar; i++) {
        for (size_t j = 0; j < br; j++) {

            #pragma omp simd
            for (size_t k = 0; k < bc; k++) {
                c_data[i*bc+k] += a_data[i*ac+j] * b_data[j*bc+k];
            }
        }
    }
}

__attribute__((used)) void BlasDotProd(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t ar = aDims[0];
    const size_t ac = aDims[1];
    const size_t bc = bDims[1];

    const float* __restrict aData = a.Data();
    const float* __restrict bData = b.Data();
    float* __restrict cData = c.Data();

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        ar, bc, ac,
        1.0f, aData, ac, bData, bc,
        1.0f, cData, bc
    );
}
__attribute__((used)) void BasicDotProdCompiler(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const float* __restrict a_data = a.Data();
    const float* __restrict b_data = b.Data();
    float* __restrict c_data = c.Data();

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t ar = aDims[0];
    const size_t ac = aDims[1];
    const size_t br = bDims[0];
    const size_t bc = bDims[1];

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ar; i++) {
        for (size_t j = 0; j < br; j++) {

            #pragma omp simd
            for (size_t k = 0; k < bc; k++) {
                c_data[i*bc+k] += a_data[i*ac+j] * b_data[j*bc+k];
            }
        }
    }
}
template <size_t BLOCK_SIZE> __attribute__((used)) void BlockedDotProdCompiler(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {

    const float* __restrict aData = a.Data();
    const float* __restrict bData = b.Data();
    float* __restrict cData = c.Data();

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t ar = aDims[0];
    const size_t ac = aDims[1];
    const size_t br = bDims[0];
    const size_t bc = bDims[1];

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ar; i += BLOCK_SIZE) {
        for (size_t j = 0; j < br; j += BLOCK_SIZE) {
            for (size_t k = 0; k < bc; k += BLOCK_SIZE) {
                const size_t iMax = std::min(i + BLOCK_SIZE, ar);
                const size_t jMax = std::min(j + BLOCK_SIZE, br);
                const size_t kMax = std::min(k + BLOCK_SIZE, bc);

                for (size_t l = i; l < iMax; l++) {
                    for (size_t m = j; m < jMax; m++) {

                        #pragma omp simd
                        for (size_t n = k; n < kMax; n++) {
                            cData[l*bc+n] += aData[l*ac+m] * bData[m*bc+n];
                        }
                    }
                }
            }
        }
    }
}
template <size_t BLOCK_SIZE> __attribute__((used)) void BlockedDotProdCompiler2(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const float* __restrict aData = a.Data();
    const float* __restrict bData = b.Data();
    float* __restrict cData = c.Data();

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t ar = aDims[0];
    const size_t ac = aDims[1];
    const size_t br = bDims[0];
    const size_t bc = bDims[1];

    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < ar; i += BLOCK_SIZE) {
        for (size_t k = 0; k < bc; k += BLOCK_SIZE) {
            for (size_t j = 0; j < br; j += BLOCK_SIZE) {
                const size_t iMax = std::min(i + BLOCK_SIZE, ar);
                const size_t jMax = std::min(j + BLOCK_SIZE, br);
                const size_t kMax = std::min(k + BLOCK_SIZE, bc);

                for (size_t l = i; l < iMax; l++) {
                    for (size_t m = j; m < jMax; m++) {

                        #pragma omp simd
                        for (size_t n = k; n < kMax; n++) {
                            cData[l*bc+n] += aData[l*ac+m] * bData[m*bc+n];
                        }
                    }
                }
            }
        }
    }
}
template <size_t L1_BLOCK_SIZE, size_t L2_BLOCK_SIZE, size_t L3_BLOCK_SIZE>__attribute__((used)) void BlockedDotProdV3(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const float* __restrict aData = a.Data();
    const float* __restrict bData = b.Data();
    float* __restrict cData = c.Data();

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t ar = aDims[0];
    const size_t ac = aDims[1];
    const size_t br = bDims[0];
    const size_t bc = bDims[1];

    for (size_t i = 0; i < ar; i += L3_BLOCK_SIZE) {
        for (size_t k = 0; k < bc; k += L3_BLOCK_SIZE) {
            for (size_t j = 0; j < br; j += L3_BLOCK_SIZE) {
                const size_t iMax = std::min(i + L3_BLOCK_SIZE, ar);
                const size_t jMax = std::min(j + L3_BLOCK_SIZE, br);
                const size_t kMax = std::min(k + L3_BLOCK_SIZE, bc);

                for (size_t i2 = i; i2 < iMax; i2 += L2_BLOCK_SIZE) {
                    for (size_t k2 = k; k2 < kMax; k2 += L2_BLOCK_SIZE) {
                        for (size_t j2 = j; j2 < jMax; j2 += L2_BLOCK_SIZE) {
                            const size_t i2Max = std::min(i2 + L2_BLOCK_SIZE, ar);
                            const size_t j2Max = std::min(j2 + L2_BLOCK_SIZE, br);
                            const size_t k2Max = std::min(k2 + L2_BLOCK_SIZE, bc);

                            for (size_t i3 = i2; i3 < i2Max; i3 += L1_BLOCK_SIZE) {
                                for (size_t k3 = k2; k3 < k2Max; k3 += L1_BLOCK_SIZE) {
                                    for (size_t j3 = j2; j3 < j2Max; j3 += L1_BLOCK_SIZE) {
                                        const size_t i3Max = std::min(i3 + L1_BLOCK_SIZE, ar);
                                        const size_t j3Max = std::min(j3 + L1_BLOCK_SIZE, br);
                                        const size_t k3Max = std::min(k3 + L1_BLOCK_SIZE, bc);

                                        for (size_t i4 = i3; i4 < i3Max; i4++) {
                                            for (size_t j4 = j3; j4 < j3Max; j4++) {

                                                #pragma omp simd
                                                for (size_t k4 = k3; k4 < k3Max; k4++) {
                                                    cData[i4*bc+k4] += aData[i4*ac+j4] * bData[j4*bc+k4];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

            }
        }
    }
}


double FlopsNeeded(size_t size) {
    return 2.0 * (size*size*size);
}
int main() {
    Benchmarker::cAcBC verify = &BlasDotProd;
    Settings settings(32, 32, 2048);
    settings.FlopsNeeded = &FlopsNeeded;

    Benchmarker::RunBenchmark<0>("Blas" , settings, &BlasDotProd, verify);
    Benchmarker::RunBenchmark<0>("Blocked2<128>", settings, &BlockedDotProdCompiler2<128>, verify);

}
