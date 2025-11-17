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
__attribute__((used)) void VariableBlockDotProd(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const size_t BLOCK_SIZE = a.Dimensions()[0] / 2;

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

int main() {
    Benchmarker::cAcBC verify = &BlasDotProd;
    Settings settings(128, 32, 4096);

    Benchmarker::RunBenchmark("Blas" , settings, &BlasDotProd, verify);
    //Benchmarker::RunBenchmark("Basic Compiler"  , settings, &BasicDotProdCompiler  , verify);
    //Benchmarker::RunBenchmark<0>("Blocked Compiler<16>", settings, &BlockedDotProdCompiler<16>, verify);
    //Benchmarker::RunBenchmark<0>("Blocked Compiler<32>", settings, &BlockedDotProdCompiler<32>, verify);
    //Benchmarker::RunBenchmark<0>("Blocked Compiler<64>", settings, &BlockedDotProdCompiler<64>, verify);
    //Benchmarker::RunBenchmark<0>("Blocked<128>", settings, &BlockedDotProdCompiler<128>, verify);
    Benchmarker::RunBenchmark<0>("Blocked2<128>", settings, &BlockedDotProdCompiler2<128>, verify);
    //Benchmarker::RunBenchmark<0>("Variable Block", settings, &VariableBlockDotProd, verify);
}
