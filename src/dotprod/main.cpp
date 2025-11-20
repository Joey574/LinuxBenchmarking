#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include <cblas.h>
#include <immintrin.h>

#include "../../dependencies/Tensor.hpp"
#include "../../dependencies/Benchmarker.hpp"


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
__attribute__((used)) void BlockedDotProdV2Dispatch(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const size_t size = a.Size();

    if (size <= 128*128) {
        BlockedDotProdCompiler2<64>(a, b, c);
    } else if (size <= 1024*1024) {
        BlockedDotProdCompiler2<128>(a, b, c);
    } else {
        BlockedDotProdCompiler2<256>(a, b, c);
    }
}

inline void V3MicroKernel(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t M, size_t N, size_t K, size_t iL1, size_t jL1, size_t kL1, size_t iL1Max, size_t jL1Max, size_t kL1Max) {
    for (size_t i = iL1; i + 3 < iL1Max; i += 4) {

        const float* __restrict aRow0 = &a[(i+0)*K];
        const float* __restrict aRow1 = &a[(i+1)*K];
        const float* __restrict aRow2 = &a[(i+2)*K];
        const float* __restrict aRow3 = &a[(i+3)*K];

        float* __restrict cRow0 = &c[(i+0)*N];
        float* __restrict cRow1 = &c[(i+1)*N];
        float* __restrict cRow2 = &c[(i+2)*N];
        float* __restrict cRow3 = &c[(i+3)*N];

        for (size_t j = jL1; j < jL1Max; j += 4) {
            const size_t jjMax = std::min(j + 4, jL1Max);

            for (size_t k = kL1; k + 7 < kL1Max; k += 8) {

                // load 4x8 chunk of c
                __m256 _c0 = _mm256_load_ps(&cRow0[k]);
                __m256 _c1 = _mm256_load_ps(&cRow1[k]);
                __m256 _c2 = _mm256_load_ps(&cRow2[k]);
                __m256 _c3 = _mm256_load_ps(&cRow3[k]);

                // Accumulate over the small inner tile (up to 4)
                #pragma GCC unroll 4
                for (size_t jj = j; jj < jjMax; jj++) {
                    const __m256 _a0 = _mm256_set1_ps(aRow0[jj]);
                    const __m256 _a1 = _mm256_set1_ps(aRow1[jj]);
                    const __m256 _a2 = _mm256_set1_ps(aRow2[jj]);
                    const __m256 _a3 = _mm256_set1_ps(aRow3[jj]);

                    const __m256 _b = _mm256_load_ps(&b[jj * N + k]);

                    _c0 = _mm256_fmadd_ps(_a0, _b, _c0);
                    _c1 = _mm256_fmadd_ps(_a1, _b, _c1);
                    _c2 = _mm256_fmadd_ps(_a2, _b, _c2);
                    _c3 = _mm256_fmadd_ps(_a3, _b, _c3);
                }

                _mm256_store_ps(&cRow0[k], _c0);
                _mm256_store_ps(&cRow1[k], _c1);
                _mm256_store_ps(&cRow2[k], _c2);
                _mm256_store_ps(&cRow3[k], _c3);
            } // end k vector chunks

            // Scalar remainder for k (the tail columns)
            size_t kTailStart = kL1 + ((kL1Max - kL1) / 8) * 8;
            for (size_t kk = kTailStart; kk < kL1Max; kk++) {
                for (size_t jj = j; jj < jjMax; ++jj) {
                    const float bv = b[jj*N + kk];

                    cRow0[kk] += aRow0[jj] * bv;
                    cRow1[kk] += aRow1[jj] * bv;
                    cRow2[kk] += aRow2[jj] * bv;
                    cRow3[kk] += aRow3[jj] * bv;
                }
            }
        } // end j-blocks (4-wide)
    } // end i blocks (4 rows)

    // Handle leftover i rows (i rem < 4)
    for (size_t iRem = (iL1Max / 4) * 4; iRem < iL1Max; iRem++) {

        const float* __restrict aRow = &a[iRem*K];
        float* __restrict cRow = &c[iRem*N];

        for (size_t j = jL1; j < jL1Max; j++) {
            const __m256 _a = _mm256_set1_ps(aRow[j]);

            size_t k = kL1;
            for (; k + 7 < kL1Max; k += 8) {
                const __m256 _b = _mm256_load_ps(&b[j*N + k]);
                __m256 _c = _mm256_load_ps(&cRow[k]);

                _c = _mm256_fmadd_ps(_a, _b, _c);
                _mm256_store_ps(&cRow[k], _c);
            }

            for (; k < kL1Max; ++k) {
                cRow[k] += aRow[j] * b[j*N + k];
            }
        }
    }
}
template <size_t L1_BLOCK_SIZE, size_t L2_BLOCK_SIZE, size_t L3_BLOCK_SIZE> __attribute__((used)) void BlockedDotProdV3(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C) {
    const float* __restrict aData = A.Data();
    const float* __restrict bData = B.Data();
    float* __restrict cData = C.Data();

    const auto aDims = A.Dimensions();
    const auto bDims = B.Dimensions();

    const size_t M = aDims[0];
    const size_t K = aDims[1];
    const size_t N = bDims[1];

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t iL3 = 0; iL3 < M; iL3 += L3_BLOCK_SIZE) {
        for (size_t kL3 = 0; kL3 < N; kL3 += L3_BLOCK_SIZE) {
            const size_t iL3Max = std::min(iL3 + L3_BLOCK_SIZE, M);
            const size_t kL3Max = std::min(kL3 + L3_BLOCK_SIZE, N);

            for (size_t jL3 = 0; jL3 < K; jL3 += L3_BLOCK_SIZE) {
                const size_t jL3Max = std::min(jL3 + L3_BLOCK_SIZE, K);

                // L2 blocking
                for (size_t iL2 = iL3; iL2 < iL3Max; iL2 += L2_BLOCK_SIZE) {
                    const size_t iL2Max = std::min(iL2 + L2_BLOCK_SIZE, iL3Max);

                    for (size_t kL2 = kL3; kL2 < kL3Max; kL2 += L2_BLOCK_SIZE) {
                        const size_t kL2Max = std::min(kL2 + L2_BLOCK_SIZE, kL3Max);

                        for (size_t jL2 = jL3; jL2 < jL3Max; jL2 += L2_BLOCK_SIZE) {
                            const size_t jL2Max = std::min(jL2 + L2_BLOCK_SIZE, jL3Max);

                            // L1 blocking
                            for (size_t iL1 = iL2; iL1 < iL2Max; iL1 += L1_BLOCK_SIZE) {
                                const size_t iL1Max = std::min(iL1 + L1_BLOCK_SIZE, iL2Max);

                                for (size_t kL1 = kL2; kL1 < kL2Max; kL1 += L1_BLOCK_SIZE) {
                                    const size_t kL1Max = std::min(kL1 + L1_BLOCK_SIZE, kL2Max);

                                    for (size_t jL1 = jL2; jL1 < jL2Max; jL1 += L1_BLOCK_SIZE) {
                                        const size_t jL1Max = std::min(jL1 + L1_BLOCK_SIZE, jL2Max);

                                        // ---- 4x4 register-blocked microkernel ----
                                        // Process 4 rows (i..i+3) and up to 4 inner cols (j..j+3).
                                        V3MicroKernel(aData, bData, cData, M, N, K, iL1, jL1, kL1, iL1Max, jL1Max, kL1Max);
                                        
                                    } // end jL1
                                } // end kL1
                            } // end iL1
                        } // end jL2
                    } // end kL2
                } // end iL2
            } // end jL3
        } // end kL3
    } // end iL3
}
__attribute__((used)) void BlockedDotProdV3Dispatch(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const size_t size = a.Size();

    if (size <= 256*256) {
        BlockedDotProdV3<16, 32, 64>(a, b, c);
    } else if (size <= 1024*1024) {
        BlockedDotProdV3<32, 64, 128>(a, b, c);
    } else {
        BlockedDotProdV3<64, 128, 512>(a, b, c);
    }
}


inline void V4MicroKernel(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t M, size_t N, size_t K, size_t iL1, size_t jL1, size_t kL1, size_t iL1Max, size_t jL1Max, size_t kL1Max, size_t kCount) {
    size_t i = iL1;
    for (; i + 3 < iL1Max; i += 4) {

        const float* __restrict aRow0 = &a[(i+0)*K];
        const float* __restrict aRow1 = &a[(i+1)*K];
        const float* __restrict aRow2 = &a[(i+2)*K];
        const float* __restrict aRow3 = &a[(i+3)*K];

        float* __restrict cRow0 = &c[(i+0)*N];
        float* __restrict cRow1 = &c[(i+1)*N];
        float* __restrict cRow2 = &c[(i+2)*N];
        float* __restrict cRow3 = &c[(i+3)*N];

        for (size_t j = jL1; j < jL1Max; j += 4) {
            const size_t jjMax = std::min(j + 4, jL1Max);
            const size_t jjCount = jjMax - j;

            size_t kLocal = 0;

            for (; kLocal + 7 < kCount; kLocal += 8) {
                const size_t kGlobal = kL1+kLocal;

                // load 4x8 chunk of c
                __m256 _c0 = _mm256_load_ps(&cRow0[kGlobal]);
                __m256 _c1 = _mm256_load_ps(&cRow1[kGlobal]);
                __m256 _c2 = _mm256_load_ps(&cRow2[kGlobal]);
                __m256 _c3 = _mm256_load_ps(&cRow3[kGlobal]);

                // accumulate over the small inner tile
                #pragma GCC unroll 4
                for (size_t jjLocal = 0; jjLocal < jjCount; jjLocal++) {
                    const size_t jjGlobal = j+jjLocal;
                    const size_t jjRow = jjGlobal-jL1;

                    const __m256 _a0 = _mm256_set1_ps(aRow0[jjGlobal]);
                    const __m256 _a1 = _mm256_set1_ps(aRow1[jjGlobal]);
                    const __m256 _a2 = _mm256_set1_ps(aRow2[jjGlobal]);
                    const __m256 _a3 = _mm256_set1_ps(aRow3[jjGlobal]);

                    const float* __restrict bRow = &b[jjRow*kCount];
                    const __m256 _b = _mm256_load_ps(&bRow[kLocal]);

                    _c0 = _mm256_fmadd_ps(_a0, _b, _c0);
                    _c1 = _mm256_fmadd_ps(_a1, _b, _c1);
                    _c2 = _mm256_fmadd_ps(_a2, _b, _c2);
                    _c3 = _mm256_fmadd_ps(_a3, _b, _c3);
                }

                _mm256_store_ps(&cRow0[kGlobal], _c0);
                _mm256_store_ps(&cRow1[kGlobal], _c1);
                _mm256_store_ps(&cRow2[kGlobal], _c2);
                _mm256_store_ps(&cRow3[kGlobal], _c3);
            } // end k vector chunks

            // Scalar remainder for k, <= 7
            #pragma GCC unroll 7
            for (; kLocal < kCount; kLocal++) {
                const size_t kkGlobal = kL1+kLocal;

                for (size_t jjLocal = 0; jjLocal < jjCount; jjLocal++) {
                    const size_t jjGlobal = j+jjLocal;
                    const size_t jjRow = jjGlobal-jL1;

                    const float bv = b[jjRow*kCount + kLocal];
                    cRow0[kkGlobal] += aRow0[jjGlobal] * bv;
                    cRow1[kkGlobal] += aRow1[jjGlobal] * bv;
                    cRow2[kkGlobal] += aRow2[jjGlobal] * bv;
                    cRow3[kkGlobal] += aRow3[jjGlobal] * bv;
                }
            }
        } // end j-blocks
    } // end i blocks

    // scalar i rows, <= 3
    #pragma GCC unroll 3
    for (; i < iL1Max; i++) {

        const float* __restrict aRow = &a[i*K];
        float* __restrict cRow = &c[i*N];

        for (size_t jGlobal = jL1; jGlobal < jL1Max; jGlobal++) {
            const float* __restrict bRow = &b[(jGlobal-jL1)*kCount];
            const __m256 _a = _mm256_set1_ps(aRow[jGlobal]);

            size_t kLocal = 0;
            for (; kLocal + 7 < kCount; kLocal += 8) {
                const size_t kGlobal = kL1+kLocal;

                const __m256 _b = _mm256_load_ps(&bRow[kLocal]);
                __m256 _c = _mm256_load_ps(&cRow[kGlobal]);

                _c = _mm256_fmadd_ps(_a, _b, _c);
                _mm256_store_ps(&cRow[kGlobal], _c);
            }

            #pragma GCC unroll 7
            for (; kLocal < kCount; kLocal++) {
                cRow[kL1+kLocal] += aRow[jGlobal] * bRow[kLocal];
            }
        }
    }
}
template <size_t L1_BLOCK_SIZE, size_t L2_BLOCK_SIZE, size_t L3_BLOCK_SIZE> __attribute__((used)) void BlockedDotProdV4(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C) {
    const float* __restrict aData = A.Data();
    const float* __restrict bData = B.Data();
    float* __restrict cData = C.Data();

    const auto aDims = A.Dimensions();
    const auto bDims = B.Dimensions();

    const size_t M = aDims[0];
    const size_t K = aDims[1];
    const size_t N = bDims[1];

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t iL3 = 0; iL3 < M; iL3 += L3_BLOCK_SIZE) {
        for (size_t kL3 = 0; kL3 < N; kL3 += L3_BLOCK_SIZE) {
            const size_t iL3Max = std::min(iL3 + L3_BLOCK_SIZE, M);
            const size_t kL3Max = std::min(kL3 + L3_BLOCK_SIZE, N);

            for (size_t jL3 = 0; jL3 < K; jL3 += L3_BLOCK_SIZE) {
                const size_t jL3Max = std::min(jL3 + L3_BLOCK_SIZE, K);

                Tensor<float> bPacked(L1_BLOCK_SIZE * L1_BLOCK_SIZE);
                float* __restrict bPackedData = bPacked.Data();

                // L2 blocking
                for (size_t iL2 = iL3; iL2 < iL3Max; iL2 += L2_BLOCK_SIZE) {
                    const size_t iL2Max = std::min(iL2 + L2_BLOCK_SIZE, iL3Max);

                    for (size_t kL2 = kL3; kL2 < kL3Max; kL2 += L2_BLOCK_SIZE) {
                        const size_t kL2Max = std::min(kL2 + L2_BLOCK_SIZE, kL3Max);

                        for (size_t jL2 = jL3; jL2 < jL3Max; jL2 += L2_BLOCK_SIZE) {
                            const size_t jL2Max = std::min(jL2 + L2_BLOCK_SIZE, jL3Max);

                            // L1 blocking
                            for (size_t iL1 = iL2; iL1 < iL2Max; iL1 += L1_BLOCK_SIZE) {
                                const size_t iL1Max = std::min(iL1 + L1_BLOCK_SIZE, iL2Max);

                                for (size_t kL1 = kL2; kL1 < kL2Max; kL1 += L1_BLOCK_SIZE) {
                                    const size_t kL1Max = std::min(kL1 + L1_BLOCK_SIZE, kL2Max);

                                    for (size_t jL1 = jL2; jL1 < jL2Max; jL1 += L1_BLOCK_SIZE) {
                                        const size_t jL1Max = std::min(jL1 + L1_BLOCK_SIZE, jL2Max);

                                        const size_t jjCount = jL1Max - jL1;
                                        const size_t kCount = kL1Max - kL1;

                                        // prep bPacked
                                        for (size_t jj = jL1; jj < jL1Max; jj++) {
                                            memcpy(&bPackedData[(jj-jL1)*kCount], &bData[jj*N + kL1], kCount*sizeof(float));
                                        }

                                        // ---- 4x4 register-blocked microkernel ----
                                        // Process 4 rows (i..i+3) and up to 4 inner cols (j..j+3).
                                        V4MicroKernel(aData, bPackedData, cData, M, N, K, iL1, jL1, kL1, iL1Max, jL1Max, kL1Max, kCount);
                                            
                                    } // end jL1
                                } // end kL1
                            } // end iL1
                        } // end jL2
                    } // end kL2
                } // end iL2
            } // end jL3
        } // end kL3
    } // end iL3
}
__attribute__((used)) void BlockedDotProdV4Dispatch(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const size_t size = a.Size();

    if (size <= 256*256) {
        BlockedDotProdV4<16, 32, 64>(a, b, c);
    } else if (size <= 512*512) {
        BlockedDotProdV4<32, 64, 128>(a, b, c);
    } else if (size <= 1024*1024) {
        BlockedDotProdV4<64, 128, 256>(a, b, c);
    } else {
        BlockedDotProdV4<64, 128, 512>(a, b, c);
    }
}


double FlopsNeeded(size_t size) {
    return 2.0 * (size*size*size);
}
int main() {
    Benchmarker::cAcBC verify = &BlasDotProd;
    Settings settings(32, 128, 4096);
    settings.FlopsNeeded = &FlopsNeeded;

    Benchmarker::RunBenchmark<0>("Blas" , settings, &BlasDotProd, verify);
    Benchmarker::RunBenchmark<0>("BlockedV4Dispatch", settings, &BlockedDotProdV4Dispatch, verify);
    Benchmarker::RunBenchmark<0>("Basic", settings, &BasicDotProdCompiler, verify);
    //Benchmarker::RunBenchmark<0>("BlockedV3Dispatch", settings, &BlockedDotProdV3Dispatch, verify);
}
