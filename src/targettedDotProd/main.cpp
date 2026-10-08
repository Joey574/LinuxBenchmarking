#include <vector>
#include <immintrin.h>
#include <cblas.h>
#include <immintrin.h>

#include "../../dependencies/Tensor.hpp"
#include "../../dependencies/Benchmarker.hpp"

void Blas(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
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
void BaseCase(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const float* __restrict a_data = a.Data();
    const float* __restrict b_data = b.Data();
    float* __restrict c_data = c.Data();

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t M = aDims[0];
    const size_t N = bDims[1];
    const size_t K = aDims[1];

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {

            const float av = a_data[i*K+j];
            const __m256 _a = _mm256_set1_ps(av);

            size_t k = 0;
            for (; k+15 < N; k += 16) {
                const __m256 _b1 = _mm256_loadu_ps(&b_data[j*N+k]);
                const __m256 _b2 = _mm256_loadu_ps(&b_data[j*N+k+8]);
                __m256 _c1 = _mm256_loadu_ps(&c_data[i*N+k]);
                __m256 _c2 = _mm256_loadu_ps(&c_data[i*N+k+8]);

                _c1 = _mm256_fmadd_ps(_a, _b1, _c1);
                _c2 = _mm256_fmadd_ps(_a, _b2, _c2);

                _mm256_storeu_ps(&c_data[i*N+k], _c1);
                _mm256_storeu_ps(&c_data[i*N+k+8], _c2);
            }

            for (; k < N; k++) {
                c_data[i*N+k] += av * b_data[j*N+k];
            }
        }
    }
}
void AppliedCase1(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    /*

        Theory:
            Zen3+ Architecture has 2 FPUs capable of SIMD and 2 FPUs capable of normal float, if we do
            both SIMD and serial operations in our hot loop, we would expect to see improved performance

        Concerns:
            Loads may be the bottleneck here and using more FPUs may not improve our performance, also
            the compiler or CPU itself may already be aware of this and may attempt to optimize this

        Practice:
            The overhead acrued by unaligned loads and stores for simd far outweighs anything we may gain
            from the added FPUs

    */

    const float* __restrict a_data = a.Data();
    const float* __restrict b_data = b.Data();
    float* __restrict c_data = c.Data();

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t M = aDims[0];
    const size_t N = bDims[1];
    const size_t K = aDims[1];

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {

            const float av = a_data[i*K+j];
            const __m256 _a = _mm256_set1_ps(av);

            size_t k = 0;
            for (; k+17 < N; k += 18) {
                const __m256 _b1 = _mm256_loadu_ps(&b_data[j*N+k]);
                const __m256 _b2 = _mm256_loadu_ps(&b_data[j*N+k+8]);
                __m256 _c1 = _mm256_loadu_ps(&c_data[i*N+k]);
                __m256 _c2 = _mm256_loadu_ps(&c_data[i*N+k+8]);

                _c1 = _mm256_fmadd_ps(_a, _b1, _c1);
                _c2 = _mm256_fmadd_ps(_a, _b2, _c2);
                c_data[i*N+k+16] += av * b_data[j*N+k+16];
                c_data[i*N+k+17] += av * b_data[j*N+k+17];

                _mm256_storeu_ps(&c_data[i*N+k], _c1);
                _mm256_storeu_ps(&c_data[i*N+k+8], _c2);
            }

            for (; k < N; k++) {
                c_data[i*N+k] += av * b_data[j*N+k];
            }
        }
    }
}


inline void AVX2_8x8_MicroKernal(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t M, size_t N, size_t K, size_t i1, size_t j1, size_t k1) {
    for (size_t i = i1; i < i1+8; i++) {
        for (size_t j = j1; j < j1+8; j++) {
            for (size_t k = k1; k < k1+8; k++) {
                c[i*N+k] += a[i*K+j] * b[j*N+k];
            }
        }
    }
}

void GeneralCase(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    constexpr size_t KW = 8;
    constexpr size_t KH = 8;

    const float* __restrict a_data = a.Data();
    const float* __restrict b_data = b.Data();
    float* __restrict c_data = c.Data();

    const auto aDims = a.Dimensions();
    const auto bDims = b.Dimensions();

    const size_t M = aDims[0];
    const size_t N = bDims[1];
    const size_t K = aDims[1];

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j+7 < K; j += KH) {
            for (size_t k = 0; k+7 < N; k += KW) {
                AVX2_8x8_MicroKernal(a_data, b_data, c_data, M, N, K, i, j, k);
            }
        }
    }
}


double FlopsNeeded(size_t size) {
    return 2.0 * (size*size*size);
}
int main() {
    Benchmarker::cAcBC verify = &Blas;
    Settings settings(1024, 128, 512);
    settings.FlopsNeeded = &FlopsNeeded;

    Benchmarker::RunBenchmark<0>("GeneralCase", settings, &GeneralCase, verify);
}
