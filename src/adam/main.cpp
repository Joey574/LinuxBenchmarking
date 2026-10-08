#include <immintrin.h>
#include <cblas.h>

#include "../../dependencies/Tensor.hpp"
#include "../../dependencies/Benchmarker.hpp"

__attribute__((used)) void OldAdam(Tensor<float>& parameters, const Tensor<float>& derivatives, Tensor<float>& velocity, Tensor<float>& squares, size_t iteration, size_t elements, float learningRate) {
    const float epsilon = 1e-8;
    const float b1 = 0.9f;
    const float b2 = 0.999f;

    //const float factor = learningRate / (float)elements;
    const float factor = learningRate;
    const float b1Rate = 1.0f-b1;
    const float b2Rate = 1.0f-b2;

    const float b1InvDenominator = 1.0f/(1.0f-powf(b1, (float)iteration));
    const float b2InvDenominator = 1.0f/(1.0f-powf(b2, (float)iteration));

    const size_t numParameters = parameters.Size();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        velocity.Data()[i] = (velocity.Data()[i]*b1)+b1Rate*derivatives.Data()[i];
        squares.Data()[i] = (squares.Data()[i]*b2)+b2Rate*derivatives.Data()[i]*derivatives.Data()[i];

        const float mh = velocity.Data()[i]*b1InvDenominator;
        const float vh = squares.Data()[i]*b2InvDenominator;

        parameters.Data()[i] -= factor*(mh/(sqrtf(vh+epsilon)));
    }
}
__attribute__((used)) void BasicAdam(Tensor<float>& parameters, const Tensor<float>& derivatives, Tensor<float>& velocity, Tensor<float>& squares, size_t iteration, size_t elements, float learningRate) {
    const float epsilon = 1e-8;
    const float b1 = 0.9f;
    const float b2 = 0.999f;

    //const float factor = learningRate / (float)elements;
    const float factor = learningRate;
    const float b1Rate = 1.0f-b1;
    const float b2Rate = 1.0f-b2;

    const float b1InvDenominator = 1.0f/(1.0f-powf(b1, (float)iteration));
    const float b2InvDenominator = 1.0f/(1.0f-powf(b2, (float)iteration));

    const size_t numParameters = parameters.Size();

    const float* __restrict dData = derivatives.Data();
    float* __restrict vData = velocity.Data();
    float* __restrict sData = squares.Data();
    float* __restrict pData = parameters.Data();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        vData[i] = (vData[i]*b1)+b1Rate*dData[i];
    }

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        sData[i] = (squares.Data()[i]*b2)+b2Rate*dData[i]*dData[i];
    }

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        const float mh = vData[i]*b1InvDenominator;
        const float vh = sData[i]*b2InvDenominator;

        parameters.Data()[i] -= factor*(mh/(sqrtf(vh+epsilon)));
    }
}
__attribute__((used)) void BlasAdam(Tensor<float>& parameters, const Tensor<float>& derivatives, Tensor<float>& velocity, Tensor<float>& squares, size_t iteration, size_t elements, float learningRate) {
    const float epsilon = 1e-8;
    const float b1 = 0.9f;
    const float b2 = 0.999f;

    //const float factor = learningRate / (float)elements;
    const float factor = learningRate;
    const float b1Rate = 1.0f-b1;
    const float b2Rate = 1.0f-b2;

    const float b1InvDenominator = 1.0f/(1.0f-powf(b1, (float)iteration));
    const float b2InvDenominator = 1.0f/(1.0f-powf(b2, (float)iteration));

    const size_t numParameters = parameters.Size();

    const float* __restrict dData = derivatives.Data();
    float* __restrict vData = velocity.Data();
    float* __restrict sData = squares.Data();
    float* __restrict pData = parameters.Data();

    cblas_saxpby(numParameters, b1Rate, derivatives.Data(), 1, b1, velocity.Data(), 1);

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        sData[i] = (squares.Data()[i]*b2)+b2Rate*dData[i]*dData[i];

        const float mh = vData[i]*b1InvDenominator;
        const float vh = sData[i]*b2InvDenominator;

        parameters.Data()[i] -= factor*(mh/(sqrtf(vh+epsilon)));
    }
}

Tensor<float> dSq(65536);
Tensor<float> mh(65536);
Tensor<float> vh(65536);
__attribute__((used)) void BlasOnlyAdam(Tensor<float>& parameters, const Tensor<float>& derivatives, Tensor<float>& velocity, Tensor<float>& squares, size_t iteration, size_t elements, float learningRate) {
    constexpr const float epsilon = 1e-8;
    constexpr const float b1 = 0.9f;
    constexpr const float b2 = 0.999f;

    //const float factor = learningRate / (float)elements;
    const float factor = learningRate;
    const float b1Rate = 1.0f-b1;
    const float b2Rate = 1.0f-b2;

    const float b1InvDenominator = 1.0f/(1.0f-powf(b1, (float)iteration));
    const float b2InvDenominator = 1.0f/(1.0f-powf(b2, (float)iteration));

    const size_t numParameters = parameters.Size();
    mh.Zero();
    vh.Zero();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        dSq.Data()[i] = derivatives.Data()[i] * derivatives.Data()[i];
    }

    cblas_saxpby(numParameters, b1Rate, derivatives.Data(), 1, b1, velocity.Data(), 1);
    cblas_saxpby(numParameters, b2Rate, dSq.Data(), 1, b2, squares.Data(), 1);
    cblas_saxpy(numParameters, b1InvDenominator, velocity.Data(), 1, mh.Data(), 1);
    cblas_saxpy(numParameters, b2InvDenominator, squares.Data(), 1, vh.Data(), 1);

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < numParameters; i++) {
        parameters.Data()[i] -= factor*(mh.Data()[i]/(sqrtf(vh.Data()[i]+epsilon)));
    }
}

int main() {
    Benchmarker::AcBCD verify = &BasicAdam;
    Settings settings(128, 2048, 65536);

    Benchmarker::RunBenchmark<0>("Old", settings, &OldAdam, verify);
    //Benchmarker::RunBenchmark<0>("Basic", settings, &BasicAdam, verify);
    //Benchmarker::RunBenchmark<0>("Blas", settings, &BlasAdam, verify);
    Benchmarker::RunBenchmark<0>("BlasOnly", settings, &BlasOnlyAdam, verify);
}
