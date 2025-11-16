#pragma once
#include <chrono>
#include <csignal>
#include <vector>
#include <string>
#include <immintrin.h>
#include <string.h>
#include <ranges>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <atomic>
#include <thread>
#include <fstream>
#include <random>
#include <format>
#include <cctype>
#include <math.h>
#include <complex>
#include <execution>
#include <functional>
#include <numeric>
#include <cassert>
#include <omp.h>

template <typename T>
struct Tensor {
    public:

    /// @brief Constructor
    template <typename... Dims> Tensor(Dims... dims) : dimensions{dims...}, owner(true) {
        size_t size = Size();
        data = (T*)aligned_alloc(32, size*sizeof(T));
    }


    /// @brief Contstructor
    Tensor(float* data, std::vector<size_t>& dimensions, bool owner=true) : data(data), dimensions(dimensions), owner(owner) {}


    /// @brief Move constructor
    Tensor(Tensor&& other) noexcept : data(other.data), dimensions(std::move(other.dimensions)), owner(true) { other.data = nullptr; }


    /// @brief Copy constructor
    Tensor(const Tensor& other) : dimensions(other.dimensions), owner(true) {
        size_t size = Size();
        std::cout << "[-] Tensor copy constructor (" << size*sizeof(T) << " bytes)\n";

        data = (T*)aligned_alloc(32, size*sizeof(T));
        memcpy(data, other.data, size*sizeof(T));
    }


    /// @brief Deconstructor
    ~Tensor() { if (data && owner) { std::free(data); } }


    /// @brief Move operator
    Tensor& operator = (Tensor&& other) noexcept {
        if (data && owner && this != &other) { std::free(data); }

        data = other.data;
        dimensions = std::move(other.dimensions);
        other.data = nullptr;
        owner = true;
        return *this;
    }


    /// @brief Copy operator
    Tensor& operator = (const Tensor& other) {
        if (data && owner && this != &other) { free(data); }
        dimensions = other.dimensions;

        size_t size = Size();
        std::cout << "[-] Tensor copy assignment (" << size*sizeof(T) << " bytes)\n";

        data = (T*)aligned_alloc(32, size*sizeof(T));
        memcpy(data, other.data, size*sizeof(T));
        owner = true;
        return *this;
    }


    /// @return Const pointer to raw data 
    inline const T* Data() const {
        assert(data != nullptr);
        return data; 
    }


    /// @return Pointer to raw data
    inline T* Data() {
        assert(data != nullptr);
        return data; 
    }


    /// @return The dimensionality of the tensor
    inline constexpr size_t Dimensionality() const {
        return dimensions.size(); 
    }


    /// @return vector of dimensions
    inline constexpr const std::vector<size_t>& Dimensions() const { return dimensions; }


    /// @return The number of elements in the tensor
    inline const size_t Size() const {
        if (dimensions.empty()) { return 0; }
        return std::reduce(std::execution::unseq, dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
    }


    /// @brief Creates a tensor of 1 less dimensionality
    /// @param start The element the view should start at
    /// @param n The number of elements to include
    /// @return A new non-owning tensor from start
    inline Tensor ViewFrom(size_t start, size_t n) {
        assert(!dimensions.empty());
        assert(Size() != 0);

        size_t stride = 1;
        if (dimensions.size() > 1) {
            stride = std::reduce(std::execution::unseq, dimensions.begin(), dimensions.end()-1, 1, std::multiplies<size_t>());
        }

        T* offsetData = data + stride*start;
        auto d = dimensions;
        d[d.size()-1] = n;

        return Tensor(offsetData, d, false);
    }


    inline std::string ToString() const {
        std::string res;
        for (size_t i = 0; i < Size(); i++) {
            res += data[i] + ", ";
        }
        return res;
    }


    inline void Randomize(T min, T max, uint64_t seed) {
        const size_t n = Size();

        const int tid = omp_get_thread_num();
        std::mt19937 gen(seed+tid);
        std::uniform_real_distribution<T> dist(min, max);

        for (size_t i = 0; i < n; i++) {
            data[i] = dist(gen);
        }
    }


    template <typename... Dims> inline void Resize(Dims... dims) {
        if (data && owner) { free(data); }
        dimensions = std::vector<size_t>{dims...};
        size_t size = Size();

        data = (T*)aligned_alloc(32, size*sizeof(T));
        owner = true;
    }


    inline void Zero() {
        memset(data, 0, Size()*sizeof(T));
    }


    inline T MeanPercentDiff(const Tensor& other) {
        if constexpr (std::is_floating_point_v<T>) {
            const size_t n = Size();
            float error = 0.0f;

            #pragma omp parallel for simd schedule(static) reduction(+:error)
            for (size_t i = 0; i < n; i++) {
                error += fabsf(data[i] - other.data[i]);
            }

            return error / (float)n;
        }
    }



    inline bool Equals(const Tensor& other, T tol=0) const {
        const size_t n = Size();

        if constexpr (std::is_floating_point_v<T>) {
            for (size_t i = 0; i < n; i++) {
                if (fabsf(data[i] - other.data[i]) > tol) {
                    std::cout << data[i] << " | " << other.data[i] << "\n";
                    return false;
                }
            }
        }
        

        return true;
    }

    private:

    T* data;
    bool owner;
    std::vector<size_t> dimensions;
};
