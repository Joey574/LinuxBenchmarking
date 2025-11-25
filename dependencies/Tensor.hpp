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

        capacity = (size + 32) & ~31;
        data = (T*)aligned_alloc(32, capacity*sizeof(T));
    }


    /// @brief Contstructor
    Tensor(T* data, std::vector<size_t>& dimensions, bool owner=true) : data(data), dimensions(dimensions), owner(owner) {
        capacity = (Size()+32) & ~31;
    }


    /// @brief Move constructor
    Tensor(Tensor&& other) noexcept : data(other.data), dimensions(std::move(other.dimensions)), owner(other.owner), capacity(other.capacity) { 
        other.owner = false;
        other.data = nullptr;
    }


    /// @brief Copy constructor
    Tensor(const Tensor& other) : dimensions(other.dimensions), owner(true) {
        size_t size = Size();
        std::cout << "[-] Tensor copy constructor (" << size*sizeof(T) << " bytes)\n";

        capacity = (size + 32) & ~31;
        data = (T*)aligned_alloc(32, capacity*sizeof(T));
        memcpy(data, other.data, size*sizeof(T));
    }


    /// @brief Deconstructor
    ~Tensor() { if (data && owner) { std::free(data); } }


    /// @brief Move operator
    Tensor& operator = (Tensor&& other) noexcept {
        if (data && owner && this != &other) { std::free(data); }

        data = other.data;
        owner = other.owner;
        capacity = other.capacity;
        dimensions = std::move(other.dimensions);
        other.data = nullptr;
        other.owner = false;
        return *this;
    }


    /// @brief Copy operator
    Tensor& operator = (const Tensor& other) {
        if (data && owner && this != &other) { free(data); }
        dimensions = other.dimensions;

        size_t size = Size();
        std::cout << "[-] Tensor copy assignment (" << size*sizeof(T) << " bytes)\n";

        capacity = (size + 32) & ~31;
        data = (T*)aligned_alloc(32, capacity*sizeof(T));
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


    inline void Randomize(T min, T max, uint64_t seed) {
        const size_t n = Size();

        std::mt19937 gen(seed);
        std::uniform_real_distribution<T> dist(min, max);

        for (size_t i = 0; i < n; i++) {
            data[i] = dist(gen);
        }
    }


    inline void Zero() {
        memset(data, 0, capacity*sizeof(T));
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
    size_t capacity;
    std::vector<size_t> dimensions;
};
