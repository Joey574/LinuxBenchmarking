#pragma once
#include "Tensor.hpp"

#include <string>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <execution>
#include <chrono>
#include <type_traits>
#include <variant>
#include <fstream>
#include <unistd.h>
#include <limits.h>

struct Settings {
    public:
    Settings(size_t runs, size_t startSize, size_t endSize) : runs(runs), startSize(startSize), endSize(endSize), flops(TheoreticalFlops()) {}
    double(*FlopsNeeded)(size_t size);
    double flops;

    size_t runs;
    size_t startSize;
    size_t endSize;

    static inline double TheoreticalFlops() {
        double cores = 8;
        double vecWidth = 8;
        double frequency = 4.769e9;

        return cores*vecWidth*frequency;
    }
};


struct Benchmarker {
    using cAcBC = void(*)(const Tensor<float>&, const Tensor<float>&, Tensor<float>&);
    using cAB = void(*)(const Tensor<float>&, Tensor<float>&);
    using AcBCD = void(*)(Tensor<float>&, const Tensor<float>&, Tensor<float>&, Tensor<float>&, size_t, size_t, float);
    using aXpbY = void(*)(float, const Tensor<float>&, float, Tensor<float>&);

    public:
    template <int TYPE=0> static inline void RunBenchmark(const std::string& name, const Settings& settings, const cAcBC test, const cAcBC verify=nullptr) {
        std::cout << "Benchmarking: \033[33m" + name + "\033[0m\n";

        const size_t start_size = settings.startSize;
        const size_t max_size = settings.endSize;
        const size_t runs = settings.runs;

        const int slen = NumDigits(max_size) * 2 + 2;

        auto times = std::vector<double>(runs, 0.0);
        auto error = std::vector<double>(runs, 0.0);
        auto flops = std::vector<double>(runs, 0.0);

        for (size_t size = start_size; size <= max_size; size *= 2) {
            Tensor<float> a;
            Tensor<float> b;
            Tensor<float> c;
            Tensor<float> check;

            if constexpr (TYPE == 0) {
                a = Tensor<float>(size, size);
                b = Tensor<float>(size, size);
                c = Tensor<float>(size, size);
                check = Tensor<float>(size, size);
            } else if constexpr (TYPE == 1) {
                a = Tensor<float>(size);
                b = Tensor<float>(size);
                c = Tensor<float>(size);
                check = Tensor<float>(size);
            }

            for (size_t r = 0; r < runs; r++) {
                a.Randomize(-100.0f, 100.0f, 14563+r);
                b.Randomize(-100.0f, 100.0f, 87741+r);
                c.Zero();

                // warmups
                test(a, b, check);
                test(a, b, check);

                auto start = std::chrono::steady_clock::now();
                test(a, b, c);
                auto end = std::chrono::steady_clock::now();
                times[r] = std::chrono::duration<double, std::milli>(end - start).count();

                if (verify) {
                    check.Zero();
                    verify(a, b, check);
                    error[r] = c.MeanPercentDiff(check);
                }

                if (settings.FlopsNeeded) {
                    flops[r] = settings.FlopsNeeded(size) / (times[r] / 1000.00);
                }
            }

            OutputResults(settings, size, slen, times, error, flops, verify != nullptr, settings.FlopsNeeded != nullptr);
        }
    }
    template <int TYPE=0> static inline void RunBenchmark(const std::string& name, const Settings& settings, const cAB test, const cAB verify=nullptr) {
        std::cout << "Benchmarking: \033[33m" + name + "\033[0m\n";

        const size_t start_size = settings.startSize;
        const size_t max_size = settings.endSize;
        const size_t runs = settings.runs;

        const int slen = NumDigits(max_size) * 2 + 2;

        auto times = std::vector<double>(runs, 0.0);
        auto error = std::vector<double>(runs, 0.0);
        auto flops = std::vector<double>(runs, 0.0);

        for (size_t size = start_size; size <= max_size; size *= 2) {

            Tensor<float> a, b, check;
            if constexpr (TYPE == 0) {
                a = Tensor<float>(size, size);
                b = Tensor<float>(size);
                check = Tensor<float>(size);
            } else if constexpr (TYPE == 1) {
                a = Tensor<float>(size);
                b = Tensor<float>(size, size);
                check = Tensor<float>(size, size);
            } else if constexpr (TYPE == 2) {
                a = Tensor<float>(size);
                b = Tensor<float>(size);
                check = Tensor<float>(size);
            }

            // warmups
            test(a, b);
            test(a, b);

            for (size_t r = 0; r < runs; r++) {
                a.Randomize(-100.0f, 100.0f, 14563+r);
                b.Zero();

                auto start = std::chrono::steady_clock::now();
                test(a, b);
                auto end = std::chrono::steady_clock::now();
                times[r] = std::chrono::duration<double, std::milli>(end - start).count();

                if (verify) {
                    check.Zero();
                    verify(a, check);
                    error[r] = b.MeanPercentDiff(check);
                }
            }

            OutputResults(settings, size, slen, times, error, flops, verify != nullptr, settings.FlopsNeeded != nullptr);
        }
    }
    template <int TYPE=0> static inline void RunBenchmark(const std::string& name, const Settings& settings, const AcBCD test, const AcBCD verify=nullptr) {
        std::cout << "Benchmarking: \033[33m" + name + "\033[0m\n";

        const size_t start_size = settings.startSize;
        const size_t max_size = settings.endSize;
        const size_t runs = settings.runs;

        const int slen = NumDigits(max_size) * 2 + 2;

        auto times = std::vector<double>(runs, 0.0);
        auto error = std::vector<double>(runs, 0.0);
        auto flops = std::vector<double>(runs, 0.0);

        std::random_device rd;
        std::mt19937 gen(rd());

        size_t elements = 512;
        std::uniform_real_distribution<float> lrDist(0.001f, 0.1f);
        

        for (size_t size = start_size; size <= max_size; size *= 2) {
            auto a = Tensor<float>(size);
            auto b = Tensor<float>(size);
            auto c = Tensor<float>(size);
            auto d = Tensor<float>(size);
            auto check = Tensor<float>(size);

            // warmups
            size_t iteration = 1;
            test(a, b, c, d, iteration, elements, lrDist(gen));
            test(a, b, c, d, iteration, elements, lrDist(gen));

            for (size_t r = 0; r < runs; r++) {
                float lr = lrDist(gen);
                b.Randomize(-1.0f, 1.0f, 57932+r);
                c.Randomize(0.1f, 1.0f, 35534+r);
                d.Randomize(0.1f, 1.0f, 24731+r);
                a.Zero();

                auto start = std::chrono::steady_clock::now();
                test(a, b, c, d, iteration, elements, lr);
                auto end = std::chrono::steady_clock::now();
                times[r] = std::chrono::duration<double, std::milli>(end - start).count();

                if (verify) {
                    b.Randomize(-1.0f, 1.0f, 57932+r);
                    c.Randomize(0.1f, 1.0f, 35534+r);
                    d.Randomize(0.1f, 1.0f, 24731+r);
                    check.Zero();
                    verify(check, b, c, d, iteration, elements, lr);
                    error[r] = a.MeanPercentDiff(check);
                }

                iteration++;
            }

            OutputResults(settings, size, slen, times, error, flops, verify != nullptr, settings.FlopsNeeded != nullptr);
        }
    }
    template <int TYPE=0> static inline void RunBenchmark(const std::string& name, const Settings& settings, const aXpbY test, const aXpbY verify=nullptr) {
        std::cout << "Benchmarking: \033[33m" + name + "\033[0m\n";

        const size_t start_size = settings.startSize;
        const size_t max_size = settings.endSize;
        const size_t runs = settings.runs;

        const int slen = NumDigits(max_size) * 2 + 2;

        auto times = std::vector<double>(runs, 0.0);
        auto error = std::vector<double>(runs, 0.0);
        auto flops = std::vector<double>(runs, 0.0);

        for (size_t size = start_size; size <= max_size; size *= 2) {
            const float a = (double)size / (double)max_size;
            const float b = (double)size / (double)start_size;

            Tensor<float> x;
            Tensor<float> y;
            Tensor<float> check;

            if constexpr (TYPE == 0) {
                x = Tensor<float>(size);
                y = Tensor<float>(size);
                check = Tensor<float>(size);
            }

            for (size_t r = 0; r < runs; r++) {
                x.Randomize(-100.0f, 100.0f, 87741+r);
                y.Randomize(-100.0f, 100.0f, 64937+r);

                // warmups
                test(a, x, b, check);
                test(a, x, b, check);

                auto start = std::chrono::steady_clock::now();
                test(a, x, b, y);
                auto end = std::chrono::steady_clock::now();
                times[r] = std::chrono::duration<double, std::milli>(end - start).count();

                if (verify) {
                    check.Randomize(-100.0f, 100.0f, 64937+r);
                    verify(a, x, b, check);
                    error[r] = y.MeanPercentDiff(check);
                }

                if (settings.FlopsNeeded) {
                    flops[r] = settings.FlopsNeeded(size) / (times[r] / 1000.00);
                }
            }

            OutputResults(settings, size, slen, times, error, flops, verify != nullptr, settings.FlopsNeeded != nullptr);
        }

        std::cout << std::endl;
    }

    private:
    static inline void OutputResults(const Settings& settings, size_t size, size_t slen, const std::vector<double>& times, const std::vector<double>& error, const std::vector<double>& flops, bool verified, bool theoretical) {
        auto it = std::minmax_element(times.begin(), times.end()); 
        double best = *it.first; 
        double worst = *it.second;
        double sum = std::reduce(times.begin(), times.end(), 0.0, std::plus<double>());
        double mean = sum / (double)settings.runs;

        double meanPercentError = verified ? Mean(error) : 0.0;
        double meanFlops = theoretical ? Mean(flops) : 0.0;
        double sd = StdDev(times, mean);

        double CV = VariationCoefficient(sd, mean) * 100.0;
        double CI = CIHalfWidth(sd, times.size());
        double percentTheoretical = theoretical ? (Mean(flops) / settings.flops * 100.0) : 0.0;

        constexpr const size_t COL_BEST = 14;
        constexpr const size_t COL_WORST = 14;
        constexpr const size_t COL_MEAN = 14;
        constexpr const size_t COL_RUNS = 6;
        constexpr const size_t COL_MPE = 10;
        constexpr const size_t COL_CV = 14;
        constexpr const size_t COL_CI = 16;
        constexpr const size_t COL_PTH = 10;

        auto formatRaw = [](double v, int precision = 4) {
            if (!std::isfinite(v)) { return std::string("-"); }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(precision) << v;
            return oss.str();
        };

        auto padRaw = [](const std::string& s, int width, bool leftAlign = false) {
            if (s.size() >= width) { return s.substr(0, width); }
            if (leftAlign) {
                return s + std::string(width - s.size(), ' ');
            } else {
                return std::string(width - s.size(), ' ') + s;
            }
        };

        auto colorWrap = [](const std::string& paddedRaw, const char* color) {
            if (!color) { return paddedRaw; }
            return std::string(color) + paddedRaw + "\033[0m";
        };

        // raw fields
        std::string rawSize = std::to_string(size);
        std::string rawBest = formatRaw(best, 5) + "ms";
        std::string rawWorst = formatRaw(worst, 5) + "ms";
        std::string rawMean = formatRaw(mean, 5) + "ms";
        std::string rawRuns = std::to_string(settings.runs);
        std::string rawCV = formatRaw(CV, 2) + "%";
        std::string rawCI = "± " + formatRaw(CI, 3) + "ms";
        std::string rawMpe = verified ? (formatRaw(meanPercentError, 3) + "%") : "";
        std::string rawPth = theoretical ? (formatRaw(percentTheoretical, 2) + "%") : "";

        // pad all to width
        std::string paddedSize  = padRaw(rawSize,  slen,      true);
        std::string paddedRuns  = padRaw(rawRuns,  COL_RUNS,  true);
        std::string paddedBest  = padRaw(rawBest,  COL_BEST,  true);
        std::string paddedWorst = padRaw(rawWorst, COL_WORST, true);
        std::string paddedMean  = padRaw(rawMean,  COL_MEAN,  true);
        std::string paddedCV    = padRaw(rawCV,    COL_CV,    true);
        std::string paddedCI    = padRaw(rawCI,    COL_CI,    true);
        std::string paddedMpe   = padRaw(rawMpe,   COL_MPE,   true);
        std::string paddedPth   = padRaw(rawPth,   COL_PTH,   true);

        // apply colors
        std::string sizeCol  = colorWrap(paddedSize,  "\033[0m");
        std::string bestCol  = colorWrap(paddedBest,  "\033[32m");
        std::string worstCol = colorWrap(paddedWorst, "\033[31m");
        std::string meanCol  = colorWrap(paddedMean,  "\033[34m");
        std::string runsCol  = colorWrap(paddedRuns,  "\033[33m");
        std::string CVCol    = colorWrap(paddedCV,    "\033[33m");
        std::string CICol    = colorWrap(paddedCI,    "\033[33m");
        std::string mpeCol   = verified    ? colorWrap(paddedMpe, "\033[33m") : paddedMpe;
        std::string pthCol   = theoretical ? colorWrap(paddedPth, "\033[32m") : paddedPth;

        // output header once
        static bool printedHeader = false;
        if (!printedHeader) {
            std::ostringstream hdr;
            hdr << padRaw("size",  slen,      true) << " "
                << padRaw("runs",  COL_RUNS,  true) << " "
                << padRaw("best",  COL_BEST,  true) << " "
                << padRaw("worst", COL_WORST, true) << " "
                << padRaw("mean",  COL_MEAN,  true) << " "
                << padRaw("CV",    COL_CV,    true) << "  "
                << padRaw("CI95",  COL_CI,    true) << " "
                << padRaw("MPE",   COL_MPE,   true) << "  "
                << padRaw("PTH",   COL_PTH,   true) << "\n";
            std::cout << hdr.str();
            printedHeader = true;
        }

        std::ostringstream oss;
        oss << sizeCol  << " "
            << runsCol  << " "
            << bestCol  << " "
            << worstCol << " "
            << meanCol  << " "
            << CVCol    << "  "
            << CICol    << "  "
            << mpeCol   << "  "
            << pthCol   << "\n";

        std::cout << oss.str();
    }
    static constexpr inline int NumDigits(size_t n) {
        int digits = 0;
        do {
            ++digits;
            n /= 10;
        } while (n != 0);
        return digits;
    }

    static inline double Mean(const std::vector<double>& t) {
        return std::reduce(std::execution::unseq, t.begin(), t.end(), 0.00, std::plus<double>()) / t.size();
    }
    static inline double StdDev(const std::vector<double>& t, double mu) {
        double s = std::transform_reduce(std::execution::unseq, t.begin(), t.end(), 0.0, std::plus<double>(), [&](double x){ return (x-mu) *(x-mu); });
        return std::sqrt(s / (t.size()-1));
    }
    static inline double VariationCoefficient(double sd, double mu) {
        return sd / mu;
    }
    static inline double CIHalfWidth(double sd, size_t n) {
        double t = 1.96;
        return t * sd / std::sqrt((double)n);
    }
    static inline double RCIW(double upper, double lower, double mean) {
        return (upper - lower) / (2.0*mean);
    }
};
