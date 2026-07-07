// Shared timing summary for netkit / TFLM MNIST benchmarks.
#pragma once

#include <cstdio>

namespace BenchmarkStats {

constexpr int kDefaultRuns = 100;
constexpr int kCnnDefaultRuns = 10;
constexpr int kMaxRuns = 100;

struct Summary {
    int num_runs = 0;
    double mean_us = 0.0;
};

inline Summary Compute(const double* run_averages_us, int num_runs)
{
    Summary summary{};
    summary.num_runs = num_runs;
    if (num_runs <= 0)
    {
        return summary;
    }

    double total = 0.0;
    for (int i = 0; i < num_runs; ++i)
    {
        total += run_averages_us[i];
    }
    summary.mean_us = total / static_cast<double>(num_runs);

    return summary;
}

inline void PrintSummary(const char* runtime, const char* model, const char* backend,
                         const Summary& summary)
{
    std::printf("\n");
    std::printf("%s MNIST %s benchmark summary (%s)\n", runtime, model, backend);
    std::printf("  method:      %d runs x 10 images, discard first invoke each run\n",
                summary.num_runs);
    std::printf("  per-run avg: avg of images 1-9 (us)\n");
    std::printf("\n");
    std::printf("  mean:   %8.3f us (%6.3f ms)\n", summary.mean_us, summary.mean_us / 1000.0);

    std::printf("BENCHMARK_SUMMARY runtime=%s model=%s backend=%s mean_us=%.3f runs=%d\n",
                runtime, model, backend, summary.mean_us, summary.num_runs);
}

}  // namespace BenchmarkStats
