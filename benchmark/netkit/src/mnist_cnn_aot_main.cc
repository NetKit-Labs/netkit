// netkit MNIST CNN AOT benchmark — lowered static Kernels:: call chain.

#include "benchmark_stats.hpp"
#include "generated/aot/mnist_cnn_aot.hpp"
#include "mnist_cnn_test_images.h"

#include "arena.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>

#ifndef NETKIT_BENCH_BACKEND
#define NETKIT_BENCH_BACKEND "aot"
#endif

namespace aot = netkit::aot::mnist_cnn;

namespace {

constexpr int kRuns = BenchmarkStats::kDefaultRuns;

int ArgMax10(const float* values)
{
    int best = 0;
    float max_val = values[0];
    for (int i = 1; i < 10; ++i)
    {
        if (values[i] > max_val)
        {
            max_val = values[i];
            best = i;
        }
    }
    return best;
}

int RunBenchmark()
{
    constexpr std::size_t kArenaCapacity =
        std::max<std::size_t>(aot::kArenaBytesRecommended, 4u * 1024u * 1024u);
    alignas(std::max_align_t) static unsigned char arena_memory[kArenaCapacity];
    Arena arena;
    if (!aot::InitArena(arena, arena_memory, sizeof(arena_memory)))
    {
        std::fprintf(stderr, "arena init failed\n");
        return 1;
    }

    aot::Model model;
    if (!model.load(arena))
    {
        std::fprintf(stderr, "AOT model load failed\n");
        return 1;
    }

    std::printf("netkit MNIST CNN benchmark\n");
    std::printf("  backend:     %s\n", NETKIT_BENCH_BACKEND);
    std::printf("  model:       mnist_cnn (lowered AOT)\n");
    std::printf("  images:      %d per run\n", kMnistCnnBenchmarkImageCount);
    std::printf("  runs:        %d (discard first invoke each run)\n", kRuns);
    std::printf("  arena bytes: %zu (recommended %zu)\n", kArenaCapacity,
                static_cast<std::size_t>(aot::kArenaBytesRecommended));

    std::array<float, aot::kOutputElements> output{};
    std::array<double, BenchmarkStats::kMaxRuns> run_averages_us{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;

        for (int i = 0; i < kMnistCnnBenchmarkImageCount; ++i)
        {
            const MnistCnnBenchmarkSample& sample = kMnistCnnBenchmarkImages[i];

            const auto start = std::chrono::steady_clock::now();
            const bool ok = model.forward(arena, sample.pixels, output.data());
            const auto end = std::chrono::steady_clock::now();

            if (!ok)
            {
                std::fprintf(stderr, "forward failed on run %d image %d (%s)\n", run + 1, i,
                             sample.name);
                return 1;
            }

            const double elapsed_us =
                std::chrono::duration<double, std::micro>(end - start).count();

            if (i > 0)
            {
                run_total_us += elapsed_us;
                ++counted;
            }

            if (run == kRuns - 1)
            {
                const int predicted = ArgMax10(output.data());
                if (predicted == sample.label)
                {
                    ++correct;
                }
            }
        }

        run_averages_us[static_cast<size_t>(run)] =
            run_total_us / static_cast<double>(counted);
    }

    std::printf("  accuracy:    %d/%d on final run\n", correct, kMnistCnnBenchmarkImageCount);
    BenchmarkStats::PrintSummary("netkit", "cnn", NETKIT_BENCH_BACKEND,
                                 BenchmarkStats::Compute(run_averages_us.data(), kRuns));

    return correct == kMnistCnnBenchmarkImageCount ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;
    return RunBenchmark();
}
