// netkit MNIST CNN invoke-time benchmark (pairs with benchmark/tflm/).

#include "arena.hpp"
#include "arena_util.hpp"
#include "benchmark_stats.hpp"
#include "cnn.hpp"
#include "mnist_cnn_test_images.h"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <span>

#ifndef NETKIT_BENCH_BACKEND
#define NETKIT_BENCH_BACKEND "reference"
#endif

namespace {

constexpr const char* kDefaultModelPath = "models/mnist_cnn.nk";
constexpr size_t kArenaCapacity = 4 * 1024 * 1024;
constexpr int kRuns = BenchmarkStats::kCnnDefaultRuns;

Tensor MakeNhwcView(float* data, uint32_t h, uint32_t w, uint32_t c)
{
    const std::array<uint32_t, 3> shape{h, w, c};
    return TensorFactory::ViewND(data, 3, std::span<const uint32_t>(shape));
}

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

int RunBenchmark(const char* model_path)
{
    NkLoader::ParsedModel parsed{};
    const NkLoader::LoadResult parse_result = NkLoader::ParseFile(model_path, parsed);
    if (parse_result.status != NkLoader::LoadStatus::Ok)
    {
        std::fprintf(stderr, "parse failed: %s\n",
                     parse_result.message ? parse_result.message
                                          : NkLoader::StatusMessage(parse_result.status));
        return 1;
    }
    if (parsed.header.network_kind != NkFormat::NetworkKind::Cnn)
    {
        std::fprintf(stderr, "expected CNN model\n");
        return 1;
    }

    alignas(std::max_align_t) static unsigned char arena_memory[kArenaCapacity];
    ArenaUtil::Scoped arena_scope(kArenaCapacity, arena_memory);
    if (!arena_scope)
    {
        std::fprintf(stderr, "arena init failed\n");
        return 1;
    }
    Arena& arena = arena_scope.Get();

    CNNNetwork* network = nullptr;
    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    const NkLoader::LoadResult load_result =
        NkLoader::LoadCNN(model_path, arena, network, input_shape, input_rank);
    if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        std::fprintf(stderr, "load failed: %s\n",
                     load_result.message ? load_result.message
                                         : NkLoader::StatusMessage(load_result.status));
        return 1;
    }

    if (input_rank != 3 || input_shape[0] != 28 || input_shape[1] != 28 || input_shape[2] != 1)
    {
        std::fprintf(stderr, "unexpected CNN input shape\n");
        return 1;
    }

    alignas(16) float input_buffer[kMnistCnnBenchmarkInputSize] = {};

    std::printf("netkit MNIST CNN benchmark\n");
    std::printf("  backend:     %s\n", NETKIT_BENCH_BACKEND);
    std::printf("  model:       %s\n", model_path);
    std::printf("  images:      %d per run\n", kMnistCnnBenchmarkImageCount);
    std::printf("  runs:        %d (discard first invoke each run)\n", kRuns);
    std::printf("  arena bytes: %zu\n", kArenaCapacity);

    std::array<double, BenchmarkStats::kMaxRuns> run_averages_us{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;

        for (int i = 0; i < kMnistCnnBenchmarkImageCount; ++i)
        {
            const MnistCnnBenchmarkSample& sample = kMnistCnnBenchmarkImages[i];
            std::memcpy(input_buffer, sample.pixels, kMnistCnnBenchmarkInputSize * sizeof(float));

            Tensor input = MakeNhwcView(input_buffer, input_shape[0], input_shape[1], input_shape[2]);

            const auto start = std::chrono::steady_clock::now();
            Tensor& output = network->forward(input, arena);
            const auto end = std::chrono::steady_clock::now();

            if (!output.data)
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
                const float* output_data = static_cast<const float*>(output.data);
                const int predicted = ArgMax10(output_data);
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
    const char* model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    return RunBenchmark(model_path);
}
