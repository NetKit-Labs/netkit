// netkit MNIST CNN int8 host benchmark (pairs with benchmark/tflite/mnist_cnn_int8_bench.py).
//
// Loads models/mnist_cnn_int8.nk. Feeds prequantized int8 digits (Python export).
// Methodology matches MCU peer: 10 runs x 10 images, discard first invoke each run.

#include "arena.hpp"
#include "arena_util.hpp"
#include "benchmark_stats.hpp"
#include "cnn.hpp"
#include "cmsis_dsp_util.hpp"
#include "mnist_cnn_int8_test_images.h"
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

constexpr const char* kDefaultModelPath = "models/mnist_cnn_int8.nk";
constexpr size_t kArenaCapacity = 4 * 1024 * 1024;
constexpr int kRuns = BenchmarkStats::kCnnDefaultRuns;

Tensor MakeNhwcViewInt8(int8_t* data, uint32_t h, uint32_t w, uint32_t c)
{
    Tensor input{};
    input.data = data;
    input.type = DataType::Int8;
    input.rank = 3;
    input.shape[0] = h;
    input.shape[1] = w;
    input.shape[2] = c;
    input.stride[0] = w * c;
    input.stride[1] = c;
    input.stride[2] = 1;
    input.num_elements = h * w * c;
    input.bytes = input.num_elements * sizeof(int8_t);
    return input;
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
    if (!network->IsQuantized())
    {
        std::fprintf(stderr, "expected quantized int8 model\n");
        return 1;
    }
    if (input_rank != 3 || input_shape[0] != 28 || input_shape[1] != 28 || input_shape[2] != 1)
    {
        std::fprintf(stderr, "unexpected CNN input shape\n");
        return 1;
    }

    alignas(16) int8_t input_buffer[kMnistCnnInt8BenchmarkInputSize] = {};

    std::printf("netkit MNIST CNN int8 benchmark\n");
    std::printf("  backend:     %s\n", NETKIT_BENCH_BACKEND);
    std::printf("  dtype:       int8\n");
    std::printf("  model:       %s\n", model_path);
    std::printf("  images:      %d per run\n", kMnistCnnInt8BenchmarkImageCount);
    std::printf("  runs:        %d (discard first invoke each run)\n", kRuns);
    std::printf("  arena bytes: %zu\n", kArenaCapacity);

    std::array<double, BenchmarkStats::kMaxRuns> run_averages_us{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;

        for (int i = 0; i < kMnistCnnInt8BenchmarkImageCount; ++i)
        {
            const MnistCnnInt8BenchmarkSample& sample = kMnistCnnInt8BenchmarkImages[i];
            std::memcpy(input_buffer, sample.pixels,
                        static_cast<size_t>(kMnistCnnInt8BenchmarkInputSize));

            Tensor input =
                MakeNhwcViewInt8(input_buffer, input_shape[0], input_shape[1], input_shape[2]);

            const auto start = std::chrono::steady_clock::now();
            Tensor& output = network->forward(input, arena);
            const auto end = std::chrono::steady_clock::now();

            if (!output.data || output.type != DataType::Int8)
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
                const int predicted = static_cast<int>(CmsisQuantUtil::ArgMaxInt8(
                    static_cast<const int8_t*>(output.data), 10u));
                if (predicted == sample.label)
                {
                    ++correct;
                }
                std::printf("  image %d label=%d pred=%d %s\n", i, sample.label, predicted,
                            predicted == sample.label ? "OK" : "MISS");
            }
        }

        run_averages_us[static_cast<size_t>(run)] =
            run_total_us / static_cast<double>(counted);
    }

    std::printf("  accuracy:    %d/%d on final run\n", correct, kMnistCnnInt8BenchmarkImageCount);
    BenchmarkStats::PrintSummary("netkit", "cnn_int8", NETKIT_BENCH_BACKEND,
                                 BenchmarkStats::Compute(run_averages_us.data(), kRuns));

    return correct == kMnistCnnInt8BenchmarkImageCount ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    return RunBenchmark(model_path);
}
