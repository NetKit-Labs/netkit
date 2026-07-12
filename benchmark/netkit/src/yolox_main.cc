// netkit YOLOX MNv4-PAFPN float32 host benchmark (320x320 detect).
//
// Loads models/yolox_mnv4_pafpn_trained.nk. Times network->forward() only
// (no decode/NMS). Methodology matches ImageNet MNv4: 10 images x 5 loops,
// warm_mean discards the entire first image pass.

#include "arena.hpp"
#include "arena_util.hpp"
#include "cnn.hpp"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"
#include "yolox_test_images.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

#ifndef NETKIT_BENCH_BACKEND
#define NETKIT_BENCH_BACKEND "reference"
#endif

namespace {

constexpr const char* kDefaultModelPath = "models/yolox_mnv4_pafpn_trained.nk";
constexpr int kLoops = 5;
constexpr uint32_t kInH = kYoloxBenchmarkHeight;
constexpr uint32_t kInW = kYoloxBenchmarkWidth;
constexpr uint32_t kInC = kYoloxBenchmarkChannels;
// 320² YOLOX + PAFPN + tap buffers need a large host arena. 256 MiB matches the
// ImageNet host bench floor; avoid multi-GiB calloc (can fail on constrained hosts).
constexpr size_t kArenaCapacity = 256ull * 1024 * 1024;

Tensor MakeNhwcView(float* data, uint32_t h, uint32_t w, uint32_t c)
{
    const std::array<uint32_t, 3> shape{h, w, c};
    return TensorFactory::ViewND(data, 3, std::span<const uint32_t>(shape));
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

#if defined(NETKIT_ARENA_HEAP)
    ArenaUtil::Scoped arena_scope(kArenaCapacity, nullptr);
#else
    static unsigned char* arena_memory = new unsigned char[kArenaCapacity];
    ArenaUtil::Scoped arena_scope(kArenaCapacity, arena_memory);
#endif
    if (!arena_scope)
    {
        std::fprintf(stderr, "arena init failed (requested %zu bytes)\n", kArenaCapacity);
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

    if (input_rank != 3 || input_shape[0] != kInH || input_shape[1] != kInW ||
        input_shape[2] != kInC)
    {
        std::fprintf(stderr, "unexpected input shape [%u,%u,%u], expected [%u,%u,%u]\n",
                     input_shape[0], input_shape[1], input_shape[2], kInH, kInW, kInC);
        return 1;
    }

    if (network->IsQuantized())
    {
        std::fprintf(stderr, "quantized YOLOX not supported in this float32 bench\n");
        return 1;
    }

    const uint32_t output_cols = NkLoader::OutputElements(parsed);
    if (output_cols != static_cast<uint32_t>(kYoloxBenchmarkOutputElems))
    {
        std::fprintf(stderr, "unexpected output count %u (want %d)\n", output_cols,
                     kYoloxBenchmarkOutputElems);
        return 1;
    }

    const char* dtype = "float32";
    const int num_images = kYoloxBenchmarkImageCount;
    std::printf("netkit YOLOX MNv4-PAFPN float32 benchmark\n");
    std::printf("  backend:     %s\n", NETKIT_BENCH_BACKEND);
    std::printf("  dtype:       %s\n", dtype);
    std::printf("  model:       %s\n", model_path);
    std::printf("  input:       %ux%ux%u  outputs: %u\n", kInH, kInW, kInC, output_cols);
    std::printf("  arena:       %zu bytes\n", kArenaCapacity);
    std::printf("  method:      %d images x %d loops = %d invokes (all timed)\n", num_images,
                kLoops, num_images * kLoops);
    std::printf("  note:        times forward() only (no decode/NMS)\n");

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(num_images * kLoops));

    for (int loop = 0; loop < kLoops; ++loop)
    {
        for (int i = 0; i < num_images; ++i)
        {
            std::vector<float> input_buf(
                kYoloxBenchmarkImages[i].pixels,
                kYoloxBenchmarkImages[i].pixels + kYoloxBenchmarkInputSize);
            Tensor input = MakeNhwcView(input_buf.data(), kInH, kInW, kInC);

            const auto start = std::chrono::steady_clock::now();
            Tensor& output = network->forward(input, arena);
            const auto end = std::chrono::steady_clock::now();

            if (!output.data)
            {
                std::fprintf(stderr, "forward failed on loop %d image %d\n", loop, i);
                return 1;
            }
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
            if (loop == 0)
            {
                std::printf("  image %d %-28s out_elems=%u\n", i,
                            kYoloxBenchmarkImages[i].name, output.num_elements);
            }
        }
    }

    const double cold_us = samples.front();
    double first_pass_sum = 0.0;
    for (int i = 0; i < num_images; ++i)
        first_pass_sum += samples[static_cast<size_t>(i)];
    const double first_pass_mean = first_pass_sum / static_cast<double>(num_images);

    const size_t warm_begin = static_cast<size_t>(num_images);
    const size_t warm_n = samples.size() - warm_begin;
    double warm_sum = 0.0;
    double warm_min = samples[warm_begin];
    double warm_max = samples[warm_begin];
    for (size_t k = warm_begin; k < samples.size(); ++k)
    {
        warm_sum += samples[k];
        if (samples[k] < warm_min)
            warm_min = samples[k];
        if (samples[k] > warm_max)
            warm_max = samples[k];
    }
    const double warm_mean = warm_sum / static_cast<double>(warm_n);
    double var = 0.0;
    for (size_t k = warm_begin; k < samples.size(); ++k)
    {
        const double d = samples[k] - warm_mean;
        var += d * d;
    }
    const double warm_std = std::sqrt(var / static_cast<double>(warm_n));
    std::vector<double> warm_sorted(samples.begin() + static_cast<std::ptrdiff_t>(warm_begin),
                                    samples.end());
    std::sort(warm_sorted.begin(), warm_sorted.end());
    const double warm_median = warm_sorted[warm_sorted.size() / 2];

    std::printf("\n");
    std::printf("netkit YOLOX MNv4-PAFPN summary (%s)\n", NETKIT_BENCH_BACKEND);
    std::printf("  10-image mean:    %9.3f us (%7.3f ms)\n", first_pass_mean,
                first_pass_mean / 1000.0);
    std::printf("  cold invoke:      %9.3f us (%7.3f ms)\n", cold_us, cold_us / 1000.0);
    std::printf("  warm median:      %9.3f us (%7.3f ms)\n", warm_median, warm_median / 1000.0);
    std::printf("  warm mean:        %9.3f us (%7.3f ms)  over %zu invokes\n", warm_mean,
                warm_mean / 1000.0, warm_n);
    std::printf("  warm min/max:     %9.3f / %.3f us\n", warm_min, warm_max);
    std::printf("  warm stddev:      %9.3f us\n", warm_std);
    std::printf(
        "BENCHMARK_SUMMARY runtime=netkit model=yolox_mnv4_pafpn dtype=%s backend=%s "
        "ten_image_mean_us=%.3f warm_median_us=%.3f warm_mean_us=%.3f cold_us=%.3f "
        "invokes=%zu\n",
        dtype, NETKIT_BENCH_BACKEND, first_pass_mean, warm_median, warm_mean, cold_us,
        samples.size());

    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    return RunBenchmark(model_path);
}
