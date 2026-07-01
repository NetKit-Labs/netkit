#pragma once
#include "arena.hpp"
#include "cnn.hpp"
#include "json_parser.hpp"
#include "mlp.hpp"
#include "tensor.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace ModelLoader
{
    constexpr std::size_t kMaxJsonBytes = 8192;
    constexpr std::size_t kMaxPathLen = 256;
    constexpr uint32_t kMaxLayers = 16;

    enum class NetworkKind
    {
        Unknown,
        MLP,
        CNN
    };

    enum class LoadStatus
    {
        Ok,
        JsonOpenFailed,
        BinOpenFailed,
        JsonParseFailed,
        UnsupportedNetwork,
        VersionMismatch,
        LayerConfigError,
        BinSizeMismatch,
        ArenaOverflow
    };

    struct LoadResult
    {
        LoadStatus status = LoadStatus::Ok;
        NetworkKind kind = NetworkKind::Unknown;
        const char* message = nullptr;
    };

    struct DenseLayerConfig
    {
        uint32_t units = 0;
        char activation[Json::kMaxStringLen] = "none";
        float alpha = 0.01f;
    };

    struct ConvLayerConfig
    {
        uint32_t kernel_size = 0;
        uint32_t stride = 1;
        uint32_t filters = 0;
        char activation[Json::kMaxStringLen] = "none";
        float alpha = 0.01f;
    };

    enum class CnnLayerKind : uint8_t
    {
        Conv2D,
        MaxPool2D,
        Flatten,
        Dense
    };

    struct PoolLayerConfig
    {
        uint32_t pool_size = 2;
        uint32_t stride = 2;
    };

    struct ArchitectureSpec
    {
        uint32_t version = 0;
        NetworkKind kind = NetworkKind::Unknown;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;
        uint32_t num_layers = 0;
        DenseLayerConfig dense_layers[kMaxLayers]{};
        ConvLayerConfig conv_layers[kMaxLayers]{};
        CnnLayerKind cnn_layer_kinds[kMaxLayers]{};
        PoolLayerConfig pool_layers[kMaxLayers]{};
        DenseLayerConfig cnn_dense_layers[kMaxLayers]{};
        std::size_t expected_weight_floats = 0;
    };

    // Parse JSON architecture only (no .bin load, no network construction).
    LoadResult ParseArchitecture(const char* json_path, ArchitectureSpec& spec);

    uint32_t ComputeMlpOutputElements(const ArchitectureSpec& spec);
    uint32_t ComputeCnnOutputElements(const ArchitectureSpec& spec);

    void PrintArchitecture(const ArchitectureSpec& spec);

    void PrintWeightsSummary(const char* json_path,
                             float* weights,
                             std::size_t float_count,
                             std::size_t expected_float_count);

    // model.json -> model.bin (same path, extension swapped)
    bool JsonPathToBinPath(const char* json_path, char* bin_path, std::size_t bin_path_capacity);

    // Load float32 weights from the companion .bin file into arena memory.
    LoadStatus LoadWeightsBin(const char* json_path,
                              Arena& arena,
                              float*& weights,
                              std::size_t& float_count,
                              const char** error_message = nullptr);

    LoadResult LoadMLP(const char* json_path,
                       Arena& arena,
                       MLPNetwork*& network,
                       std::array<uint32_t, kMaxTensorRank>& input_shape,
                       uint32_t& input_rank);

    LoadResult LoadCNN(const char* json_path,
                       Arena& arena,
                       CNNNetwork*& network,
                       std::array<uint32_t, kMaxTensorRank>& input_shape,
                       uint32_t& input_rank);

    // Detects network type from JSON and loads the matching network.
    LoadResult Load(const char* json_path,
                    Arena& arena,
                    NetworkKind& kind,
                    MLPNetwork*& mlp,
                    CNNNetwork*& cnn,
                    std::array<uint32_t, kMaxTensorRank>& input_shape,
                    uint32_t& input_rank);
}
