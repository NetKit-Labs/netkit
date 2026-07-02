#pragma once

#include "tensor.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace OnnxGraph
{
    constexpr std::size_t kMaxStringLen = 64;
    constexpr uint32_t kMaxLayers = 16;

    enum class NetworkKind
    {
        Unknown,
        MLP,
        CNN
    };

    enum class CnnLayerKind : uint8_t
    {
        Conv2D,
        MaxPool2D,
        Flatten,
        Dense
    };

    struct DenseLayerConfig
    {
        uint32_t units = 0;
        char activation[kMaxStringLen] = "none";
        float alpha = 0.01f;
    };

    struct ConvLayerConfig
    {
        uint32_t kernel_size = 0;
        uint32_t stride = 1;
        uint32_t filters = 0;
        char activation[kMaxStringLen] = "none";
        float alpha = 0.01f;
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
}
