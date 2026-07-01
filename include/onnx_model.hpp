#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace Onnx
{
    constexpr std::size_t kMaxNameLen = 128;
    constexpr std::size_t kMaxNodes = 32;
    constexpr std::size_t kMaxInitializers = 64;
    constexpr std::size_t kMaxDims = 8;
    constexpr std::size_t kMaxAttributes = 16;
    constexpr std::size_t kMaxNodeInputs = 8;
    constexpr std::size_t kMaxNodeOutputs = 4;
    constexpr std::size_t kMaxGraphInputs = 4;
    constexpr std::size_t kMaxGraphOutputs = 4;

    struct Tensor
    {
        char name[kMaxNameLen]{};
        std::array<int64_t, kMaxDims> dims{};
        uint32_t dim_count = 0;
        int32_t data_type = 0;
        std::vector<float> float_data;
    };

    struct Attribute
    {
        char name[kMaxNameLen]{};
        int32_t type = 0;
        int64_t i = 0;
        float f = 0.0f;
        std::vector<int64_t> ints;
        std::vector<float> floats;
    };

    struct Node
    {
        char op_type[64]{};
        std::array<char[kMaxNameLen], kMaxNodeInputs> inputs{};
        uint32_t input_count = 0;
        std::array<char[kMaxNameLen], kMaxNodeOutputs> outputs{};
        uint32_t output_count = 0;
        std::array<Attribute, kMaxAttributes> attributes{};
        uint32_t attribute_count = 0;
    };

    struct ValueInfo
    {
        char name[kMaxNameLen]{};
        std::array<int64_t, kMaxDims> dims{};
        uint32_t dim_count = 0;
    };

    struct Graph
    {
        std::array<Node, kMaxNodes> nodes{};
        uint32_t node_count = 0;
        std::array<Tensor, kMaxInitializers> initializers{};
        uint32_t initializer_count = 0;
        std::array<ValueInfo, kMaxGraphInputs> inputs{};
        uint32_t input_count = 0;
        std::array<ValueInfo, kMaxGraphOutputs> outputs{};
        uint32_t output_count = 0;
    };

    struct Model
    {
        Graph graph{};
    };

    enum class ParseStatus
    {
        Ok,
        FileOpenFailed,
        FileTooLarge,
        InvalidProtobuf,
        UnsupportedFeature
    };

    ParseStatus ParseModelFile(const char* path, Model& model, char* error, std::size_t error_capacity);
    ParseStatus ParseModelBytes(std::span<const uint8_t> bytes, Model& model, char* error, std::size_t error_capacity);

    const Tensor* FindInitializer(const Graph& graph, const char* name);
    const Attribute* FindAttribute(const Node& node, const char* name);
    int64_t AttributeInt(const Node& node, const char* name, int64_t default_value);
    bool AttributeInts(const Node& node, const char* name, std::span<const int64_t>& out);
}
