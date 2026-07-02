#include "onnx_importer.hpp"
#include "onnx_graph.hpp"
#include "onnx_model.hpp"
#include "tensor_factory.hpp"

#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace OnnxImporter
{
    namespace
    {
        struct ConvertedModel
        {
            OnnxGraph::ArchitectureSpec spec{};
            std::vector<float> weights;
            char error[256]{};
        };

        void SetError(ConvertedModel& out, const char* message)
        {
            std::strncpy(out.error, message, sizeof(out.error) - 1);
            out.error[sizeof(out.error) - 1] = '\0';
        }

        bool IsOp(const Onnx::Node& node, const char* op)
        {
            return std::strcmp(node.op_type, op) == 0;
        }

        bool NextIsRelu(const Onnx::Graph& graph, uint32_t index)
        {
            return index + 1 < graph.node_count && IsOp(graph.nodes[index + 1], "Relu");
        }

        bool NextIsSoftmax(const Onnx::Graph& graph, uint32_t index)
        {
            return index + 1 < graph.node_count && IsOp(graph.nodes[index + 1], "Softmax");
        }

        bool SquareKernelFromAttribute(std::span<const int64_t> kernel_shape,
                                       uint32_t& kernel,
                                       ConvertedModel& out,
                                       const char* op_name)
        {
            if (kernel_shape.empty())
            {
                char message[128];
                std::snprintf(message, sizeof(message), "%s missing kernel_shape", op_name);
                SetError(out, message);
                return false;
            }

            if (kernel_shape.size() == 1)
            {
                kernel = static_cast<uint32_t>(kernel_shape[0]);
                return kernel > 0;
            }

            if (kernel_shape.size() == 2 && kernel_shape[0] == kernel_shape[1] && kernel_shape[0] > 0)
            {
                kernel = static_cast<uint32_t>(kernel_shape[0]);
                return true;
            }

            char message[128];
            std::snprintf(message, sizeof(message), "%s requires square kernel_shape attribute", op_name);
            SetError(out, message);
            return false;
        }

        uint32_t FirstStride(std::span<const int64_t> strides, uint32_t default_stride)
        {
            if (strides.empty())
                return default_stride;
            return static_cast<uint32_t>(strides[0]);
        }

        bool TensorMatchesElements(const Onnx::Tensor& tensor, std::size_t expected)
        {
            return tensor.float_data.size() == expected;
        }

        void CopyGemmWeight(const Onnx::Tensor& weight,
                            bool trans_b,
                            uint32_t in_features,
                            uint32_t out_features,
                            std::vector<float>& out)
        {
            const float* src = weight.float_data.data();
            out.resize(out.size() + static_cast<std::size_t>(in_features) * out_features);
            float* dst = out.data() + out.size() - static_cast<std::size_t>(in_features) * out_features;

            if (!trans_b)
            {
                std::memcpy(dst, src, static_cast<std::size_t>(in_features) * out_features * sizeof(float));
                return;
            }

            for (uint32_t o = 0; o < out_features; ++o)
            {
                for (uint32_t i = 0; i < in_features; ++i)
                    dst[static_cast<std::size_t>(i) * out_features + o] =
                        src[static_cast<std::size_t>(o) * in_features + i];
            }
        }

        void CopyBias(const Onnx::Tensor& bias, uint32_t out_features, std::vector<float>& out)
        {
            out.insert(out.end(), bias.float_data.begin(), bias.float_data.begin() + out_features);
        }

        void CopyConvWeightNchwToNetkit(const Onnx::Tensor& weight,
                                        uint32_t out_channels,
                                        uint32_t in_channels,
                                        uint32_t kernel,
                                        std::vector<float>& out)
        {
            const float* src = weight.float_data.data();
            const std::size_t block = static_cast<std::size_t>(out_channels) * kernel * kernel * in_channels;
            out.resize(out.size() + block);
            float* dst = out.data() + out.size() - block;

            for (uint32_t oc = 0; oc < out_channels; ++oc)
            {
                for (uint32_t ic = 0; ic < in_channels; ++ic)
                {
                    for (uint32_t kh = 0; kh < kernel; ++kh)
                    {
                        for (uint32_t kw = 0; kw < kernel; ++kw)
                        {
                            const std::size_t src_idx =
                                (((static_cast<std::size_t>(oc) * in_channels + ic) * kernel + kh) * kernel) + kw;
                            const std::size_t dst_idx =
                                (((static_cast<std::size_t>(oc) * kernel + kh) * kernel + kw) * in_channels) + ic;
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
        }

        bool ResolveInputShape(const Onnx::Graph& graph, ConvertedModel& out)
        {
            if (graph.input_count == 0)
            {
                SetError(out, "ONNX graph has no declared inputs");
                return false;
            }

            const Onnx::ValueInfo& input = graph.inputs[0];
            if (input.dim_count == 2)
            {
                out.spec.kind = OnnxGraph::NetworkKind::MLP;
                out.spec.input_rank = 2;
                out.spec.input_shape[0] = static_cast<uint32_t>(input.dims[0] > 0 ? input.dims[0] : 1);
                out.spec.input_shape[1] = static_cast<uint32_t>(input.dims[1]);
                return out.spec.input_shape[1] > 0;
            }

            if (input.dim_count == 4)
            {
                out.spec.kind = OnnxGraph::NetworkKind::CNN;
                out.spec.input_rank = 3;
                const uint32_t height = static_cast<uint32_t>(input.dims[2]);
                const uint32_t width = static_cast<uint32_t>(input.dims[3]);
                const uint32_t channels = static_cast<uint32_t>(input.dims[1]);
                out.spec.input_shape[0] = height;
                out.spec.input_shape[1] = width;
                out.spec.input_shape[2] = channels;
                return height > 0 && width > 0 && channels > 0;
            }

            if (input.dim_count == 3)
            {
                out.spec.kind = OnnxGraph::NetworkKind::CNN;
                out.spec.input_rank = 3;
                out.spec.input_shape[0] = static_cast<uint32_t>(input.dims[0]);
                out.spec.input_shape[1] = static_cast<uint32_t>(input.dims[1]);
                out.spec.input_shape[2] = static_cast<uint32_t>(input.dims[2]);
                return true;
            }

            SetError(out, "Unsupported ONNX input rank (expected 2 for MLP or 3/4 for CNN NHWC/NCHW)");
            return false;
        }

        bool AppendDense(ConvertedModel& out,
                         uint32_t out_features,
                         const char* activation,
                         uint32_t in_features,
                         const Onnx::Tensor& weight,
                         const Onnx::Tensor* bias,
                         bool trans_b)
        {
            if (out.spec.num_layers >= OnnxGraph::kMaxLayers)
            {
                SetError(out, "Too many ONNX layers for netkit");
                return false;
            }

            if (!TensorMatchesElements(weight, static_cast<std::size_t>(in_features) * out_features))
            {
                SetError(out, "Gemm/MatMul weight shape mismatch");
                return false;
            }

            if (bias && !TensorMatchesElements(*bias, out_features))
            {
                SetError(out, "Gemm/MatMul bias shape mismatch");
                return false;
            }

            const uint32_t idx = out.spec.num_layers;
            if (out.spec.kind == OnnxGraph::NetworkKind::MLP)
            {
                out.spec.dense_layers[idx].units = out_features;
                std::strncpy(out.spec.dense_layers[idx].activation, activation, OnnxGraph::kMaxStringLen - 1);
            }
            else
            {
                out.spec.cnn_layer_kinds[idx] = OnnxGraph::CnnLayerKind::Dense;
                out.spec.cnn_dense_layers[idx].units = out_features;
                std::strncpy(out.spec.cnn_dense_layers[idx].activation, activation, OnnxGraph::kMaxStringLen - 1);
            }

            CopyGemmWeight(weight, trans_b, in_features, out_features, out.weights);
            if (bias)
                CopyBias(*bias, out_features, out.weights);
            else
                out.weights.insert(out.weights.end(), out_features, 0.0f);

            ++out.spec.num_layers;
            return true;
        }

        bool ConvertGraph(const Onnx::Model& model, ConvertedModel& out)
        {
            const Onnx::Graph& graph = model.graph;
            if (!ResolveInputShape(graph, out))
                return false;

            uint32_t in_features = out.spec.input_rank == 2 ? out.spec.input_shape[1] : 0;
            uint32_t spatial_h = out.spec.input_rank == 3 ? out.spec.input_shape[0] : 0;
            uint32_t spatial_w = out.spec.input_rank == 3 ? out.spec.input_shape[1] : 0;
            uint32_t channels = out.spec.input_rank == 3 ? out.spec.input_shape[2] : 0;
            uint32_t dense_in = 0;

            for (uint32_t i = 0; i < graph.node_count;)
            {
                const Onnx::Node& node = graph.nodes[i];
                const char* next_activation = "none";
                uint32_t skip = 0;
                if (NextIsRelu(graph, i))
                {
                    next_activation = "relu";
                    skip = 1;
                }
                else if (NextIsSoftmax(graph, i))
                {
                    next_activation = "softmax";
                    skip = 1;
                }

                if (IsOp(node, "Relu") || IsOp(node, "Softmax"))
                {
                    ++i;
                    continue;
                }

                if (IsOp(node, "Gemm"))
                {
                    if (node.input_count < 2)
                    {
                        SetError(out, "Gemm node missing weights");
                        return false;
                    }

                    const Onnx::Tensor* weight = Onnx::FindInitializer(graph, node.inputs[1]);
                    const Onnx::Tensor* bias =
                        node.input_count >= 3 ? Onnx::FindInitializer(graph, node.inputs[2]) : nullptr;
                    if (!weight || weight->dim_count != 2)
                    {
                        SetError(out, "Gemm weight initializer missing or invalid");
                        return false;
                    }

                    const bool trans_b = Onnx::AttributeInt(node, "transB", 0) != 0;
                    const uint32_t out_features = static_cast<uint32_t>(trans_b ? weight->dims[0] : weight->dims[1]);

                    if (out.spec.kind == OnnxGraph::NetworkKind::MLP)
                    {
                        if (!AppendDense(out, out_features, next_activation, in_features, *weight, bias, trans_b))
                            return false;
                        in_features = out_features;
                    }
                    else
                    {
                        if (!AppendDense(out, out_features, next_activation, dense_in > 0 ? dense_in : in_features,
                                         *weight, bias, trans_b))
                            return false;
                        const uint32_t dense_out = out_features;
                        in_features = dense_out;
                        dense_in = dense_out;
                    }
                    i += 1 + skip;
                    continue;
                }

                if (IsOp(node, "Conv") && out.spec.kind == OnnxGraph::NetworkKind::CNN)
                {
                    if (node.input_count < 2)
                    {
                        SetError(out, "Conv node missing weights");
                        return false;
                    }

                    const Onnx::Tensor* weight = Onnx::FindInitializer(graph, node.inputs[1]);
                    const Onnx::Tensor* bias =
                        node.input_count >= 3 ? Onnx::FindInitializer(graph, node.inputs[2]) : nullptr;
                    if (!weight || weight->dim_count != 4)
                    {
                        SetError(out, "Conv weight initializer missing or invalid");
                        return false;
                    }

                    std::span<const int64_t> kernel_shape;
                    std::span<const int64_t> strides;
                    uint32_t kernel = 0;
                    if (!Onnx::AttributeInts(node, "kernel_shape", kernel_shape) ||
                        !SquareKernelFromAttribute(kernel_shape, kernel, out, "Conv"))
                        return false;

                    uint32_t stride = kernel;
                    if (Onnx::AttributeInts(node, "strides", strides))
                        stride = FirstStride(strides, kernel);

                    const uint32_t out_c = static_cast<uint32_t>(weight->dims[0]);
                    const uint32_t in_c = static_cast<uint32_t>(weight->dims[1]);
                    if (in_c != channels)
                    {
                        SetError(out, "Conv input channel count mismatch");
                        return false;
                    }

                    const uint32_t idx = out.spec.num_layers;
                    out.spec.cnn_layer_kinds[idx] = OnnxGraph::CnnLayerKind::Conv2D;
                    out.spec.conv_layers[idx].kernel_size = kernel;
                    out.spec.conv_layers[idx].stride = stride;
                    out.spec.conv_layers[idx].filters = out_c;
                    std::strncpy(out.spec.conv_layers[idx].activation, next_activation, OnnxGraph::kMaxStringLen - 1);

                    CopyConvWeightNchwToNetkit(*weight, out_c, in_c, kernel, out.weights);
                    if (bias)
                        CopyBias(*bias, out_c, out.weights);
                    else
                        out.weights.insert(out.weights.end(), out_c, 0.0f);

                    ++out.spec.num_layers;

                    spatial_h = (spatial_h - kernel) / stride + 1;
                    spatial_w = (spatial_w - kernel) / stride + 1;
                    channels = out_c;
                    i += 1 + skip;
                    continue;
                }

                if (IsOp(node, "MaxPool") && out.spec.kind == OnnxGraph::NetworkKind::CNN)
                {
                    std::span<const int64_t> kernel_shape;
                    std::span<const int64_t> strides;
                    uint32_t pool = 0;
                    if (!Onnx::AttributeInts(node, "kernel_shape", kernel_shape) ||
                        !SquareKernelFromAttribute(kernel_shape, pool, out, "MaxPool"))
                        return false;

                    uint32_t stride = pool;
                    if (Onnx::AttributeInts(node, "strides", strides))
                        stride = FirstStride(strides, pool);

                    const uint32_t idx = out.spec.num_layers;
                    out.spec.cnn_layer_kinds[idx] = OnnxGraph::CnnLayerKind::MaxPool2D;
                    out.spec.pool_layers[idx].pool_size = pool;
                    out.spec.pool_layers[idx].stride = stride;
                    ++out.spec.num_layers;

                    spatial_h = (spatial_h - pool) / stride + 1;
                    spatial_w = (spatial_w - pool) / stride + 1;
                    i += 1 + skip;
                    continue;
                }

                if (IsOp(node, "Flatten") && out.spec.kind == OnnxGraph::NetworkKind::CNN)
                {
                    const uint32_t idx = out.spec.num_layers;
                    out.spec.cnn_layer_kinds[idx] = OnnxGraph::CnnLayerKind::Flatten;
                    ++out.spec.num_layers;
                    in_features = spatial_h * spatial_w * channels;
                    dense_in = in_features;
                    i += 1 + skip;
                    continue;
                }

                char message[128];
                std::snprintf(message, sizeof(message), "Unsupported ONNX op: %s", node.op_type);
                SetError(out, message);
                return false;
            }

            out.spec.version = 1;
            out.spec.expected_weight_floats = out.weights.size();
            return out.spec.num_layers > 0;
        }

        NkLoader::NetworkKind ToNkKind(OnnxGraph::NetworkKind kind)
        {
            switch (kind)
            {
                case OnnxGraph::NetworkKind::MLP: return NkLoader::NetworkKind::Mlp;
                case OnnxGraph::NetworkKind::CNN: return NkLoader::NetworkKind::Cnn;
                default: return NkLoader::NetworkKind::Unknown;
            }
        }

        ImportResult Fail(ImportStatus status,
                          const char* message,
                          NkLoader::NetworkKind kind = NkLoader::NetworkKind::Unknown)
        {
            return ImportResult{status, kind, message};
        }

        ImportResult FailGraph(ImportStatus status, const char* message, OnnxGraph::NetworkKind kind)
        {
            return Fail(status, message, ToNkKind(kind));
        }

        ImportResult ConvertOnnxFile(const char* onnx_path, ConvertedModel& converted)
        {
            Onnx::Model model{};
            if (Onnx::ParseModelFile(onnx_path, model, converted.error, sizeof(converted.error)) != Onnx::ParseStatus::Ok)
                return Fail(ImportStatus::ParseFailed, converted.error);

            if (!ConvertGraph(model, converted))
                return FailGraph(ImportStatus::UnsupportedGraph, converted.error, converted.spec.kind);

            return ImportResult{ImportStatus::Ok, ToNkKind(converted.spec.kind), nullptr};
        }

        ImportResult LoadConverted(const ConvertedModel& converted,
                                   Arena& arena,
                                   NkLoader::NetworkKind& kind,
                                   MLPNetwork*& mlp,
                                   CNNNetwork*& cnn,
                                   std::array<uint32_t, kMaxTensorRank>& input_shape,
                                   uint32_t& input_rank)
        {
            mlp = nullptr;
            cnn = nullptr;
            kind = ToNkKind(converted.spec.kind);
            input_shape = converted.spec.input_shape;
            input_rank = converted.spec.input_rank;

            const std::size_t bytes = converted.weights.size() * sizeof(float);
            float* weights = static_cast<float*>(arena.alloc(bytes, alignof(float)));
            if (!weights)
                return Fail(ImportStatus::ArenaOverflow, "Arena out of memory while loading ONNX weights", kind);

            std::memcpy(weights, converted.weights.data(), bytes);

            if (converted.spec.kind == OnnxGraph::NetworkKind::MLP)
            {
                void* network_mem = arena.alloc(sizeof(MLPNetwork), alignof(MLPNetwork));
                if (!network_mem)
                    return Fail(ImportStatus::ArenaOverflow, "Arena out of memory while creating MLPNetwork", kind);

                mlp = new (network_mem) MLPNetwork(converted.spec.num_layers, arena);
                if (!mlp->IsValid())
                    return Fail(ImportStatus::ArenaOverflow, "Arena out of memory while allocating MLP layers", kind);

                std::size_t offset = 0;
                uint32_t in_features = input_shape[1];
                for (uint32_t i = 0; i < converted.spec.num_layers; ++i)
                {
                    const uint32_t out_features = converted.spec.dense_layers[i].units;
                    const std::size_t weight_elems = static_cast<std::size_t>(in_features) * out_features;
                    Tensor W = TensorFactory::View2D(weights + offset, in_features, out_features);
                    offset += weight_elems;
                    Tensor B = TensorFactory::View2D(weights + offset, 1, out_features);
                    offset += out_features;

                    ActivationType activation = ActivationType::None;
                    const char* act = converted.spec.dense_layers[i].activation;
                    if (std::strcmp(act, "relu") == 0) activation = ActivationType::ReLU;
                    else if (std::strcmp(act, "softmax") == 0) activation = ActivationType::Softmax;
                    else if (std::strcmp(act, "sigmoid") == 0) activation = ActivationType::Sigmoid;
                    else if (std::strcmp(act, "tanh") == 0) activation = ActivationType::Tanh;

                    mlp->InitLayer(i, W, B, activation, converted.spec.dense_layers[i].alpha);
                    in_features = out_features;
                }

                if (!mlp->InitActivationBuffers(arena, input_shape[0]))
                    return Fail(ImportStatus::ArenaOverflow, "Arena out of memory while allocating MLP activation buffers", kind);
            }
            else if (converted.spec.kind == OnnxGraph::NetworkKind::CNN)
            {
                void* network_mem = arena.alloc(sizeof(CNNNetwork), alignof(CNNNetwork));
                if (!network_mem)
                    return Fail(ImportStatus::ArenaOverflow, "Arena out of memory while creating CNNNetwork", kind);

                cnn = new (network_mem) CNNNetwork(converted.spec.num_layers, arena);
                if (!cnn->IsValid())
                    return Fail(ImportStatus::ArenaOverflow, "Arena out of memory while allocating CNN layers", kind);

                std::size_t offset = 0;
                uint32_t in_channels = input_shape[2];
                uint32_t h = input_shape[0];
                uint32_t w = input_shape[1];
                uint32_t dense_in = 0;

                for (uint32_t i = 0; i < converted.spec.num_layers; ++i)
                {
                    switch (converted.spec.cnn_layer_kinds[i])
                    {
                        case OnnxGraph::CnnLayerKind::Conv2D:
                        {
                            const OnnxGraph::ConvLayerConfig& layer = converted.spec.conv_layers[i];
                            const std::size_t kernel_elems = static_cast<std::size_t>(layer.kernel_size) *
                                                             layer.kernel_size * in_channels;
                            const std::size_t weight_elems = kernel_elems * layer.filters;
                            float* layer_weights = weights + offset;
                            offset += weight_elems;
                            float* layer_bias = weights + offset;
                            offset += layer.filters;

                            ConvActivationType activation = ConvActivationType::None;
                            if (std::strcmp(layer.activation, "relu") == 0) activation = ConvActivationType::ReLU;
                            else if (std::strcmp(layer.activation, "softmax") == 0) activation = ConvActivationType::Softmax;

                            cnn->InitConvLayer(i,
                                               static_cast<int>(layer.kernel_size),
                                               static_cast<int>(layer.stride),
                                               static_cast<int>(in_channels),
                                               static_cast<int>(layer.filters),
                                               layer_weights,
                                               layer_bias,
                                               activation,
                                               layer.alpha);
                            h = (h - layer.kernel_size) / layer.stride + 1;
                            w = (w - layer.kernel_size) / layer.stride + 1;
                            in_channels = layer.filters;
                            break;
                        }
                        case OnnxGraph::CnnLayerKind::MaxPool2D:
                        {
                            const OnnxGraph::PoolLayerConfig& layer = converted.spec.pool_layers[i];
                            cnn->InitPoolLayer(i, static_cast<int>(layer.pool_size), static_cast<int>(layer.stride));
                            h = (h - layer.pool_size) / layer.stride + 1;
                            w = (w - layer.pool_size) / layer.stride + 1;
                            break;
                        }
                        case OnnxGraph::CnnLayerKind::Flatten:
                            dense_in = h * w * in_channels;
                            cnn->InitFlattenLayer(i);
                            break;
                        case OnnxGraph::CnnLayerKind::Dense:
                        {
                            const OnnxGraph::DenseLayerConfig& layer = converted.spec.cnn_dense_layers[i];
                            const std::size_t weight_elems = static_cast<std::size_t>(dense_in) * layer.units;
                            Tensor W = TensorFactory::View2D(weights + offset, dense_in, layer.units);
                            offset += weight_elems;
                            Tensor B = TensorFactory::View2D(weights + offset, 1, layer.units);
                            offset += layer.units;

                            ActivationType activation = ActivationType::None;
                            if (std::strcmp(layer.activation, "relu") == 0) activation = ActivationType::ReLU;
                            else if (std::strcmp(layer.activation, "softmax") == 0) activation = ActivationType::Softmax;

                            cnn->InitDenseLayer(i, W, B, activation, layer.alpha);
                            dense_in = layer.units;
                            break;
                        }
                    }
                }

                if (!cnn->InitActivationBuffers(arena, input_shape[0], input_shape[1], input_shape[2]))
                    return Fail(ImportStatus::ArenaOverflow, "Arena out of memory while allocating CNN activation buffers", kind);
            }
            else
            {
                return Fail(ImportStatus::UnsupportedGraph, "Unsupported ONNX network kind", kind);
            }

            return ImportResult{ImportStatus::Ok, kind, nullptr};
        }
    }

    const char* StatusMessage(ImportStatus status)
    {
        switch (status)
        {
            case ImportStatus::Ok: return "ok";
            case ImportStatus::FileOpenFailed: return "file open failed";
            case ImportStatus::ParseFailed: return "parse failed";
            case ImportStatus::UnsupportedGraph: return "unsupported graph";
            case ImportStatus::UnsupportedOp: return "unsupported op";
            case ImportStatus::ShapeMismatch: return "shape mismatch";
            case ImportStatus::ArenaOverflow: return "arena overflow";
        }
        return "unknown";
    }

    ImportResult LoadFromOnnx(const char* onnx_path,
                              Arena& arena,
                              NkLoader::NetworkKind& kind,
                              MLPNetwork*& mlp,
                              CNNNetwork*& cnn,
                              std::array<uint32_t, kMaxTensorRank>& input_shape,
                              uint32_t& input_rank)
    {
        ConvertedModel converted{};
        ImportResult result = ConvertOnnxFile(onnx_path, converted);
        if (result.status != ImportStatus::Ok)
            return result;

        return LoadConverted(converted, arena, kind, mlp, cnn, input_shape, input_rank);
    }
}
