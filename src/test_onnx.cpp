#include "test_onnx.hpp"
#include "json_parser.hpp"
#include "model_loader.hpp"
#include "onnx_importer.hpp"
#include "mlp.hpp"
#include "cnn.hpp"
#include "tensor_factory.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace
{
    constexpr uint32_t kMaxHandFloats = 256;
    constexpr uint32_t kMnistInputDim = 784;
    constexpr uint32_t kMnistOutputDim = 10;
    constexpr uint32_t kMaxMnistCases = 16;
    constexpr std::size_t kHandArenaCapacity = 64u * 1024u;
    constexpr std::size_t kMnistArenaCapacity = 2u * 1024u * 1024u;
    constexpr std::size_t kMnistCnnArenaCapacity = 4u * 1024u * 1024u;
    constexpr std::size_t kManifestBytes = 8192;

    alignas(std::max_align_t) unsigned char g_hand_arena[kHandArenaCapacity];
    alignas(std::max_align_t) unsigned char g_mnist_arena[kMnistArenaCapacity];
    alignas(std::max_align_t) unsigned char g_mnist_cnn_arena[kMnistCnnArenaCapacity];

    struct VectorsParitySpec
    {
        const char* vectors_path;
        const char* onnx_path;
    };

    struct MnistParitySpec
    {
        const char* manifest_path;
        const char* onnx_path;
        bool is_cnn;
    };

    constexpr VectorsParitySpec kVectorParitySpecs[] = {
        {"models/test_mlp.vectors.json", "models/test_mlp.onnx"},
        {"models/mlp_hand.vectors.json", "models/mlp_hand.onnx"},
        {"models/test_cnn.vectors.json", "models/test_cnn.onnx"},
        {"models/cnn_4x4_single.vectors.json", "models/cnn_4x4_single.onnx"},
        {"models/cnn_hand.vectors.json", "models/cnn_hand.onnx"},
    };

    constexpr MnistParitySpec kMnistParitySpecs[] = {
        {"models/mnist/manifest.json", "models/mnist_mlp.onnx", false},
        {"models/mnist_cnn/manifest.json", "models/mnist_cnn.onnx", true},
    };

    bool FileReadable(const char* path)
    {
        std::FILE* file = std::fopen(path, "rb");
        if (!file)
            return false;
        std::fclose(file);
        return true;
    }

    const char* ResolveModelPath(const char* rel_path, char* buffer, std::size_t buffer_size)
    {
        if (FileReadable(rel_path))
            return rel_path;

        std::snprintf(buffer, buffer_size, "../%s", rel_path);
        if (FileReadable(buffer))
            return buffer;

        return rel_path;
    }

    void ResolveSiblingPath(const char* base_path, const char* relative, char* out, std::size_t out_capacity)
    {
        std::strncpy(out, base_path, out_capacity - 1);
        out[out_capacity - 1] = '\0';

        char* slash = std::strrchr(out, '/');
        if (!slash)
            slash = std::strrchr(out, '\\');

        if (slash)
            slash[1] = '\0';
        else
            out[0] = '\0';

        std::strncat(out, relative, out_capacity - std::strlen(out) - 1);
    }

    bool ReadTextFile(const char* path, char* buffer, std::size_t capacity, std::size_t& bytes_read)
    {
        std::FILE* file = std::fopen(path, "rb");
        if (!file)
            return false;

        bytes_read = std::fread(buffer, 1, capacity - 1, file);
        buffer[bytes_read] = '\0';
        const bool ok = !std::ferror(file);
        std::fclose(file);
        return ok;
    }

    bool ReadBinaryFloats(const char* path, float* values, uint32_t count)
    {
        std::FILE* file = std::fopen(path, "rb");
        if (!file)
            return false;

        const std::size_t want = static_cast<std::size_t>(count) * sizeof(float);
        const std::size_t got = std::fread(values, 1, want, file);
        const bool ok = !std::ferror(file) && got == want;
        std::fclose(file);
        return ok;
    }

    bool ParseCaseName(const char* object_begin, const char* object_end, char* out, std::size_t out_capacity)
    {
        const char* value = Json::FindKeyValue(object_begin, object_end, "name");
        if (!value)
        {
            std::strncpy(out, "case", out_capacity);
            out[out_capacity - 1] = '\0';
            return true;
        }

        const char* p = value;
        return Json::ParseString(p, object_end, out, out_capacity);
    }

    bool ParseFloatArray(const char* object_begin,
                         const char* object_end,
                         const char* key,
                         float* values,
                         uint32_t& count,
                         uint32_t max_count)
    {
        const char* value = Json::FindKeyValue(object_begin, object_end, key);
        if (!value || *value != '[')
            return false;

        const char* cursor = value;
        const char* elem_begin = nullptr;
        const char* elem_end = nullptr;
        count = 0;

        while (count < max_count && Json::NextArrayElement(cursor, object_end, elem_begin, elem_end))
        {
            const char* p = elem_begin;
            float number = 0.0f;
            if (!Json::ParseFloat(p, elem_end, number))
                return false;
            values[count++] = number;
        }

        return count > 0;
    }

    bool FloatNear(float a, float b, float eps)
    {
        return std::fabs(a - b) <= eps;
    }

    bool CompareOutputs(const float* json_out,
                        const float* onnx_out,
                        uint32_t count,
                        float tol,
                        const char* label)
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!FloatNear(json_out[i], onnx_out[i], tol))
            {
                std::cout << "FAIL " << label << ": json/onnx mismatch at out[" << i << "] "
                          << json_out[i] << " vs " << onnx_out[i] << "\n";
                return false;
            }
        }
        return true;
    }

    uint32_t ExpectedMlpOutputElements(const ModelLoader::ArchitectureSpec& spec)
    {
        if (spec.num_layers == 0)
            return 0;
        return spec.input_shape[0] * spec.dense_layers[spec.num_layers - 1].units;
    }

    Tensor MakeNhwcInput(float* data, uint32_t h, uint32_t w, uint32_t c)
    {
        Tensor input{};
        input.data = data;
        input.type = DataType::Float32;
        input.rank = 3;
        input.shape[0] = h;
        input.shape[1] = w;
        input.shape[2] = c;
        input.stride[0] = w * c;
        input.stride[1] = c;
        input.stride[2] = 1;
        input.num_elements = h * w * c;
        input.bytes = input.num_elements * sizeof(float);
        return input;
    }

    bool ForwardMlpJson(const char* json_path,
                        const float* input_values,
                        uint32_t input_count,
                        float* output,
                        uint32_t output_count,
                        Arena& arena)
    {
        MLPNetwork* network = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;

        if (ModelLoader::LoadMLP(json_path, arena, network, input_shape, input_rank).status !=
                ModelLoader::LoadStatus::Ok ||
            !network || !network->IsValid())
            return false;

        Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
        Tensor out = TensorFactory::Create2D(arena, input_shape[0], output_count / input_shape[0]);
        if (!input.data || !out.data)
            return false;

        float* input_data = static_cast<float*>(input.data);
        for (uint32_t i = 0; i < input_count; ++i)
            input_data[i] = input_values[i];

        network->forward(input, out, arena);
        std::memcpy(output, out.data, static_cast<std::size_t>(output_count) * sizeof(float));
        return true;
    }

    bool ForwardMlpOnnx(const char* onnx_path,
                        const float* input_values,
                        uint32_t input_count,
                        float* output,
                        uint32_t output_count,
                        Arena& arena)
    {
        MLPNetwork* network = nullptr;
        CNNNetwork* cnn = nullptr;
        ModelLoader::NetworkKind kind = ModelLoader::NetworkKind::Unknown;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;

        if (OnnxImporter::LoadFromOnnx(onnx_path, arena, kind, network, cnn, input_shape, input_rank).status !=
                OnnxImporter::ImportStatus::Ok ||
            kind != ModelLoader::NetworkKind::MLP || !network)
            return false;

        Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
        Tensor out = TensorFactory::Create2D(arena, input_shape[0], output_count / input_shape[0]);
        if (!input.data || !out.data)
            return false;

        float* input_data = static_cast<float*>(input.data);
        for (uint32_t i = 0; i < input_count; ++i)
            input_data[i] = input_values[i];

        network->forward(input, out, arena);
        std::memcpy(output, out.data, static_cast<std::size_t>(output_count) * sizeof(float));
        return true;
    }

    bool ForwardCnnJson(const char* json_path,
                        const std::array<uint32_t, kMaxTensorRank>& input_shape,
                        float* input_values,
                        uint32_t /*input_count*/,
                        float* output,
                        uint32_t& output_count,
                        Arena& arena)
    {
        CNNNetwork* network = nullptr;
        std::array<uint32_t, kMaxTensorRank> loaded_shape{};
        uint32_t input_rank = 0;

        if (ModelLoader::LoadCNN(json_path, arena, network, loaded_shape, input_rank).status !=
                ModelLoader::LoadStatus::Ok ||
            !network || !network->IsValid())
            return false;

        Tensor input = MakeNhwcInput(input_values, input_shape[0], input_shape[1], input_shape[2]);
        Tensor& out = network->forward(input, arena);
        if (!out.data)
            return false;

        output_count = out.num_elements;
        std::memcpy(output, out.data, static_cast<std::size_t>(output_count) * sizeof(float));
        return true;
    }

    bool ForwardCnnOnnx(const char* onnx_path,
                        const std::array<uint32_t, kMaxTensorRank>& input_shape,
                        float* input_values,
                        uint32_t /*input_count*/,
                        float* output,
                        uint32_t& output_count,
                        Arena& arena)
    {
        MLPNetwork* mlp = nullptr;
        CNNNetwork* network = nullptr;
        ModelLoader::NetworkKind kind = ModelLoader::NetworkKind::Unknown;
        std::array<uint32_t, kMaxTensorRank> loaded_shape{};
        uint32_t input_rank = 0;

        if (OnnxImporter::LoadFromOnnx(onnx_path, arena, kind, mlp, network, loaded_shape, input_rank).status !=
                OnnxImporter::ImportStatus::Ok ||
            kind != ModelLoader::NetworkKind::CNN || !network)
            return false;

        Tensor input = MakeNhwcInput(input_values, input_shape[0], input_shape[1], input_shape[2]);
        Tensor& out = network->forward(input, arena);
        if (!out.data)
            return false;

        output_count = out.num_elements;
        std::memcpy(output, out.data, static_cast<std::size_t>(output_count) * sizeof(float));
        return true;
    }

    VectorsLoader::RunSummary RunVectorsParity(const VectorsParitySpec& spec)
    {
        VectorsLoader::RunSummary summary{};

        char vectors_buffer[ModelLoader::kMaxPathLen] = {};
        char onnx_buffer[ModelLoader::kMaxPathLen] = {};
        const char* resolved_vectors =
            ResolveModelPath(spec.vectors_path, vectors_buffer, sizeof(vectors_buffer));
        const char* resolved_onnx = ResolveModelPath(spec.onnx_path, onnx_buffer, sizeof(onnx_buffer));

        char json_buffer[ModelLoader::kMaxJsonBytes] = {};
        std::size_t json_bytes = 0;
        if (!ReadTextFile(resolved_vectors, json_buffer, sizeof(json_buffer), json_bytes))
        {
            std::cout << "FAIL ONNX parity: cannot read " << resolved_vectors << "\n";
            ++summary.failed;
            return summary;
        }

        const char* json = json_buffer;
        const char* json_end = json_buffer + json_bytes;

        char model_relative[Json::kMaxStringLen] = {};
        const char* model_value = Json::FindKeyValue(json, json_end, "model");
        if (!model_value || !Json::ParseString(model_value, json_end, model_relative, sizeof(model_relative)))
        {
            std::cout << "FAIL ONNX parity: invalid model path in " << resolved_vectors << "\n";
            ++summary.failed;
            return summary;
        }

        char model_path[ModelLoader::kMaxPathLen] = {};
        ResolveSiblingPath(resolved_vectors, model_relative, model_path, sizeof(model_path));
        char model_fallback[ModelLoader::kMaxPathLen] = {};
        const char* resolved_model = ResolveModelPath(model_path, model_fallback, sizeof(model_fallback));

        ModelLoader::ArchitectureSpec arch{};
        if (ModelLoader::ParseArchitecture(resolved_model, arch).status != ModelLoader::LoadStatus::Ok)
        {
            std::cout << "FAIL ONNX parity: cannot parse " << resolved_model << "\n";
            ++summary.failed;
            return summary;
        }

        std::cout << "ONNX parity vectors: " << resolved_vectors << "\n";
        std::cout << "  JSON model: " << resolved_model << "\n";
        std::cout << "  ONNX model: " << resolved_onnx << "\n";

        const char* cases_value = Json::FindKeyValue(json, json_end, "cases");
        if (!cases_value || *cases_value != '[')
        {
            std::cout << "FAIL ONNX parity: missing cases in " << resolved_vectors << "\n";
            ++summary.failed;
            return summary;
        }

        const char* cursor = cases_value;
        const char* case_begin = nullptr;
        const char* case_end = nullptr;

        while (Json::NextArrayElement(cursor, json_end, case_begin, case_end))
        {
            char case_name[Json::kMaxStringLen] = {};
            ParseCaseName(case_begin, case_end, case_name, sizeof(case_name));

            float input_values[kMaxHandFloats] = {};
            float expected_values[kMaxHandFloats] = {};
            uint32_t input_count = 0;
            uint32_t expected_count = 0;

            if (!ParseFloatArray(case_begin, case_end, "input", input_values, input_count, kMaxHandFloats) ||
                !ParseFloatArray(case_begin, case_end, "expected", expected_values, expected_count, kMaxHandFloats))
            {
                std::cout << "FAIL ONNX parity " << case_name << ": invalid arrays\n";
                ++summary.failed;
                continue;
            }

            float json_out[kMaxHandFloats] = {};
            float onnx_out[kMaxHandFloats] = {};
            uint32_t json_out_count = expected_count;
            uint32_t onnx_out_count = expected_count;

            Arena json_arena;
            json_arena.init(g_hand_arena, sizeof(g_hand_arena));

            bool ok = false;
            if (arch.kind == ModelLoader::NetworkKind::MLP)
            {
                json_out_count = ExpectedMlpOutputElements(arch);
                if (!ForwardMlpJson(resolved_model, input_values, input_count, json_out, json_out_count, json_arena))
                {
                    std::cout << "FAIL ONNX parity " << case_name << ": JSON forward failed\n";
                    ++summary.failed;
                    continue;
                }

                Arena onnx_arena;
                onnx_arena.init(g_hand_arena, sizeof(g_hand_arena));
                if (!ForwardMlpOnnx(resolved_onnx, input_values, input_count, onnx_out, json_out_count, onnx_arena))
                {
                    std::cout << "FAIL ONNX parity " << case_name << ": ONNX forward failed\n";
                    ++summary.failed;
                    continue;
                }

                ok = CompareOutputs(json_out, onnx_out, json_out_count, 1e-5f, case_name);
            }
            else if (arch.kind == ModelLoader::NetworkKind::CNN)
            {
                if (!ForwardCnnJson(resolved_model, arch.input_shape, input_values, input_count, json_out,
                                    json_out_count, json_arena))
                {
                    std::cout << "FAIL ONNX parity " << case_name << ": JSON forward failed\n";
                    ++summary.failed;
                    continue;
                }

                Arena onnx_arena;
                onnx_arena.init(g_hand_arena, sizeof(g_hand_arena));
                if (!ForwardCnnOnnx(resolved_onnx, arch.input_shape, input_values, input_count, onnx_out,
                                    onnx_out_count, onnx_arena))
                {
                    std::cout << "FAIL ONNX parity " << case_name << ": ONNX forward failed\n";
                    ++summary.failed;
                    continue;
                }

                if (json_out_count != onnx_out_count)
                {
                    std::cout << "FAIL ONNX parity " << case_name << ": output size "
                              << json_out_count << " vs " << onnx_out_count << "\n";
                    ++summary.failed;
                    continue;
                }

                ok = CompareOutputs(json_out, onnx_out, json_out_count, 1e-5f, case_name);
            }

            if (ok)
            {
                std::cout << "PASS ONNX parity " << case_name << " (json vs onnx, " << json_out_count
                          << " outputs)\n";
                ++summary.passed;
            }
            else
                ++summary.failed;
        }

        return summary;
    }

    bool ParseCaseStringField(const char* object_begin,
                              const char* object_end,
                              const char* key,
                              char* out,
                              std::size_t out_capacity)
    {
        const char* value = Json::FindKeyValue(object_begin, object_end, key);
        if (!value)
            return false;

        const char* p = value;
        return Json::ParseString(p, object_end, out, out_capacity);
    }

    VectorsLoader::RunSummary RunMnistParity(const MnistParitySpec& spec)
    {
        VectorsLoader::RunSummary summary{};

        char manifest_buffer[ModelLoader::kMaxPathLen] = {};
        char onnx_buffer[ModelLoader::kMaxPathLen] = {};
        const char* manifest_path =
            ResolveModelPath(spec.manifest_path, manifest_buffer, sizeof(manifest_buffer));
        const char* resolved_onnx = ResolveModelPath(spec.onnx_path, onnx_buffer, sizeof(onnx_buffer));

        char json_buffer[kManifestBytes] = {};
        std::size_t json_bytes = 0;
        if (!ReadTextFile(manifest_path, json_buffer, sizeof(json_buffer), json_bytes))
        {
            std::cout << "FAIL MNIST ONNX parity: cannot read " << manifest_path << "\n";
            ++summary.failed;
            return summary;
        }

        const char* json = json_buffer;
        const char* json_end = json_buffer + json_bytes;

        char model_relative[ModelLoader::kMaxPathLen] = {};
        const char* model_value = Json::FindKeyValue(json, json_end, "model");
        if (!model_value || !Json::ParseString(model_value, json_end, model_relative, sizeof(model_relative)))
        {
            std::cout << "FAIL MNIST ONNX parity: missing model in manifest\n";
            ++summary.failed;
            return summary;
        }

        char model_path[ModelLoader::kMaxPathLen] = {};
        ResolveSiblingPath(manifest_path, model_relative, model_path, sizeof(model_path));
        char model_fallback[ModelLoader::kMaxPathLen] = {};
        const char* resolved_model = ResolveModelPath(model_path, model_fallback, sizeof(model_fallback));

        std::cout << "ONNX parity manifest: " << manifest_path << "\n";
        std::cout << "  JSON model: " << resolved_model << "\n";
        std::cout << "  ONNX model: " << resolved_onnx << "\n";

        const char* cases_value = Json::FindKeyValue(json, json_end, "cases");
        if (!cases_value || *cases_value != '[')
        {
            std::cout << "FAIL MNIST ONNX parity: missing cases\n";
            ++summary.failed;
            return summary;
        }

        const char* cursor = cases_value;
        const char* case_begin = nullptr;
        const char* case_end = nullptr;

        while (summary.passed + summary.failed < kMaxMnistCases &&
               Json::NextArrayElement(cursor, json_end, case_begin, case_end))
        {
            char case_name[Json::kMaxStringLen] = {};
            char input_rel[ModelLoader::kMaxPathLen] = {};
            if (!ParseCaseName(case_begin, case_end, case_name, sizeof(case_name)) ||
                !ParseCaseStringField(case_begin, case_end, "input", input_rel, sizeof(input_rel)))
            {
                std::cout << "FAIL MNIST ONNX parity: case parse error\n";
                ++summary.failed;
                continue;
            }

            char input_path[ModelLoader::kMaxPathLen] = {};
            ResolveSiblingPath(manifest_path, input_rel, input_path, sizeof(input_path));
            char input_fallback[ModelLoader::kMaxPathLen] = {};
            const char* resolved_input = ResolveModelPath(input_path, input_fallback, sizeof(input_fallback));

            float input_values[kMnistInputDim] = {};
            if (!ReadBinaryFloats(resolved_input, input_values, kMnistInputDim))
            {
                std::cout << "FAIL ONNX parity " << case_name << ": cannot read input\n";
                ++summary.failed;
                continue;
            }

            float json_out[kMnistOutputDim] = {};
            float onnx_out[kMnistOutputDim] = {};

            if (spec.is_cnn)
            {
                std::array<uint32_t, kMaxTensorRank> input_shape = {28, 28, 1};
                float cnn_input[kMnistInputDim] = {};
                for (uint32_t i = 0; i < kMnistInputDim; ++i)
                    cnn_input[i] = input_values[i];

                Arena json_arena;
                json_arena.init(g_mnist_cnn_arena, sizeof(g_mnist_cnn_arena));
                uint32_t json_out_count = kMnistOutputDim;
                if (!ForwardCnnJson(resolved_model, input_shape, cnn_input, kMnistInputDim, json_out,
                                    json_out_count, json_arena))
                {
                    std::cout << "FAIL ONNX parity " << case_name << ": JSON forward failed\n";
                    ++summary.failed;
                    continue;
                }

                Arena onnx_arena;
                onnx_arena.init(g_mnist_cnn_arena, sizeof(g_mnist_cnn_arena));
                uint32_t onnx_out_count = kMnistOutputDim;
                if (!ForwardCnnOnnx(resolved_onnx, input_shape, cnn_input, kMnistInputDim, onnx_out,
                                    onnx_out_count, onnx_arena))
                {
                    std::cout << "FAIL ONNX parity " << case_name << ": ONNX forward failed\n";
                    ++summary.failed;
                    continue;
                }

                if (CompareOutputs(json_out, onnx_out, kMnistOutputDim, 1e-5f, case_name))
                {
                    std::cout << "PASS ONNX parity " << case_name << " (json vs onnx, 10 outputs)\n";
                    ++summary.passed;
                }
                else
                    ++summary.failed;
            }
            else
            {
                Arena json_arena;
                json_arena.init(g_mnist_arena, sizeof(g_mnist_arena));
                if (!ForwardMlpJson(resolved_model, input_values, kMnistInputDim, json_out, kMnistOutputDim,
                                    json_arena))
                {
                    std::cout << "FAIL ONNX parity " << case_name << ": JSON forward failed\n";
                    ++summary.failed;
                    continue;
                }

                Arena onnx_arena;
                onnx_arena.init(g_mnist_arena, sizeof(g_mnist_arena));
                if (!ForwardMlpOnnx(resolved_onnx, input_values, kMnistInputDim, onnx_out, kMnistOutputDim,
                                    onnx_arena))
                {
                    std::cout << "FAIL ONNX parity " << case_name << ": ONNX forward failed\n";
                    ++summary.failed;
                    continue;
                }

                if (CompareOutputs(json_out, onnx_out, kMnistOutputDim, 1e-5f, case_name))
                {
                    std::cout << "PASS ONNX parity " << case_name << " (json vs onnx, 10 outputs)\n";
                    ++summary.passed;
                }
                else
                    ++summary.failed;
            }
        }

        return summary;
    }
}

VectorsLoader::RunSummary run_onnx_import_tests()
{
    VectorsLoader::RunSummary summary{};

    std::cout << "\n============================\n";
    std::cout << " ONNX PARITY TESTS\n";
    std::cout << "============================\n";

    for (const VectorsParitySpec& spec : kVectorParitySpecs)
    {
        const VectorsLoader::RunSummary part = RunVectorsParity(spec);
        summary.passed += part.passed;
        summary.failed += part.failed;
    }

    for (const MnistParitySpec& spec : kMnistParitySpecs)
    {
        const VectorsLoader::RunSummary part = RunMnistParity(spec);
        summary.passed += part.passed;
        summary.failed += part.failed;
    }

    return summary;
}
