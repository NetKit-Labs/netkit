#include "test_mnist.hpp"
#include "json_parser.hpp"
#include "model_loader.hpp"
#include "mlp.hpp"
#include "tensor_factory.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace
{
    constexpr uint32_t kInputDim = 784;
    constexpr uint32_t kOutputDim = 10;
    constexpr uint32_t kMaxCases = 16;
    constexpr std::size_t kMnistArenaCapacity = 2u * 1024u * 1024u;
    constexpr std::size_t kManifestBytes = 8192;

    alignas(std::max_align_t) unsigned char g_mnist_arena_buffer[kMnistArenaCapacity];

    struct MnistCase
    {
        char name[Json::kMaxStringLen] = {};
        char input_path[ModelLoader::kMaxPathLen] = {};
        char expected_path[ModelLoader::kMaxPathLen] = {};
        uint32_t label = 0;
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

    bool ParseCaseUintField(const char* object_begin, const char* object_end, const char* key, uint32_t& out)
    {
        const char* value = Json::FindKeyValue(object_begin, object_end, key);
        if (!value)
            return false;

        const char* p = value;
        return Json::ParseUint(p, object_end, out);
    }

    bool FloatNear(float a, float b, float eps)
    {
        return std::fabs(a - b) <= eps;
    }

    uint32_t ArgMax(const float* values, uint32_t count)
    {
        uint32_t best = 0;
        for (uint32_t i = 1; i < count; ++i)
        {
            if (values[i] > values[best])
                best = i;
        }
        return best;
    }

    void PrintOutputNeurons(const float* actual, const float* expected, float tolerance)
    {
        std::cout << std::fixed;
        for (uint32_t i = 0; i < kOutputDim; ++i)
        {
            const bool ok = FloatNear(actual[i], expected[i], tolerance);
            std::cout << "  neuron[" << i << "]: actual=" << actual[i]
                      << " expected=" << expected[i]
                      << (ok ? " OK" : " MISMATCH") << "\n";
        }
    }
}

VectorsLoader::RunSummary run_mnist_tests()
{
    VectorsLoader::RunSummary summary{};

    char manifest_path_buffer[ModelLoader::kMaxPathLen] = {};
    const char* manifest_path =
        ResolveModelPath("models/mnist/manifest.json", manifest_path_buffer, sizeof(manifest_path_buffer));

    char json_buffer[kManifestBytes] = {};
    std::size_t json_bytes = 0;
    if (!ReadTextFile(manifest_path, json_buffer, sizeof(json_buffer), json_bytes))
    {
        std::cout << "FAIL MNIST manifest: could not read " << manifest_path << "\n";
        summary.failed++;
        return summary;
    }

    const char* json = json_buffer;
    const char* json_end = json_buffer + json_bytes;

    char model_relative[ModelLoader::kMaxPathLen] = {};
    const char* model_value = Json::FindKeyValue(json, json_end, "model");
    if (!model_value || !Json::ParseString(model_value, json_end, model_relative, sizeof(model_relative)))
    {
        std::cout << "FAIL MNIST manifest: missing model field\n";
        summary.failed++;
        return summary;
    }

    float tolerance = 0.05f;
    const char* tol_value = Json::FindKeyValue(json, json_end, "output_tolerance");
    if (tol_value)
    {
        const char* p = tol_value;
        Json::ParseFloat(p, json_end, tolerance);
    }

    char model_path[ModelLoader::kMaxPathLen] = {};
    ResolveSiblingPath(manifest_path, model_relative, model_path, sizeof(model_path));

    char model_fallback[ModelLoader::kMaxPathLen] = {};
    const char* resolved_model = ResolveModelPath(model_path, model_fallback, sizeof(model_fallback));

    std::cout << "Manifest: " << manifest_path << "\n";
    std::cout << "Model:    " << resolved_model << "\n";
    std::cout << "Architecture: 784 -> 128 (ReLU) -> 10 (softmax)\n";
    std::cout << "Output tolerance: " << tolerance << "\n";

    Arena arena;
    arena.init(g_mnist_arena_buffer, sizeof(g_mnist_arena_buffer));

    MLPNetwork* network = nullptr;
    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;

    const ModelLoader::LoadResult load_result =
        ModelLoader::LoadMLP(resolved_model, arena, network, input_shape, input_rank);

    if (load_result.status != ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        std::cout << "FAIL MNIST model load: "
                  << (load_result.message ? load_result.message : "unknown error") << "\n";
        summary.failed++;
        return summary;
    }

    const char* cases_value = Json::FindKeyValue(json, json_end, "cases");
    if (!cases_value || *cases_value != '[')
    {
        std::cout << "FAIL MNIST manifest: missing cases array\n";
        summary.failed++;
        return summary;
    }

    const char* cursor = cases_value;
    const char* case_begin = nullptr;
    const char* case_end = nullptr;

    while (summary.passed + summary.failed < kMaxCases &&
           Json::NextArrayElement(cursor, json_end, case_begin, case_end))
    {
        MnistCase test_case{};
        if (!ParseCaseName(case_begin, case_end, test_case.name, sizeof(test_case.name)) ||
            !ParseCaseStringField(case_begin, case_end, "input", test_case.input_path, sizeof(test_case.input_path)) ||
            !ParseCaseStringField(case_begin, case_end, "expected", test_case.expected_path,
                                  sizeof(test_case.expected_path)) ||
            !ParseCaseUintField(case_begin, case_end, "label", test_case.label))
        {
            std::cout << "FAIL MNIST case parse\n";
            summary.failed++;
            continue;
        }

        char input_path[ModelLoader::kMaxPathLen] = {};
        char expected_path[ModelLoader::kMaxPathLen] = {};
        ResolveSiblingPath(manifest_path, test_case.input_path, input_path, sizeof(input_path));
        ResolveSiblingPath(manifest_path, test_case.expected_path, expected_path, sizeof(expected_path));

        char input_fallback[ModelLoader::kMaxPathLen] = {};
        char expected_fallback[ModelLoader::kMaxPathLen] = {};
        const char* resolved_input = ResolveModelPath(input_path, input_fallback, sizeof(input_fallback));
        const char* resolved_expected =
            ResolveModelPath(expected_path, expected_fallback, sizeof(expected_fallback));

        float input_values[kInputDim] = {};
        float expected_values[kOutputDim] = {};

        if (!ReadBinaryFloats(resolved_input, input_values, kInputDim) ||
            !ReadBinaryFloats(resolved_expected, expected_values, kOutputDim))
        {
            std::cout << "FAIL " << test_case.name << ": could not read binary case files\n";
            summary.failed++;
            continue;
        }

        arena.reset();

        const ModelLoader::LoadResult reload =
            ModelLoader::LoadMLP(resolved_model, arena, network, input_shape, input_rank);
        if (reload.status != ModelLoader::LoadStatus::Ok || !network || !network->IsValid())
        {
            std::cout << "FAIL " << test_case.name << ": model reload after arena reset\n";
            summary.failed++;
            continue;
        }

        Tensor input = TensorFactory::Create2D(arena, 1, kInputDim);
        Tensor output = TensorFactory::Create2D(arena, 1, kOutputDim);
        if (!input.data || !output.data)
        {
            std::cout << "FAIL " << test_case.name << ": arena overflow allocating tensors\n";
            summary.failed++;
            continue;
        }

        float* input_data = static_cast<float*>(input.data);
        for (uint32_t i = 0; i < kInputDim; ++i)
            input_data[i] = input_values[i];

        network->forward(input, output, arena);

        const float* actual = static_cast<const float*>(output.data);
        const uint32_t predicted = ArgMax(actual, kOutputDim);

        std::cout << "\nCase: " << test_case.name << "\n";
        std::cout << "Label: " << test_case.label << "  Predicted: " << predicted << "\n";
        PrintOutputNeurons(actual, expected_values, tolerance);

        bool outputs_ok = true;
        for (uint32_t i = 0; i < kOutputDim; ++i)
        {
            if (!FloatNear(actual[i], expected_values[i], tolerance))
                outputs_ok = false;
        }

        const bool class_ok = predicted == test_case.label;

        if (outputs_ok && class_ok)
        {
            std::cout << "PASS " << test_case.name << " (all neurons within tolerance, classification correct)\n";
            summary.passed++;
        }
        else
        {
            if (!outputs_ok)
                std::cout << "FAIL " << test_case.name << ": output neuron mismatch\n";
            if (!class_ok)
                std::cout << "FAIL " << test_case.name << ": classification mismatch\n";
            summary.failed++;
        }
    }

    return summary;
}
