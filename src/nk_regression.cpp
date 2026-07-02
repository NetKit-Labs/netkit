#include "nk_regression.hpp"
#include "nk_loader.hpp"
#include "onnx_importer.hpp"
#include "tensor_factory.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace NkRegression
{
    namespace
    {
        constexpr std::size_t kHandArenaCapacity = Arena::kDefaultCapacity;
        constexpr std::size_t kMnistMlpArenaCapacity = 2u * 1024u * 1024u;
        constexpr std::size_t kMnistCnnArenaCapacity = 4u * 1024u * 1024u;

        alignas(std::max_align_t) unsigned char g_hand_arena[kHandArenaCapacity];
        alignas(std::max_align_t) unsigned char g_mnist_mlp_arena[kMnistMlpArenaCapacity];
        alignas(std::max_align_t) unsigned char g_mnist_cnn_arena[kMnistCnnArenaCapacity];

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

        std::size_t ArenaCapacityForModel(const NkLoader::ParsedModel& model)
        {
            const uint32_t input_elements = NkLoader::InputElements(model);
            if (input_elements >= 784)
            {
                if (model.header.network_kind == NkFormat::NetworkKind::Cnn)
                    return kMnistCnnArenaCapacity;
                return kMnistMlpArenaCapacity;
            }
            return kHandArenaCapacity;
        }

        unsigned char* ArenaBufferForCapacity(std::size_t capacity)
        {
            if (capacity == kMnistCnnArenaCapacity)
                return g_mnist_cnn_arena;
            if (capacity == kMnistMlpArenaCapacity)
                return g_mnist_mlp_arena;
            return g_hand_arena;
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

        void PrintElementComparison(const float* actual,
                                    const float* expected,
                                    uint32_t count,
                                    float eps)
        {
            std::cout << std::fixed << std::setprecision(4);
            for (uint32_t i = 0; i < count; ++i)
            {
                const bool ok = FloatNear(actual[i], expected[i], eps);
                std::cout << "  out[" << i << "]: actual=" << actual[i]
                          << "  expected=" << expected[i]
                          << (ok ? "  OK" : "  MISMATCH") << "\n";
            }
        }

        void PrintClassificationSummary(const float* actual,
                                        const float* expected,
                                        uint32_t output_count,
                                        int32_t label,
                                        float tolerance)
        {
            const uint32_t predicted = ArgMax(actual, output_count);

            std::cout << std::fixed << std::setprecision(4);
            std::cout << "  predicted class: " << predicted
                      << "  (label " << label << ")\n";
            std::cout << "  winner out[" << predicted << "]: actual=" << actual[predicted]
                      << "  expected=" << expected[predicted];
            if (FloatNear(actual[predicted], expected[predicted], tolerance))
                std::cout << "  OK\n";
            else
                std::cout << "  MISMATCH\n";

            constexpr float kRunnerUpThreshold = 0.01f;
            for (uint32_t i = 0; i < output_count; ++i)
            {
                if (i == predicted)
                    continue;
                if (actual[i] >= kRunnerUpThreshold || expected[i] >= kRunnerUpThreshold)
                {
                    std::cout << "  runner-up out[" << i << "]: actual=" << actual[i]
                              << "  expected=" << expected[i];
                    if (FloatNear(actual[i], expected[i], tolerance))
                        std::cout << "  OK\n";
                    else
                        std::cout << "  MISMATCH\n";
                }
            }
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

        bool RunMlpCase(MLPNetwork& network,
                        const NkLoader::ParsedModel& model,
                        const std::array<uint32_t, kMaxTensorRank>& input_shape,
                        const NkLoader::TestCase& test_case,
                        float tolerance,
                        Arena& arena)
        {
            const uint32_t required = input_shape[0] * input_shape[1];
            if (test_case.input_count != required)
            {
                std::cout << "FAIL " << test_case.name << ": input length " << test_case.input_count
                          << " != expected " << required << "\n";
                return false;
            }

            const uint32_t output_elements = NkLoader::OutputElements(model);
            if (test_case.output_count != output_elements)
            {
                std::cout << "FAIL " << test_case.name << ": expected length " << test_case.output_count
                          << " != model output " << output_elements << "\n";
                return false;
            }

            Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
            if (!input.data)
            {
                std::cout << "FAIL " << test_case.name << ": arena overflow while allocating input\n";
                return false;
            }

            float* input_data = static_cast<float*>(input.data);
            for (uint32_t i = 0; i < test_case.input_count; ++i)
                input_data[i] = test_case.input[i];

            const uint32_t output_cols = output_elements / input_shape[0];
            Tensor output = TensorFactory::Create2D(arena, input_shape[0], output_cols);
            if (!output.data)
            {
                std::cout << "FAIL " << test_case.name << ": arena overflow while allocating output\n";
                return false;
            }

            TensorFactory::PrintLabeled("Input", input);
            network.forward(input, output, arena);

            const float* actual = static_cast<const float*>(output.data);
            PrintElementComparison(actual, test_case.expected, test_case.output_count, tolerance);

            bool outputs_ok = true;
            for (uint32_t i = 0; i < test_case.output_count; ++i)
            {
                if (!FloatNear(actual[i], test_case.expected[i], tolerance))
                    outputs_ok = false;
            }

            if (test_case.label >= 0)
            {
                PrintClassificationSummary(actual, test_case.expected, test_case.output_count, test_case.label,
                                           tolerance);
                const bool class_ok = ArgMax(actual, test_case.output_count) == static_cast<uint32_t>(test_case.label);
                if (outputs_ok && class_ok)
                {
                    std::cout << "PASS " << test_case.name
                              << " (all neurons within tolerance, classification correct)\n";
                    return true;
                }

                if (!outputs_ok)
                    std::cout << "FAIL " << test_case.name << ": output neuron mismatch\n";
                if (!class_ok)
                    std::cout << "FAIL " << test_case.name << ": classification mismatch\n";
                return false;
            }

            if (outputs_ok)
            {
                std::cout << "PASS " << test_case.name << " (" << test_case.output_count
                          << " outputs match within tolerance)\n";
                return true;
            }

            std::cout << "FAIL " << test_case.name << " (mismatch in outputs)\n";
            return false;
        }

        bool RunCnnCase(CNNNetwork& network,
                        const std::array<uint32_t, kMaxTensorRank>& input_shape,
                        const NkLoader::TestCase& test_case,
                        float tolerance,
                        Arena& arena)
        {
            const uint32_t required = input_shape[0] * input_shape[1] * input_shape[2];
            if (test_case.input_count != required)
            {
                std::cout << "FAIL " << test_case.name << ": input length " << test_case.input_count
                          << " != expected " << required << "\n";
                return false;
            }

            float input_buffer[NkFormat::kMaxCaseFloats] = {};
            for (uint32_t i = 0; i < test_case.input_count; ++i)
                input_buffer[i] = test_case.input[i];

            Tensor input = MakeNhwcInput(input_buffer, input_shape[0], input_shape[1], input_shape[2]);
            TensorFactory::PrintLabeled("Input", input);

            Tensor& output = network.forward(input, arena);
            if (!output.data)
            {
                std::cout << "FAIL " << test_case.name << ": arena overflow during CNN forward pass\n";
                return false;
            }

            if (test_case.output_count != output.num_elements)
            {
                std::cout << "FAIL " << test_case.name << ": expected length " << test_case.output_count
                          << " != output elements " << output.num_elements << "\n";
                return false;
            }

            const float* actual = static_cast<const float*>(output.data);
            PrintElementComparison(actual, test_case.expected, test_case.output_count, tolerance);

            for (uint32_t i = 0; i < test_case.output_count; ++i)
            {
                if (!FloatNear(actual[i], test_case.expected[i], tolerance))
                {
                    std::cout << "FAIL " << test_case.name << " (mismatch at out[" << i << "])\n";
                    return false;
                }
            }

            if (test_case.label >= 0)
            {
                PrintClassificationSummary(actual, test_case.expected, test_case.output_count, test_case.label,
                                           tolerance);
                const bool class_ok = ArgMax(actual, test_case.output_count) == static_cast<uint32_t>(test_case.label);
                if (!class_ok)
                {
                    std::cout << "FAIL " << test_case.name << ": classification mismatch\n";
                    return false;
                }
                std::cout << "PASS " << test_case.name
                          << " (all neurons within tolerance, classification correct)\n";
                return true;
            }

            std::cout << "PASS " << test_case.name << " (" << test_case.output_count
                      << " outputs match within tolerance)\n";
            return true;
        }

        bool ForwardMlpNk(const char* nk_path,
                          const float* input_values,
                          uint32_t input_count,
                          float* output,
                          uint32_t output_count,
                          Arena& arena)
        {
            MLPNetwork* network = nullptr;
            std::array<uint32_t, kMaxTensorRank> input_shape{};
            uint32_t input_rank = 0;

            if (NkLoader::LoadMLP(nk_path, arena, network, input_shape, input_rank).status !=
                    NkLoader::LoadStatus::Ok ||
                !network || !network->IsValid())
                return false;

            if (input_count != input_shape[0] * input_shape[1])
                return false;

            Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
            Tensor out = TensorFactory::Create2D(arena, input_shape[0], output_count / input_shape[0]);
            if (!input.data || !out.data)
                return false;

            float* input_data = static_cast<float*>(input.data);
            for (uint32_t i = 0; i < input_count; ++i)
                input_data[i] = input_values[i];

            network->forward(input, out, arena);
            if (!out.data || out.num_elements != output_count)
                return false;

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
            MLPNetwork* mlp = nullptr;
            CNNNetwork* cnn = nullptr;
            NkLoader::NetworkKind kind = NkLoader::NetworkKind::Unknown;
            std::array<uint32_t, kMaxTensorRank> input_shape{};
            uint32_t input_rank = 0;

            if (OnnxImporter::LoadFromOnnx(onnx_path, arena, kind, mlp, cnn, input_shape, input_rank).status !=
                    OnnxImporter::ImportStatus::Ok ||
                kind != NkLoader::NetworkKind::Mlp || !mlp)
                return false;

            if (input_count != input_shape[0] * input_shape[1])
                return false;

            Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
            Tensor out = TensorFactory::Create2D(arena, input_shape[0], output_count / input_shape[0]);
            if (!input.data || !out.data)
                return false;

            float* input_data = static_cast<float*>(input.data);
            for (uint32_t i = 0; i < input_count; ++i)
                input_data[i] = input_values[i];

            mlp->forward(input, out, arena);
            if (!out.data || out.num_elements != output_count)
                return false;

            std::memcpy(output, out.data, static_cast<std::size_t>(output_count) * sizeof(float));
            return true;
        }

        bool ForwardCnnNk(const char* nk_path,
                          const std::array<uint32_t, kMaxTensorRank>& input_shape,
                          const float* input_values,
                          uint32_t input_count,
                          float* output,
                          uint32_t& output_count,
                          Arena& arena)
        {
            CNNNetwork* network = nullptr;
            std::array<uint32_t, kMaxTensorRank> loaded_shape{};
            uint32_t input_rank = 0;

            if (NkLoader::LoadCNN(nk_path, arena, network, loaded_shape, input_rank).status !=
                    NkLoader::LoadStatus::Ok ||
                !network || !network->IsValid())
                return false;

            const uint32_t required = input_shape[0] * input_shape[1] * input_shape[2];
            if (input_count != required)
                return false;

            float input_buffer[NkFormat::kMaxCaseFloats] = {};
            for (uint32_t i = 0; i < input_count; ++i)
                input_buffer[i] = input_values[i];

            Tensor input = MakeNhwcInput(input_buffer, input_shape[0], input_shape[1], input_shape[2]);
            Tensor& out = network->forward(input, arena);
            if (!out.data)
                return false;

            output_count = out.num_elements;
            std::memcpy(output, out.data, static_cast<std::size_t>(output_count) * sizeof(float));
            return true;
        }

        bool ForwardCnnOnnx(const char* onnx_path,
                            const std::array<uint32_t, kMaxTensorRank>& input_shape,
                            const float* input_values,
                            uint32_t input_count,
                            float* output,
                            uint32_t& output_count,
                            Arena& arena)
        {
            MLPNetwork* mlp = nullptr;
            CNNNetwork* network = nullptr;
            NkLoader::NetworkKind kind = NkLoader::NetworkKind::Unknown;
            std::array<uint32_t, kMaxTensorRank> loaded_shape{};
            uint32_t input_rank = 0;

            if (OnnxImporter::LoadFromOnnx(onnx_path, arena, kind, mlp, network, loaded_shape, input_rank).status !=
                    OnnxImporter::ImportStatus::Ok ||
                kind != NkLoader::NetworkKind::Cnn || !network)
                return false;

            const uint32_t required = input_shape[0] * input_shape[1] * input_shape[2];
            if (input_count != required)
                return false;

            float input_buffer[NkFormat::kMaxCaseFloats] = {};
            for (uint32_t i = 0; i < input_count; ++i)
                input_buffer[i] = input_values[i];

            Tensor input = MakeNhwcInput(input_buffer, input_shape[0], input_shape[1], input_shape[2]);
            Tensor& out = network->forward(input, arena);
            if (!out.data)
                return false;

            output_count = out.num_elements;
            std::memcpy(output, out.data, static_cast<std::size_t>(output_count) * sizeof(float));
            return true;
        }

        bool CompareOutputs(const float* nk_out,
                            const float* onnx_out,
                            uint32_t count,
                            float tol,
                            const char* label)
        {
            for (uint32_t i = 0; i < count; ++i)
            {
                if (!FloatNear(nk_out[i], onnx_out[i], tol))
                {
                    std::cout << "FAIL " << label << ": nk/onnx mismatch at out[" << i << "] "
                              << nk_out[i] << " vs " << onnx_out[i] << "\n";
                    return false;
                }
            }
            return true;
        }
    }

    RunSummary RunModelTests(const char* nk_path)
    {
        RunSummary summary{};

        char path_buffer[NkLoader::kMaxPathLen] = {};
        const char* resolved = ResolveModelPath(nk_path, path_buffer, sizeof(path_buffer));

        NkLoader::ParsedModel parsed{};
        const NkLoader::LoadResult parse_result = NkLoader::ParseFile(resolved, parsed);
        if (parse_result.status != NkLoader::LoadStatus::Ok)
        {
            std::cout << "  Model parse failed (" << resolved << "): "
                      << (parse_result.message ? parse_result.message
                                               : NkLoader::StatusMessage(parse_result.status))
                      << "\n";
            ++summary.failed;
            return summary;
        }

        NkLoader::TestSuite tests{};
        const NkLoader::LoadResult test_result = NkLoader::ReadTestSuite(resolved, tests);
        if (test_result.status != NkLoader::LoadStatus::Ok)
        {
            std::cout << "  Test load failed (" << resolved << "): "
                      << (test_result.message ? test_result.message
                                              : NkLoader::StatusMessage(test_result.status))
                      << "\n";
            ++summary.failed;
            return summary;
        }

        const NkLoader::NetworkKind kind = parse_result.kind;
        const std::size_t arena_capacity = ArenaCapacityForModel(parsed);
        unsigned char* arena_buffer = ArenaBufferForCapacity(arena_capacity);

        std::cout << "Model: " << resolved << "\n";
        std::cout << "Embedded cases: " << tests.num_cases << "\n";
        std::cout << "Output tolerance: " << tests.tolerance << "\n";

        for (uint32_t i = 0; i < tests.num_cases; ++i)
        {
            const NkLoader::TestCase& test_case = tests.cases[i];
            std::cout << "\nCase: " << test_case.name << "\n";

            Arena arena;
            arena.init(arena_buffer, arena_capacity);

            if (kind == NkLoader::NetworkKind::Mlp)
            {
                MLPNetwork* network = nullptr;
                std::array<uint32_t, kMaxTensorRank> input_shape{};
                uint32_t input_rank = 0;

                if (NkLoader::LoadMLP(resolved, arena, network, input_shape, input_rank).status !=
                        NkLoader::LoadStatus::Ok ||
                    !network || !network->IsValid())
                {
                    std::cout << "FAIL " << test_case.name << ": could not load MLP weights\n";
                    ++summary.failed;
                    continue;
                }

                if (RunMlpCase(*network, parsed, input_shape, test_case, tests.tolerance, arena))
                    ++summary.passed;
                else
                    ++summary.failed;
            }
            else if (kind == NkLoader::NetworkKind::Cnn)
            {
                CNNNetwork* network = nullptr;
                std::array<uint32_t, kMaxTensorRank> input_shape{};
                uint32_t input_rank = 0;

                if (NkLoader::LoadCNN(resolved, arena, network, input_shape, input_rank).status !=
                        NkLoader::LoadStatus::Ok ||
                    !network || !network->IsValid())
                {
                    std::cout << "FAIL " << test_case.name << ": could not load CNN weights\n";
                    ++summary.failed;
                    continue;
                }

                if (RunCnnCase(*network, input_shape, test_case, tests.tolerance, arena))
                    ++summary.passed;
                else
                    ++summary.failed;
            }
            else
            {
                std::cout << "FAIL " << test_case.name << ": unsupported network kind\n";
                ++summary.failed;
            }
        }

        return summary;
    }

    RunSummary RunNkOnnxParity(const char* nk_path, const char* onnx_path)
    {
        RunSummary summary{};

        char nk_buffer[NkLoader::kMaxPathLen] = {};
        char onnx_buffer[NkLoader::kMaxPathLen] = {};
        const char* resolved_nk = ResolveModelPath(nk_path, nk_buffer, sizeof(nk_buffer));
        const char* resolved_onnx = ResolveModelPath(onnx_path, onnx_buffer, sizeof(onnx_buffer));

        NkLoader::ParsedModel parsed{};
        if (NkLoader::ParseFile(resolved_nk, parsed).status != NkLoader::LoadStatus::Ok)
        {
            std::cout << "FAIL ONNX parity: cannot parse " << resolved_nk << "\n";
            ++summary.failed;
            return summary;
        }

        NkLoader::TestSuite tests{};
        if (NkLoader::ReadTestSuite(resolved_nk, tests).status != NkLoader::LoadStatus::Ok)
        {
            std::cout << "FAIL ONNX parity: no embedded tests in " << resolved_nk << "\n";
            ++summary.failed;
            return summary;
        }

        const NkLoader::NetworkKind kind = parsed.header.network_kind == NkFormat::NetworkKind::Mlp
                                               ? NkLoader::NetworkKind::Mlp
                                               : NkLoader::NetworkKind::Cnn;

        std::cout << "ONNX parity model: " << resolved_nk << "\n";
        std::cout << "  ONNX model: " << resolved_onnx << "\n";

        const std::size_t arena_capacity = ArenaCapacityForModel(parsed);
        unsigned char* arena_buffer = ArenaBufferForCapacity(arena_capacity);

        for (uint32_t i = 0; i < tests.num_cases; ++i)
        {
            const NkLoader::TestCase& test_case = tests.cases[i];

            float nk_out[NkFormat::kMaxCaseFloats] = {};
            float onnx_out[NkFormat::kMaxCaseFloats] = {};
            uint32_t nk_out_count = test_case.output_count;
            uint32_t onnx_out_count = test_case.output_count;

            Arena nk_arena;
            nk_arena.init(arena_buffer, arena_capacity);

            bool ok = false;
            if (kind == NkLoader::NetworkKind::Mlp)
            {
                nk_out_count = NkLoader::OutputElements(parsed);
                if (!ForwardMlpNk(resolved_nk, test_case.input, test_case.input_count, nk_out, nk_out_count, nk_arena))
                {
                    std::cout << "FAIL ONNX parity " << test_case.name << ": NK forward failed\n";
                    ++summary.failed;
                    continue;
                }

                Arena onnx_arena;
                onnx_arena.init(arena_buffer, arena_capacity);
                if (!ForwardMlpOnnx(resolved_onnx, test_case.input, test_case.input_count, onnx_out, nk_out_count,
                                    onnx_arena))
                {
                    std::cout << "FAIL ONNX parity " << test_case.name << ": ONNX forward failed\n";
                    ++summary.failed;
                    continue;
                }

                ok = CompareOutputs(nk_out, onnx_out, nk_out_count, tests.tolerance, test_case.name);
            }
            else
            {
                std::array<uint32_t, kMaxTensorRank> input_shape{};
                for (uint32_t r = 0; r < parsed.header.input_rank; ++r)
                    input_shape[r] = parsed.header.input_shape[r];

                if (!ForwardCnnNk(resolved_nk, input_shape, test_case.input, test_case.input_count, nk_out,
                                  nk_out_count, nk_arena))
                {
                    std::cout << "FAIL ONNX parity " << test_case.name << ": NK forward failed\n";
                    ++summary.failed;
                    continue;
                }

                Arena onnx_arena;
                onnx_arena.init(arena_buffer, arena_capacity);
                if (!ForwardCnnOnnx(resolved_onnx, input_shape, test_case.input, test_case.input_count, onnx_out,
                                    onnx_out_count, onnx_arena))
                {
                    std::cout << "FAIL ONNX parity " << test_case.name << ": ONNX forward failed\n";
                    ++summary.failed;
                    continue;
                }

                if (nk_out_count != onnx_out_count)
                {
                    std::cout << "FAIL ONNX parity " << test_case.name << ": output size " << nk_out_count << " vs "
                              << onnx_out_count << "\n";
                    ++summary.failed;
                    continue;
                }

                ok = CompareOutputs(nk_out, onnx_out, nk_out_count, tests.tolerance, test_case.name);
            }

            if (ok)
            {
                std::cout << "PASS ONNX parity " << test_case.name << " (nk vs onnx, " << nk_out_count
                          << " outputs)\n";
                ++summary.passed;
            }
            else
                ++summary.failed;
        }

        return summary;
    }
}
