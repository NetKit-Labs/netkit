#include "test_onnx.hpp"

#include <iostream>

namespace
{
    struct ParitySpec
    {
        const char* nk_path;
        const char* onnx_path;
    };

    constexpr ParitySpec kParitySpecs[] = {
        {"models/test_mlp.nk", "models/test_mlp.onnx"},
        {"models/mlp_hand.nk", "models/mlp_hand.onnx"},
        {"models/test_cnn.nk", "models/test_cnn.onnx"},
        {"models/cnn_4x4_single.nk", "models/cnn_4x4_single.onnx"},
        {"models/cnn_hand.nk", "models/cnn_hand.onnx"},
        {"models/mnist_mlp.nk", "models/mnist_mlp.onnx"},
        {"models/mnist_cnn.nk", "models/mnist_cnn.onnx"},
    };
}

NkRegression::RunSummary run_onnx_import_tests()
{
    NkRegression::RunSummary summary{};

    std::cout << "\n============================\n";
    std::cout << " ONNX PARITY TESTS\n";
    std::cout << "============================\n";

    for (const ParitySpec& spec : kParitySpecs)
    {
        const NkRegression::RunSummary part = NkRegression::RunNkOnnxParity(spec.nk_path, spec.onnx_path);
        summary.passed += part.passed;
        summary.failed += part.failed;
    }

    return summary;
}
