#include "nk_regression.hpp"
#include "test_onnx.hpp"
#include <iostream>

NkRegression::RunSummary run_all_tests()
{
    NkRegression::RunSummary total{};

    auto merge = [&](const NkRegression::RunSummary& part) {
        total.passed += part.passed;
        total.failed += part.failed;
    };

    std::cout << "\n============================\n";
    std::cout << " C++ API TESTS\n";
    std::cout << "============================\n";

    std::cout << "\n============================\n";
    std::cout << " MLP TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/test_mlp.nk"));
    merge(NkRegression::RunModelTests("models/mlp_hand.nk"));

    std::cout << "\n============================\n";
    std::cout << " CNN TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/test_cnn.nk"));
    merge(NkRegression::RunModelTests("models/cnn_4x4_single.nk"));
    merge(NkRegression::RunModelTests("models/cnn_hand.nk"));

    std::cout << "\n============================\n";
    std::cout << " MNIST MLP TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/mnist_mlp.nk"));

    std::cout << "\n============================\n";
    std::cout << " MNIST CNN TESTS\n";
    std::cout << "============================\n";
    merge(NkRegression::RunModelTests("models/mnist_cnn.nk"));

    merge(run_onnx_import_tests());

    return total;
}
