#include "vectors_loader.hpp"
#include "test_mnist.hpp"
#include "test_onnx.hpp"
#include <iostream>

VectorsLoader::RunSummary run_all_tests()
{
    VectorsLoader::RunSummary total{};

    auto merge = [&](const VectorsLoader::RunSummary& part) {
        total.passed += part.passed;
        total.failed += part.failed;
    };

    std::cout << "\n============================\n";
    std::cout << " C++ API TESTS\n";
    std::cout << "============================\n";

    std::cout << "\n============================\n";
    std::cout << " MLP TESTS\n";
    std::cout << "============================\n";
    merge(VectorsLoader::RunVectorsFile("models/test_mlp.vectors.json"));
    merge(VectorsLoader::RunVectorsFile("models/mlp_hand.vectors.json"));

    std::cout << "\n============================\n";
    std::cout << " CNN TESTS\n";
    std::cout << "============================\n";
    merge(VectorsLoader::RunVectorsFile("models/test_cnn.vectors.json"));
    merge(VectorsLoader::RunVectorsFile("models/cnn_4x4_single.vectors.json"));
    merge(VectorsLoader::RunVectorsFile("models/cnn_hand.vectors.json"));

    std::cout << "\n============================\n";
    std::cout << " MNIST MLP TESTS\n";
    std::cout << "============================\n";
    merge(run_mnist_tests());

    std::cout << "\n============================\n";
    std::cout << " MNIST CNN TESTS\n";
    std::cout << "============================\n";
    merge(run_mnist_cnn_tests());

    merge(run_onnx_import_tests());

    return total;
}
