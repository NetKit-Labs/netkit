// MNIST MLP invoke-time benchmark against TensorFlow Lite Micro (TFLM).
//
// Uses the same trained 784->128->10 weights as models/mnist_mlp.nk and
// 10 embedded MNIST test vectors (one per digit). Times each Invoke() call
// with std::chrono and prints run statistics at the end.

#include "benchmark_stats.hpp"
#include "generated/mnist_mlp_model_data.h"
#include "generated/mnist_test_images.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kTensorArenaSize = 128 * 1024;
constexpr int kRuns = BenchmarkStats::kDefaultRuns;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

using MnistOpResolver = tflite::MicroMutableOpResolver<4>;

int ArgMax10(const float* values) {
  int best = 0;
  float max_val = values[0];
  for (int i = 1; i < 10; ++i) {
    if (values[i] > max_val) {
      max_val = values[i];
      best = i;
    }
  }
  return best;
}

TfLiteStatus RunBenchmark() {
  MnistOpResolver op_resolver;
  if (op_resolver.AddFullyConnected() != kTfLiteOk) {
    MicroPrintf("Failed to register FULLY_CONNECTED");
    return kTfLiteError;
  }
  if (op_resolver.AddSoftmax() != kTfLiteOk) {
    MicroPrintf("Failed to register SOFTMAX");
    return kTfLiteError;
  }

  const tflite::Model* model = tflite::GetModel(g_mnist_mlp_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d != supported %d", model->version(),
                TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return kTfLiteError;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (input == nullptr || output == nullptr) {
    MicroPrintf("Missing input or output tensor");
    return kTfLiteError;
  }
  if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) {
    MicroPrintf("Expected float32 tensors");
    return kTfLiteError;
  }

  MicroPrintf("TFLM MNIST MLP benchmark");
  MicroPrintf("  model bytes: %u", g_mnist_mlp_model_data_size);
  MicroPrintf("  images:      %d per run", kMnistBenchmarkImageCount);
  MicroPrintf("  runs:        %d (discard first invoke each run)", kRuns);
  MicroPrintf("  arena bytes: %d", kTensorArenaSize);

  std::array<double, BenchmarkStats::kMaxRuns> run_averages_us{};
  int correct = 0;

  for (int run = 0; run < kRuns; ++run) {
    double run_total_us = 0.0;
    int counted = 0;

    for (int i = 0; i < kMnistBenchmarkImageCount; ++i) {
      const MnistBenchmarkSample& sample = kMnistBenchmarkImages[i];

      if (input->bytes < kMnistBenchmarkInputSize * sizeof(float)) {
        MicroPrintf("Input tensor too small");
        return kTfLiteError;
      }
      std::memcpy(input->data.f, sample.pixels,
                  kMnistBenchmarkInputSize * sizeof(float));

      const auto start = std::chrono::steady_clock::now();
      const TfLiteStatus invoke_status = interpreter.Invoke();
      const auto end = std::chrono::steady_clock::now();

      if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed on run %d image %d (%s)", run + 1, i,
                    sample.name);
        return kTfLiteError;
      }

      const double elapsed_us =
          std::chrono::duration<double, std::micro>(end - start).count();

      if (i > 0) {
        run_total_us += elapsed_us;
        ++counted;
      }

      if (run == kRuns - 1) {
        const int predicted = ArgMax10(output->data.f);
        if (predicted == sample.label) {
          ++correct;
        }
      }
    }

    run_averages_us[static_cast<size_t>(run)] =
        run_total_us / static_cast<double>(counted);
  }

  MicroPrintf("  accuracy:    %d/%d on final run", correct, kMnistBenchmarkImageCount);
  BenchmarkStats::PrintSummary("tflm", "mlp", "reference",
                               BenchmarkStats::Compute(run_averages_us.data(), kRuns));

  return correct == kMnistBenchmarkImageCount ? kTfLiteOk : kTfLiteError;
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  tflite::InitializeTarget();
  const TfLiteStatus status = RunBenchmark();
  return status == kTfLiteOk ? 0 : 1;
}
