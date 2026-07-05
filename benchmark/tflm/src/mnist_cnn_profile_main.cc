// TFLM MNIST CNN invoke profiler — benchmark-only translation unit.
//
// Linked only into mnist_cnn_profile_benchmark (see Makefile.inc). Never compiled
// into the production mnist_cnn_benchmark binary.

#ifndef NETKIT_TFLM_CNN_PROFILE
#define NETKIT_TFLM_CNN_PROFILE 1
#endif

#include "benchmark_stats.hpp"
#include "generated/mnist_cnn_model_data.h"
#include "generated/mnist_cnn_test_images.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/kernels/reshape.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kTensorArenaSize = 256 * 1024;
constexpr int kRuns = BenchmarkStats::kDefaultRuns;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

using MnistCnnOpResolver = tflite::MicroMutableOpResolver<10>;

constexpr int kMaxProfileEvents = 64;
constexpr int kMaxProfileTags = 32;

class TagTickAggregator : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char* tag) override {
    if (num_events_ >= kMaxProfileEvents) {
      return 0;
    }
    tags_[num_events_] = tag;
    start_ns_[num_events_] = NowNs();
    end_ns_[num_events_] = start_ns_[num_events_];
    return static_cast<uint32_t>(num_events_++);
  }

  void EndEvent(uint32_t event_handle) override {
    if (event_handle >= static_cast<uint32_t>(num_events_)) {
      return;
    }
    end_ns_[event_handle] = NowNs();
  }

  void AccumulateInvoke() {
    for (int i = 0; i < num_events_; ++i) {
      const uint64_t delta = end_ns_[i] - start_ns_[i];
      AddTagNanoseconds(tags_[i], delta);
    }
    num_events_ = 0;
    ++num_invokes_;
  }

  void AddWallMicros(double wall_us) { total_wall_us_ += wall_us; }

  int num_invokes() const { return num_invokes_; }

  double mean_wall_us() const {
    return num_invokes_ > 0 ? total_wall_us_ / static_cast<double>(num_invokes_) : 0.0;
  }

  double NanosToMicros(uint64_t ns) const {
    return static_cast<double>(ns) / 1000.0;
  }

  uint64_t TotalOpNanoseconds() const {
    uint64_t sum = 0;
    for (int i = 0; i < num_tags_; ++i) {
      sum += tag_ns_[i];
    }
    return sum;
  }

  double MeanOpMicrosPerInvoke() const {
    return num_invokes_ > 0
               ? NanosToMicros(TotalOpNanoseconds() /
                                 static_cast<uint64_t>(num_invokes_))
               : 0.0;
  }

  void PrintProfileSummaries() const {
    if (num_invokes_ <= 0) {
      return;
    }

    const double mean_wall = mean_wall_us();
    const double mean_op_us = MeanOpMicrosPerInvoke();
    const double gap_us = mean_wall - mean_op_us;
    const double gap_pct = mean_wall > 0.0 ? (gap_us / mean_wall) * 100.0 : 0.0;

    MicroPrintf(
        "PROFILE_SUMMARY runtime=tflm model=cnn kind=meta tag=wall_clock "
        "mean_us=%.3f invoke_count=%d",
        mean_wall, num_invokes_);
    MicroPrintf(
        "PROFILE_SUMMARY runtime=tflm model=cnn kind=meta tag=op_time_sum "
        "mean_us=%.3f invoke_count=%d",
        mean_op_us, num_invokes_);
    MicroPrintf(
        "PROFILE_SUMMARY runtime=tflm model=cnn kind=meta tag=between_op_overhead "
        "mean_us=%.3f pct=%.1f invoke_count=%d",
        gap_us, gap_pct, num_invokes_);

    struct TagRow {
      const char* tag;
      uint64_t ns;
      double mean_us;
      double pct;
    };
    std::array<TagRow, kMaxProfileTags> rows{};
    int row_count = 0;
    for (int i = 0; i < num_tags_; ++i) {
      const uint64_t tag_mean_ns =
          tag_ns_[i] / static_cast<uint64_t>(num_invokes_);
      const double tag_mean_us = NanosToMicros(tag_mean_ns);
      const double pct = mean_wall > 0.0 ? (tag_mean_us / mean_wall) * 100.0 : 0.0;
      rows[row_count++] = TagRow{tag_names_[i], tag_ns_[i], tag_mean_us, pct};
    }

    std::sort(rows.begin(), rows.begin() + row_count,
              [](const TagRow& a, const TagRow& b) { return a.mean_us > b.mean_us; });

    for (int i = 0; i < row_count; ++i) {
      MicroPrintf(
          "PROFILE_SUMMARY runtime=tflm model=cnn kind=op tag=%s mean_us=%.3f pct=%.1f "
          "invoke_count=%d",
          rows[i].tag, rows[i].mean_us, rows[i].pct, num_invokes_);
    }
  }

 private:
  static uint64_t NowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  void AddTagNanoseconds(const char* tag, uint64_t delta) {
    for (int i = 0; i < num_tags_; ++i) {
      if (std::strcmp(tag_names_[i], tag) == 0) {
        tag_ns_[i] += delta;
        return;
      }
    }
    if (num_tags_ >= kMaxProfileTags) {
      return;
    }
    tag_names_[num_tags_] = tag;
    tag_ns_[num_tags_] = delta;
    ++num_tags_;
  }

  const char* tags_[kMaxProfileEvents]{};
  uint64_t start_ns_[kMaxProfileEvents]{};
  uint64_t end_ns_[kMaxProfileEvents]{};
  int num_events_ = 0;

  const char* tag_names_[kMaxProfileTags]{};
  uint64_t tag_ns_[kMaxProfileTags]{};
  int num_tags_ = 0;

  int num_invokes_ = 0;
  double total_wall_us_ = 0.0;
};

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
  MnistCnnOpResolver op_resolver;
  if (op_resolver.AddConv2D() != kTfLiteOk ||
      op_resolver.AddMaxPool2D() != kTfLiteOk ||
      op_resolver.AddFullyConnected() != kTfLiteOk ||
      op_resolver.AddSoftmax() != kTfLiteOk ||
      op_resolver.AddReshape() != kTfLiteOk) {
    MicroPrintf("Failed to register CNN ops");
    return kTfLiteError;
  }

  const tflite::Model* model = tflite::GetModel(g_mnist_cnn_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d != supported %d", model->version(),
                TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

  TagTickAggregator profiler;
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize, nullptr, &profiler);
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

  MicroPrintf("TFLM MNIST CNN profile benchmark");
  MicroPrintf("  model bytes: %u", g_mnist_cnn_model_data_size);
  MicroPrintf("  images:      %d per run", kMnistCnnBenchmarkImageCount);
  MicroPrintf("  runs:        %d (discard first invoke each run)", kRuns);
  MicroPrintf("  arena bytes: %d", kTensorArenaSize);
  MicroPrintf("  profiler:    MicroProfiler per-op (benchmark-only build)");

  std::array<double, BenchmarkStats::kMaxRuns> run_averages_us{};
  int correct = 0;

  for (int run = 0; run < kRuns; ++run) {
    double run_total_us = 0.0;
    int counted = 0;

    for (int i = 0; i < kMnistCnnBenchmarkImageCount; ++i) {
      const MnistCnnBenchmarkSample& sample = kMnistCnnBenchmarkImages[i];

      if (input->bytes < kMnistCnnBenchmarkInputSize * sizeof(float)) {
        MicroPrintf("Input tensor too small");
        return kTfLiteError;
      }
      std::memcpy(input->data.f, sample.pixels,
                  kMnistCnnBenchmarkInputSize * sizeof(float));

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
        profiler.AddWallMicros(elapsed_us);
        profiler.AccumulateInvoke();
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

  MicroPrintf("  accuracy:    %d/%d on final run", correct, kMnistCnnBenchmarkImageCount);
  BenchmarkStats::PrintSummary("tflm", "cnn", "profile",
                               BenchmarkStats::Compute(run_averages_us.data(), kRuns));
  profiler.PrintProfileSummaries();

  return correct == kMnistCnnBenchmarkImageCount ? kTfLiteOk : kTfLiteError;
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  tflite::InitializeTarget();
  const TfLiteStatus status = RunBenchmark();
  return status == kTfLiteOk ? 0 : 1;
}
