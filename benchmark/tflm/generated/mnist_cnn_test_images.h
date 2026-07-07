#pragma once

#include <cstdint>

constexpr int kMnistCnnBenchmarkImageCount = 10;
constexpr int kMnistCnnBenchmarkInputSize = 784;
constexpr bool kMnistCnnBenchmarkHasInt8Pixels = true;
constexpr float kMnistCnnBenchmarkInputScale = 0.00787402f;
constexpr int kMnistCnnBenchmarkInputZeroPoint = 0;

struct MnistCnnBenchmarkSample {
  const char* name;
  int label;
  const float* pixels;
  const int8_t* pixels_i8;
};

extern const MnistCnnBenchmarkSample kMnistCnnBenchmarkImages[10];

