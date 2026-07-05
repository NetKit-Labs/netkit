#pragma once

#include <cstdint>

constexpr int kMnistBenchmarkImageCount = 10;
constexpr int kMnistBenchmarkInputSize = 784;

struct MnistBenchmarkSample {
  const char* name;
  int label;
  const float* pixels;
};

extern const MnistBenchmarkSample kMnistBenchmarkImages[10];

