#pragma once

#include <cstdint>

// Quantized inference output: int8 activations end-to-end; float32 only at the boundary when requested.
enum class QuantOutputFormat : uint8_t
{
    Int8 = 0,
    Float32 = 1,
};

namespace QuantOps
{
// TFLite int8 softmax output quantization (fixed by spec).
constexpr float kSoftmaxOutputScale = 1.0f / 256.0f;
constexpr int32_t kSoftmaxOutputZeroPoint = -128;

struct SoftmaxS8Params
{
    int32_t mult = 0;
    int32_t shift = 0;
    int32_t diff_min = 0;
};

SoftmaxS8Params ComputeSoftmaxS8Params(float logit_scale, float beta = 1.0f);

void SoftmaxS8(const int8_t* logits,
               uint32_t count,
               float logit_scale,
               int8_t* output);

float DequantizeSoftmaxOutput(int8_t value);

void DequantizeSoftmaxOutput(const int8_t* src, float* dst, uint32_t count);

uint32_t ArgMaxInt8(const int8_t* values, uint32_t count);

}  // namespace QuantOps
