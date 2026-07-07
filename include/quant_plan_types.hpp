#pragma once

#include "quant_output.hpp"

#include <cstdint>

namespace CmsisQuantPlan
{

constexpr uint32_t kMaxPerChannel = 512;

enum class LayerKind : uint8_t
{
    Conv2D,
    MaxPool2D,
    FlattenView,
    Dense,
    DenseSoftmax,
};

struct Conv2DPlan
{
    int32_t input_offset = 0;
    int32_t output_offset = 0;
    int32_t stride = 1;
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    bool apply_relu = false;
    int32_t in_h = 0;
    int32_t in_w = 0;
    int32_t in_c = 0;
    int32_t out_h = 0;
    int32_t out_w = 0;
    int32_t out_c = 0;
    int32_t kernel_size = 0;
    int32_t workspace_bytes = 0;
    int32_t* multipliers = nullptr;
    int32_t* shifts = nullptr;
    bool ready = false;
};

struct Pool2DPlan
{
    int32_t stride = 1;
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    int32_t pool_h = 0;
    int32_t pool_w = 0;
    int32_t in_h = 0;
    int32_t in_w = 0;
    int32_t in_c = 0;
    int32_t out_h = 0;
    int32_t out_w = 0;
    bool ready = false;
};

struct FcPlan
{
    int32_t input_offset = 0;
    int32_t filter_offset = 0;
    int32_t output_offset = 0;
    bool apply_relu = false;
    int32_t in_features = 0;
    int32_t out_features = 0;
    int32_t multiplier = 0;
    int32_t shift = 0;
    int32_t workspace_bytes = 0;
    bool ready = false;
};

struct SoftmaxPlan
{
    QuantOps::SoftmaxS8Params params{};
    int32_t row_size = 0;
    bool ready = false;
};

struct LayerPlan
{
    LayerKind kind = LayerKind::Conv2D;
    uint32_t output_elements = 0;
    Conv2DPlan conv{};
    Pool2DPlan pool{};
    FcPlan fc{};
    SoftmaxPlan softmax{};
};

}  // namespace CmsisQuantPlan
