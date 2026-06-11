#pragma once
#include <cstdint>
#include "tensor.hpp"

// raw float access
float* tensor_data_f32(Tensor& t);
const float* tensor_data_f32(const Tensor& t);

// NHWC indexing (H, W, C)
uint32_t index_nhwc(const Tensor& t, uint32_t h, uint32_t w, uint32_t c);
