#pragma once
#include <cstdint>
#include "tensor.hpp"

// raw float access (returns nullptr if t.type != DataType::Float32)
float* tensor_data_f32(Tensor& t);
const float* tensor_data_f32(const Tensor& t);

// raw int8 access (returns nullptr if t.type != DataType::Int8)
int8_t* tensor_data_i8(Tensor& t);
const int8_t* tensor_data_i8(const Tensor& t);

// raw int32 access (returns nullptr if t.type != DataType::Int32)
int32_t* tensor_data_i32(Tensor& t);
const int32_t* tensor_data_i32(const Tensor& t);

// NHWC indexing (H, W, C)
uint32_t index_nhwc(const Tensor& t, uint32_t h, uint32_t w, uint32_t c);
