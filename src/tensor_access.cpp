#include "tensor_access.hpp"

float* tensor_data_f32(Tensor& t)
{
    return t.type == DataType::Float32 ? static_cast<float*>(t.data) : nullptr;
}

const float* tensor_data_f32(const Tensor& t)
{
    return t.type == DataType::Float32 ? static_cast<const float*>(t.data) : nullptr;
}

int8_t* tensor_data_i8(Tensor& t)
{
    return t.type == DataType::Int8 ? static_cast<int8_t*>(t.data) : nullptr;
}

const int8_t* tensor_data_i8(const Tensor& t)
{
    return t.type == DataType::Int8 ? static_cast<const int8_t*>(t.data) : nullptr;
}

uint32_t index_nhwc(const Tensor& t, uint32_t h, uint32_t w, uint32_t c)
{
    const uint32_t W = t.shape[1];
    const uint32_t C = t.shape[2];
    return (h * W + w) * C + c;
}
