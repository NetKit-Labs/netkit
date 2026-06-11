#include "tensor_access.hpp"

float* tensor_data_f32(Tensor& t)
{
    return static_cast<float*>(t.data);
}

const float* tensor_data_f32(const Tensor& t)
{
    return static_cast<const float*>(t.data);
}

uint32_t index_nhwc(const Tensor& t, uint32_t h, uint32_t w, uint32_t c)
{
    const uint32_t W = t.shape[1];
    const uint32_t C = t.shape[2];
    return (h * W + w) * C + c;
}
