#pragma once
#include "tensor_access.hpp"
#include <cstdint>

struct Conv2D
{
    int kernel_size = 3;
    int stride = 1;
    int in_channels;
    int out_channels;

    float* weights; // [out][kh][kw][in]
    float* bias;    // [out]

    void forward(const Tensor& input, Tensor& output);
};
