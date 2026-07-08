#pragma once
#include "kernel_activation.hpp"
#include "tensor_access.hpp"
#include <cstdint>

struct DepthwiseConv2D
{
    int kernel_h = 3;
    int kernel_w = 3;
    int stride = 1;
    int pad_h = 0;
    int pad_w = 0;
    int pad_h_end = -1;
    int pad_w_end = -1;
    int channels = 0;

    float* weights; // [ch][kh][kw]
    float* bias;    // [ch]
    const int8_t* weights_q = nullptr;
    const int32_t* bias_q = nullptr;

    bool forward(const Tensor& input,
                 Tensor& output,
                 NetkitKernelActivation fuse_activation = NetkitKernelActivation::None);
};
