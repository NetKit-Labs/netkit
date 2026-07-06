#pragma once

#include "kernel_activation.hpp"

#include <cstddef>
#include <cstdint>

struct Tensor;

// Internal Conv2D lowering — policy selects direct / partial im2col / full im2col.
// Not part of the public C/C++ API.

enum class Conv2dExecMode : uint8_t
{
    Direct = 0,
    PartialIm2Col = 1,
    FullIm2Col = 2,
};

Conv2dExecMode SelectConv2dExecMode(int kernel_h,
                                    int kernel_w,
                                    int stride,
                                    uint32_t in_channels,
                                    uint32_t out_h,
                                    uint32_t out_w);

std::size_t Conv2dWorkspaceBytes(uint32_t out_h,
                                 uint32_t out_w,
                                 uint32_t kernel_h,
                                 uint32_t kernel_w,
                                 uint32_t in_channels,
                                 int stride);

bool Conv2dDispatchForward(const Tensor& input,
                           float* weights,
                           float* bias,
                           int kernel_size,
                           int stride,
                           int pad_h,
                           int pad_w,
                           int pad_h_end,
                           int pad_w_end,
                           int in_channels,
                           int out_channels,
                           NetkitKernelActivation fuse_activation,
                           Tensor& output,
                           const float* weights_hwio);
