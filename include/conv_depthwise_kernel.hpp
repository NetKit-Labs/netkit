#pragma once

#include "kernel_activation.hpp"

#include <cstdint>

struct Tensor;

bool ConvDepthwiseForward(const Tensor& input,
                          float* weights,
                          float* bias,
                          int kernel_h,
                          int kernel_w,
                          int stride,
                          int pad_h,
                          int pad_w,
                          int pad_h_end,
                          int pad_w_end,
                          int channels,
                          NetkitKernelActivation fuse_activation,
                          Tensor& output);
