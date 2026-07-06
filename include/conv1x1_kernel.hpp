#pragma once

#include "kernel_activation.hpp"

#include <cstdint>

bool Conv1x1Forward(const float* in,
                    const float* weights_oki,
                    const float* bias,
                    float* out,
                    uint32_t in_h,
                    uint32_t in_w,
                    uint32_t in_ch,
                    uint32_t out_h,
                    uint32_t out_w,
                    uint32_t out_ch,
                    int out_channels,
                    NetkitKernelActivation fuse_activation);
