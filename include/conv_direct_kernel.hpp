#pragma once

#include "kernel_activation.hpp"

#include <cstdint>

bool ConvDirectForward3x3S1P0(const float* in,
                              float* weights,
                              const float* bias,
                              float* out,
                              uint32_t in_w,
                              uint32_t in_ch,
                              uint32_t out_h,
                              uint32_t out_w,
                              uint32_t out_ch,
                              int out_channels,
                              NetkitKernelActivation fuse_activation);

bool ConvDirectForwardNoPad(const float* in,
                            float* weights,
                            const float* bias,
                            float* out,
                            uint32_t in_w,
                            uint32_t in_ch,
                            uint32_t out_h,
                            uint32_t out_w,
                            uint32_t out_ch,
                            int kernel_size,
                            int stride,
                            int out_channels,
                            NetkitKernelActivation fuse_activation);

bool ConvDirectForwardPadded(const float* in,
                             float* weights,
                             const float* bias,
                             float* out,
                             uint32_t in_h,
                             uint32_t in_w,
                             uint32_t in_ch,
                             uint32_t out_h,
                             uint32_t out_w,
                             uint32_t out_ch,
                             int kernel_size,
                             int stride,
                             int pad_h,
                             int pad_w,
                             int out_channels,
                             NetkitKernelActivation fuse_activation);

bool ConvDirectTryInputStationaryForward(const float* in,
                                         const float* weights_hwio,
                                         const float* bias,
                                         float* out,
                                         uint32_t in_h,
                                         uint32_t in_w,
                                         uint32_t in_ch,
                                         uint32_t out_h,
                                         uint32_t out_w,
                                         uint32_t out_ch,
                                         uint32_t kernel_h,
                                         uint32_t kernel_w,
                                         int stride,
                                         int pad_h,
                                         int pad_w,
                                         int pad_h_end,
                                         int pad_w_end,
                                         int out_channels,
                                         NetkitKernelActivation fuse_activation);
