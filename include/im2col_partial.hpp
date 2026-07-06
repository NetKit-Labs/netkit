#pragma once

#include "kernel_activation.hpp"

#include <cstddef>
#include <cstdint>

std::size_t ConvPartialIm2ColWorkspaceBytes(uint32_t kernel_h,
                                          uint32_t kernel_w,
                                          uint32_t in_channels);

bool ConvPartialIm2ColForward(const float* in,
                              const float* weights_oki,
                              const float* bias,
                              float* out,
                              float* patch_workspace,
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
