#pragma once

#include "kernel_activation.hpp"

#include <cstddef>
#include <cstdint>

struct Arena;
struct Conv2D;
struct Tensor;

// Repack [out, kh, kw, in] (OIHW) -> [kh, kw, in, out] (HWIO) for input-stationary conv.
void RepackConv2dOkiToHwio(const float* oki,
                           float* hwio,
                           uint32_t kernel_h,
                           uint32_t kernel_w,
                           uint32_t in_channels,
                           uint32_t out_channels);

bool RepackConv2dWeights(Conv2D& conv, Arena& arena);

std::size_t Conv2dIm2ColWorkspaceBytes(uint32_t out_h,
                                       uint32_t out_w,
                                       uint32_t kernel_h,
                                       uint32_t kernel_w,
                                       uint32_t in_channels);

bool Conv2dTryIm2ColForward(const float* in,
                            const float* weights_oki,
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

bool Conv2dTryInputStationaryForward(const float* in,
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

bool Conv2dShouldUseIm2Col(uint32_t out_h,
                           uint32_t out_w,
                           uint32_t kernel_h,
                           uint32_t kernel_w,
                           uint32_t in_channels);

bool Conv2dShouldUseInputStationary(uint32_t out_channels, const float* weights_hwio);
