#pragma once

#include <cstddef>
#include <cstdint>

// Int8 QuantOps Conv2D im2col (NETKIT_IM2COL 1/2). Pad value is the raw int8
// zero-point contribution (0 when offsets are folded into bias).

std::size_t ConvPartialIm2ColS8WorkspaceBytes(uint32_t kernel_h,
                                              uint32_t kernel_w,
                                              uint32_t in_channels);

std::size_t ConvFullIm2ColS8WorkspaceBytes(uint32_t out_h,
                                           uint32_t out_w,
                                           uint32_t kernel_h,
                                           uint32_t kernel_w,
                                           uint32_t in_channels);

// Arena bytes for the selected NETKIT_IM2COL policy (0 when Direct).
std::size_t Conv2dQuantIm2ColWorkspaceBytes(uint32_t out_h,
                                            uint32_t out_w,
                                            uint32_t kernel_h,
                                            uint32_t kernel_w,
                                            uint32_t in_channels,
                                            int stride);

bool ConvPartialIm2ColS8Forward(const int8_t* input,
                                const int8_t* weights_ohwi,
                                const int32_t* bias,
                                int8_t* output,
                                int8_t* patch_workspace,
                                uint32_t in_h,
                                uint32_t in_w,
                                uint32_t in_c,
                                uint32_t out_h,
                                uint32_t out_w,
                                int out_channels,
                                uint32_t kernel_h,
                                uint32_t kernel_w,
                                int stride,
                                int pad_h,
                                int pad_w,
                                int8_t pad_value,
                                bool plain_mac,
                                int32_t input_offset,
                                const int32_t* multipliers,
                                const int32_t* shifts,
                                int32_t output_zero_point,
                                int32_t baked_min,
                                int32_t baked_max);

bool ConvFullIm2ColS8Forward(const int8_t* input,
                             const int8_t* weights_ohwi,
                             const int32_t* bias,
                             int8_t* output,
                             int8_t* col_workspace,
                             uint32_t in_h,
                             uint32_t in_w,
                             uint32_t in_c,
                             uint32_t out_h,
                             uint32_t out_w,
                             int out_channels,
                             uint32_t kernel_h,
                             uint32_t kernel_w,
                             int stride,
                             int pad_h,
                             int pad_w,
                             int8_t pad_value,
                             bool plain_mac,
                             int32_t input_offset,
                             const int32_t* multipliers,
                             const int32_t* shifts,
                             int32_t output_zero_point,
                             int32_t baked_min,
                             int32_t baked_max);
