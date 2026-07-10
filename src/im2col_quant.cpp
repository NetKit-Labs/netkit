#include "im2col_quant.hpp"

#include "conv_im2col_policy.hpp"
#include "quant_integer.hpp"

#include <cstddef>
#include <cstdint>

namespace
{
    void Im2ColPatchNhwcS8(const int8_t* in,
                           int8_t* col,
                           uint32_t in_h,
                           uint32_t in_w,
                           uint32_t in_ch,
                           uint32_t oh,
                           uint32_t ow,
                           uint32_t kernel_h,
                           uint32_t kernel_w,
                           int stride,
                           int pad_h,
                           int pad_w,
                           int8_t pad_value)
    {
        uint32_t k_idx = 0;
        for (uint32_t kh = 0; kh < kernel_h; ++kh)
        {
            const int ih = static_cast<int>(oh) * stride + static_cast<int>(kh) - pad_h;
            for (uint32_t kw = 0; kw < kernel_w; ++kw)
            {
                const int iw = static_cast<int>(ow) * stride + static_cast<int>(kw) - pad_w;
                const bool in_bounds =
                    ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 && iw < static_cast<int>(in_w);
                for (uint32_t ic = 0; ic < in_ch; ++ic)
                {
                    int8_t value = pad_value;
                    if (in_bounds)
                    {
                        value = in[(static_cast<uint32_t>(ih) * in_w + static_cast<uint32_t>(iw)) *
                                       in_ch +
                                   ic];
                    }
                    col[k_idx++] = value;
                }
            }
        }
    }

    void Im2ColNhwcS8(const int8_t* in,
                      int8_t* col,
                      uint32_t in_h,
                      uint32_t in_w,
                      uint32_t in_ch,
                      uint32_t out_h,
                      uint32_t out_w,
                      uint32_t kernel_h,
                      uint32_t kernel_w,
                      int stride,
                      int pad_h,
                      int pad_w,
                      int8_t pad_value)
    {
        const uint32_t patch_elems = kernel_h * kernel_w * in_ch;
        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                const uint32_t spatial_idx = oh * out_w + ow;
                Im2ColPatchNhwcS8(in,
                                  col + spatial_idx * patch_elems,
                                  in_h,
                                  in_w,
                                  in_ch,
                                  oh,
                                  ow,
                                  kernel_h,
                                  kernel_w,
                                  stride,
                                  pad_h,
                                  pad_w,
                                  pad_value);
            }
        }
    }

    int32_t DotProductS8(const int8_t* weights,
                         const int8_t* patch,
                         uint32_t count,
                         bool plain_mac,
                         int32_t input_offset)
    {
        int32_t acc = 0;
        if (plain_mac)
        {
            for (uint32_t i = 0; i < count; ++i)
                acc += static_cast<int32_t>(weights[i]) * static_cast<int32_t>(patch[i]);
        }
        else
        {
            for (uint32_t i = 0; i < count; ++i)
            {
                acc += static_cast<int32_t>(weights[i]) *
                       (static_cast<int32_t>(patch[i]) + input_offset);
            }
        }
        return acc;
    }
}  // namespace

std::size_t ConvPartialIm2ColS8WorkspaceBytes(uint32_t kernel_h,
                                              uint32_t kernel_w,
                                              uint32_t in_channels)
{
    const uint32_t patch = kernel_h * kernel_w * in_channels;
    if (patch == 0)
        return 0;
    return static_cast<std::size_t>(patch) * sizeof(int8_t);
}

std::size_t ConvFullIm2ColS8WorkspaceBytes(uint32_t out_h,
                                           uint32_t out_w,
                                           uint32_t kernel_h,
                                           uint32_t kernel_w,
                                           uint32_t in_channels)
{
    const uint32_t patch = kernel_h * kernel_w * in_channels;
    const uint32_t spatial = out_h * out_w;
    if (patch == 0 || spatial == 0)
        return 0;
    return static_cast<std::size_t>(patch) * static_cast<std::size_t>(spatial) * sizeof(int8_t);
}

std::size_t Conv2dQuantIm2ColWorkspaceBytes(uint32_t out_h,
                                            uint32_t out_w,
                                            uint32_t kernel_h,
                                            uint32_t kernel_w,
                                            uint32_t in_channels,
                                            int stride)
{
    const Conv2dExecMode mode = SelectConv2dExecMode(static_cast<int>(kernel_h),
                                                     static_cast<int>(kernel_w),
                                                     stride,
                                                     in_channels,
                                                     out_h,
                                                     out_w);
    if (mode == Conv2dExecMode::Direct)
        return 0;
    if (!ConvIm2ColVolumeWarrants(kernel_h, kernel_w, in_channels, out_h, out_w))
        return 0;
    if (mode == Conv2dExecMode::PartialIm2Col)
        return ConvPartialIm2ColS8WorkspaceBytes(kernel_h, kernel_w, in_channels);
    return ConvFullIm2ColS8WorkspaceBytes(out_h, out_w, kernel_h, kernel_w, in_channels);
}

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
                                int32_t baked_max)
{
    if (!input || !weights_ohwi || !bias || !output || !patch_workspace || !multipliers || !shifts)
        return false;

    const uint32_t patch_elems = kernel_h * kernel_w * in_c;
    const uint32_t oc_count = static_cast<uint32_t>(out_channels);

    for (uint32_t oh = 0; oh < out_h; ++oh)
    {
        for (uint32_t ow = 0; ow < out_w; ++ow)
        {
            Im2ColPatchNhwcS8(input,
                              patch_workspace,
                              in_h,
                              in_w,
                              in_c,
                              oh,
                              ow,
                              kernel_h,
                              kernel_w,
                              stride,
                              pad_h,
                              pad_w,
                              pad_value);

            const uint32_t out_spatial_base = (oh * out_w + ow) * oc_count;
            for (int oc = 0; oc < out_channels; ++oc)
            {
                const int8_t* wt_row = weights_ohwi + static_cast<uint32_t>(oc) * patch_elems;
                const int32_t acc = bias[oc] + DotProductS8(wt_row,
                                                            patch_workspace,
                                                            patch_elems,
                                                            plain_mac,
                                                            input_offset);
                output[out_spatial_base + static_cast<uint32_t>(oc)] =
                    QuantInteger::RequantizeAccToInt8(
                        acc, multipliers[oc], shifts[oc], output_zero_point, baked_min, baked_max);
            }
        }
    }
    return true;
}

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
                             int32_t baked_max)
{
    if (!input || !weights_ohwi || !bias || !output || !col_workspace || !multipliers || !shifts)
        return false;

    const uint32_t patch_elems = kernel_h * kernel_w * in_c;
    const uint32_t out_spatial = out_h * out_w;
    const uint32_t oc_count = static_cast<uint32_t>(out_channels);

    Im2ColNhwcS8(input,
                 col_workspace,
                 in_h,
                 in_w,
                 in_c,
                 out_h,
                 out_w,
                 kernel_h,
                 kernel_w,
                 stride,
                 pad_h,
                 pad_w,
                 pad_value);

    for (uint32_t s = 0; s < out_spatial; ++s)
    {
        const int8_t* patch = col_workspace + s * patch_elems;
        const uint32_t out_base = s * oc_count;
        for (int oc = 0; oc < out_channels; ++oc)
        {
            const int8_t* wt_row = weights_ohwi + static_cast<uint32_t>(oc) * patch_elems;
            const int32_t acc =
                bias[oc] + DotProductS8(wt_row, patch, patch_elems, plain_mac, input_offset);
            output[out_base + static_cast<uint32_t>(oc)] = QuantInteger::RequantizeAccToInt8(
                acc, multipliers[oc], shifts[oc], output_zero_point, baked_min, baked_max);
        }
    }
    return true;
}
