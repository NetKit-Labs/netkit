#include "conv_depthwise_kernel.hpp"

#include "kernel_activation.hpp"
#include "tensor_access.hpp"

#include <cstddef>

namespace
{
    float ConvOutputValue(float sum, NetkitKernelActivation fuse_activation)
    {
        return ApplyKernelActivation(sum, fuse_activation);
    }
}

bool ConvDepthwiseForward(const Tensor& input,
                          float* weights,
                          float* bias,
                          int kernel_h,
                          int kernel_w,
                          int stride,
                          int pad_h,
                          int pad_w,
                          int /*pad_h_end*/,
                          int /*pad_w_end*/,
                          int channels,
                          NetkitKernelActivation fuse_activation,
                          Tensor& output)
{
    float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];
    const int in_h = static_cast<int>(input.shape[0]);
    const int in_w = static_cast<int>(input.shape[1]);
    const uint32_t in_w_u = input.shape[1];
    const uint32_t ch_u = static_cast<uint32_t>(channels);

    for (size_t oh = 0; oh < out_h; ++oh)
    {
        for (size_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t out_spatial_base = (oh * out_w + ow) * ch_u;

            for (int c = 0; c < channels; ++c)
            {
                float sum = bias ? bias[c] : 0.0f;

                for (int kh = 0; kh < kernel_h; ++kh)
                {
                    const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                    if (ih < 0 || ih >= in_h)
                        continue;

                    const uint32_t in_row = static_cast<uint32_t>(ih) * in_w_u;

                    for (int kw = 0; kw < kernel_w; ++kw)
                    {
                        const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                        if (iw < 0 || iw >= in_w)
                            continue;

                        const uint32_t in_idx =
                            (in_row + static_cast<uint32_t>(iw)) * ch_u + static_cast<uint32_t>(c);
                        const uint32_t w_idx =
                            (static_cast<uint32_t>(c) * static_cast<uint32_t>(kernel_h) +
                             static_cast<uint32_t>(kh)) *
                                static_cast<uint32_t>(kernel_w) +
                            static_cast<uint32_t>(kw);
                        sum += in[in_idx] * weights[w_idx];
                    }
                }

                out[out_spatial_base + static_cast<uint32_t>(c)] = ConvOutputValue(sum, fuse_activation);
            }
        }
    }

    return kernel_activation_is_fused(fuse_activation);
}
