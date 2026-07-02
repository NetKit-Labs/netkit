#include "conv2d.hpp"
#include "netkit_backend.h"

bool Conv2D::forward(const Tensor& input, Tensor& output, NetkitBackendActivation fuse_activation)
{
    if (netkit_cmsis_conv2d_forward(&input,
                                    weights,
                                    bias,
                                    kernel_size,
                                    stride,
                                    pad_h,
                                    pad_w,
                                    in_channels,
                                    out_channels,
                                    fuse_activation,
                                    &output))
        return true;

    float* in  = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    uint32_t outH = output.shape[0];
    uint32_t outW = output.shape[1];

    for (int oc = 0; oc < out_channels; oc++)
    {
        for (uint32_t oh = 0; oh < outH; oh++)
        {
            for (uint32_t ow = 0; ow < outW; ow++)
            {
                float sum = bias ? bias[oc] : 0.0f;

                for (int kh = 0; kh < kernel_size; kh++)
                {
                    for (int kw = 0; kw < kernel_size; kw++)
                    {
                        for (int ic = 0; ic < in_channels; ic++)
                        {
                            const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                            const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                            if (ih < 0 || iw < 0 || ih >= static_cast<int>(input.shape[0]) ||
                                iw >= static_cast<int>(input.shape[1]))
                                continue;

                            const uint32_t in_idx =
                                index_nhwc(input, static_cast<uint32_t>(ih), static_cast<uint32_t>(iw), ic);

                            const uint32_t w_idx =
                                (((oc * kernel_size + kh) * kernel_size + kw) * in_channels) + ic;

                            sum += in[in_idx] * weights[w_idx];
                        }
                    }
                }

                uint32_t out_idx = (oh * outW + ow) * out_channels + oc;
                out[out_idx] = sum;
            }
        }
    }

    return false;
}
