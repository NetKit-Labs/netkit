#include "conv2d.hpp"

void Conv2D::forward(const Tensor& input, Tensor& output)
{
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
                            uint32_t ih = oh * stride + kh;
                            uint32_t iw = ow * stride + kw;

                            uint32_t in_idx = index_nhwc(input, ih, iw, ic);

                            uint32_t w_idx = (((oc * kernel_size + kh) * kernel_size + kw) * in_channels) + ic;

                            sum += in[in_idx] * weights[w_idx];
                        }
                    }
                }

                uint32_t out_idx = (oh * outW + ow) * out_channels + oc;
                out[out_idx] = sum;
            }
        }
    }
}
