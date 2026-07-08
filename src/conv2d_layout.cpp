#include "conv2d_layout.hpp"

#include "arena.hpp"
#include "conv2d.hpp"

void RepackConv2dOkiToHwio(const float* oki,
                           float* hwio,
                           uint32_t kernel_h,
                           uint32_t kernel_w,
                           uint32_t in_channels,
                           uint32_t out_channels)
{
    if (!oki || !hwio || kernel_h == 0 || kernel_w == 0 || in_channels == 0 || out_channels == 0)
        return;

    const uint32_t filter_spatial = kernel_h * kernel_w;

    for (size_t oc = 0; oc < out_channels; ++oc)
    {
        for (size_t kh = 0; kh < kernel_h; ++kh)
        {
            for (size_t kw = 0; kw < kernel_w; ++kw)
            {
                for (size_t ic = 0; ic < in_channels; ++ic)
                {
                    const uint32_t oki_idx =
                        ((oc * filter_spatial + kh * kernel_w + kw) * in_channels) + ic;
                    const uint32_t hwio_idx =
                        ((kh * kernel_w + kw) * in_channels + ic) * out_channels + oc;
                    hwio[hwio_idx] = oki[oki_idx];
                }
            }
        }
    }
}

bool RepackConv2dWeights(Conv2D& conv, Arena& arena)
{
    if (conv.weights_hwio || !conv.weights || conv.kernel_size <= 0 || conv.in_channels <= 0 ||
        conv.out_channels <= 0)
    {
        return conv.weights != nullptr;
    }

    const uint32_t kh = static_cast<uint32_t>(conv.kernel_size);
    const uint32_t kw = kh;
    const uint32_t ic = static_cast<uint32_t>(conv.in_channels);
    const uint32_t oc = static_cast<uint32_t>(conv.out_channels);
    const std::size_t bytes = static_cast<std::size_t>(kh) * kw * ic * oc * sizeof(float);

    float* hwio = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    if (!hwio)
        return false;

    RepackConv2dOkiToHwio(conv.weights, hwio, kh, kw, ic, oc);
    conv.weights_hwio = hwio;
    return true;
}
