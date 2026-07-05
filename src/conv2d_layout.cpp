#include "conv2d_layout.hpp"

#include "arena.hpp"
#include "conv2d.hpp"
#include "kernel_activation.hpp"
#include "kernel_workspace.hpp"
#include "netkit_loop_unroll.hpp"
#include "tensor_access.hpp"

namespace
{
    constexpr uint32_t kIm2ColMinPatchVolume = 2048u;
    constexpr uint32_t kInputStationaryMinOutChannels = 16u;

    float ConvOutputValue(float sum, NetkitKernelActivation fuse_activation)
    {
        return ApplyKernelActivation(sum, fuse_activation);
    }

    void im2col_nhwc(const float* in,
                     float* col,
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
                     int pad_h_end,
                     int pad_w_end)
    {
        const uint32_t patch_elems = kernel_h * kernel_w * in_ch;

        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                const uint32_t spatial_idx = oh * out_w + ow;
                uint32_t k_idx = 0;

                for (uint32_t kh = 0; kh < kernel_h; ++kh)
                {
                    const int ih = static_cast<int>(oh) * stride + static_cast<int>(kh) - pad_h;

                    for (uint32_t kw = 0; kw < kernel_w; ++kw)
                    {
                        const int iw = static_cast<int>(ow) * stride + static_cast<int>(kw) - pad_w;
                        const bool in_bounds = ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 &&
                                               iw < static_cast<int>(in_w);

                        for (uint32_t ic = 0; ic < in_ch; ++ic)
                        {
                            float value = 0.0f;
                            if (in_bounds)
                            {
                                value = in[(static_cast<uint32_t>(ih) * in_w + static_cast<uint32_t>(iw)) *
                                             in_ch +
                                         ic];
                            }
                            col[spatial_idx * patch_elems + k_idx++] = value;
                        }
                    }
                }
            }
        }
    }

    bool Conv2dForwardIm2Col(const float* in,
                             const float* weights_oki,
                             const float* bias,
                             float* out,
                             float* workspace,
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
                             NetkitKernelActivation fuse_activation)
    {
        const uint32_t patch_elems = kernel_h * kernel_w * in_ch;
        const uint32_t out_spatial = out_h * out_w;

        im2col_nhwc(in,
                    workspace,
                    in_h,
                    in_w,
                    in_ch,
                    out_h,
                    out_w,
                    kernel_h,
                    kernel_w,
                    stride,
                    pad_h,
                    pad_w,
                    pad_h_end,
                    pad_w_end);

        const bool fuse_in_kernel = kernel_activation_is_fused(fuse_activation);

        for (int oc = 0; oc < out_channels; ++oc)
        {
            const float* wt_row = weights_oki + static_cast<uint32_t>(oc) * patch_elems;
            const float b = bias ? bias[oc] : 0.0f;

            for (uint32_t s = 0; s < out_spatial; ++s)
            {
                const float sum =
                    b + NetkitLoopUnroll::dot_contiguous(wt_row, workspace + s * patch_elems, patch_elems);
                out[s * out_ch + static_cast<uint32_t>(oc)] = ConvOutputValue(sum, fuse_activation);
            }
        }

        return fuse_in_kernel;
    }

    bool Conv2dForwardInputStationary(const float* in,
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
                                      NetkitKernelActivation fuse_activation)
    {
        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                const uint32_t out_spatial_base = (oh * out_w + ow) * out_ch;

                for (int oc = 0; oc < out_channels; ++oc)
                {
                    out[out_spatial_base + static_cast<uint32_t>(oc)] = bias ? bias[oc] : 0.0f;
                }

                for (uint32_t kh = 0; kh < kernel_h; ++kh)
                {
                    const int ih = static_cast<int>(oh) * stride + static_cast<int>(kh) - pad_h;

                    for (uint32_t kw = 0; kw < kernel_w; ++kw)
                    {
                        const int iw = static_cast<int>(ow) * stride + static_cast<int>(kw) - pad_w;
                        if (ih < 0 || ih >= static_cast<int>(in_h) || iw < 0 ||
                            iw >= static_cast<int>(in_w))
                        {
                            continue;
                        }

                        const uint32_t in_base =
                            (static_cast<uint32_t>(ih) * in_w + static_cast<uint32_t>(iw)) * in_ch;

                        for (uint32_t ic = 0; ic < in_ch; ++ic)
                        {
                            const float value = in[in_base + ic];
                            if (value == 0.0f)
                                continue;

                            const float* w_slice = weights_hwio +
                                                   ((kh * kernel_w + kw) * in_ch + ic) * out_ch;

                            for (uint32_t oc = 0; oc < out_ch; ++oc)
                            {
                                out[out_spatial_base + oc] += value * w_slice[oc];
                            }
                        }
                    }
                }

                for (int oc = 0; oc < out_channels; ++oc)
                {
                    out[out_spatial_base + static_cast<uint32_t>(oc)] =
                        ConvOutputValue(out[out_spatial_base + static_cast<uint32_t>(oc)], fuse_activation);
                }
            }
        }

        return kernel_activation_is_fused(fuse_activation);
    }
}

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

    for (uint32_t oc = 0; oc < out_channels; ++oc)
    {
        for (uint32_t kh = 0; kh < kernel_h; ++kh)
        {
            for (uint32_t kw = 0; kw < kernel_w; ++kw)
            {
                for (uint32_t ic = 0; ic < in_channels; ++ic)
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

std::size_t Conv2dIm2ColWorkspaceBytes(uint32_t out_h,
                                       uint32_t out_w,
                                       uint32_t kernel_h,
                                       uint32_t kernel_w,
                                       uint32_t in_channels)
{
    const uint32_t patch = kernel_h * kernel_w * in_channels;
    const uint32_t spatial = out_h * out_w;
    if (patch == 0 || spatial == 0 || patch * spatial < kIm2ColMinPatchVolume)
        return 0;

    return static_cast<std::size_t>(patch) * static_cast<std::size_t>(spatial) * sizeof(float);
}

bool Conv2dShouldUseIm2Col(uint32_t out_h,
                           uint32_t out_w,
                           uint32_t kernel_h,
                           uint32_t kernel_w,
                           uint32_t in_channels)
{
    const uint32_t patch = kernel_h * kernel_w * in_channels;
    const uint32_t spatial = out_h * out_w;
    return patch > 0 && spatial > 0 && patch * spatial >= kIm2ColMinPatchVolume;
}

bool Conv2dShouldUseInputStationary(uint32_t out_channels, const float* weights_hwio)
{
    return weights_hwio != nullptr && out_channels >= static_cast<int>(kInputStationaryMinOutChannels);
}

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
                            NetkitKernelActivation fuse_activation)
{
    if (!Conv2dShouldUseIm2Col(out_h, out_w, kernel_h, kernel_w, in_ch))
        return false;

    KernelWorkspace* workspace = GetActiveKernelWorkspace();
    if (!workspace || !workspace->data)
        return false;

    const std::size_t required = Conv2dIm2ColWorkspaceBytes(
        out_h, out_w, kernel_h, kernel_w, in_ch);
    if (required == 0 || workspace->size_bytes < required)
        return false;

    Conv2dForwardIm2Col(in,
                        weights_oki,
                        bias,
                        out,
                        reinterpret_cast<float*>(workspace->data),
                        in_h,
                        in_w,
                        in_ch,
                        out_h,
                        out_w,
                        out_ch,
                        kernel_h,
                        kernel_w,
                        stride,
                        pad_h,
                        pad_w,
                        pad_h_end,
                        pad_w_end,
                        out_channels,
                        fuse_activation);
    return true;
}

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
                                     NetkitKernelActivation fuse_activation)
{
    if (!Conv2dShouldUseInputStationary(static_cast<uint32_t>(out_channels), weights_hwio))
        return false;

    Conv2dForwardInputStationary(in,
                                 weights_hwio,
                                 bias,
                                 out,
                                 in_h,
                                 in_w,
                                 in_ch,
                                 out_h,
                                 out_w,
                                 out_ch,
                                 kernel_h,
                                 kernel_w,
                                 stride,
                                 pad_h,
                                 pad_w,
                                 pad_h_end,
                                 pad_w_end,
                                 out_channels,
                                 fuse_activation);
    return true;
}
