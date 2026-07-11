#include "yolox_pafpn.hpp"

#include "active_kernel.hpp"
#include "fused_kernel_ops.hpp"
#include "tensor_access.hpp"

#include <cmath>
#include <cstring>

namespace
{
    void ApplySiluInPlace(Tensor& tensor)
    {
        float* data = tensor_data_f32(tensor);
        for (size_t i = 0; i < tensor.num_elements; ++i)
        {
            const float x = data[i];
            const float sigmoid = 1.0f / (1.0f + std::exp(-x));
            data[i] = x * sigmoid;
        }
    }

    void Conv2dNhwc(const Tensor& input,
                    float* weights,
                    float* bias,
                    int kernel_size,
                    int stride,
                    int pad,
                    int in_channels,
                    int out_channels,
                    Tensor& output)
    {
        Kernels::Conv2dForward(input,
                               weights,
                               bias,
                               kernel_size,
                               stride,
                               pad,
                               pad,
                               in_channels,
                               out_channels,
                               NetkitKernelActivation::None,
                               output);
    }

    void DepthwiseConv2dNhwc(const Tensor& input,
                             float* weights,
                             float* bias,
                             int kernel,
                             int stride,
                             int pad,
                             int channels,
                             Tensor& output)
    {
        Kernels::DepthwiseConv2dForward(input,
                                        weights,
                                        bias,
                                        kernel,
                                        kernel,
                                        stride,
                                        pad,
                                        pad,
                                        pad,
                                        pad,
                                        channels,
                                        NetkitKernelActivation::None,
                                        output);
    }

    void UpsampleNearest2x(const Tensor& input, Tensor& output)
    {
        const uint32_t h = input.shape[0];
        const uint32_t w = input.shape[1];
        const uint32_t c = input.shape[2];
        const float* in = tensor_data_f32(input);
        float* out = tensor_data_f32(output);
        for (uint32_t y = 0; y < h; ++y)
        {
            for (uint32_t x = 0; x < w; ++x)
            {
                const std::size_t in_base =
                    (static_cast<std::size_t>(y) * w + x) * static_cast<std::size_t>(c);
                for (int dy = 0; dy < 2; ++dy)
                {
                    for (int dx = 0; dx < 2; ++dx)
                    {
                        const uint32_t oy = y * 2u + static_cast<uint32_t>(dy);
                        const uint32_t ox = x * 2u + static_cast<uint32_t>(dx);
                        const std::size_t out_base =
                            (static_cast<std::size_t>(oy) * (w * 2u) + ox) *
                            static_cast<std::size_t>(c);
                        std::memcpy(out + out_base, in + in_base,
                                    static_cast<std::size_t>(c) * sizeof(float));
                    }
                }
            }
        }
    }

    void AddInPlace(Tensor& dst, const Tensor& src)
    {
        float* d = tensor_data_f32(dst);
        const float* s = tensor_data_f32(src);
        for (size_t i = 0; i < dst.num_elements; ++i)
            d[i] += s[i];
    }

    void AddInPlaceBuf(float* dst, const float* src, uint32_t elems)
    {
        for (uint32_t i = 0; i < elems; ++i)
            dst[i] += src[i];
    }

    void DwPwSilu(const Tensor& input,
                  float* dw_w,
                  float* dw_b,
                  float* pw_w,
                  float* pw_b,
                  int channels,
                  int stride,
                  float* dw_buf,
                  float* out_buf,
                  uint32_t out_h,
                  uint32_t out_w)
    {
        Tensor dw_out = fused_ops::NhwcView(dw_buf, out_h, out_w, static_cast<uint32_t>(channels));
        DepthwiseConv2dNhwc(input, dw_w, dw_b, 3, stride, 1, channels, dw_out);
        Tensor pw_out = fused_ops::NhwcView(out_buf, out_h, out_w, static_cast<uint32_t>(channels));
        Conv2dNhwc(dw_out, pw_w, pw_b, 1, 1, 0, channels, channels, pw_out);
        ApplySiluInPlace(pw_out);
    }
}

void YoloxPafpnMultiscale::forward(const Tensor& c5_input, Tensor& output)
{
    if (!scratch || scratch_elems == 0 || !output.data || !c3_data || !c4_data)
        return;

    const uint32_t h5 = c5_input.shape[0];
    const uint32_t w5 = c5_input.shape[1];
    const uint32_t h4 = h5 * 2u;
    const uint32_t w4 = w5 * 2u;
    const uint32_t h3 = h5 * 4u;
    const uint32_t w3 = w5 * 4u;
    const uint32_t H = static_cast<uint32_t>(hidden_dim);
    const int out_c = output_channels();

    if (c3_h != h3 || c3_w != w3 || c4_h != h4 || c4_w != w4)
        return;

    const uint32_t e3 = h3 * w3 * H;
    const uint32_t e4 = h4 * w4 * H;
    const uint32_t e5 = h5 * w5 * H;
    // Layout: p3 | p4 | p5 | work_a | work_b | head_scratch(3*e3)
    const uint32_t required = e3 + e4 + e5 + e3 + e3 + 3u * e3;
    if (scratch_elems < required)
        return;

    float* p3 = scratch;
    float* p4 = p3 + e3;
    float* p5 = p4 + e4;
    float* work_a = p5 + e5;
    float* work_b = work_a + e3;
    float* head_scratch = work_b + e3;

    Tensor c3 = fused_ops::NhwcView(c3_data, h3, w3, static_cast<uint32_t>(c3_channels));
    Tensor c4 = fused_ops::NhwcView(c4_data, h4, w4, static_cast<uint32_t>(c4_channels));

    // Laterals + top-down
    Tensor l5 = fused_ops::NhwcView(p5, h5, w5, H);
    Conv2dNhwc(c5_input, lat5_weights, lat5_bias, 1, 1, 0, c5_channels, hidden_dim, l5);

    Tensor l4 = fused_ops::NhwcView(work_a, h4, w4, H);
    Conv2dNhwc(c4, lat4_weights, lat4_bias, 1, 1, 0, c4_channels, hidden_dim, l4);

    Tensor up5 = fused_ops::NhwcView(work_b, h4, w4, H);
    UpsampleNearest2x(l5, up5);
    AddInPlace(l4, up5);
    DwPwSilu(l4, td_p4_dw_weights, td_p4_dw_bias, td_p4_pw_weights, td_p4_pw_bias, hidden_dim, 1,
             work_b, p4, h4, w4);

    Tensor l3 = fused_ops::NhwcView(work_a, h3, w3, H);
    Conv2dNhwc(c3, lat3_weights, lat3_bias, 1, 1, 0, c3_channels, hidden_dim, l3);

    Tensor p4_t = fused_ops::NhwcView(p4, h4, w4, H);
    Tensor up4 = fused_ops::NhwcView(work_b, h3, w3, H);
    UpsampleNearest2x(p4_t, up4);
    AddInPlace(l3, up4);
    DwPwSilu(l3, td_p3_dw_weights, td_p3_dw_bias, td_p3_pw_weights, td_p3_pw_bias, hidden_dim, 1,
             work_b, p3, h3, w3);

    // Bottom-up: N3=P3; N4 = SiLU(DW_s2+PW(N3))+P4; N5 = SiLU(DW_s2+PW(N4))+P5
    Tensor n3 = fused_ops::NhwcView(p3, h3, w3, H);
    DwPwSilu(n3, bu_n4_dw_weights, bu_n4_dw_bias, bu_n4_pw_weights, bu_n4_pw_bias, hidden_dim, 2,
             work_a, work_b, h4, w4);
    AddInPlaceBuf(work_b, p4, e4);
    std::memcpy(p4, work_b, static_cast<std::size_t>(e4) * sizeof(float));

    Tensor n4 = fused_ops::NhwcView(p4, h4, w4, H);
    Tensor p5_t = fused_ops::NhwcView(p5, h5, w5, H);
    DwPwSilu(n4, bu_n5_dw_weights, bu_n5_dw_bias, bu_n5_pw_weights, bu_n5_pw_bias, hidden_dim, 2,
             work_a, work_b, h5, w5);
    AddInPlaceBuf(work_b, tensor_data_f32(p5_t), e5);
    std::memcpy(p5, work_b, static_cast<std::size_t>(e5) * sizeof(float));

    float* out_data = tensor_data_f32(output);
    uint32_t out_offset = 0;
    const uint32_t scales_h[3] = {h3, h4, h5};
    const uint32_t scales_w[3] = {w3, w4, w5};
    float* scale_bufs[3] = {p3, p4, p5};

    for (int s = 0; s < kNumScales; ++s)
    {
        YoloxDecoupledHead& head = heads[s];
        head.scratch = head_scratch;
        head.scratch_elems = 3u * e3;

        Tensor feat = fused_ops::NhwcView(scale_bufs[s], scales_h[s], scales_w[s], H);
        const uint32_t spat = scales_h[s] * scales_w[s];
        const uint32_t head_elems = spat * static_cast<uint32_t>(out_c);
        Tensor head_out =
            fused_ops::NhwcView(out_data + out_offset, scales_h[s], scales_w[s],
                                static_cast<uint32_t>(out_c));
        head.forward(feat, head_out);
        out_offset += head_elems;
    }
}
