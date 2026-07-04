#include "yolox_decoupled_head.hpp"

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
        for (uint32_t i = 0; i < tensor.num_elements; ++i)
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

    void RunBranchConvs(const Tensor& stem,
                        int num_convs,
                        int hidden_dim,
                        float* const* conv_weights,
                        float* const* conv_bias,
                        float* work_a,
                        float* work_b)
    {
        const uint32_t h = stem.shape[0];
        const uint32_t w = stem.shape[1];
        Tensor cur = stem;
        float* next_data = work_a;
        float* alt_data = work_b;

        for (int i = 0; i < num_convs; ++i)
        {
            Tensor next = fused_ops::NhwcView(next_data, h, w, static_cast<uint32_t>(hidden_dim));
            Conv2dNhwc(cur,
                       conv_weights[i],
                       conv_bias[i],
                       3,
                       1,
                       1,
                       hidden_dim,
                       hidden_dim,
                       next);
            ApplySiluInPlace(next);
            cur = next;
            std::swap(next_data, alt_data);
        }

        if (cur.data != work_a)
            std::memcpy(work_a, cur.data, static_cast<std::size_t>(h * w * hidden_dim) * sizeof(float));
    }

    void ScatterChannels(float* output,
                         uint32_t h,
                         uint32_t w,
                         int out_channels,
                         int channel_offset,
                         int pred_channels,
                         const float* pred)
    {
        for (uint32_t y = 0; y < h; ++y)
        {
            for (uint32_t x = 0; x < w; ++x)
            {
                const std::size_t out_base =
                    (static_cast<std::size_t>(y) * w + x) * static_cast<std::size_t>(out_channels);
                const std::size_t pred_base =
                    (static_cast<std::size_t>(y) * w + x) * static_cast<std::size_t>(pred_channels);
                for (int c = 0; c < pred_channels; ++c)
                {
                    output[out_base + static_cast<std::size_t>(channel_offset + c)] =
                        pred[pred_base + static_cast<std::size_t>(c)];
                }
            }
        }
    }
}

void YoloxDecoupledHead::forward(const Tensor& input, Tensor& output)
{
    if (!scratch || scratch_elems == 0 || !output.data)
        return;

    const uint32_t h = input.shape[0];
    const uint32_t w = input.shape[1];
    const uint32_t spatial = h * w;
    const uint32_t hidden = static_cast<uint32_t>(hidden_dim);
    const int out_ch = output_channels();

    const std::size_t stem_elems = static_cast<std::size_t>(spatial) * hidden;
    const std::size_t required = stem_elems * 3u;
    if (scratch_elems < required)
        return;

    float* stem_buf = scratch;
    float* work_a = scratch + stem_elems;
    float* work_b = scratch + stem_elems * 2u;

    Tensor stem = fused_ops::NhwcView(stem_buf, h, w, hidden);
    Conv2dNhwc(input,
               stem_weights,
               stem_bias,
               1,
               1,
               0,
               in_channels,
               hidden_dim,
               stem);
    ApplySiluInPlace(stem);

    RunBranchConvs(stem, num_convs, hidden_dim, cls_conv_weights, cls_conv_bias, work_a, work_b);

    Tensor cls_feat = fused_ops::NhwcView(work_a, h, w, hidden);
    Tensor cls_pred = fused_ops::NhwcView(work_b, h, w, static_cast<uint32_t>(num_classes));
    Conv2dNhwc(cls_feat,
               cls_pred_weights,
               cls_pred_bias,
               1,
               1,
               0,
               hidden_dim,
               num_classes,
               cls_pred);

    float* out_data = tensor_data_f32(output);
    std::memset(out_data, 0, static_cast<std::size_t>(spatial) * static_cast<std::size_t>(out_ch) * sizeof(float));
    ScatterChannels(out_data, h, w, out_ch, 5, num_classes, tensor_data_f32(cls_pred));

    RunBranchConvs(stem, num_convs, hidden_dim, reg_conv_weights, reg_conv_bias, work_a, work_b);

    Tensor reg_feat = fused_ops::NhwcView(work_a, h, w, hidden);
    Tensor reg_pred = fused_ops::NhwcView(work_b, h, w, 4u);
    Conv2dNhwc(reg_feat,
               reg_pred_weights,
               reg_pred_bias,
               1,
               1,
               0,
               hidden_dim,
               4,
               reg_pred);
    ScatterChannels(out_data, h, w, out_ch, 0, 4, tensor_data_f32(reg_pred));

    Tensor obj_pred = fused_ops::NhwcView(work_b, h, w, 1u);
    Conv2dNhwc(reg_feat,
               obj_pred_weights,
               obj_pred_bias,
               1,
               1,
               0,
               hidden_dim,
               1,
               obj_pred);
    ScatterChannels(out_data, h, w, out_ch, 4, 1, tensor_data_f32(obj_pred));
}
