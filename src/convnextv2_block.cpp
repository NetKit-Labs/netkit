#include "convnextv2_block.hpp"

#include "active_kernel.hpp"
#include "fused_kernel_ops.hpp"
#include "tensor_access.hpp"

#include <cstring>

void ConvNeXtV2Block::forward(const Tensor& input, Tensor& output)
{
    const uint32_t height = input.shape[0];
    const uint32_t width = input.shape[1];
    const uint32_t channel_count = static_cast<uint32_t>(channels);
    const uint32_t expanded = channel_count * static_cast<uint32_t>(kMlpRatio);
    const uint32_t spatial = height * width;
    const uint32_t required_scratch = spatial * expanded + expanded;

    if (!scratch || scratch_elems < required_scratch)
        return;

    float* branch = tensor_data_f32(output);
    float* expanded_buf = scratch;
    float* grn_norms = scratch + spatial * expanded;

    Kernels::DepthwiseConv2dForward(input,
                                    dw_weights,
                                    dw_bias,
                                    kDwKernel,
                                    kDwKernel,
                                    1,
                                    kDwPad,
                                    kDwPad,
                                    kDwPad,
                                    kDwPad,
                                    channels,
                                    NetkitKernelActivation::None,
                                    output);

    Tensor branch_view = fused_ops::NhwcView(branch, height, width, channel_count);
    Kernels::LayerNorm2dForward(
        branch_view, ln_weight, ln_bias, channels, eps, branch_view);

    for (uint32_t i = 0; i < spatial; ++i)
    {
        fused_ops::FullyConnected1x1(branch + i * channel_count,
                                     channels,
                                     static_cast<int>(expanded),
                                     pw1_weight,
                                     pw1_bias,
                                     expanded_buf + i * expanded);
    }

    Tensor expanded_view = fused_ops::NhwcView(expanded_buf, height, width, expanded);
    fused_ops::GeluInPlace(expanded_view);
    fused_ops::Grn2dInPlace(expanded_view,
                            static_cast<int>(expanded),
                            grn_gamma,
                            grn_beta,
                            eps,
                            grn_norms);

    float* out = tensor_data_f32(output);
    for (uint32_t i = 0; i < spatial; ++i)
    {
        fused_ops::FullyConnected1x1(expanded_buf + i * expanded,
                                     static_cast<int>(expanded),
                                     channels,
                                     pw2_weight,
                                     pw2_bias,
                                     out + i * channel_count);
    }

    fused_ops::MatAddInPlace(output, input);
}
