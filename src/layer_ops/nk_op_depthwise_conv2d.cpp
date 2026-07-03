#include "layer_ops/nk_depthwise_conv2d_op.hpp"

#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanDepthwiseConv2D(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const uint32_t out_h = CalcOutputDim(plan.h,
                                         block.depthwise_conv.depthwise.kernel_size,
                                         block.depthwise_conv.depthwise.stride,
                                         block.depthwise_conv.depthwise.pad_h);
    const uint32_t out_w = CalcOutputDim(plan.w,
                                         block.depthwise_conv.depthwise.kernel_size,
                                         block.depthwise_conv.depthwise.stride,
                                         block.depthwise_conv.depthwise.pad_w);
    const uint32_t out_c = static_cast<uint32_t>(block.depthwise_conv.depthwise.channels);
    BumpMaxActivation(plan, out_h * out_w * out_c);
    plan.h = out_h;
    plan.w = out_w;
    plan.channels = out_c;
    return true;
}

bool NkPrepareDepthwiseConv2D(const NkCnnOpContext& ctx)
{
    const uint32_t out_h = CalcOutputDim(ctx.input.shape[0],
                                         ctx.block.depthwise_conv.depthwise.kernel_size,
                                         ctx.block.depthwise_conv.depthwise.stride,
                                         ctx.block.depthwise_conv.depthwise.pad_h);
    const uint32_t out_w = CalcOutputDim(ctx.input.shape[1],
                                         ctx.block.depthwise_conv.depthwise.kernel_size,
                                         ctx.block.depthwise_conv.depthwise.stride,
                                         ctx.block.depthwise_conv.depthwise.pad_w);
    const uint32_t out_c = static_cast<uint32_t>(ctx.block.depthwise_conv.depthwise.channels);
    const std::array<uint32_t, 3> shape = {out_h, out_w, out_c};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalDepthwiseConv2D(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.depthwise_conv.forward(input, output);
}
