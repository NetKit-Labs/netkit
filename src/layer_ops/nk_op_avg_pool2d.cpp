#include "layer_ops/nk_avg_pool2d_op.hpp"

#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanAvgPool2D(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const AvgPool2DLayer& pool = block.avg_pool;
    const uint32_t out_h =
        CalcOutputDimAsymmetric(plan.h, pool.pool_h, pool.stride, pool.pad_h, pool.pad_h_end);
    const uint32_t out_w =
        CalcOutputDimAsymmetric(plan.w, pool.pool_w, pool.stride, pool.pad_w, pool.pad_w_end);
    BumpMaxActivation(plan, out_h * out_w * plan.channels);
    plan.h = out_h;
    plan.w = out_w;
    return true;
}

bool NkPrepareAvgPool2D(const NkCnnOpContext& ctx)
{
    const AvgPool2DLayer& pool = ctx.block.avg_pool;
    const uint32_t out_h = CalcOutputDimAsymmetric(
        ctx.input.shape[0], pool.pool_h, pool.stride, pool.pad_h, pool.pad_h_end);
    const uint32_t out_w = CalcOutputDimAsymmetric(
        ctx.input.shape[1], pool.pool_w, pool.stride, pool.pad_w, pool.pad_w_end);
    const std::array<uint32_t, 3> shape = {out_h, out_w, ctx.input.shape[2]};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalAvgPool2D(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.avg_pool.forward(input, output);
}
