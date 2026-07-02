#include "layer_op_registry.hpp"

#include "cnn.hpp"
#include "tensor_factory.hpp"
#include <array>
#include <cstring>

using namespace TensorFactory;

namespace
{
    uint32_t CalcOutputDim(uint32_t input_dim, int kernel_size, int stride, int pad = 0)
    {
        return static_cast<uint32_t>((static_cast<int>(input_dim) + 2 * pad - kernel_size) / stride + 1);
    }

    void FlattenNhwc(const Tensor& input, Tensor& output)
    {
        const float* in = tensor_data_f32(const_cast<Tensor&>(input));
        float* out = tensor_data_f32(output);
        std::memcpy(out, in, static_cast<std::size_t>(input.num_elements) * sizeof(float));
    }

    void BumpMaxActivation(NkCnnSpatialPlan& plan, uint32_t elements)
    {
        if (!plan.max_activation_elements)
            return;

        if (elements > *plan.max_activation_elements)
            *plan.max_activation_elements = elements;
    }
}

bool NkPlanConv2D(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const uint32_t out_h =
        CalcOutputDim(plan.h, block.conv.conv.kernel_size, block.conv.conv.stride, block.conv.conv.pad_h);
    const uint32_t out_w =
        CalcOutputDim(plan.w, block.conv.conv.kernel_size, block.conv.conv.stride, block.conv.conv.pad_w);
    const uint32_t out_c = static_cast<uint32_t>(block.conv.conv.out_channels);
    BumpMaxActivation(plan, out_h * out_w * out_c);
    plan.h = out_h;
    plan.w = out_w;
    plan.channels = out_c;
    return true;
}

bool NkPlanMaxPool2D(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const uint32_t out_h = CalcOutputDim(plan.h, block.pool.pool_size, block.pool.stride, block.pool.pad_h);
    const uint32_t out_w = CalcOutputDim(plan.w, block.pool.pool_size, block.pool.stride, block.pool.pad_w);
    BumpMaxActivation(plan, out_h * out_w * plan.channels);
    plan.h = out_h;
    plan.w = out_w;
    return true;
}

bool NkPlanAvgPool2D(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const uint32_t out_h =
        CalcOutputDim(plan.h, block.avg_pool.pool_size, block.avg_pool.stride, block.avg_pool.pad_h);
    const uint32_t out_w =
        CalcOutputDim(plan.w, block.avg_pool.pool_size, block.avg_pool.stride, block.avg_pool.pad_w);
    BumpMaxActivation(plan, out_h * out_w * plan.channels);
    plan.h = out_h;
    plan.w = out_w;
    return true;
}

bool NkPlanBatchNorm2d(CnnBlock& /*block*/, NkCnnSpatialPlan& plan)
{
    BumpMaxActivation(plan, plan.h * plan.w * plan.channels);
    return true;
}

bool NkPlanFlatten(CnnBlock& /*block*/, NkCnnSpatialPlan& plan)
{
    const uint32_t features = plan.h * plan.w * plan.channels;
    BumpMaxActivation(plan, features);
    plan.h = 1;
    plan.w = features;
    plan.channels = 1;
    return true;
}

bool NkPlanDense(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const uint32_t out_features = block.dense.weights.shape[0];
    BumpMaxActivation(plan, out_features);
    plan.w = out_features;
    return true;
}

bool NkPrepareConv2D(const NkCnnOpContext& ctx)
{
    const uint32_t out_h = CalcOutputDim(ctx.input.shape[0],
                                         ctx.block.conv.conv.kernel_size,
                                         ctx.block.conv.conv.stride,
                                         ctx.block.conv.conv.pad_h);
    const uint32_t out_w = CalcOutputDim(ctx.input.shape[1],
                                         ctx.block.conv.conv.kernel_size,
                                         ctx.block.conv.conv.stride,
                                         ctx.block.conv.conv.pad_w);
    const uint32_t out_c = static_cast<uint32_t>(ctx.block.conv.conv.out_channels);
    const std::array<uint32_t, 3> shape = {out_h, out_w, out_c};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

bool NkPrepareMaxPool2D(const NkCnnOpContext& ctx)
{
    const uint32_t out_h =
        CalcOutputDim(ctx.input.shape[0], ctx.block.pool.pool_size, ctx.block.pool.stride, ctx.block.pool.pad_h);
    const uint32_t out_w =
        CalcOutputDim(ctx.input.shape[1], ctx.block.pool.pool_size, ctx.block.pool.stride, ctx.block.pool.pad_w);
    const std::array<uint32_t, 3> shape = {out_h, out_w, ctx.input.shape[2]};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

bool NkPrepareAvgPool2D(const NkCnnOpContext& ctx)
{
    const uint32_t out_h = CalcOutputDim(ctx.input.shape[0],
                                         ctx.block.avg_pool.pool_size,
                                         ctx.block.avg_pool.stride,
                                         ctx.block.avg_pool.pad_h);
    const uint32_t out_w = CalcOutputDim(ctx.input.shape[1],
                                         ctx.block.avg_pool.pool_size,
                                         ctx.block.avg_pool.stride,
                                         ctx.block.avg_pool.pad_w);
    const std::array<uint32_t, 3> shape = {out_h, out_w, ctx.input.shape[2]};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

bool NkPrepareBatchNorm2d(const NkCnnOpContext& ctx)
{
    const std::array<uint32_t, 3> shape = {ctx.input.shape[0], ctx.input.shape[1], ctx.input.shape[2]};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

bool NkPrepareFlatten(const NkCnnOpContext& ctx)
{
    const uint32_t features = ctx.input.num_elements;
    ctx.output = View2D(ctx.write_buffer, 1, features);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

bool NkPrepareDense(const NkCnnOpContext& ctx)
{
    const uint32_t out_features = ctx.block.dense.weights.shape[0];
    ctx.output = View2D(ctx.write_buffer, 1, out_features);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalConv2D(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.conv.forward(input, output);
}

void NkEvalMaxPool2D(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.pool.forward(input, output);
}

void NkEvalAvgPool2D(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.avg_pool.forward(input, output);
}

void NkEvalBatchNorm2d(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.batch_norm.forward(input, output);
}

void NkEvalFlatten(CnnBlock& /*block*/, const Tensor& input, Tensor& output)
{
    FlattenNhwc(input, output);
}

void NkEvalDense(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.dense.forward(input, output);
}

NkOpCode ToOpCode(CnnBlockType block_type)
{
    switch (block_type)
    {
        case CnnBlockType::Conv2D:
            return NkOpCode::Conv2D;
        case CnnBlockType::MaxPool2D:
            return NkOpCode::MaxPool2D;
        case CnnBlockType::AvgPool2D:
            return NkOpCode::AvgPool2D;
        case CnnBlockType::BatchNorm2d:
            return NkOpCode::BatchNorm2d;
        case CnnBlockType::Flatten:
            return NkOpCode::Flatten;
        case CnnBlockType::Dense:
            return NkOpCode::Dense;
    }
    return NkOpCode::Dense;
}

const NkOpsResolver& GetDefaultOpsResolver()
{
    return NkAllLayerOps::View();
}
