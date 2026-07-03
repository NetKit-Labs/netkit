#pragma once

#include "ops_resolver.hpp"

bool NkPlanDepthwiseConv2D(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareDepthwiseConv2D(const NkCnnOpContext& ctx);
void NkEvalDepthwiseConv2D(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkDepthwiseConv2DOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::DepthwiseConv2D),
        "depthwise_conv2d",
        NkPlanDepthwiseConv2D,
        NkPrepareDepthwiseConv2D,
        NkEvalDepthwiseConv2D,
    };
};
