#pragma once

#include "ops_resolver.hpp"

bool NkPlanYoloxDecoupledHead(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareYoloxDecoupledHead(const NkCnnOpContext& ctx);
void NkEvalYoloxDecoupledHead(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkYoloxDecoupledHeadOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::YoloxDecoupledHead),
        "yolox_decoupled_head",
        NkPlanYoloxDecoupledHead,
        NkPrepareYoloxDecoupledHead,
        NkEvalYoloxDecoupledHead,
    };
};
