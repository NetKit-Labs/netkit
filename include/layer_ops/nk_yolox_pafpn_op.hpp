#pragma once

#include "ops_resolver.hpp"

bool NkPlanYoloxPafpnMultiscale(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareYoloxPafpnMultiscale(const NkCnnOpContext& ctx);
void NkEvalYoloxPafpnMultiscale(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkYoloxPafpnMultiscaleOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::YoloxPafpnMultiscale),
        "yolox_pafpn_multiscale",
        NkPlanYoloxPafpnMultiscale,
        NkPrepareYoloxPafpnMultiscale,
        NkEvalYoloxPafpnMultiscale,
    };
};
