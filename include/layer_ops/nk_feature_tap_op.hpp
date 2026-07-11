#pragma once

#include "ops_resolver.hpp"

bool NkPlanFeatureTap(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareFeatureTap(const NkCnnOpContext& ctx);
void NkEvalFeatureTap(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkFeatureTapOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::FeatureTap),
        "feature_tap",
        NkPlanFeatureTap,
        NkPrepareFeatureTap,
        NkEvalFeatureTap,
    };
};
