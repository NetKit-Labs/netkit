#include "layer_ops/nk_feature_tap_op.hpp"

#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_access.hpp"
#include "tensor_factory.hpp"

#include <array>
#include <cstring>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanFeatureTap(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const FeatureTapLayer& tap = block.feature_tap;
    BumpMaxActivation(plan, plan.h * plan.w * static_cast<uint32_t>(tap.channels));
    (void)tap;
    return true;
}

bool NkPrepareFeatureTap(const NkCnnOpContext& ctx)
{
    const FeatureTapLayer& tap = ctx.block.feature_tap;
    const std::array<uint32_t, 3> shape = {ctx.input.shape[0], ctx.input.shape[1],
                                           static_cast<uint32_t>(tap.channels)};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalFeatureTap(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.feature_tap.forward(input, output);
}
