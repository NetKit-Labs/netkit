#include "layer_ops/nk_yolox_pafpn_op.hpp"

#include "cmsis_buffer_size.hpp"
#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include "yolox_pafpn.hpp"

#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanYoloxPafpnMultiscale(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    YoloxPafpnMultiscale& neck = block.yolox_pafpn.block;
    const uint32_t h5 = plan.h;
    const uint32_t w5 = plan.w;
    const uint32_t h4 = h5 * 2u;
    const uint32_t w4 = w5 * 2u;
    const uint32_t h3 = h5 * 4u;
    const uint32_t w3 = w5 * 4u;
    const uint32_t H = static_cast<uint32_t>(neck.hidden_dim);
    const uint32_t out_c = static_cast<uint32_t>(neck.output_channels());
    const uint32_t out_elems = neck.output_elements(h5, w5);

    // Lateral 1x1
    CmsisBumpConv2dWorkspace(h3, w3, 1, 1, 0, 0, neck.c3_channels, neck.hidden_dim);
    CmsisBumpConv2dWorkspace(h4, w4, 1, 1, 0, 0, neck.c4_channels, neck.hidden_dim);
    CmsisBumpConv2dWorkspace(h5, w5, 1, 1, 0, 0, neck.c5_channels, neck.hidden_dim);

    // Top-down DW+PW
    CmsisBumpDepthwiseConv2dWorkspace(h4, w4, 3, 3, 1, 1, 1, static_cast<int>(H));
    CmsisBumpConv2dWorkspace(h4, w4, 1, 1, 0, 0, neck.hidden_dim, neck.hidden_dim);
    CmsisBumpDepthwiseConv2dWorkspace(h3, w3, 3, 3, 1, 1, 1, static_cast<int>(H));
    CmsisBumpConv2dWorkspace(h3, w3, 1, 1, 0, 0, neck.hidden_dim, neck.hidden_dim);

    // Bottom-up DW stride2 + PW
    CmsisBumpDepthwiseConv2dWorkspace(h3, w3, 3, 3, 2, 1, 1, static_cast<int>(H));
    CmsisBumpConv2dWorkspace(h4, w4, 1, 1, 0, 0, neck.hidden_dim, neck.hidden_dim);
    CmsisBumpDepthwiseConv2dWorkspace(h4, w4, 3, 3, 2, 1, 1, static_cast<int>(H));
    CmsisBumpConv2dWorkspace(h5, w5, 1, 1, 0, 0, neck.hidden_dim, neck.hidden_dim);

    // Heads (largest grid first)
    for (int s = 0; s < YoloxPafpnMultiscale::kNumScales; ++s)
    {
        const YoloxDecoupledHead& head = neck.heads[s];
        const uint32_t hh = (s == 0) ? h3 : (s == 1) ? h4 : h5;
        const uint32_t ww = (s == 0) ? w3 : (s == 1) ? w4 : w5;
        CmsisBumpConv2dWorkspace(hh, ww, 1, 1, 0, 0, head.in_channels, head.hidden_dim);
        for (int i = 0; i < head.num_convs; ++i)
            CmsisBumpConv2dWorkspace(hh, ww, 3, 1, 1, 1, head.hidden_dim, head.hidden_dim);
        CmsisBumpConv2dWorkspace(hh, ww, 1, 1, 0, 0, head.hidden_dim, head.num_classes);
        for (int i = 0; i < head.num_convs; ++i)
            CmsisBumpConv2dWorkspace(hh, ww, 3, 1, 1, 1, head.hidden_dim, head.hidden_dim);
        CmsisBumpConv2dWorkspace(hh, ww, 1, 1, 0, 0, head.hidden_dim, 4);
        CmsisBumpConv2dWorkspace(hh, ww, 1, 1, 0, 0, head.hidden_dim, 1);
    }

    BumpMaxActivation(plan, h3 * w3 * static_cast<uint32_t>(neck.c3_channels));
    BumpMaxActivation(plan, h3 * w3 * H);
    BumpMaxActivation(plan, out_elems);

    plan.h = 1;
    plan.w = 1;
    plan.channels = out_elems;
    return true;
}

bool NkPrepareYoloxPafpnMultiscale(const NkCnnOpContext& ctx)
{
    YoloxPafpnMultiscale& neck = ctx.block.yolox_pafpn.block;
    const uint32_t out_elems = neck.output_elements(ctx.input.shape[0], ctx.input.shape[1]);
    const std::array<uint32_t, 1> shape = {out_elems};
    ctx.output = ViewND(ctx.write_buffer, 1, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalYoloxPafpnMultiscale(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.yolox_pafpn.forward(input, output);
}
