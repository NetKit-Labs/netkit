/*
 * Float32 XNNPACK full-network subgraph for netkit CNNs.
 *
 * Mirrors the int8 CreateNetworkSubgraph path: one persistent runtime, shared
 * weights cache + workspace, XNN_FLAG_DONT_SPIN_WORKERS, single invoke/forward.
 * Ephemeral per-op create/run/delete stays in xnnpack_backend.cpp as fallback.
 */
#include "xnnpack_float.hpp"

#include "netkit_config.h"

#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED

#include "arena.hpp"
#include "cnn.hpp"
#include "mlp.hpp"
#include "mobilenetv4_uib.hpp"
#include "nk_op_detail.hpp"
#include "yolox_decoupled_head.hpp"
#include "yolox_pafpn.hpp"

#include <xnnpack.h>

#include <cfloat>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>

namespace
{
std::once_flag g_xnn_init_flag;
bool g_xnn_ready = false;

bool EnsureXnnInitialized()
{
    std::call_once(g_xnn_init_flag, []() {
        g_xnn_ready = (xnn_initialize(/*allocator=*/nullptr) == xnn_status_success);
    });
    return g_xnn_ready;
}

const char* XnnStatusName(enum xnn_status st)
{
    switch (st)
    {
        case xnn_status_success:
            return "success";
        case xnn_status_uninitialized:
            return "uninitialized";
        case xnn_status_invalid_parameter:
            return "invalid_parameter";
        case xnn_status_invalid_state:
            return "invalid_state";
        case xnn_status_unsupported_parameter:
            return "unsupported_parameter";
        case xnn_status_unsupported_hardware:
            return "unsupported_hardware";
        case xnn_status_out_of_memory:
            return "out_of_memory";
        default:
            return "unknown";
    }
}

bool ActivationClamp(ConvActivationType activation, float& out_min, float& out_max)
{
    out_min = -FLT_MAX;
    out_max = FLT_MAX;
    switch (activation)
    {
        case ConvActivationType::None:
        case ConvActivationType::Softmax:
            return true;
        case ConvActivationType::ReLU:
            out_min = 0.0f;
            return true;
        case ConvActivationType::ReLU6:
            out_min = 0.0f;
            out_max = 6.0f;
            return true;
        default:
            return false;
    }
}

bool ActivationClampDense(ActivationType activation, float& out_min, float& out_max)
{
    out_min = -FLT_MAX;
    out_max = FLT_MAX;
    switch (activation)
    {
        case ActivationType::None:
        case ActivationType::Softmax:
            // Softmax omitted in network subgraph (logits), matching int8 path.
            return true;
        case ActivationType::ReLU:
            out_min = 0.0f;
            return true;
        case ActivationType::ReLU6:
            out_min = 0.0f;
            out_max = 6.0f;
            return true;
        default:
            return false;
    }
}

bool DefineFp32(xnn_subgraph_t subgraph,
                size_t num_dims,
                const size_t* dims,
                const void* data,
                uint32_t external_id,
                uint32_t flags,
                uint32_t* id_out)
{
    return xnn_define_tensor_value(subgraph,
                                   xnn_datatype_fp32,
                                   num_dims,
                                   dims,
                                   data,
                                   external_id,
                                   flags,
                                   id_out) == xnn_status_success;
}

bool DefineActNhwc(xnn_subgraph_t subgraph,
                   size_t h,
                   size_t w,
                   size_t c,
                   uint32_t external_id,
                   uint32_t flags,
                   uint32_t* id_out)
{
    const size_t dims[4] = {1, h, w, c};
    return DefineFp32(subgraph, 4, dims, nullptr, external_id, flags, id_out);
}

void RepackDwGhwToHwc(const float* ghw, float* hwc, int kh, int kw, int channels)
{
    for (int c = 0; c < channels; ++c)
    {
        for (int y = 0; y < kh; ++y)
        {
            for (int x = 0; x < kw; ++x)
                hwc[(y * kw + x) * channels + c] = ghw[(c * kh + y) * kw + x];
        }
    }
}

bool PushDwHwc(XnnpackFloat::Runtime& runtime, float* owned)
{
    if (!owned)
        return true;
    float** next = new (std::nothrow) float*[runtime.dw_hwc_count + 1];
    if (!next)
    {
        delete[] owned;
        return false;
    }
    for (uint32_t i = 0; i < runtime.dw_hwc_count; ++i)
        next[i] = runtime.dw_hwc_owned[i];
    next[runtime.dw_hwc_count] = owned;
    delete[] runtime.dw_hwc_owned;
    runtime.dw_hwc_owned = next;
    ++runtime.dw_hwc_count;
    return true;
}

float* OwnDwHwc(XnnpackFloat::Runtime& runtime,
                Arena& arena,
                const float* ghw,
                int kh,
                int kw,
                int channels)
{
    const size_t n = static_cast<size_t>(kh) * static_cast<size_t>(kw) *
                     static_cast<size_t>(channels);
    float* hwc = static_cast<float*>(arena.alloc(n * sizeof(float), alignof(float)));
    bool arena_owned = (hwc != nullptr);
    if (!hwc)
    {
        hwc = new (std::nothrow) float[n];
        if (!hwc)
            return nullptr;
    }
    RepackDwGhwToHwc(ghw, hwc, kh, kw, channels);
    if (!arena_owned && !PushDwHwc(runtime, hwc))
        return nullptr;
    return hwc;
}

bool DefineFilterBiasConv(xnn_subgraph_t subgraph,
                          const float* weights_ohwi,
                          const float* bias,
                          size_t out_c,
                          size_t kh,
                          size_t kw,
                          size_t in_c,
                          uint32_t* filter_id,
                          uint32_t* bias_id)
{
    const size_t filter_dims[4] = {out_c, kh, kw, in_c};
    if (!DefineFp32(subgraph,
                    4,
                    filter_dims,
                    weights_ohwi,
                    XNN_INVALID_VALUE_ID,
                    /*flags=*/0,
                    filter_id))
        return false;
    if (bias)
    {
        const size_t bias_dims[1] = {out_c};
        if (!DefineFp32(subgraph,
                        1,
                        bias_dims,
                        bias,
                        XNN_INVALID_VALUE_ID,
                        /*flags=*/0,
                        bias_id))
            return false;
    }
    else
    {
        *bias_id = XNN_INVALID_VALUE_ID;
    }
    return true;
}

bool DefineFilterBiasDw(xnn_subgraph_t subgraph,
                        const float* weights_hwc,
                        const float* bias,
                        size_t channels,
                        size_t kh,
                        size_t kw,
                        uint32_t* filter_id,
                        uint32_t* bias_id)
{
    // Subgraph DW filter layout: [1, Kh, Kw, C].
    const size_t filter_dims[4] = {1, kh, kw, channels};
    if (!DefineFp32(subgraph,
                    4,
                    filter_dims,
                    weights_hwc,
                    XNN_INVALID_VALUE_ID,
                    /*flags=*/0,
                    filter_id))
        return false;
    if (bias)
    {
        const size_t bias_dims[1] = {channels};
        if (!DefineFp32(subgraph,
                        1,
                        bias_dims,
                        bias,
                        XNN_INVALID_VALUE_ID,
                        /*flags=*/0,
                        bias_id))
            return false;
    }
    else
    {
        *bias_id = XNN_INVALID_VALUE_ID;
    }
    return true;
}

bool DefineConvNode(xnn_subgraph_t subgraph,
                    uint32_t pad_h,
                    uint32_t pad_w,
                    uint32_t kh,
                    uint32_t kw,
                    uint32_t stride,
                    size_t in_c,
                    size_t out_c,
                    float out_min,
                    float out_max,
                    uint32_t input_id,
                    uint32_t filter_id,
                    uint32_t bias_id,
                    uint32_t output_id)
{
    return xnn_define_convolution_2d(subgraph,
                                     pad_h,
                                     pad_w,
                                     pad_h,
                                     pad_w,
                                     kh,
                                     kw,
                                     stride,
                                     stride,
                                     /*dilation_height=*/1,
                                     /*dilation_width=*/1,
                                     /*groups=*/1,
                                     in_c,
                                     out_c,
                                     out_min,
                                     out_max,
                                     input_id,
                                     filter_id,
                                     bias_id,
                                     output_id,
                                     /*flags=*/0) == xnn_status_success;
}

bool DefineDwNode(xnn_subgraph_t subgraph,
                  uint32_t pad_h,
                  uint32_t pad_w,
                  uint32_t kh,
                  uint32_t kw,
                  uint32_t stride,
                  size_t channels,
                  float out_min,
                  float out_max,
                  uint32_t input_id,
                  uint32_t filter_id,
                  uint32_t bias_id,
                  uint32_t output_id)
{
    return xnn_define_depthwise_convolution_2d(subgraph,
                                               pad_h,
                                               pad_w,
                                               pad_h,
                                               pad_w,
                                               kh,
                                               kw,
                                               stride,
                                               stride,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*depth_multiplier=*/1,
                                               channels,
                                               out_min,
                                               out_max,
                                               input_id,
                                               filter_id,
                                               bias_id,
                                               output_id,
                                               /*flags=*/0) == xnn_status_success;
}

using nk_op_detail::CalcOutputDim;

bool AppendUibBody(xnn_subgraph_t subgraph,
                   XnnpackFloat::Runtime& runtime,
                   Arena& arena,
                   MobileNetV4Uib& uib,
                   uint32_t in_h,
                   uint32_t in_w,
                   uint32_t input_id,
                   uint32_t external_output_id,
                   uint32_t* output_id_out,
                   uint32_t* out_h_out,
                   uint32_t* out_w_out)
{
    uib.FoldBatchNorm();

    const uint32_t in_c = static_cast<uint32_t>(uib.in_channels);
    const uint32_t out_c = static_cast<uint32_t>(uib.out_channels);
    const uint32_t expand_c = uib.expanded_channels();
    const uint32_t block_in_id = input_id;
    uint32_t cur_id = input_id;
    uint32_t cur_h = in_h;
    uint32_t cur_w = in_w;

    if (uib.start_dw_kernel > 0)
    {
        if (!uib.start_dw_weights)
            return false;
        const int kh = uib.start_dw_kernel;
        const int pad = (kh - 1) / 2;
        const uint32_t stride = uib.start_dw_stride();
        const uint32_t next_h = CalcOutputDim(cur_h, kh, static_cast<int>(stride), pad);
        const uint32_t next_w = CalcOutputDim(cur_w, kh, static_cast<int>(stride), pad);
        float* hwc = OwnDwHwc(runtime, arena, uib.start_dw_weights, kh, kh, static_cast<int>(in_c));
        if (!hwc)
            return false;
        uint32_t filter_id = 0;
        uint32_t bias_id = 0;
        uint32_t out_id = 0;
        if (!DefineFilterBiasDw(subgraph,
                                hwc,
                                uib.start_dw_bias,
                                in_c,
                                static_cast<size_t>(kh),
                                static_cast<size_t>(kh),
                                &filter_id,
                                &bias_id))
            return false;
        if (!DefineActNhwc(subgraph, next_h, next_w, in_c, XNN_INVALID_VALUE_ID, 0, &out_id))
            return false;
        if (!DefineDwNode(subgraph,
                          static_cast<uint32_t>(pad),
                          static_cast<uint32_t>(pad),
                          static_cast<uint32_t>(kh),
                          static_cast<uint32_t>(kh),
                          stride,
                          in_c,
                          -FLT_MAX,
                          FLT_MAX,
                          cur_id,
                          filter_id,
                          bias_id,
                          out_id))
            return false;
        cur_id = out_id;
        cur_h = next_h;
        cur_w = next_w;
    }

    {
        if (!uib.expand_weights)
            return false;
        uint32_t filter_id = 0;
        uint32_t bias_id = 0;
        uint32_t out_id = 0;
        if (!DefineFilterBiasConv(subgraph,
                                  uib.expand_weights,
                                  uib.expand_bias,
                                  expand_c,
                                  1,
                                  1,
                                  in_c,
                                  &filter_id,
                                  &bias_id))
            return false;
        if (!DefineActNhwc(subgraph, cur_h, cur_w, expand_c, XNN_INVALID_VALUE_ID, 0, &out_id))
            return false;
        if (!DefineConvNode(subgraph,
                            0,
                            0,
                            1,
                            1,
                            1,
                            in_c,
                            expand_c,
                            /*out_min=*/0.0f,
                            FLT_MAX,
                            cur_id,
                            filter_id,
                            bias_id,
                            out_id))
            return false;
        cur_id = out_id;
    }

    if (uib.middle_dw_kernel > 0)
    {
        if (!uib.middle_dw_weights)
            return false;
        const int kh = uib.middle_dw_kernel;
        const int pad = (kh - 1) / 2;
        const uint32_t stride = uib.middle_dw_stride();
        const uint32_t next_h = CalcOutputDim(cur_h, kh, static_cast<int>(stride), pad);
        const uint32_t next_w = CalcOutputDim(cur_w, kh, static_cast<int>(stride), pad);
        float* hwc =
            OwnDwHwc(runtime, arena, uib.middle_dw_weights, kh, kh, static_cast<int>(expand_c));
        if (!hwc)
            return false;
        uint32_t filter_id = 0;
        uint32_t bias_id = 0;
        uint32_t out_id = 0;
        if (!DefineFilterBiasDw(subgraph,
                                hwc,
                                uib.middle_dw_bias,
                                expand_c,
                                static_cast<size_t>(kh),
                                static_cast<size_t>(kh),
                                &filter_id,
                                &bias_id))
            return false;
        if (!DefineActNhwc(subgraph, next_h, next_w, expand_c, XNN_INVALID_VALUE_ID, 0, &out_id))
            return false;
        if (!DefineDwNode(subgraph,
                          static_cast<uint32_t>(pad),
                          static_cast<uint32_t>(pad),
                          static_cast<uint32_t>(kh),
                          static_cast<uint32_t>(kh),
                          stride,
                          expand_c,
                          /*out_min=*/0.0f,
                          FLT_MAX,
                          cur_id,
                          filter_id,
                          bias_id,
                          out_id))
            return false;
        cur_id = out_id;
        cur_h = next_h;
        cur_w = next_w;
    }

    {
        if (!uib.proj_weights)
            return false;
        uint32_t filter_id = 0;
        uint32_t bias_id = 0;
        if (!DefineFilterBiasConv(subgraph,
                                  uib.proj_weights,
                                  uib.proj_bias,
                                  out_c,
                                  1,
                                  1,
                                  expand_c,
                                  &filter_id,
                                  &bias_id))
            return false;

        if (uib.has_residual())
        {
            uint32_t proj_out_id = 0;
            if (!DefineActNhwc(
                    subgraph, cur_h, cur_w, out_c, XNN_INVALID_VALUE_ID, 0, &proj_out_id))
                return false;
            if (!DefineConvNode(subgraph,
                                0,
                                0,
                                1,
                                1,
                                1,
                                expand_c,
                                out_c,
                                -FLT_MAX,
                                FLT_MAX,
                                cur_id,
                                filter_id,
                                bias_id,
                                proj_out_id))
                return false;

            uint32_t out_id = 0;
            const uint32_t out_flags =
                (external_output_id != XNN_INVALID_VALUE_ID) ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
            if (!DefineActNhwc(
                    subgraph, cur_h, cur_w, out_c, external_output_id, out_flags, &out_id))
                return false;
            if (xnn_define_binary(subgraph,
                                  xnn_binary_add,
                                  /*params=*/nullptr,
                                  proj_out_id,
                                  block_in_id,
                                  out_id,
                                  /*flags=*/0) != xnn_status_success)
                return false;
            *output_id_out = out_id;
        }
        else
        {
            uint32_t out_id = 0;
            const uint32_t out_flags =
                (external_output_id != XNN_INVALID_VALUE_ID) ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
            if (!DefineActNhwc(
                    subgraph, cur_h, cur_w, out_c, external_output_id, out_flags, &out_id))
                return false;
            if (!DefineConvNode(subgraph,
                                0,
                                0,
                                1,
                                1,
                                1,
                                expand_c,
                                out_c,
                                -FLT_MAX,
                                FLT_MAX,
                                cur_id,
                                filter_id,
                                bias_id,
                                out_id))
                return false;
            *output_id_out = out_id;
        }
    }

    *out_h_out = cur_h;
    *out_w_out = cur_w;
    return true;
}

bool DefineSilu(xnn_subgraph_t subgraph,
                uint32_t h,
                uint32_t w,
                uint32_t c,
                uint32_t input_id,
                uint32_t output_id)
{
    uint32_t sigmoid_id = 0;
    if (!DefineActNhwc(subgraph, h, w, c, XNN_INVALID_VALUE_ID, 0, &sigmoid_id))
        return false;
    if (xnn_define_unary(subgraph,
                         xnn_unary_sigmoid,
                         /*params=*/nullptr,
                         input_id,
                         sigmoid_id,
                         /*flags=*/0) != xnn_status_success)
        return false;
    return xnn_define_binary(subgraph,
                             xnn_binary_multiply,
                             /*params=*/nullptr,
                             input_id,
                             sigmoid_id,
                             output_id,
                             /*flags=*/0) == xnn_status_success;
}

bool DefineConv2dMaybeSilu(xnn_subgraph_t subgraph,
                           const float* weights_ohwi,
                           const float* bias,
                           uint32_t kh,
                           uint32_t pad,
                           uint32_t stride,
                           uint32_t in_c,
                           uint32_t out_c,
                           uint32_t in_h,
                           uint32_t in_w,
                           uint32_t input_id,
                           uint32_t* output_id_out,
                           uint32_t* out_h_out,
                           uint32_t* out_w_out,
                           bool apply_silu)
{
    const uint32_t out_h = CalcOutputDim(in_h, static_cast<int>(kh), static_cast<int>(stride),
                                         static_cast<int>(pad));
    const uint32_t out_w = CalcOutputDim(in_w, static_cast<int>(kh), static_cast<int>(stride),
                                         static_cast<int>(pad));
    uint32_t filter_id = 0;
    uint32_t bias_id = 0;
    if (!DefineFilterBiasConv(subgraph, weights_ohwi, bias, out_c, kh, kh, in_c, &filter_id,
                              &bias_id))
        return false;

    uint32_t conv_out_id = 0;
    if (!DefineActNhwc(subgraph, out_h, out_w, out_c, XNN_INVALID_VALUE_ID, 0, &conv_out_id))
        return false;
    if (!DefineConvNode(subgraph, pad, pad, kh, kh, stride, in_c, out_c, -FLT_MAX, FLT_MAX,
                        input_id, filter_id, bias_id, conv_out_id))
        return false;

    if (!apply_silu)
    {
        *output_id_out = conv_out_id;
        *out_h_out = out_h;
        *out_w_out = out_w;
        return true;
    }

    uint32_t silu_id = 0;
    if (!DefineActNhwc(subgraph, out_h, out_w, out_c, XNN_INVALID_VALUE_ID, 0, &silu_id))
        return false;
    if (!DefineSilu(subgraph, out_h, out_w, out_c, conv_out_id, silu_id))
        return false;
    *output_id_out = silu_id;
    *out_h_out = out_h;
    *out_w_out = out_w;
    return true;
}

bool DefineDwPwSilu(xnn_subgraph_t subgraph,
                    XnnpackFloat::Runtime& runtime,
                    Arena& arena,
                    float* dw_w,
                    float* dw_b,
                    float* pw_w,
                    float* pw_b,
                    uint32_t channels,
                    uint32_t stride,
                    uint32_t in_h,
                    uint32_t in_w,
                    uint32_t input_id,
                    uint32_t* output_id_out,
                    uint32_t* out_h_out,
                    uint32_t* out_w_out)
{
    const uint32_t pad = 1;
    const uint32_t kh = 3;
    const uint32_t dw_h = CalcOutputDim(in_h, static_cast<int>(kh), static_cast<int>(stride),
                                        static_cast<int>(pad));
    const uint32_t dw_w_out = CalcOutputDim(in_w, static_cast<int>(kh), static_cast<int>(stride),
                                            static_cast<int>(pad));
    float* hwc = OwnDwHwc(runtime, arena, dw_w, static_cast<int>(kh), static_cast<int>(kh),
                          static_cast<int>(channels));
    if (!hwc)
        return false;

    uint32_t filter_id = 0;
    uint32_t bias_id = 0;
    uint32_t dw_out_id = 0;
    if (!DefineFilterBiasDw(subgraph, hwc, dw_b, channels, kh, kh, &filter_id, &bias_id))
        return false;
    if (!DefineActNhwc(subgraph, dw_h, dw_w_out, channels, XNN_INVALID_VALUE_ID, 0, &dw_out_id))
        return false;
    if (!DefineDwNode(subgraph, pad, pad, kh, kh, stride, channels, -FLT_MAX, FLT_MAX, input_id,
                      filter_id, bias_id, dw_out_id))
        return false;

    return DefineConv2dMaybeSilu(subgraph, pw_w, pw_b, /*kh=*/1, /*pad=*/0, /*stride=*/1, channels,
                                 channels, dw_h, dw_w_out, dw_out_id, output_id_out, out_h_out,
                                 out_w_out, /*apply_silu=*/true);
}

bool DefineNearestUpsample2x(xnn_subgraph_t subgraph,
                             uint32_t h,
                             uint32_t w,
                             uint32_t c,
                             uint32_t input_id,
                             uint32_t* output_id_out)
{
    // [1,H,W,C] -> [1,H,1,W,1,C] -> broadcast [1,H,2,W,2,C] -> [1,2H,2W,C]
    uint32_t expanded_id = 0;
    const size_t expand_dims[6] = {1, h, 1, w, 1, c};
    if (!DefineFp32(subgraph, 6, expand_dims, nullptr, XNN_INVALID_VALUE_ID, 0, &expanded_id))
        return false;
    if (xnn_define_static_reshape(subgraph, 6, expand_dims, input_id, expanded_id, 0) !=
        xnn_status_success)
        return false;

    uint32_t broadcast_id = 0;
    const size_t bcast_dims[6] = {1, h, 2, w, 2, c};
    if (!DefineFp32(subgraph, 6, bcast_dims, nullptr, XNN_INVALID_VALUE_ID, 0, &broadcast_id))
        return false;
    if (xnn_define_static_broadcast(subgraph, 6, bcast_dims, expanded_id, broadcast_id, 0) !=
        xnn_status_success)
        return false;

    uint32_t out_id = 0;
    const uint32_t out_h = h * 2u;
    const uint32_t out_w = w * 2u;
    if (!DefineActNhwc(subgraph, out_h, out_w, c, XNN_INVALID_VALUE_ID, 0, &out_id))
        return false;
    const size_t out_dims[4] = {1, out_h, out_w, c};
    if (xnn_define_static_reshape(subgraph, 4, out_dims, broadcast_id, out_id, 0) !=
        xnn_status_success)
        return false;
    *output_id_out = out_id;
    return true;
}

bool DefineAdd(xnn_subgraph_t subgraph,
               uint32_t h,
               uint32_t w,
               uint32_t c,
               uint32_t a_id,
               uint32_t b_id,
               uint32_t* output_id_out)
{
    uint32_t out_id = 0;
    if (!DefineActNhwc(subgraph, h, w, c, XNN_INVALID_VALUE_ID, 0, &out_id))
        return false;
    if (xnn_define_binary(subgraph, xnn_binary_add, nullptr, a_id, b_id, out_id, 0) !=
        xnn_status_success)
        return false;
    *output_id_out = out_id;
    return true;
}

bool AppendDecoupledHead(xnn_subgraph_t subgraph,
                         const YoloxDecoupledHead& head,
                         uint32_t h,
                         uint32_t w,
                         uint32_t input_id,
                         uint32_t* output_id_out)
{
    const uint32_t hidden = static_cast<uint32_t>(head.hidden_dim);
    const uint32_t in_c = static_cast<uint32_t>(head.in_channels);
    const uint32_t num_classes = static_cast<uint32_t>(head.num_classes);
    const uint32_t out_c = static_cast<uint32_t>(head.output_channels());

    uint32_t stem_id = 0;
    uint32_t sh = 0;
    uint32_t sw = 0;
    if (!DefineConv2dMaybeSilu(subgraph, head.stem_weights, head.stem_bias, 1, 0, 1, in_c, hidden, h,
                               w, input_id, &stem_id, &sh, &sw, true))
        return false;

    uint32_t cls_id = stem_id;
    for (int i = 0; i < head.num_convs; ++i)
    {
        uint32_t next = 0;
        if (!DefineConv2dMaybeSilu(subgraph, head.cls_conv_weights[i], head.cls_conv_bias[i], 3, 1, 1,
                                   hidden, hidden, sh, sw, cls_id, &next, &sh, &sw, true))
            return false;
        cls_id = next;
    }
    uint32_t cls_pred = 0;
    uint32_t ph = 0;
    uint32_t pw = 0;
    if (!DefineConv2dMaybeSilu(subgraph, head.cls_pred_weights, head.cls_pred_bias, 1, 0, 1, hidden,
                               num_classes, sh, sw, cls_id, &cls_pred, &ph, &pw, false))
        return false;

    // Reg/obj share the stem; spatial stays HxW through 3x3 pad1.
    uint32_t reg_id = stem_id;
    sh = h;
    sw = w;
    for (int i = 0; i < head.num_convs; ++i)
    {
        uint32_t next = 0;
        if (!DefineConv2dMaybeSilu(subgraph, head.reg_conv_weights[i], head.reg_conv_bias[i], 3, 1, 1,
                                   hidden, hidden, sh, sw, reg_id, &next, &sh, &sw, true))
            return false;
        reg_id = next;
    }
    uint32_t reg_pred = 0;
    if (!DefineConv2dMaybeSilu(subgraph, head.reg_pred_weights, head.reg_pred_bias, 1, 0, 1, hidden, 4,
                               sh, sw, reg_id, &reg_pred, &ph, &pw, false))
        return false;
    uint32_t obj_pred = 0;
    if (!DefineConv2dMaybeSilu(subgraph, head.obj_pred_weights, head.obj_pred_bias, 1, 0, 1, hidden, 1,
                               sh, sw, reg_id, &obj_pred, &ph, &pw, false))
        return false;

    // Channel order matches reference ScatterChannels: reg(4) | obj(1) | cls(C).
    uint32_t head_out = 0;
    if (!DefineActNhwc(subgraph, h, w, out_c, XNN_INVALID_VALUE_ID, 0, &head_out))
        return false;
    const uint32_t parts[3] = {reg_pred, obj_pred, cls_pred};
    if (xnn_define_concatenate(subgraph, /*axis=*/3, 3, parts, head_out, 0) != xnn_status_success)
        return false;
    *output_id_out = head_out;
    return true;
}

bool AppendYoloxPafpn(xnn_subgraph_t subgraph,
                      XnnpackFloat::Runtime& runtime,
                      Arena& arena,
                      YoloxPafpnMultiscale& neck,
                      uint32_t c5_h,
                      uint32_t c5_w,
                      uint32_t c5_id,
                      uint32_t c3_id,
                      uint32_t c4_id,
                      uint32_t external_output_id,
                      uint32_t* output_id_out,
                      uint32_t* out_elems_out)
{
    if (c3_id == XNN_INVALID_VALUE_ID || c4_id == XNN_INVALID_VALUE_ID)
        return false;
    if (!neck.lat3_weights || !neck.lat4_weights || !neck.lat5_weights)
        return false;

    const uint32_t h5 = c5_h;
    const uint32_t w5 = c5_w;
    const uint32_t h4 = h5 * 2u;
    const uint32_t w4 = w5 * 2u;
    const uint32_t h3 = h5 * 4u;
    const uint32_t w3 = w5 * 4u;
    const uint32_t H = static_cast<uint32_t>(neck.hidden_dim);
    const uint32_t c3 = static_cast<uint32_t>(neck.c3_channels);
    const uint32_t c4 = static_cast<uint32_t>(neck.c4_channels);
    const uint32_t c5 = static_cast<uint32_t>(neck.c5_channels);

    uint32_t l5 = 0;
    uint32_t th = 0;
    uint32_t tw = 0;
    if (!DefineConv2dMaybeSilu(subgraph, neck.lat5_weights, neck.lat5_bias, 1, 0, 1, c5, H, h5, w5,
                               c5_id, &l5, &th, &tw, false))
        return false;

    uint32_t l4 = 0;
    if (!DefineConv2dMaybeSilu(subgraph, neck.lat4_weights, neck.lat4_bias, 1, 0, 1, c4, H, h4, w4,
                               c4_id, &l4, &th, &tw, false))
        return false;

    uint32_t up5 = 0;
    if (!DefineNearestUpsample2x(subgraph, h5, w5, H, l5, &up5))
        return false;
    uint32_t l4_sum = 0;
    if (!DefineAdd(subgraph, h4, w4, H, l4, up5, &l4_sum))
        return false;

    uint32_t p4 = 0;
    if (!DefineDwPwSilu(subgraph, runtime, arena, neck.td_p4_dw_weights, neck.td_p4_dw_bias,
                        neck.td_p4_pw_weights, neck.td_p4_pw_bias, H, /*stride=*/1, h4, w4, l4_sum,
                        &p4, &th, &tw))
        return false;

    uint32_t l3 = 0;
    if (!DefineConv2dMaybeSilu(subgraph, neck.lat3_weights, neck.lat3_bias, 1, 0, 1, c3, H, h3, w3,
                               c3_id, &l3, &th, &tw, false))
        return false;
    uint32_t up4 = 0;
    if (!DefineNearestUpsample2x(subgraph, h4, w4, H, p4, &up4))
        return false;
    uint32_t l3_sum = 0;
    if (!DefineAdd(subgraph, h3, w3, H, l3, up4, &l3_sum))
        return false;

    uint32_t p3 = 0;
    if (!DefineDwPwSilu(subgraph, runtime, arena, neck.td_p3_dw_weights, neck.td_p3_dw_bias,
                        neck.td_p3_pw_weights, neck.td_p3_pw_bias, H, /*stride=*/1, h3, w3, l3_sum,
                        &p3, &th, &tw))
        return false;

    // Bottom-up
    uint32_t n4_body = 0;
    if (!DefineDwPwSilu(subgraph, runtime, arena, neck.bu_n4_dw_weights, neck.bu_n4_dw_bias,
                        neck.bu_n4_pw_weights, neck.bu_n4_pw_bias, H, /*stride=*/2, h3, w3, p3,
                        &n4_body, &th, &tw))
        return false;
    uint32_t n4 = 0;
    if (!DefineAdd(subgraph, h4, w4, H, n4_body, p4, &n4))
        return false;

    uint32_t n5_body = 0;
    if (!DefineDwPwSilu(subgraph, runtime, arena, neck.bu_n5_dw_weights, neck.bu_n5_dw_bias,
                        neck.bu_n5_pw_weights, neck.bu_n5_pw_bias, H, /*stride=*/2, h4, w4, n4,
                        &n5_body, &th, &tw))
        return false;
    uint32_t n5 = 0;
    if (!DefineAdd(subgraph, h5, w5, H, n5_body, l5, &n5))
        return false;

    const uint32_t scale_h[3] = {h3, h4, h5};
    const uint32_t scale_w[3] = {w3, w4, w5};
    const uint32_t scale_in[3] = {p3, n4, n5};
    uint32_t flat_ids[3] = {};
    const uint32_t out_c = static_cast<uint32_t>(neck.output_channels());
    uint32_t total_elems = 0;

    for (int s = 0; s < YoloxPafpnMultiscale::kNumScales; ++s)
    {
        neck.heads[s].in_channels = neck.hidden_dim;
        neck.heads[s].hidden_dim = neck.hidden_dim;
        neck.heads[s].num_classes = neck.num_classes;
        neck.heads[s].num_convs = neck.num_convs;

        uint32_t head_id = 0;
        if (!AppendDecoupledHead(subgraph, neck.heads[s], scale_h[s], scale_w[s], scale_in[s],
                                 &head_id))
            return false;

        const uint32_t spat = scale_h[s] * scale_w[s];
        const uint32_t elems = spat * out_c;
        total_elems += elems;
        uint32_t flat_id = 0;
        const size_t flat_dims[4] = {1, 1, 1, elems};
        if (!DefineFp32(subgraph, 4, flat_dims, nullptr, XNN_INVALID_VALUE_ID, 0, &flat_id))
            return false;
        if (xnn_define_static_reshape(subgraph, 4, flat_dims, head_id, flat_id, 0) !=
            xnn_status_success)
            return false;
        flat_ids[s] = flat_id;
    }

    uint32_t out_id = 0;
    const uint32_t out_flags =
        (external_output_id != XNN_INVALID_VALUE_ID) ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
    const size_t out_dims[4] = {1, 1, 1, total_elems};
    if (!DefineFp32(subgraph, 4, out_dims, nullptr, external_output_id, out_flags, &out_id))
        return false;
    if (xnn_define_concatenate(subgraph, /*axis=*/3, 3, flat_ids, out_id, 0) != xnn_status_success)
        return false;

    *output_id_out = out_id;
    *out_elems_out = total_elems;
    return true;
}

bool FinishAfterWeightsCache(XnnpackFloat::Runtime& runtime)
{
    if (runtime.xnn_network_ready)
        return true;
    if (!runtime.xnn_network_runtime || runtime.in_h == 0)
        return false;

    auto* xnn_rt = static_cast<xnn_runtime_t>(runtime.xnn_network_runtime);
    const size_t in_dims[4] = {1,
                               static_cast<size_t>(runtime.in_h),
                               static_cast<size_t>(runtime.in_w),
                               static_cast<size_t>(runtime.in_c)};
    if (xnn_reshape_external_value(xnn_rt, runtime.xnn_net_ext_in, 4, in_dims) !=
        xnn_status_success)
    {
        std::fprintf(stderr, "XnnpackFloat: reshape_external_value failed\n");
        return false;
    }
    if (xnn_reshape_runtime(xnn_rt) != xnn_status_success)
    {
        std::fprintf(stderr, "XnnpackFloat: reshape_runtime failed\n");
        return false;
    }
    runtime.xnn_network_ready = true;
    return true;
}

}  // namespace

namespace XnnpackFloat
{

void DestroyRuntime(Runtime& runtime)
{
    if (runtime.xnn_network_runtime)
    {
        (void)xnn_delete_runtime(static_cast<xnn_runtime_t>(runtime.xnn_network_runtime));
        runtime.xnn_network_runtime = nullptr;
    }
    if (runtime.xnn_weights_cache)
    {
        (void)xnn_delete_weights_cache(static_cast<xnn_weights_cache_t>(runtime.xnn_weights_cache));
        runtime.xnn_weights_cache = nullptr;
    }
    if (runtime.xnn_workspace)
    {
        (void)xnn_release_workspace(static_cast<xnn_workspace_t>(runtime.xnn_workspace));
        runtime.xnn_workspace = nullptr;
    }
    if (runtime.dw_hwc_owned)
    {
        for (uint32_t i = 0; i < runtime.dw_hwc_count; ++i)
            delete[] runtime.dw_hwc_owned[i];
        delete[] runtime.dw_hwc_owned;
        runtime.dw_hwc_owned = nullptr;
        runtime.dw_hwc_count = 0;
    }
    runtime.xnn_network_ready = false;
    runtime.xnn_net_ext_in = 0;
    runtime.xnn_net_ext_out = 1;
    runtime.bound_input = nullptr;
    runtime.bound_output = nullptr;
    runtime.in_h = runtime.in_w = runtime.in_c = 0;
    runtime.out_elements = 0;
}

bool BuildNetworkRuntime(CNNNetwork& network,
                         Arena& arena,
                         uint32_t in_h,
                         uint32_t in_w,
                         uint32_t in_c,
                         Runtime*& out_runtime)
{
    out_runtime = nullptr;
    if (!network.IsValid() || network.layer_count() == 0 || !EnsureXnnInitialized())
        return false;

    Runtime* runtime = new (std::nothrow) Runtime{};
    if (!runtime)
        return false;

    auto fail = [&](const char* reason) {
        std::fprintf(stderr, "XnnpackFloat::BuildNetworkRuntime: %s\n", reason);
        DestroyRuntime(*runtime);
        delete runtime;
        return false;
    };

    // Match int8 / TF Lite: large weights cache + shared workspace.
    constexpr size_t kWeightsCacheBytes = 16u * 1024u * 1024u;
    xnn_weights_cache_t cache = nullptr;
    if (xnn_create_weights_cache_with_size(kWeightsCacheBytes, &cache) == xnn_status_success ||
        xnn_create_weights_cache(&cache) == xnn_status_success)
        runtime->xnn_weights_cache = cache;
    xnn_workspace_t ws = nullptr;
    if (xnn_create_workspace(&ws) == xnn_status_success)
        runtime->xnn_workspace = ws;

    runtime->xnn_net_ext_in = 0;
    runtime->xnn_net_ext_out = 1;
    runtime->in_h = in_h;
    runtime->in_w = in_w;
    runtime->in_c = in_c;

    xnn_subgraph_t subgraph = nullptr;
    if (xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph) !=
        xnn_status_success)
        return fail("xnn_create_subgraph failed");

    auto fail_sg = [&](const char* reason) {
        (void)xnn_delete_subgraph(subgraph);
        return fail(reason);
    };

    uint32_t cur_id = 0;
    if (!DefineActNhwc(subgraph,
                       in_h,
                       in_w,
                       in_c,
                       runtime->xnn_net_ext_in,
                       XNN_VALUE_FLAG_EXTERNAL_INPUT,
                       &cur_id))
        return fail_sg("define external input failed");

    uint32_t cur_h = in_h;
    uint32_t cur_w = in_w;
    uint32_t cur_c = in_c;
    const uint32_t n = network.layer_count();
    // Feature taps (YOLOX C3/C4): identity in the subgraph; keep value IDs for PAFPN.
    constexpr uint32_t kMaxTaps = 4;
    uint32_t tap_value_ids[kMaxTaps];
    for (uint32_t t = 0; t < kMaxTaps; ++t)
        tap_value_ids[t] = XNN_INVALID_VALUE_ID;

    for (uint32_t i = 0; i < n; ++i)
    {
        CnnBlock& block = network.GetBlock(i);
        const bool is_last = i + 1 == n;
        char layer_err[96];

        switch (block.type)
        {
            case CnnBlockType::Flatten:
            {
                // Reshape NHWC [1,H,W,C] → [1,1,1,H*W*C] so Dense last-dim matches
                // (MNIST CNN). ImageNet head is already 1×1×C after global pool.
                if (cur_h == 1 && cur_w == 1)
                    break;
                const uint32_t features = cur_h * cur_w * cur_c;
                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime->xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                if (!DefineActNhwc(subgraph, 1, 1, features, out_ext, out_flags, &out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "flatten act failed layer %u", i);
                    return fail_sg(layer_err);
                }
                const size_t new_shape[4] = {1, 1, 1, static_cast<size_t>(features)};
                if (xnn_define_static_reshape(subgraph,
                                              /*num_dims=*/4,
                                              new_shape,
                                              cur_id,
                                              out_id,
                                              /*flags=*/0) != xnn_status_success)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "flatten reshape failed layer %u", i);
                    return fail_sg(layer_err);
                }
                cur_id = out_id;
                cur_h = 1;
                cur_w = 1;
                cur_c = features;
                break;
            }

            case CnnBlockType::Conv2D:
            {
                const Conv2D& conv = block.conv.conv;
                if (conv.pad_h != conv.pad_h_end || conv.pad_w != conv.pad_w_end)
                    return fail_sg("asymmetric conv padding unsupported");
                if (!conv.weights)
                    return fail_sg("conv weights missing");
                float out_min = -FLT_MAX;
                float out_max = FLT_MAX;
                if (!ActivationClamp(block.conv.activation, out_min, out_max))
                    return fail_sg("unsupported conv activation");

                const uint32_t out_h = CalcOutputDim(
                    cur_h, conv.kernel_size, conv.stride, conv.pad_h);
                const uint32_t out_w = CalcOutputDim(
                    cur_w, conv.kernel_size, conv.stride, conv.pad_w);
                const uint32_t out_c = static_cast<uint32_t>(conv.out_channels);

                uint32_t filter_id = 0;
                uint32_t bias_id = 0;
                if (!DefineFilterBiasConv(subgraph,
                                          conv.weights,
                                          conv.bias,
                                          out_c,
                                          static_cast<size_t>(conv.kernel_size),
                                          static_cast<size_t>(conv.kernel_size),
                                          static_cast<size_t>(conv.in_channels),
                                          &filter_id,
                                          &bias_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "conv filter/bias failed layer %u", i);
                    return fail_sg(layer_err);
                }

                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime->xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                if (!DefineActNhwc(subgraph, out_h, out_w, out_c, out_ext, out_flags, &out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "conv act failed layer %u", i);
                    return fail_sg(layer_err);
                }
                if (!DefineConvNode(subgraph,
                                    static_cast<uint32_t>(conv.pad_h),
                                    static_cast<uint32_t>(conv.pad_w),
                                    static_cast<uint32_t>(conv.kernel_size),
                                    static_cast<uint32_t>(conv.kernel_size),
                                    static_cast<uint32_t>(conv.stride),
                                    static_cast<size_t>(conv.in_channels),
                                    out_c,
                                    out_min,
                                    out_max,
                                    cur_id,
                                    filter_id,
                                    bias_id,
                                    out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "conv node failed layer %u", i);
                    return fail_sg(layer_err);
                }
                cur_id = out_id;
                cur_h = out_h;
                cur_w = out_w;
                cur_c = out_c;
                break;
            }

            case CnnBlockType::DepthwiseConv2D:
            {
                const DepthwiseConv2D& dw = block.depthwise_conv.depthwise;
                if (dw.pad_h != dw.pad_h_end || dw.pad_w != dw.pad_w_end)
                    return fail_sg("asymmetric dw padding unsupported");
                if (!dw.weights)
                    return fail_sg("dw weights missing");
                float out_min = -FLT_MAX;
                float out_max = FLT_MAX;
                if (!ActivationClamp(block.depthwise_conv.activation, out_min, out_max))
                    return fail_sg("unsupported dw activation");

                const uint32_t channels = static_cast<uint32_t>(dw.channels);
                const uint32_t out_h =
                    CalcOutputDim(cur_h, dw.kernel_h, dw.stride, dw.pad_h);
                const uint32_t out_w =
                    CalcOutputDim(cur_w, dw.kernel_w, dw.stride, dw.pad_w);
                float* hwc =
                    OwnDwHwc(*runtime, arena, dw.weights, dw.kernel_h, dw.kernel_w, dw.channels);
                if (!hwc)
                    return fail_sg("dw weight repack failed");

                uint32_t filter_id = 0;
                uint32_t bias_id = 0;
                if (!DefineFilterBiasDw(subgraph,
                                        hwc,
                                        dw.bias,
                                        channels,
                                        static_cast<size_t>(dw.kernel_h),
                                        static_cast<size_t>(dw.kernel_w),
                                        &filter_id,
                                        &bias_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "dw filter/bias failed layer %u", i);
                    return fail_sg(layer_err);
                }

                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime->xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                if (!DefineActNhwc(
                        subgraph, out_h, out_w, channels, out_ext, out_flags, &out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "dw act failed layer %u", i);
                    return fail_sg(layer_err);
                }
                if (!DefineDwNode(subgraph,
                                  static_cast<uint32_t>(dw.pad_h),
                                  static_cast<uint32_t>(dw.pad_w),
                                  static_cast<uint32_t>(dw.kernel_h),
                                  static_cast<uint32_t>(dw.kernel_w),
                                  static_cast<uint32_t>(dw.stride),
                                  channels,
                                  out_min,
                                  out_max,
                                  cur_id,
                                  filter_id,
                                  bias_id,
                                  out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "dw node failed layer %u", i);
                    return fail_sg(layer_err);
                }
                cur_id = out_id;
                cur_h = out_h;
                cur_w = out_w;
                cur_c = channels;
                break;
            }

            case CnnBlockType::MobilenetV4Uib:
            {
                MobileNetV4Uib& uib = block.mobilenetv4_uib.block;
                uint32_t out_id = 0;
                uint32_t out_h = 0;
                uint32_t out_w = 0;
                const uint32_t out_ext =
                    is_last ? runtime->xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                if (!AppendUibBody(subgraph,
                                   *runtime,
                                   arena,
                                   uib,
                                   cur_h,
                                   cur_w,
                                   cur_id,
                                   out_ext,
                                   &out_id,
                                   &out_h,
                                   &out_w))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "UIB body failed layer %u", i);
                    return fail_sg(layer_err);
                }
                cur_id = out_id;
                cur_h = out_h;
                cur_w = out_w;
                cur_c = static_cast<uint32_t>(uib.out_channels);
                break;
            }

            case CnnBlockType::MaxPool2D:
            {
                const MaxPool2DLayer& pool = block.pool;
                if (pool.pad_h != pool.pad_h_end || pool.pad_w != pool.pad_w_end)
                    return fail_sg("asymmetric maxpool padding unsupported");
                float out_min = -FLT_MAX;
                float out_max = FLT_MAX;
                if (!ActivationClamp(pool.activation, out_min, out_max))
                    return fail_sg("unsupported maxpool activation");

                const uint32_t out_h =
                    CalcOutputDim(cur_h, pool.pool_h, pool.stride, pool.pad_h);
                const uint32_t out_w =
                    CalcOutputDim(cur_w, pool.pool_w, pool.stride, pool.pad_w);
                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime->xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                if (!DefineActNhwc(
                        subgraph, out_h, out_w, cur_c, out_ext, out_flags, &out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "maxpool act failed layer %u", i);
                    return fail_sg(layer_err);
                }
                if (xnn_define_max_pooling_2d(
                        subgraph,
                        static_cast<uint32_t>(pool.pad_h),
                        static_cast<uint32_t>(pool.pad_w),
                        static_cast<uint32_t>(pool.pad_h),
                        static_cast<uint32_t>(pool.pad_w),
                        static_cast<uint32_t>(pool.pool_h),
                        static_cast<uint32_t>(pool.pool_w),
                        static_cast<uint32_t>(pool.stride),
                        static_cast<uint32_t>(pool.stride),
                        /*dilation_height=*/1,
                        /*dilation_width=*/1,
                        out_min,
                        out_max,
                        cur_id,
                        out_id,
                        /*flags=*/0) != xnn_status_success)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "maxpool node failed layer %u", i);
                    return fail_sg(layer_err);
                }
                cur_id = out_id;
                cur_h = out_h;
                cur_w = out_w;
                break;
            }

            case CnnBlockType::AvgPool2D:
            {
                const AvgPool2DLayer& pool = block.avg_pool;
                if (pool.pad_h != pool.pad_h_end || pool.pad_w != pool.pad_w_end)
                    return fail_sg("asymmetric avgpool padding unsupported");

                // Global / large-kernel avgpool: prefer reduce_mean KEEP_DIMS so
                // a following 1×1 conv still sees NHWC [1,1,1,C] (matches int8).
                const bool global =
                    pool.pool_h == static_cast<int>(cur_h) &&
                    pool.pool_w == static_cast<int>(cur_w) && pool.pad_h == 0 &&
                    pool.pad_w == 0;

                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime->xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                uint32_t out_h = 0;
                uint32_t out_w = 0;

                if (global)
                {
                    out_h = 1;
                    out_w = 1;
                    if (!DefineActNhwc(
                            subgraph, out_h, out_w, cur_c, out_ext, out_flags, &out_id))
                    {
                        std::snprintf(layer_err, sizeof(layer_err),
                                      "avgpool act failed layer %u", i);
                        return fail_sg(layer_err);
                    }
                    const int64_t axes[2] = {1, 2};
                    if (xnn_define_static_reduce_v2(subgraph,
                                                    xnn_reduce_mean,
                                                    /*num_reduction_axes=*/2,
                                                    axes,
                                                    cur_id,
                                                    out_id,
                                                    XNN_FLAG_KEEP_DIMS) != xnn_status_success)
                    {
                        std::snprintf(layer_err, sizeof(layer_err),
                                      "avgpool reduce_mean failed layer %u", i);
                        return fail_sg(layer_err);
                    }
                }
                else
                {
                    out_h = CalcOutputDim(cur_h, pool.pool_h, pool.stride, pool.pad_h);
                    out_w = CalcOutputDim(cur_w, pool.pool_w, pool.stride, pool.pad_w);
                    if (!DefineActNhwc(
                            subgraph, out_h, out_w, cur_c, out_ext, out_flags, &out_id))
                    {
                        std::snprintf(layer_err, sizeof(layer_err),
                                      "avgpool act failed layer %u", i);
                        return fail_sg(layer_err);
                    }
                    if (xnn_define_average_pooling_2d(
                            subgraph,
                            static_cast<uint32_t>(pool.pad_h),
                            static_cast<uint32_t>(pool.pad_w),
                            static_cast<uint32_t>(pool.pad_h),
                            static_cast<uint32_t>(pool.pad_w),
                            static_cast<uint32_t>(pool.pool_h),
                            static_cast<uint32_t>(pool.pool_w),
                            static_cast<uint32_t>(pool.stride),
                            static_cast<uint32_t>(pool.stride),
                            -FLT_MAX,
                            FLT_MAX,
                            cur_id,
                            out_id,
                            /*flags=*/0) != xnn_status_success)
                    {
                        std::snprintf(layer_err, sizeof(layer_err),
                                      "avgpool node failed layer %u", i);
                        return fail_sg(layer_err);
                    }
                }
                cur_id = out_id;
                cur_h = out_h;
                cur_w = out_w;
                break;
            }

            case CnnBlockType::Dense:
            {
                const MLPLayer& dense = block.dense;
                if (!dense.weights.data || dense.weights.rank != 2)
                    return fail_sg("dense weights missing/invalid");
                float out_min = -FLT_MAX;
                float out_max = FLT_MAX;
                if (!ActivationClampDense(dense.activation, out_min, out_max))
                    return fail_sg("unsupported dense activation");

                const size_t out_features = dense.weights.shape[0];
                const size_t in_features = dense.weights.shape[1];
                // After global pool + 1×1 head, tensor is NHWC 1×1×C; FC last-dim
                // must match C (Flatten is a no-op in the subgraph).
                if (in_features != cur_c)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "dense in_features mismatch layer %u "
                                  "(want %u got %zu)",
                                  i,
                                  cur_c,
                                  in_features);
                    return fail_sg(layer_err);
                }

                const size_t filter_dims[2] = {out_features, in_features};
                uint32_t filter_id = 0;
                if (!DefineFp32(subgraph,
                                2,
                                filter_dims,
                                dense.weights.data,
                                XNN_INVALID_VALUE_ID,
                                0,
                                &filter_id))
                    return fail_sg("dense filter failed");

                uint32_t bias_id = XNN_INVALID_VALUE_ID;
                if (dense.bias.data)
                {
                    size_t bias_out = 0;
                    if (dense.bias.rank == 1)
                        bias_out = dense.bias.shape[0];
                    else if (dense.bias.rank == 2 && dense.bias.shape[0] == 1)
                        bias_out = dense.bias.shape[1];
                    else
                        return fail_sg("dense bias shape invalid");
                    if (bias_out != out_features)
                        return fail_sg("dense bias size mismatch");
                    const size_t bias_dims[1] = {out_features};
                    if (!DefineFp32(subgraph,
                                    1,
                                    bias_dims,
                                    dense.bias.data,
                                    XNN_INVALID_VALUE_ID,
                                    0,
                                    &bias_id))
                        return fail_sg("dense bias failed");
                }

                // Keep output rank matched to NHWC input (1×1×out) like int8 path.
                uint32_t out_id = 0;
                const uint32_t out_ext =
                    is_last ? runtime->xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
                const size_t out_dims[4] = {1, 1, 1, out_features};
                if (!DefineFp32(subgraph, 4, out_dims, nullptr, out_ext, out_flags, &out_id))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "dense act failed layer %u", i);
                    return fail_sg(layer_err);
                }
                if (xnn_define_fully_connected(subgraph,
                                               out_min,
                                               out_max,
                                               cur_id,
                                               filter_id,
                                               bias_id,
                                               out_id,
                                               /*flags=*/0) != xnn_status_success)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "dense node failed layer %u", i);
                    return fail_sg(layer_err);
                }
                cur_id = out_id;
                cur_h = 1;
                cur_w = 1;
                cur_c = static_cast<uint32_t>(out_features);
                runtime->out_elements = static_cast<uint32_t>(out_features);
                break;
            }

            case CnnBlockType::FeatureTap:
            {
                const FeatureTapLayer& tap = block.feature_tap;
                if (tap.tap_id >= kMaxTaps)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "feature_tap id %u out of range at layer %u",
                                  static_cast<unsigned>(tap.tap_id), i);
                    return fail_sg(layer_err);
                }
                if (static_cast<uint32_t>(tap.channels) != cur_c)
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "feature_tap channels mismatch at layer %u", i);
                    return fail_sg(layer_err);
                }
                // Identity: PAFPN reads this activation via the stored value id.
                tap_value_ids[tap.tap_id] = cur_id;
                break;
            }

            case CnnBlockType::YoloxPafpnMultiscale:
            {
                YoloxPafpnMultiscale& neck = block.yolox_pafpn.block;
                const uint32_t out_ext =
                    is_last ? runtime->xnn_net_ext_out : XNN_INVALID_VALUE_ID;
                uint32_t out_id = 0;
                uint32_t out_elems = 0;
                if (!AppendYoloxPafpn(subgraph,
                                      *runtime,
                                      arena,
                                      neck,
                                      cur_h,
                                      cur_w,
                                      cur_id,
                                      tap_value_ids[0],
                                      tap_value_ids[1],
                                      out_ext,
                                      &out_id,
                                      &out_elems))
                {
                    std::snprintf(layer_err, sizeof(layer_err),
                                  "yolox_pafpn failed layer %u", i);
                    return fail_sg(layer_err);
                }
                cur_id = out_id;
                cur_h = 1;
                cur_w = 1;
                cur_c = out_elems;
                runtime->out_elements = out_elems;
                break;
            }

            default:
                std::snprintf(layer_err, sizeof(layer_err),
                              "unsupported layer type %d at %u",
                              static_cast<int>(block.type),
                              i);
                return fail_sg(layer_err);
        }
    }

    if (runtime->out_elements == 0)
        runtime->out_elements = cur_h * cur_w * cur_c;

    xnn_runtime_t xnn_rt = nullptr;
    const xnn_status create_st =
        xnn_create_runtime_v4(subgraph,
                              static_cast<xnn_weights_cache_t>(runtime->xnn_weights_cache),
                              static_cast<xnn_workspace_t>(runtime->xnn_workspace),
                              /*threadpool=*/nullptr,
                              XNN_FLAG_DONT_SPIN_WORKERS,
                              &xnn_rt);
    (void)xnn_delete_subgraph(subgraph);
    subgraph = nullptr;
    if (create_st != xnn_status_success || !xnn_rt)
    {
        std::fprintf(stderr,
                     "XnnpackFloat::BuildNetworkRuntime: xnn_create_runtime_v4 failed (%s)\n",
                     XnnStatusName(create_st));
        DestroyRuntime(*runtime);
        delete runtime;
        return false;
    }
    runtime->xnn_network_runtime = xnn_rt;

    if (runtime->xnn_weights_cache)
    {
        auto* cache_ptr = static_cast<xnn_weights_cache_t>(runtime->xnn_weights_cache);
        if (xnn_finalize_weights_cache(cache_ptr, xnn_weights_cache_finalization_kind_hard) !=
            xnn_status_success)
        {
            (void)xnn_finalize_weights_cache(cache_ptr,
                                             xnn_weights_cache_finalization_kind_soft);
        }
    }

    if (!FinishAfterWeightsCache(*runtime))
    {
        DestroyRuntime(*runtime);
        delete runtime;
        return false;
    }

    std::fprintf(stderr, "XnnpackFloat: network xnn_subgraph ready\n");
    out_runtime = runtime;
    return true;
}

bool BuildMlpRuntime(MLPNetwork& network,
                     Arena& arena,
                     uint32_t in_features,
                     Runtime*& out_runtime)
{
    (void)arena;
    out_runtime = nullptr;
    if (!network.IsValid() || network.layer_count() == 0 || in_features == 0 ||
        !EnsureXnnInitialized())
        return false;

    Runtime* runtime = new (std::nothrow) Runtime{};
    if (!runtime)
        return false;

    auto fail = [&](const char* reason) {
        std::fprintf(stderr, "XnnpackFloat::BuildMlpRuntime: %s\n", reason);
        DestroyRuntime(*runtime);
        delete runtime;
        return false;
    };

    constexpr size_t kWeightsCacheBytes = 16u * 1024u * 1024u;
    xnn_weights_cache_t cache = nullptr;
    if (xnn_create_weights_cache_with_size(kWeightsCacheBytes, &cache) == xnn_status_success ||
        xnn_create_weights_cache(&cache) == xnn_status_success)
        runtime->xnn_weights_cache = cache;
    xnn_workspace_t ws = nullptr;
    if (xnn_create_workspace(&ws) == xnn_status_success)
        runtime->xnn_workspace = ws;

    runtime->xnn_net_ext_in = 0;
    runtime->xnn_net_ext_out = 1;
    runtime->in_h = 1;
    runtime->in_w = 1;
    runtime->in_c = in_features;

    xnn_subgraph_t subgraph = nullptr;
    if (xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph) !=
        xnn_status_success)
        return fail("xnn_create_subgraph failed");

    auto fail_sg = [&](const char* reason) {
        (void)xnn_delete_subgraph(subgraph);
        return fail(reason);
    };

    // External input [1, in_features] (2D) — matches TF Lite MLP layout.
    uint32_t cur_id = 0;
    const size_t in_dims[2] = {1, static_cast<size_t>(in_features)};
    if (!DefineFp32(subgraph,
                    2,
                    in_dims,
                    nullptr,
                    runtime->xnn_net_ext_in,
                    XNN_VALUE_FLAG_EXTERNAL_INPUT,
                    &cur_id))
        return fail_sg("define external input failed");

    uint32_t cur_features = in_features;
    const uint32_t n = network.layer_count();
    for (uint32_t i = 0; i < n; ++i)
    {
        MLPLayer& layer = network.GetLayer(i);
        const bool is_last = i + 1 == n;
        char layer_err[96];

        if (!layer.weights.data || layer.weights.rank != 2)
            return fail_sg("dense weights missing/invalid");
        float out_min = -FLT_MAX;
        float out_max = FLT_MAX;
        if (!ActivationClampDense(layer.activation, out_min, out_max))
            return fail_sg("unsupported dense activation");

        const size_t out_features = layer.weights.shape[0];
        const size_t layer_in = layer.weights.shape[1];
        if (layer_in != cur_features)
        {
            std::snprintf(layer_err, sizeof(layer_err),
                          "dense in_features mismatch layer %u (want %u got %zu)",
                          i,
                          cur_features,
                          layer_in);
            return fail_sg(layer_err);
        }

        const size_t filter_dims[2] = {out_features, layer_in};
        uint32_t filter_id = 0;
        if (!DefineFp32(subgraph,
                        2,
                        filter_dims,
                        layer.weights.data,
                        XNN_INVALID_VALUE_ID,
                        0,
                        &filter_id))
            return fail_sg("dense filter failed");

        uint32_t bias_id = XNN_INVALID_VALUE_ID;
        if (layer.bias.data)
        {
            size_t bias_out = 0;
            if (layer.bias.rank == 1)
                bias_out = layer.bias.shape[0];
            else if (layer.bias.rank == 2 && layer.bias.shape[0] == 1)
                bias_out = layer.bias.shape[1];
            else
                return fail_sg("dense bias shape invalid");
            if (bias_out != out_features)
                return fail_sg("dense bias size mismatch");
            const size_t bias_dims[1] = {out_features};
            if (!DefineFp32(subgraph,
                            1,
                            bias_dims,
                            layer.bias.data,
                            XNN_INVALID_VALUE_ID,
                            0,
                            &bias_id))
                return fail_sg("dense bias failed");
        }

        uint32_t out_id = 0;
        const uint32_t out_ext =
            is_last ? runtime->xnn_net_ext_out : XNN_INVALID_VALUE_ID;
        const uint32_t out_flags = is_last ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
        const size_t out_dims[2] = {1, out_features};
        if (!DefineFp32(subgraph, 2, out_dims, nullptr, out_ext, out_flags, &out_id))
        {
            std::snprintf(layer_err, sizeof(layer_err), "dense act failed layer %u", i);
            return fail_sg(layer_err);
        }
        if (xnn_define_fully_connected(subgraph,
                                       out_min,
                                       out_max,
                                       cur_id,
                                       filter_id,
                                       bias_id,
                                       out_id,
                                       /*flags=*/0) != xnn_status_success)
        {
            std::snprintf(layer_err, sizeof(layer_err), "dense node failed layer %u", i);
            return fail_sg(layer_err);
        }
        cur_id = out_id;
        cur_features = static_cast<uint32_t>(out_features);
        if (is_last)
            runtime->out_elements = static_cast<uint32_t>(out_features);
    }

    xnn_runtime_t xnn_rt = nullptr;
    const xnn_status create_st =
        xnn_create_runtime_v4(subgraph,
                              static_cast<xnn_weights_cache_t>(runtime->xnn_weights_cache),
                              static_cast<xnn_workspace_t>(runtime->xnn_workspace),
                              /*threadpool=*/nullptr,
                              XNN_FLAG_DONT_SPIN_WORKERS,
                              &xnn_rt);
    (void)xnn_delete_subgraph(subgraph);
    subgraph = nullptr;
    if (create_st != xnn_status_success || !xnn_rt)
    {
        std::fprintf(stderr,
                     "XnnpackFloat::BuildMlpRuntime: xnn_create_runtime_v4 failed (%s)\n",
                     XnnStatusName(create_st));
        DestroyRuntime(*runtime);
        delete runtime;
        return false;
    }
    runtime->xnn_network_runtime = xnn_rt;

    if (runtime->xnn_weights_cache)
    {
        auto* cache_ptr = static_cast<xnn_weights_cache_t>(runtime->xnn_weights_cache);
        if (xnn_finalize_weights_cache(cache_ptr, xnn_weights_cache_finalization_kind_hard) !=
            xnn_status_success)
        {
            (void)xnn_finalize_weights_cache(cache_ptr,
                                             xnn_weights_cache_finalization_kind_soft);
        }
    }

    // Reshape external values then finalize runtime.
    const size_t reshape_in[2] = {1, static_cast<size_t>(in_features)};
    if (xnn_reshape_external_value(xnn_rt, runtime->xnn_net_ext_in, 2, reshape_in) !=
        xnn_status_success)
    {
        DestroyRuntime(*runtime);
        delete runtime;
        return false;
    }
    const size_t reshape_out[2] = {1, static_cast<size_t>(runtime->out_elements)};
    if (xnn_reshape_external_value(xnn_rt, runtime->xnn_net_ext_out, 2, reshape_out) !=
        xnn_status_success)
    {
        DestroyRuntime(*runtime);
        delete runtime;
        return false;
    }
    if (xnn_reshape_runtime(xnn_rt) != xnn_status_success)
    {
        DestroyRuntime(*runtime);
        delete runtime;
        return false;
    }

    runtime->xnn_network_ready = true;
    std::fprintf(stderr, "XnnpackFloat: MLP xnn_subgraph ready\n");
    out_runtime = runtime;
    return true;
}

bool InvokeNetwork(Runtime& runtime, const float* input, float* output)
{
    if (!runtime.xnn_network_ready || !runtime.xnn_network_runtime || !input || !output)
        return false;

    auto* xnn_rt = static_cast<xnn_runtime_t>(runtime.xnn_network_runtime);
    if (input != runtime.bound_input || output != runtime.bound_output)
    {
        const xnn_external_value externals[2] = {
            {runtime.xnn_net_ext_in, const_cast<float*>(input)},
            {runtime.xnn_net_ext_out, output},
        };
        if (xnn_setup_runtime_v2(xnn_rt, 2, externals) != xnn_status_success)
            return false;
        runtime.bound_input = input;
        runtime.bound_output = output;
    }
    return xnn_invoke_runtime(xnn_rt) == xnn_status_success;
}

}  // namespace XnnpackFloat

#else  // !XNNPACK

namespace XnnpackFloat
{

void DestroyRuntime(Runtime&) {}

}  // namespace XnnpackFloat

#endif
