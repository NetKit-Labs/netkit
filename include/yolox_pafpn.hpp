#pragma once

#include "tensor.hpp"
#include "yolox_decoupled_head.hpp"
#include <cstdint>

// YOLOX Nano-style PAFPN (depthwise, add-based) + three decoupled heads.
// Reads tap C3/C4 side buffers + layer input C5; emits flat concat [P3|P4|P5].
struct YoloxPafpnMultiscale
{
    static constexpr int kNumScales = 3;

    int c3_channels = 0;
    int c4_channels = 0;
    int c5_channels = 0;
    int hidden_dim = 64;
    int num_classes = 80;
    int num_convs = 2;

    // Lateral 1x1 (C3/C4/C5 -> H)
    float* lat3_weights = nullptr;
    float* lat3_bias = nullptr;
    float* lat4_weights = nullptr;
    float* lat4_bias = nullptr;
    float* lat5_weights = nullptr;
    float* lat5_bias = nullptr;

    // Top-down refine: DW 3x3 s1 + PW 1x1 (P4, then P3)
    float* td_p4_dw_weights = nullptr;
    float* td_p4_dw_bias = nullptr;
    float* td_p4_pw_weights = nullptr;
    float* td_p4_pw_bias = nullptr;
    float* td_p3_dw_weights = nullptr;
    float* td_p3_dw_bias = nullptr;
    float* td_p3_pw_weights = nullptr;
    float* td_p3_pw_bias = nullptr;

    // Bottom-up: DW 3x3 s2 + PW 1x1 (N4, then N5)
    float* bu_n4_dw_weights = nullptr;
    float* bu_n4_dw_bias = nullptr;
    float* bu_n4_pw_weights = nullptr;
    float* bu_n4_pw_bias = nullptr;
    float* bu_n5_dw_weights = nullptr;
    float* bu_n5_dw_bias = nullptr;
    float* bu_n5_pw_weights = nullptr;
    float* bu_n5_pw_bias = nullptr;

    YoloxDecoupledHead heads[kNumScales]{};

    // Side buffers: tap[0]=C3, tap[1]=C4 (owned by CNNNetwork)
    float* c3_data = nullptr;
    uint32_t c3_h = 0;
    uint32_t c3_w = 0;
    float* c4_data = nullptr;
    uint32_t c4_h = 0;
    uint32_t c4_w = 0;

    float* scratch = nullptr;
    uint32_t scratch_elems = 0;

    int output_channels() const { return 4 + 1 + num_classes; }

    uint32_t output_elements(uint32_t c5_h, uint32_t c5_w) const
    {
        const uint32_t h3 = c5_h * 4u;
        const uint32_t w3 = c5_w * 4u;
        const uint32_t h4 = c5_h * 2u;
        const uint32_t w4 = c5_w * 2u;
        const uint32_t out_c = static_cast<uint32_t>(output_channels());
        return (h3 * w3 + h4 * w4 + c5_h * c5_w) * out_c;
    }

    void forward(const Tensor& c5_input, Tensor& output);
};
