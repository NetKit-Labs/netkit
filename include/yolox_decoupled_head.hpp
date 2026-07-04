#pragma once

#include "tensor.hpp"
#include <cstdint>

// YOLOX single-scale decoupled detection head (anchor-free cls/reg/obj branches).
struct YoloxDecoupledHead
{
    static constexpr int kMaxStackedConvs = 4;

    int in_channels = 0;
    int hidden_dim = 256;
    int num_classes = 80;
    int num_convs = 2;

    float* stem_weights = nullptr;
    float* stem_bias = nullptr;

    float* cls_conv_weights[kMaxStackedConvs]{};
    float* cls_conv_bias[kMaxStackedConvs]{};
    float* reg_conv_weights[kMaxStackedConvs]{};
    float* reg_conv_bias[kMaxStackedConvs]{};

    float* cls_pred_weights = nullptr;
    float* cls_pred_bias = nullptr;
    float* reg_pred_weights = nullptr;
    float* reg_pred_bias = nullptr;
    float* obj_pred_weights = nullptr;
    float* obj_pred_bias = nullptr;

    float* scratch = nullptr;
    uint32_t scratch_elems = 0;

    int output_channels() const { return 4 + 1 + num_classes; }

    void forward(const Tensor& input, Tensor& output);
};
