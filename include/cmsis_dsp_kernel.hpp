#pragma once

#include "kernel_activation.hpp"
#include "tensor.hpp"

struct CmsisDspKernel
{
    static bool TryMatAdd(const Tensor& a, const Tensor& b, Tensor& c);
    static bool TryMul(const Tensor& a, const Tensor& b, Tensor& c);
    static bool TryMulScalar(const Tensor& a, float scalar, Tensor& c);
    static bool TryMatMul(const Tensor& a, const Tensor& b, Tensor& c);
    static bool TryClip(const Tensor& input, Tensor& output, float low, float high);
    static bool TryFullyConnectedWithBias(const Tensor& input,
                                          const Tensor& weights,
                                          const Tensor& bias,
                                          Tensor& output);
    static bool TryBatchNorm2dForward(const Tensor& input,
                                      const float* scale,
                                      const float* bias,
                                      int channels,
                                      Tensor& output);
    static bool TryLayerNorm2dForward(const Tensor& input,
                                      const float* weight,
                                      const float* bias,
                                      int channels,
                                      float eps,
                                      Tensor& output);
    static bool TryGeluForward(const Tensor& input, Tensor& output);
    static bool TryGrn2dForward(const Tensor& input,
                                const float* gamma,
                                const float* beta,
                                int channels,
                                float eps,
                                float* channel_norm_scratch,
                                Tensor& output);
};
