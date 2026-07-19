/*
 * ESP-NN float LayerFast adapters.
 *
 * ESP-NN provides int8 kernels only. This TU is linked when NETKIT_ESP_NN is
 * effective so EspNnKernel symbols exist; every Try* returns false and
 * ComposedKernel falls back to ReferenceKernel (same pattern as an accel miss).
 */
#include "esp_nn_kernel.hpp"
#include "netkit_config.h"

#if defined(NETKIT_USE_ESP_NN) && NETKIT_USE_ESP_NN && NETKIT_ESP_NN_ALLOWED

bool EspNnKernel::TryConv2dForward(const Tensor&,
                                   float*,
                                   float*,
                                   int,
                                   int,
                                   int,
                                   int,
                                   int,
                                   int,
                                   NetkitKernelActivation,
                                   Tensor&)
{
    return false;
}

bool EspNnKernel::TryDepthwiseConv2dForward(const Tensor&,
                                            float*,
                                            float*,
                                            int,
                                            int,
                                            int,
                                            int,
                                            int,
                                            int,
                                            NetkitKernelActivation,
                                            Tensor&)
{
    return false;
}

bool EspNnKernel::TryMaxPool2dForward(const Tensor&,
                                      int,
                                      int,
                                      int,
                                      int,
                                      NetkitKernelActivation,
                                      Tensor&)
{
    return false;
}

bool EspNnKernel::TryAvgPool2dForward(const Tensor&, int, int, int, int, Tensor&)
{
    return false;
}

bool EspNnKernel::TryBatchNorm2dForward(const Tensor&, const float*, const float*, int, Tensor&)
{
    return false;
}

bool EspNnKernel::TryFullyConnectedWithBias(const Tensor&,
                                            const Tensor&,
                                            const Tensor&,
                                            NetkitKernelActivation,
                                            Tensor&)
{
    return false;
}

bool EspNnKernel::TryActivationForward(const Tensor&,
                                       Tensor&,
                                       NetkitKernelActivation,
                                       float)
{
    return false;
}

bool EspNnKernel::TrySoftmaxForward(const Tensor&, Tensor&)
{
    return false;
}

bool EspNnKernel::TryGeluForward(const Tensor&, Tensor&)
{
    return false;
}

bool EspNnKernel::TryMatAdd(const Tensor&, const Tensor&, Tensor&)
{
    return false;
}

#endif /* NETKIT_USE_ESP_NN */
