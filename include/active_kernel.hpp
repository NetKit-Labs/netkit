#pragma once

#include "kernel_dispatch.hpp"
#include "netkit_config.h"
#include "reference_kernel.hpp"

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
#include "cmsis_nn_kernel.hpp"
#endif

#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
#include "xnnpack_kernel.hpp"
#endif

// Backend composition:
//   MCU (mcu_arm): CMSIS-NN LayerFast over reference VectorFast (int8 production path)
//   CPU/MPU:       XNNPACK LayerFast when enabled, else reference
//   No CMSIS-DSP — float32 MCU is reference-only (rare; production is int8)
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
using Kernels = detail::ComposedKernel<ReferenceKernel, CmsisNnKernel>;
#elif defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
using Kernels = detail::ComposedKernel<ReferenceKernel, XnnpackKernel>;
#else
using Kernels = ReferenceKernel;
#endif
