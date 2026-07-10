#pragma once

#include <cstdint>

#include "netkit_loop_unroll.hpp"  // NetkitLoopUnroll::dot_contiguous + NETKIT_USE_CMSIS_DSP via config

namespace CmsisDspUtil
{
// Vector copy (arm_copy_q7 / arm_copy_f32 when NETKIT_USE_CMSIS_DSP).
void CopyInt8(const int8_t* src, int8_t* dst, uint32_t count);
void CopyF32(const float* src, float* dst, uint32_t count);

// Index of maximum element.
uint32_t ArgMaxInt8(const int8_t* values, uint32_t count);
uint32_t ArgMaxF32(const float* values, uint32_t count);

#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
// arm_dot_prod_f32 shim, defined in cmsis_dsp_util.cpp so <arm_math.h> stays out of this
// widely-included header.
float DotProductF32Cmsis(const float* a, const float* b, uint32_t count);
#endif

// Contiguous dot product. Header-inline so the hot FC/CNN reduction loops inline into their
// kernels at -O2 (no cross-translation-unit call, no LTO needed).
//
// Policy: CMSIS-DSP is VectorFast helpers only by default. Hot dots stay on the inlined
// 4-accumulator path even when NETKIT_USE_CMSIS_DSP=1 (portable host CMSIS dots were a large
// regression vs reference). Opt into arm_dot_prod_f32 with -DNETKIT_CMSIS_DSP_DOT=1.
//
// Code-size note: the 4-accumulator body is inlined into every caller (conv_direct/conv1x1/
// im2col_partial/dense FC), costing ~+6 KB in conv_direct alone at -O2 (Cortex-M4). See
// docs/KERNELS.md "Hot dot product is header-inline".
inline float DotProductF32(const float* a, const float* b, uint32_t count)
{
#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP && defined(NETKIT_CMSIS_DSP_DOT) &&       \
    NETKIT_CMSIS_DSP_DOT
    return DotProductF32Cmsis(a, b, count);
#else
    return NetkitLoopUnroll::dot_contiguous(a, b, count);
#endif
}

void MulF32(const float* a, const float* b, float* c, uint32_t count);
void AddF32(const float* a, const float* b, float* c, uint32_t count);
void MulScalarF32(const float* a, float scalar, float* c, uint32_t count);
void ScaleF32(float* c, float scalar, uint32_t count);
}  // namespace CmsisDspUtil

// Back-compat aliases for quant call sites.
namespace CmsisQuantUtil
{
using CmsisDspUtil::ArgMaxInt8;
using CmsisDspUtil::CopyInt8;
}  // namespace CmsisQuantUtil
