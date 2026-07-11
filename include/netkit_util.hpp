#pragma once

#include <cstdint>

#include "netkit_loop_unroll.hpp"

namespace NetkitUtil
{
// Portable vector helpers (memcpy / loops / NetkitLoopUnroll). CMSIS-DSP is not used.
void CopyInt8(const int8_t* src, int8_t* dst, uint32_t count);
void CopyF32(const float* src, float* dst, uint32_t count);

uint32_t ArgMaxInt8(const int8_t* values, uint32_t count);
uint32_t ArgMaxF32(const float* values, uint32_t count);

// Contiguous dot product — header-inline so hot FC/CNN loops inline at -O2.
// See docs/KERNELS.md "Hot dot product is header-inline".
inline float DotProductF32(const float* a, const float* b, uint32_t count)
{
    return NetkitLoopUnroll::dot_contiguous(a, b, count);
}

void MulF32(const float* a, const float* b, float* c, uint32_t count);
void AddF32(const float* a, const float* b, float* c, uint32_t count);
void MulScalarF32(const float* a, float scalar, float* c, uint32_t count);
void ScaleF32(float* c, float scalar, uint32_t count);
}  // namespace NetkitUtil

// Back-compat aliases for quant / board call sites.
namespace CmsisQuantUtil
{
using NetkitUtil::ArgMaxInt8;
using NetkitUtil::CopyInt8;
}  // namespace CmsisQuantUtil
