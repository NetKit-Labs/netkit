#include "netkit_util.hpp"

#include "netkit_loop_unroll.hpp"

#include <cstring>

namespace NetkitUtil
{
void CopyInt8(const int8_t* src, int8_t* dst, uint32_t count)
{
    if (!src || !dst || count == 0)
        return;
    std::memcpy(dst, src, static_cast<std::size_t>(count));
}

void CopyF32(const float* src, float* dst, uint32_t count)
{
    if (!src || !dst || count == 0)
        return;
    std::memcpy(dst, src, static_cast<std::size_t>(count) * sizeof(float));
}

uint32_t ArgMaxInt8(const int8_t* values, uint32_t count)
{
    if (!values || count == 0)
        return 0;
    if (count == 1)
        return 0;

    uint32_t best = 0;
    for (uint32_t i = 1; i < count; ++i)
    {
        if (values[i] > values[best])
            best = i;
    }
    return best;
}

uint32_t ArgMaxF32(const float* values, uint32_t count)
{
    if (!values || count == 0)
        return 0;
    if (count == 1)
        return 0;

    uint32_t best = 0;
    for (uint32_t i = 1; i < count; ++i)
    {
        if (values[i] > values[best])
            best = i;
    }
    return best;
}

void MulF32(const float* a, const float* b, float* c, uint32_t count)
{
    NetkitLoopUnroll::mul_contiguous(a, b, c, count);
}

void AddF32(const float* a, const float* b, float* c, uint32_t count)
{
    NetkitLoopUnroll::add_contiguous(a, b, c, count);
}

void MulScalarF32(const float* a, float scalar, float* c, uint32_t count)
{
    NetkitLoopUnroll::mul_scalar_contiguous(a, scalar, c, count);
}

void ScaleF32(float* c, float scalar, uint32_t count)
{
    NetkitLoopUnroll::scale_contiguous(c, scalar, count);
}
}  // namespace NetkitUtil
