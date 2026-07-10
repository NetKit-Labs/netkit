#pragma once

#include "netkit_config.h"

#include <cstdint>

// Shared float/int8 Conv2D im2col policy (NETKIT_IM2COL 0/1/2).
// Depthwise and 1x1 stride-1 stay on direct loops.

enum class Conv2dExecMode : uint8_t
{
    Direct = 0,
    PartialIm2Col = 1,
    FullIm2Col = 2,
};

inline bool ConvIm2ColVolumeWarrants(uint32_t kernel_h,
                                     uint32_t kernel_w,
                                     uint32_t in_channels,
                                     uint32_t out_h,
                                     uint32_t out_w)
{
    constexpr uint32_t kIm2ColMinPatchVolume = 2048u;
    const uint32_t patch = kernel_h * kernel_w * in_channels;
    const uint32_t spatial = out_h * out_w;
    return patch > 0 && spatial > 0 && patch * spatial >= kIm2ColMinPatchVolume;
}

inline bool ConvIm2ColFullVolumeWarrants(uint32_t kernel_h,
                                         uint32_t kernel_w,
                                         uint32_t in_channels,
                                         uint32_t out_h,
                                         uint32_t out_w)
{
    constexpr uint32_t kIm2ColFullMinPatchVolume = 32768u;
    const uint32_t patch = kernel_h * kernel_w * in_channels;
    const uint32_t spatial = out_h * out_w;
    return patch > 0 && spatial > 0 && patch * spatial >= kIm2ColFullMinPatchVolume;
}

inline Conv2dExecMode SelectConv2dExecMode(int kernel_h,
                                           int kernel_w,
                                           int stride,
                                           uint32_t in_channels,
                                           uint32_t out_h,
                                           uint32_t out_w)
{
    if (kernel_h == 1 && kernel_w == 1 && stride == 1)
        return Conv2dExecMode::Direct;

#if NETKIT_IM2COL >= 1
    const uint32_t kh = static_cast<uint32_t>(kernel_h);
    const uint32_t kw = static_cast<uint32_t>(kernel_w);
    const bool large_volume = ConvIm2ColVolumeWarrants(kh, kw, in_channels, out_h, out_w);
#if NETKIT_IM2COL >= 2
    const bool full_gemm_volume =
        ConvIm2ColFullVolumeWarrants(kh, kw, in_channels, out_h, out_w);
#endif

    if (kernel_h == 3 && kernel_w == 3 && stride == 1)
    {
#if NETKIT_IM2COL >= 2
        if (full_gemm_volume)
            return Conv2dExecMode::FullIm2Col;
#endif
        if (large_volume)
            return Conv2dExecMode::PartialIm2Col;
        return Conv2dExecMode::Direct;
    }

    if (kernel_h >= 5 || kernel_w >= 5)
    {
#if NETKIT_IM2COL >= 2
        if (large_volume)
            return Conv2dExecMode::FullIm2Col;
#endif
        if (large_volume)
            return Conv2dExecMode::PartialIm2Col;
        return Conv2dExecMode::Direct;
    }

#if NETKIT_IM2COL >= 2
    if (large_volume)
        return Conv2dExecMode::FullIm2Col;
#endif
    if (large_volume)
        return Conv2dExecMode::PartialIm2Col;
    return Conv2dExecMode::Direct;
#else
    (void)kernel_h;
    (void)kernel_w;
    (void)stride;
    (void)in_channels;
    (void)out_h;
    (void)out_w;
    return Conv2dExecMode::Direct;
#endif
}
