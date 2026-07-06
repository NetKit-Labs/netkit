#pragma once

#include <cstddef>
#include <cstdint>

struct Arena;
struct Conv2D;

// Repack [out, kh, kw, in] (OIHW) -> [kh, kw, in, out] (HWIO) for input-stationary conv.
void RepackConv2dOkiToHwio(const float* oki,
                           float* hwio,
                           uint32_t kernel_h,
                           uint32_t kernel_w,
                           uint32_t in_channels,
                           uint32_t out_channels);

bool RepackConv2dWeights(Conv2D& conv, Arena& arena);
