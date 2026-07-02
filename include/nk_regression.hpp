#pragma once

#include <cstdint>

namespace NkRegression
{
    struct RunSummary
    {
        uint32_t passed = 0;
        uint32_t failed = 0;
    };

    RunSummary RunModelTests(const char* nk_path);
    RunSummary RunNkOnnxParity(const char* nk_path, const char* onnx_path);
}
