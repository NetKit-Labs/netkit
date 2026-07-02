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

    // CPU heap regression: one malloc for the full suite; call EndRegressionArena() when done.
    void BeginRegressionArena();
    void EndRegressionArena();
}
