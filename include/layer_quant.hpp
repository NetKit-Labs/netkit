#pragma once

#include "nk_format.hpp"

struct LayerQuant
{
    NkFormat::LayerQuantDesc params{};
    bool enabled = false;
};
