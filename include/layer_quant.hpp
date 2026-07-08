#pragma once

#include "nk_format.hpp"

struct LayerQuant
{
    NkFormat::MlpLayerQuantDesc params{};
    bool enabled = false;
};
