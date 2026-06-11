#pragma once
#include <cstddef>
#include <cstdint>
#include "tensor.hpp"

namespace Inference
{
    struct TopKResult
    {
        uint32_t index;
        float value;
    };

    bool BinaryClassify(const Tensor& output, float threshold = 0.5f);
    void MultiLabelClassify(const Tensor& output, float threshold, uint8_t* labels);
    uint32_t ArgMax(const Tensor& output);
    void TopK(const Tensor& A, TopKResult* results, uint32_t K);
}
