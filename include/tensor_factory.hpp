#pragma once
#include "arena.hpp"
#include "tensor.hpp"
#include <iostream>

namespace TensorFactory
{
    void Print(const Tensor& t);
    Tensor Create2D(Arena& arena, uint32_t rows, uint32_t cols);
    Tensor CreateND(Arena& arena, uint32_t rank, const uint32_t* shape);
    void Fill(Tensor& t, const float* values);
}
