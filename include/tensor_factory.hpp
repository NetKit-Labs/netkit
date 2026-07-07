#pragma once
#include "arena.hpp"
#include "tensor.hpp"
#include "netkit_log.hpp"
#include <span>

namespace TensorFactory
{
    void Print(const Tensor& t);
    // max_values == 0 prints every element; otherwise prints the first max_values plus a total count.
    void PrintLabeled(const char* label, const Tensor& t, uint32_t max_values = 0);
    Tensor Create2D(Arena& arena, uint32_t rows, uint32_t cols);
    Tensor CreateND(Arena& arena, uint32_t rank, std::span<const uint32_t> shape);
    Tensor View2D(float* data, uint32_t rows, uint32_t cols);
    Tensor View2DInt8(int8_t* data, uint32_t rows, uint32_t cols);
    Tensor View3DInt8(int8_t* data, uint32_t depth, uint32_t rows, uint32_t cols);
    Tensor View1DInt32(int32_t* data, uint32_t length);
    Tensor ViewND(float* data, uint32_t rank, std::span<const uint32_t> shape);
    void Fill(Tensor& t, std::span<const float> values);
}
