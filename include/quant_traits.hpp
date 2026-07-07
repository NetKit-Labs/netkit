#pragma once

#include "tensor.hpp"

#include <cstdint>

// Compile-time element traits for float vs int8 quantized kernels.

template<typename Element>
struct QuantElementTraits;

template<>
struct QuantElementTraits<float>
{
    using ElementType = float;
    using AccumType = float;
    static constexpr DataType kDataType = DataType::Float32;
    static constexpr bool kQuantized = false;
};

template<>
struct QuantElementTraits<int8_t>
{
    using ElementType = int8_t;
    using AccumType = int32_t;
    static constexpr DataType kDataType = DataType::Int8;
    static constexpr bool kQuantized = true;
};

template<>
struct QuantElementTraits<int32_t>
{
    using ElementType = int32_t;
    using AccumType = int32_t;
    static constexpr DataType kDataType = DataType::Int32;
    static constexpr bool kQuantized = true;
};
