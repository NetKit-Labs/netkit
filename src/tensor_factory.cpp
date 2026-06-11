#include "tensor_factory.hpp"

namespace TensorFactory
{
    void Print(const Tensor& t)
    {
        const float* p = static_cast<const float*>(t.data);

        std::cout << "Tensor shape: [";
        for (uint32_t i = 0; i < t.rank; i++)
        {
            std::cout << t.shape[i];
            if (i + 1 < t.rank) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "Values: ";
        for (uint32_t i = 0; i < t.num_elements; i++)
        {
            std::cout << p[i] << " ";
        }
        std::cout << "\n";
    }

    Tensor Create2D(Arena& arena, uint32_t rows, uint32_t cols)
    {
        Tensor t;

        t.type = DataType::Float32;
        t.rank = 2;

        t.shape[0] = rows;
        t.shape[1] = cols;

        t.stride[0] = cols;
        t.stride[1] = 1;

        t.num_elements = rows * cols;
        t.bytes = t.num_elements * sizeof(float);

        t.data = arena.alloc(t.bytes);

        return t;
    }

    Tensor CreateND(Arena& arena, uint32_t rank, const uint32_t* shape)
    {
        Tensor t;

        t.type = DataType::Float32;
        t.rank = rank;

        uint32_t num_elements = 1;

        for (uint32_t i = 0; i < rank; i++)
        {
            t.shape[i] = shape[i];
            num_elements *= shape[i];
        }

        uint32_t stride_val = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            t.stride[i] = stride_val;
            stride_val *= shape[i];
        }

        t.num_elements = num_elements;
        t.bytes = num_elements * sizeof(float);

        t.data = arena.alloc(t.bytes);

        return t;
    }

    void Fill(Tensor& t, const float* values)
    {
        float* p = static_cast<float*>(t.data);

        for (uint32_t i = 0; i < t.num_elements; i++)
        {
            p[i] = values[i];
        }
    }
}
