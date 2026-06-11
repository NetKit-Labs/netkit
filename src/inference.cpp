#include "inference.hpp"

namespace Inference
{
    bool BinaryClassify(const Tensor& output, float threshold)
    {
        const float* p = static_cast<const float*>(output.data);
        return p[0] > threshold;
    }

    void MultiLabelClassify(const Tensor& output, float threshold, uint8_t* labels)
    {
        const float* p = static_cast<const float*>(output.data);

        for (uint32_t i = 0; i < output.num_elements; i++)
        {
            labels[i] = (p[i] > threshold) ? 1 : 0;
        }
    }

    uint32_t ArgMax(const Tensor& output)
    {
        const float* p = static_cast<const float*>(output.data);

        uint32_t idx = 0;
        float max_val = p[0];

        for (uint32_t i = 1; i < output.num_elements; i++)
        {
            if (p[i] > max_val)
            {
                max_val = p[i];
                idx = i;
            }
        }

        return idx;
    }

    void TopK(const Tensor& A, TopKResult* results, uint32_t K)
    {
        const float* a = static_cast<const float*>(A.data);
        const uint32_t n = A.num_elements;

        // initialize results with very small values
        for (uint32_t k = 0; k < K; k++)
        {
            results[k].index = 0;
            results[k].value = -3.4028235e38f; // ~ -infinity float32
        }

        // scan all elements
        for (uint32_t i = 0; i < n; i++)
        {
            float val = a[i];

            // try to insert into top-K list
            for (uint32_t k = 0; k < K; k++)
            {
                if (val > results[k].value)
                {
                    // shift down to make space
                    for (uint32_t j = K - 1; j > k; j--)
                    {
                        results[j] = results[j - 1];
                    }

                    results[k].value = val;
                    results[k].index = i;
                    break;
                }
            }
        }
    }
}
