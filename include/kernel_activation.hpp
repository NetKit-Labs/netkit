#pragma once

enum class NetkitKernelActivation
{
    None = 0,
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ReLU6,
    Softmax,
};

constexpr bool kernel_activation_is_fused(NetkitKernelActivation activation)
{
    return activation == NetkitKernelActivation::ReLU || activation == NetkitKernelActivation::ReLU6;
}

inline float ApplyKernelActivation(float value, NetkitKernelActivation activation)
{
    switch (activation)
    {
        case NetkitKernelActivation::ReLU:
            return value > 0.0f ? value : 0.0f;
        case NetkitKernelActivation::ReLU6:
            if (value < 0.0f)
                return 0.0f;
            if (value > 6.0f)
                return 6.0f;
            return value;
        default:
            return value;
    }
}
