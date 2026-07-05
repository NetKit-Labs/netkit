#include "mlp.hpp"
#include "active_kernel.hpp"
#include "activation_followup.hpp"
#include "tensor_factory.hpp"

#include <chrono>
#include <cstdint>

namespace
{
    void ForwardLayer(MLPLayer& layer, const Tensor& input, Tensor& output)
    {
        const NetkitKernelActivation kernel_activation = ToKernelActivation(layer.activation);
        const bool fused_in_kernel = Kernels::FullyConnectedWithBias(
            input, layer.weights, layer.bias, kernel_activation, output);
        ApplyFusedOutputActivation(kernel_activation, fused_in_kernel, output, layer.leaky_alpha);
    }

    void ForwardLayerTimed(MLPLayer& layer,
                           const Tensor& input,
                           Tensor& output,
                           MLPNetwork::LayerTimingFn timing_fn,
                           void* user_data)
    {
        const auto layer_start = std::chrono::steady_clock::now();
        ForwardLayer(layer, input, output);
        const auto layer_end = std::chrono::steady_clock::now();

        if (timing_fn)
        {
            const uint64_t duration_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(layer_end - layer_start)
                    .count());
            timing_fn("FullyConnected", duration_ns, user_data);
        }
    }
}

void MLPLayer::forward(const Tensor& input, Tensor& output)
{
    ForwardLayer(*this, input, output);
}

MLPNetwork::MLPNetwork(uint32_t num_layers, Arena& arena)
    : layers(nullptr), num_layers(num_layers)
{
    layers = static_cast<MLPLayer*>(arena.alloc(sizeof(MLPLayer) * num_layers, alignof(MLPLayer)));
}

bool MLPNetwork::InitActivationBuffers(Arena& arena, uint32_t batch_rows)
{
    ping_a = nullptr;
    ping_b = nullptr;
    max_activation_elements = 0;
    hidden_activation_ = {};
    ping_view_a_ = {};
    ping_view_b_ = {};

    if (num_layers <= 1 || !layers)
        return layers != nullptr;

    uint32_t max_hidden_cols = 0;
    for (uint32_t i = 0; i < num_layers - 1; ++i)
    {
        const uint32_t cols = layers[i].weights.shape[0];
        const uint32_t elements = batch_rows * cols;
        if (elements > max_activation_elements)
            max_activation_elements = elements;
        if (cols > max_hidden_cols)
            max_hidden_cols = cols;
    }

    if (max_activation_elements == 0 || batch_rows == 0)
        return false;

    const std::size_t bytes = static_cast<std::size_t>(max_activation_elements) * sizeof(float);
    ping_a = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    ping_b = static_cast<float*>(arena.alloc(bytes, alignof(float)));
    if (!ping_a || !ping_b)
        return false;

    ping_view_a_ = TensorFactory::View2D(ping_a, batch_rows, max_hidden_cols);
    ping_view_b_ = TensorFactory::View2D(ping_b, batch_rows, max_hidden_cols);
    hidden_activation_ = ping_view_a_;
    return true;
}

void MLPNetwork::InitLayer(uint32_t layer_idx,
                           const Tensor& weights,
                           const Tensor& bias,
                           ActivationType activation,
                           float leaky_alpha)
{
    if (!layers || layer_idx >= num_layers)
        return;

    layers[layer_idx].weights = weights;
    layers[layer_idx].bias = bias;
    layers[layer_idx].activation = activation;
    layers[layer_idx].leaky_alpha = leaky_alpha;
}

void MLPNetwork::forward(const Tensor& input, Tensor& output, Arena& /*arena*/)
{
    if (!IsValid() || !HasActivationBuffers() || num_layers == 0)
        return;

    if (num_layers == 1)
    {
        ForwardLayer(layers[0], input, output);
        return;
    }

    if (num_layers == 2)
    {
        ForwardLayer(layers[0], input, hidden_activation_);
        ForwardLayer(layers[1], hidden_activation_, output);
        return;
    }

    const Tensor* current_input = &input;
    Tensor* write_view = &ping_view_a_;

    for (uint32_t i = 0; i < num_layers; ++i)
    {
        if (i == num_layers - 1)
        {
            ForwardLayer(layers[i], *current_input, output);
            return;
        }

        const uint32_t rows = current_input->shape[0];
        const uint32_t cols = layers[i].weights.shape[0];
        if (rows * cols > max_activation_elements)
            return;

        *write_view = TensorFactory::View2D(write_view == &ping_view_a_ ? ping_a : ping_b, rows, cols);
        ForwardLayer(layers[i], *current_input, *write_view);
        current_input = write_view;
        write_view = (write_view == &ping_view_a_) ? &ping_view_b_ : &ping_view_a_;
    }
}

void MLPNetwork::forward_timed(const Tensor& input,
                               Tensor& output,
                               LayerTimingFn timing_fn,
                               void* user_data)
{
    if (!IsValid() || !HasActivationBuffers() || num_layers == 0)
        return;

    if (num_layers == 1)
    {
        ForwardLayerTimed(layers[0], input, output, timing_fn, user_data);
        return;
    }

    if (num_layers == 2)
    {
        ForwardLayerTimed(layers[0], input, hidden_activation_, timing_fn, user_data);
        ForwardLayerTimed(layers[1], hidden_activation_, output, timing_fn, user_data);
        return;
    }

    const Tensor* current_input = &input;
    Tensor* write_view = &ping_view_a_;

    for (uint32_t i = 0; i < num_layers; ++i)
    {
        if (i == num_layers - 1)
        {
            ForwardLayerTimed(layers[i], *current_input, output, timing_fn, user_data);
            return;
        }

        const uint32_t rows = current_input->shape[0];
        const uint32_t cols = layers[i].weights.shape[0];
        if (rows * cols > max_activation_elements)
            return;

        *write_view = TensorFactory::View2D(write_view == &ping_view_a_ ? ping_a : ping_b, rows, cols);
        ForwardLayerTimed(layers[i], *current_input, *write_view, timing_fn, user_data);
        current_input = write_view;
        write_view = (write_view == &ping_view_a_) ? &ping_view_b_ : &ping_view_a_;
    }
}
