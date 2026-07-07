#include "cnn.hpp"
#include "cmsis_quant_plan.hpp"
#include "netkit_config.h"
#include "quant_trace.hpp"
#include "tensor_factory.hpp"

#include <algorithm>

bool CNNNetwork::InitQuantizedActivationBuffers(Arena& arena, uint32_t in_h, uint32_t in_w, uint32_t in_c)
{
    ping_a = nullptr;
    ping_b = nullptr;
    ping_i8_a = nullptr;
    ping_i8_b = nullptr;
    kernel_workspace_ = nullptr;
    kernel_workspace_bytes_ = 0;
    max_activation_elements = 0;
    layer_output_views_ = nullptr;
    output_cache_ = {};
    float_output_buffer_ = nullptr;
    float_output_elements_ = 0;
    quant_runtime_ = nullptr;

    if (!blocks || num_layers == 0)
        return blocks != nullptr;

    if (!CmsisQuantPlan::BuildRuntime(*this, arena, in_h, in_w, in_c))
        return false;

    const CmsisQuantPlan::Runtime* runtime = quant_runtime_;
    if (!runtime)
        return false;

    ping_i8_a = runtime->act_a;
    ping_i8_b = runtime->act_b;
    kernel_workspace_ = runtime->workspace;
    kernel_workspace_bytes_ = runtime->workspace_bytes;
    max_activation_elements = std::max(runtime->act_a_bytes, runtime->act_b_bytes);
    return true;
}

void CNNNetwork::InitQuantizedConvLayer(uint32_t layer_idx,
                                        int kernel_size,
                                        int stride,
                                        int in_channels,
                                        int out_channels,
                                        int8_t* weights,
                                        int32_t* bias,
                                        const NkFormat::LayerQuantDesc& quant,
                                        ConvActivationType activation,
                                        float leaky_alpha,
                                        int pad_h,
                                        int pad_w,
                                        int pad_h_end,
                                        int pad_w_end)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::Conv2D;
    Conv2D& conv = blocks[layer_idx].conv.conv;
    conv.kernel_size = kernel_size;
    conv.stride = stride;
    conv.pad_h = pad_h;
    conv.pad_w = pad_w;
    conv.pad_h_end = pad_h_end >= 0 ? pad_h_end : pad_h;
    conv.pad_w_end = pad_w_end >= 0 ? pad_w_end : pad_w;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.weights = nullptr;
    conv.weights_hwio = nullptr;
    conv.bias = nullptr;
    conv.weights_q = weights;
    conv.bias_q = bias;
    blocks[layer_idx].conv.activation = activation;
    blocks[layer_idx].conv.leaky_alpha = leaky_alpha;
    blocks[layer_idx].conv.quant.params = quant;
    blocks[layer_idx].conv.quant.enabled = true;
}

void CNNNetwork::InitQuantizedDenseLayer(uint32_t layer_idx,
                                         const Tensor& weights,
                                         const Tensor& bias,
                                         const NkFormat::LayerQuantDesc& quant,
                                         ActivationType activation,
                                         float leaky_alpha)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::Dense;
    blocks[layer_idx].dense.weights = weights;
    blocks[layer_idx].dense.bias = bias;
    blocks[layer_idx].dense.activation = activation;
    blocks[layer_idx].dense.leaky_alpha = leaky_alpha;
    blocks[layer_idx].dense.quant.params = quant;
    blocks[layer_idx].dense.quant.enabled = true;
}

Tensor& CNNNetwork::forward_quantized(const Tensor& input)
{
    static Tensor empty{};

    if (!IsValid() || !HasActivationBuffers() || num_layers == 0 || !quant_runtime_)
        return empty;

    if (!CmsisQuantPlan::Forward(*quant_runtime_,
                                 *this,
                                 input,
                                 quant_output_format_,
                                 output_cache_))
    {
        return empty;
    }

    return output_cache_;
}
