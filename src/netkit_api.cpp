#include "netkit.h"
#include "model_loader.hpp"
#include "tensor_factory.hpp"
#include "mlp.hpp"
#include "cnn.hpp"
#include <cstdio>
#include <cstring>
#include <new>

namespace
{
    thread_local char g_last_error[NK_MAX_MESSAGE_LEN] = {};

    void SetLastError(const char* message)
    {
        if (!message)
        {
            g_last_error[0] = '\0';
            return;
        }

        std::strncpy(g_last_error, message, sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
    }

    nk_status_t FromLoadStatus(ModelLoader::LoadStatus status)
    {
        switch (status)
        {
            case ModelLoader::LoadStatus::Ok: return NK_OK;
            case ModelLoader::LoadStatus::JsonOpenFailed: return NK_ERR_JSON_OPEN;
            case ModelLoader::LoadStatus::BinOpenFailed: return NK_ERR_BIN_OPEN;
            case ModelLoader::LoadStatus::JsonParseFailed: return NK_ERR_JSON_PARSE;
            case ModelLoader::LoadStatus::UnsupportedNetwork: return NK_ERR_UNSUPPORTED_NETWORK;
            case ModelLoader::LoadStatus::VersionMismatch: return NK_ERR_VERSION_MISMATCH;
            case ModelLoader::LoadStatus::LayerConfigError: return NK_ERR_LAYER_CONFIG;
            case ModelLoader::LoadStatus::BinSizeMismatch: return NK_ERR_BIN_SIZE_MISMATCH;
            case ModelLoader::LoadStatus::ArenaOverflow: return NK_ERR_ARENA_OVERFLOW;
        }

        return NK_ERR_INVALID_ARGUMENT;
    }

    nk_network_kind_t FromNetworkKind(ModelLoader::NetworkKind kind)
    {
        switch (kind)
        {
            case ModelLoader::NetworkKind::MLP: return NK_NETWORK_MLP;
            case ModelLoader::NetworkKind::CNN: return NK_NETWORK_CNN;
            default: return NK_NETWORK_UNKNOWN;
        }
    }

    struct ModelState
    {
        nk_network_kind_t kind = NK_NETWORK_UNKNOWN;
        MLPNetwork* mlp = nullptr;
        CNNNetwork* cnn = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;
        uint32_t input_elements = 0;
        uint32_t output_elements = 0;
        bool loaded = false;
    };

    static_assert(sizeof(ModelState) <= NK_MODEL_STORAGE_BYTES);
    static_assert(sizeof(Arena) <= NK_ARENA_STORAGE_BYTES);

    Arena* ArenaPtr(nk_arena_t* arena)
    {
        return reinterpret_cast<Arena*>(arena->storage);
    }

    const Arena* ArenaPtr(const nk_arena_t* arena)
    {
        return reinterpret_cast<const Arena*>(arena->storage);
    }

    ModelState* ModelPtr(nk_model_t* model)
    {
        return reinterpret_cast<ModelState*>(model->storage);
    }

    const ModelState* ModelPtr(const nk_model_t* model)
    {
        return reinterpret_cast<const ModelState*>(model->storage);
    }

    bool FileReadable(const char* path)
    {
        std::FILE* file = std::fopen(path, "rb");
        if (!file)
            return false;
        std::fclose(file);
        return true;
    }

    const char* ResolveModelPath(const char* rel_path, char* buffer, std::size_t buffer_size)
    {
        if (FileReadable(rel_path))
            return rel_path;

        std::snprintf(buffer, buffer_size, "../%s", rel_path);
        if (FileReadable(buffer))
            return buffer;

        return rel_path;
    }

    uint32_t InputElementCount(const ModelLoader::ArchitectureSpec& spec)
    {
        uint32_t count = 1;
        for (uint32_t i = 0; i < spec.input_rank; ++i)
            count *= spec.input_shape[i];
        return count;
    }

    uint32_t MlpOutputElements(const ModelLoader::ArchitectureSpec& spec)
    {
        if (spec.num_layers == 0)
            return 0;
        return spec.input_shape[0] * spec.dense_layers[spec.num_layers - 1].units;
    }

    uint32_t CnnOutputElements(const ModelLoader::ArchitectureSpec& spec)
    {
        uint32_t h = spec.input_shape[0];
        uint32_t w = spec.input_shape[1];
        uint32_t c = spec.input_shape[2];

        for (uint32_t i = 0; i < spec.num_layers; ++i)
        {
            const ModelLoader::ConvLayerConfig& layer = spec.conv_layers[i];
            h = (h - layer.kernel_size) / layer.stride + 1;
            w = (w - layer.kernel_size) / layer.stride + 1;
            c = layer.filters;
        }

        return h * w * c;
    }

    void FillArchInfo(const ModelLoader::ArchitectureSpec& spec, nk_arch_info_t* info)
    {
        info->version = spec.version;
        info->kind = FromNetworkKind(spec.kind);
        info->input_rank = spec.input_rank;
        info->num_layers = spec.num_layers;
        info->expected_weight_floats = spec.expected_weight_floats;
        info->input_elements = InputElementCount(spec);

        for (uint32_t i = 0; i < NK_MAX_TENSOR_RANK; ++i)
            info->input_shape[i] = i < spec.input_rank ? spec.input_shape[i] : 0;

        if (spec.kind == ModelLoader::NetworkKind::MLP)
            info->output_elements = MlpOutputElements(spec);
        else if (spec.kind == ModelLoader::NetworkKind::CNN)
            info->output_elements = CnnOutputElements(spec);
        else
            info->output_elements = 0;
    }

    Tensor MakeNhwcInput(float* data, uint32_t h, uint32_t w, uint32_t c)
    {
        Tensor input{};
        input.data = data;
        input.type = DataType::Float32;
        input.rank = 3;
        input.shape[0] = h;
        input.shape[1] = w;
        input.shape[2] = c;
        input.stride[0] = w * c;
        input.stride[1] = c;
        input.stride[2] = 1;
        input.num_elements = h * w * c;
        input.bytes = input.num_elements * sizeof(float);
        return input;
    }
}

extern "C" {

const char* nk_version_string(void)
{
    return "0.1.0";
}

const char* nk_status_string(nk_status_t status)
{
    switch (status)
    {
        case NK_OK: return "ok";
        case NK_ERR_JSON_OPEN: return "json open failed";
        case NK_ERR_BIN_OPEN: return "bin open failed";
        case NK_ERR_JSON_PARSE: return "json parse failed";
        case NK_ERR_UNSUPPORTED_NETWORK: return "unsupported network";
        case NK_ERR_VERSION_MISMATCH: return "version mismatch";
        case NK_ERR_LAYER_CONFIG: return "layer config error";
        case NK_ERR_BIN_SIZE_MISMATCH: return "bin size mismatch";
        case NK_ERR_ARENA_OVERFLOW: return "arena overflow";
        case NK_ERR_INVALID_ARGUMENT: return "invalid argument";
        case NK_ERR_BUFFER_TOO_SMALL: return "buffer too small";
        case NK_ERR_MODEL_NOT_LOADED: return "model not loaded";
    }

    return "unknown";
}

const char* nk_last_error(void)
{
    return g_last_error;
}

void nk_arena_init(nk_arena_t* arena, void* memory, size_t size)
{
    if (!arena)
        return;

    std::memset(arena->storage, 0, sizeof(arena->storage));
    ArenaPtr(arena)->init(memory, size);
}

void nk_arena_reset(nk_arena_t* arena)
{
    if (!arena)
        return;
    ArenaPtr(arena)->reset();
}

size_t nk_arena_capacity(const nk_arena_t* arena)
{
    if (!arena)
        return 0;
    return ArenaPtr(arena)->capacity;
}

size_t nk_arena_used(const nk_arena_t* arena)
{
    if (!arena)
        return 0;
    return ArenaPtr(arena)->offset;
}

size_t nk_arena_remaining(const nk_arena_t* arena)
{
    if (!arena)
        return 0;
    return ArenaPtr(arena)->remaining();
}

nk_status_t nk_parse_architecture(const char* json_path, nk_arch_info_t* info)
{
    if (!json_path || !info)
    {
        SetLastError("json_path and info are required");
        return NK_ERR_INVALID_ARGUMENT;
    }

    char path_buffer[ModelLoader::kMaxPathLen] = {};
    const char* resolved = ResolveModelPath(json_path, path_buffer, sizeof(path_buffer));

    ModelLoader::ArchitectureSpec spec{};
    const ModelLoader::LoadResult result = ModelLoader::ParseArchitecture(resolved, spec);
    if (result.status != ModelLoader::LoadStatus::Ok)
    {
        SetLastError(result.message ? result.message : "architecture parse failed");
        return FromLoadStatus(result.status);
    }

    std::memset(info, 0, sizeof(*info));
    FillArchInfo(spec, info);
    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_model_load(const char* json_path, nk_arena_t* arena, nk_model_t* model)
{
    if (!json_path || !arena || !model)
    {
        SetLastError("json_path, arena, and model are required");
        return NK_ERR_INVALID_ARGUMENT;
    }

    char path_buffer[ModelLoader::kMaxPathLen] = {};
    const char* resolved = ResolveModelPath(json_path, path_buffer, sizeof(path_buffer));

    ModelLoader::ArchitectureSpec spec{};
    const ModelLoader::LoadResult arch_result = ModelLoader::ParseArchitecture(resolved, spec);
    if (arch_result.status != ModelLoader::LoadStatus::Ok)
    {
        SetLastError(arch_result.message ? arch_result.message : "architecture parse failed");
        return FromLoadStatus(arch_result.status);
    }

    std::memset(model->storage, 0, sizeof(model->storage));
    ModelState* state = ModelPtr(model);
    state->input_rank = spec.input_rank;
    state->input_elements = InputElementCount(spec);
    for (uint32_t i = 0; i < spec.input_rank; ++i)
        state->input_shape[i] = spec.input_shape[i];

    Arena* cpp_arena = ArenaPtr(arena);

    if (spec.kind == ModelLoader::NetworkKind::MLP)
    {
        const ModelLoader::LoadResult load_result =
            ModelLoader::LoadMLP(resolved, *cpp_arena, state->mlp, state->input_shape, state->input_rank);

        if (load_result.status != ModelLoader::LoadStatus::Ok || !state->mlp || !state->mlp->IsValid())
        {
            SetLastError(load_result.message ? load_result.message : "MLP load failed");
            return FromLoadStatus(load_result.status);
        }

        state->kind = NK_NETWORK_MLP;
        state->output_elements = MlpOutputElements(spec);
        state->loaded = true;
    }
    else if (spec.kind == ModelLoader::NetworkKind::CNN)
    {
        const ModelLoader::LoadResult load_result =
            ModelLoader::LoadCNN(resolved, *cpp_arena, state->cnn, state->input_shape, state->input_rank);

        if (load_result.status != ModelLoader::LoadStatus::Ok || !state->cnn || !state->cnn->IsValid())
        {
            SetLastError(load_result.message ? load_result.message : "CNN load failed");
            return FromLoadStatus(load_result.status);
        }

        state->kind = NK_NETWORK_CNN;
        state->output_elements = CnnOutputElements(spec);
        state->loaded = true;
    }
    else
    {
        SetLastError("unsupported network kind");
        return NK_ERR_UNSUPPORTED_NETWORK;
    }

    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_model_get_arch(const nk_model_t* model, nk_arch_info_t* info)
{
    if (!model || !info)
    {
        SetLastError("model and info are required");
        return NK_ERR_INVALID_ARGUMENT;
    }

    const ModelState* state = ModelPtr(model);
    if (!state->loaded)
    {
        SetLastError("model is not loaded");
        return NK_ERR_MODEL_NOT_LOADED;
    }

    std::memset(info, 0, sizeof(*info));
    info->version = 1;
    info->kind = state->kind;
    info->input_rank = state->input_rank;
    info->input_elements = state->input_elements;
    info->output_elements = state->output_elements;

    for (uint32_t i = 0; i < NK_MAX_TENSOR_RANK; ++i)
        info->input_shape[i] = i < state->input_rank ? state->input_shape[i] : 0;

    SetLastError(nullptr);
    return NK_OK;
}

uint32_t nk_model_input_count(const nk_model_t* model)
{
    if (!model)
        return 0;
    return ModelPtr(model)->input_elements;
}

uint32_t nk_model_output_count(const nk_model_t* model)
{
    if (!model)
        return 0;
    return ModelPtr(model)->output_elements;
}

nk_network_kind_t nk_model_kind(const nk_model_t* model)
{
    if (!model)
        return NK_NETWORK_UNKNOWN;
    return ModelPtr(model)->kind;
}

nk_status_t nk_model_run(const nk_model_t* model,
                         nk_arena_t* arena,
                         const float* input,
                         uint32_t input_count,
                         float* output,
                         uint32_t output_capacity,
                         uint32_t* output_count)
{
    if (!model || !arena || !input || !output || !output_count)
    {
        SetLastError("model, arena, input, output, and output_count are required");
        return NK_ERR_INVALID_ARGUMENT;
    }

    const ModelState* state = ModelPtr(model);
    if (!state->loaded)
    {
        SetLastError("model is not loaded");
        return NK_ERR_MODEL_NOT_LOADED;
    }

    if (input_count != state->input_elements)
    {
        SetLastError("input element count does not match model");
        return NK_ERR_INVALID_ARGUMENT;
    }

    if (output_capacity < state->output_elements)
    {
        SetLastError("output buffer too small");
        return NK_ERR_BUFFER_TOO_SMALL;
    }

    Arena* cpp_arena = ArenaPtr(arena);

    if (state->kind == NK_NETWORK_MLP)
    {
        Tensor input_tensor =
            TensorFactory::Create2D(*cpp_arena, state->input_shape[0], state->input_shape[1]);
        if (!input_tensor.data)
        {
            SetLastError("arena overflow while allocating input");
            return NK_ERR_ARENA_OVERFLOW;
        }

        float* input_data = static_cast<float*>(input_tensor.data);
        for (uint32_t i = 0; i < input_count; ++i)
            input_data[i] = input[i];

        const uint32_t output_cols = state->output_elements / state->input_shape[0];
        Tensor output_tensor = TensorFactory::Create2D(*cpp_arena, state->input_shape[0], output_cols);
        if (!output_tensor.data)
        {
            SetLastError("arena overflow while allocating output");
            return NK_ERR_ARENA_OVERFLOW;
        }

        state->mlp->forward(input_tensor, output_tensor, *cpp_arena);

        const float* out_data = static_cast<const float*>(output_tensor.data);
        for (uint32_t i = 0; i < state->output_elements; ++i)
            output[i] = out_data[i];
    }
    else if (state->kind == NK_NETWORK_CNN)
    {
        float input_buffer[4096] = {};
        if (input_count > 4096)
        {
            SetLastError("input too large");
            return NK_ERR_INVALID_ARGUMENT;
        }

        for (uint32_t i = 0; i < input_count; ++i)
            input_buffer[i] = input[i];

        Tensor input_tensor = MakeNhwcInput(input_buffer,
                                            state->input_shape[0],
                                            state->input_shape[1],
                                            state->input_shape[2]);

        Tensor& output_tensor = state->cnn->forward(input_tensor, *cpp_arena);
        if (!output_tensor.data)
        {
            SetLastError("arena overflow during CNN forward pass");
            return NK_ERR_ARENA_OVERFLOW;
        }

        if (output_tensor.num_elements != state->output_elements)
        {
            SetLastError("unexpected CNN output size");
            return NK_ERR_INVALID_ARGUMENT;
        }

        const float* out_data = static_cast<const float*>(output_tensor.data);
        for (uint32_t i = 0; i < state->output_elements; ++i)
            output[i] = out_data[i];
    }
    else
    {
        SetLastError("unsupported network kind");
        return NK_ERR_UNSUPPORTED_NETWORK;
    }

    *output_count = state->output_elements;
    SetLastError(nullptr);
    return NK_OK;
}

nk_status_t nk_inspect_model(const char* json_path, nk_arena_t* arena, nk_inspect_info_t* info)
{
    if (!json_path || !arena || !info)
    {
        SetLastError("json_path, arena, and info are required");
        return NK_ERR_INVALID_ARGUMENT;
    }

    std::memset(info, 0, sizeof(*info));

    const nk_status_t arch_status = nk_parse_architecture(json_path, &info->arch);
    if (arch_status != NK_OK)
        return arch_status;

    nk_model_t model{};
    const nk_status_t load_status = nk_model_load(json_path, arena, &model);
    if (load_status != NK_OK)
        return load_status;

    info->arena_bytes_after_load = nk_arena_used(arena);

    float zero_input[4096] = {};
    if (info->arch.input_elements > 4096)
    {
        SetLastError("input too large for inspect");
        return NK_ERR_INVALID_ARGUMENT;
    }

    float output_buffer[4096] = {};
    uint32_t output_count = 0;
    const nk_status_t run_status = nk_model_run(&model,
                                                arena,
                                                zero_input,
                                                info->arch.input_elements,
                                                output_buffer,
                                                4096,
                                                &output_count);
    if (run_status != NK_OK)
        return run_status;

    float* weights = nullptr;
    std::size_t float_count = 0;
    const char* weight_error = nullptr;

    char path_buffer[ModelLoader::kMaxPathLen] = {};
    const char* resolved = ResolveModelPath(json_path, path_buffer, sizeof(path_buffer));

    Arena scratch{};
    alignas(std::max_align_t) unsigned char scratch_buffer[512];
    scratch.init(scratch_buffer, sizeof(scratch_buffer));

    const ModelLoader::LoadStatus weight_status =
        ModelLoader::LoadWeightsBin(resolved, scratch, weights, float_count, &weight_error);

    if (weight_status == ModelLoader::LoadStatus::Ok)
        info->weight_floats = float_count;

    info->arena_bytes_after_forward = nk_arena_used(arena);
    info->arena_remaining = nk_arena_remaining(arena);

    SetLastError(nullptr);
    return NK_OK;
}

} /* extern "C" */
