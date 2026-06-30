/*
 * netkit.h — C23 public API for netkit inference
 *
 * Link against libnetkit (built from C++26 sources). When compiling this header
 * from C, use -std=c23. The implementation bridge is C++26.
 */
#ifndef NETKIT_H
#define NETKIT_H

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NK_VERSION_MAJOR 0
#define NK_VERSION_MINOR 1
#define NK_VERSION_PATCH 0

#define NK_MAX_TENSOR_RANK 4
#define NK_MAX_LAYERS      16
#define NK_MAX_PATH_LEN    256
#define NK_MAX_MESSAGE_LEN 128
#define NK_ARENA_DEFAULT_CAPACITY (64U * 1024U)

#define NK_ARENA_STORAGE_BYTES 32
#define NK_MODEL_STORAGE_BYTES  64

typedef enum nk_status
{
    NK_OK = 0,
    NK_ERR_JSON_OPEN,
    NK_ERR_BIN_OPEN,
    NK_ERR_JSON_PARSE,
    NK_ERR_UNSUPPORTED_NETWORK,
    NK_ERR_VERSION_MISMATCH,
    NK_ERR_LAYER_CONFIG,
    NK_ERR_BIN_SIZE_MISMATCH,
    NK_ERR_ARENA_OVERFLOW,
    NK_ERR_INVALID_ARGUMENT,
    NK_ERR_BUFFER_TOO_SMALL,
    NK_ERR_MODEL_NOT_LOADED
} nk_status_t;

typedef enum nk_network_kind
{
    NK_NETWORK_UNKNOWN = 0,
    NK_NETWORK_MLP,
    NK_NETWORK_CNN
} nk_network_kind_t;

/* Opaque storage sized for embedded stack allocation (no heap). */
typedef struct nk_arena
{
    alignas(max_align_t) unsigned char storage[NK_ARENA_STORAGE_BYTES];
} nk_arena_t;

typedef struct nk_model
{
    alignas(max_align_t) unsigned char storage[NK_MODEL_STORAGE_BYTES];
} nk_model_t;

typedef struct nk_arch_info
{
    uint32_t version;
    nk_network_kind_t kind;
    uint32_t input_shape[NK_MAX_TENSOR_RANK];
    uint32_t input_rank;
    uint32_t num_layers;
    size_t expected_weight_floats;
    uint32_t input_elements;
    uint32_t output_elements;
} nk_arch_info_t;

typedef struct nk_inspect_info
{
    nk_arch_info_t arch;
    size_t weight_floats;
    size_t arena_bytes_after_load;
    size_t arena_bytes_after_forward;
    size_t arena_remaining;
} nk_inspect_info_t;

/* Version / errors */
const char* nk_version_string(void);
const char* nk_status_string(nk_status_t status);
const char* nk_last_error(void);

/* Arena */
void nk_arena_init(nk_arena_t* arena, void* memory, size_t size);
void nk_arena_reset(nk_arena_t* arena);
size_t nk_arena_capacity(const nk_arena_t* arena);
size_t nk_arena_used(const nk_arena_t* arena);
size_t nk_arena_remaining(const nk_arena_t* arena);

/* Architecture (JSON only, no weight load) */
nk_status_t nk_parse_architecture(const char* json_path, nk_arch_info_t* info);

/* Model load + inference */
nk_status_t nk_model_load(const char* json_path, nk_arena_t* arena, nk_model_t* model);
nk_status_t nk_model_get_arch(const nk_model_t* model, nk_arch_info_t* info);
uint32_t nk_model_input_count(const nk_model_t* model);
uint32_t nk_model_output_count(const nk_model_t* model);
nk_network_kind_t nk_model_kind(const nk_model_t* model);

nk_status_t nk_model_run(const nk_model_t* model,
                         nk_arena_t* arena,
                         const float* input,
                         uint32_t input_count,
                         float* output,
                         uint32_t output_capacity,
                         uint32_t* output_count);

/* Load model into arena, run zero-input forward pass, report arena usage */
nk_status_t nk_inspect_model(const char* json_path, nk_arena_t* arena, nk_inspect_info_t* info);

#ifdef __cplusplus
}
#endif

#endif /* NETKIT_H */
