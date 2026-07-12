/*
 * nk_infer — run one forward pass.
 *
 * Usage:
 *   nk_infer <model.nk> <input floats...>              # CSV stdout
 *   nk_infer <model.nk> @<in.bin>                      # CSV stdout
 *   nk_infer <model.nk> @<in.bin> --out-bin <out.bin>  # raw float32 file
 *
 * @path.bin is little-endian float32 with exactly input_elements values.
 * Build: make tools/nk_infer (cpu only)
 */
#include "netkit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>

static int load_input_bin(const char* path, float* dst, uint32_t n)
{
    FILE* f = fopen(path, "rb");
    if (!f)
        return 0;
    const size_t got = fread(dst, sizeof(float), n, f);
    fclose(f);
    return got == (size_t)n;
}

static int write_output_bin(const char* path, const float* src, uint32_t n)
{
    FILE* f = fopen(path, "wb");
    if (!f)
        return 0;
    const size_t put = fwrite(src, sizeof(float), n, f);
    fclose(f);
    return put == (size_t)n;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        fprintf(stderr,
                "Usage: %s <model.nk> <input floats...>\n"
                "       %s <model.nk> @<in.bin> [--out-bin <out.bin>]\n",
                argv[0],
                argv[0]);
        return 1;
    }

    const char* nk_path = argv[1];
    const int use_bin = (argv[2][0] == '@');
    const char* out_bin = nullptr;
    int input_argc_end = argc;
    if (use_bin && argc >= 5 && strcmp(argv[3], "--out-bin") == 0)
    {
        out_bin = argv[4];
        input_argc_end = 3;
    }
    else if (!use_bin)
    {
        for (int i = 2; i + 1 < argc; ++i)
        {
            if (strcmp(argv[i], "--out-bin") == 0)
            {
                out_bin = argv[i + 1];
                input_argc_end = i;
                break;
            }
        }
    }

    nk_arch_info_t arch = {0};
    if (nk_parse_architecture(nk_path, &arch) != NK_OK)
    {
        fprintf(stderr, "nk_parse_architecture failed for %s\n", nk_path);
        return 1;
    }

    float* input = (float*)malloc((size_t)arch.input_elements * sizeof(float));
    float* output = (float*)malloc((size_t)arch.output_elements * sizeof(float));
    if (!input || !output)
    {
        fprintf(stderr, "malloc failed for I/O buffers (%u in / %u out)\n", arch.input_elements,
                arch.output_elements);
        free(input);
        free(output);
        return 1;
    }

    if (use_bin)
    {
        const char* bin_path = argv[2] + 1;
        if (input_argc_end != 3 || !load_input_bin(bin_path, input, arch.input_elements))
        {
            fprintf(stderr, "failed to read %u float32s from %s\n", arch.input_elements, bin_path);
            free(input);
            free(output);
            return 1;
        }
    }
    else
    {
        const int input_arg_count = input_argc_end - 2;
        if ((uint32_t)input_arg_count != arch.input_elements)
        {
            fprintf(stderr, "input count %d != model input_elements %u\n", input_arg_count,
                    arch.input_elements);
            free(input);
            free(output);
            return 1;
        }
        for (int i = 0; i < input_arg_count; ++i)
            input[i] = strtof(argv[i + 2], nullptr);
    }

#if defined(NETKIT_ARENA_HEAP)
    const size_t arena_capacity = nk_recommended_arena_bytes(nk_path);
    if (arena_capacity == 0)
    {
        fprintf(stderr, "nk_recommended_arena_bytes returned 0\n");
        free(input);
        free(output);
        return 1;
    }

    nk_arena_t arena;
    if (nk_arena_init_heap(&arena, arena_capacity) != NK_OK)
    {
        fprintf(stderr, "nk_arena_init_heap failed capacity=%zu\n", arena_capacity);
        free(input);
        free(output);
        return 1;
    }
#else
    alignas(max_align_t) static unsigned char arena_memory[NK_ARENA_DEFAULT_CAPACITY];
    nk_arena_t arena;
    nk_arena_init(&arena, arena_memory, sizeof(arena_memory));
#endif

    nk_model_t model;
    if (nk_model_load(nk_path, &arena, &model) != NK_OK)
    {
        fprintf(stderr, "nk_model_load failed for %s (arena=%zu)\n", nk_path, arena_capacity);
#if defined(NETKIT_ARENA_HEAP)
        nk_arena_destroy_heap(&arena);
#endif
        free(input);
        free(output);
        return 1;
    }

    uint32_t output_count = 0;
    const nk_status_t run_st =
        nk_model_run(&model, &arena, input, arch.input_elements, output, arch.output_elements,
                     &output_count);
    if (run_st != NK_OK)
    {
        fprintf(stderr, "nk_model_run failed status=%d\n", (int)run_st);
#if defined(NETKIT_ARENA_HEAP)
        nk_arena_destroy_heap(&arena);
#endif
        free(input);
        free(output);
        return 1;
    }

    int ok = 1;
    if (out_bin)
    {
        ok = write_output_bin(out_bin, output, output_count);
        if (ok)
            printf("wrote %s (%u floats)\n", out_bin, output_count);
        else
            fprintf(stderr, "failed to write %s\n", out_bin);
    }
    else
    {
        for (uint32_t i = 0; i < output_count; ++i)
        {
            if (i > 0)
                putchar(',');
            printf("%.9g", output[i]);
        }
        putchar('\n');
    }

#if defined(NETKIT_ARENA_HEAP)
    nk_arena_destroy_heap(&arena);
#endif
    free(input);
    free(output);
    return ok ? 0 : 1;
}
