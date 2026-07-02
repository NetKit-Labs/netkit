#pragma once
#include "netkit_config.h"
#include <cstddef>
#include <cstdint>

struct Arena {
#if defined(NETKIT_TARGET_MPU)
    static constexpr std::size_t kDefaultCapacity = 128 * 1024; // 128 KiB
#elif defined(NETKIT_TARGET_MCU)
    static constexpr std::size_t kDefaultCapacity = 64 * 1024; // 64 KiB
#else
    static constexpr std::size_t kDefaultCapacity = 4 * 1024 * 1024; // 4 MiB (CPU)
#endif

    std::byte* base{};
    std::size_t capacity = 0;
    std::size_t offset = 0;
    bool heap_owned = false;

    void init(void* memory, std::size_t size);
#if defined(NETKIT_ARENA_HEAP)
    // Heap-backed backing store (malloc once). CPU may free via destroy_heap(); MCU/MPU never free.
    bool init_heap(std::size_t size);
    void destroy_heap();
#endif
    // Returns nullptr when the arena is exhausted, not initialized, or alignment is invalid.
    // alignment must be a power of two. Padding is inserted before the block when needed.
    void* alloc(std::size_t size, std::size_t alignment);
    void reset();
    std::size_t remaining() const;
};
