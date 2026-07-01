#include "arena.hpp"
#include <cstdint>
#include <cstddef>

namespace
{
    bool IsPowerOfTwo(std::size_t value)
    {
        return value != 0 && (value & (value - 1)) == 0;
    }

    bool AddWouldOverflow(std::size_t a, std::size_t b)
    {
        return a > SIZE_MAX - b;
    }
}

void Arena::init(void* memory, std::size_t size)
{
    base = static_cast<std::byte*>(memory);
    capacity = size;
    offset = 0;
}

void* Arena::alloc(std::size_t size, std::size_t alignment)
{
    if (!base)
        return nullptr;

    if (size == 0 || alignment == 0 || !IsPowerOfTwo(alignment))
        return nullptr;

    if (offset > capacity)
        return nullptr;

    const std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(base);
    const std::uintptr_t raw_addr = base_addr + offset;
    if (raw_addr < base_addr)
        return nullptr;

    const std::uintptr_t mask = static_cast<std::uintptr_t>(alignment - 1);
    const std::uintptr_t aligned_addr = (raw_addr + mask) & ~mask;
    const std::size_t aligned_offset = static_cast<std::size_t>(aligned_addr - base_addr);
    if (aligned_offset < offset)
        return nullptr;

    if (AddWouldOverflow(aligned_offset, size))
        return nullptr;

    const std::size_t new_offset = aligned_offset + size;
    if (new_offset > capacity)
        return nullptr;

    offset = new_offset;
    return reinterpret_cast<void*>(aligned_addr);
}

void Arena::reset()
{
    offset = 0;
}

std::size_t Arena::remaining() const
{
    if (!base || offset > capacity)
        return 0;
    return capacity - offset;
}
