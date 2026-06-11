#include "arena.hpp"

void Arena::init(void* memory, std::size_t size)
{
    base = static_cast<unsigned char*>(memory);
    capacity = size;
    offset = 0;
}

void* Arena::alloc(std::size_t size)
{
    void* p = base + offset;
    offset += size;
    return p;
}

void Arena::reset()
{
    offset = 0;
}
