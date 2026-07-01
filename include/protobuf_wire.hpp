#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace ProtoWire
{
    enum class WireType : uint32_t
    {
        Varint = 0,
        Fixed64 = 1,
        LengthDelimited = 2,
        Fixed32 = 5
    };

    class Reader
    {
    public:
        Reader(const uint8_t* data, std::size_t size);

        bool eof() const { return pos_ >= size_; }
        std::size_t remaining() const { return pos_ < size_ ? size_ - pos_ : 0; }

        bool read_tag(uint32_t& field_number, WireType& wire_type);
        bool read_varint(uint64_t& value);
        bool read_fixed32(uint32_t& value);
        bool read_fixed64(uint64_t& value);
        bool read_length_delimited(std::span<const uint8_t>& payload);
        bool skip_field(WireType wire_type);

    private:
        const uint8_t* data_;
        std::size_t size_;
        std::size_t pos_;
    };
}
