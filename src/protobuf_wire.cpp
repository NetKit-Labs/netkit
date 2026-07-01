#include "protobuf_wire.hpp"

namespace ProtoWire
{
    Reader::Reader(const uint8_t* data, std::size_t size)
        : data_(data), size_(size), pos_(0)
    {
    }

    bool Reader::read_varint(uint64_t& value)
    {
        value = 0;
        uint32_t shift = 0;

        while (pos_ < size_)
        {
            const uint8_t byte = data_[pos_++];
            value |= static_cast<uint64_t>(byte & 0x7Fu) << shift;
            if ((byte & 0x80u) == 0)
                return true;
            shift += 7;
            if (shift >= 64)
                return false;
        }

        return false;
    }

    bool Reader::read_tag(uint32_t& field_number, WireType& wire_type)
    {
        uint64_t tag = 0;
        if (!read_varint(tag))
            return false;

        field_number = static_cast<uint32_t>(tag >> 3);
        wire_type = static_cast<WireType>(tag & 0x7u);
        return field_number != 0;
    }

    bool Reader::read_fixed32(uint32_t& value)
    {
        if (pos_ + 4 > size_)
            return false;

        const uint32_t b0 = data_[pos_++];
        const uint32_t b1 = data_[pos_++];
        const uint32_t b2 = data_[pos_++];
        const uint32_t b3 = data_[pos_++];
        value = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        return true;
    }

    bool Reader::read_fixed64(uint64_t& value)
    {
        if (pos_ + 8 > size_)
            return false;

        uint64_t out = 0;
        for (int i = 0; i < 8; ++i)
            out |= static_cast<uint64_t>(data_[pos_++]) << (8 * i);
        value = out;
        return true;
    }

    bool Reader::read_length_delimited(std::span<const uint8_t>& payload)
    {
        uint64_t len = 0;
        if (!read_varint(len))
            return false;
        if (pos_ + len > size_)
            return false;

        payload = std::span<const uint8_t>(data_ + pos_, static_cast<std::size_t>(len));
        pos_ += static_cast<std::size_t>(len);
        return true;
    }

    bool Reader::skip_field(WireType wire_type)
    {
        switch (wire_type)
        {
            case WireType::Varint:
            {
                uint64_t ignored = 0;
                return read_varint(ignored);
            }
            case WireType::Fixed64:
            {
                uint64_t ignored = 0;
                return read_fixed64(ignored);
            }
            case WireType::Fixed32:
            {
                uint32_t ignored = 0;
                return read_fixed32(ignored);
            }
            case WireType::LengthDelimited:
            {
                std::span<const uint8_t> ignored;
                return read_length_delimited(ignored);
            }
        }
        return false;
    }
}
