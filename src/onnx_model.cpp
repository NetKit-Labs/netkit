#include "onnx_model.hpp"
#include "protobuf_wire.hpp"

#include <cstdio>
#include <cstring>
#include <span>

namespace Onnx
{
    namespace
    {
        constexpr int32_t kTensorFloat = 1;

        void CopyName(char* dst, std::size_t dst_capacity, const std::span<const uint8_t>& bytes)
        {
            const std::size_t n = bytes.size() < dst_capacity - 1 ? bytes.size() : dst_capacity - 1;
            std::memcpy(dst, bytes.data(), n);
            dst[n] = '\0';
        }

        bool ParseTensorShape(const std::span<const uint8_t>& bytes,
                              std::array<int64_t, kMaxDims>& dims,
                              uint32_t& dim_count)
        {
            ProtoWire::Reader reader(bytes.data(), bytes.size());
            dim_count = 0;

            while (!reader.eof())
            {
                uint32_t field = 0;
                ProtoWire::WireType wire = ProtoWire::WireType::Varint;
                if (!reader.read_tag(field, wire))
                    break;

                if (field == 1 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> dim_bytes;
                    if (!reader.read_length_delimited(dim_bytes))
                        return false;

                    ProtoWire::Reader dim_reader(dim_bytes.data(), dim_bytes.size());
                    while (!dim_reader.eof())
                    {
                        uint32_t dim_field = 0;
                        ProtoWire::WireType dim_wire = ProtoWire::WireType::Varint;
                        if (!dim_reader.read_tag(dim_field, dim_wire))
                            break;

                        if (dim_field == 1 && dim_wire == ProtoWire::WireType::Varint)
                        {
                            uint64_t value = 0;
                            if (!dim_reader.read_varint(value))
                                return false;
                            if (dim_count < kMaxDims)
                                dims[dim_count++] = static_cast<int64_t>(value);
                        }
                        else if (!dim_reader.skip_field(dim_wire))
                            return false;
                    }
                }
                else if (!reader.skip_field(wire))
                    return false;
            }

            return true;
        }

        bool ParseTypeProto(const std::span<const uint8_t>& bytes,
                            std::array<int64_t, kMaxDims>& dims,
                            uint32_t& dim_count)
        {
            ProtoWire::Reader reader(bytes.data(), bytes.size());
            dim_count = 0;

            while (!reader.eof())
            {
                uint32_t field = 0;
                ProtoWire::WireType wire = ProtoWire::WireType::Varint;
                if (!reader.read_tag(field, wire))
                    break;

                if (field == 1 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> tensor_type;
                    if (!reader.read_length_delimited(tensor_type))
                        return false;

                    ProtoWire::Reader type_reader(tensor_type.data(), tensor_type.size());
                    while (!type_reader.eof())
                    {
                        uint32_t inner_field = 0;
                        ProtoWire::WireType inner_wire = ProtoWire::WireType::Varint;
                        if (!type_reader.read_tag(inner_field, inner_wire))
                            break;

                        if (inner_field == 1 && inner_wire == ProtoWire::WireType::Varint)
                        {
                            uint64_t elem_type = 0;
                            if (!type_reader.read_varint(elem_type))
                                return false;
                            (void)elem_type;
                        }
                        else if (inner_field == 2 && inner_wire == ProtoWire::WireType::LengthDelimited)
                        {
                            std::span<const uint8_t> shape_bytes;
                            if (!type_reader.read_length_delimited(shape_bytes))
                                return false;
                            if (!ParseTensorShape(shape_bytes, dims, dim_count))
                                return false;
                        }
                        else if (!type_reader.skip_field(inner_wire))
                            return false;
                    }
                }
                else if (!reader.skip_field(wire))
                    return false;
            }

            return true;
        }

        bool ParseAttribute(const std::span<const uint8_t>& bytes, Attribute& attr)
        {
            ProtoWire::Reader reader(bytes.data(), bytes.size());
            attr = Attribute{};

            while (!reader.eof())
            {
                uint32_t field = 0;
                ProtoWire::WireType wire = ProtoWire::WireType::Varint;
                if (!reader.read_tag(field, wire))
                    break;

                if (field == 1 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> name;
                    if (!reader.read_length_delimited(name))
                        return false;
                    CopyName(attr.name, sizeof(attr.name), name);
                }
                else if (field == 2 && wire == ProtoWire::WireType::Fixed32)
                {
                    uint32_t bits = 0;
                    if (!reader.read_fixed32(bits))
                        return false;
                    std::memcpy(&attr.f, &bits, sizeof(float));
                }
                else if (field == 3 && wire == ProtoWire::WireType::Varint)
                {
                    uint64_t value = 0;
                    if (!reader.read_varint(value))
                        return false;
                    attr.i = static_cast<int64_t>(value);
                }
                else if (field == 7 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> packed;
                    if (!reader.read_length_delimited(packed))
                        return false;
                    for (std::size_t offset = 0; offset + 4 <= packed.size(); offset += 4)
                    {
                        uint32_t bits = static_cast<uint32_t>(packed[offset]) |
                                        (static_cast<uint32_t>(packed[offset + 1]) << 8) |
                                        (static_cast<uint32_t>(packed[offset + 2]) << 16) |
                                        (static_cast<uint32_t>(packed[offset + 3]) << 24);
                        float value = 0.0f;
                        std::memcpy(&value, &bits, sizeof(float));
                        attr.floats.push_back(value);
                    }
                }
                else if (field == 7 && wire == ProtoWire::WireType::Fixed32)
                {
                    uint32_t bits = 0;
                    if (!reader.read_fixed32(bits))
                        return false;
                    float value = 0.0f;
                    std::memcpy(&value, &bits, sizeof(float));
                    attr.floats.push_back(value);
                }
                else if (field == 8 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> packed;
                    if (!reader.read_length_delimited(packed))
                        return false;
                    ProtoWire::Reader packed_reader(packed.data(), packed.size());
                    while (!packed_reader.eof())
                    {
                        uint64_t value = 0;
                        if (!packed_reader.read_varint(value))
                            return false;
                        attr.ints.push_back(static_cast<int64_t>(value));
                    }
                }
                else if (field == 8 && wire == ProtoWire::WireType::Varint)
                {
                    uint64_t value = 0;
                    if (!reader.read_varint(value))
                        return false;
                    attr.ints.push_back(static_cast<int64_t>(value));
                }
                else if (field == 20 && wire == ProtoWire::WireType::Varint)
                {
                    uint64_t value = 0;
                    if (!reader.read_varint(value))
                        return false;
                    attr.type = static_cast<int32_t>(value);
                }
                else if (!reader.skip_field(wire))
                    return false;
            }

            return true;
        }

        bool ParseTensorProto(const std::span<const uint8_t>& bytes, Tensor& tensor)
        {
            ProtoWire::Reader reader(bytes.data(), bytes.size());
            tensor = Tensor{};

            while (!reader.eof())
            {
                uint32_t field = 0;
                ProtoWire::WireType wire = ProtoWire::WireType::Varint;
                if (!reader.read_tag(field, wire))
                    break;

                if (field == 1 && wire == ProtoWire::WireType::Varint)
                {
                    uint64_t dim = 0;
                    if (!reader.read_varint(dim))
                        return false;
                    if (tensor.dim_count < kMaxDims)
                        tensor.dims[tensor.dim_count++] = static_cast<int64_t>(dim);
                }
                else if (field == 2 && wire == ProtoWire::WireType::Varint)
                {
                    uint64_t value = 0;
                    if (!reader.read_varint(value))
                        return false;
                    tensor.data_type = static_cast<int32_t>(value);
                }
                else if (field == 4 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> packed;
                    if (!reader.read_length_delimited(packed))
                        return false;
                    for (std::size_t offset = 0; offset + 4 <= packed.size(); offset += 4)
                    {
                        uint32_t bits = static_cast<uint32_t>(packed[offset]) |
                                        (static_cast<uint32_t>(packed[offset + 1]) << 8) |
                                        (static_cast<uint32_t>(packed[offset + 2]) << 16) |
                                        (static_cast<uint32_t>(packed[offset + 3]) << 24);
                        float value = 0.0f;
                        std::memcpy(&value, &bits, sizeof(float));
                        tensor.float_data.push_back(value);
                    }
                }
                else if (field == 8 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> name;
                    if (!reader.read_length_delimited(name))
                        return false;
                    CopyName(tensor.name, sizeof(tensor.name), name);
                }
                else if (field == 9 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> raw;
                    if (!reader.read_length_delimited(raw))
                        return false;
                    if ((raw.size() % sizeof(float)) != 0)
                        return false;
                    const std::size_t count = raw.size() / sizeof(float);
                    tensor.float_data.resize(count);
                    std::memcpy(tensor.float_data.data(), raw.data(), raw.size());
                }
                else if (!reader.skip_field(wire))
                    return false;
            }

            return true;
        }

        bool ParseNode(const std::span<const uint8_t>& bytes, Node& node)
        {
            ProtoWire::Reader reader(bytes.data(), bytes.size());
            node = Node{};

            while (!reader.eof())
            {
                uint32_t field = 0;
                ProtoWire::WireType wire = ProtoWire::WireType::Varint;
                if (!reader.read_tag(field, wire))
                    break;

                if (field == 1 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> value;
                    if (!reader.read_length_delimited(value))
                        return false;
                    if (node.input_count < kMaxNodeInputs)
                        CopyName(node.inputs[node.input_count++], kMaxNameLen, value);
                }
                else if (field == 2 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> value;
                    if (!reader.read_length_delimited(value))
                        return false;
                    if (node.output_count < kMaxNodeOutputs)
                        CopyName(node.outputs[node.output_count++], kMaxNameLen, value);
                }
                else if (field == 4 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> value;
                    if (!reader.read_length_delimited(value))
                        return false;
                    CopyName(node.op_type, sizeof(node.op_type), value);
                }
                else if (field == 5 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> attr_bytes;
                    if (!reader.read_length_delimited(attr_bytes))
                        return false;
                    if (node.attribute_count < kMaxAttributes)
                    {
                        if (!ParseAttribute(attr_bytes, node.attributes[node.attribute_count++]))
                            return false;
                    }
                }
                else if (!reader.skip_field(wire))
                    return false;
            }

            return true;
        }

        bool ParseValueInfo(const std::span<const uint8_t>& bytes, ValueInfo& info)
        {
            ProtoWire::Reader reader(bytes.data(), bytes.size());
            info = ValueInfo{};

            while (!reader.eof())
            {
                uint32_t field = 0;
                ProtoWire::WireType wire = ProtoWire::WireType::Varint;
                if (!reader.read_tag(field, wire))
                    break;

                if (field == 1 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> name;
                    if (!reader.read_length_delimited(name))
                        return false;
                    CopyName(info.name, sizeof(info.name), name);
                }
                else if (field == 2 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> type_bytes;
                    if (!reader.read_length_delimited(type_bytes))
                        return false;
                    if (!ParseTypeProto(type_bytes, info.dims, info.dim_count))
                        return false;
                }
                else if (!reader.skip_field(wire))
                    return false;
            }

            return true;
        }

        bool ParseGraph(const std::span<const uint8_t>& bytes, Graph& graph)
        {
            ProtoWire::Reader reader(bytes.data(), bytes.size());
            graph = Graph{};

            while (!reader.eof())
            {
                uint32_t field = 0;
                ProtoWire::WireType wire = ProtoWire::WireType::Varint;
                if (!reader.read_tag(field, wire))
                    break;

                if (field == 1 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> node_bytes;
                    if (!reader.read_length_delimited(node_bytes))
                        return false;
                    if (graph.node_count < kMaxNodes)
                    {
                        if (!ParseNode(node_bytes, graph.nodes[graph.node_count++]))
                            return false;
                    }
                }
                else if (field == 5 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> tensor_bytes;
                    if (!reader.read_length_delimited(tensor_bytes))
                        return false;
                    if (graph.initializer_count < kMaxInitializers)
                    {
                        if (!ParseTensorProto(tensor_bytes, graph.initializers[graph.initializer_count++]))
                            return false;
                    }
                }
                else if (field == 11 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> info_bytes;
                    if (!reader.read_length_delimited(info_bytes))
                        return false;
                    if (graph.input_count < kMaxGraphInputs)
                    {
                        if (!ParseValueInfo(info_bytes, graph.inputs[graph.input_count++]))
                            return false;
                    }
                }
                else if (field == 12 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> info_bytes;
                    if (!reader.read_length_delimited(info_bytes))
                        return false;
                    if (graph.output_count < kMaxGraphOutputs)
                    {
                        if (!ParseValueInfo(info_bytes, graph.outputs[graph.output_count++]))
                            return false;
                    }
                }
                else if (!reader.skip_field(wire))
                    return false;
            }

            return true;
        }

        bool ParseModelMessage(const std::span<const uint8_t>& bytes, Model& model)
        {
            ProtoWire::Reader reader(bytes.data(), bytes.size());
            model = Model{};

            while (!reader.eof())
            {
                uint32_t field = 0;
                ProtoWire::WireType wire = ProtoWire::WireType::Varint;
                if (!reader.read_tag(field, wire))
                    break;

                if (field == 7 && wire == ProtoWire::WireType::LengthDelimited)
                {
                    std::span<const uint8_t> graph_bytes;
                    if (!reader.read_length_delimited(graph_bytes))
                        return false;
                    if (!ParseGraph(graph_bytes, model.graph))
                        return false;
                }
                else if (!reader.skip_field(wire))
                    return false;
            }

            return model.graph.node_count > 0;
        }

        void SetError(char* error, std::size_t error_capacity, const char* message)
        {
            if (!error || error_capacity == 0)
                return;
            std::strncpy(error, message, error_capacity - 1);
            error[error_capacity - 1] = '\0';
        }
    }

    const Tensor* FindInitializer(const Graph& graph, const char* name)
    {
        for (uint32_t i = 0; i < graph.initializer_count; ++i)
        {
            if (std::strcmp(graph.initializers[i].name, name) == 0)
                return &graph.initializers[i];
        }
        return nullptr;
    }

    const Attribute* FindAttribute(const Node& node, const char* name)
    {
        for (uint32_t i = 0; i < node.attribute_count; ++i)
        {
            if (std::strcmp(node.attributes[i].name, name) == 0)
                return &node.attributes[i];
        }
        return nullptr;
    }

    int64_t AttributeInt(const Node& node, const char* name, int64_t default_value)
    {
        const Attribute* attr = FindAttribute(node, name);
        if (!attr)
            return default_value;
        return attr->i;
    }

    bool AttributeInts(const Node& node, const char* name, std::span<const int64_t>& out)
    {
        const Attribute* attr = FindAttribute(node, name);
        if (!attr || attr->ints.empty())
            return false;
        out = std::span<const int64_t>(attr->ints.data(), attr->ints.size());
        return true;
    }

    ParseStatus ParseModelBytes(std::span<const uint8_t> bytes, Model& model, char* error, std::size_t error_capacity)
    {
        if (!ParseModelMessage(bytes, model))
        {
            SetError(error, error_capacity, "Invalid or empty ONNX ModelProto graph");
            return ParseStatus::InvalidProtobuf;
        }

        for (uint32_t i = 0; i < model.graph.initializer_count; ++i)
        {
            const Tensor& tensor = model.graph.initializers[i];
            if (tensor.data_type != 0 && tensor.data_type != kTensorFloat)
            {
                SetError(error, error_capacity, "Only float32 ONNX initializers are supported");
                return ParseStatus::UnsupportedFeature;
            }
        }

        return ParseStatus::Ok;
    }

    ParseStatus ParseModelFile(const char* path, Model& model, char* error, std::size_t error_capacity)
    {
        std::FILE* file = std::fopen(path, "rb");
        if (!file)
        {
            SetError(error, error_capacity, "Failed to open ONNX file");
            return ParseStatus::FileOpenFailed;
        }

        std::fseek(file, 0, SEEK_END);
        const long file_size = std::ftell(file);
        std::fseek(file, 0, SEEK_SET);

        if (file_size <= 0 || file_size > static_cast<long>(32 * 1024 * 1024))
        {
            std::fclose(file);
            SetError(error, error_capacity, "ONNX file size out of supported range");
            return ParseStatus::FileTooLarge;
        }

        std::vector<uint8_t> bytes(static_cast<std::size_t>(file_size));
        const std::size_t read_count = std::fread(bytes.data(), 1, bytes.size(), file);
        std::fclose(file);

        if (read_count != bytes.size())
        {
            SetError(error, error_capacity, "Failed to read ONNX file");
            return ParseStatus::FileOpenFailed;
        }

        return ParseModelBytes(bytes, model, error, error_capacity);
    }
}
