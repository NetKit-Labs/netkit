#pragma once

#include "arena.hpp"
#include "cnn.hpp"
#include "mlp.hpp"
#include "nk_loader.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace OnnxImporter
{
    enum class ImportStatus
    {
        Ok,
        FileOpenFailed,
        ParseFailed,
        UnsupportedGraph,
        UnsupportedOp,
        ShapeMismatch,
        ArenaOverflow
    };

    struct ImportResult
    {
        ImportStatus status = ImportStatus::Ok;
        NkLoader::NetworkKind kind = NkLoader::NetworkKind::Unknown;
        const char* message = nullptr;
    };

    ImportResult LoadFromOnnx(const char* onnx_path,
                              Arena& arena,
                              NkLoader::NetworkKind& kind,
                              MLPNetwork*& mlp,
                              CNNNetwork*& cnn,
                              std::array<uint32_t, kMaxTensorRank>& input_shape,
                              uint32_t& input_rank);

    const char* StatusMessage(ImportStatus status);
}
