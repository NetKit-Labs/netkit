#pragma once

#include "arena.hpp"
#include "cnn.hpp"
#include "mlp.hpp"
#include "model_loader.hpp"
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
        WriteFailed,
        ArenaOverflow
    };

    struct ImportResult
    {
        ImportStatus status = ImportStatus::Ok;
        ModelLoader::NetworkKind kind = ModelLoader::NetworkKind::Unknown;
        const char* message = nullptr;
    };

    // Convert ONNX -> netkit JSON + float32 .bin on disk.
    ImportResult ImportToNetkitFiles(const char* onnx_path, const char* json_out_path, const char* bin_out_path);

    // Load ONNX directly into arena-backed MLP/CNN networks (same result as Import + Load).
    ImportResult LoadFromOnnx(const char* onnx_path,
                              Arena& arena,
                              ModelLoader::NetworkKind& kind,
                              MLPNetwork*& mlp,
                              CNNNetwork*& cnn,
                              std::array<uint32_t, kMaxTensorRank>& input_shape,
                              uint32_t& input_rank);

    const char* StatusMessage(ImportStatus status);
}
