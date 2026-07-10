#pragma once

#include "netkit_config.h"

#include <cstddef>
#include <cstdint>

#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED

namespace CmsisQuantPlan
{

// Persistent XNNPACK qs8 operator created once at BuildRuntime / first use.
// Forward path only calls setup + run (TFLite-style), avoiding create/delete.
// When using a shared weights_cache, create packs weights first; reshape (which
// caches absolute packed-weight pointers) runs only after cache finalize so
// macOS allocate+copy growth cannot leave stale pointers.
struct XnnpackOpHoist
{
    void* op = nullptr;  // xnn_operator_t
    uint8_t* workspace = nullptr;
    size_t workspace_bytes = 0;
    bool workspace_heap = false;  // true → DestroyXnnpackOp must delete[]
    bool per_channel = false;     // qs8_qc8w vs qs8 (conv/FC)
    bool ready = false;
    // Geometry for deferred reshape after weights_cache finalize (conv/dw only).
    size_t reshape_in_h = 0;
    size_t reshape_in_w = 0;
    size_t reshape_out_h = 0;
    size_t reshape_out_w = 0;
};

}  // namespace CmsisQuantPlan

#define NETKIT_XNNPACK_PLAN_HOIST 1

#else

#define NETKIT_XNNPACK_PLAN_HOIST 0

#endif
