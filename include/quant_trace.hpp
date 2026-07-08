#pragma once

#include <cstddef>
#include <cstdint>

// Optional quantized-kernel dispatch tracing (CMSIS vs reference).
// Enable with -DNETKIT_QUANT_TRACE=1; on MCU firmware add -DNETKIT_QUANT_TRACE_UART=1
// and link uart.c so PrintSummaryUart() emits over serial.

namespace QuantTrace
{

enum class Conv2dFail : uint8_t
{
    NullPtr = 1,
    AsymmetricPad = 2,
    TooManyChannels = 3,
    BadQuant = 4,
    BindContext = 5,
    CmsisStatus = 6,
    Disabled = 7,
};

enum class FcFail : uint8_t
{
    NullPtr = 1,
    FloatOutput = 2,
    BadShape = 3,
    BindContext = 4,
    CmsisStatus = 5,
    Disabled = 6,
};

struct Stats
{
    uint32_t conv_cmsis_ok = 0;
    uint32_t conv_ref = 0;
    uint32_t conv_cmsis_fail = 0;
    Conv2dFail conv_last_fail = Conv2dFail::Disabled;
    int32_t conv_last_buf_bytes = 0;
    uint32_t conv_last_workspace_bytes = 0;

    uint32_t pool_cmsis_ok = 0;
    uint32_t pool_ref = 0;

    uint32_t fc_cmsis_ok = 0;
    uint32_t fc_ref = 0;
    uint32_t fc_cmsis_fail = 0;
    FcFail fc_last_fail = FcFail::Disabled;
    int32_t fc_last_buf_bytes = 0;

    uint32_t softmax_cmsis_ok = 0;
    uint32_t softmax_ref = 0;

    uint32_t kernel_workspace_bytes = 0;
    uint32_t ping_buffer_bytes = 0;
};

#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE

void Reset();

const Stats& GetStats();

void RecordKernelPlan(uint32_t kernel_workspace_bytes, uint32_t ping_buffer_bytes);

void RecordConv2dCmsisOk();
void RecordConv2dCmsisFail(Conv2dFail reason, int32_t buf_bytes = 0, uint32_t workspace_bytes = 0);
void RecordConv2dReference();

void RecordPoolCmsisOk();
void RecordPoolReference();

void RecordFcCmsisOk();
void RecordFcCmsisFail(FcFail reason, int32_t buf_bytes = 0, uint32_t workspace_bytes = 0);
void RecordFcReference();

void RecordSoftmaxCmsisOk();
void RecordSoftmaxReference();

// Write a multi-line summary into buf (NUL-terminated). Returns bytes written excluding NUL.
std::size_t FormatSummary(char* buf, std::size_t capacity);

void PrintSummaryUart();

#else  // !NETKIT_QUANT_TRACE

inline void Reset() {}
inline const Stats& GetStats()
{
    static const Stats kEmpty{};
    return kEmpty;
}
inline void RecordKernelPlan(uint32_t, uint32_t) {}
inline void RecordConv2dCmsisOk() {}
inline void RecordConv2dCmsisFail(Conv2dFail, int32_t = 0, uint32_t = 0) {}
inline void RecordConv2dReference() {}
inline void RecordPoolCmsisOk() {}
inline void RecordPoolReference() {}
inline void RecordFcCmsisOk() {}
inline void RecordFcCmsisFail(FcFail, int32_t = 0, uint32_t = 0) {}
inline void RecordFcReference() {}
inline void RecordSoftmaxCmsisOk() {}
inline void RecordSoftmaxReference() {}
inline std::size_t FormatSummary(char*, std::size_t) { return 0; }
inline void PrintSummaryUart() {}

#endif  // NETKIT_QUANT_TRACE

}  // namespace QuantTrace
