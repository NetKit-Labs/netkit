#include "quant_trace.hpp"

#include "kernel_workspace.hpp"

#include <cstdio>

#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE

#if defined(NETKIT_QUANT_TRACE_UART) && NETKIT_QUANT_TRACE_UART
#include "uart.h"
#endif

namespace QuantTrace
{
    namespace
    {
        Stats g_stats{};

        const char* Conv2dFailName(Conv2dFail reason)
        {
            switch (reason)
            {
                case Conv2dFail::NullPtr:
                    return "null_ptr";
                case Conv2dFail::AsymmetricPad:
                    return "asym_pad";
                case Conv2dFail::TooManyChannels:
                    return "too_many_oc";
                case Conv2dFail::BadQuant:
                    return "bad_quant";
                case Conv2dFail::BindContext:
                    return "bind_ctx";
                case Conv2dFail::CmsisStatus:
                    return "cmsis_status";
                case Conv2dFail::Disabled:
                    return "disabled";
            }
            return "?";
        }

        const char* FcFailName(FcFail reason)
        {
            switch (reason)
            {
                case FcFail::NullPtr:
                    return "null_ptr";
                case FcFail::FloatOutput:
                    return "float_out";
                case FcFail::BadShape:
                    return "bad_shape";
                case FcFail::BindContext:
                    return "bind_ctx";
                case FcFail::CmsisStatus:
                    return "cmsis_status";
                case FcFail::Disabled:
                    return "disabled";
            }
            return "?";
        }

        uint32_t ActiveWorkspaceBytes()
        {
            const KernelWorkspace* ws = GetActiveKernelWorkspace();
            return ws ? static_cast<uint32_t>(ws->size_bytes) : 0u;
        }
    }

    void Reset()
    {
        const uint32_t workspace = g_stats.kernel_workspace_bytes;
        const uint32_t ping = g_stats.ping_buffer_bytes;
        g_stats = {};
        g_stats.kernel_workspace_bytes = workspace;
        g_stats.ping_buffer_bytes = ping;
    }

    const Stats& GetStats()
    {
        return g_stats;
    }

    void RecordKernelPlan(uint32_t kernel_workspace_bytes, uint32_t ping_buffer_bytes)
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        g_stats.kernel_workspace_bytes = kernel_workspace_bytes;
        g_stats.ping_buffer_bytes = ping_buffer_bytes;
#endif
    }

    void RecordConv2dCmsisOk()
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        ++g_stats.conv_cmsis_ok;
#endif
    }

    void RecordConv2dCmsisFail(Conv2dFail reason, int32_t buf_bytes, uint32_t workspace_bytes)
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        ++g_stats.conv_cmsis_fail;
        g_stats.conv_last_fail = reason;
        g_stats.conv_last_buf_bytes = buf_bytes;
        g_stats.conv_last_workspace_bytes = workspace_bytes;
#endif
    }

    void RecordConv2dReference()
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        ++g_stats.conv_ref;
#endif
    }

    void RecordPoolCmsisOk()
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        ++g_stats.pool_cmsis_ok;
#endif
    }

    void RecordPoolReference()
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        ++g_stats.pool_ref;
#endif
    }

    void RecordFcCmsisOk()
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        ++g_stats.fc_cmsis_ok;
#endif
    }

    void RecordFcCmsisFail(FcFail reason, int32_t buf_bytes, uint32_t workspace_bytes)
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        (void)workspace_bytes;
        ++g_stats.fc_cmsis_fail;
        g_stats.fc_last_fail = reason;
        g_stats.fc_last_buf_bytes = buf_bytes;
#endif
    }

    void RecordFcReference()
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        ++g_stats.fc_ref;
#endif
    }

    void RecordSoftmaxCmsisOk()
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        ++g_stats.softmax_cmsis_ok;
#endif
    }

    void RecordSoftmaxReference()
    {
#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
        ++g_stats.softmax_ref;
#endif
    }

    std::size_t FormatSummary(char* buf, std::size_t capacity)
    {
        if (!buf || capacity == 0)
            return 0;

        const int n = std::snprintf(
            buf,
            capacity,
            "quant trace:\r\n"
            "  plan workspace=%u ping=%u\r\n"
            "  conv2d cmsis_ok=%u ref=%u fail=%u last=%s buf=%ld ws=%u\r\n"
            "  pool   cmsis_ok=%u ref=%u\r\n"
            "  fc     cmsis_ok=%u ref=%u fail=%u last=%s buf=%ld\r\n"
            "  softmax cmsis_ok=%u ref=%u\r\n",
            g_stats.kernel_workspace_bytes,
            g_stats.ping_buffer_bytes,
            g_stats.conv_cmsis_ok,
            g_stats.conv_ref,
            g_stats.conv_cmsis_fail,
            Conv2dFailName(g_stats.conv_last_fail),
            static_cast<long>(g_stats.conv_last_buf_bytes),
            g_stats.conv_last_workspace_bytes,
            g_stats.pool_cmsis_ok,
            g_stats.pool_ref,
            g_stats.fc_cmsis_ok,
            g_stats.fc_ref,
            g_stats.fc_cmsis_fail,
            FcFailName(g_stats.fc_last_fail),
            static_cast<long>(g_stats.fc_last_buf_bytes),
            g_stats.softmax_cmsis_ok,
            g_stats.softmax_ref);
        if (n < 0)
        {
            buf[0] = '\0';
            return 0;
        }
        return static_cast<std::size_t>(n);
    }

#if defined(NETKIT_QUANT_TRACE) && NETKIT_QUANT_TRACE
    void PrintSummaryUart()
    {
#if defined(NETKIT_QUANT_TRACE_UART) && NETKIT_QUANT_TRACE_UART
        char buf[384];
        FormatSummary(buf, sizeof(buf));
        uart_write(buf);
        uart_write("\r\n");
#else
        (void)FormatSummary;
#endif
    }
#endif

}  // namespace QuantTrace

#endif  // NETKIT_QUANT_TRACE
