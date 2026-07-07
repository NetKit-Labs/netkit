// NUCLEO-F446RE TFLM MNIST MLP invoke benchmark — same 10 images as benchmark/netkit.
// Methodology: 100 runs x 10 images, discard first invoke each run; mean of per-run
// averages (images 1-9). DWT cycle timing @ SystemCoreClock.

#include "dwt_time.h"
#include "generated/mnist_mlp_model_data.h"
#include "generated/mnist_test_images.h"
#include "stm32f446xx.h"
#include "uart.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <array>
#include <cstring>

namespace {

constexpr int kRuns = 100;
constexpr int kTensorArenaSize = 64 * 1024;
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];

using MnistOpResolver = tflite::MicroMutableOpResolver<4>;

int ArgMax10(const float* values)
{
    int best = 0;
    float max_val = values[0];
    for (int i = 1; i < 10; ++i)
    {
        if (values[i] > max_val)
        {
            max_val = values[i];
            best = i;
        }
    }
    return best;
}

}  // namespace

extern "C" void uart_debug_log(const char* message);

extern "C" int main(void)
{
    uart_init();
    dwt_time_init();
    RegisterDebugLogCallback(uart_debug_log);

    tflite::InitializeTarget();

    MnistOpResolver op_resolver;
    if (op_resolver.AddFullyConnected() != kTfLiteOk || op_resolver.AddSoftmax() != kTfLiteOk)
    {
        uart_write("ERR op resolver\r\n");
        for (;;)
        {
        }
    }

    const tflite::Model* model = tflite::GetModel(g_mnist_mlp_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        uart_write("ERR schema version\r\n");
        for (;;)
        {
        }
    }

    tflite::MicroInterpreter interpreter(model, op_resolver, g_tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk)
    {
        uart_write("ERR AllocateTensors\r\n");
        for (;;)
        {
        }
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    if (input == nullptr || output == nullptr || input->type != kTfLiteFloat32 ||
        output->type != kTfLiteFloat32)
    {
        uart_write("ERR tensors\r\n");
        for (;;)
        {
        }
    }

    uart_write("\r\nTFLM NUCLEO-F446RE MNIST MLP benchmark\r\n");
    uart_printf("  backend:     tflm reference (MCU CM4)\r\n");
    uart_printf("  model bytes: %u\r\n", g_mnist_mlp_model_data_size);
    uart_printf("  images:      %d per run\r\n", kMnistBenchmarkImageCount);
    uart_printf("  runs:        %d (discard first invoke each run)\r\n", kRuns);
    uart_printf("  arena bytes: %d\r\n", kTensorArenaSize);
    uart_printf("  sysclk:      %lu Hz\r\n", static_cast<unsigned long>(SystemCoreClock));

    std::array<double, kRuns> run_averages_us{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;

        for (int i = 0; i < kMnistBenchmarkImageCount; ++i)
        {
            const MnistBenchmarkSample& sample = kMnistBenchmarkImages[i];

            if (input->bytes < kMnistBenchmarkInputSize * static_cast<int>(sizeof(float)))
            {
                uart_write("ERR input size\r\n");
                for (;;)
                {
                }
            }
            std::memcpy(input->data.f, sample.pixels, kMnistBenchmarkInputSize * sizeof(float));

            const uint32_t start_cycles = dwt_cycles();
            const TfLiteStatus invoke_status = interpreter.Invoke();
            const uint32_t elapsed_cycles = dwt_cycles() - start_cycles;

            if (invoke_status != kTfLiteOk)
            {
                uart_printf("ERR invoke run %d image %d\r\n", run + 1, i);
                for (;;)
                {
                }
            }

            const double elapsed_us = dwt_cycles_to_us(elapsed_cycles);
            if (i > 0)
            {
                run_total_us += elapsed_us;
                ++counted;
            }

            if (run == kRuns - 1)
            {
                const int predicted = ArgMax10(output->data.f);
                if (predicted == sample.label)
                {
                    ++correct;
                }
            }
        }

        run_averages_us[static_cast<size_t>(run)] =
            run_total_us / static_cast<double>(counted);
    }

    double mean_us = 0.0;
    for (int i = 0; i < kRuns; ++i)
    {
        mean_us += run_averages_us[static_cast<size_t>(i)];
    }
    mean_us /= static_cast<double>(kRuns);

    uart_printf("  accuracy:    %d/%d on final run\r\n", correct, kMnistBenchmarkImageCount);
    uart_write("\r\nTFLM MNIST mlp benchmark summary (reference)\r\n");
    uart_write("  method:      100 runs x 10 images, discard first invoke each run\r\n");
    uart_write("  per-run avg: avg of images 1-9 (us)\r\n\r\n");
    uart_printf("  mean:   %8.3f us (%6.3f ms)\r\n", mean_us, mean_us / 1000.0);
    uart_printf(
        "BENCHMARK_SUMMARY runtime=tflm model=mlp backend=reference mean_us=%.3f runs=%d\r\n",
        mean_us,
        kRuns);
    uart_write("\r\nDONE\r\n");

    for (;;)
    {
    }
}
