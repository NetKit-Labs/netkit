# ESP-NN int8 sources for conv / pool / FC used by netkit (mirrors cmsis_nn.mk).
# Default: ANSI + portable opt C (host smoke / ESP32 / C3 / C6).
# ESP32-S3 / ESP32-P4 assembly is added when NETKIT_ARCH selects those chips
# and NETKIT_HOST_SMOKE is not set.
ESP_NN_DIR ?= third_party/ESP-NN

ESP_NN_CFLAGS = -std=c11 -O2 \
	-I$(ESP_NN_DIR)/include \
	-I$(ESP_NN_DIR)/src/common \
	$(NETKIT_ARCH_CFLAGS)

# Portable C sources (always).
ESP_NN_SOURCES = \
	$(ESP_NN_DIR)/src/activation_functions/esp_nn_relu_ansi.c \
	$(ESP_NN_DIR)/src/activation_functions/esp_nn_hard_swish_ansi.c \
	$(ESP_NN_DIR)/src/common/esp_nn_mean_ansi.c \
	$(ESP_NN_DIR)/src/basic_math/esp_nn_add_ansi.c \
	$(ESP_NN_DIR)/src/basic_math/esp_nn_mul_ansi.c \
	$(ESP_NN_DIR)/src/convolution/esp_nn_conv_ansi.c \
	$(ESP_NN_DIR)/src/convolution/esp_nn_conv_opt.c \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_ansi.c \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_opt.c \
	$(ESP_NN_DIR)/src/fully_connected/esp_nn_fully_connected_ansi.c \
	$(ESP_NN_DIR)/src/softmax/esp_nn_softmax_ansi.c \
	$(ESP_NN_DIR)/src/softmax/esp_nn_softmax_opt.c \
	$(ESP_NN_DIR)/src/logistic/esp_nn_logistic_ansi.c \
	$(ESP_NN_DIR)/src/pooling/esp_nn_avg_pool_ansi.c \
	$(ESP_NN_DIR)/src/pooling/esp_nn_max_pool_ansi.c

# Chip-tuned assembly/C (cross-compile only; skipped for host smoke).
ifneq ($(NETKIT_HOST_SMOKE),1)
  ifeq ($(NETKIT_ARCH_IS_ESP32S3),1)
    ESP_NN_SOURCES += \
	$(ESP_NN_DIR)/src/common/esp_nn_common_functions_esp32s3.S \
	$(ESP_NN_DIR)/src/common/esp_nn_dot_s8_esp32s3.S \
	$(ESP_NN_DIR)/src/common/esp_nn_multiply_by_quantized_mult_esp32s3.S \
	$(ESP_NN_DIR)/src/common/esp_nn_multiply_by_quantized_mult_ver1_esp32s3.S \
	$(ESP_NN_DIR)/src/activation_functions/esp_nn_relu_s8_esp32s3.S \
	$(ESP_NN_DIR)/src/activation_functions/esp_nn_hard_swish_s8_esp32s3.c \
	$(ESP_NN_DIR)/src/common/esp_nn_mean_s8_esp32s3.c \
	$(ESP_NN_DIR)/src/basic_math/esp_nn_add_s8_esp32s3.S \
	$(ESP_NN_DIR)/src/basic_math/esp_nn_mul_s8_esp32s3.S \
	$(ESP_NN_DIR)/src/basic_math/esp_nn_mul_broadcast_s8_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_conv_esp32s3.c \
	$(ESP_NN_DIR)/src/convolution/esp_nn_conv_s8_1x1_esp32s3.c \
	$(ESP_NN_DIR)/src/convolution/esp_nn_conv_s8_3x3_opt_esp32s3.c \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_s8_esp32s3.c \
	$(ESP_NN_DIR)/src/convolution/esp_nn_conv_s16_mult8_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_conv_s8_mult8_1x1_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_conv_s16_mult4_1x1_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_conv_s8_filter_aligned_input_padded_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_s16_mult1_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_s16_mult1_3x3_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_s16_mult1_3x3_no_pad_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_s16_mult8_3x3_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_s16_mult4_esp32s3.S \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_s16_mult8_esp32s3.S \
	$(ESP_NN_DIR)/src/fully_connected/esp_nn_fully_connected_esp32s3.c \
	$(ESP_NN_DIR)/src/fully_connected/esp_nn_fc_s8_mac16_esp32s3.S \
	$(ESP_NN_DIR)/src/fully_connected/esp_nn_fully_connected_s8_esp32s3.S \
	$(ESP_NN_DIR)/src/fully_connected/esp_nn_fully_connected_per_ch_s8_esp32s3.S \
	$(ESP_NN_DIR)/src/pooling/esp_nn_max_pool_s8_esp32s3.S \
	$(ESP_NN_DIR)/src/pooling/esp_nn_avg_pool_s8_esp32s3.c \
	$(ESP_NN_DIR)/src/pooling/esp_nn_avg_pool_s8_esp32s3.S \
	$(ESP_NN_DIR)/src/softmax/esp_nn_softmax_s8_esp32s3.c
  else ifeq ($(NETKIT_ARCH_IS_ESP32P4),1)
    ESP_NN_SOURCES += \
	$(ESP_NN_DIR)/src/common/esp_nn_mean_s8_esp32p4.c \
	$(ESP_NN_DIR)/src/common/esp_nn_multiply_by_quantized_mult_esp32p4.S \
	$(ESP_NN_DIR)/src/activation_functions/esp_nn_hard_swish_s8_esp32p4.c \
	$(ESP_NN_DIR)/src/activation_functions/esp_nn_relu_s8_esp32p4.c \
	$(ESP_NN_DIR)/src/basic_math/esp_nn_add_s8_esp32p4.c \
	$(ESP_NN_DIR)/src/basic_math/esp_nn_mul_s8_esp32p4.c \
	$(ESP_NN_DIR)/src/convolution/esp_nn_conv_esp32p4.c \
	$(ESP_NN_DIR)/src/convolution/esp_nn_depthwise_conv_esp32p4.c \
	$(ESP_NN_DIR)/src/fully_connected/esp_nn_fully_connected_s8_esp32p4.c \
	$(ESP_NN_DIR)/src/pooling/esp_nn_avg_pool_s8_esp32p4.c \
	$(ESP_NN_DIR)/src/pooling/esp_nn_max_pool_s8_esp32p4.c \
	$(ESP_NN_DIR)/src/softmax/esp_nn_softmax_s8_esp32p4.c
  endif
endif

ESP_NN_OBJECTS = $(ESP_NN_SOURCES:$(ESP_NN_DIR)/%.c=build/esp_nn/%.o)
ESP_NN_OBJECTS := $(ESP_NN_OBJECTS:$(ESP_NN_DIR)/%.S=build/esp_nn/%.o)

build/esp_nn/%.o: $(ESP_NN_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(ESP_NN_CFLAGS) -c $< -o $@

build/esp_nn/%.o: $(ESP_NN_DIR)/%.S
	@mkdir -p $(dir $@)
	$(CC) $(ESP_NN_CFLAGS) -c $< -o $@
