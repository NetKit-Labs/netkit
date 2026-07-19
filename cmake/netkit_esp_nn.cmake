# Minimal ESP-NN integration for netkit (mirrors third_party/esp_nn.mk / netkit_cmsis.cmake).

function(netkit_add_esp_nn target)
    if(NOT NETKIT_ESP_NN)
        return()
    endif()
    if(NOT NETKIT_TARGET STREQUAL "mcu_esp")
        message(WARNING
            "NETKIT_ESP_NN=ON ignored — requires NETKIT_TARGET=mcu_esp; using reference kernels")
        return()
    endif()

    set(ESP_NN_DIR "${CMAKE_SOURCE_DIR}/third_party/ESP-NN")
    if(NOT EXISTS "${ESP_NN_DIR}/include/esp_nn.h")
        message(FATAL_ERROR "NETKIT_ESP_NN=ON requires ESP-NN at ${ESP_NN_DIR} — run ./tools/fetch_esp_nn.sh")
    endif()

    set(ESP_NN_SOURCES
        src/activation_functions/esp_nn_relu_ansi.c
        src/activation_functions/esp_nn_hard_swish_ansi.c
        src/common/esp_nn_mean_ansi.c
        src/basic_math/esp_nn_add_ansi.c
        src/basic_math/esp_nn_mul_ansi.c
        src/convolution/esp_nn_conv_ansi.c
        src/convolution/esp_nn_conv_opt.c
        src/convolution/esp_nn_depthwise_conv_ansi.c
        src/convolution/esp_nn_depthwise_conv_opt.c
        src/fully_connected/esp_nn_fully_connected_ansi.c
        src/softmax/esp_nn_softmax_ansi.c
        src/softmax/esp_nn_softmax_opt.c
        src/logistic/esp_nn_logistic_ansi.c
        src/pooling/esp_nn_avg_pool_ansi.c
        src/pooling/esp_nn_max_pool_ansi.c
    )

    set(_esp_objects "")
    foreach(src ${ESP_NN_SOURCES})
        list(APPEND _esp_objects "${ESP_NN_DIR}/${src}")
    endforeach()

    add_library(netkit_esp_nn OBJECT ${_esp_objects})
    target_include_directories(netkit_esp_nn PUBLIC
        "${ESP_NN_DIR}/include"
        "${ESP_NN_DIR}/src/common")
    target_compile_options(netkit_esp_nn PRIVATE -std=c11 -O2 -Wno-unused-function)

    # Chip tags from NETKIT_ARCH (host smoke uses ANSI without CONFIG_NN_OPTIMIZED).
    if(NETKIT_ENV_ESP_DEFINES AND NOT NETKIT_HOST_SMOKE)
        foreach(def ${NETKIT_ENV_ESP_DEFINES})
            target_compile_definitions(netkit_esp_nn PRIVATE ${def})
        endforeach()
    endif()

    target_sources(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/esp_nn_backend.cpp
        $<TARGET_OBJECTS:netkit_esp_nn>)
    target_include_directories(${target} PUBLIC
        "${ESP_NN_DIR}/include"
        "${ESP_NN_DIR}/src/common")
    target_compile_definitions(${target} PUBLIC NETKIT_USE_ESP_NN=1)
endfunction()
