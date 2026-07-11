# Minimal CMSIS-NN integration for netkit (mirrors third_party/cmsis_nn.mk).
# CMSIS-DSP is not used as a netkit backend.

function(netkit_apply_cmsis_target_flags cmsis_target)
    if(HOST)
        target_compile_definitions(${cmsis_target} PRIVATE __GNUC_PYTHON__)
    endif()

    if(NETKIT_ENV_ARM_MATH_NEON)
        target_compile_definitions(${cmsis_target} PRIVATE ARM_MATH_NEON)
    endif()

    if(NETKIT_ENV_ARM_MATH_LOOPUNROLL)
        target_compile_definitions(${cmsis_target} PRIVATE ARM_MATH_LOOPUNROLL)
    endif()

    foreach(def ${NETKIT_ENV_ARM_MATH_DEFINES})
        target_compile_definitions(${cmsis_target} PRIVATE ${def})
    endforeach()

    if(CMSISCORE AND EXISTS "${CMSISCORE}")
        target_include_directories(${cmsis_target} PRIVATE "${CMSISCORE}")
    endif()

    # Helium (MVE) builds may need relaxed vector conversions (matches CMSIS-NN CMake).
    if(NETKIT_ENV_ARM_MATH_MVE)
        target_compile_options(${cmsis_target} PRIVATE
            $<$<C_COMPILER_ID:GNU>:-flax-vector-conversions>
            $<$<C_COMPILER_ID:ARMClang>:-flax-vector-conversions=integer>)
    endif()
endfunction()

function(netkit_add_cmsis_nn target)
    if(NOT NETKIT_CMSIS_NN)
        return()
    endif()
    if(NOT NETKIT_TARGET STREQUAL "mcu_arm")
        message(WARNING
            "NETKIT_CMSIS_NN=ON ignored — requires NETKIT_TARGET=mcu_arm and Cortex-M NETKIT_ARCH; using reference kernels")
        return()
    endif()
    # Mirror Make: CMSIS-NN needs a Cortex-M ARM_MATH_* define from NETKIT_ARCH.
    set(_netkit_cmsis_nn_cortex_m FALSE)
    foreach(_def ${NETKIT_ENV_ARM_MATH_DEFINES})
        if(_def MATCHES "^ARM_MATH_(CM0|CM0PLUS|CM3|CM4|CM7|ARMV8MBL|ARMV8MML|M55|M85)$")
            set(_netkit_cmsis_nn_cortex_m TRUE)
        endif()
    endforeach()
    if(NOT _netkit_cmsis_nn_cortex_m)
        message(WARNING
            "NETKIT_CMSIS_NN=ON ignored — set NETKIT_ARCH=CM4|M33|... (Cortex-M); using reference kernels")
        return()
    endif()

    set(CMSIS_NN_DIR "${CMAKE_SOURCE_DIR}/third_party/CMSIS-NN")
    if(NOT EXISTS "${CMSIS_NN_DIR}/Include/arm_nnfunctions.h")
        message(FATAL_ERROR "NETKIT_CMSIS_NN=ON requires CMSIS-NN at ${CMSIS_NN_DIR} — run ./tools/fetch_cmsis_nn.sh")
    endif()

    set(CMSIS_NN_SOURCES
        Source/ConvolutionFunctions/arm_convolve_f32.c
        Source/ConvolutionFunctions/arm_convolve_1_x_n_f32.c
        Source/ConvolutionFunctions/arm_convolve_1x1_f32.c
        Source/ConvolutionFunctions/arm_depthwise_conv_f32.c
        Source/NNSupportFunctions/arm_get_buffer_size_f32.c
        Source/NNSupportFunctions/arm_nn_pack_conv_patch_f32.c
        Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_f32.c
        Source/NNSupportFunctions/arm_nn_mat_mult_nt_n_packed_f32.c
        Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_f32.c
        Source/NNSupportFunctions/arm_nn_depthwise_conv3x3_f32.c
        Source/NNSupportFunctions/arm_nn_depthwise_conv1d_k3_f32.c
        Source/NNSupportFunctions/arm_nn_conv1d_k3_f32.c
        Source/NNSupportFunctions/arm_nn_conv1d_k3_packed_f32.c
        Source/NNSupportFunctions/arm_nn_conv1d_k5_f32.c
        Source/NNSupportFunctions/arm_nn_conv1d_k5_packed_f32.c
        Source/NNSupportFunctions/arm_nntables_flt.c
        Source/FullyConnectedFunctions/arm_fully_connected_f32.c
        Source/PoolingFunctions/arm_max_pool_f32.c
        Source/PoolingFunctions/arm_avg_pool_f32.c
        Source/NNSupportFunctions/arm_nn_maxpool1d_f32.c
        Source/NNSupportFunctions/arm_batch_norm_f32.c
        Source/ActivationFunctions/arm_nn_activation_f32.c
        Source/SoftmaxFunctions/arm_softmax_f32.c
        Source/BasicMathFunctions/arm_elementwise_add_f32.c
    )

    foreach(src ${CMSIS_NN_SOURCES})
        list(APPEND cmsis_nn_objects "${CMSIS_NN_DIR}/${src}")
    endforeach()

    add_library(netkit_cmsis_nn OBJECT ${cmsis_nn_objects})
    target_include_directories(netkit_cmsis_nn PUBLIC "${CMSIS_NN_DIR}/Include")
    target_compile_definitions(netkit_cmsis_nn PRIVATE ARM_NN_ENABLE_F32=1)
    target_compile_options(netkit_cmsis_nn PRIVATE -std=c11 -O2)
    netkit_apply_cmsis_target_flags(netkit_cmsis_nn)

    if(NEON AND NOT NETKIT_ENV_ARM_MATH_NEON)
        message(STATUS "NEON enabled; CMSIS-NN uses architecture NEON paths where available")
    endif()

    target_sources(${target} PRIVATE src/cmsis_nn_backend.cpp)
    target_compile_definitions(${target} PUBLIC NETKIT_USE_CMSIS_NN=1 ARM_NN_ENABLE_F32=1)
    target_include_directories(${target} PUBLIC "${CMSIS_NN_DIR}/Include")
    target_link_libraries(${target} PUBLIC netkit_cmsis_nn)
endfunction()

function(netkit_add_cmsis_softmax_s8 target)
    set(CMSIS_NN_DIR "${CMAKE_SOURCE_DIR}/third_party/CMSIS-NN")
    set(CMSIS_SOFTMAX_SOURCE "${CMSIS_NN_DIR}/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c")
    if(NOT EXISTS "${CMSIS_SOFTMAX_SOURCE}")
        return()
    endif()

    add_library(netkit_cmsis_softmax_s8 OBJECT "${CMSIS_SOFTMAX_SOURCE}")
    target_include_directories(netkit_cmsis_softmax_s8 PUBLIC "${CMSIS_NN_DIR}/Include")
    target_compile_options(netkit_cmsis_softmax_s8 PRIVATE -std=c11 -O2)

    target_compile_definitions(${target} PUBLIC NETKIT_USE_CMSIS_SOFTMAX_S8=1)
    target_include_directories(${target} PUBLIC "${CMSIS_NN_DIR}/Include")
    target_link_libraries(${target} PUBLIC netkit_cmsis_softmax_s8)
endfunction()
