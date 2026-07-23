#pragma once

#include "netkit_config.h"

#if defined(NETKIT_TARGET_MCU_ESP) && NETKIT_TARGET_MCU_ESP
// Espressif MCU: EspNnQuant when ESP-NN is on; same TU provides Try*/Finalize* stubs when off.
#include "esp_nn_quant.hpp"
namespace ActiveQuant = EspNnQuant;
#elif defined(NETKIT_USE_NMSIS_NN) && NETKIT_USE_NMSIS_NN && NETKIT_NMSIS_NN_ALLOWED
#include "nmsis_nn_quant.hpp"
namespace ActiveQuant = NmsisNnQuant;
#elif defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
#include "cmsis_nn_quant.hpp"
namespace ActiveQuant = CmsisNnQuant;
#else
// No MCU NN accel — CmsisNnQuant stubs: Try* returns false, Finalize* is a no-op.
#include "cmsis_nn_quant.hpp"
namespace ActiveQuant = CmsisNnQuant;
#endif
