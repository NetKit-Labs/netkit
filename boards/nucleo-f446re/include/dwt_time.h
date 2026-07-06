#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void dwt_time_init(void);
uint32_t dwt_cycles(void);
double dwt_cycles_to_us(uint32_t cycles);

#ifdef __cplusplus
}
#endif
