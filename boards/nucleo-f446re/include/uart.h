#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void uart_init(void);
void uart_write(const char* text);
void uart_printf(const char* fmt, ...);

#ifdef __cplusplus
}
#endif
