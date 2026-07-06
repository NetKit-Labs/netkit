#include "dwt_time.h"

#include "stm32f446xx.h"

void dwt_time_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0u;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    __asm volatile("" ::: "memory");
}

uint32_t dwt_cycles(void)
{
    return DWT->CYCCNT;
}

double dwt_cycles_to_us(uint32_t cycles)
{
    return (double)cycles * 1000000.0 / (double)SystemCoreClock;
}
