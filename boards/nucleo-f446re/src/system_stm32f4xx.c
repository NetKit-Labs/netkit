#include "stm32f446xx.h"

uint32_t SystemCoreClock = 180000000u;
uint32_t SystemUSART2Clock = 45000000u;

static volatile uint32_t* const kFlashAcr = (volatile uint32_t*)0x40023C00u;
static volatile uint32_t* const kPwrCr = (volatile uint32_t*)0x40007000u;

static const uint32_t kCrHseOn = 1u << 16;
static const uint32_t kCrHseRdy = 1u << 17;
static const uint32_t kCrPllOn = 1u << 24;
static const uint32_t kCrPllRdy = 1u << 25;

static const uint32_t kCfgrSwPll = 2u << 0;
static const uint32_t kCfgrSwsPll = 2u << 2;
static const uint32_t kCfgrPpre1Div4 = 5u << 10;
static const uint32_t kCfgrPpre2Div2 = 4u << 13;

static const uint32_t kApb1EnrPwrEn = 1u << 28;

static int WaitReady(volatile uint32_t* reg, uint32_t mask, uint32_t spins)
{
    while (spins-- > 0u)
    {
        if ((*reg & mask) != 0u)
        {
            return 1;
        }
    }
    return 0;
}

static void ConfigurePllFromHse(void)
{
    RCC->PLLCFGR = (7u << 24) | (1u << 22) | (360u << 6) | 8u;
}

static void ConfigurePllFromHsi(void)
{
    RCC->PLLCFGR = (7u << 24) | (0u << 22) | (180u << 6) | 8u;
}

static void EnablePllAndSwitchSysclk(void)
{
    RCC->CR |= kCrPllOn;
    while ((RCC->CR & kCrPllRdy) == 0u)
    {
    }

    RCC->CFGR = (RCC->CFGR & ~((7u << 10) | (7u << 13) | (3u << 0))) | kCfgrPpre1Div4 |
                kCfgrPpre2Div2 | kCfgrSwPll;
    while ((RCC->CFGR & kCfgrSwsPll) != kCfgrSwsPll)
    {
    }
}

void SystemInit(void)
{
    /* Enable CP10/CP11 (FPU) — required for -mfloat-abi=hard / CMSIS-DSP. */
    SCB->CPACR |= (3u << (10u * 2u)) | (3u << (11u * 2u));

    /* Scale 1 regulator + flash wait states before raising SYSCLK to 180 MHz. */
    RCC->APB1ENR |= kApb1EnrPwrEn;
    *kPwrCr = (*kPwrCr & ~(3u << 14)) | (1u << 14);
    *kFlashAcr = (*kFlashAcr & ~0xFu) | 5u;

    RCC->CR |= kCrHseOn;
    if (WaitReady(&RCC->CR, kCrHseRdy, 0x5000u))
    {
        ConfigurePllFromHse();
    }
    else
    {
        ConfigurePllFromHsi();
    }

    EnablePllAndSwitchSysclk();

    SystemCoreClock = 180000000u;
    /* USART2 kernel clock = PCLK1 (APB1 = HCLK/4 at 180 MHz SYSCLK). */
    SystemUSART2Clock = 45000000u;
}
