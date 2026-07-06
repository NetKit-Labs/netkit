#pragma once

#include <stdint.h>

/* Minimal STM32F446 register definitions for USART2 bring-up and DWT timing. */

#define PERIPH_BASE           0x40000000u
#define AHB1PERIPH_BASE       (PERIPH_BASE + 0x00020000u)
#define APB1PERIPH_BASE       PERIPH_BASE

#define RCC_BASE              (AHB1PERIPH_BASE + 0x3800u)
#define GPIOA_BASE            (AHB1PERIPH_BASE + 0x0000u)
#define GPIOD_BASE            (AHB1PERIPH_BASE + 0x0C00u)
#define USART2_BASE           (APB1PERIPH_BASE + 0x4400u)

#define RCC                 ((RCC_TypeDef*)RCC_BASE)
#define GPIOA               ((GPIO_TypeDef*)GPIOA_BASE)
#define GPIOD               ((GPIO_TypeDef*)GPIOD_BASE)
#define USART2              ((USART_TypeDef*)USART2_BASE)

#define SCS_BASE            0xE000E000u
#define SCB_BASE            (SCS_BASE + 0x0D00u)
#define DWT_BASE            0xE0001000u
#define CoreDebug_BASE      0xE000EDF0u

#define SCB                 ((SCB_TypeDef*)SCB_BASE)
#define DWT                 ((DWT_TypeDef*)DWT_BASE)
#define CoreDebug           ((CoreDebug_TypeDef*)CoreDebug_BASE)

typedef struct
{
    volatile uint32_t CR;
    volatile uint32_t PLLCFGR;
    volatile uint32_t CFGR;
    volatile uint32_t CIR;
    volatile uint32_t AHB1RSTR;
    volatile uint32_t AHB2RSTR;
    volatile uint32_t AHB3RSTR;
    volatile uint32_t RESERVED0;
    volatile uint32_t APB1RSTR;
    volatile uint32_t APB2RSTR;
    volatile uint32_t RESERVED_AHB[2];
    volatile uint32_t AHB1ENR;
    volatile uint32_t AHB2ENR;
    volatile uint32_t AHB3ENR;
    volatile uint32_t RESERVED_APB;
    volatile uint32_t APB1ENR;
    volatile uint32_t APB2ENR;
} RCC_TypeDef;

typedef struct
{
    volatile uint32_t MODER;
    volatile uint32_t OTYPER;
    volatile uint32_t OSPEEDR;
    volatile uint32_t PUPDR;
    volatile uint32_t IDR;
    volatile uint32_t ODR;
    volatile uint32_t BSRR;
    volatile uint32_t LCKR;
    volatile uint32_t AFR[2];
} GPIO_TypeDef;

typedef struct
{
    volatile uint32_t SR;
    volatile uint32_t DR;
    volatile uint32_t BRR;
    volatile uint32_t CR1;
    volatile uint32_t CR2;
    volatile uint32_t CR3;
    volatile uint32_t GTPR;
} USART_TypeDef;

typedef struct
{
    volatile uint32_t RESERVED[34];
    volatile uint32_t CPACR;
} SCB_TypeDef;

typedef struct
{
    volatile uint32_t CTRL;
    volatile uint32_t CYCCNT;
    volatile uint32_t CPICNT;
    volatile uint32_t EXCCNT;
    volatile uint32_t SLEEPCNT;
    volatile uint32_t LSUCNT;
    volatile uint32_t FOLDCNT;
    volatile uint32_t PCSR;
} DWT_TypeDef;

typedef struct
{
    volatile uint32_t RESERVED[3];
    volatile uint32_t DEMCR;
} CoreDebug_TypeDef;

#define RCC_AHB1ENR_GPIOAEN   (1u << 0)
#define RCC_AHB1ENR_GPIODEN   (1u << 3)
#define RCC_APB1ENR_USART2EN  (1u << 17)

#define USART_SR_TXE          (1u << 7)
#define USART_SR_TC           (1u << 6)
#define USART_CR1_UE          (1u << 13)
#define USART_CR1_TE          (1u << 3)

#define CoreDebug_DEMCR_TRCENA_Msk (1u << 24)
#define DWT_CTRL_CYCCNTENA_Msk     (1u << 0)

extern uint32_t SystemCoreClock;
/* USART2 is on APB1; kernel clock is PCLK1 (not the 2x timer multiplier). */
extern uint32_t SystemUSART2Clock;
