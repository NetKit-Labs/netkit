#include "uart.h"

#include "stm32f446xx.h"

#include <stdarg.h>
#include <stdio.h>

void uart_init(void)
{
    /* NUCLEO-F446RE ST-Link VCP: USART2 on PA2 (TX) / PA3 (RX), AF7 (SB13/SB14 on). */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    RCC->APB1ENR |= RCC_APB1ENR_USART2EN;

    GPIOA->MODER &= ~((3u << (2u * 2u)) | (3u << (3u * 2u)));
    GPIOA->MODER |= (2u << (2u * 2u)) | (2u << (3u * 2u));
    GPIOA->AFR[0] &= ~((0xFu << (2u * 4u)) | (0xFu << (3u * 4u)));
    GPIOA->AFR[0] |= (7u << (2u * 4u)) | (7u << (3u * 4u));
    GPIOA->OSPEEDR |= (3u << (2u * 2u)) | (3u << (3u * 2u));

    USART2->CR1 = 0u;
    USART2->BRR = (SystemUSART2Clock + 57600u) / 115200u;
    USART2->CR1 = USART_CR1_TE | USART_CR1_UE;
}

static void uart_putc(char ch)
{
    while ((USART2->SR & USART_SR_TXE) == 0u)
    {
    }
    USART2->DR = (uint32_t)(unsigned char)ch;
}

void uart_write(const char* text)
{
    if (!text)
    {
        return;
    }
    while (*text)
    {
        if (*text == '\n')
        {
            uart_putc('\r');
        }
        uart_putc(*text++);
    }
}

void uart_printf(const char* fmt, ...)
{
    char buffer[192];
    va_list args;
    va_start(args, fmt);
    const int n = vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    if (n > 0)
    {
        uart_write(buffer);
    }
}
