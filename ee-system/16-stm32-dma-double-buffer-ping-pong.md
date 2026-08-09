# 嵌入式知识体系 · #16 · STM32 DMA 双缓冲：让数据传输飞起来

---

CPU 跑来跑去搬数据，本质上是一种浪费。

SPI 收 1 个字节、UART 发 1000 个字节、ADC 连续采样 4096 个点——这些操作如果用 CPU 逐字处理，CPU 就被死死摁在搬运工的岗位上，做不了任何有"算力"的事情。

**DMA（Direct Memory Access）** 就是专门干这个的搬运工。它不需要 CPU 参与，能在外设和内存之间、内存和内存之间自动传数据。

而当数据传输是连续不断的——比如双通道音频播放、高速 ADC 采样——**DMA 双缓冲（双缓冲又称乒乓缓冲）**就成了必备技能。

---

## 一、DMA 传输的三种模式

### 1. 单次传输（Normal）

DMA 收到一次触发，传输 N 个数据，然后停止。下次需要重新配置传输计数。

适用场景：单次 ADC 转换结果读取、一次性 SPI 通信。

### 2. 循环传输（Circular）

DMA 传输完 N 个数据后，自动从起始地址重新开始。地址指针绕圈跑，永不停。

适用场景：连续 ADC 采样（如音频采集）、持续 SPI 数据流。

### 3. 双缓冲传输（Double Buffer / Ping-Pong）

两个缓冲区交替使用。DMA 写满 buffer A 后自动切到 buffer B 写入，同时 CPU 处理 buffer A 的数据；写满 buffer B 后切回 buffer A，CPU 处理 buffer B。

**数据流从不中断**，CPU 和 DMA 各干各的，互不冲突。

---

## 二、核心机制：半满中断与完全中断

双缓冲的精髓在于**两个中断的交替触发**。

STM32 的 DMA 流（Stream）支持两个中断标志位：

| 中断标志 | 触发时机 | 含义 |
|:--------:|:--------:|:-----|
| **HTIF**（Half Transfer） | 传输到缓冲区的一半 | buffer A 满了 |
| **TCIF**（Transfer Complete） | 传输到缓冲区的末尾 | buffer B 满了 |

对于大小为 N 的双缓冲，DMA 传输 N/2 个数据时触发 HT，传完全部 N 个数据时触发 TC。然后计数自动重置，从头开始——HT 再次对应 buffer A 满，TC 再次对应 buffer B 满。

```
时间轴：                  HT            TC            HT            TC
                         ↓             ↓             ↓             ↓
双缓冲大小=16： [0 1 2 3 4 5 6 7|8 9 A B C D E F|0 1 2 3 4 5 6 7|8 9 A B C D E F]
                 └─── buffer A ──┘ └─── buffer B ──┘ └─── buffer A ──┘ └─── buffer B ──┘
CPU 处理：                       buffer A         buffer B         buffer A         buffer B
```

### 配置代码（以 STM32F4 SPI DMA 双缓冲为例）

```c
#define BUF_SIZE  256

uint16_t dma_rx_buf[BUF_SIZE];         /* DMA 硬件双缓冲 */
volatile uint8_t buf_ready = 0;        /* 标志位：哪个缓冲区就绪 */

void MX_DMA_Init(void)
{
    HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);
}

void start_double_buffer(void)
{
    /* 开启 SPI DMA 接收 */
    HAL_SPI_Receive_DMA(&hspi1, dma_rx_buf, BUF_SIZE);

    /* 使能 DMA 流双缓冲模式 */
    hdma_spi_rx.Instance->CR |= DMA_SxCR_DBM;       /* Double Buffer Mode */
    hdma_spi_rx.Instance->CR |= DMA_SxCR_CIRC;      /* Circular Mode */

    /* 设置两个缓冲区地址 */
    hdma_spi_rx.Instance->M0AR = (uint32_t)&dma_rx_buf[0];           /* buffer A */
    hdma_spi_rx.Instance->M1AR = (uint32_t)&dma_rx_buf[BUF_SIZE/2]; /* buffer B */

    /* 使能半满中断和完全中断 */
    hdma_spi_rx.Instance->CR |= DMA_SxCR_HTIE | DMA_SxCR_TCIE;
}
```

### 中断处理

```c
void DMA2_Stream0_IRQHandler(void)
{
    uint32_t flags = hdma_spi_rx.Instance->ISR;

    /* 半满中断：buffer A 已满 */
    if (flags & DMA_FLAG_HTIF0_4) {
        hdma_spi_rx.Instance->IFCR |= DMA_FLAG_HTIF0_4;  /* 清标志 */
        process_buffer(&dma_rx_buf[0], BUF_SIZE / 2);    /* 处理 A */
    }

    /* 完全中断：buffer B 已满 */
    if (flags & DMA_FLAG_TCIF0_4) {
        hdma_spi_rx.Instance->IFCR |= DMA_FLAG_TCIF0_4;  /* 清标志 */
        process_buffer(&dma_rx_buf[BUF_SIZE / 2], BUF_SIZE / 2);  /* 处理 B */
    }
}
```

这样，`process_buffer()` 总是在处理"旧数据"，而 DMA 同时在写入"新数据"。两个缓冲区交替使用，**数据流零停顿**。

---

## 三、乒乓缓冲的典型应用

### 音频 I²S 播放

I²S 连续输出音频数据到 DAC。如果用单缓冲，DMA 中断来临时必须**立即填好下一批数据**——留给 CPU 的时间只有一次 DMA 传输的时间。

假设 44.1 kHz 采样率、16 位立体声、缓冲区 512 样本。CPU 大约有 `512 / 44100 ≈ 11.6 ms` 处理时间。双缓冲把时间翻倍到 23 ms，还允许 CPU 在后台执行其他逻辑。

### 高速 ADC + 数字信号处理

用定时器触发 ADC 以 1 MHz 采样，DMA 乒乓缓冲连续搬数据。CPU 在 TC 中断中处理 buffer A 的 FFT，DMA 同时在往 buffer B 写——互不干扰。

### 串口高速接收

UART 115200 波特率下，每字节约 87 μs。不用 DMA 的话，CPU 几乎每 100 μs 就要进一次中断。用 DMA + 双缓冲，CPU 只需要在缓冲区满时批量处理一批数据，中断频率大幅降低。

---

## 四、DMA 与 Cache 一致性问题（Cortex-M7）

如果你用的是 STM32H7 或 STM32F7（Cortex-M7 内核），DMA 双缓冲会引入一个新的麻烦：**Cache 一致性**。

### 问题在哪

Cortex-M7 有 L1-Cache（数据缓存）。CPU 读内存时，优先读 Cache，而不是直接读 SRAM。

```
场景：DMA 写 SRAM buffer → CPU 通过内存地址读取
```

DMA 是**总线主设备**，它直接访问 SRAM，**不经过 Cache**。所以 DMA 把数据写进了 SRAM 的地址 0x24001000，但 CPU 可能读的是 Cache 里的旧数据——地址相同，但内容不同。

### 解决方法

**方法一：使用非缓存内存段**

STM32H7 的 SRAM 区域中，DTCM（地址 0x20000000）默认不使用 Cache。把 DMA 缓冲区放在 DTCM 即可避免一致性问题：

```c
/* 放在 DTCM 区域，默认不缓存 */
__attribute__((section(".dtcmram")))
uint16_t dma_rx_buf[BUF_SIZE];
```

**方法二：手动 Cache 维护**

如果不方便改内存段，可以在 CPU 读之前做 Cache 失效：

```c
void DMA2_Stream0_IRQHandler(void)
{
    uint32_t flags = hdma_spi_rx.Instance->ISR;

    if (flags & DMA_FLAG_HTIF0_4) {
        hdma_spi_rx.Instance->IFCR |= DMA_FLAG_HTIF0_4;

        /* 读之前使 Cache 失效，强制从 SRAM 重新加载 */
        SCB_InvalidateDCache_by_Addr(
            (uint32_t*)&dma_rx_buf[0], BUF_SIZE / 2 * sizeof(uint16_t));

        process_buffer(&dma_rx_buf[0], BUF_SIZE / 2);
    }
    /* ... TC 同样处理 ... */
}
```

**方法三：MPU 配置**

通过 MPU 将 DMA 缓冲区所在的内存区域配置为**非缓存（Non-cacheable）**。CubeMX 在 Cache 配置页提供了这个选项。

### 三个方法怎么选

| 方法 | 优点 | 缺点 |
|:----|:----|:-----|
| DTCM 段 | 零代码开销，性能最高 | DTCM 空间有限（通常 128 KB） |
| Cache 失效 | 灵活，任意内存区域可用 | 每次中断都有 ~20 周期开销 |
| MPU 配置 | 全局生效，无需每个缓冲区单独处理 | MPU 配置复杂，容易配错 |

> **黄金法则**：如果 DMA 缓冲区的数据是"生产者-消费者"模式（DMA 写，CPU 读），**必须保证 Cache 一致性**。不处理的话，读到的数据可能是错的。

---

## 五、常见 DMA 故障排查

| 症状 | 可能原因 | 检查方法 |
|:----|:--------|:--------|
| DMA 不传输 | 未使能 DMA 时钟 | 查 `RCC->AHBxENR` DMA 位 |
| 只传一次就停 | 忘记配 Circular 模式 | 查 `DMA_SxCR` 的 CIRC 位 |
| 数据错位 | 数据宽度不匹配 | 外设宽度 vs 内存宽度需一致 |
| 数据全 0 | 地址写错或未使能外设 DMA 请求 | 查外设 DMA 使能位 |
| 数据跳变/乱序 | Cache 未维护（M7） | 加 `SCB_InvalidateDCache` |
| 中断不触发 | 中断标志未清或 NVIC 未使能 | 查 NVIC->ISER 寄存器 |

---

## 总结

DMA 双缓冲是嵌入式系统中"CPU 干正事，DMA 干搬运"的最佳实践。它的核心思想很简单：**用两个缓冲区轮流装数据，让 CPU 和 DMA 并行工作，谁也不等谁**。

从 SPI 数据采集到音频 I²S 播放，从多通道 ADC 到高速串口，只要数据流是连续不断的，乒乓缓冲都是最高效的方案——但别忘了 Cache 一致性的雷。

> 🏷️ STM32 DMA 双缓冲 乒乓缓冲 Ping-Pong HTIF TCIF Cache 一致性 Cortex-M7 嵌入式
