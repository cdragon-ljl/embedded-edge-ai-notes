# 嵌入式知识体系 · #14 · STM32 时钟树完全解析：从 HSE 到 SYSCLK

---

刚入门的嵌入式开发者，最容易踩的坑之一就是**时钟配置**。

明明代码逻辑没问题，UART 乱码、定时器不准、PWM 频率对不上——追根溯源，十有八九是时钟树没配明白。

STM32 的时钟树极其灵活，但也因此复杂。HSE、HSI、PLL、AHB、APB1、APB2……这一大堆缩写就像一团乱麻。今天我们就把它一根一根理清楚。

---

## 一、时钟源四兄弟

STM32 内部可以接入 4 种独立的时钟源，各有各的脾气：

| 时钟源 | 全称 | 典型频率 | 精度 | 用途 |
|:------|:-----|:--------:|:----:|:-----|
| **HSI** | High-Speed Internal | 8/16 MHz | ±1% | 上电默认、低成本方案 |
| **HSE** | High-Speed External | 4~26 MHz | ±50 ppm | 高精度系统时钟 |
| **LSI** | Low-Speed Internal | 32 kHz | ±5% | 独立看门狗（IWDG）、RTC |
| **LSE** | Low-Speed External | 32.768 kHz | ±20 ppm | RTC 精准走时 |

### HSI — 即插即用，但精度堪忧

HSI 是芯片内部自带的 RC 振荡器，上电即启动。STM32 复位后默认使用 HSI，所以你写个 `while(1)` 点灯不需要配时钟也能跑。

但 HSI 的精度只有 ±1%，对于 UART 通信来说已经临界了。如果你用 HSI 跑 115200 波特率，连续发几百个字节后可能出现帧错误——收发双方的累积误差超过了容限。

### HSE — 高精度的底气

HSE 需要外接晶振（通常是 8 MHz 或 25 MHz），精度可达 ±50 ppm（0.005%）。需要让 PLL 倍频到上百 MHz 跑系统时钟，就必须用 HSE。

使用 HSE 时注意两点：

1. **起振时间**：晶振从加电到稳定输出需几百微秒到几毫秒，配置 PLL 前必须等待 HSE 就绪
2. **负载电容**：晶振两端的匹配电容（典型 12~22 pF）影响起振可靠性，不能随便

### LSI 与 LSE — 低速的世界

LSI 主要喂给**独立看门狗（IWDG）**——即使在睡眠模式下 IWDG 也要跑，只有 LSI 能在低功耗状态持续工作。

LSE 则是 RTC 的黄金搭档。32.768 kHz = 2¹⁵ Hz，分频 15 次刚好得到 1 秒，所以 RTC 用它走时最省事。

---

## 二、PLL：把频率"吹"上去

系统要跑到上百 MHz，光靠 8 MHz HSE 是不够的。**锁相环（PLL）**的作用就是把输入频率乘以一个系数。

### PLL 配置三件套

STM32 的 PLL 一般由 3 个参数决定输出频率：

```
PLL_CLK = (输入频率 / PLL_M) × PLL_N / PLL_P
```

以经典的 STM32F407 + 8 MHz HSE → 168 MHz SYSCLK 为例：

| 参数 | 值 | 含义 |
|:---|:--:|:-----|
| PLL_M | 8 | 分频，8 MHz ÷ 8 = 1 MHz |
| PLL_N | 336 | 倍频，1 MHz × 336 = 336 MHz |
| PLL_P | 2 | 分频，336 MHz ÷ 2 = 168 MHz ✅ |

所以 `PLL_CLK = (8 / 8) × 336 / 2 = 168 MHz`。

> **注意**：PLL_N 的输入频率（即 PLL_M 之后的频率）必须在 **1~2 MHz** 之间（F4 系列），否则 PLL 失锁。这是很多开发者配不通的原因——把 M 设大了或小了。

### CubeMX 的自动计算 vs 手动配置

用 CubeMX 时，你只需在"Clock Configuration"页面填入目标频率，软件自动算系数。但你要**看懂它算出来的值**，才能判断是否合理。

手动配置的代码长这样：

```c
RCC_OscInitTypeDef osc = {0};
RCC_ClkInitTypeDef clk = {0};

/* 1. 开启 HSE，配置 PLL */
osc.OscillatorType = RCC_OSCILLATORTYPE_HSE;
osc.HSEState = RCC_HSE_ON;
osc.PLL.PLLState = RCC_PLL_ON;
osc.PLL.PLLSource = RCC_PLLSOURCE_HSE;
osc.PLL.PLLM = 8;
osc.PLL.PLLN = 336;
osc.PLL.PLLP = RCC_PLLP_DIV2;
osc.PLL.PLLQ = 7;  /* USB/SDIO 需要 48 MHz：336/7 = 48 */
HAL_RCC_OscConfig(&osc);

/* 2. 选择 PLL 输出作为系统时钟 */
clk.ClockType = RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK
              | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
clk.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
clk.AHBCLKDivider = RCC_SYSCLK_DIV1;     /* HCLK = SYSCLK = 168 MHz */
clk.APB1CLKDivider = RCC_HCLK_DIV4;      /* APB1 = 42 MHz （上限 42 MHz） */
clk.APB2CLKDivider = RCC_HCLK_DIV2;      /* APB2 = 84 MHz */
HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_5);
```

PLL 还有另一个输出 `PLLQ`，专门给 USB OTG（需要 48 MHz）和 SDIO 用。忘了配它，USB 就枚举不上。

---

## 三、AHB / APB1 / APB2：总线分级的学问

系统时钟搞定后，还得通过**总线桥**分发给各个外设。

```
SYSCLK (168 MHz)
   ├── AHB 分频器 → HCLK (168 MHz)
   │    ├── 内核、DCode、SysTick
   │    ├── 存储器（Flash、SRAM）
   │    ├── DMA
   │    ├── GPIO（！）← GPIO 挂在 AHB 上
   │    └── APB1 分频器 → PCLK1 (42 MHz)
   │    │    ├── TIM2~7、USART2~5、I2C1~3、SPI2/3 ...
   │    │    └── 定时器时钟 = PCLK1 × 2（如果分频系数 ≠ 1）
   │    └── APB2 分频器 → PCLK2 (84 MHz)
   │         ├── TIM1/8、USART1、SPI1、ADC1~3 ...
   │         └── 定时器时钟 = PCLK2 × 2（同理）
   └── 其他 AHB 外设（CRC、RNG、USB OTG HS ...）
```

### 容易踩的坑

**坑 1：APB 定时器时钟翻倍规则**

APB 分频系数不为 1 时，定时器的实际时钟是 PCLK 的**两倍**。F407 上 APB1 分频为 4，所以 TIM2~7 的时钟是 `42 × 2 = 84 MHz`。如果你不知道这条规则，配定时器重装值时频率永远差一倍。

**坑 2：APB1 和 APB2 频率上限不同**

- APB1：最高 **42 MHz**（F407）
- APB2：最高 **84 MHz**

把 USART1（APB2 外设）的频率算成 42 MHz 会导致波特率异常，但代码逻辑完全正确——最难排查。

**坑 3：Flash 等待周期**

SYSCLK 超过一定频率后，必须增加 Flash 等待周期（`FLASH_LATENCY`）。F407 的参考关系：

| SYSCLK (MHz) | 等待周期 |
|:-----------:|:--------:|
| ≤ 30 | 0 |
| ≤ 60 | 1 |
| ≤ 90 | 2 |
| ≤ 120 | 3 |
| ≤ 150 | 4 |
| ≤ 168 | 5 |

等待周期不够，**直接 HardFault**。CubeMX 生成的代码会自动设对，但手动配时经常漏。

---

## 四、CubeMX 配置 vs 手动寄存器

### CubeMX 法（推荐起步）

1. 在 "Pinout & Configuration" → "RCC" 选中 HSE（Crystal/Ceramic Resonator）
2. 切到 "Clock Configuration" 页
3. 在 HCLK 框输入 168，回车，软件自动算好所有分频系数
4. 检查左下角红色"×"全部消失，变成绿色"✓"

### 手动寄存器法（需要透彻理解）

直接操作 RCC 寄存器可以实现更精细的控制，比如动态切换时钟源来省电：

```c
/* 切换到 HSI 运行，然后关闭 HSE 省电 */
RCC->CR &= ~RCC_CR_HSION;          /* 确保 HSI 已开启（默认开启） */
RCC->CFGR &= ~RCC_CFGR_SW;         /* 清除 SW 位 */
RCC->CFGR |= RCC_CFGR_SW_HSI;      /* 切换系统时钟到 HSI */
while (!(RCC->CFGR & RCC_CFGR_SWS_HSI));  /* 等待切换完成 */

RCC->CR &= ~RCC_CR_HSEON;          /* 关闭 HSE */
```

这段代码在低功耗场景非常实用：满速跑完任务后切到 HSI，关 HSE 省电。

---

## 五、常见时钟故障排查清单

遇到"外设不工作"的玄学问题，按这个清单查一遍：

1. **确认 HSE 起振**：检查 `RCC->CR` 的 `HSERDY` 标志位
2. **确认 PLL 锁定**：检查 `RCC->CR` 的 `PLLRDY` 标志位
3. **确认系统时钟源切换**：检查 `RCC->CFGR` 的 `SWS` 位是否为 `10`（PLL）
4. **确认 APB 分频系数**：`RCC->CFGR` 的 `PPRE1/PPRE2` 位域
5. **计算定时器时钟**：`TIM_CLK = PCLK × 2`（当 APB 分频 ≠ 1 时）
6. **检查 Flash 等待周期**：`FLASH->ACR` 的 `LATENCY` 位域
7. **USB 检查 PLLQ**：PLLQ 输出是否被正确配置为 48 MHz

---

## 总结

STM32 的时钟树就像城市的供水系统：水源（时钟源）→ 加压站（PLL）→ 主管道（AHB）→ 支线（APB1/2）→ 每家每户（外设）。任何一个环节的阀门没开对，水就流不到目的地。

理解时钟树的层级关系，比记住某个系列的具体参数更重要——因为换了芯片系列（F1→F4→H7），参数变但结构不变。

下次遇到"UART乱码"或者"PWM频率不对"，先查时钟树，而不是先怀疑代码逻辑。这不是玄学，这是基本功。

> 🏷️ STM32 时钟树 HSE HSI PLL AHB APB1 APB2 SYSCLK 嵌入式 C语言
