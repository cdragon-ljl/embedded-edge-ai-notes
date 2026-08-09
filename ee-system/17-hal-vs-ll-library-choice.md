# 嵌入式知识体系 · #17 · HAL 库 vs LL 库：选择困难症的终结指南

---

"用 HAL 还是 LL？"

这是 STM32 生态里争论最久的问题之一。HAL 说"我方便"，LL 说"我快"，开发者夹在中间，不知道该站哪边。

更迷惑的是：**它们其实可以一起用**。

ST 官方提供了两套外设库，但这并不是一个"二选一"的选择题，而是一个"根据场景选工具"的决策框架。本文带你拆解两者的本质区别，并给出选型指南。

---

## 一、HAL 库：封装到极致，上手快

### 分层架构

HAL（Hardware Abstraction Layer）库把外设操作封装成了 3 层：

```
应用层
   ↓ 调用 HAL 函数（如 HAL_UART_Transmit()）
HAL 层
   ↓ 调用底层寄存器操作
外设寄存器层（LL 等价操作在这里）
```

以一次 UART 发送为例，HAL 库内部的大致路径：

```c
// 你写的
HAL_UART_Transmit(&huart1, data, len, timeout);

// 内部大致做了这些事：
// 1. 检查 huart1 状态（是否忙）
// 2. 如果忙碌，根据 timeout 等待或返回超时
// 3. 打开发送使能位
// 4. 逐字节发送（轮询模式下）
// 5. 等待发送完成
// 6. 返回 HAL_OK 或 HAL_ERROR
```

### 回调机制

HAL 库的一大特色是**中断回调函数**：

```c
// 在 stm32f4xx_hal_uart.c 的中断处理函数里
void UART_IRQHandler(UART_HandleTypeDef *huart)
{
    // ... 判断中断源 ...
    // ... 收数据到缓冲区 ...
    HAL_UART_RxCpltCallback(huart);  // ← 回调用户函数
}

// 你只需要在自己的代码里实现这个函数：
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        process_byte(huart->Instance->DR);  /* 处理收到的字节 */
    }
}
```

回调机制让你不用碰中断向量表，也不用读堆栈现场，**专注业务逻辑**。

### 状态机管理

每个 HAL 句柄（`UART_HandleTypeDef`、`SPI_HandleTypeDef` 等）内部维护了一个**状态机**：

```c
typedef struct {
    USART_TypeDef *Instance;
    UART_InitTypeDef Init;
    uint8_t *pTxBuffPtr;
    uint16_t TxXferSize;
    uint16_t TxXferCount;
    __IO HAL_UART_StateTypeDef State;   // ← 状态
    __IO uint32_t ErrorCode;
    // ... 其他字段 ...
} UART_HandleTypeDef;
```

`State` 字段可能的取值：

| 状态 | 含义 |
|:----|:------|
| `HAL_UART_STATE_RESET` | 未初始化 |
| `HAL_UART_STATE_READY` | 就绪 |
| `HAL_UART_STATE_BUSY` | 正在传输 |
| `HAL_UART_STATE_BUSY_TX` | 发送中 |
| `HAL_UART_STATE_BUSY_RX` | 接收中 |
| `HAL_UART_STATE_ERROR` | 出错 |

这套状态机制的好处是**防止重入**——如果在传输中又调用同一个外设的函数，HAL 会直接返回 `HAL_BUSY`，避免寄存器冲突。

但这也带来了性能损失：每个函数调用都要查一次状态。

---

## 二、LL 库：不绕弯子，直奔寄存器

LL（Low-Layer）库相当于**带函数壳的寄存器操作**。

对比一下两者做同一件事：

```c
// ============ HAL 方式 ============
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);

// 内部展开后：
// 1. 断言检查参数
// 2. 读取 GPIOA->BSRR 寄存器
// 3. 计算位掩码
// 4. 写入 BSRR

// ============ LL 方式 ============
LL_GPIO_SetOutputPin(GPIOA, GPIO_PIN_5);

// 内部就是一行：
// WRITE_REG(GPIOA->BSRR, GPIO_PIN_5)
```

### LL 的典型用法

```c
/* GPIO 初始化 — LL 版 */
void gpio_init_ll(void)
{
    LL_GPIO_InitTypeDef gpio = {0};

    gpio.Pin = LL_GPIO_PIN_5;
    gpio.Mode = LL_GPIO_MODE_OUTPUT;
    gpio.Speed = LL_GPIO_SPEED_FREQ_LOW;
    gpio.OutputType = LL_GPIO_OUTPUT_PUSHPULL;
    gpio.Pull = LL_GPIO_PULL_NO;
    LL_GPIO_Init(GPIOA, &gpio);
}

/* UART 发送 — LL 版 */
void uart_send_ll(uint8_t c)
{
    /* 等待上一次发送完成 */
    while (!LL_USART_IsActiveFlag_TXE(USART1));
    /* 直接写数据寄存器 */
    LL_USART_TransmitData8(USART1, c);
}
```

### LL 库的优点

1. **代码量小**：编译后机器码通常只有 HAL 版本的 1/3 ~ 1/5
2. **执行快**：没有状态检查、超时循环等额外开销
3. **可预测**：每一行代码做了什么都一目了然
4. **中断里用**：不会被状态机的 `__HAL_LOCK()` 阻塞

### LL 库的缺点

1. **没有回调机制**：中断处理函数需要自己写完整逻辑
2. **没有超时保护**：轮询发送时需要自己加 while 等待
3. **可读性差一点**：不如 HAL 函数名直观
4. **移植成本高**：换了芯片系列，LL 函数名可能有差异

---

## 三、性能对比：到底差多少？

我们测一组实际数据（STM32F407 @168 MHz，优化等级 -O2）：

| 操作 | HAL | LL | 差距 |
|:----|:---:|:--:|:----:|
| GPIO 拉高 | ~60 cycles | ~6 cycles | **10x** |
| UART 发送 1 字节（轮询） | ~80 cycles | ~12 cycles | **6.7x** |
| SPI 传输 1 字（轮询） | ~85 cycles | ~15 cycles | **5.7x** |
| 定时器配置 | ~300 cycles | ~50 cycles | **6x** |
| ADC 单次读取 | ~120 cycles | ~25 cycles | **4.8x** |

差距主要来自：
- HAL 的状态检查（`__HAL_LOCK()` 和 `State` 判断）
- 超时计算和检查（`HAL_GetTick()` 调用）
- 参数校验（`assert_param()`）

**结论**：对于普通应用（GPIO 翻转频率 < 100 kHz、UART < 1 Mbps），HAL 的性能损失可以忽略。只有在**高频翻转**或**硬实时**场景下，LL 的差异才显现出来。

---

## 四、混用 HAL 和 LL：最佳实践

ST 官方设计上就允许混用。正确的混用策略是：

### 原则

1. **初始化用 HAL**：CubeMX 生成的初始化代码质量高、配置完整，没必要自己写
2. **运行时数据通道用 LL**：GPIO 快速翻转、ISR 中的数据收发用 LL
3. **复杂流程用 HAL**：多步骤外设操作（如写 Flash、配置 DMA）用 HAL
4. **不要同时操作同一个外设的同一个资源**：HAL 的状态机和 LL 的直接寄存器操作可能冲突

### 混用示例

```c
/* 初始化（HAL 生成） */
void MX_USART1_UART_Init(void)
{
    huart1.Instance = USART1;
    huart1.Init.BaudRate = 115200;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    HAL_UART_Init(&huart1);
}

/* 中断接收（LL + HAL 混合） */
void USART1_IRQHandler(void)
{
    /* 用 LL 快速判断中断源 */
    if (LL_USART_IsActiveFlag_RXNE(USART1)) {
        uint8_t byte = LL_USART_ReceiveData8(USART1);
        ring_buffer_write(&uart_rx_buf, byte);

        /* 如果收到换行符，用 HAL 回调通知上层 */
        if (byte == '\n') {
            HAL_UART_RxCpltCallback(&huart1);
        }
    }
}
```

这套模式的优点：
- 中断处理极快（LL），不影响其他中断的延迟
- 初始化代码可靠（HAL），减少手写寄存器配置的错误
- 只读和单字节操作用 LL，复杂控制流用 HAL

### 哪些场景适合纯 LL

- **资源极度受限**：Flash < 32 KB，RAM < 8 KB 的芯片（如 STM32G0）
- **高频 PWM / GPIO**：需要 ≥ 1 MHz 翻转频率
- **Bootloader**：需要精确控制外设状态，不允许 HAL 的状态机干扰
- **RTOS 临界区内的外设操作**：避免 HAL 的 `HAL_GetTick()` 造成延时

### 哪些场景适合纯 HAL

- **新手**：不熟悉寄存器细节，需要尽快出成果
- **快速原型开发**：CubeMX 一键生成，3 小时出 Demo
- **复杂外设**：USB、Ethernet、SDMMC 等协议栈本身就很复杂，HAL 帮你处理了大量状态细节
- **多人团队**：HAL 的标准化接口方便代码审查和交接

---

## 五、混用时的常见陷阱

### 陷阱 1：HAL 状态机未复位

```c
/* HAL 操作中断（比如超时）→ 外设处于 BUSY 状态 */
HAL_UART_Transmit(&huart1, data, len, 100);  /* 返回 HAL_TIMEOUT */

/* 然后试图用 LL 发送 */
LL_USART_TransmitData8(USART1, 'A');           /* ✅ LL 可以发 */

/* 但再调 HAL */
HAL_UART_Transmit(&huart1, data2, len2, 100);  /* ❌ 返回 HAL_BUSY！ */
```

**解决方案**：在两个代码域切换时，手动复位 HAL 状态机：

```c
huart1.State = HAL_UART_STATE_READY;
```

### 陷阱 2：中断标志冲突

HAL 的中断处理函数会清除某些中断标志。如果在 HAL 的中断处理函数之前用 LL 读了标志，HAL 的处理逻辑可能被跳过。

**解决方案**：要么完全用 HAL 处理中断，中断回调里混用 LL；要么完全用 LL 处理中断，不用 `HAL_UART_IRQHandler()`。

---

## 总结

HAL 和 LL 不是敌人，而是工具箱里的两把改锥：

| 场景 | 推荐 |
|:----|:----:|
| 快速开发、原型验证 | HAL |
| 资源受限芯片（G0/G4） | LL 或混合 |
| 频率敏感（LCD/电机控制） | LL |
| 复杂外设（USB/Ethernet） | HAL |
| 中断处理 | LL |
| 初始化代码 | HAL（CubeMX 生成） |

**混用是常态，不是例外**。把 HAL 当作"对外接口"、LL 当作"对内加速"，既能享受 CubeMX 的生产力，又能获得 LL 的高性能。

> 🏷️ STM32 HAL库 LL库 CubeMX 寄存器操作 嵌入式开发 性能优化 代码效率
