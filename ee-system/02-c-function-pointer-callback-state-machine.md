# 嵌入式知识体系 · #2 · 函数指针与回调机制：嵌入式状态机的灵魂

---

假设你正在写一个 UART 驱动，需要支持不同波特率、不同数据位长度、不同校验模式。如果每换一种配置就写一个 `if-else` 分支，代码会膨胀到难以维护。更棘手的是：当接收一帧数据时，有时需要检查起始位，有时需要校验 CRC，有时需要解析应用层协议——这些逻辑在运行时才能确定。

这时候就需要函数指针和回调机制出场了。它们是嵌入式世界中"面向接口编程"的基石，也是实现状态机、驱动抽象层和 HAL 库的核心工具。

---

## 一、函数指针声明、赋值与调用

### 1.1 声明：记忆诀窍

函数指针的声明语法看上去很古怪，但有一个口诀：**先把 `(*p)` 当成变量名，再套上返回值类型和参数表**。

```c
// 声明一个函数指针 pFunc，指向：
// 返回值是 void，参数是 int 的函数
void (*pFunc)(int);
```

对比一下函数声明和函数指针声明：

```c
// 函数声明
void delay_ms(uint32_t ms);

// 函数指针声明 —— 把 delay_ms 换成 (*pFunc)
void (*pFunc)(uint32_t ms);
```

这样就清晰了：`(*pFunc)` 是"一个指针，指向函数"，这个函数返回值是 `void`，参数是 `uint32_t`。

### 1.2 赋值

```c
void delay_ms(uint32_t ms) {
    for(volatile uint32_t i = 0; i < ms * 1000; i++);
}

// 三种赋值方式，效果完全一样
pFunc = delay_ms;        // 方式一：最常用
pFunc = &delay_ms;       // 方式二：取地址，可选
pFunc = &delay_ms;       // 方式三：同上
```

在 C 语言中，函数名本身就是地址，所以 `delay_ms` 和 `&delay_ms` 等价。

### 1.3 调用

```c
// 两种调用方式等价
pFunc(100);      // 方式一：隐式解引用
(*pFunc)(100);   // 方式二：显式解引用（与声明语法更一致）
```

> **实战建议**：推荐用 `pFunc(100)` 方式，代码更简洁。但在初始化时用 `&delay_ms` 赋值可读性更好，能清晰表明"我在取函数地址"。

---

## 二、typedef 简化：从此告别丑陋语法

函数指针类型声明写多了会发现很冗长，C 语言提供了 `typedef` 来"命名"这类类型：

```c
// 定义：FuncPtr 是一种类型——指向 void(uint32_t) 的函数指针
typedef void (*FuncPtr)(uint32_t);

// 用法：声明变量就像声明 int 一样简单
FuncPtr p = delay_ms;
p(100);
```

再看实际嵌入式中的例子：

```c
// HAL 库中的定时器回调类型
typedef void (*TIM_CallbackFunc)(TIM_HandleTypeDef *htim);

// 声明两个回调变量
TIM_CallbackFunc onPeriodElapsed = TIM_ElapsedCallback;
TIM_CallbackFunc onCaptureComplete = TIM_CaptureCallback;

// 统一调用
onPeriodElapsed(&htim2);
```

用 `typedef` 之后，函数指针变量的声明读起来跟普通变量一样自然。

---

## 三、回调函数在 HAL 库中的实际应用

回调（callback）的本质很简单：**你注册一个函数指针给我，我在合适的时候调用它"回"你**。

### 3.1 STM32 HAL 的定时器回调注册

```c
// stm32f1xx_hal_tim.h
typedef void (*TIM_TypeDef)(TIM_HandleTypeDef *);

// 回调注册：弱符号，用户可重写
__weak void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    // 用户需要在外部实现这个函数
}

// HAL 内部怎么调用的？
void HAL_TIM_IRQHandler(TIM_HandleTypeDef *htim) {
    if (__HAL_TIM_GET_FLAG(htim, TIM_FLAG_UPDATE)) {
        __HAL_TIM_CLEAR_FLAG(htim, TIM_FLAG_UPDATE);
        HAL_TIM_PeriodElapsedCallback(htim);  // ← 回调用户代码
    }
}
```

用户只需要在 main.c 中实现 HAL 预留的"弱符号"函数：

```c
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM2) {
        led_toggle();      // 每个定时周期翻转 LED
    }
}
```

### 3.2 更灵活的方式：函数指针表

弱符号回调只能有一个全局实现。如果要用多个定时器，每个触发不同行为，就需要更灵活的方式：

```c
#define MAX_TIM_CALLBACKS  8

typedef struct {
    TIM_HandleTypeDef *htim;
    void (*callback)(void *arg);
    void *arg;
} TIM_CallbackEntry;

static TIM_CallbackEntry s_callbacks[MAX_TIM_CALLBACKS];
static uint8_t s_cb_count = 0;

int TIM_RegisterCallback(TIM_HandleTypeDef *htim,
                         void (*cb)(void *), void *arg) {
    if (s_cb_count >= MAX_TIM_CALLBACKS) return -1;
    s_callbacks[s_cb_count].htim = htim;
    s_callbacks[s_cb_count].callback = cb;
    s_callbacks[s_cb_count].arg = arg;
    s_cb_count++;
    return 0;
}

void TIM_GlobalCallback(TIM_HandleTypeDef *htim) {
    for (uint8_t i = 0; i < s_cb_count; i++) {
        if (s_callbacks[i].htim == htim) {
            s_callbacks[i].callback(s_callbacks[i].arg);
        }
    }
}
```

这样注册时就可以传自定义参数：

```c
void button_check(void *arg) {
    uint32_t pin = (uint32_t)arg;
    if (HAL_GPIO_ReadPin(GPIOA, pin) == 0) {
        // 按键按下
    }
}

TIM_RegisterCallback(&htim2, button_check, (void *)GPIO_PIN_0);
TIM_RegisterCallback(&htim3, button_check, (void *)GPIO_PIN_1);
```

这就是回调机制的核心：**把一个函数"借"给底层驱动，让底层在恰当的时候执行你的逻辑**。

---

## 四、函数指针式状态机：完整实战

这是本文最核心的部分——用函数指针表实现一个**数据帧接收状态机**。

### 4.1 问题场景

MCU 通过 UART 接收数据帧：`STX(0xAA) + LEN + DATA[N] + CRC`。按字节逐个接收，需要根据当前处于哪个阶段来决定下一步行为。

### 4.2 状态定义

```c
typedef enum {
    ST_IDLE = 0,   // 等待帧头 STX
    ST_STX,        // 收到 STX，准备收长度
    ST_LEN,        // 收到长度，准备收数据
    ST_DATA,       // 收数据载荷
    ST_CRC,        // 收 CRC 校验字节
    ST_DONE        // 完整帧已收到
} FrameState_t;
```

### 4.3 状态处理函数

每个状态对应一个处理函数：

```c
// 状态处理函数 —— 统一接口类型
typedef FrameState_t (*StateHandler)(uint8_t byte, FrameCtx *ctx);

FrameState_t state_idle(uint8_t byte, FrameCtx *ctx) {
    if (byte == 0xAA) {
        ctx->crc = 0;
        ctx->crc ^= byte;
        return ST_STX;           // 收到帧头，进入 STX 状态
    }
    return ST_IDLE;              // 没收到，继续等
}

FrameState_t state_stx(uint8_t byte, FrameCtx *ctx) {
    // 已收到 STX，下一个字节是长度
    // 检查长度合法性
    if (byte == 0 || byte > MAX_FRAME_LEN) {
        return ST_IDLE;          // 非法长度，回到空闲
    }
    ctx->len = byte;
    ctx->idx = 0;
    ctx->crc ^= byte;
    return ST_LEN;
}

FrameState_t state_len(uint8_t byte, FrameCtx *ctx) {
    // 刚收到长度，这里可以做扩展校验（高字节等）
    ctx->crc ^= byte;
    return (ctx->len > 0) ? ST_DATA : ST_CRC;
}

FrameState_t state_data(uint8_t byte, FrameCtx *ctx) {
    ctx->data[ctx->idx++] = byte;
    ctx->crc ^= byte;
    if (ctx->idx >= ctx->len)
        return ST_CRC;           // 收完数据，准备校验
    return ST_DATA;
}

FrameState_t state_crc(uint8_t byte, FrameCtx *ctx) {
    if (byte == ctx->crc) {
        return ST_DONE;          // CRC 校验通过！
    }
    return ST_IDLE;              // CRC 失败，丢弃整帧
}
```

### 4.4 函数指针表：状态机的核心

将所有状态处理函数放入一张表，**用枚举值作为索引**：

```c
// 函数指针表 —— 状态机的"核心调度器"
static const StateHandler s_state_table[] = {
    [ST_IDLE] = state_idle,
    [ST_STX]  = state_stx,
    [ST_LEN]  = state_len,
    [ST_DATA] = state_data,
    [ST_CRC]  = state_crc,
    // 若编译环境不支持 designated initializer，则按枚举顺序手动写
};
```

### 4.5 运行调度

```c
typedef struct {
    FrameState_t state;
    uint8_t      data[MAX_FRAME_LEN];
    uint8_t      len;
    uint8_t      idx;
    uint8_t      crc;
} FrameCtx;

FrameCtx g_ctx = {.state = ST_IDLE};

void UART_RxByte_Handler(uint8_t byte) {
    // 查表 -> 调用 -> 更新状态 —— 核心就一行！
    g_ctx.state = s_state_table[g_ctx.state](byte, &g_ctx);

    if (g_ctx.state == ST_DONE) {
        ProcessFrame(&g_ctx);          // 处理完整帧
        g_ctx.state = ST_IDLE;         // 状态机复位，等待下一帧
    }
}
```

**整个状态机的核心逻辑只有一条语句**：找到当前状态对应的函数 → 调用 → 用返回结果更新状态。没有 `switch-case`，没有 `if-else`，纯粹的"查表+调用"。

### 4.6 对比：switch-case 实现 vs 函数指针表

```
switch-case 实现                          函数指针表实现
──────────────────────────               ──────────────────────────
switch(state) {                          static handler_t tbl[] = {
    case ST_IDLE:                             state_idle,
        state = idle(byte);                   state_stx,
        break;                                ...
    case ST_STX:                          };
        state = stx(byte);               while(1) {
        break;                                state = tbl[state](byte);
    ...                                   }
}
```

两者的函数体实现一模一样（`state_idle`、`state_stx` 等），**但调度方式完全不同**。

---

## 五、坑与注意

### 5.1 回调函数与中断的"时间账"

回调函数是在中断上下文中执行的。如果回调耗时过长，会阻塞其他中断甚至导致系统崩溃。

```c
// ❌ 错误：回调里做耗时操作
void TIM_ElapsedCallback(TIM_HandleTypeDef *htim) {
    HAL_Delay(100);           // 在中断里延时？灾难！
    printf("tick\n");         // printf 可能阻塞
}

// ✅ 正确：回调里只做标记，耗时的放主循环
volatile uint8_t g_tick_flag = 0;
void TIM_ElapsedCallback(TIM_HandleTypeDef *htim) {
    g_tick_flag = 1;          // 只设标志
}
```

**黄金法则**：中断回调只做"传递"不做"处理"——设标志、放队列、触发 DMA，但绝不阻塞或做复杂计算。

### 5.2 函数指针的"裸奔"风险

函数指针本身没有任何类型安全保证——你可以把一个 `void(*)(int)` 赋值给 `int(*)(void)`，编译器可能只给警告甚至不报错，但运行时必定崩溃。

```c
int add(int a, int b) { return a + b; }
void blink(void) { ... }

void (*p)(void) = (void (*)(void))add;  // ❌ 强制转型，编译器不拦
p();  // 栈帧错乱，HardFault！
```

**防御措施**：
1. 始终用 `typedef` 定义统一的回调类型
2. 回调注册接口做参数校验（`if (cb == NULL) return -1;`）
3. 永远不要对函数指针做强制类型转换

### 5.3 const 函数指针表必须加 const

```c
// ✅ 正确：函数指针表放在 Flash 中
static const StateHandler s_state_table[] = { ... };

// ❌ 错误：不加 const，表放在 RAM 中浪费空间
static StateHandler s_state_table[] = { ... };
```

状态机函数指针表在运行期不会改变，必须加 `const`。否则这张表会被放入 `.data` 段占用宝贵的 RAM，而它本应待在 Flash 里。

### 5.4 函数指针大小：不同架构不同

```c
printf("sizeof(function pointer) = %zu\n", sizeof(void (*)(void)));
```

| 架构 | 函数指针大小 | 说明 |
|------|-------------|------|
| Cortex-M0/M0+ | 2 bytes | Thumb 指令，函数地址 16 位对齐 |
| Cortex-M3/M4/M7 | 4 bytes | 32 位地址空间 |
| Cortex-A (32-bit) | 4 bytes | 标准 ARM 32 位 |
| Cortex-A (64-bit) | 8 bytes | AArch64 模式 |

移植代码时要注意：函数指针大小是**平台相关的**。在 64 位 Linux 上编译的代码放到 32 位 MCU 上，函数指针大小不同可能导致结构体布局出错。

---

## 六、总结

| 概念 | 一句话 | 嵌入式实战场景 |
|------|--------|---------------|
| 函数指针声明 | `void (*p)(int)` — 把函数名换成 `(*p)` | 驱动函数间接调用 |
| typedef 简化 | `typedef void (*Handler)(int);` | 定义统一回调类型 |
| 回调注册 | 用户注册，底层在适当时机回调 | HAL 库中断回调、定时器回调 |
| 函数指针表 | 用数组索引替代 switch-case | 状态机、命令解析、协议处理 |
| const 约束 | 运行时不变的表必须加 const | 函数指针表放 Flash，省 RAM |

函数指针的威力不在于语法本身，而在于它带来的架构灵活性：**调用方和实现方在编译时解耦，在运行时组合**。这也是从"写代码"到"设计代码"的一条分水岭。

```
            ┌──────────────────────┐
            │   UART 接收字节       │
            └────────┬─────────────┘
                     │ byte
                     ▼
            ┌──────────────────────┐
            │  s_state_table[state] │  ← 查表
            └────────┬─────────────┘
                     │ (*handler)(byte)
                     ▼
            ┌──────────────────────┐
            │  状态处理函数 (如     │
            │  state_idle,         │
            │  state_stx, ...)     │
            └────────┬─────────────┘
                     │ return next_state
                     ▼
            ┌──────────────────────┐
            │  更新 g_ctx.state    │
            └──────────────────────┘
```

我是 学嵌入式的长路 ，用代码和思考陪你走通嵌入式的每一公里。

---

> 🏷️ #C语言 #函数指针 #回调机制 #状态机 #嵌入式开发 #驱动设计