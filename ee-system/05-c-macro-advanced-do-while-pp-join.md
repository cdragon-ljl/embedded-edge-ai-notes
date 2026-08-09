# 嵌入式知识体系 · #05 · 宏的高级写法：从 `do{}while(0)` 到 `##` 拼接

---

C 语言的宏（`#define`）是嵌入式开发中最被低估的利器。很多人只知道用宏定义常量，却不知道宏可以写出媲美模板的代码生成能力。本文带你深入宏的四个高阶技巧，每个都配嵌入式实战场景。

---

## 一、`do{}while(0)`：宏的最佳外套

### 问题：普通宏的陷阱

你写过这样的宏吗？

```c
#define LED_ON()    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, SET)
#define LED_OFF()   HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, RESET)
```

单条语句没问题，但换成多条：

```c
#define SAFE_PRINT(msg) \
    if (uart_busy)      \
        printf("busy\n"); \
    printf(msg)
```

调用 `if (cond) SAFE_PRINT("hello");` 时，展开后 else 会错误地挂到内部的 `if` 上——经典的「悬空 else」bug。

### 解决方案：`do{}while(0)`

```c
#define SAFE_PRINT(msg)          \
    do {                         \
        if (uart_busy)           \
            printf("busy\n");    \
        else                     \
            printf(msg);         \
    } while (0)
```

**为什么是 `do{}while(0)` 而不是 `{}`？**

```c
// ❌ 大括号方案：
#define BAD_MACRO() { func1(); func2(); }
if (cond)
    BAD_MACRO();
else
    other();  // 编译错误：else 前多了一个分号！

// ✅ do{}while(0) 方案：
#define GOOD_MACRO() do { func1(); func2(); } while (0)
if (cond)
    GOOD_MACRO();
else
    other();  // ✅ 完美编译
```

`do{}while(0)` 末尾的分号是语句的自然结束符，而 `{}` 末尾加 `;` 就会在 `if` 后面多出一条空语句，导致 else 悬空。

### 实战：断言宏

```c
#define ASSERT(cond, msg)                        \
    do {                                         \
        if (!(cond)) {                           \
            printf("ASSERT: %s in %s:%d\n",      \
                   msg, __FILE__, __LINE__);      \
            hard_fault_handler();                \
        }                                        \
    } while (0)

void app_task(void) {
    uint32_t *p = get_buffer();
    ASSERT(p != NULL, "buffer pointer is NULL");
    *p = 0x55;  // 安全访问
}
```

---

## 二、`##` 与 `#`：字符串化与拼接魔法

### `#`：参数变成字符串

```c
#define TO_STRING(x)   #x

printf("%s\n", TO_STRING(12345));   // 输出 "12345"
printf("%s\n", TO_STRING(GPIOA));   // 输出 "GPIOA"
```

**实战：注册表打印**

```c
#define REG_DUMP(reg)                         \
    do {                                      \
        printf("%s = 0x%08lX\n",              \
               #reg, (unsigned long)(reg));    \
    } while (0)

// 用法
REG_DUMP(USART1->SR);      // 输出: USART1->SR = 0x000000C0
REG_DUMP(TIM2->CNT);       // 输出: TIM2->CNT = 0x00000452
```

### `##`：拼接标识符

```c
#define MAKE_REG(name, num)   name ## num

// 展开为：GPIOA、GPIOB
MAKE_REG(GPIO, A)
MAKE_REG(GPIO, B)
```

**实战：批量定义寄存器访问宏**

```c
// 批量生成引脚控制宏
#define PIN_FUNC(port, num)                     \
    do {                                        \
        MAKE_REG(GPIO, port)->BSRR = (1 << num); \
    } while (0)

// 用法
PIN_FUNC(A, 5);   // GPIOA->BSRR = (1 << 5);  → PA5 置高
PIN_FUNC(B, 3);   // GPIOB->BSRR = (1 << 3);  → PB3 置高
```

### 实战：自动生成中断向量表名称

```c
#define IRQ_HANDLER(irq_num)                    \
    void MAKE_REG(USART, irq_num)##_IRQHandler(void)

// 展开为：void USART1_IRQHandler(void)
IRQ_HANDLER(1) {
    // USART1 中断处理
    uint8_t data = USART1->DR;
}
```

---

## 三、可变参数宏：灵活的日志系统

### 基本用法

```c
#define LOG(fmt, ...)                       \
    printf("[LOG] " fmt "\n", __VA_ARGS__)

// 用法
LOG("temp = %d°C", temperature);  // 输出: [LOG] temp = 28°C
```

### 问题：零参数的兼容性

```c
LOG("system init OK");  // 展开为 printf("[LOG] system init OK\n", );
                        // 多了一个逗号 → 编译错误
```

### 解决方案 C99：`##__VA_ARGS__`（GCC 扩展）

```c
#define LOG(fmt, ...)                       \
    printf("[LOG] " fmt "\n", ##__VA_ARGS__)
// ^^ 双 ## 表示：如果 __VA_ARGS__ 为空，删除前面的逗号

LOG("system init OK");             // ✅ printf("[LOG] system init OK\n");
LOG("temp = %d°C", temperature);   // ✅ printf("[LOG] temp = %d°C\n", 28);
```

### 实战：带等级的日志系统

```c
#define LOG_LEVEL_NONE  0
#define LOG_LEVEL_ERR   1
#define LOG_LEVEL_WARN  2
#define LOG_LEVEL_INFO  3

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_INFO
#endif

#define LOG_ERR(fmt, ...)                        \
    do {                                         \
        if (LOG_LEVEL >= LOG_LEVEL_ERR)          \
            printf("[ERR]  " fmt "\n", ##__VA_ARGS__); \
    } while (0)

#define LOG_WARN(fmt, ...)                       \
    do {                                         \
        if (LOG_LEVEL >= LOG_LEVEL_WARN)         \
            printf("[WARN] " fmt "\n", ##__VA_ARGS__); \
    } while (0)

#define LOG_INFO(fmt, ...)                       \
    do {                                         \
        if (LOG_LEVEL >= LOG_LEVEL_INFO)         \
            printf("[INFO] " fmt "\n", ##__VA_ARGS__); \
    } while (0)

// ⚡ 编译时就能裁剪！空宏无运行时开销
void test_log(void) {
    LOG_INFO("ADC value: %d", adc_val);
    LOG_WARN("buffer 80%% full (%d/%d)", used, total);
    LOG_ERR("I2C timeout on bus %d", bus_id);
}
```

> 编译时，如果 `LOG_LEVEL` 设为 `LOG_LEVEL_WARN`，所有 `LOG_INFO()` 的 `if` 条件恒假，优化器会直接删除——**零运行时开销**。

---

## 四、X-Macro：数据驱动的代码生成

X-Macro 是最被低估的宏技，它用同一个「数据表」同时生成枚举、字符串表、处理函数，保证三者始终同步。

### 实战：GPIO 功能表

先定义一个数据驱动表：

```c
// === gpio_pins.def ===
X(GPIO_PIN_LED,     'A', 5)   // PA5 - LED
X(GPIO_PIN_BTN,     'B', 3)   // PB3 - Button
X(GPIO_PIN_UART_TX, 'A', 9)   // PA9 - USART1 TX
X(GPIO_PIN_UART_RX, 'A', 10)  // PA10 - USART1 RX
#undef X
```

然后，用不同的 `#define X` 生成不同内容：

```c
// === gpio_enum.h — 生成枚举 ===
typedef enum {
#define X(name, port, pin) name,
#include "gpio_pins.def"
    GPIO_PIN_COUNT
} gpio_pin_t;

// 展开为：
// typedef enum {
//     GPIO_PIN_LED,
//     GPIO_PIN_BTN,
//     GPIO_PIN_UART_TX,
//     GPIO_PIN_UART_RX,
//     GPIO_PIN_COUNT
// } gpio_pin_t;
```

```c
// === gpio_names.h — 生成字符串表 ===
static const char *gpio_pin_names[] = {
#define X(name, port, pin) #name,
#include "gpio_pins.def"
};

// 展开为：
// static const char *gpio_pin_names[] = {
//     "GPIO_PIN_LED",
//     "GPIO_PIN_BTN",
//     "GPIO_PIN_UART_TX",
//     "GPIO_PIN_UART_RX",
// };
```

```c
// === gpio_init.c — 生成初始化代码 ===
static void gpio_init_all(void) {
#define X(name, port, pin)                                 \
    do {                                                    \
        GPIO_InitTypeDef init = {0};                        \
        init.Pin   = MAKE_REG(GPIO_PIN_, pin);              \
        init.Mode  = GPIO_MODE_OUTPUT_PP;                   \
        init.Speed = GPIO_SPEED_FREQ_LOW;                   \
        MAKE_REG(HAL_GPIO_Init, )(MAKE_REG(GPIO, port), &init); \
    } while (0);

#include "gpio_pins.def"
}
```

### 一次性宏（不用头文件）

如果不想分离文件，用宏参数传递：

```c
#define GPIO_PIN_TABLE          \
    X(GPIO_PIN_LED,     A, 5)  \
    X(GPIO_PIN_BTN,     B, 3)  \
    X(GPIO_PIN_UART_TX, A, 9)  \
    X(GPIO_PIN_UART_RX, A, 10)

// 生成枚举
typedef enum {
#define X(name, port, pin) name,
    GPIO_PIN_TABLE
    GPIO_PIN_COUNT
} gpio_pin_t;
#undef X
```

### X-Macro 的真正价值

普通做法是手写枚举、手写字符串表、手写初始化函数——三份信息靠人工同步，改一个引脚要改三个地方。X-Macro 把数据集中在**一个地方**，所有代码都从这个数据推导生成，**不可能出现不一致**。

这在管理几十上百个引脚的项目中，节省的时间是量级的。

---

## 五、终极综合：注册表风格的设备驱动框架

把前面所有技巧结合：

```c
// 1. do{}while(0) 确保宏安全
// 2. ## 拼接寄存器名
// 3. 可变参数做日志
// 4. X-Macro 做设备表

#define DEVICE_TABLE                        \
    X(DEV_USART1, USART1, 115200, 'A', 9)   \
    X(DEV_USART2, USART2, 9600,   'D', 5)   \
    X(DEV_I2C1,   I2C1,   400000, 'B', 6)   \
    X(DEV_SPI1,   SPI1,   8000000,'A', 5)

// 生成设备枚举
typedef enum {
#define X(name, reg, baud, port, pin) name,
    DEVICE_TABLE
    DEV_COUNT
} device_id_t;
#undef X

// 生成初始化统一入口
#define DEV_INIT(name, reg, baud, port, pin)          \
    static void MAKE_REG(name, _init)(void) {          \
        LOG_INFO("Init %s @ %d baud", #name, baud);   \
        /* 实际初始化... */                              \
        LED_ON();                                       \
    }

DEVICE_TABLE
#undef X

// 统一调用
void all_devices_init(void) {
#define X(name, reg, baud, port, pin)  MAKE_REG(name, _init)();
    DEVICE_TABLE
#undef X
}
```

---

> 🏷️ 宏技巧 C语言 嵌入式编程 代码生成 do-while-0 X-Macro 可变参数宏 预处理器
