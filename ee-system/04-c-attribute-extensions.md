# 嵌入式知识体系 · #4 · `__attribute__` 妙用：编译器扩展让 C 更强大

---

假设你在调试一个诡异的启动问题——程序还没进 `main()` 就卡死了。你用 JTAG 连上去，发现 PC 停在某个函数里。但代码里没人调过它。

又或者，你希望在 HAL 库中提供一个**默认的 GPIO 初始化函数**，用户如果没写自己的版本，就自动用默认的——但 C 语言没有"弱符号"的概念。

再或者，你想把一段初始化代码放到某个特殊内存段，比如 DTCM 或紧耦合内存（TCM），在链接脚本里精细控制它的位置。

这些需求，**标准 C 语言**一个都满足不了。但 GCC 的 `__attribute__` 全部能搞定。

`__attribute__` 是 GCC 提供的编译器扩展机制，用 `__attribute__((属性名))` 的语法给变量、函数、类型添加**编译期元信息**。在嵌入式开发中，它几乎是不可或缺的瑞士军刀。

---

## 一、`section`：把变量/函数放到指定内存段

链接脚本把代码分成 `.text`（代码）、`.data`（初始化的全局变量）、`.bss`（未初始化的全局变量）等段。默认情况下，所有函数都进 `.text`，所有全局变量都根据初始化情况进 `.data` 或 `.bss`。

`__attribute__((section("name")))` 让你**自定义段名**，把特定变量或函数放到指定的内存区域。

### 场景 1：把数据放到特定 RAM 区域

某些 MCU 有片内 SRAM 和紧耦合内存（TCM），TCM 访问延迟更低。把关键变量放到 TCM：

```c
// 将中断频繁访问的变量放到 TCM
#define ITCM_RAM  __attribute__((section(".itcm_ram")))

ITCM_RAM volatile uint32_t system_tick;
ITCM_RAM struct can_rx_buf rx_buffers[16];
```

然后在链接脚本中定义 `.itcm_ram` 段映射到 TCM 地址范围。**不需要修改链接脚本就无法匹配**——但如果链接脚本不包含这个段，链接会报错，确保你意识到需要适配。

### 场景 2：构建初始化函数表

在 RTOS 或框架代码中，经常需要让各个模块的初始化函数自动注册到一个表里，然后统一遍历调用：

```c
// 定义一个初始化函数指针类型
typedef void (*init_fn_t)(void);

// 定义一个段标记
#define INIT_PRIO(n)  __attribute__((section(".init_call." #n)))

// 各模块注册初始化函数
void spi_init(void)     INIT_PRIO(1);
void i2c_init(void)     INIT_PRIO(2);
void uart_init(void)    INIT_PRIO(3);

// 链接器会把这些函数指针按优先级顺序排列在 .init_call 段
// 然后统一遍历：
extern init_fn_t __init_call_start[];
extern init_fn_t __init_call_end[];

void run_init_chain(void) {
    for (init_fn_t *fn = __init_call_start; fn < __init_call_end; fn++) {
        (*fn)();
    }
}
```

这种模式在 FreeRTOS 的 `CreateThread()` 中、Linux 内核的 `__initcall` 中广泛使用。核心思路是：**用编译期的段分配 + 链接脚本的排序，替代运行时的全局数组维护。**

### 场景 3：把版本信息固话到固定地址

把固件版本字符串放到固定的 Flash 地址，上位机可以通过读取该地址来判断固件版本：

```c
__attribute__((section(".fw_version")))
const char firmware_version[] = "v2.1.0-build20260714";
```

---

## 二、`aligned` / `packed`：精细控制对齐

### `aligned(n)`：提升对齐要求

之前讲到内存对齐时，关注的是自然对齐的**最低要求**。`__attribute__((aligned(n)))` 可以**提升**对齐等级，适合以下场景：

```c
// 让缓冲区按 32 字节对齐（DMA 优化、Cache Line 对齐）
uint8_t dma_buffer[1024] __attribute__((aligned(32)));

// 让结构体整体按 16 字节对齐
struct __attribute__((aligned(16))) matrix {
    float data[4][4];
};
```

Cortex-M7 的 Cache Line 通常是 32 字节，DMA 缓冲区如果跨越 Cache Line 边界，会引发一致性问题。用 `aligned(32)` 确保缓冲区首地址对齐，是解决这类问题的最简单手段。

### `packed`：取消填充

`__attribute__((packed))` 在第 #3 篇已经详细讲过，这里只做总结：

```c
struct __attribute__((packed)) can_frame {
    uint16_t id;      // offset 0
    uint8_t  dlc;     // offset 2
    uint8_t  data[8]; // offset 3
};  // sizeof = 11（而不是正常对齐的 12）
```

适用于**协议解析**和**存储格式**，但要注意非对齐访问的性能代价。

> **对比：** `aligned(n)` 是往上加的（要求更严格），`packed` 是往下去掉的（要求更宽松）。两者可以组合：
> ```c
> struct __attribute__((packed, aligned(4))) {
>     uint8_t  a;
>     uint32_t b;
> };  // 内部不填充，但整体按 4 字节对齐
> ```

---

## 三、`weak`：弱符号实现默认函数与重写

这是嵌入式开发中**最有用**的 `__attribute__` 之一。

### 问题

在标准 C 中，如果定义了两个相同名称的函数，链接器会报 "multiple definition" 错误。但在很多框架中，我们希望：**提供一个默认实现，允许用户在自己的代码中覆盖它。**

这就是弱符号（weak symbol）的用武之地。

### 语法

```c
// 弱符号定义：提供一个默认实现
__attribute__((weak)) void HAL_GPIO_Init(void) {
    // 默认初始化：全部设为推挽输出
    GPIOA->CRL = 0x33333333;
    GPIOA->CRH = 0x33333333;
}

// 用户在自己的代码中定义一个同名函数（强符号）
void HAL_GPIO_Init(void) {
    // 自定义初始化
    RCC->APB2ENR |= RCC_APB2ENR_IOPAEN;
    GPIOA->CRL = 0x44444444;  // 开漏输出
}

// 链接器优先选择强符号。如果用户定义了，弱符号被忽略。
```

### 嵌入式中 weak 的经典用法

**HAL 库的中断回调：**

```c
// HAL 库提供的弱符号回调
__attribute__((weak)) void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    // 什么都不做——用户可以自行实现
}

// 用户代码中实现
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    // 处理接收完成事件
    rx_buffer[rx_index++] = huart->Instance->DR;
}
```

STM32 HAL 库中几乎所有回调函数都是 `weak` 的。用户只需要定义同名的函数，链接器自动忽略弱符号版本——**不需要修改库代码，不需要条件编译宏**，比 `#ifdef` 方案优雅得多。

**RTOS 的默认钩子函数：**

```c
__attribute__((weak)) void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName) {
    // 默认死循环——至少让你知道栈溢出了
    for(;;);
}

__attribute__((weak)) void vApplicationIdleHook(void) {
    // 默认什么都不做
}
```

### weak 的工作原理

1. 编译器给弱符号打上特殊标记
2. 链接器在解析符号时，**强符号优先于弱符号**
3. 如果同一个符号有多个弱定义，链接器随机选取一个（所以 `.o` 文件的链接顺序会影响结果）
4. 如果弱符号没有被替换，它正常工作；如果被替换，弱符号被丢弃

> ⚠️ **注意：** weak 必须用在**函数定义**或**变量定义**上（分配存储空间），不能用在声明（`extern`）上。`extern __attribute__((weak))` 只是声明一个可能存在也可能不存在的符号——如果没人提供就会链接失败。

---

## 四、`constructor` / `destructor`：main 前后的自动执行

这两个属性让函数在 `main()` 之前或 `exit()` 之后自动执行。

### 基本用法

```c
__attribute__((constructor)) void before_main(void) {
    // 在 main() 之前自动执行
    init_hardware();
    init_heap();
}

__attribute__((destructor)) void after_main(void) {
    // 在 exit() 或 main() 返回后自动执行
    cleanup_resources();
}
```

### 执行顺序控制

可以给 `constructor` 指定优先级（数值越小越靠前）：

```c
__attribute__((constructor(101))) void early_init(void) {
    // 最先执行
}
__attribute__((constructor(102))) void mid_init(void) {
    // 第二个执行
}
__attribute__((constructor(103))) void late_init(void) {
    // 第三个执行
}
```

优先级范围一般用 **101~65535**，0~100 保留给编译器内部使用。

### 嵌入式中不要过度依赖 constructor

在资源受限的 MCU 上，`constructor` 的使用场景有限，原因有三：

1. **依赖 C 运行时初始化**：constructor 由 `__libc_init_array()` 遍历段中的函数指针来调用。如果程序不用 C 标准库（比如裸机 `-nostdlib`），这段代码可能不存在
2. **启动顺序不可控**：在不同 .c 文件中定义的 constructor 函数，执行顺序取决于链接器排列的 `.init_array` 段顺序，不易预测
3. **替代方案更可靠**：在启动文件（`startup_xxx.s`）或系统初始化函数中显式调用的方案更可控

> **推荐做法：** 在 `main()` 入口处显式调用 `SystemInit()` + `__libc_init_array()` + 各模块初始化函数。constructor 更适合**PC 端测试代码**或**不需要极端可靠性的场景**。

---

## 五、其他高频 `__attribute__`

### `used`：防止编译器优化掉"没用"的变量

编译器会在 `-Os` 优化下移除没有任何引用的变量。但如果你希望变量**因为链接脚本的段定义而被链接器保留**，需要用 `used` 标记：

```c
// 基于 section 的初始化表：这个函数没人直接调用，但不能被优化掉
static void __attribute__((used, section(".init_call.1"))) spi_pins_init(void) {
    // ...
}
```

### `always_inline`：强制内联

`inline` 关键字只是**建议**编译器内联。`__attribute__((always_inline))` 强制内联：

```c
static inline __attribute__((always_inline)) uint32_t read_reg(volatile uint32_t *addr) {
    return *addr;
}
```

在 MCU 上，高频调用的关键路径（如中断服务程序中的位操作）适合强制内联，避免函数调用开销。但要注意**代码膨胀**——内联后的代码会被复制到每个调用点。

### `naked`：裸函数（无编译器生成的序言/尾声）

中断服务程序在某些编译器上要求用 `naked` 来避免编译器干扰栈帧：

```c
__attribute__((naked)) void SysTick_Handler(void) {
    __asm volatile (
        "push {r4-r11, lr}\n\t"
        "bl systick_isr\n\t"
        "pop {r4-r11, pc}\n\t"
    );
}
```

> ⚠️ `naked` 函数中**不能有局部变量和函数调用**（除了内联汇编中的调用），因为编译器不会帮你管理栈帧。

### `unused`：消除"未使用"警告

```c
static void __attribute__((unused)) debug_print(const char *msg) {
    // 暂时不用，但不删
}
```

适合那些在调试阶段保留、但在发布版本中可能不用到的函数。

---

## 六、综合实战：构建模块化初始化框架

下面组合 `section` + `used` + `constructor` 实现一个**松耦合的模块注册系统**，它可以替代手动维护全局初始化列表。

```c
/* module_init.h */
#ifndef MODULE_INIT_H
#define MODULE_INIT_H

typedef void (*module_init_fn_t)(void);

// 模块注册宏：按优先级把初始化函数放入 .module_init 段
#define MODULE_INIT(priority) \
    static void module_init_##__LINE__(void); \
    static module_init_fn_t __attribute__((used, section(".module_init." #priority))) \
        __module_init_ptr_##__LINE__ = module_init_##__LINE__; \
    static void module_init_##__LINE__(void)

/* main.c */
#include "module_init.h"

// 模块 A：优先级 1（最先）
MODULE_INIT(1) {
    // 初始化 GPIO、时钟等基础外设
    RCC->APB2ENR = 0xFFFFFFFF;
}

// 模块 B：优先级 2
MODULE_INIT(2) {
    // 初始化 UART
    USART1->CR1 = USART_CR1_UE;
}

// 模块 C：优先级 3
MODULE_INIT(3) {
    // 初始化 I2C
    I2C1->CR1 = I2C_CR1_PE;
}

// 在 main() 中统一执行
extern module_init_fn_t __module_init_start[];
extern module_init_fn_t __module_init_end[];

int main(void) {
    for (module_init_fn_t *fn = __module_init_start; fn < __module_init_end; fn++) {
        (*fn)();
    }
    // ... 主循环
}
```

链接脚本中需要添加：

```ld
SECTIONS {
    .module_init : {
        __module_init_start = .;
        KEEP(*(.module_init.*))
        __module_init_end = .;
    } > FLASH
}
```

**这样做的优势：**

- ✅ **松耦合**：每个模块只需要包含头文件，不需要手动注册
- ✅ **自动排序**：优先级数字决定执行顺序
- ✅ **零运行时开销**：初始化的遍历是线性，没有链表管理开销
- ✅ **编译时检查**：函数签名不匹配会在编译时报错

---

## 七、注意事项与最佳实践

### 1. 可移植性

`__attribute__` 是 GCC 扩展，**不是 C 标准的一部分**。如果要跨编译器（ARMCC、IAR、Clang），需要用宏抽象：

```c
#ifdef __GNUC__
#define WEAK        __attribute__((weak))
#define PACKED      __attribute__((packed))
#define SECTION(x)  __attribute__((section(x)))
#else
#ifdef __ICCARM__
#define WEAK        __weak
#define PACKED      __packed
#define SECTION(x)  @ x  // IAR 语法
#endif
#endif
```

### 2. 多用宏封装

直接写 `__attribute__` 会降低代码可读性。建议封装成有意义的名字：

```c
// 好的做法
#define RAM_FUNC   __attribute__((section(".ram_functions"), long_call))
#define IN_ITCM    __attribute__((section(".itcm_code")))
#define ALIGN_32   __attribute__((aligned(32)))
#define WEAK       __attribute__((weak))
#define USED       __attribute__((used))
#define PACKED     __attribute__((packed))
```

### 3. 理解链接脚本

`section` 属性的能力上限取决于链接脚本。**不了解链接脚本，section 属性只能发挥一半功力。** 建议在写 section 属性前，至少读过一遍 MCU 的 `.ld` 文件。

### 4. 推荐还是谨慎

| 属性 | 推荐指数 | 说明 |
|:----|:--------:|------|
| `packed` | ⭐⭐⭐ | 协议解析必备，注意非对齐代价 |
| `aligned` | ⭐⭐⭐ | DMA/Cache 场景必用 |
| `weak` | ⭐⭐⭐⭐⭐ | HAL 库回调、框架扩展，最实用的属性 |
| `section` | ⭐⭐⭐⭐ | 高端用法，构建初始化链的利器 |
| `constructor` | ⭐⭐ | 裸机 MCU 上慎用，PC 调试可用 |
| `used` | ⭐⭐⭐ | 配合 section 属性使用 |
| `always_inline` | ⭐⭐⭐ | 关键路径优化，注意代码膨胀 |
| `naked` | ⭐⭐ | 高级用户专用，新手容易踩坑 |

---

## 总结

`__attribute__` 是 C 语言通往底层控制能力的钥匙：

- **`section`** 让你操控变量的物理位置，上对链接脚本，下接分散加载
- **`packed` / `aligned`** 让你精细掌控内存布局
- **`weak`** 是框架设计的利器，是 HAL 库可扩展性的基石
- **`constructor` / `destructor`** 提供了执行时序的钩子
- **`used` / `always_inline` / `naked`** 等小工具在特定场景下不可或缺

这些能力都不是 C 标准提供的，但 GCC 经过几十年的打磨，这些扩展已经成了事实标准——ARMCC、IAR 的对应语法几乎都是等价的。掌握它们，你的 C 代码就有了"编译期元编程"的能力，许多复杂的设计模式变得举重若轻。

下一期顺理成章——宏的进阶技巧里，`##` 拼接和 X-Macro 的魔法能让代码生成自动化，继续深化"用编译器帮你写代码"的主题。

---

> 🏷️ 技术关键词：__attribute__ · GCC扩展 · section · weak · packed · aligned · 嵌入式C · 编译器魔法
