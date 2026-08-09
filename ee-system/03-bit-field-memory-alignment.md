# 嵌入式知识体系 · #3 · 结构体位域与内存对齐：省内存也要讲效率

---

你写了一个结构体：

```c
struct sensor_data {
    uint8_t   id;
    uint32_t  value;
    uint8_t   status;
};
```

直觉告诉你，这应该占 `1 + 4 + 1 = 6` 个字节。但 `sizeof(struct sensor_data)` 的结果是 **12**。

足足多了一倍。

更诡异的是，调换一下成员顺序：

```c
struct sensor_data_compact {
    uint32_t  value;
    uint8_t   id;
    uint8_t   status;
};
```

`sizeof` 变成了 **8** —— 什么都没丢，省了 4 字节。

这种"看不见的浪费"，来自 CPU 对**内存对齐**的硬性要求。而与之对应的**位域（bit-field）**则把打包做到极致，在控制寄存器映射和协议解析中不可或缺。理解这两者，是写出高效嵌入式代码的基本功。

---

## 一、内存对齐：为什么 CPU 要"垫东西"

### 对齐规则

现代 CPU 访问内存时，不是以字节为单位逐个取的，而是以**字（word）**为单位批量读取。ARM Cortex-M 默认用 4 字节对齐：一个 4 字节的 `int` 如果放在地址 0x1000（4 的倍数），CPU 一次就能读完；如果放在 0x1001，CPU 需要读两次再拼接——既慢又可能触发硬件异常。

这就是**自然对齐（natural alignment）**规则：**一个 N 字节的变量必须存放在 N 的倍数地址上。**

| 类型 | 大小 | 自然对齐要求 |
|:----|:----:|:-----------:|
| `uint8_t` | 1 | 1 字节对齐（任意地址）|
| `uint16_t` | 2 | 2 字节对齐（偶数地址）|
| `uint32_t` | 4 | 4 字节对齐 |
| `uint64_t` | 8 | 8 字节对齐 |

### padding 怎么来的

编译器在结构体成员之间**插入填充字节（padding）**，确保每个成员都满足自然对齐。

以第一个结构体为例：

```c
struct sensor_data {            // 起始地址 0x1000
    uint8_t   id;               // offset 0 (1 byte)
    // padding: 3 bytes         // offset 1~3 (对齐 uint32_t)
    uint32_t  value;            // offset 4 (4 bytes)
    uint8_t   status;           // offset 8 (1 byte)
    // padding: 3 bytes         // offset 9~11 (对齐整个结构体)
};                              // sizeof = 12
```

总共 12 字节，其中 **6 字节是填充**——一半都浪费了。

第二个版本：

```c
struct sensor_data_compact {    // 起始地址 0x1000
    uint32_t  value;            // offset 0 (4 bytes)
    uint8_t   id;               // offset 4 (1 byte)
    uint8_t   status;           // offset 5 (1 byte)
    // padding: 2 bytes         // offset 6~7 (对齐整个结构体)
};                              // sizeof = 8
```

只浪费 2 字节。**把大的成员放前面，小的放后面，是最简单有效的优化。**

### 结构体的整体对齐

编译器还会确保结构体本身的大小是**最大成员对齐值**的整数倍。这就是为什么 `sensor_data_compact` 末尾还要垫 2 字节——`uint32_t` 要求 4 字节对齐，而结构体大小为 6 不是 4 的倍数，必须补到 8。

> **黄金口诀：结构体对齐 = max(成员对齐) 的整数倍，末尾补 padding 直到满足。**

---

## 二、`__attribute__((packed))`：破除对齐的利刃

GCC 提供了 `__attribute__((packed))` 来强制取消填充：

```c
struct __attribute__((packed)) sensor_data_packed {
    uint8_t   id;               // offset 0
    uint32_t  value;            // offset 1（不对齐！）
    uint8_t   status;           // offset 5
};                              // sizeof = 6
```

`sizeof` 终于变成 6 了！但这把剑有代价——

### packed 的代价

**非对齐访问（unaligned access）**。`value` 在地址 0x1001 上，不是 4 的倍数。在 Cortex-M0/M0+ 上，这种访问会直接触发 **HardFault**。在 Cortex-M3/M4/M7 上，虽然硬件能处理（多次读取后拼接），但会产生额外总线访问，**性能下降 2~5 倍**。

```c
// Cortex-M4 上测试
struct packed_s {
    uint8_t  a;
    uint32_t b;
} __attribute__((packed));

volatile struct packed_s s;
s.b = 0x12345678;   // 3 次总线写操作 vs 正常 1 次
```

### 正确用法

- ✅ **协议解析场景**：接收网络包（TCP/IP 头部）、读取文件系统元数据时，数据是对端按字节排布的，必须用 packed
- ✅ **与硬件共用数据结构**：某些控制器的寄存器按非对齐方式排列
- ❌ **高频访问的热路径**：不要对频繁读写的结构体用 packed

更安全的折中方案是用 `__attribute__((aligned))` 控制整体对齐，而非完全取消：

```c
struct __attribute__((aligned(4))) partial_packed {
    uint8_t  a;
    uint32_t b;
    uint16_t c;
};  // 内部取消填充，但结构体整体按 4 字节对齐
```

---

## 三、位域：把比特用到极致

位域（bit-field）允许以**位（bit）**为单位定义结构体成员，是硬件寄存器映射的标配工具。

### 基本语法

```c
struct status_reg {
    unsigned int ready   : 1;  // bit 0
    unsigned int error   : 1;  // bit 1
    unsigned int mode    : 2;  // bit 2~3
    unsigned int reserved: 4;  // bit 4~7
};
```

冒号后的数字表示占多少位。`sizeof(struct status_reg)` 通常是 4 字节（一个 `unsigned int` 宽度）。

### 实战：硬件寄存器映射

以 STM32 USART 的状态寄存器（SR）为例——实际项目中通常用 `volatile` 指针映射到固定地址：

```c
typedef struct {
    volatile uint32_t SR;       // 0x00: 状态寄存器
    volatile uint32_t DR;       // 0x04: 数据寄存器
    volatile uint32_t BRR;      // 0x08: 波特率寄存器
    volatile uint32_t CR1;      // 0x0C: 控制寄存器 1
    volatile uint32_t CR2;      // 0x10: 控制寄存器 2
    volatile uint32_t CR3;      // 0x14: 控制寄存器 3
    volatile uint32_t GTPR;     // 0x18: 保护时间和预分频器
} USART_TypeDef;

#define USART1  ((USART_TypeDef *)0x40011000)
// 用法：
// while (!(USART1->SR & (1 << 6)));  // 等待 TC 位
```

但位域可以更直观地表达寄存器位：

```c
typedef struct {
    volatile uint32_t PE    : 1;  // bit 0: 奇偶校验错误
    volatile uint32_t FE    : 1;  // bit 1: 帧错误
    volatile uint32_t NE    : 1;  // bit 2: 噪声错误标志
    volatile uint32_t ORE   : 1;  // bit 3: 溢出错误
    volatile uint32_t IDLE  : 1;  // bit 4: 空闲线路检测
    volatile uint32_t RXNE  : 1;  // bit 5: 读数据寄存器非空
    volatile uint32_t TC    : 1;  // bit 6: 发送完成
    volatile uint32_t TXE   : 1;  // bit 7: 发送数据寄存器空
    volatile uint32_t LBD   : 1;  // bit 8: LIN 中断检测
    volatile uint32_t CTS   : 1;  // bit 9: CTS 标志
    volatile uint32_t       : 22; // bit 10~31: 保留
} USART_SR_Bits;

#define USART1_SR  (*(volatile USART_SR_Bits *)0x40011000)
// 用法：
// while (!USART1_SR.TC);  // 等发送完成
```

### 位域的陷阱

**跨平台不可移植。** C 标准对位域的布局只做了最宽松的约定，不同编译器处理方式不一样：

1. **位序（bit ordering）**：有些编译器从低位开始分配（LSB first），有些从高位（MSB first）
2. **分配方向**：位域能否跨过字节边界？不同架构不同
3. **基本类型**：位域底层用 `int` / `unsigned int`，但有些编译器允许 `uint8_t` 等更窄类型

```c
// 危险：以下代码在不同平台行为不同
struct {
    unsigned int a : 3;
    unsigned int b : 3;
    unsigned int c : 2;
} reg;

uint8_t *p = (uint8_t *)&reg;
// p[0] 是第一字节，但 a/b/c 各占几位取决于是小端还是大端
```

> **铁律：位域适合同一平台上的寄存器映射和协议解析，不适合跨平台通信。**

---

## 四、`offsetof` 宏：揭开地址的魔法

`offsetof` 是 `<stddef.h>` 中定义的宏，返回结构体成员在结构体中的字节偏移量：

```c
#include <stddef.h>

struct example {
    uint8_t  a;
    uint32_t b;
    uint16_t c;
};

offsetof(struct example, b);   // 返回 4（跳过 a + padding 3）
offsetof(struct example, c);   // 返回 8
```

### 经典实现

```c
#define offsetof(TYPE, MEMBER)  ((size_t)&((TYPE *)0)->MEMBER)
```

这个宏看似疯狂——对**空指针**解引用！但它利用了编译器的求址机制而不是真正的内存访问：

1. `(TYPE *)0`：假设地址 0 处有一个 TYPE 类型的结构体
2. `->MEMBER`：编译器计算该成员的相对于结构体起始的偏移量（不产生实际代码）
3. `&`：取该成员的地址
4. `(size_t)`：转成整数，就是偏移量

在 Linux 内核中，`offsetof` 被广泛用于**从结构体成员找到包含它的结构体**——`container_of` 宏：

```c
#define container_of(ptr, type, member) ({                      \
    const typeof(((type *)0)->member) *__mptr = (ptr);          \
    (type *)((char *)__mptr - offsetof(type, member));          \
})

// 用法：从链表节点找到包含该节点的结构体
struct list_head *node = ...;
struct sensor_data *data = container_of(node, struct sensor_data, list);
```

---

## 五、实战对比：哪种方案更优？

假设我们需要存储 1000 个传感器数据点，每个点包含温度（`int16_t`）、湿度（`uint8_t`）、标志（`uint8_t` 中的 2 位）。

| 方案 | 定义 | sizeof | 1000 个占用 | 访问速度 | 适用场景 |
|:----|:----|:-----:|:----------:|:--------:|:--------|
| 对齐结构体 | 常规顺序 | 8 | 8 KB | ⚡最快 | 常规场景 |
| packed 结构体 | `__attribute__((packed))` | 4 | 4 KB | 🐢 慢 2~5x | 传输/存储 |
| 位域结构体 | bit-field | 4 | 4 KB | 🐢 稍慢 | 寄存器映射 |
| 手动位移 | `#define GET_TEMP(x) ((x) >> ...)` | 4 | 4 KB | ⚡最快 | 极致优化 |

### 选择建议

- **普通存储** → 对齐结构体（大成员放前面），简单可靠
- **协议/文件格式** → packed 结构体，但要确认目标 CPU 支持非对齐访问
- **硬件寄存器** → 位域 + volatile，可读性最佳
- **极致省 RAM（如几十 KB 的 MCU）** → 手动位移宏或 packed，配合 `memcpy` 安全访问

---

## 六、常见坑与最佳实践

### 坑 1：结构体作为函数参数

按值传递结构体时，实参被整体拷贝到栈上。如果结构体有 100 字节，每调一次函数就多花 100 字节栈空间。

✅ **传指针而不是传值：**

```c
void process(struct sensor_data *data);  // 只传 4 字节指针
// 而不是
void process(struct sensor_data data);   // 拷贝整个结构体
```

### 坑 2：packed 结构体中的 volatile

对 packed 结构体成员加 `volatile`，在非对齐访问时 volatile 语义可能导致额外总线操作。建议拆成两步：

```c
// 不好的做法
volatile struct __attribute__((packed)) {
    uint32_t reg;
} *p;

// 更好的做法：用 memcpy 安全读取
uint32_t val;
memcpy(&val, &p->reg, sizeof(val));
```

### 坑 3：位域的原子性

对位域成员的操作不是原子的——即使是一个位。Cortex-M 的 `set/clear` 操作适合用 `__IO` 宏或 `__set_BIT()` / `__clear_BIT()` 内联函数来完成。

```c
// ❌ 非原子：可能涉及 RMW（读-改-写）
status_reg->ready = 1;

// ✅ 原子（硬件支持）：直接写寄存器值
USART1->SR |= (1 << 5);       // 用移位表达式强制整字写入
```

### 坑 4：sizeof 不是运行时函数

`sizeof` 是**编译期运算符**，不是函数。`sizeof(struct sensor_data)` 在编译时就被常数替换了，没有运行时开销。`offsetof` 同理。

---

## 总结

| 概念 | 本质 | 省内存 | 影响性能 | 可移植性 |
|:----|:----|:-----:|:--------:|:--------:|
| 自然对齐 | CPU 硬件要求 | ❌ | ✅ 提升 | ✅ 通用 |
| padding | 编译器插的对齐填充 | ❌ 浪费 | ✅ 保证 | ✅ 通用 |
| `__attribute__((packed))` | 取消 padding | ✅ 省 | ❌ 降速 | ❌ 依赖 CPU |
| 位域（bit-field） | 按位打包 | ✅ 极致 | ❌ 稍慢 | ❌ 编译器相关 |
| `offsetof` | 编译期计算偏移 | ✅ 零开销 | ✅ 零开销 | ✅ 通用 |

**一条原则：对齐是常态，packed 是例外。** 在大多数情况下，把大成员放到结构体前面，就能在不牺牲性能的前提下把内存浪费降到最低。只在协议解析和寄存器映射时才动用 packed 和 bit-field 这些重型工具。

---

> 🏷️ 技术关键词：C语言 · 结构体 · 内存对齐 · 位域 · packed · offsetof · 嵌入式编程
