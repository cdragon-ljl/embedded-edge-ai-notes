# 嵌入式知识体系 · #12 · 链接脚本逐行解读：从编译到运行的最后一步

在嵌入式开发中，编译通过不等于代码能跑。你写好 `main()`，烧录进去，上电——什么反应都没有。检查了 GPIO 配置、时钟初始化、中断向量表，都没有问题。最后发现是**链接脚本里的一个段地址没对上**。

链接脚本（`.ld` 文件）是编译流程的最后一站，也是嵌入式工程师最不愿意碰的那一块。它长着一张「不是你写的东西出了问题」的脸，但恰恰是它决定了你的程序能不能在硬件上正确运行。

## 一、预处理→编译→汇编→链接全流程回顾

先快速过一遍编译工具链的完整流水线：

```
source.c
    │
    ├─ 预处理 (cpp)     → source.i    (展开宏、处理 #include、条件编译)
    │
    ├─ 编译 (cc1)       → source.s    (C/ASM → ARM 汇编)
    │
    ├─ 汇编 (as)        → source.o    (汇编 → 机器码 / ELF 目标文件)
    │
    └─ 链接 (ld)        → firmware.elf (合并所有 .o、解析符号、分配地址)
                              │
                              ├─ objcopy → firmware.bin   (纯二进制，烧录用)
                              └─ objdump → firmware.lst   (反汇编清单，调试用)
```

前面三步处理的是**代码逻辑**：把人类可读的 C 代码变成处理器可执行的二进制指令。但到了链接阶段，问题从「逻辑是否正确」变成了**「这些指令和数据应该放在内存的哪个位置」**。

**这就是链接脚本回答的问题。**

## 二、`MEMORY` 命令定义内存区域

MCU 的内存不是同构的。STM32F407 的内存分布是：

| 区域 | 起始地址 | 大小 | 属性 |
|------|:-------:|:----:|:----:|
| Flash | 0x08000000 | 1MB | rx（只读+执行） |
| RAM | 0x20000000 | 128KB | rwx（读写+执行） |
| CCM RAM | 0x10000000 | 64KB | rw（仅CPU核可直接访问） |
| Backup SRAM | 0x40024000 | 4KB | rw（掉电保持） |

在 `.ld` 文件中用 `MEMORY` 命令声明：

```ld
MEMORY
{
    FLASH (rx)   : ORIGIN = 0x08000000, LENGTH = 1M
    RAM   (rwx)  : ORIGIN = 0x20000000, LENGTH = 128K
    CCM   (rw)   : ORIGIN = 0x10000000, LENGTH = 64K
}
```

每一行定义了一块**命名的内存区域**，包含起始地址（ORIGIN）、大小（LENGTH）和访问属性（r=读，w=写，x=执行）。链接器在后续的 `SECTIONS` 中会检查：某个输出段是否超出所属区域的边界。

**如果不定义 `MEMORY`，默认工作模式是把所有内容塞进一个连续虚地址空间**——这在 x86 上可能行得通，在 MCU 上必然出问题，因为 Flash 和 RAM 的地址空间是不连续的。

## 三、`SECTIONS` 命令分配段布局

`SECTIONS` 是链接脚本的核心，它告诉链接器：每个输入段（来自 `.o` 文件）应该放到哪个输出段、哪个内存区域、以什么顺序排列。

```ld
SECTIONS
{
    /* 1. 中断向量表——必须从 Flash 起始地址开始 */
    .isr_vector :
    {
        . = ALIGN(4);
        _svector = .;
        KEEP(*(.isr_vector))    /* KEEP 防止 --gc-sections 剪掉 */
        _evector = .;
    } > FLASH

    /* 2. 代码段 */
    .text :
    {
        . = ALIGN(4);
        _stext = .;
        *(.text)                /* 所有 .o 的 .text 段 */
        *(.text*)
        *(.glue_7)
        *(.glue_7t)
        *(.rodata)              /* 只读数据也放 Flash */
        *(.rodata*)
        _etext = .;
    } > FLASH

    /* 3. 初始化数据——运行时复制到 RAM */
    .data : AT(_etext)          /* AT 指定加载地址在 Flash */
    {
        . = ALIGN(4);
        _sdata = .;             /* RAM 中的起始地址 */
        *(.data)
        *(.data*)
        _edata = .;
    } > RAM                     /* VMA = RAM */

    /* 4. 零初始化数据——启动代码会清零 */
    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss)
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > RAM

    /* 5. 堆（可选，有 RTOS 时通常由 OS 管理） */
    .heap :
    {
        . = ALIGN(8);
        _sheap = .;
        . = . + HEAP_SIZE;
        . = ALIGN(8);
        _eheap = .;
    } > RAM

    /* 6. 栈——从 RAM 顶部向下生长 */
    .stack :
    {
        . = ALIGN(8);
        _sstack = .;
        . = . + STACK_SIZE;
        . = ALIGN(8);
        _estack = .;
    } > RAM
}
```

几个关键符号解释：

- **`.` （当前位置计数器）**：链接器内部维护的地址游标。`. = ALIGN(4)` 表示将当前位置对齐到 4 字节边界。
- **`_sdata` / `_edata`**：C 代码中 `extern char _sdata;` 就能获取到这些地址。
- **`AT(_etext)`**：指定 `.data` 段的加载地址（LMA）在 Flash 中紧接着 `.text` 之后。启动代码运行时需要把这段数据从 Flash 复制到 RAM。

## 四、`.data` / `.bss` / `.text` / `.rodata` 的来龙去脉

这是整个嵌入式链接中最核心的概念，也是很多初学者最容易混淆的部分。

| 段名 | 存放内容 | 链接后地址 | 运行时 | 大小确定时机 |
|------|:-------:|:----------:|:------:|:----------:|
| `.text` | 函数指令 | Flash (LMA) | 直接在 Flash 执行 | 编译时 |
| `.rodata` | 字符串/查表常量 | Flash (LMA) | 在 Flash 中只读访问 | 编译时 |
| `.data` | 初始化的全局/静态变量 | RAM (VMA)，加载地址在 Flash | 启动时从 Flash 复制到 RAM | 编译时已知初值 |
| `.bss` | 零初始化的全局/静态变量 | RAM | 启动时清零 | 编译时已知大小 |
| `.stack` | 函数调用栈 | RAM | 自动管理 | 链接时固定大小 |

**一个常见的误解：「函数里的局部变量也在 `.data` 或 `.bss` 里」**——不对。局部变量在栈上，运行到函数时才分配。`.data` 和 `.bss` 只包含全局变量和 `static` 变量。

**为什么 `.data` 需要复制？** 因为全局变量在代码中可能有初始值（比如 `int counter = 100;`），Flash 掉电不丢失但无法原位修改，RAM 可以修改但掉电丢失。所以初值存放在 Flash（LMA），启动时搬运到 RAM（VMA），之后程序操作的是 RAM 中的副本。

## 五、自定义 section 与启动代码的配合

你可以在 C 代码中用 `__attribute__` 把特定变量放进自定义段，然后在链接脚本中精确控制其位置：

```c
// 1. 把一大块 DMA 缓冲区放到 CCM RAM
__attribute__((section(".ccm_ram")))
uint8_t dma_buffer[4096] __attribute__((aligned(32)));

// 2. 函数放到 Flash 的指定区域
__attribute__((section(".fast_code")))
void irq_handler(void) { /* ... */ }
```

链接脚本中增加对应输出段：

```ld
SECTIONS
{
    /* ... 标准段 ... */

    .ccm_ram (NOLOAD) :    /* NOLOAD：不初始化，不烧入 Flash */
    {
        . = ALIGN(32);
        *(.ccm_ram)
        . = ALIGN(32);
    } > CCM

    .fast_code :
    {
        *(.fast_code)
    } > FLASH
}
```

启动代码（`startup_*.s`）里则必须为自定义段做一些事：如果放在 RAM 中需要初始化，就要在 `main()` 之前搬运数据和清零 BSS。

典型的启动汇编伪代码：

```asm
reset_handler:
    /* 复制 .data 段从 Flash 到 RAM */
    ldr  r0, =_sdata        ; VMA 起点 (RAM)
    ldr  r1, =_edata        ; VMA 终点 (RAM)
    ldr  r2, =_sidata       ; LMA 起点 (Flash, 通常等于 _etext)
    bl   memcpy_loop

    /* 清零 .bss */
    ldr  r0, =_sbss
    ldr  r1, =_ebss
    bl   memset_zero

    /* 设置栈指针 */
    ldr  sp, =_estack

    /* 跳转 main */
    bl   main
```

**如果链接脚本里定义了新段，但没有在启动代码中加搬运/清零逻辑，变量初值就是随机的。** 这是新手最容易踩的坑——链接脚本改了，启动代码忘了同步。

## 小结

链接脚本是嵌入式开发中被低估的关键环节。`MEMORY` 描述硬件的物理内存划分，`SECTIONS` 决定代码和数据在其中的布局，`AT` 处理 LMA 与 VMA 之间的映射。**看懂链接脚本 = 理解你写的程序最终是怎么存在于硬件上的。** 下次 MCU 启动后程序跑飞，除了检查时钟和 GPIO，也多看一眼你的 `.ld` 文件——问题往往就在你不敢看的那一行。

> 🏷️ 链接脚本 MEMORY SECTIONS LMA VMA 启动代码 .data .bss
