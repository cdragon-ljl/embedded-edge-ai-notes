# 嵌入式知识体系 · #18 · 启动文件 startup_xxx.s 深度解析：从 Reset 到 main

嵌入式开发者写了几百个 `main()`，但很少有人问：**MCU 上电后第一条指令执行的是什么？全局变量为什么能在 main 之前就赋好初值？** 答案全藏在那个不起眼的 `startup_xxx.s` 启动文件里。

启动文件是芯片厂商提供的一段汇编代码，负责把 CPU 从复位状态带到 C 语言 `main()` 环境。它不是"可选配置"——没有它，你的程序根本跑不起来。本文将带你逐段解析启动文件的每个环节，并画出一条完整的开机时间线。

---

## 一、中断向量表：芯片的"通讯录"

启动文件的第一个"作品"就是中断向量表。以 ARM Cortex-M 为例，向量表放在 Flash 的首地址（0x08000000 或 0x00000000），格式固定：

```asm
; startup_stm32f407xx.s（简化）
                .section .isr_vector, "a", %progbits
                .type     g_pfnVectors, %object
                .size     g_pfnVectors, .-g_pfnVectors

g_pfnVectors:
                .word     _estack              ; 0: 栈顶地址（SP 初始值）
                .word     Reset_Handler        ; 1: 复位向量（PC 初始值）
                .word     NMI_Handler          ; 2: NMI 异常
                .word     HardFault_Handler    ; 3: HardFault
                .word     MemManage_Handler    ; 4: MemManage
                .word     BusFault_Handler     ; 5: BusFault
                .word     UsageFault_Handler   ; 6: UsageFault
                .word     0, 0, 0, 0           ; 7-10: 保留
                .word     SVC_Handler          ; 11: SVCall
                .word     DebugMon_Handler     ; 12: DebugMon
                .word     0                    ; 13: 保留
                .word     PendSV_Handler       ; 14: PendSV
                .word     SysTick_Handler      ; 15: SysTick
                ; 以下为外设中断（编号从 16 开始）
                .word     WWDG_IRQHandler      ; Window WatchDog
                .word     PVD_IRQHandler       ; PVD through EXTI
                ; ... 每个外设一个表项
```

**Cortex-M 的独特之处：** 复位后硬件自动从地址 0x00000000 加载 SP，从 0x00000004 加载 PC，跳转到 `Reset_Handler`。这不同于传统 ARM（PC 从 0 开始执行指令），完全是硬件设计的魔法。

> 向量表的位置由 `__Vectors` 标号定义，链接脚本控制它放哪个 Section。如果你做 IAP（应用内编程），需要在 0x08000000 + N 偏移处放第二张向量表，并修改 SCB->VTOR 寄存器。

---

## 二、Reset_Handler：一切从这里开始

`Reset_Handler` 是系统上电/复位后第一个执行的代码。它的工作分四步：

### 2.1 搬运 .data 段

C 语言中写了 `uint32_t counter = 100;` —— 这个 100 存在哪里？

- **编译时：** `100` 这个值放在 Flash 的 `LOADADDR(.data)` 区域（也叫 **LMA**，Load Memory Address）
- **运行时：** `counter` 变量在 SRAM 的 `.data` 段（也叫 **VMA**，Virtual Memory Address）

启动文件必须把 100 从 Flash 拷贝到 SRAM：

```asm
Reset_Handler:
    ; 1. 复制 .data 段：Flash → SRAM
    ldr   r0, =_sidata       ; Flash 中 .data 的源地址（LMA）
    ldr   r1, =_sdata        ; SRAM 中 .data 的目标地址（VMA）
    ldr   r2, =_edata        ; SRAM 中 .data 的结束地址
    subs  r2, r2, r1         ; 计算 .data 段大小
    ble   .L_loop_data       ; 如果大小为 0，跳过
.L_copy_data:
    ldrb  r4, [r0], #1       ; 从 Flash 读 1 字节
    strb  r4, [r1], #1       ; 写入 SRAM
    subs  r2, r2, #1         ; 计数减 1
    bne   .L_copy_data       ; 循环直到拷贝完
.L_loop_data:
```

### 2.2 清零 .bss 段

未初始化或初始化为 0 的全局变量（`static int flag;`、`uint8_t buffer[1024];`）放在 `.bss` 段。SRAM 上电后内容是随机的，必须清零：

```asm
    ; 2. 清零 .bss 段
    ldr   r1, =_sbss         ; .bss 起始地址
    ldr   r2, =_ebss         ; .bss 结束地址
    movs  r3, #0
    subs  r2, r2, r1
    ble   .L_loop_bss
.L_zero_bss:
    strb  r3, [r1], #1
    subs  r2, r2, #1
    bne   .L_zero_bss
.L_loop_bss:
```

> **⚠️ 字节搬运 vs 字搬运**：上面的示例按字节搬运，效率低。实际芯片的启动文件会用 `ldmia`/`stmia` 一次搬运 4 字（16 字节），速度提升 4 倍。但为了可读性，这里用字节方式展示。

### 2.3 调用 SystemInit()

C 环境还没准备好？不，现在可以调 C 函数了——SP 已设、RAM 已初始化：

```asm
    ; 3. 系统时钟初始化
    bl    SystemInit
```

`SystemInit()` 是 CMSIS 标准定义的函数，由芯片厂商实现。它完成：
- 切换 HSI/HSE 振荡器
- 配置 PLL 锁相环倍频
- 设置 AHB/APB1/APB2 分频系数
- 配置 Flash 等待周期（适配高主频）

以 STM32F407 为例，`SystemInit()` 将 HSE 8MHz → PLL 倍频 168MHz（主频），并确保 APB1 ≤ 42MHz、APB2 ≤ 84MHz。

### 2.4 __libc_init_array() 与全局构造函数

C++ 和 GNU C 中，你可以在全局作用域写构造函数（或 `__attribute__((constructor))`）：

```c
// 这个函数在 main() 之前自动执行！
__attribute__((constructor))
void early_init(void) {
    init_hardware_early();
}
```

`__libc_init_array()` 遍历 `.init_array` 段，逐一调用所有构造函数指针：

```asm
    ; 4. 调用全局构造函数
    bl    __libc_init_array
```

对于纯 C 项目，`__libc_init_array` 也可能不存在——这时链接脚本里根本没有 `.init_array` 段。

### 2.5 跳转到 main()

最后：

```asm
    ; 5. 进入主程序
    bl    main
    ; main 返回后执行（嵌入式场景基本不会返回）
    bl    exit
```

如果 `main()` 意外返回，`exit()` 会调用注册的 `atexit` 函数，然后死循环。

---

## 三、弱定义：每个异常/中断都有"保底处理"

启动文件末尾通常会为所有中断提供一个**弱定义（weak）**的默认处理函数：

```asm
                .weak     HardFault_Handler
                .type     HardFault_Handler, %function
HardFault_Handler:
                b         .              ; 死循环
```

`weak` 属性意味着：如果用户在 C 代码中写了同名的 `HardFault_Handler()`，链接器优先选用户的；如果没写，就链接这个默认的死循环版本。这种设计让初学者不用为每个中断写 handler，也能保证程序不会跑飞。

---

## 四、完整的开机时间线

从按下复位到 `main()` 第一条语句，完整的流程：

```
时间    事件                    说明
───     ────                    ────
T+0     CPU 复位释放            硬件从 0x0000 加载 SP, PC
T+1     Reset_Handler 开始      硬件跳入启动文件的汇编入口
T+2     关闭全局中断            确保初始化不受中断干扰
T+3     搬运 .data              从 Flash 拷贝到 SRAM
T+4     清零 .bss               初始化全局/静态零值变量
T+5     调用 SystemInit()       配置时钟树（HSE→PLL→168MHz）
T+6     调用 __libc_init_array() 执行全局构造函数
T+7     跳转 main()             C 世界正式开始
        ├─ 硬件初始化            HAL_Init(), MX_GPIO_Init() 等
        ├─ 外设配置              USART, SPI, I2C, Timer...
        └─ while(1){...}        主循环
```

总耗时：在 168MHz STM32F4 上约 **200-500 μs**（取决于 .data/.bss 大小和 Flash 读取速度）。

---

## 五、常见陷阱与调试技巧

### 5.1 栈溢出在 main 之前

如果 `_estack` 指向的地址超过实际 SRAM 范围，或者在 .data 拷贝前就调用了函数（压栈），程序会在 `SystemInit()` 里面 HardFault。**症状：** 调试器看到 `Reset_Handler` 执行后就进异常。

**排查法：** 检查链接脚本中 `_estack` 计算是否正确（通常是 SRAM 起始 + 大小）。

### 5.2 .data 段复制方向反了

常见笔误：把 `_sidata`（Flash 源）和 `_sdata`（RAM 目标）搞反。结果从 SRAM 向 Flash 拷贝，读出来全是垃圾。**症状：** 全局变量的初值全是乱码。

### 5.3 VTOR 没改导致 IAP 失败

做了 Bootloader + App 分区，应用区放在 0x08010000，但向量表还在 0x08000000。中断触发时 CPU 读到的还是 Bootloader 的向量表。

**解决：** 在 `SystemInit()` 或 `Reset_Handler` 早期加入：

```c
SCB->VTOR = 0x08010000;  // 重定向向量表到 App 分区
```

### 5.4 .bss 清零过大导致延时可见

大数组（如 LCD 显存 `uint16_t lcd_buffer[320*240];`）在 .bss 段。400μs 的启动延迟在某些场景（如快速重连外设）不可接受。一个做法是把大数组放进独立的 `.noinit` 段，由程序员手动初始化。

---

## 六、不同架构的启动对比

| 架构 | 向量表位置 | 启动方式 | 关键差异 |
|------|-----------|---------|----------|
| Cortex-M (STM32) | Flash 首地址 | 硬件加载 SP+PC | 无需汇编跳转，向量表包含栈顶 |
| Cortex-A (i.MX6ULL) | 固定地址 | ROM Code 加载 SPL | 多级 boot，u-boot SPL 负责 DDR 初始化 |
| RISC-V (GD32V) | 可配置 mtvec | 从 0 取第一条指令 | 无统一标准，ECLIC/CLIC 不同 |
| ESP32 (Xtensa) | ROM + Flash | 一级 Bootloader 加载 | 两级 boot + flash cache 初始化 |

---

## 七、动手实验：改编启动文件

克隆一份官方启动文件，自己动手改一下，理解会更深刻：

1. 在 `.data` 拷贝前后插入 GPIO 翻转——用示波器量拷贝耗时
2. 强行删掉 `__libc_init_array` 调用，看全局构造函数是否还执行
3. 手动修改 `_estack` 为非法地址，观察芯片行为

> 启动文件不是黑盒，它只是用汇编写的一段"初始化保姆代码"。理解了它的每个步骤，你就真正掌握了嵌入式程序从通电到运行的全部秘密。

> 🏷️ 启动文件 | startup.s | Reset_Handler | 中断向量表 | .data 段搬运 | .bss 清零 | SystemInit | __libc_init_array | VTOR | 链接脚本 | Cortex-M | ARM 汇编 | 嵌入式底层
