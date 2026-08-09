# 嵌入式知识体系 · #13 · ARM 汇编基础：Thumb / Thumb-2 与 AAPCS 调用约定

很多嵌入式工程师对汇编的态度是：「这辈子不会手写，但反汇编一定要能看懂。」

这个态度很务实——实际工作中，你很少需要从头写汇编文件，但分析 crash dump、优化临界区、理解启动代码、debug 反汇编窗口——这些场景几乎每周都遇到。如果看不懂 ARM 汇编，你就像在迷雾中做调试。

这篇从指令集演进讲到调用约定，目标是：**下次看反汇编或 crash dump 时，你能认出每条指令在做什么，以及栈上的数据是怎么组织的。**

## 一、ARM vs Thumb vs Thumb-2 指令集演进

ARM 指令集经历了几次大的架构变更，理解这些变更是看懂反汇编的前提：

| 指令集 | 指令宽度 | 出现于 | 特点 |
|:------:|:-------:|:-------:|:---:|
| ARM | 32-bit | ARMv4+ | 功能齐全，但代码密度低 |
| Thumb | 16-bit | ARMv4T | 压缩指令，密度高，功能受限 |
| Thumb-2 | 16/32-bit 混编 | ARMv6T2+ | 兼顾密度和功能，Cortex-M 标配 |

**关键转折点：Cortex-M 系列只支持 Thumb/Thumb-2，不支持 ARM 指令集。** 如果你看到 `LDR R0, =0x40020000` 这样的指令以为自己不认识——别紧张，它只是一个被汇编器转成 PC 相对加载的伪指令。

看懂 Thumb-2 的几个高频指令比全部背下来更实用：

```assembly
@ 数据传输类
LDR   R0, [R1, #4]      @ 从 R1+4 加载 32 位到 R0
STR   R0, [R1]          @ 将 R0 写入 R1 指向的地址
LDRB  R0, [R1]          @ 加载 1 个字节（零扩展）
STRB  R0, [R1]          @ 存储 1 个字节
LDMIA R0!, {R1-R4}      @ 从 R0 加载多个寄存器，R0 自增
STMIA R0!, {R1-R4}      @ 存储多个寄存器到 R0 指向的内存

@ 算术逻辑类
ADD   R0, R1, R2        @ R0 = R1 + R2
SUB   R0, R1, #1        @ R0 = R1 - 1
MOV   R0, #0xFF         @ R0 = 0xFF
AND   R0, R1, R2        @ R0 = R1 & R2
ORR   R0, R1, R2        @ R0 = R1 | R2
LSL   R0, R1, #2        @ R0 = R1 << 2 (逻辑左移)
LSR   R0, R1, #2        @ R0 = R1 >> 2 (逻辑右移)

@ 分支跳转类
B     label             @ 无条件跳转（±2KB 范围）
BL    func              @ 带链接跳转（用于函数调用）
BX    LR                @ 返回（跳转到 LR 中的地址）
CBZ   R0, label         @ 如果 R0 == 0 则跳转
CBNZ  R0, label         @ 如果 R0 != 0 则跳转
```

一个快速识别技巧：**看指令的第一个数字**——16-bit Thumb 指令的操作码通常短于 32-bit Thumb-2 指令。Cortex-M 反汇编时，指令地址是 2 的倍数（Thumb 地址的 bit[0] 固定为 1），但指令本身可能是 2 字节或 4 字节。

## 二、AAPCS 调用约定的寄存器分工

AAPCS（ARM Architecture Procedure Call Standard）定义了函数调用时寄存器的职责分工。这是理解反汇编的基石：

| 寄存器 | 别名 | 功能 | 调用者保存？ |
|:------:|:----:|:---:|:----------:|
| R0-R3 | a1-a4 | 参数传递 / 返回值 | ✅ 调用者保存 |
| R4-R11 | v1-v8 | 局部变量 | ❌ 被调用者保存 |
| R12 | IP | 内部调用暂存（intra-procedure-call scratch） | ✅ 不保存 |
| R13 | SP | 栈指针 | 永远不参与普通传参 |
| R14 | LR | 链接寄存器（保存返回地址） | 被调用者保存 |
| R15 | PC | 程序计数器 | — |

**规则速记：**

- **前 4 个参数**放在 R0~R3。第 5 个参数开始压栈。
- **返回值**在 R0（32 位）或 R0+R1（64 位）。
- **R4~R11** 由被调用函数负责保存。如果一个函数要用 R4，必须先在函数入口处压栈，退出时恢复。
- **R12 (IP)** 临时寄存器，函数调用过程中可能会被链接器生成的 veneer 代码覆盖——不要在函数调用之间期望它的值不变。

看一个实际的函数调用反汇编：

```c
// C 代码
int add_four(int a, int b, int c, int d, int e) {
    return a + b + c + d + e;
}
```

```assembly
@ 反汇编之后的 ARM 代码（示意）
add_four:
    @ R0=a, R1=b, R2=c, R3=d, (e 在栈上)
    ADD   R0, R0, R1          @ R0=a+b
    ADD   R0, R0, R2          @ R0+=c
    ADD   R0, R0, R3          @ R0+=d
    LDR   R3, [SP, #0]        @ 从栈取第五个参数 e
    ADD   R0, R0, R3          @ R0+=e
    BX    LR                  @ 返回，返回值在 R0
```

## 三、栈帧结构与函数序言/尾声

当一个函数使用超过 4 个局部变量、或者需要调用其他函数（需要使用 LR），编译器会在栈上开辟一块区域——**栈帧**。

```assembly
@ 典型的 Cortex-M 函数序言 (prologue)
my_function:
    PUSH  {R4-R7, LR}         @ 保存被调用者保存的寄存器 + LR
    SUB   SP, SP, #32         @ 为局部变量分配 32 字节
    @ ... 函数体 ...

@ 典型的函数尾声 (epilogue)
    ADD   SP, SP, #32         @ 回收局部变量空间
    POP   {R4-R7, PC}         @ 恢复寄存器并返回
    @ 注意：POP {..., PC} 等价于恢复 LR 后再 BX LR
    @ Cortex-M 硬件直接支持 POP 到 PC 并完成返回
```

**分析 crash dump 时的关键思维框架：**

```
当前 SP → [局部变量空间]       ← SP + 偏移
           [R7 (备份)]
           [R6 (备份)]
           [R5 (备份)]
           [R4 (备份)]
           [LR (返回地址)]    ← 函数返回时 POP 到 PC
上层 SP →  ...
```

当发生 HardFault 时，MSP/PSP 指向的栈帧中存储了发生异常前的 R0-R3、R12、LR、PC、xPSR。**查看 `PC` 的备份值（通常叫 `saved_pc`）**，就到了发生异常的指令地址，再对照反汇编清单定位具体代码行。

## 四、异常入口的汇编处理

Cortex-M 的异常处理是硬件自动完成的——这比传统 ARM7/ARM9 手动压栈省很多事。但启动代码和中断处理函数仍然需要一些汇编包装。

**硬件自动保存的寄存器（入异常时）**：xPSR, PC, LR, R12, R3, R2, R1, R0——共 8 个 32 位寄存器，32 字节。CPU 自动压入当前栈（MSP 或 PSP）。

```assembly
@ Cortex-M 的 PendSV 处理——RTOS 上下文切换的核心
@ 典型的 PendSV_Handler 实现
PendSV_Handler:
    @ 关闭中断，防止保存时被打断
    CPSID   I

    @ 检查是否发生了嵌套
    MRS     R0, PSP           @ 获取当前 PSP
    CBZ     R0, skip_save     @ PSP=0 表示第一次切换

    @ 保存当前任务的寄存器
    STMDB   R0!, {R4-R11}     @ 保存 R4~R11 到任务栈
    @ 更新任务的栈指针
    LDR     R1, =current_tcb
    STR     R0, [R1]          @ TCB->sp = R0

skip_save:
    @ 切换到下一个任务
    LDR     R1, =next_tcb
    LDR     R2, [R1]
    LDR     R0, [R2]          @ 新任务的栈指针

    @ 恢复新任务的寄存器
    LDMIA   R0!, {R4-R11}     @ 弹出 R4~R11
    MSR     PSP, R0           @ 更新 PSP

    @ 恢复中断
    CPSIE   I

    @ 异常返回——硬件自动恢复 R0-R3, R12, LR, PC, xPSR
    BX      LR
```

这段汇编是 FreeRTOS、RT-Thread、uCOS 等 RTOS 上下文切换的核心。看懂它需要三样东西：**理解 AAPCS 的寄存器分工、知道硬件自动压栈的 8 个寄存器、熟悉多寄存器加载/存储指令（LDMIA/STMDB）。**

如果你能完整解释 `STMDB R0!, {R4-R11}` 和 `LDMIA R0!, {R4-R11}` 做了什么，以及为什么不需要保存 R0-R3——说明你已经掌握了 ARM 汇编在嵌入式领域的核心应用场景。

## 小结

ARM 汇编不必畏惧。把它当成一门有规律的符号语言来理解：**R0-R3 传参数、R4-R11 存变量、R13 管栈、R14 记返回地址、R15 指下一条指令。** Thumb-2 混编指令看似复杂，但反汇编中 80% 的场景只是 `LDR/STR/ADD/SUB/B/BLX` 这几种。下次 HardFault 找不出原因时，与其在 C 代码里加九个 `printf`，不如打开 `.lst` 文件找那条 `saved_pc` 对应的指令——答案通常就在那里。

> 🏷️ ARM汇编 Thumb Thumb-2 AAPCS 栈帧 Cortex-M 异常处理
