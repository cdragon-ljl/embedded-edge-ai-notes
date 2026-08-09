# 嵌入式知识体系 · #09 · Rust 嵌入式编程：no_std 裸机开发实战

如果你的嵌入式开发只停留在 C 语言阶段，那你一定听说过 Rust 在嵌入式领域的「黑马」名声——零成本抽象、无垃圾回收、所有权系统在编译期消灭内存安全漏洞。但真正动手时，很多人第一个困惑就是：**为什么我的 Rust 项目不能 `println!`？为什么 `Vec` 用不了？**

这就是我们要聊的——**no_std**。

## 一、no_std vs std：何时放弃标准库

Rust 的标准库（std）依赖操作系统提供的基础设施：文件系统、网络套接字、线程、堆分配器。当你的目标平台是裸机 MCU（STM32、ESP32、RP2040 等）时，这些统统不存在。因此 Rust 嵌入式项目默认使用 `#![no_std]` 属性。

```rust
// no_std 入口
#![no_std]
#![no_main]

extern crate panic_halt; // 一个只执行 wfi 的 panic 处理

#[no_mangle]
pub extern "C" fn main() -> ! {
    loop {}
}
```

**no_std 和 std 的关键区别：**

| 特性 | std | no_std |
|------|:---:|:------:|
| `Vec` / `Box` / `String` | ✅ 可用 | ❌ 需 alloc |
| `println!` / 文件 I/O | ✅ | ❌ 需自己实现 |
| `Box` / `Rc` | ✅ | ❌ 需 alloc + 全局分配器 |
| `core` 库 | 包含 | ✅ 完全可用 |
| 堆分配器 | 系统提供 | 需手动实现 |

> 一种常见的误解是：no_std = 没有堆。实际上只要你实现一个全局分配器，`alloc` crate 的 `Vec`、`Box`、`String` 一样能用。但很多低端 MCU 只有几 KB RAM，用堆反而是个坑。

核心原则是：**能用 `core` 就不用 `std`，能静态分配就不用堆。**

## 二、所有权与借用检查如何防止内存 Bug

C 嵌入式的三大噩梦：野指针、释放后使用（use-after-free）、缓冲区溢出。Rust 编译器在编译期就堵死了这些路。

```rust
// 用 C 写一个常见的 DMA 缓冲区问题
uint8_t *buf = malloc(256);
free(buf);
// ... 100 行之后 ...
dma_transfer(buf, 256); // 已释放！UB！

// Rust 等效代码——编译不通过
let buf: &mut [u8] = &mut [0u8; 256];
drop(buf); // 所有权转移
dma_transfer(buf, 256); // ❌ 编译错误：buf 已被移动
```

**在裸机场景下，所有权系统的一个关键用法是「外设单例」**。外设寄存器映射在特定内存地址上，整个系统的全局唯一性由 Rust 的类型系统保证：

```rust
use core::ptr::NonNull;

struct Uart {
    base: NonNull<u32>,
}

// !Send 和 !Sync——防止多个上下文同时操作 UART
unsafe impl Send for Uart {}   // 单核 MCU 上安全
// 不实现 Sync——防止共享引用

fn main() -> ! {
    let uart = Uart { base: NonNull::new(0x4001_3800 as *mut _).unwrap() };
    // 所有权在 main 中独占，不允许其他函数拿走
    loop {}
}
```

MCU 的每个外设寄存器、每个中断控制器都天然具有「一次只有一个所有者」的特性——和 Rust 的所有权模型天作之合。

## 三、embedded-hal 生态：统一外设抽象

这是 Rust 嵌入式生态的杀手锏。**embedded-hal** 定义了一套 trait，让外设驱动独立于具体芯片：

```rust
// embedded-hal v1.0 中的数字输出 pin
pub trait OutputPin {
    fn set_high(&mut self) -> Result<(), Self::Error>;
    fn set_low(&mut self) -> Result<(), Self::Error>;
}

pub trait DelayNs {
    fn delay_ns(&mut self, ns: u32);
}
```

**这意味着什么？** 你写一个 SSD1306 OLED 驱动，只要依赖 `embedded-hal` 的 `I2C` 和 `DelayMs` trait，它就能在 STM32、ESP32、nRF52、RP2040——任何实现了这些 trait 的 HAL 上运行，一行代码不改：

```rust
// 一段驱动代码，跨平台
use embedded_hal::delay::DelayNs;
use embedded_hal::i2c::I2c;

pub struct Ssd1306<I2C> {
    i2c: I2C,
    addr: u8,
}

impl<I2C: I2c, D: DelayNs> Ssd1306<I2C> {
    pub fn init(&mut self, delay: &mut D) -> Result<(), Error> {
        // 通用的初始化时序
        delay.delay_ms(100);
        self.i2c.write(self.addr, &[0x00, 0xAE])?;
        Ok(())
    }
}
```

当前生态已经覆盖了绝大多数常见外设：传感器（BME280、MPU6050）、显示（SSD1306、ST7735）、无线（nRF24、LoRa）。**换芯片就像换货架上的产品，驱动代码无需改。**

## 四、FFI：Rust 与 C 的互操作实践

现实世界没有完美的绿场开发。你的 MCU 可能有成熟的 C 驱动库、FreeRTOS 移植、或者厂家提供的 HAL。Rust 必须能优雅地和 C 共存。

```rust
// 声明 C 函数
extern "C" {
    static HAL_UART_Handle: u32;
    fn HAL_UART_Transmit(huart: *const u32, data: *const u8, len: u16, timeout: u32) -> u32;
}

fn uart_send(data: &[u8]) -> Result<(), u32> {
    let ret = unsafe {
        HAL_UART_Transmit(
            &HAL_UART_Handle,
            data.as_ptr(),
            data.len() as u16,
            1000,
        )
    };
    if ret == 0 { Ok(()) } else { Err(ret) }
}
```

关键实践原则：

1. **`unsafe` 是隔离的，不是放任的**——把 unsafe 封装在安全接口内，外部使用者不需要碰 `unsafe`。
2. **`#[repr(C)]` 保证布局兼容**——Rust 结构体默认布局未定义，FFI 边界必须指定 C 布局。
3. **链接器不分语言**——Rust 编译出的 `.o` 和 C 编译出的 `.o` 一起链接，入口由 Rust 或 C 定义均可。

一种常见的混合架构：**C 提供底层 HAL + RTOS，Rust 提供业务逻辑和协议栈**。C 的链接脚本和启动汇编不动，Rust 编译成 `.o` 参与链接。

## 五、给 C 工程师的 Rust 入门对照

| C 概念 | Rust 对应 | 说明 |
|--------|-----------|------|
| `int x = 0;` | `let mut x: i32 = 0;` | `let` 默认不可变，`mut` 可变 |
| `#define GPIOA ((GPIO_TypeDef*)0x40020000)` | `const GPIOA: *const Gpio = 0x40020000 as *const _;` | 指针操作需要 `unsafe` |
| `void foo(int x)` | `fn foo(x: i32)` | 最后表达式不加分号即返回值 |
| `int* p = &x;` | `let p = &x;` | 引用有生命周期约束 |
| `struct { int a; char b; }` | `struct Foo { a: i32, b: u8 }` | `#[repr(C)]` 保证 C 兼容布局 |
| `if (x) { ... }` | `if x != 0 { ... }` | Rust 不会隐式转 bool |
| `enum { A, B, C };` | `enum E { A, B, C }` | Rust 枚举可以携带数据 |
| `volatile uint32_t*` | `core::ptr::read_volatile` / `write_volatile` | 没有关键字，用函数 |
| 条件编译 `#ifdef` | `#[cfg(feature = "...")]` | 属性级别更精细 |

最让 C 工程师感到不适的可能是**所有权转移**和**生命周期标注**，但恰恰是这两样东西让你解放了调试内存问题的时间。一旦跨过门槛，你会发现自己已经好几个月没遇到段错误了。

## 小结

Rust 在嵌入式领域不是「更好用的 C」，而是一种**更安全的系统编程范式**。no_std 环境让它在最小的 MCU 上也能跑；embedded-hal 的 trait 抽象让驱动真正可移植；所有权系统则在编译期替你挡下了成千上万个运行时崩溃。如果你正在用 C 写一个超过 5000 行的嵌入式项目，尝试用 Rust 重写其中一小块模块——你会回来的。

> 🏷️ 嵌入式 Rust no_std embedded-hal FFI 裸机编程 所有权系统
