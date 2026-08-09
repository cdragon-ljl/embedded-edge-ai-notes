# 嵌入式知识体系 · #08 · C++11/14/17/20 在嵌入式中的落地实践

---

许多嵌入式开发者对 C++ 的印象还停留在「类 + 虚函数 = 很重」。事实上，C++11 之后的现代 C++（Modern C++）提供了大量**零开销抽象**的特性，特别适合嵌入式场景。本文逐一落地这些特性，每个都配合实际例子。

---

## 一、`auto`：让类型推导替你工作

### 基本用法

```cpp
// ❌ 以前：写冗长的类型
std::vector<uint16_t>::iterator it = vec.begin();
uint32_t volatile * const reg = reinterpret_cast<uint32_t volatile *>(0x40020000);

// ✅ 现代 C++：让 auto 推导
auto it = vec.begin();
auto reg = reinterpret_cast<uint32_t volatile *>(0x40020000);
```

### 嵌入式实战：寄存器定义

```cpp
// 使用 auto 简化外设结构体指针
auto usart = reinterpret_cast<USART_TypeDef *>(USART1_BASE);
auto tim   = reinterpret_cast<TIM_TypeDef *>(TIM2_BASE);

// 修改：清晰且简洁
usart->BRR = 0x16E;  // 115200 @ 16 MHz
tim->ARR  = 999;     // 自动重装值
```

### 范围 `for` 循环

```cpp
// 传统方式
for (size_t i = 0; i < NUM_CHANNELS; i++)
    adc_values[i] = read_adc(i);

// 范围 for + auto&
uint16_t adc_values[8];
for (auto &val : adc_values)   // auto& → 引用，可修改
    val = read_adc(&val - adc_values);

// 只读访问用 const auto&
for (const auto &v : adc_values)
    printf("ADC: %d\n", v);
```

**性能一致**：范围 for 展开后和传统 for 循环的汇编完全一样。

---

## 二、`nullptr` vs `NULL`：这不是名字的差别

### 根本区别

```c
// C 中：
#define NULL ((void *)0)
```

```cpp
// C++ 中：
// NULL 通常是 0（整数零），不是指针！

void func(int *p)   { printf("pointer\n"); }
void func(int n)    { printf("int\n"); }

func(NULL);    // C++ 输出: int（NULL 被解析为整型 0）
func(nullptr); // C++ 输出: pointer（nullptr 是真正的空指针常量）
```

### 嵌入式场景的陷阱

```cpp
// ❌ NULL 导致重载决议错误
UART_HandleTypeDef *huart = NULL;  // 如果 NULL 是 0，类型是 int
if (huart == NULL) { ... }         // OK：指针与 0 比较

// ✅ nullptr 保证类型安全
UART_HandleTypeDef *huart = nullptr;
if (huart == nullptr) { ... }

// 模板中零和空指针的区分至关重要
template <typename T>
bool is_valid(T *p) {
    return p != nullptr;  // ✅ 正确
    // return p != 0;     // ❌ 也可工作，但语义不清晰
}
```

**黄金法则**：所有新代码用 `nullptr`，不再使用 `NULL`。

---

## 三、`std::array`：C 数组的现代替代

### C 数组的问题

```cpp
// 问题 1：退化到指针
void process(int arr[5]) {
    // arr 其实是 int*，丢失了长度信息
    sizeof(arr);  // = sizeof(int*) = 8（不是 20）
}

// 问题 2：没有 begin/end
// 问题 3：赋值是地址拷贝，不是元素拷贝
int a[5] = {1,2,3,4,5};
int b[5] = a;  // ❌ 编译错误！不能数组赋值
```

### `std::array` 的解决方案

```cpp
#include <array>

// 明确大小，不退化
void process(std::array<int, 5> &arr) {
    arr.size();       // = 5（不会丢长度）
    arr.begin();      // 迭代器
    arr.data();       // 需要裸指针时
}

void test_array(void) {
    // 声明 & 初始化
    std::array<uint16_t, 8> adc_vals = {0};
    // 等价于 uint16_t adc_vals[8] = {0};

    // 安全访问
    adc_vals[3] = 1024;          // ✅ operator[]（无边界检查）
    adc_vals.at(15);             // ✅ at()（有边界检查，抛异常）
                                 // 嵌入式建议禁用异常，用 []

    // 赋值
    std::array<uint16_t, 8> backup;
    backup = adc_vals;           // ✅ 可以整体赋值！

    // 范围 for
    for (auto val : adc_vals)
        printf("%d\n", val);

    // 零开销
    static_assert(sizeof(std::array<uint16_t, 8>) == sizeof(uint16_t[8]));
    // ✅ 大小完全一致，无额外开销
}
```

### 实战：查找表

```cpp
// ✅ 用 std::array 替代 C 数组
constexpr std::array<uint8_t, 256> sin_lut = {
    #include "sin_table.inc"   // 同样可以用 include 注入
};

// 编译期检查：确保 LUT 大小正确
static_assert(sin_lut.size() == 256, "sin LUT must have 256 entries");

// 访问
uint8_t sample = sin_lut[index & 0xFF];  // 与 C 数组一样高效
```

---

## 四、lambda 表达式：闭包机制与应用

### 基本语法

```cpp
auto greet = []() {
    printf("Hello from lambda!\n");
};

greet();  // 调用

// 带参数 + 返回值
auto sum = [](int a, int b) -> int { return a + b; };
printf("%d\n", sum(3, 4));  // 7
```

### 捕获：访问外部变量

```cpp
uint32_t counter = 0;

// [&]：按引用捕获所有变量
auto inc = [&]() { counter++; };

// [=]：按值捕获所有变量（不推荐在嵌入式使用，浪费栈空间）
// [this]：捕获当前对象指针
// [&counter]：只捕获 counter 的引用

inc();  // counter → 1
```

### 实战：回调函数注册

```cpp
// 传统方式：定义全局/静态函数
static void timer_callback(void) {
    // 不能访问调用者上下文
}

// ✅ lambda 方式：捕获上下文
class PWMManager {
public:
    void start(void) {
        // 用 lambda 替代普通回调函数
        timer_.on_timeout([this]() {
            // this 指针被捕获，可以访问成员变量
            toggle_pin(output_pin_);
        });
    }

private:
    uint16_t output_pin_;
    Timer timer_;
};
```

### 实战：排序/过滤

```cpp
struct SensorReading {
    uint32_t timestamp;
    int16_t  value;
};

std::array<SensorReading, 32> readings;

// 按时间戳排序
std::sort(readings.begin(), readings.end(),
    [](const auto &a, const auto &b) {
        return a.timestamp < b.timestamp;
    });

// 过滤异常值
auto valid_end = std::remove_if(readings.begin(), readings.end(),
    [](const auto &r) { return r.value < 0; });
```

### lambda vs 函数指针

```cpp
// 无捕获的 lambda 可以隐式转为函数指针
using Cb = void(*)(int);
Cb cb = [](int x) { printf("%d", x); };  // ✅ 无捕获，可转换

// 有捕获的 lambda 不能转为函数指针
int threshold = 100;
// Cb cb2 = [threshold](int x) { printf("%d", x); };  // ❌ 编译错误
```

> **嵌入式提示**：如果硬件外设需要注册函数指针回调（如定时器中断），使用**无捕获 lambda** 或普通函数。有捕获的 lambda 需要 `std::function`，这会引入额外的堆分配开销。

---

## 五、虚函数在嵌入式中的实际开销

### 解剖虚函数调用

```cpp
class Peripheral {
public:
    virtual void init() = 0;  // 纯虚函数
    virtual ~Peripheral() = default;

    int get_id() { return id_; }  // 非虚函数

private:
    int id_;
};

class SPI_Device : public Peripheral {
public:
    void init() override {
        // 初始化 SPI 外设
    }

private:
    uint32_t baudrate_;
};
```

**编译器生成的结构**：

```
SPI_Device 对象布局：
+--------+
| vptr   |  ← 指向 vtable 的指针（8 字节）
+--------+
| id_    |  ← Peripheral::id_（4 字节）
+--------+
| baudrate_| ← SPI_Device::baudrate_（4 字节）
+--------+

vtable（.rodata 段）：
+-------------------------+
| typeinfo ptr           |
+-------------------------+
| ~Peripheral()         |  ← 虚析构函数地址
+-------------------------+
| Peripheral::init()    |  ← SPI_Device::init() 的地址
+-------------------------+
```

### 性能开销实测

| 开销项 | 消耗 |
|--------|:----:|
| ROM（vtable） | 每个类约 16~24 字节 |
| RAM（vptr）   | 每个对象 8 字节 |
| 调用延时 | 比直接调用多 2~4 个周期（vptr → vtable → 跳转） |
| 内联 | 虚函数无法内联 |
| switch-case | 零间接开销 |

### 实战替代方案

```cpp
// ✅ 对已知设备集：enum + if/switch 替代虚函数
enum DeviceType { DEV_SPI_FLASH, DEV_SPI_LCD, DEV_SPI_SD };

void init_device(DeviceType type) {
    switch (type) {
    case DEV_SPI_FLASH: spi_flash_init(); break;
    case DEV_SPI_LCD:   spi_lcd_init();   break;
    case DEV_SPI_SD:    spi_sd_init();    break;
    }
}
// 编译器可能优化为跳转表，但 inline 更简单
```

### 什么时候可以使用虚函数

- 设备类型在**运行期才能确定**（如 USB 枚举）
- 设备数量少（< 10 个）
- ROM/RAM 充裕（> 64 KB Flash）
- 修改/扩展频率高，虚函数便于维护

**现实建议**：STM32F4/F7/H7、ESP32 等较丰富的 MCU 上，少量虚函数（3~5 个类）的开销完全可以接受。Cortex-M0/M0+ 或 32 KB Flash 以下——用 switch-case 或函数指针表。

---

## 六、其他实用特性速览

### `constexpr` 取代 `#define`

```cpp
// ❌ C 风格
#define BUFFER_SIZE  256
#define GPIO_HIGH(x) HAL_GPIO_WritePin(x, GPIO_PIN_SET)

// ✅ C++17 风格
constexpr size_t BUFFER_SIZE = 256;
constexpr void gpio_high(GPIO_TypeDef *port, uint16_t pin) {
    HAL_GPIO_WritePin(port, pin, GPIO_PIN_SET);
}
// 编译期检查类型，可调试，可放在命名空间
```

### `enum class` 取代普通枚举

```cpp
// ❌ 传统枚举：污染命名空间，可隐式转为 int
enum State { IDLE, BUSY, ERROR };

// ✅ C++11 强类型枚举：不泄露，不能隐式转 int
enum class State : uint8_t { IDLE, BUSY, ERROR };

State s = State::IDLE;  // ✅ 必须用作用域
// if (s == 0)          // ❌ 编译错误！不能与 int 比较
if (s == State::IDLE)   // ✅

// 显式转换（需要时）
uint8_t val = static_cast<uint8_t>(State::BUSY);
```

### `static_assert` 编译期断言

```cpp
static_assert(sizeof(uint32_t) == 4, "uint32_t must be 4 bytes");
static_assert(offsetof(GPIO_TypeDef, ODR) == 0x14,
              "GPIO ODR offset mismatch");
static_assert(CHAR_BIT == 8, "This code requires 8-bit bytes");
```

### `override` 和 `final`

```cpp
class UART_Driver {
public:
    virtual void send(const uint8_t *data, size_t len) = 0;
};

class USART1_Driver : public UART_Driver {
public:
    // ✅ override：编译器检查签名是否匹配
    void send(const uint8_t *data, size_t len) override {
        // ...
    }
    // ❌ 手误：void snd(...) — 不是 override，不会被调用！
};

class STM32_UART final : public UART_Driver {
    // final：禁止进一步继承
};
```

---

### 小结

现代 C++ 不是「什么都加一层抽象」的庞然大物。`nullptr`、`auto`、`std::array`、`enum class`、`constexpr`——这些特性要么零开销，要么帮助编译器生成更好的代码。关键是**按需选取**，不必全盘接受。对于嵌入式开发者，从 `nullptr` 和 `std::array` 开始引入，逐步尝试 lambda 和编译期多态，是一条平滑的升级路径。

---

> 🏷️ C++11 C++14 C++17 C++20 现代C++ 嵌入式开发 auto nullptr std::array lambda 虚函数 constexpr enum class
