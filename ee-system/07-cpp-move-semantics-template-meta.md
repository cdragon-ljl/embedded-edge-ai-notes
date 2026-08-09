# 嵌入式知识体系 · #07 · C++ 移动语义与模板元编程入门

---

C++11 引入的两个革命性特性——移动语义（move semantics）和模板元编程（template metaprogramming）——在嵌入式场景中有巨大的实用价值。移动语义消除不必要的拷贝开销，模板元编程将运行期计算转移至编译期。

---

## 一、左值与右值的本质区别

### 直观理解

```cpp
int a = 42;    // a 是左值：有地址，可取 &a
int b = a + 1; // a + 1 是右值：临时结果，无地址

// &(a + 1);   ← ❌ 编译错误！不能取右值的地址
```

**左值（lvalue）**：有名字、有地址、生命周期超出当前表达式。
**右值（rvalue）**：无名临时对象、无地址、表达式结束后立即销毁。

### C++11 的右值引用 `&&`

```cpp
void process(int &x)  { printf("lvalue ref\n"); }   // 左值引用
void process(int &&x) { printf("rvalue ref\n"); }   // 右值引用

int main() {
    int a = 5;
    process(a);        // 输出: lvalue ref（a 是左值）
    process(5);        // 输出: rvalue ref（5 是字面量，右值）
    process(std::move(a)); // 输出: rvalue ref（std::move 将 a 转为右值）
}
```

**`std::move` 做了啥？** 它不移动任何数据，只是**类型转换**——将左值转为右值引用，告诉编译器"请用移动构造函数而非拷贝构造函数"。

---

## 二、`std::move` 与移动构造函数的实际效果

### 先看拷贝的代价

```cpp
class RingBuffer {
public:
    RingBuffer(size_t size)
        : size_(size), data_(new uint8_t[size]) {}

    // 拷贝构造函数：深拷贝 — 昂贵
    RingBuffer(const RingBuffer &other)
        : size_(other.size_),
          data_(new uint8_t[other.size_]) {
        memcpy(data_, other.data_, size_);
        printf("COPY: %zu bytes\n", size_);
    }

    // 移动构造函数：浅拷贝 + 置空 — 便宜
    RingBuffer(RingBuffer &&other) noexcept
        : size_(other.size_),
          data_(other.data_) {
        other.data_ = nullptr;
        other.size_ = 0;
        printf("MOVE: stolen %zu bytes\n", size_);
    }

    ~RingBuffer() { delete[] data_; }

private:
    size_t size_;
    uint8_t *data_;
};
```

### 调用对比

```cpp
RingBuffer make_temp_buffer(void) {
    RingBuffer buf(1024);
    // 填充数据...
    return buf;  // C++11 前：拷贝构造（O(1024)）
                 // C++11 后：移动构造（O(1)）
}

void test(void) {
    RingBuffer a(1024);

    RingBuffer b = a;                      // 拷贝：深复制 1KB
    RingBuffer c = std::move(a);           // 移动：偷指针，O(1)
    RingBuffer d = make_temp_buffer();     // 移动（甚至 NRVO 优化直接构造）
}
```

### 嵌入式实战：大型缓冲区的高效传递

```cpp
using AudioBuffer = std::vector<int16_t>;  // 假设 16K 采样点 ≈ 32KB

// 采集线程产生数据（不拷贝！）
AudioBuffer capture_audio(void) {
    AudioBuffer buf(16000);
    // I2S DMA 填充...
    return buf;  // 移动语义 → 不拷贝 32KB，只传递 3 个指针
}

// 处理线程消费数据
void process_audio(AudioBuffer buf) {            // 传值，但移动构造
    apply_fft(buf);
}

// 主循环
void audio_loop(void) {
    AudioBuffer raw = capture_audio();  // 移动构造，无拷贝
    process_audio(std::move(raw));      // 移动，无拷贝
    // raw 现在是空，但安全
}
```

没有移动语义时，每经过一个函数边界就要拷贝 32KB。有了移动语义，全程只有**一次分配、零次拷贝**。

### `noexcept` 为什么重要

```cpp
// ❌ 没有 noexcept：vector 扩张时不敢用移动（怕抛异常后无法回滚）
// → 选择拷贝，慢！
RingBuffer(RingBuffer &&other) /* 没写 noexcept */ {
    // ...
}

// ✅ 有 noexcept：vector 扩张时放心移动
// → 快！
RingBuffer(RingBuffer &&other) noexcept {
    // ...
}
```

**规则：移动构造函数和移动赋值运算符永远加上 `noexcept`**。

---

## 三、模板特化：编译期类型分发

### 基础：函数模板

```cpp
template <typename T>
T max_value(T a, T b) {
    return (a > b) ? a : b;
}

// 展开（T = int）：
// int max_value(int a, int b) { return (a > b) ? a : b; }
```

### 模板特化：不同类型不同实现

```cpp
// 通用模板：所有整数类型的默认 CRC 计算
template <typename T>
T crc8(const T *data, size_t len) {
    uint8_t crc = 0;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++)
            crc = (crc & 0x80) ? (crc << 1) ^ 0x07 : (crc << 1);
    }
    return crc;
}

// 全特化：uint8_t 类型用查表法加速
template <>
uint8_t crc8<uint8_t>(const uint8_t *data, size_t len) {
    static const uint8_t table[256] = {
        // 预计算好的 CRC8 表
        0x00, 0x07, 0x0E, 0x09, 0x1C, 0x1B, 0x12, 0x15, // ...
    };
    uint8_t crc = 0;
    for (size_t i = 0; i < len; i++)
        crc = table[crc ^ data[i]];
    return crc;
}

// 使用
void test_crc(void) {
    uint8_t  buf8[]  = {0x01, 0x02, 0x03};
    uint16_t buf16[] = {0x0102, 0x0304};

    uint8_t c1 = crc8(buf8, 3);   // 调用查表版本（全特化）
    uint8_t c2 = crc8(buf16, 2);  // 调用通用版本
}
```

### 偏特化：按特征选择实现

```cpp
// 辅助 traits：判断是否为指针类型
template <typename T>
struct is_pointer {
    static constexpr bool value = false;
};

template <typename T>
struct is_pointer<T*> {
    static constexpr bool value = true;
};

// 利用偏特化写安全的寄存器写入
template <typename T>
void safe_write(volatile T *reg, T value) {
    *reg = value;
}

// 对 bool 类型特殊处理：不能写 0x55 到 1-bit 位域
template <>
void safe_write<bool>(volatile bool *reg, bool value) {
    *reg = value;
    __DSB();  // 对 bit-band 操作加内存屏障
}
```

---

## 四、`constexpr`：让编译器替你算

### 运行期 vs 编译期

```c
// ❌ C 方式：运行期算
int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}
// 运行时算 factorial(10)，浪费 CPU
int table[10] = {factorial(1), factorial(2), ...};
```

```cpp
// ✅ C++17 constexpr：编译期算
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

// 编译期就得到 3628800，运行时直接查表
constexpr int fact_10 = factorial(10);

// 甚至可以生成编译期查找表
template <size_t N>
struct FactorialTable {
    constexpr FactorialTable() : data{} {
        for (size_t i = 0; i < N; i++)
            data[i] = factorial(static_cast<int>(i));
    }
    int data[N];
};

constexpr auto fact_table = FactorialTable<10>();
// 编译期填充好，运行期零计算
```

### 实战：编译期 CRC 表

```cpp
// 编译期生成 CRC32 表
class Crc32Table {
public:
    constexpr Crc32Table() : table{} {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = i;
            for (int j = 0; j < 8; j++)
                crc = (crc >> 1) ^ (0xEDB88320 & ~((crc & 1) - 1));
            table[i] = crc;
        }
    }
    uint32_t table[256];
};

// 编译期计算完成，放在 .rodata 中，RAM 占用为 0
constexpr auto crc32_table = Crc32Table{};
```

### `constinit`（C++20）

`constexpr` 变量必须在编译期初始化，但有时你想**保证常量初始化**但又不想在编译期算完（比如函数指针表）：

```cpp
// C++20 constinit：保证静态初始化顺序安全
// ❌ 可能动态初始化（顺序不确定）
static int global = compute_value();

// ✅ 保证编译期常量初始化
constinit int global = compute_value();  // 编译期检查
```

---

## 五、综合：编译期多态 vs 运行期多态

嵌入式中的经典选择题：虚函数（运行期多态）的代价是 vtable + 间接跳转，能否用模板（编译期多态）替代？

```cpp
// 运行期多态（虚函数）
class Sensor {
public:
    virtual int16_t read() = 0;
};

class BMP280 : public Sensor {
    int16_t read() override { return i2c_read(0x76, 0xFA); }
};

class MPU6050 : public Sensor {
    int16_t read() override { return i2c_read(0x68, 0x3B); }
};

void process(Sensor &s) {
    int16_t val = s.read();  // 虚函数调用：vptr → vtable → 间接跳转
}
```

```cpp
// 编译期多态（CRTP + C++20 concept）
template <typename Derived>
concept SensorConcept = requires(Derived d) {
    { d.read() } -> std::same_as<int16_t>;
};

class BMP280 {
public:
    int16_t read() { return i2c_read(0x76, 0xFA); }
};

class MPU6050 {
public:
    int16_t read() { return i2c_read(0x68, 0x3B); }
};

template <SensorConcept T>
void process(T &sensor) {
    int16_t val = sensor.read();  // 编译期确定 → 直接调用
}
```

| 对比 | 虚函数 | 模板（编译期多态） |
|------|:------:|:------------------:|
| 调用开销 | vptr 间接寻址 | 直接调用 |
| ROM 开销 | vtable（每个类 1 个） | 无 |
| 灵活性 | 运行期动态绑定 | 编译期固定 |
| 适用场景 | 动态设备枚举 | 已知设备集合 |

**选型建议**：如果你在裸机或 RTOS 上，设备类型在编译期已知，模板多态是更好的选择——零间接开销，编译器可内联优化。

---

> 🏷️ C++ 移动语义 右值引用 模板元编程 constexpr CRTP 编译期计算 嵌入式 C++
