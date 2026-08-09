# 嵌入式知识体系 · #06 · C++ 智能指针：嵌入式中的 RAII 与所有权管理

---

**裸指针三宗罪**：忘记 `free` 导致内存泄漏、多次 `free` 导致 double-free、野指针访问导致 hardfault。智能指针（smart pointer）就是用 RAII 包裹这些风险，在离开作用域时自动释放资源。

本文针对嵌入式场景，聚焦 C++11 的三种智能指针，重点分析**零开销原则**和**实际开销**。

---

## 一、RAII 哲学：资源获取即初始化

RAII（Resource Acquisition Is Initialization）的核心思想很简单：**构造函数获取资源，析构函数释放资源**。

```cpp
class I2CTransaction {
public:
    I2CTransaction(uint8_t dev_addr) : addr_(dev_addr) {
        // 构造函数：获取总线锁
        while (i2c_busy)
            ;
        i2c_busy = true;
        i2c_start(addr_);
    }

    ~I2CTransaction() {
        // 析构函数：释放总线锁
        i2c_stop();
        i2c_busy = false;
    }

    void write_byte(uint8_t reg, uint8_t data) {
        i2c_write_byte(reg);
        i2c_write_byte(data);
    }

private:
    uint8_t addr_;
};

// 用法：离开 {} 自动释放总线
void read_sensor(void) {
    I2CTransaction trans(0x76);  // 自动 lock
    trans.write_byte(0x01, 0xAA);
    // 函数结束或抛出异常 → 自动 unlock → ✅ 不会死锁
}
```

RAII 使资源管理变得**声明式**——你只需告知"这个资源归我管"，编译器自动处理 cleanup。

---

## 二、`unique_ptr`：独占所有权，零开销

### 原理

`std::unique_ptr` 是独占所有权的智能指针。它不可复制（copy 被删除），只能移动（move）。当 `unique_ptr` 离开作用域时，自动 `delete` 管理对象。

```cpp
#include <memory>

void test_unique(void) {
    // 分配一个 1KB 的 DMA 缓冲区
    auto buf = std::make_unique<uint8_t[]>(1024);
    // buf 的类型是 std::unique_ptr<uint8_t[]>

    // ✅ 移动所有权
    auto buf2 = std::move(buf);
    // buf 现在是 nullptr，buf2 拥有缓冲区

    // ❌ 编译错误：不可复制
    // auto buf3 = buf2;

    // 函数结束 → buf2 析构 → delete[] 自动调用
}
```

### 嵌入式零开销证明

**裸指针版本：**
```cpp
void old_way(void) {
    uint8_t *buf = new uint8_t[1024];
    // ... 使用 buf
    delete[] buf;  // 手动释放
}
```

**`unique_ptr` 版本：**
```cpp
void new_way(void) {
    auto buf = std::make_unique<uint8_t[]>(1024);
    // ... 使用 buf
    // 离开作用域自动释放
}
```

编译器优化后，两个函数生成的**汇编完全相同**。`unique_ptr` 是真正的零开销抽象。

### 实战：外设对象生命周期管理

```cpp
class SPIDevice {
public:
    SPIDevice(SPI_TypeDef *spi, uint16_t cs_pin)
        : spi_(spi), cs_(cs_pin) {}

    void transfer(const uint8_t *tx, uint8_t *rx, size_t len);

private:
    SPI_TypeDef *spi_;
    uint16_t cs_;
};

class SensorManager {
public:
    SensorManager()
        : accel_(nullptr), gyro_(nullptr) {}

    // 使用 unique_ptr 明确所有权
    void set_accel(std::unique_ptr<SPIDevice> dev) {
        accel_ = std::move(dev);  // 转移所有权
    }

    void set_gyro(std::unique_ptr<SPIDevice> dev) {
        gyro_ = std::move(dev);
    }

private:
    std::unique_ptr<SPIDevice> accel_;  // 明确：我独占 accel
    std::unique_ptr<SPIDevice> gyro_;   // 明确：我独占 gyro
};
```

**关键优势**：代码表达意图。`unique_ptr` 告诉读者"这个资源只有一个所有者"，编译器强制检查——无法意外复制产生两个释放者。

---

## 三、`shared_ptr`：引用计数与控制块开销

### 原理

`shared_ptr` 通过引用计数实现共享所有权。复制时计数 +1，析构时 -1，归零时释放资源。

```cpp
void test_shared(void) {
    auto p1 = std::make_shared<int>(42);
    // p1 的控制块：ref_count = 1

    {
        auto p2 = p1;  // 复制 → ref_count = 2
        *p2 = 100;
    }  // p2 析构 → ref_count = 1

    // p1 仍有效
    std::cout << *p1;  // 输出 100
}  // p1 析构 → ref_count = 0 → delete
```

### 开销分析（对嵌入式非常重要）

| 开销项 | `unique_ptr` | `shared_ptr` |
|--------|:------------:|:------------:|
| 内存大小 | 8 B（同裸指针） | 16 B（指针 + 控制块指针） |
| 控制块分配 | 无 | 一次堆分配 |
| 复制/赋值 | 不可复制（move） | 原子操作 ref_count++ |
| 析构 | 一次 `delete` | 原子操作 ref_count-- + 条件 delete |
| 异常安全 | ✅ | ✅ |

### 什么时候不能用 `shared_ptr`

**在中断中不能使用 `shared_ptr` 的拷贝/析构！**

```cpp
// ❌ 危险！中断中修改 ref_count（原子操作可能死锁）
void IRQ_Handler(void) {
    auto temp = shared_buffer_;  // atomic increment → 不可重入
    process(temp);
}  // atomic decrement → 仍不安全
```

**在 ISR 中使用引用或裸指针传递已存在的对象是安全的**，但不要执行任何涉及 `shared_ptr` 复制或析构的代码。

### 实战场景：缓冲区共享

```cpp
// 共享音频数据缓冲区
class AudioPipeline {
public:
    using Buffer = std::shared_ptr<std::vector<int16_t>>;

    // 采集线程产生数据
    Buffer capture_block(void) {
        auto buf = std::make_shared<std::vector<int16_t>>(256);
        // I2S 采集填充...
        return buf;  // 移动构造，ref_count 仍是 1
    }

    // 处理线程共享访问
    void process_block(Buffer buf) {
        // 复制 shared_ptr → ref_count +1
        // 即使 capture 线程释放了原 buf，数据仍在
        apply_filter(*buf);
    }

private:
    // 在 RTOS 任务间共享
    Buffer latest_block_;
};
```

> 大部分嵌入式场景中，**`unique_ptr` 已经足够**。`shared_ptr` 只在明确的共享所有权场景使用，否则不仅浪费内存，还引入原子操作的开销。

---

## 四、`weak_ptr`：打破循环引用

### 问题：循环引用导致内存泄漏

```cpp
struct Node {
    std::shared_ptr<Node> next;
    ~Node() { printf("~Node()\n"); }
};

void test_leak(void) {
    auto a = std::make_shared<Node>();
    auto b = std::make_shared<Node>();
    a->next = b;
    b->next = a;  // 循环引用！

    // 离开作用域 → a 的 ref_count = 1（b 持有），
    //                b 的 ref_count = 1（a 持有）
    // → 永远不会析构 → 内存泄漏 💀
}
```

### 解决方案：`weak_ptr`

```cpp
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // 弱引用：不增加 ref_count

    ~Node() { printf("~Node()\n"); }
};

void test_no_leak(void) {
    auto a = std::make_shared<Node>();
    auto b = std::make_shared<Node>();
    a->next = b;           // b ref_count = 1
    b->prev = a;           // 不增加 ref_count，a ref_count 仍为 1

    // 离开作用域 → 都正常析构 ✅

    // 如果需要访问 prev：必须先 lock() 提升为 shared_ptr
    if (auto sp = b->prev.lock()) {
        // 安全使用 sp（指向 a）
        printf("prev is alive\n");
    } else {
        printf("prev has been destroyed\n");
    }
}
```

`weak_ptr` 是「观察者」——它能看到对象，但不延长生命周期。嵌入式中的典型场景是**驱动层与硬件抽象层之间的反向引用**。

---

## 五、嵌入式场景选型建议

### 黄金法则

| 场景 | 推荐 | 原因 |
|------|:----:|:----:|
| 外设驱动独占某个资源 | `unique_ptr` | 零开销，明确所有权 |
| 缓冲区在任务间传递 | `unique_ptr`（move） | 所有权随数据走 |
| 多个模块共享同一个配置 | `shared_ptr` | 共享所有权 |
| 父子层级中的反向引用 | `weak_ptr` | 避免循环引用 |
| 中断处理函数 | **裸指针** | 不可重入问题 |

### 使用 `make_unique` / `make_shared`

```cpp
// ❌ 不推荐
std::unique_ptr<SPIDevice> dev(new SPIDevice(SPI1, PIN_CS));

// ✅ 推荐
auto dev = std::make_unique<SPIDevice>(SPI1, PIN_CS);
```

原因：
1. **异常安全**：`make_*` 保证不会因参数求值顺序问题导致内存泄漏
2. **单次分配**（仅 `make_shared`）：`shared_ptr` 的对象和控制块一次性分配，减少堆碎片
3. **代码简洁**：类型只写一次

### 谨慎使用场景

- **超低 RAM 设备**（< 16 KB）：`unique_ptr` 的堆分配仍然有开销，考虑静态分配 + 裸指针
- **硬实时中断**：避免任何智能指针的复制/析构操作
- **bootloader**：不推荐，裸指针更直接

---

### 小结

`unique_ptr` 是嵌入式 C++ 的「默认选择」——零开销、所有权清晰、编译器强制检查。`shared_ptr` 仅在跨模块共享生命周期时使用，必须清楚其原子操作成本。`weak_ptr` 是解决循环引用的精确工具。RAII 哲学（而非具体指针类型）才是最重要的：**让析构函数替你做清理**。

---

> 🏷️ 智能指针 C++ RAII unique_ptr shared_ptr weak_ptr 嵌入式开发 内存管理
