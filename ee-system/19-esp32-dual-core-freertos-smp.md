# 嵌入式知识体系 · #19 · ESP32 双核 FreeRTOS：一核不够，两核怎么用

ESP32 搭载两片 Xtensa LX6（或 LX7）核心，主频最高 240MHz。它的 FreeRTOS 不是标准的单核版本，而是经过了乐鑫深度改造的 **ESP-IDF FreeRTOS SMP（对称多处理）** 版本。这对从单核 MCU 迁移过来的开发者来说，既是礼物，也是陷阱。

---

## 一、双核调度机制：SMP 的 ESP32 实践

### 1.1 单核 vs SMP 调度器

标准 FreeRTOS 每次 Tick 中断只检查一个就绪队列，把最高优先级的任务放到唯一的核心上执行：

```
单核：Task_A(就绪) → Tick → Task_B(运行)  | 只有一个 CPU
```

ESP-IDF 的 SMP 版本维护一个全局就绪队列（`xReadyLists`）加上每个核独立的**当前运行任务**。Tick 中断触发调度时，调度器会尝试给**两个核同时分配任务**：

```
双核：CPU0: Task_A(运行)   CPU1: Task_B(运行)
      └─ Tick 中断 ─┘
      CPU0: Task_C(运行)   CPU1: Task_A(继续)
```

调度器核心逻辑（简化伪码）：

```c
/* ESP-IDF FreeRTOS SMP 调度核心 */
void vTaskSwitchContext(void) {
    UBaseType_t uxCurrentPriority;
    
    for (uxCurrentPriority = configMAX_PRIORITIES - 1;
         uxCurrentPriority >= 0;
         uxCurrentPriority--) {
        
        List_t *pxReadyList = &xReadyLists[uxCurrentPriority];
        
        if (listCURRENT_LIST_LENGTH(pxReadyList) == 0)
            continue;
        
        // 找一个未在当前核上运行的同优先级任务
        TCB_t *pxNewTCB = uxListRemoveUnpinned(pxReadyList, xPortGetCoreID());
        
        if (pxNewTCB != NULL) {
            vTaskTCBStore(pxNewTCB);
            return;
        }
    }
}
```

关键差异：调度器不会把同一个任务同时分配给两个核，但会尽量让两个核都"有事做"。

### 1.2 空闲任务的差异

单核 FreeRTOS 有一个空闲任务 `IDLE`。ESP-IDF SMP 版本在每个核上各创建一个空闲任务：

```c
/* ESP-IDF 初始化时创建两个空闲任务 */
void vTaskStartScheduler(void) {
    /* 为 CPU0 创建空闲任务 */
    xTaskCreatePinnedToCore(prvIdleTask, "IDLE0",
                            configMINIMAL_STACK_SIZE,
                            NULL, tskIDLE_PRIORITY, NULL, 0);
    
    /* 为 CPU1 创建空闲任务 */
    xTaskCreatePinnedToCore(prvIdleTask, "IDLE1",
                            configMINIMAL_STACK_SIZE,
                            NULL, tskIDLE_PRIORITY, NULL, 1);
}
```

当核心空转时，该核的空闲任务才会运行。这意味着 **IDLE0 和 IDLE1 是两个独立的任务实例**，可以在各自的空闲钩子中做不同的事。

---

## 二、任务绑核：xTaskCreatePinnedToCore

ESP-IDF 的核心 API：

```c
BaseType_t xTaskCreatePinnedToCore(
    TaskFunction_t pvTaskCode,   // 任务函数
    const char *const pcName,    // 任务名
    configSTACK_DEPTH_TYPE uxStackDepth,  // 栈深度（字）
    void *pvParameters,          // 参数
    UBaseType_t uxPriority,      // 优先级
    TaskHandle_t *pxCreatedTask, // 任务句柄
    BaseType_t xCoreID           // 核心ID：0/1 或 tskNO_AFFINITY
);
```

三个绑核策略：

| xCoreID 值 | 含义 | 使用场景 |
|-----------|------|---------|
| `0` | 固定在 PRO_CPU（协议核） | WiFi/BT 协议栈、蓝牙 |
| `1` | 固定在 APP_CPU（应用核） | 用户主逻辑、传感器处理 |
| `tskNO_AFFINITY` | 无亲和性，调度器自动分配 | 负载均衡的任务 |

### 常见分工模式

**模式一：协议栈隔离（推荐）**

```c
// CPU0：WiFi/BT 协议栈（ESP-IDF 默认）
// CPU1：用户任务
xTaskCreatePinnedToCore(user_app_task, "app", 4096, NULL, 5, NULL, 1);
xTaskCreatePinnedToCore(sensor_read_task, "sensor", 2048, NULL, 4, NULL, 1);
```

这种模式下，WiFi 中断和 TCP/IP 栈跑在 CPU0，应用逻辑不受无线协议的脉冲式负载影响。

**模式二：流水线型分核**

```c
// CPU0：数据采集
xTaskCreatePinnedToCore(adc_sampler, "sampler", 2048, NULL, 5, NULL, 0);
// CPU1：数据处理 + 输出
xTaskCreatePinnedToCore(data_processor, "processor", 4096, NULL, 5, NULL, 1);
```

两个核心间通过队列传递数据，关键数据字段按 4 字节对齐，避免 CPU0 写一半 CPU1 读到脏数据。

---

## 三、双核间通信：队列与共享内存

### 3.1 FreeRTOS 队列（跨核安全）

ESP-IDF 的 `xQueueSend()` 是跨核安全的——它内部用自旋锁保护了队列结构体，无论发送者和接收者在哪个核，都不会产生竞争：

```c
// CPU0 发送
xQueueSend(data_queue, &sensor_val, portMAX_DELAY);

// CPU1 接收
xQueueReceive(data_queue, &sensor_val, portMAX_DELAY);
```

**性能注意：** 跨核队列操作比同核队列慢约 30-50%（因为涉及自旋锁和内存屏障）。如果发送和接收同在一个核，可以考虑用 `xQueueSendFromISR` 配合同核调度。

### 3.2 共享内存的坑

两个核可以任意读写同一片 SRAM——没有 MMU 隔离。这带来了**数据一致性问题**：

```c
// 错误示例：CPU0 和 CPU1 同时操作
volatile uint32_t shared_counter = 0;  // volatile 只是防止优化

// CPU0
shared_counter++;  
// CPU1
shared_counter++;  
// 结果可能是 1 而不是 2（读-改-写非原子）
```

正确的做法是使用原子操作或互斥量：

```c
// 方式一：原子操作（推荐，无锁）
#include "esp_attr.h"

int32_t shared_counter = 0;

// CPU0
atomic_fetch_add(&shared_counter, 1);  // C11 原子操作，硬件保证

// CPU1
atomic_fetch_add(&shared_counter, 1);

// 方式二：互斥量（适用于复杂数据）
SemaphoreHandle_t xMutex = xSemaphoreCreateMutex();

xSemaphoreTake(xMutex, portMAX_DELAY);
shared_struct.field_a = new_value;
shared_struct.field_b = another;
xSemaphoreGive(xMutex);
```

### 3.3 双核环形缓冲区

高性能场景下，用无锁环形缓冲区（lock-free ring buffer）比队列更高效。ESP-IDF 不直接提供，但可以用 `atomic` 指令自行实现：

```c
#define RING_SIZE 256

typedef struct {
    int32_t data[RING_SIZE];
    volatile uint32_t head;      // 写指针（原子）
    volatile uint32_t tail;      // 读指针（原子）
} lockfree_ring_t;

bool ring_push(lockfree_ring_t *ring, int32_t val) {
    uint32_t h = atomic_load(&ring->head);
    uint32_t t = atomic_load(&ring->tail);
    
    if ((h - t) >= RING_SIZE) // 满
        return false;
    
    ring->data[h % RING_SIZE] = val;
    atomic_store(&ring->head, h + 1);  // 写完再更新 head
    return true;
}
```

> 注意：使用 `atomic_store`/`atomic_load` 时，编译器保证不会重排内存操作。

---

## 四、核心级临界区保护：Spinlock vs 关中断

### 4.1 单核临界区：关中断

标准 FreeRTOS 用 `taskENTER_CRITICAL()` 进入临界区——本质是关中断 + 保存全局中断掩码：

```c
/* 单核：关全局中断即可 */
void taskENTER_CRITICAL(void) {
    portDISABLE_INTERRUPTS();   // 关中断
    uxCriticalNesting++;        // 嵌套计数
}
```

但这在双核上不够用——CPU0 关了自身中断，CPU1 依然可以访问共享资源。

### 4.2 双核临界区：Spinlock

ESP-IDF 使用**自旋锁（Spinlock）**保护跨核共享资源：

```c
#include "esp_spinlock.h"

spinlock_t my_lock = SPINLOCK_INIT;

void critical_section_dual_core(void) {
    spinlock_acquire(&my_lock, portMAX_DELAY);
    // 访问共享资源（CPU0 和 CPU1 互斥）
    shared_buffer[index++] = new_data;
    spinlock_release(&my_lock);
}
```

自旋锁的原理：如果锁被 CPU1 持有，CPU0 会一直**空转等待**（spin）。这对短临界区（几微秒）高效，但长临界区极度浪费 CPU。

### 4.3 实际代码中的混合使用

ESP-IDF 内部的 FreeRTOS API 使用了**双层保护**：

```c
/* ESP-IDF 中的 xQueueSend 简化实现 */
BaseType_t xQueueGenericSend(QueueHandle_t xQueue, ...) {
    taskENTER_CRITICAL(&xQueue->mux);      // 自旋锁 + 关本地中断
    if (xQueue->uxMessagesWaiting < xQueue->uxLength) {
        xQueue->pcHead[xQueue->uxMessagesWaiting] = ...;
        xQueue->uxMessagesWaiting++;
    }
    taskEXIT_CRITICAL(&xQueue->mux);
}
```

`taskENTER_CRITICAL(&mux)` 在 SMP 版本中等价于：

1. **关本地核中断**（防止同核 ISR 打断）
2. **自旋等待**（如果另一个核持有锁）

退出时反向操作。

---

## 五、双核下的性能调优原则

1. **不绑核的陷阱：** `tskNO_AFFINITY` 任务可能会在两个核间"弹跳"——每次调度换一个核，导致缓存频繁失效。固定绑核通常更高效。

2. **中断亲和性：** 使用 `esp_intr_alloc()` 时指定 `ESP_INTR_FLAG_LEVEL1` 等标志，或用 `xCoreID` 参数把 WiFi 中断固定在 CPU0：

    ```c
    esp_intr_alloc(ETS_WIFI_MAC_INTR_SOURCE,
                   ESP_INTR_FLAG_LEVEL1,
                   wifi_isr_handler, NULL, NULL);
    // 默认绑在 PRO_CPU(0)
    ```

3. **栈分配：** 每个核独立维护当前任务栈。任务栈不在核之间迁移。`uxTaskGetStackHighWaterMark()` 返回的是该任务在绑定核上的实际栈使用量。

4. **Tick 中断：** 两个核都有自己的 Tick 中断（通过 `timer_group` 实现），每个核独立触发调度。如果一个核空闲，它自动进入 `WFI`（Wait For Interrupt）省电模式。

---

## 六、经典双核架构实战：WiFi 数据采集器

一个典型场景：ESP32 作为传感器节点，通过 WiFi 上报数据。

| 核心 | 任务 | 优先级 | 功能 |
|------|------|--------|------|
| CPU0 | WiFi/TCP 协议栈 | 3 | 维护 WiFi 连接、TCP 发送 |
| CPU1 | 传感器采样 | 5 | I2C 读取温湿度、ADC 采集 |
| CPU1 | 数据聚合 | 4 | 打包、校验、入队列 |
| CPU0 | 看门狗喂狗 | 1 | 系统健康检查 |

```c
void app_main(void) {
    // CPU0：WiFi 初始化（ESP-IDF 默认在 CPU0 运行协议栈）
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_sta();
    
    // CPU1：传感器任务
    xTaskCreatePinnedToCore(sensor_task, "sensor", 3072,
                            NULL, 5, &sensor_handle, 1);
    // CPU1：数据聚合任务
    xTaskCreatePinnedToCore(packer_task, "packer", 2048,
                            NULL, 4, &packer_handle, 1);
}
```

这种架构下，WiFi 发送数据时的脉冲负载（TCP 重传、ARP 广播）不会影响传感器 I2C 时序，采样精度得以保证。

---

> 双核不是魔法——它只是给了你两个独立的执行流。合理分配任务、正确处理跨核同步，才能真正发挥 ESP32 双核的威力。

> 🏷️ ESP32 | 双核 | FreeRTOS SMP | xTaskCreatePinnedToCore | 任务绑核 | 自旋锁 | 共享内存 | 原子操作 | 跨核通信 | 临界区 | PRO_CPU | APP_CPU | 嵌入式RTOS
