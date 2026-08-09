# 嵌入式知识体系 · #20 · WiFi / BLE 共存：ESP32 无线协议的时分复用

ESP32 只有 **一根天线，一个射频前端**，但它同时支持 WiFi（2.4G 802.11 b/g/n）和蓝牙（BLE + Classic）。当两个协议都需要占用射频时，冲突不可避免。乐鑫在 ESP-IDF 中实现了一套**共存调度策略（Coexistence Arbitration）**，让两者有序地分时共享天线。

---

## 一、单天线分时复用的硬件约束

### 1.1 物理层冲突的本质

ESP32 内部的射频收发器同一时刻只能工作在一种模式：

```
时间轴：
├── BLE 广播/连接事件 ──┤
                         ├── WiFi 信标/数据收发 ──┤
                                                    ├── BLE 扫描 ──┤
```

WiFi 和 BLE 在 2.4GHz 频段工作在不同的频道：

| 协议 | 频段 | 频道宽度 | 典型频道 |
|------|------|---------|---------|
| WiFi | 2.400-2.4835 GHz | 20/40 MHz | Ch1(2412), Ch6(2437), Ch11(2462) |
| BLE | 2.400-2.4835 GHz | 2 MHz × 40 | Ch0(2404), Ch19(2440), Ch39(2480) |

两者频率范围完全重叠。当 WiFi 的 20MHz 带宽覆盖了 BLE 正在使用的频道时，两边的信号会互相干扰。

### 1.2 硬件层次的支持

ESP32 的共存机制不是纯软件策略，需要 MAC 层的硬件配合：

- **硬件仲裁器（Coexistence Arbiter）：** 内置在 ESP32 MAC 层，根据优先级裁决哪边获得天线
- **PTA（Packet Traffic Arbiter）：** 传统 WiFi+BT 外置共存握手的三线接口（GPIO），但 ESP32 内部集成了类似逻辑
- **信号指示：** WiFi MAC 和 BLE 控制器之间有专用硬件信号线（非 GPIO），延迟仅几个时钟周期

> 旧芯片（如 ESP32 第一代）使用内部 PTA 握手；ESP32-S3/C3 等后续芯片优化了仲裁逻辑，减少了 BLE 丢包率。

---

## 二、ESP-IDF 的共存调度策略

### 2.1 优先级体系

内部仲裁器将请求按优先级从高到低排列：

| 优先级 | 请求类型 | 说明 |
|:------:|---------|------|
| 1（最高） | BLE 高优先级事件 | 如连接请求、已建立连接的保活 |
| 2 | WiFi 管理帧 | Beacon 接收、ACK、RTS/CTS |
| 3 | WiFi 数据收发 | TCP/UDP 数据包 |
| 4 | BLE 扫描/广播 | 低优先级的广告事件 |
| 5（最低） | BLE 空闲扫描 | Background RSSI 扫描 |

**关键原则：** WiFi 的 Beacon（信标）和 BLE 的连接事件 **不可被延迟** ——错过信标会导致丢关联，错过 BLE 连接窗口会断连。仲裁器保证这两种事件不被对方阻塞。

### 2.2 共存状态机

ESP-IDF 的 coex 模块维护一个状态机，跟踪当前射频属于哪个协议：

```c
/* 共存状态机（简化） */
typedef enum {
    COEX_STATE_WIFI_ONLY,       // 仅 WiFi 活动
    COEX_STATE_BLE_ONLY,        // 仅 BLE 活动
    COEX_STATE_WIFI_ACTIVE_BLE_IDLE, // WiFi 正用，BLE 等待
    COEX_STATE_BLE_ACTIVE_WIFI_IDLE, // BLE 正用，WiFi 等待
    COEX_STATE_DUAL_IDLE,       // 都空闲（省电）
} coex_state_t;
```

状态切换由硬件仲裁器在微秒级完成，软件层面通过 `esp_coex_status_bit_t` 位掩码获知当前状态。

### 2.3 配置共存策略

ESP-IDF 提供了 `esp_coex_` API：

```c
#include "esp_coex_i.h"

void app_main(void) {
    // 初始化共存机制（会在 WiFi 和 BLE 初始化时自动调用）
    esp_coex_adapter_register(NULL);
    
    // 可选：调整共存偏好
    esp_coex_preference_set(ESP_COEX_PREFER_WIFI);  // WiFi 优先
    // 或
    esp_coex_preference_set(ESP_COEX_PREFER_BALANCE); // 均衡模式（默认）
}
```

三种预设偏好：

| 偏好 | 效果 | 适用场景 |
|------|------|---------|
| `ESP_COEX_PREFER_BALANCE` | 两边公平分配（默认） | 普通物联网网关 |
| `ESP_COEX_PREFER_WIFI` | WiFi 吞吐优先，BLE 可延迟 | 视频流传输+BLE配置 |
| `ESP_COEX_PREFER_BLE` | BLE 连接稳定性优先 | BLE Mesh + 少量WiFi |

---

## 三、连接间隔与 WiFi DTIM 的冲突处理

### 3.1 BLE 连接间隔 vs WiFi DTIM

这是共存中最经典的矛盾：

- **BLE 连接间隔（Connection Interval）：** 7.5ms ~ 4s 不等，每个间隔内主从双方交换一次数据
- **WiFi DTIM（Delivery Traffic Indication Message）：** 通常每 1-3 个 Beacon（100-300ms）一次，AP 宣告有缓存数据

当两者同时发生时，天线只有一根：

```
时间轴（冲突场景）：
BLE:  ┊← 连接事件 →┊      ┊← 连接事件 →┊
      ┊ ↑           ┊      ┊              ┊
WiFi: ┊←  DTIM Beacon →┊   ┊ 正常 Beacon →┊
              ↑ 冲突！
```

如果 BLE 连接事件与 DTIM Beacon 时间重合，仲裁器会优先处理 BLE（如果偏好是 BALANCE），WiFi 的 DTIM 被推迟，可能导致 AP 认为 STA 已休眠，触发丢包重传。

### 3.2 测量冲突的方法

在 ESP-IDF 中启用共存日志：

```c
// 在 menuconfig 中启用：
// Component config → Wi-Fi → WiFi BT/BLE coexistence → Coexistence log level → VERBOSE
```

或者通过运行时打印统计信息：

```c
// 获取共存统计
esp_coex_status_t status;
esp_coex_status_get(&status);
ESP_LOGI("coex", "WiFi lost %d  BLE lost %d",
         status.wifi_lost, status.ble_lost);
```

### 3.3 调整连接参数来避让

通过修改 BLE 连接参数，让 BLE 事件"避"开 WiFi Beacon 的时间窗口：

```c
// BLE 从机端：建议连接参数
esp_ble_conn_update_params_t conn_params = {
    .min_conn_interval = 0x0014,  // 20 × 1.25ms = 25ms
    .max_conn_interval = 0x0028,  // 40 × 1.25ms = 50ms
    .min_conn_latency  = 0,
    .max_conn_timeout  = 0x01F4,  // 500 × 10ms = 5000ms
};

// 发送连接参数更新请求
esp_ble_gap_update_conn_params(&conn_params);
```

**实用建议：**

| 场景 | BLE 连接间隔 | WiFi DTIM | 说明 |
|------|-------------|-----------|------|
| 高吞吐 | 25-50ms | 1（每100ms） | 频繁冲突，建议 WiFi 优先 |
| 低功耗 | 200-500ms | 3（每300ms） | 冲突概率低，推荐 |
| 实时控制 | 10-15ms | 3 | 冲突概率高，需 WiFi 牺牲吞吐 |

---

## 四、降低共存冲突的实用技巧

### 4.1 硬件层面

1. **天线分集（Antenna Diversity）：** 虽然 ESP32 只有一根射频链，但可以通过外部 Switch 连接两个天线，用物理位置分离减少干扰。ESP-IDF 的 `esp_coex_ant_set()` 支持配置天线切换逻辑。

2. **调整 WiFi 信道：** 避开 BLE 的工作频道。例如 BLE 使用 Ch19（2440MHz），WiFi 应选 Ch1（2412MHz）而不是 Ch6（2437MHz），减少频谱重叠：

    ```c
    // WiFi 信道和 BLE 频道的最小重叠
    // BLE Ch19(2440MHz) ↔ WiFi Ch6(2437MHz) → 高度重叠 ❌
    // BLE Ch19(2440MHz) ↔ WiFi Ch1(2412MHz) → 较少重叠 ✅
    wifi_config_t wifi_config = {
        .sta = {
            .channel = 1,   // 选择与 BLE 最小重叠的信道
        },
    };
    ```

3. **电源管理：** 开启 WiFi 的 **DTIM Sleep** 和 BLE 的 **Sniff Subrate**，让两者在对方工作时主动休眠，减少争抢：

    ```c
    // 启用 WiFi 省电模式（配合 DTIM）
    esp_wifi_set_ps(WIFI_PS_MIN_MODEM);  // 最小调制解调器休眠
    ```

### 4.2 软件层面

1. **错开活动窗口（最有效）：** 如果应用层可以控制，让 BLE 的连接事件和 WiFi 的 Beacon 间隔错开。例如将 BLE 连接间隔设为 45ms（36 个 1.25ms 时隙 = 45ms），WiFi Beacon 间隔 100ms，两者自然形成周期性偏移。

2. **调低 WiFi 发送功率：** 降低 WiFi 发射功率可以减少对 BLE 接收的阻塞：

    ```c
    // 降低 WiFi TX 功率（范围 0-80，对应 -8dBm ~ 20dBm）
    esp_wifi_set_max_tx_power(64);  // 约 16dBm（默认 20dBm）
    ```

3. **减少 BLE 广播包数量：** 广播阶段的 BLE 占用大量射频时间。尽快从广播态进入连接态，或增大广播间隔：

    ```c
    // 增大广播间隔（默认 100ms）
    esp_ble_adv_params_t adv_params = {
        .adv_int_min    = 0x400,   // 1024 × 0.625ms = 640ms
        .adv_int_max    = 0x800,   // 2048 × 0.625ms = 1280ms
        .adv_type       = ADV_TYPE_IND,
        // ...
    };
    ```

4. **必要时禁止共存：** 如果某个时刻确定只用一种协议，可以动态关闭另一方的射频活动：

    ```c
    // 只保留 WiFi，关闭 BLE
    esp_ble_gap_stop_advertising();
    esp_bt_controller_disable();
    
    // 需要 BLE 时再重新使能
    esp_bt_controller_enable(ESP_BT_MODE_BTDM);
    ```

### 4.3 共存性能预期

| 场景 | WiFi 吞吐量 | BLE 丢包率 |
|------|-----------|-----------|
| 仅 WiFi（无 BLE） | 20-30 Mbps | — |
| WiFi + BLE（均衡） | 15-25 Mbps | < 5% |
| WiFi + BLE（WiFi 优先） | 18-28 Mbps | 5-10% |
| WiFi + BLE（BLE 优先） | 10-18 Mbps | < 2% |

以上数据基于 ESP32 标准模组，室内 10m 距离，实际结果受环境干扰影响。

---

## 五、实战：调试共存问题的 Checklist

当产品同时使用 WiFi 和 BLE 时出现异常，按以下顺序排查：

1. **现象确认：** 是 WiFi 频繁断连、吞吐下降，还是 BLE 丢包率高、连接中断？
2. **看共存日志：** 启用 CONFIG_ESP_COEX_DEBUG_PRINTS，观察冲突次数
3. **测隔离场景：** 关闭 BLE 只测 WiFi 吞吐，再关闭 WiFi 只测 BLE 丢包——确认问题源于共存而非单协议问题
4. **调参数：** 调整 BLE 连接间隔（50-300ms），观察对 WiFi 吞吐的影响曲线
5. **改信道：** 切换 WiFi 信道（1/6/11），观察 BLE 扫描结果的变化
6. **降功率：** 降低 WiFi TX 功率，观察 BLE 接收率是否改善

---

> 一根天线撑起两个无线协议，本身就是工程权衡的艺术。了解 ESP32 的共存机制，不是为了消灭冲突——那是物理定律决定的——而是为了在你的应用场景下找到最合理的平衡点。

> 🏷️ ESP32 | WiFi BLE 共存 | 时分复用 | 共存仲裁 | 连接间隔 | DTIM | PTA | 射频冲突 | 天线分集 | 共存调试 | BLE 丢包 | WiFi 吞吐 | 嵌入式无线
