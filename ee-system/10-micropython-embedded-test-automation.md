# 嵌入式知识体系 · #10 · MicroPython 与嵌入式测试自动化

调试嵌入式软件往往让人想起一句玩笑：「改一行代码，烧录两分钟，等串口输出十秒钟，发现 bug 还在原地。」如果测试流程全靠手动操作——给 JTAG 接跳线、盯着逻辑分析仪看波形、用示波器量电平——那相当一部分时间其实耗在了「准备测试」上。

**MicroPython 给这条痛苦的生产线提供了一个简单粗暴的解决方案：把 MCU 变成一台可以实时执行 Python 脚本的测试仪器。**

## 一、MicroPython vs CircuitPython：选择与区别

这两个分支都源自 Damien George 2013 年发起的 MicroPython 项目。2016 年 Adafruit 为了降低创客入门门槛，fork 出 CircuitPython，两者自此分道扬镳。

| 维度 | MicroPython | CircuitPython |
|------|:-----------:|:-------------:|
| 内存占用 | 低，≈ 256KB Flash + 16KB RAM | 略高，推荐 512KB+ |
| USB 工作流 | REPL 通过串口/UART | 即插即用类 U 盘编辑 |
| 中断处理 | ✅ 有限支持 | ❌ 不暴露中断 |
| 外设库 | `machine` 模块 | `board` + `busio` 模块 |
| 并发 | `_thread`（有限） | 无 |
| 典型芯片 | STM32、ESP32、RP2040 | SAMD21、nRF52840、RP2040 |

**选型建议**：如果你的目标是**开发工具、测试夹具、原型验证**——MicroPython 更合适，因为它暴露了底层中断和硬件定时器，适合编写灵活的测试脚本。CircuitPython 更适合最终用户产品的教学场景，你不太可能用它来做批产测试。

## 二、用 Python 编写嵌入式自动化测试脚本

假设你在产线测试一块 STM32F103 控制板，需要验证：UART 数据环回、GPIO 电平翻转、ADC 采样精度、RTC 走时误差、Flash 读写完整性。

传统做法：写一个 C 测试固件 → 烧录 → 复位 → 用串口助手看 log → 人工判断。做五块板子就头大。

MicroPython 方式：把待测板刷上 MicroPython 固件（只需一次），然后用 Python 脚本自动测试：

```python
# test_fixture.py —— 产线测试骨架
import machine
import time
import ubinascii
from machine import Pin, UART, ADC, RTC

def test_uart_loopback(uart_id=1, tx_pin=6, rx_pin=7):
    """UART 环回测试：发一组数据，收回来对比"""
    uart = UART(uart_id, baudrate=115200, tx=Pin(tx_pin), rx=Pin(rx_pin))
    pattern = b"TEST_PATTERN_2026_ABCD"
    uart.write(pattern)
    time.sleep_ms(10)
    recv = uart.read(len(pattern))
    passed = recv == pattern
    print(f"[UART] {'PASS' if passed else 'FAIL'}: sent={pattern} recv={recv}")
    return passed

def test_gpio_toggle(pin_id=13, count=100):
    """GPIO 翻转测试：测量翻转频率"""
    p = Pin(pin_id, Pin.OUT)
    start = time.ticks_us()
    for i in range(count):
        p.value(i & 1)
    elapsed = time.ticks_diff(time.ticks_us(), start)
    freq = (count * 1_000_000) / elapsed
    print(f"[GPIO] freq={freq:.1f} Hz (expect ~{freq:.0f})")
    return 400_000 < freq < 1_600_000  # 正常范围

def test_adc_precision(adc_pin=36, vref=3.3):
    """ADC 精度：读取内部参考电压"""
    adc = ADC(Pin(adc_pin))
    adc.atten(ADC.ATTN_11DB)  # 0-3.3V 范围
    raw = adc.read()
    voltage = raw / 4095 * vref
    pct_error = abs(voltage - 1.1) / 1.1 * 100  # 假设 1.1V 参考点
    print(f"[ADC] raw={raw} voltage={voltage:.3f}V error={pct_error:.2f}%")
    return pct_error < 5.0  # ±5% 容差

# 主流程
results = []
results.append(test_uart_loopback())
results.append(test_gpio_toggle())
results.append(test_adc_precision())
print(f"\n{'='*30}\nTest Suite: {sum(results)}/{len(results)} PASSED")
```

这段脚本可以直接在 MicroPython 的 REPL 里运行，也可以上传到板子的 Flash 上自动执行。批量测试时，PC 端只需一根 USB 线 + 一个 `screen` 或 `picocom` 就能完成所有板子的自动化。

## 三、日志解析与数据分析实战

嵌入式测试的另一半工作是**数据收集与分析**。MicroPython 内建了 `json` 模块，可以输出结构化日志：

```python
import json

def log_event(event, value, unit=""):
    """输出 JSON 格式日志，方便 PC 端解析"""
    record = {
        "ts": time.time(),
        "event": event,
        "value": value,
        "unit": unit,
    }
    print(json.dumps(record))

# 采样 100 次 ADC 值
for i in range(100):
    v = adc.read() / 4095 * 3.3
    log_event("adc_sample", round(v, 3), "V")
    time.sleep_ms(50)
```

PC 端用 `pyserial` 接收并做数据分析：

```python
# pc_analyze.py —— PC 端实时采集与绘图
import serial
import json

ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
samples = []

while len(samples) < 100:
    line = ser.readline().decode().strip()
    if not line:
        continue
    try:
        record = json.loads(line)
        if record["event"] == "adc_sample":
            samples.append(record["value"])
    except (json.JSONDecodeError, KeyError):
        pass  # 跳过无法解析的行

if samples:
    print(f"Min: {min(samples):.3f}V, Max: {max(samples):.3f}V, "
          f"Mean: {sum(samples)/len(samples):.3f}V, "
          f"StdDev: {__import__('statistics').stdev(samples):.4f}V")
```

这套工作流让测试数据不再是杂乱的 ASCII 表格——你可以直接用 `matplotlib` 画曲线、用 `numpy` 做统计、用 `pandas` 存数据库。产线主管屏幕上显示的是图表，而不是一行行滚动的数字。

## 四、pyserial 与设备通信

`pyserial` 是 PC 端和设备通信的事实标准库。几个实用的生产力技巧：

```python
import serial
import serial.tools.list_ports

# 自动寻找 MicroPython 设备（不依赖固定 COM 口）
def find_micropython():
    for port in serial.tools.list_ports.comports():
        if "USB Serial" in port.description or "CP210" in port.description:
            return port.device
    return None

# 串口硬流控，防止大块数据丢失
ser = serial.Serial(
    port=find_micropython(),
    baudrate=115200,
    timeout=1,
    rtscts=True,        # 硬件流控
    write_timeout=1,
)

# 发送文件到 MicroPython（非 ampy，纯 pyserial）
def send_file(filename, content):
    ser.write(b"\x03\x03")  # Ctrl+C 两次，确保在 REPL
    time.sleep(0.5)
    ser.write(b"\x02")      # Ctrl+B 进入 raw REPL
    time.sleep(0.2)
    ser.write(content.encode())
    ser.write(b"\x04")      # Ctrl+D 执行
    time.sleep(0.5)
```

一个常见的生产落地场景：**24 小时老化测试**。PC 端用 pyserial 周期性地给待测板发指令、收数据、写入 CSV 文件、计算通过率。板子挂了几块、哪个测试项失败率最高——一目了然，不需要人工盯着。

## 小结

MicroPython 在嵌入式测试领域的价值，不在于它比 C 快，而在于它**让测试脚本和上位机分析脚本用同一种语言**。C 写固件、Python 写测试——这是目前嵌入式行业里性价比最高的测试方案之一。不需要买昂贵的测试设备，一根 USB 线 + 几行 Python + 一块刷了 MicroPython 的板子，就是一套能用的自动化测试夹具。

> 🏷️ MicroPython 嵌入式测试自动化 pyserial 产线测试 日志分析
