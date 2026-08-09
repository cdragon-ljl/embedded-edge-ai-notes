# 板端 Python 推理：rknn-toolkit-lite 快速验证

> 系列：RKNN 端侧部署实战：基于 RV1126 的模型转换与部署
> 前置：已完成 INT8 量化模型 + 板端 C 五步流程理解
> 配套硬件：RV1126 开发板

## 0. 本节目标

上节用 C 实现了板端推理。但 C 版本适合**量产代码**，不适合**快速验证**——换个模型要重编译、调试输出要自己写。板端 Python 推理用 `rknn-toolkit-lite`，模型加载、推理、结果打印全在解释器里，是原型开发的利器。

本节完成：

1. 板端安装 `rknn-toolkit-lite`（与 PC 端 `rknn-toolkit` 区别要分清）；
2. 用 Python 三行核心 API 跑通分类；
3. 用真实摄像头/图片做快速模型验证（同一个 `.rknn`，C/Python 结果一致）。

## 1. rknn-toolkit-lite 是什么

**定义 1（rknn-toolkit-lite）**：运行在**板端**的 Python 推理库，基于 `librknnmrt.so` 封装。它**只做推理，不做转换**——模型必须在 PC 上先用完整版 rknn-toolkit 转好。

| 项目 | PC 端 rknn-toolkit | 板端 rknn-toolkit-lite |
|:---|:---|:---|
| 运行位置 | x86_64 PC | RV1126 板子 |
| 版本 | 1.7.x | 1.7.x（与 PC 端配套） |
| 类名 | `RKNN` | `RKNNLite` |
| 功能 | 转换 + 量化 + 模拟推理 | **仅推理** |
| 底层 | 模拟器 | `librknnmrt.so`（真实 NPU） |
| 依赖 | 一堆转换库（onnx/tf 等） | 轻量，适合板端 |

**装错就乱套**：PC 上不能装 lite（没有转换功能），板子上不建议装完整版（依赖太重）。记住：**转换在 PC，推理在板；完整版转换，lite 推理**。

【图1：PC 转换 → 板端 lite 推理的数据流】

```text
PC (rknn-toolkit)                      RV1126 板子 (rknn-toolkit-lite)
┌───────────────────┐                  ┌──────────────────────────┐
│ model.onnx        │                  │ RKNNLite()               │
│   │ convert       │  拷贝 .rknn      │   ├─ load_rknn()         │
│   ▼               │ ──────────────▶ │   ├─ init_runtime()       │
│ model.rknn        │                  │   ├─ inference(inputs)    │
└───────────────────┘                  │   └─ release()            │
                                       └──────────────────────────┘
```

> 图1 生图 prompt：左右两栏流程图，白色背景。左栏蓝色标题"PC：rknn-toolkit"：model.onnx 经过 convert 变成 model.rknn；中间箭头标注"拷贝 .rknn 到板子"；右栏绿色标题"RV1126：rknn-toolkit-lite"：列出 RKNNLite()、load_rknn()、init_runtime()、inference()、release() 五个步骤。扁平插画风，比例 16:9，中文标注。

## 2. 板端安装

### 2.1 安装

在**板子**上执行（RV1126 的 Linux SDK 通常自带 Python 3，若没有先 `opkg install python3 python3-pip`）：

```bash
# 获取 rknn_toolkit_lite wheel（SDK 的 external/rknn-toolkit/rknn-toolkit-lite/packages/ 下，
# 或瑞芯微官方 GitHub releases 下载对应 Python 版本）
pip3 install rknn_toolkit_lite-1.7.5-cp36-cp36m-linux_armv7l.whl

# 验证
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
```

> ⚠️ 注意 wheel 的平台标签是 `linux_armv7l`（32 位 ARM），不要拿 x86_64 的包装到板子上。

### 2.2 环境确认

`rknn-toolkit-lite` 底层调 `librknnmrt.so`，所以板子上该库必须可用（上一节已拷到 `/usr/lib/`）：

```bash
python3 -c "from rknnlite.api import RKNNLite; import ctypes; ctypes.cdll.LoadLibrary('librknnmrt.so'); print('librknnmrt 可加载')"
```

## 3. 第一个 lite 推理程序

创建 `lite_classify.py`：

```python
# lite_classify.py —— 板端 Python 分类推理
import numpy as np
from PIL import Image
from rknnlite.api import RKNNLite

# 1. 创建 RKNNLite 对象
rknn_lite = RKNNLite()

# 2. 加载模型（PC 转换好的 .rknn）
ret = rknn_lite.load_rknn('mobilenetv2_int8.rknn')
if ret != 0:
    print('load_rknn 失败'); exit(ret)

# 3. 初始化运行时（真实 NPU）
ret = rknn_lite.init_runtime()
if ret != 0:
    print('init_runtime 失败'); exit(ret)

# 4. 读图 + 预处理（只需 resize 到模型输入尺寸，保持 0~255）
img = Image.open('test.jpg').convert('RGB').resize((224, 224))
img = np.array(img).astype(np.uint8)   # 注意：UINT8，不是 float32！

# 5. 推理
outputs = rknn_lite.inference(inputs=[img])
scores = outputs[0].flatten()

# 6. 打印 top5
top5 = np.argsort(scores)[::-1][:5]
for i, idx in enumerate(top5):
    print(f'Top{i+1}: class {idx}, score {scores[idx]:.4f}')

rknn_lite.release()
```

运行：

```bash
python3 lite_classify.py
```

**结果应与 C 版本一致**（同一个模型、同一个 NPU）：

```text
Top1: class 281, score 0.9021
Top2: class 282, score 0.0411
...
```

## 4. 与 C 版本的关键差异

### 4.1 输入数据类型

**lite 的输入是 `np.uint8`（0~255），不是 float32**——和 C 版 `RKNN_TENSOR_UINT8` 对应。如果你按 PC 模拟器习惯传 float32，会报错或结果错乱。

### 4.2 归一化同样由库完成

和 C 版一样，`mean/std` 在转换时已编译进模型，lite 推理时自动应用。**代码里不要手动归一化**。

### 4.3 init_runtime 参数

`init_runtime()` 可带 `core_mask` 参数（如 `RKNNLite.NPU_CORE_AUTO` / `NPU_CORE_0`），用于多核 NPU 平台（RK3566/RK3588）指定核。RV1126 是单核 NPU，通常不传即可。

### 4.4 性能

Python 版本比 C 版本**每次调用多一层解释器开销**（约几毫秒级），对原型验证无所谓，量产建议用 C。但 Python 的开发迭代速度是 C 的很多倍——**先用 lite 验证模型效果，再移植成 C**，是推荐工作流。

## 5. 实战：批量验证多个模型

lite 最实用的场景：快速对比不同模型的精度/速度。比如验证量化前后的差异：

```python
# compare_lite.py —— 板端快速对比浮点版 vs INT8 版
from rknnlite.api import RKNNLite
import numpy as np
from PIL import Image

def test_model(path, img):
    rknn = RKNNLite()
    rknn.load_rknn(path)
    rknn.init_runtime()
    out = rknn.inference(inputs=[img])[0]
    rknn.release()
    return int(np.argmax(out))

img = np.array(Image.open('test.jpg').convert('RGB').resize((224, 224))).astype(np.uint8)
print('浮点版预测类别:', test_model('mobilenetv2.rknn', img))
print('INT8版预测类别:', test_model('mobilenetv2_int8.rknn', img))
```

同一个脚本换模型文件即可，不需要重编译——这就是 lite 的价值。

## 6. 常见问题

| 现象 | 原因 | 处理 |
|:---|:---|:---|
| `ImportError: No module named rknnlite` | lite 没装或装错平台 | 确认 wheel 是 armv7l 版本 |
| `init_runtime` 失败 | librknnmrt.so 缺失 / 驱动问题 | 按上节确认 /dev/rknpu |
| 输入 float32 报错 | lite 期望 uint8 | `astype(np.uint8)` |
| 结果与 PC 模拟不一致 | 预处理路径不同 | 确认 PC 模拟也用 0~255 + 同配置 |
| 运行慢 | Python 解释器开销 | 量产换 C 版 |

## 7. 练习与里程碑

### 练习

1. **跑通 lite**：在板子上跑通 `lite_classify.py`，与 C 版结果对比；
2. **批量验证**：用第 5 节脚本对比浮点/INT8 在 10 张图上的类别一致性；
3. **换模型**：把 PC 上转换好的另一个模型（如 ResNet）拷到板子，改一行路径跑通；
4. **计时对比**：用 `time.perf_counter()` 测量 lite 的 inference 耗时，与 C 版 rknn_run 对比，量化 Python 开销。

### 里程碑自检

- [ ] 分清 rknn-toolkit（PC 转换）与 rknn-toolkit-lite（板端推理）
- [ ] 能在板子上安装 lite 并 import 成功
- [ ] 能用 RKNNLite 三行核心 API 完成推理
- [ ] 知道 lite 输入是 uint8 0~255，归一化由库完成
- [ ] 能快速对比多个模型的推理结果

## 8. 小结

- **lite = 板端推理专用**：`RKNNLite` 类，只推理不转换，基于 librknnmrt；
- **三行核心**：`load_rknn → init_runtime → inference`；
- **输入纪律**：`np.uint8` 0~255，归一化自动；
- **工作流**：PC 转换 → 板端 lite 快速验证 → 量产移植 C。

到这里，板端推理的两种姿势（C 和 Python）你都掌握了，能跑**分类**了。但真实产品里更常见的是**目标检测**——不只是"这是什么"，而是"东西在哪"。下一节用 YOLO 补齐检测能力：模型转换 + 板端解码 + NMS 后处理，输出检测框。

> 🏷️ 标签：#RKNN #rknn-toolkit-lite #板端Python #RKNNLite #原型验证
