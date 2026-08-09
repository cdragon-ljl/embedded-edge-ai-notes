# PC 端环境搭建与第一个转换：把 MobileNet 变成 .rknn

> 系列：RKNN 端侧部署实战：基于 RV1126 的模型转换与部署
> 前置：RKNN-01（平台与工具链总览）
> 配套环境：x86_64 PC + Ubuntu 18.04/20.04（本文以 18.04 为例）、Python 3.6（版本说明见下）；本节的转换与模拟推理**不需要 RV1126 板子**

## 0. 本节目标

上一节认识了 RV1126 和 RKNN 生态。本节动手走通第一段流水线：**在 PC 上把一个小型分类模型（TFLite 格式）转换成 `.rknn` 文件，并用 PC 模拟器跑一次推理，看到分类结果**。

完成本节，你就掌握了 RKNN 工具链的四个核心步骤：`config → load → build → export`，外加模拟推理 `init_runtime → inference`。这是整个系列的地基——后面所有内容（量化、精度评估、板端部署）都建立在这条流水线上。

## 1. 环境准备：PC 端转换环境

### 1.1 你需要什么

| 项 | 要求 |
|:---|:---|
| PC | x86_64 架构（rknn-toolkit 1.7.x 官方 wheel 只提供 Linux x86_64） |
| 系统 | Ubuntu 18.04 或 20.04（官方推荐 18.04） |
| Python | 3.6（推荐；1.7.5 也提供 cp37 的 wheel，以官方 release 说明为准） |
| 网络 | 能访问 GitHub 下载 wheel 与模型 |

**⚠️ 版本红线**：这里安装的是 `rknn-toolkit`（一代，1.7.x），**不是 `rknn-toolkit2`**——装错包，后面所有 API 都会对不上。

【图1：PC 转换环境与板端运行环境的分工】

```text
┌──────────────────────── PC（x86_64, Ubuntu）────────────────────────┐
│  rknn-toolkit 1.7.x（Python 包）                                     │
│   ├─ 模型转换：config → load → build → export_rknn  →  model.rknn   │
│   ├─ 模拟推理：init_runtime(target=None) 在 PC 上模拟 NPU 执行       │
│   └─ 量化/精度评估（后续篇）                                          │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ 把 model.rknn 拷贝到板子
┌───────────────────────────▼──────────────────────────RV1126 板端─────┐
│  librknnmrt.so（C 运行时） / rknn-toolkit-lite（Python 运行时）       │
│  真实 NPU 推理                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

> 图1 生图 prompt：两张卡片对比图，白色背景。上卡片蓝色标题"PC 端（x86_64 Ubuntu）"内部列出：模型转换 config→load→build→export、模拟推理 init_runtime、量化与精度评估；下卡片绿色标题"RV1126 板端"内部列出：librknnmrt.so、rknn-toolkit-lite、真实 NPU 推理。两卡片之间一个向下箭头标注"拷贝 .rknn"。扁平插画风，比例 16:9，中文标注。

### 1.2 安装步骤

```bash
# 1. 系统依赖
sudo apt update
sudo apt install -y python3-pip python3-dev

# 2. 安装 rknn-toolkit 1.7.x 的依赖（注意锁定版本，新版 numpy/onnx 会冲突）
pip3 install numpy==1.16.6 onnx==1.7.0 protobuf==3.12.2

# 3. 下载 rknn-toolkit wheel（瑞芯微官方 GitHub releases）
#    网址：https://github.com/rockchip-linux/rknn-toolkit/releases
#    选择 1.7.x 版本，下载对应 Python 版本的 wheel，例如：
#    rknn-toolkit-1.7.5-cp36-cp36m-linux_x86_64.whl（Python 3.6）
pip3 install rknn-toolkit-1.7.5-cp36-cp36m-linux_x86_64.whl

# 4. 验证安装
python3 -c "from rknn.api import RKNN; print('RKNN 工具链 OK')"
```

> ⚠️ 如果 `import rknn` 报缺失依赖，根据报错逐个补齐（常见：`onnx`、`protobuf`、`numpy`）；如果提示 wheel 平台不匹配，检查是否下载了对应 Python 版本的 wheel。不同小版本（1.7.2 / 1.7.5）对 Python 的支持略有差异，以 release 页说明为准。

### 1.3 准备模型与测试图

用一个 TensorFlow 官方的 MobileNetV1 分类模型（TFLite 格式）：

```bash
# 下载 MobileNetV1 1.0 224（TensorFlow 官方 model zoo）
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
tar xzf mobilenet_v1_1.0_224.tgz
ls mobilenet_v1_1.0_224.tflite
```

再准备一张测试图片（任意类别明显的图即可，比如一张猫的照片），命名为 `test.jpg`，放在同目录。

## 2. 第一个转换脚本：完整可照抄

创建 `convert.py`：

```python
# convert.py —— MobileNetV1 (TFLite) → .rknn
from rknn.api import RKNN

# 1. 创建 RKNN 对象（verbose 打开日志）
rknn = RKNN(verbose=True)

# 2. 配置：目标平台 + 输入预处理参数
ret = rknn.config(
    mean_values=[[127.5, 127.5, 127.5]],   # 每个通道的均值
    std_values=[[127.5, 127.5, 127.5]],    # 每个通道的标准差
    target_platform='rv1126')              # ⚠️ 一代平台，不能写 rk3568 等
if ret != 0:
    print('config 失败，退出'); exit(ret)

# 3. 加载 TFLite 模型
ret = rknn.load_tflite(model='mobilenet_v1_1.0_224.tflite')
if ret != 0:
    print('load_tflite 失败，退出'); exit(ret)

# 4. 构建（第一步先不量化，跑通流程）
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('build 失败，退出'); exit(ret)

# 5. 导出 .rknn 文件
ret = rknn.export_rknn('mobilenet_v1.rknn')
if ret != 0:
    print('export 失败，退出'); exit(ret)

print('✅ 转换完成：mobilenet_v1.rknn')
rknn.release()
```

运行：

```bash
python3 convert.py
```

如果一切正常，目录下会生成 `mobilenet_v1.rknn`（约 4~5 MB，比原始模型略大——多了 NPU 的指令和内存布局信息）。**恭喜，你的第一个 .rknn 模型诞生了。**

## 3. config 参数在干什么：预处理必须与模型训练时一致

### 3.1 mean / std 的作用

`mean_values` 和 `std_values` 不是随便填的，它们定义了一个**固定的输入预处理**，会在推理时作用到你的输入数据上：

```text
预处理后的像素 = (原始像素 - mean) / std
```

MobileNetV1 官方训练时的预处理是：输入归一化到 [-1, 1]，即 `(x / 255 - 0.5) / 0.5`。用 mean=127.5、std=127.5 展开就是同一个公式：

```text
(原始 0~255 像素 - 127.5) / 127.5  =  原始像素 / 127.5 - 1  =  [0,1] 映射到 [-1,1]
```

**规则：config 的 mean/std 必须和模型训练时用的预处理一致**，否则模型推理结果会显著变差——就像 ADC 采样率设置错了，采集到的数据对不上算法假设。

### 3.2 数据流视角

【图2：输入预处理数据流】

```text
 原始图像（0~255 整数）
      │
      │  读取 + resize 到 224×224
      ▼
 图像数组（float32, 224×224×3）
      │
      │  NPU 运行时自动应用 config 的 mean/std：
      │  像素' = (像素 - 127.5) / 127.5        ← 这一步由 RKNN 自动完成
      ▼
 送入模型（范围 ≈ [-1, 1]）
      │
      │  模型推理
      ▼
 输出（1000 个类别的得分）
```

> 图2 生图 prompt：横向数据流图，白色背景。从左到右五个节点用箭头连接：① 原始图像（0~255 整数，小图图标）→ ② 读取+resize 224×224 → ③ 图像数组 float32 → ④ 自动应用 mean/std 预处理（高亮，公式 (x-127.5)/127.5）→ ⑤ 模型推理输出 1000 类得分。扁平插画风，比例 16:9，中文标注。

**关键点**：mean/std 是在 NPU 运行时**自动**应用的，你不需要在代码里手动归一化——但要确保传入的原始像素范围（0~255）与 config 匹配。如果手动归一化到 0~1 又配了 mean=127.5，结果会差 255 倍。

## 4. 模拟推理：PC 上先看到分类结果

转换成功后，用 PC 模拟器（不接板子）跑一次推理：

创建 `infer.py`：

```python
# infer.py —— 用 PC 模拟器跑 .rknn
import numpy as np
from PIL import Image
from rknn.api import RKNN

rknn = RKNN(verbose=True)

# 加载之前转换好的模型
ret = rknn.load_rknn('mobilenet_v1.rknn')
if ret != 0:
    print('load_rknn 失败'); exit(ret)

# 初始化运行时：target=None 表示 PC 模拟
ret = rknn.init_runtime()
if ret != 0:
    print('init_runtime 失败'); exit(ret)

# 读图并 resize 到 224×224
img = Image.open('test.jpg').convert('RGB').resize((224, 224))
img = np.array(img).astype(np.float32)   # 0~255，与 config 匹配

# 推理：TFLite 模型输入是 NHWC（1,224,224,3）
outputs = rknn.inference(inputs=[img])
print('输出 shape:', outputs[0].shape)

# 取 top5
scores = outputs[0].flatten()
top5 = np.argsort(scores)[::-1][:5]
for i, idx in enumerate(top5):
    print(f'  Top{i+1}: 类别 {idx}  得分 {scores[idx]:.4f}')

rknn.release()
```

运行：

```bash
pip3 install pillow
python3 infer.py
```

**预期输出**（猫图）：

```text
输出 shape: (1, 1000)
  Top1: 类别 281  得分 0.9132
  Top2: 类别 282  得分 0.0411
  ...
```

类别 281 在 ImageNet 标签里是 tabby cat（虎斑猫）——如果测试图是猫，这个结果就对了。（ImageNet 1000 类标签文件可下载对照：`https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt`，注意该文件含背景类共 1001 行，索引要减 1 核对。）

**模拟推理的意义**：`init_runtime()` 不带 target 参数时，rknn-toolkit 在 PC 上**模拟 NPU 执行**（用 CPU 模拟指令），让你在没有板子的情况下验证"模型转得对不对、结果准不准"。模拟的速度比真实 NPU 慢很多，但精度行为一致——所以 PC 模拟是量产前快速验证的利器。

## 5. 换成 ONNX 模型：只需改两行

一代工具链同样支持 ONNX 导入。把 `convert.py` 的两行替换即可：

```python
# 替换 load_tflite 这一行：
ret = rknn.load_onnx(model='mobilenetv2-7.onnx')   # 从 ONNX Model Zoo 下载
```

其余（config / build / export）完全不变。**转换入口换框架只是换一个 load 函数**，这是 RKNN 工具链设计得好的地方——后续用 Caffe/TF 也一样。

## 6. 常见报错与排查

| 报错 | 原因 | 处理 |
|:---|:---|:---|
| `No module named 'rknn'` | 没装好 / wheel 平台不对 | 确认 wheel 是 cp36/cp37 且 Linux x86_64；重装 |
| `ImportError: onnx ...` | onnx/protobuf 版本过新 | 按 1.2 节锁定 numpy/onnx/protobuf 版本 |
| `load_tflite` 失败 | 模型算子不在支持列表 | 换官方模型；或用 ONNX 格式转换 |
| `build` 报 `unsupported op` | 模型里有 NPU 不支持的算子 | 后面转换篇专门讲算子约束与规避 |
| 推理结果全是垃圾 | mean/std 与模型训练预处理不一致 | 核对 config 的 mean/std 与模型说明 |
| 模拟推理很慢 | 正常，PC 模拟比 NPU 慢 | 用小型模型（MobileNet 级别）即可 |

## 7. 练习与里程碑

### 练习

1. **跑通全流程**：按 1.2 装环境 → 下载 MobileNet → 运行 `convert.py` → 运行 `infer.py`，确认 top1 类别与测试图内容相符。
2. **换图测试**：换 3 张不同类别的图（猫、狗、车），观察 top5 结果是否合理。
3. **对比预处理**：把 config 的 mean/std 改成 `[0,0,0]`/`[1,1,1]` 重新转换推理，观察结果变化，体会"预处理必须匹配"这条规则。
4. **ONNX 复现**：从 ONNX Model Zoo 下载 mobilenetv2，按第 5 节改两行完成转换，对比两种入口的差异。

### 里程碑自检

- [ ] 能独立装好 rknn-toolkit 1.7.x 并验证 import
- [ ] 能把 TFLite 模型转成 .rknn 并导出
- [ ] 能用 PC 模拟器跑通一次推理并看到正确的 top1
- [ ] 能解释 mean/std 的作用和"必须与训练一致"的原因
- [ ] 知道 TFLite 与 ONNX 入口只需换 load 函数

## 8. 小结

本节跑通了 RKNN 转换流水线的第一段：

- **环境**：PC（x86_64 Ubuntu）+ rknn-toolkit **1.7.x**（一代，RV1126 专用）；
- **转换四步**：`config`（目标平台 + mean/std）→ `load_tflite/load_onnx` → `build` → `export_rknn`，产出 `.rknn` 文件；
- **模拟推理**：`init_runtime()` 无参即 PC 模拟，`inference` 出分类结果；
- **关键认知**：mean/std 预处理由运行时自动完成，必须与模型训练一致；换模型框架只换 load 函数。

现在你手里的 `.rknn` 是**未量化的浮点版本**——转换跑通了，但还没发挥 NPU 的真正实力（2 TOPS 是 INT8 算力）。下一步的核心工作就是量化：用校准数据集把模型变成 INT8，同时保证精度不塌。这是整个部署流程里技术含量最高的部分。

> 🏷️ 标签：#RKNN #模型转换 #TFLite #ONNX #模拟推理 #环境搭建
