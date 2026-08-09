# 模型转换实战：参数、算子约束与报错规避

> 系列：RKNN 端侧部署实战：基于 RV1126 的模型转换与部署
> 前置：已完成 PC 端环境搭建与第一个转换（MobileNet → .rknn 全流程）
> 配套环境：PC（x86_64 Ubuntu + rknn-toolkit 1.7.x），无需板子

## 0. 本节目标

你已经能转换一个 MobileNet 了。但真实项目里的模型五花八门：有 ResNet、有 YOLO、有 Transformer 的轻量变体，转换时经常遇到"报错、转出来不对、性能莫名变慢"。

本节把转换这件事讲透：

1. **四类框架入口**怎么选、load 函数有哪些参数；
2. **config 常用参数**逐个拆解——每个参数影响什么、不设置会怎样；
3. **算子约束**——为什么有的模型转不过去、转过去跑在哪；
4. **高频报错与规避**——一张排查表 + 三个真实案例。

## 1. 支持框架与 load 函数

一代工具链（1.7.x）支持四种模型入口：

| 框架 | 加载函数 | 典型后缀 | 说明 |
|:---|:---|:---|:---|
| TensorFlow | `load_tensorflow` | `.pb` | 冻结的 TF 图 |
| TFLite | `load_tflite` | `.tflite` | 推荐入口，算子最规整 |
| Caffe | `load_caffe` | `.prototxt` + `.caffemodel` | 需要两个文件 |
| ONNX | `load_onnx` | `.onnx` | 1.7.x 起支持，生态最通用 |

**选型建议**：

- 模型从 PyTorch 训练来的 → 先导出 ONNX 再转（一代不支持 PyTorch 直转）；
- 模型从 TensorFlow 来的 → 优先转 TFLite（量化信息更完整）；
- 拿不准 → 用 ONNX 做统一入口，生态工具最多（Netron 可视化、onnx-simplifier 简化都方便）。

load 函数签名（以 ONNX 为例）：

```python
ret = rknn.load_onnx(
    model='mobilenetv2-7.onnx',   # 模型文件路径
    inputs=None)                  # 可选：输入节点名列表，默认自动识别
```

大部分情况 `inputs` 可以留空。只有当模型有多个输入、或输入名在转换后丢失时才需要显式指定（比如有些 TF 模型的输入是 `input:0` 这样的名字）。

## 2. config 常用参数详解

`rknn.config()` 是转换的"总开关"。下面是项目里真正会用到的高频参数：

| 参数 | 作用 | 默认 | 建议 |
|:---|:---|:---|:---|
| `mean_values` | 各通道均值 | 无 | 必须与训练预处理一致 |
| `std_values` | 各通道标准差 | 无 | 必须与训练预处理一致 |
| `target_platform` | 目标芯片 | 无 | **必须填** `'rv1126'` |
| `reorder_channel` | 通道顺序调整 | `'0 1 2'` | 输入是 BGR 时填 `'2 1 0'` |
| `quantized_dtype` | 量化数据类型 | 一代默认与平台相关 | 常用 `'asymmetric_quantized-8'` |
| `quantized_algorithm` | 量化算法 | `'normal'` | 精度敏感模型用 `'kl_divergence'` |
| `batch_size` | 转换时的 batch | 1 | 部署推理一般保持 1 |
| `optimization_level` | 优化强度 | 3 | 一般不动 |

### 2.1 mean_values / std_values

上节讲过：运行时预处理 = `(x - mean) / std`，必须与模型训练时一致。一代 API 的格式是**嵌套列表**：

```python
# 单输入模型：每个输入一个 [c1, c2, c3]
rknn.config(mean_values=[[127.5, 127.5, 127.5]],
            std_values=[[127.5, 127.5, 127.5]],
            target_platform='rv1126')

# 双输入模型（例如双目深度估计）：
rknn.config(mean_values=[[127.5,127.5,127.5],[127.5,127.5,127.5]],
            std_values=[[127.5,127.5,127.5],[127.5,127.5,127.5]],
            target_platform='rv1126')
```

### 2.2 target_platform

**必须填写，且必须是 `'rv1126'`**。它决定编译器生成哪一代 NPU 指令。填错（比如填 `'rk3568'`）会直接报错或生成跑不了的模型。

### 2.3 reorder_channel

一个特别容易踩的坑：**模型训练时输入是 RGB，但你的摄像头输出是 BGR**（V4L2 采集默认 NV12，转 RGB 后顺序也可能和你预处理代码不一致）。

`reorder_channel='2 1 0'` 表示把输入的通道顺序从 BGR 重排为 RGB（`'0 1 2'` 是不重排）。**用不用、怎么用，取决于你的输入数据通道顺序和模型要求的差异**，而不是随便填。

嵌入式类比：这就像 SPI 的 MSB/LSB 位序——发送方和接收方必须约定一致，否则数据全是乱的。

### 2.4 quantized_dtype 与 quantized_algorithm

量化相关的两个参数（量化的原理下一节专门讲）：

- `quantized_dtype`：量化成什么类型。一代常用 **`asymmetric_quantized-8`**（8 位非对称量化，带零点和缩放，对分布不均匀的激活值友好）；还有 `dynamic_fixed_point-8/16` 等选项，一般项目用默认即可；
- `quantized_algorithm`：怎么找量化参数。`normal` 快但粗略；`kl_divergence` 用 KL 散度最小化量化前后的分布差异，精度更好、耗时略长。**精度敏感的模型建议用 `kl_divergence`**。

## 3. 算子约束：模型不是"都能转"

### 3.1 两个基本事实

1. **NPU 不是万能图灵机**——它只实现了神经网络常用算子的硬件电路（卷积、池化、全连接、激活、拼接等）；
2. **转不过去的算子有两条出路**：要么编译器把它**放到 CPU 上跑**（混合调度），要么直接**报错拒绝转换**。

【图1：算子调度：NPU vs CPU 混合执行】

```text
                    原始模型计算图
      ┌─────────────────┼─────────────────┐
      ▼                 ▼                 ▼
   Conv2d           MaxPool          ？？自定义算子
      │                 │                 │
   NPU 执行          NPU 执行          CPU 兜底执行
   （快）            （快）            （慢，但能跑）
```

> 图1 生图 prompt：流程图，白色背景。顶部一个模型计算图节点，分三条支路：左边"Conv2d → NPU 执行（快）"绿色、中间"MaxPool → NPU 执行（快）"绿色、右边"自定义算子 → CPU 兜底执行（慢，但能跑）"橙色。底部注释"不支持的算子自动落到 CPU"。扁平插画风，比例 16:9，中文标注。

**性能含义**：混合调度能跑，但 CPU 上的算子会成为瓶颈。如果模型里 CPU 算子占比高，NPU 的优势就发挥不出来——这就是为什么"同样的模型，别人转换后 60fps，你转换后只有 10fps"的常见原因之一。

### 3.2 典型不支持/低效场景

| 场景 | 现象 | 处理思路 |
|:---|:---|:---|
| 自定义算子（自定义激活等） | build 报错或落 CPU | 改写为标准算子组合 |
| 动态 shape（输入尺寸运行时变化） | 转换失败 | 固定输入尺寸 |
| 超大模型（几十 MB 权重） | 内存不足 | 剪枝/蒸馏/换轻量模型 |
| 模型里的 RNN/LSTM | 部分支持或落 CPU | 换 CNN 方案或确认支持情况 |
| 训练时才有的节点（dropout 等） | 转换报错 | 导出前冻结/移除训练节点 |

**实操原则**：转换前用 Netron（`https://netron.app`）打开模型看一眼结构，遇到不认识的算子先查官方支持列表（`rknn-toolkit/docs/` 下有算子支持文档）。不确定就查文档，不要猜。

## 4. 完整转换示例：ONNX 模型 + 完整参数

把上节的 MobileNet 换成 ONNX 入口，同时把常用参数用齐：

```python
# convert_onnx.py —— 带完整参数的 ONNX → RKNN
from rknn.api import RKNN

rknn = RKNN(verbose=True)

# 1. 配置：全参数版本
ret = rknn.config(
    mean_values=[[127.5, 127.5, 127.5]],
    std_values=[[127.5, 127.5, 127.5]],
    target_platform='rv1126',
    reorder_channel='0 1 2',          # 输入已是 RGB 则不重排
    quantized_dtype='asymmetric_quantized-8',
    quantized_algorithm='kl_divergence',
    optimization_level=3,
    batch_size=1)
if ret != 0:
    print('config 失败'); exit(ret)

# 2. 加载 ONNX 模型（从 ONNX Model Zoo 下载 mobilenetv2-7.onnx）
ret = rknn.load_onnx(model='mobilenetv2-7.onnx')
if ret != 0:
    print('load_onnx 失败'); exit(ret)

# 3. 构建：先不量化
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('build 失败'); exit(ret)

# 4. 导出
ret = rknn.export_rknn('mobilenetv2.rknn')
if ret != 0:
    print('export 失败'); exit(ret)

print('✅ 转换完成')
rknn.release()
```

## 5. 高频报错排查表

| 报错/现象 | 原因 | 排查方向 |
|:---|:---|:---|
| `load_onnx 返回非 0` | ONNX 文件损坏 / 版本过旧 | 用 Netron 打开确认；`onnx.checker` 检查 |
| `build: unsupported op: XXX` | 算子不支持 | 查官方算子列表；换模型；改写算子 |
| 转换成功但推理结果全错 | mean/std 或通道顺序不匹配 | 核对 config 与训练预处理、reorder_channel |
| 推理速度异常慢 | 大量算子落 CPU | 看转换日志中 CPU 算子列表，改写网络 |
| `numpy/onnx 版本冲突` | 依赖版本过新 | 按环境篇锁定 numpy==1.16.6 onnx==1.7.0 |
| 转换时内存不足 | 模型太大 | 换轻量模型；batch_size=1 |

### 案例 1：`unsupported op`

现象：转换 YOLO 时 build 报一个没见过的算子。

处理流程：① Netron 定位该算子在哪一层；② 查官方支持列表；③ 若确实不支持，用 `onnx-simplifier`（`python3 -m onnxsim model.onnx model_sim.onnx`）简化图，很多"纸面算子"会被合并消除；④ 仍不行就改写网络结构，用标准算子实现同样功能。

### 案例 2：推理结果全错但转换成功

现象：转出来的模型在 PC 模拟上 top1 是垃圾。

处理流程：① 核对 mean/std 是否与模型训练一致（最容易错）；② 核对通道顺序（RGB/BGR）；③ 核对输入尺寸；④ 用一张"已知答案"的图做单步排查。

### 案例 3：转换日志大量 CPU 算子

现象：build 日志里出现 `fallback to cpu`。

处理流程：这些算子就是性能瓶颈。优先换网络结构（用 NPU 友好的算子），其次考虑模型简化，最后才接受现状并做 CPU 侧优化。

## 6. 练习与里程碑

### 练习

1. **参数实验**：分别用 `reorder_channel='0 1 2'` 和 `'2 1 0'` 转换同一个 RGB 输入模型，模拟推理同一张图，对比结果差异（预期：不匹配时结果错误）。
2. **算子实验**：找一个带 GELU/自定义激活的模型尝试转换，记录报错信息，然后用标准激活（ReLU/SiLU）替换后重新转换。
3. **Netron 练习**：用 Netron 打开 MobileNetV2 的 ONNX 文件，数一数里面有多少种算子，找出可能不受支持的。
4. **排查表实操**：故意把 mean/std 填错（如填 0/1），转换并模拟推理，验证"结果全错"的现象，然后修正。

### 里程碑自检

- [ ] 能说出四种框架入口及各自适用场景
- [ ] 能解释 config 里 mean/std、target_platform、reorder_channel 的作用
- [ ] 知道 NPU 不支持的算子会怎样（落 CPU 或报错）
- [ ] 遇到 unsupported op 知道完整排查流程
- [ ] 能用 Netron 查看模型结构

## 7. 小结

本节把"转换"这个环节补全了：

- **入口**：TF/TFLite/Caffe/ONNX 四选一，PyTorch 先导 ONNX；
- **config**：mean/std 必须匹配训练预处理，`target_platform='rv1126'` 必填，`reorder_channel` 管通道顺序，量化参数（`quantized_dtype` / `quantized_algorithm`）决定精度与性能的平衡；
- **算子约束**：NPU 只支持常用算子，不支持的落 CPU 或报错，转换前用 Netron 检查；
- **报错排查**：按表逐项核对，onnx-simplifier 是消除"纸面算子"的利器。

到目前为止，你转换的都是**浮点模型**——能跑，但没吃满 NPU 的 2 TOPS（那是 INT8 算力）。接下来进入整个部署流程技术含量最高的环节：量化。为什么浮点转 8 位整数精度不会崩、校准数据集怎么准备、量化后精度掉了怎么定位——下一阶段把这些一次讲清。

> 🏷️ 标签：#RKNN #模型转换 #算子约束 #ONNX #Netron #排错
