# INT8 量化与精度评估：把浮点模型压进 8 位

> 系列：RKNN 端侧部署实战：基于 RV1126 的模型转换与部署
> 前置：已完成浮点模型转换（config/load/build/export 全流程）
> 配套环境：PC（x86_64 Ubuntu + rknn-toolkit 1.7.x），无需板子

## 0. 本节目标

上节转换的还是浮点模型。RV1126 的 NPU 标称 **2 TOPS（INT8）**——只有量化成 8 位整数，才算真正"用上"这块 NPU。

本节解决三个问题：

1. **量化是什么**：为什么浮点转 8 位整数精度不会崩（用你熟悉的定点数直觉理解）；
2. **怎么量化**：dataset 校准集的准备、`build(do_quantization=True)` 完整流程；
3. **量化后精度掉了怎么办**：量化前后对比、`accuracy_analysis` 逐层定位精度损失。

## 1. 量化 = 定点数缩放（嵌入式类比）

### 1.1 你早就懂量化

做嵌入式的人对"浮点转定点"绝不陌生。MCU 上没有 FPU 时，我们用 **Q 格式**（Q15、Q1.14）表示小数：

```text
Q15 格式：-1.0 ~ 0.99997 的浮点数
         映射为 16 位整数 -32768 ~ 32767
         转换：q = round(x * 2^15)
```

**定义 1（量化, Quantization）**：把连续取值（浮点）映射到有限离散取值（整数）的过程。神经网络量化就是把权重和激活值从 FP32 变成 INT8——**本质上和你当年写 Q15 定点库是一回事**。

### 1.2 为什么神经网络对量化不敏感

直觉：神经网络在训练时就被设计成对微小扰动鲁棒（有大量冗余参数，中间值经过 ReLU 等非线性后范围收敛）。所以把 FP32 的权重和激活压到 8 位（256 个台阶），精度损失通常只有百分之几。

但"通常"不等于"一定"。量化损失取决于：

- **模型对精度有多敏感**（大模型更不敏感，小模型更容易崩）；
- **校准数据集是否覆盖真实分布**（下面重点讲）；
- **量化算法**（normal vs kl_divergence）。

### 1.3 量化的数学本质

对每个张量，量化器找一对参数：**缩放因子 scale**（浮点）和 **零点 zero_point**（整数）：

```text
量化：q = round(x / scale) + zero_point
反量化：x ≈ (q - zero_point) * scale
```

嵌入式类比：`scale` 就像 ADC 的 LSB 大小（每档代表多少伏），`zero_point` 就像 ADC 的偏移校准。**找 scale/zero_point 的过程就叫校准（calibration）**——它是量化精度好坏的核心。

## 2. dataset 校准集：量化的"基准数据"

### 2.1 为什么要校准

量化器需要知道**激活值的真实分布范围**才能定 scale。它怎么知道？在**代表性输入**上跑一遍模型，统计每层的激活分布。这批代表性输入就是**校准集（dataset）**。

嵌入式类比：ADC 量程设置——你得先知道信号大致范围，才能选对量程；选错量程，信号削顶或量化噪声巨大。校准集就是用来"测信号范围"的。

### 2.2 校准集要求

| 要求 | 说明 |
|:---|:---|
| **代表性** | 必须覆盖部署时的真实场景（做行人检测就放行人图，别放猫图） |
| **多样性** | 覆盖各类别、各种光照/角度 |
| **数量** | 经验值 100~1000 张；太少统计不准，太多收益递减 |
| **尺寸** | 与模型输入一致（224×224 的模型就放 224×224 的图） |

### 2.3 校准文件格式

rknn-toolkit 用**一个 txt 文件**列出校准图片路径，每行一张：

```text
# dataset.txt —— 每行一张校准图（绝对路径或相对路径）
/opt/datasets/calib_001.jpg
/opt/datasets/calib_002.jpg
...
/opt/datasets/calib_200.jpg
```

**注意**：校准图不需要标注（不需要类别标签）——它只是让模型跑一遍统计分布，不是训练。

## 3. 完整量化流程

### 3.1 量化转换

在转换脚本里把 `build` 加上 dataset：

```python
# quantize.py —— INT8 量化转换
from rknn.api import RKNN

rknn = RKNN(verbose=True)

rknn.config(
    mean_values=[[127.5, 127.5, 127.5]],
    std_values=[[127.5, 127.5, 127.5]],
    target_platform='rv1126',
    quantized_dtype='asymmetric_quantized-8',   # 8 位非对称量化
    quantized_algorithm='kl_divergence')        # 精度优先的校准算法

rknn.load_onnx(model='mobilenetv2-7.onnx')

# 关键：do_quantization=True + dataset 指向校准集 txt
ret = rknn.build(do_quantization=True, dataset='dataset.txt')
if ret != 0:
    print('build(量化) 失败'); exit(ret)

rknn.export_rknn('mobilenetv2_int8.rknn')
print('✅ 量化完成')
rknn.release()
```

对比上节的浮点转换，**只多了两处**：config 里两个量化参数 + build 的 `do_quantization=True, dataset=...`。

### 3.2 量化后模型大小

```bash
ls -lh mobilenetv2.rknn         # 浮点版：约 14 MB
ls -lh mobilenetv2_int8.rknn    # INT8 版：约 4 MB（约 1/4）
```

**约 1/4**——这就是 32 位→8 位的直接收益（权重 4 倍压缩），另外还有速度收益（INT8 乘加比 FP32 快得多，NPU 2 TOPS 就是这么来的）。

## 4. 量化前后精度对比

### 4.1 方法：同一测试集跑两个模型

准备一组有标签的测试图（比如从 ImageNet 验证集抽 100 张），分别用浮点模型和量化模型推理，对比 top1 准确率：

```python
# compare.py —— 量化前后 top1 对比（骨架）
import numpy as np
from PIL import Image
from rknn.api import RKNN

def load_model(rknn_path):
    rknn = RKNN()
    rknn.load_rknn(rknn_path)
    rknn.init_runtime()
    return rknn

def predict(rknn, img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img = np.array(img).astype(np.float32)
    out = rknn.inference(inputs=[img])
    return int(np.argmax(out[0]))

fp32 = load_model('mobilenetv2.rknn')        # 浮点版
int8 = load_model('mobilenetv2_int8.rknn')   # 量化版

# 假设 val_labels 是 {图片路径: 真实类别索引}
correct = {'fp32': 0, 'int8': 0}
for path, label in val_labels.items():
    if predict(fp32, path) == label: correct['fp32'] += 1
    if predict(int8, path) == label: correct['int8'] += 1

n = len(val_labels)
print(f'浮点 top1: {correct["fp32"]}/{n} = {correct["fp32"]/n:.2%}')
print(f'INT8 top1: {correct["int8"]}/{n} = {correct["int8"]/n:.2%}')
```

**期望结果**：MobileNetV2 这类模型量化后 top1 掉 0~2% 属于正常；掉 5% 以上就该警惕。

### 4.2 量化后精度崩了怎么办（排查顺序）

| 步骤 | 操作 | 说明 |
|:---|:---|:---|
| 1 | 检查校准集 | 是否覆盖真实场景？数量是否够？ |
| 2 | 换量化算法 | `normal` → `kl_divergence` |
| 3 | 检查 mean/std | 校准和推理的预处理必须一致 |
| 4 | 用 accuracy_analysis 定位 | 找出误差最大的层（下一节） |
| 5 | 换更宽容的量化配置 | 部分层保留 FP16/混合精度（一代支持有限，查文档） |
| 6 | 换模型 | 对量化不敏感的模型（大模型/结构规整的模型） |

## 5. accuracy_analysis：逐层定位精度损失

量化后精度崩，最怕"不知道崩在哪"。`accuracy_analysis` 帮你**逐层对比量化前后输出误差**，定位误差最大的层。

```python
# analyze.py —— 精度分析
from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.load_rknn('mobilenetv2_int8.rknn')
rknn.init_runtime()

# 输入：一批带路径的测试图（每行一个路径的 list）
# 输出：./analysis/ 目录下的逐层误差报告
ret = rknn.accuracy_analysis(
    inputs=['test_001.jpg', 'test_002.jpg', 'test_003.jpg'],
    output_dir='./analysis')
if ret != 0:
    print('accuracy_analysis 失败'); exit(ret)

rknn.release()
```

运行后在 `./analysis/` 下会生成报告文件，包含每一层的量化前后输出误差（如 cosine similarity / MSE 等指标）。**误差大的层就是量化损失的元凶**，针对它做网络结构修改（比如把该层换成对量化更友好的形式），比盲目调参高效得多。

【图1：量化精度分析闭环】

```text
        校准集 ──► 量化转换 ──► INT8 模型
                                    │
                                    ▼
                          测试集精度对比 ◄──┐
                                    │      │ 误差大
                              accuracy_analysis
                                    │      │
                              定位误差层 ◄──┘
                                    │
                              修改网络/调参
```

> 图1 生图 prompt：闭环流程图，白色背景。四个节点首尾相连成环：①"校准集 → 量化转换 → INT8 模型" ②"测试集精度对比" ③"accuracy_analysis 逐层分析" ④"定位误差层，修改网络/调参"。节点间用箭头连接，②到③标"误差大"，④回到①标"重新量化"。扁平插画风，比例 16:9，中文标注。

## 6. 练习与里程碑

### 练习

1. **完整量化**：准备 100 张校准图，用第 3 节脚本量化 MobileNetV2，确认模型从 ~14MB 变 ~4MB。
2. **精度对比**：用第 4 节脚本在 50 张带标签测试图上对比浮点/INT8 的 top1，记录差距。
3. **算法对比**：分别用 `normal` 和 `kl_divergence` 量化，对比精度，体会算法选择的影响。
4. **分析定位**：运行 `accuracy_analysis`，找出误差最大的 3 层，想想它们有什么共同点。

### 里程碑自检

- [ ] 能用自己的话解释"量化 = 定点数缩放"
- [ ] 会写 dataset.txt 校准文件
- [ ] 能把浮点模型转成 INT8 并看到体积 ~1/4
- [ ] 能对比量化前后精度并判断是否达标
- [ ] 会用 accuracy_analysis 定位精度损失层

## 7. 小结

- **量化原理** = Q 格式定点化，scale 像 ADC 的 LSB，zero_point 像偏移校准，神经网络对量化天然鲁棒；
- **校准集** = 量化器测激活分布用的代表性数据，必须覆盖真实场景、100~1000 张、与模型输入同尺寸；
- **量化流程** = `build(do_quantization=True, dataset=...)`，体积 ~1/4，精度通常掉 0~2%；
- **精度排查** = 检查校准集 → 换算法 → 核对预处理 → `accuracy_analysis` 逐层定位 → 改网络/换模型。

到这里，PC 端的工作全部完成：你手里有一个**量化好、精度验证过、能在 PC 模拟器上跑通**的 `.rknn` 模型。下一步把它搬到 RV1126 板子上，用真实的 NPU 跑起来——C 语言五步 API、Python 快速验证、目标检测后处理，板端篇见。

> 🏷️ 标签：#RKNN #INT8量化 #校准集 #accuracy_analysis #精度评估
