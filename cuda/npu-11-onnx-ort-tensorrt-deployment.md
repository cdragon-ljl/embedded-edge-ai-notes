# 模型部署生态：ONNX、ONNX Runtime 与 TensorRT 的完整链路

> 系列：CUDA 高性能算子实战 · NPU-11
> 前置：NPU-09（PyTorch 导出 ONNX）、NPU-10（TensorFlow 导出 SavedModel/TFLite）
> 配套环境：Python 3.10+、pip install onnx onnxruntime（CPU 推理即可；GPU 版另装 onnxruntime-gpu）

## 0. 本节目标

前两节我们分别用 PyTorch 和 TensorFlow 训练并导出了模型。现在的问题很实际：**一个训练好的模型，在真实生产环境里怎么跑起来？**

- 总不能每台部署机器都装 PyTorch/TensorFlow 全家桶；
- 生产环境需要**快**——最好能利用 GPU 的 Tensor Core、能自动做算子融合；
- 硬件五花八门——NVIDIA GPU、ARM、NPU、移动芯片，模型最好**一次导出、处处能跑**。

答案是部署生态的三件套：**ONNX（通用中间表示）、ONNX Runtime（通用推理引擎）、TensorRT（NVIDIA 高性能推理引擎）**。本节把它们串成一条完整的链路讲清楚，并回答贯穿全系列的问题：**这些推理引擎底层是怎么调用 CUDA 的？** 理解了这条链路，你就知道前面写的 CUDA kernel 在真实世界里站在什么位置。

## 1. 为什么需要中间表示（IR）

### 1.1 问题：框架与硬件的矩阵爆炸

假设你有 3 个训练框架（PyTorch / TensorFlow / PaddlePaddle）、4 种部署硬件（NVIDIA GPU / ARM CPU / 手机 NPU / 自研芯片）。如果每个框架要直接对接每种硬件，需要 3×4 = 12 套转换工具，且每加一个框架或硬件都要写一批新代码。

**解决办法**：引入一个与框架无关、与硬件无关的中间表示（IR, Intermediate Representation），让"框架 → IR → 硬件"变成两段解耦的流程：3 个框架只需各自导出一套 IR（3 个导出器），4 种硬件只需各自消费一套 IR（4 个后端）。总共 3 + 4 = 7 个组件。

### 1.2 嵌入式类比：这正是编译器的结构

```
源码 (.c)  ──编译器前端──▶  汇编/LLVM IR  ──编译器后端──▶  目标机器码
PyTorch   ──导出器──────▶  ONNX IR      ──推理引擎──────▶  GPU/CPU/NPU kernel
```

- **ONNX** ≈ LLVM IR / 目标文件（ELF）：统一的中间表示，描述"计算图长什么样"；
- **ONNX Runtime / TensorRT** ≈ 编译器后端 + 链接器：把 IR 翻译成目标硬件上的高效执行计划；
- **cuBLAS / cuDNN / 自研 CUDA kernel** ≈ 标准库/手写汇编：真正干活的高性能函数。

这就是为什么业界常说 **ONNX 是"模型界的 ELF"**——ELF 不是最终可执行文件，但它让"一份源码编译到任意平台"成为可能；ONNX 不是最终部署格式，但它让"一个模型导出到任意硬件"成为可能。

【图1：框架/硬件矩阵与 IR 解耦】

```text
         没有 IR：3 框架 × 4 硬件 = 12 条转换路径
  PyTorch ──┬─▶ NVIDIA ──▶ ARM ──▶ NPU ──▶ 自研
  TF      ──┼─▶ NVIDIA ──▶ ARM ──▶ NPU ──▶ 自研
  Paddle  ──┴─▶ NVIDIA ──▶ ARM ──▶ NPU ──▶ 自研

         引入 ONNX IR：3 条导出路径 + 4 条部署路径 = 7 个组件
  PyTorch ──┐
  TF      ──┼──▶  ONNX IR  ──▶ ONNX Runtime ──▶ NVIDIA/ARM/NPU
  Paddle  ──┘                └─▶ TensorRT   ──▶ NVIDIA GPU
```

## 2. ONNX：框架无关的计算图

### 2.1 ONNX 是什么

**定义 1（ONNX）**：Open Neural Network Exchange，一个开放格式，用**计算图**描述神经网络。计算图由两部分组成：

- **节点（Node）**：算子，如 `Conv`、`MatMul`、`Relu`、`Softmax`，每个算子有 `op_type`（类型）、输入张量名、输出张量名、属性（如卷积的 stride/padding）；
- **边（Edge）**：张量，即数据流。每个张量有名字、形状、数据类型。

嵌入式类比：ONNX 文件就像一张**电路原理图**——器件（算子）和连线（张量）清清楚楚，与"谁来焊板子"（哪个框架）、"板子跑多快"（哪个硬件）无关。

### 2.2 查看 ONNX 模型结构

```bash
pip install onnx onnxruntime
```

```python
import onnx

# 加载上一节导出的模型（如果没有，先按前面 PyTorch 篇的代码导出一个）
model = onnx.load('mnist_cnn.onnx')

print("算子集版本:", model.opset_import)
print("输入:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                for i in model.graph.input])
print("输出:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                for o in model.graph.output])
print("节点数:", len(model.graph.node))

# 打印前 8 个算子的连接关系
for node in model.graph.node[:8]:
    print(f"  {node.op_type:8s} ({', '.join(node.input)}) -> {', '.join(node.output)}")
```

输出示例：

```
算子集版本: [version: 17]
输入: [('input', [1, 1, 28, 28])]
输出: [('output', [1, 10])]
节点数: 14
  Conv     (input, /features/conv2d/Conv/ReadVariableOp:0, ...) -> /features/conv2d/Conv_output_0
  Relu     (/features/conv2d/Conv_output_0) -> /features/relu/Relu_output_0
  MaxPool  (...) -> ...
  ...
  MatMul   (...) -> ...
  Softmax  (...) -> ...
```

这就是"计算图"的字面含义：一系列算子按数据依赖串起来。你可以用 `netron.app`（网页版/桌面版可视化工具）把 ONNX 拖进去，看到图形化的网络结构。

## 3. ONNX Runtime：通用推理引擎

### 3.1 是什么

**定义 2（ONNX Runtime, ORT）**：微软开源的跨平台推理引擎，直接加载 ONNX 模型执行推理。它像一台"模型虚拟机"：同一份 ONNX，在 x86 CPU、ARM、NVIDIA GPU、甚至某些 NPU 上都能跑，通过**执行提供程序（Execution Provider, EP）**切换后端。

嵌入式类比：ONNX Runtime 像 JVM——同一份字节码（ONNX），在不同平台上由不同的 JIT/解释器（EP）执行。你不关心平台差异，只要选对 EP。

### 3.2 CPU 推理（最简可运行）

```python
import numpy as np
import onnxruntime as ort

# 加载 ONNX 模型
sess = ort.InferenceSession('mnist_cnn.onnx')

# 构造输入：一张 28x28 灰度图，预处理成 (1,1,28,28) float32
# 实际项目中这里要加载图片 -> 缩放 -> 归一化，与训练时保持一致
x = np.random.rand(1, 1, 28, 28).astype(np.float32)

input_name = sess.get_inputs()[0].name     # 'input'
output_name = sess.get_outputs()[0].name   # 'output'

result = sess.run([output_name], {input_name: x})
print("输出形状:", result[0].shape)          # (1, 10)
print("预测类别:", int(np.argmax(result[0])))
```

`InferenceSession` 创建时会对计算图做**图优化**（常量折叠、算子融合等），然后按拓扑顺序逐个执行算子。

### 3.3 GPU 推理：CUDA Execution Provider

```bash
pip install onnxruntime-gpu    # 带 CUDA 支持（需配套 CUDA/cuDNN 版本，以官方文档为准）
```

```python
import onnxruntime as ort

sess = ort.InferenceSession(
    'mnist_cnn.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    # 提供程序按优先级排列：GPU 优先，GPU 不可用时自动回退 CPU
)
print("当前可用的提供程序:", sess.get_providers())
# ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

**关键问题：ORT 的 GPU 后端是怎么干活的？**

`CUDAExecutionProvider` 收到一个 ONNX 算子（比如 `Conv`、`MatMul`）时，不是自己用 CUDA 写死每个算子，而是**调用 NVIDIA 官方高性能算子库**：

| ONNX 算子 | ORT 底层调用 | 说明 |
|:---|:---|:---|
| `MatMul` / `Gemm` | cuBLAS | 矩阵乘法库 |
| `Conv` | cuDNN | 卷积库 |
| `Softmax` / `Relu` / 逐元素 | 内部 CUDA kernel | 简单算子直接用模板 kernel |
| 自定义算子 | 用户注册的 CUDA kernel | 扩展点 |

嵌入式类比：cuBLAS/cuDNN 就像芯片厂商提供的**标准外设驱动库**——你不需要自己写 DMA 驱动，调用厂商库接口就行；而当厂商库不满足需求时（性能不够/算子不存在），你就得自己写 kernel（自己写驱动）并注册进去。**这正是本系列前七节手写 GEMM、优化 kernel 的终极意义**：cuBLAS 不是终点，理解它为什么快、在它不够用时能自己造，才是算子开发者的价值。

## 4. TensorRT：NVIDIA 的高性能推理引擎

### 4.1 与 ONNX Runtime 的区别

**定义 3（TensorRT）**：NVIDIA 推出的**专用于 NVIDIA GPU** 的推理优化引擎。它比 ONNX Runtime 更"重"、更专一，但也更快，因为它针对 GPU 做深度优化。

| | ONNX Runtime | TensorRT |
|:---|:---|:---|
| 定位 | 通用推理引擎，多硬件 | NVIDIA GPU 专用推理引擎 |
| 优化手段 | 通用图优化 | 图优化 + 层融合 + 精度校准 + kernel 自动选择 |
| 精度 | FP32/FP16（按 EP） | FP32/FP16/INT8（带校准） |
| 产物 | 加载 ONNX 直接跑 | 构建成 `.engine` 引擎文件（类似编译产物） |

TensorRT 的优化思路（嵌入式类比）：

1. **图优化与层融合**：把多个连续算子合并成一个 kernel，减少中间张量读写。例如 `Conv + BN + ReLU` 融合成一个算子——省掉中间结果写回显存的开销。类比：编译器 `-O2` 的指令合并，或你把"读传感器→滤波→限幅"写成一条内联函数而不是三个函数调用。
2. **精度选择**：支持 FP16 和 INT8。INT8 推理时用校准数据统计每层数值范围，把浮点权重/激活量化为 8 位整数。类比：把浮点系数转定点数，省一半以上带宽，代价是少量精度损失。
3. **Kernel 自动选择（Tactic）**：为每个算子从预置的几十种 kernel 实现中实测选优（对应不同 tile 大小、向量化宽度、Tensor Core 使用方式）。类比：编译器根据目标 CPU 特性自动选择指令集（NEON/AVX）。
4. **Tensor Core 利用**：FP16/INT8 计算自动映射到 Tensor Core，吞吐数倍于普通 CUDA Core。

### 4.2 命令行构建引擎（trtexec）

TensorRT 安装后（NVIDIA 官网下载，需注册账号），可用官方工具 `trtexec` 快速体验：

```bash
# 从 ONNX 构建 FP32 引擎
trtexec --onnx=mnist_cnn.onnx --saveEngine=mnist_cnn.engine

# FP16 引擎（Tensor Core 生效）
trtexec --onnx=mnist_cnn.onnx --fp16 --saveEngine=mnist_cnn_fp16.engine

# INT8 引擎（需要校准数据，--calib 指定校准输入）
trtexec --onnx=mnist_cnn.onnx --int8 --calib=calib_inputs.txt \
        --saveEngine=mnist_cnn_int8.engine
```

`trtexec` 构建时会打印每层选择的 tactic 和耗时，最后输出引擎文件。推理时用 TensorRT 的 C++/Python API 加载 `.engine` 执行。

> ⚠️ 说明：TensorRT 版本与 GPU 架构强相关，请从 NVIDIA 官网下载与驱动匹配的版本；构建引擎的时间与显存占用与模型大小有关。本节的目的是理解链路，不强制安装——用 onnxruntime CPU 跑通链路已经足够。

## 5. 完整链路：从训练到 GPU 算子

把前几节的碎片拼起来，一条完整的生产链路是：

```text
训练阶段                        部署阶段
─────────                       ─────────
PyTorch / TensorFlow        ONNX 文件（计算图，框架无关）
   │  torch.onnx.export          │
   │  / tf2onnx                  ├──▶ ONNX Runtime
   ▼                             │     (CUDA EP) ──▶ cuBLAS/cuDNN ──┐
ONNX 中间表示 ────────────────────┤                                 ├──▶ GPU
   ▲                             └──▶ TensorRT ──▶ 融合 kernel      ┘
   │  (Netron 可视化/验证)             + Tensor Core
训练好的权重
```

在这条链路的末尾，所有推理引擎最终都会落到**两类 CUDA 程序**上：

1. **厂商库 kernel**（cuBLAS/cuDNN）：覆盖绝大多数标准算子；
2. **自定义 kernel**：厂商库覆盖不到或性能不够的算子，由算子开发者手写。

本系列前七节练的手写 GEMM、tiling、归约、Nsight 分析，就是让你具备写第 2 类 kernel 的能力；而当你连厂商库的性能都开始较真（比如给 cuBLAS 提 issue、看懂 cuDNN 的启发式），你就站在了算子开发的第一线。

## 6. 练习与里程碑

### 练习

1. **图结构检查**：用 2.2 节的代码打印 `mnist_cnn.onnx` 的全部节点，数一数里面有几个 `Conv`、几个 `MatMul`、几个 `Relu`——和你在 PyTorch 里定义的网络结构对照验证。
2. **CPU vs GPU 推理**：安装 `onnxruntime` 和 `onnxruntime-gpu`，用 `timeit` 对比同一个 ONNX 模型在 CPU EP 和 CUDA EP 下的推理耗时（各跑 100 次取平均）。注意小模型（MNIST）GPU 可能不占优——启动/传输开销占比高，这也是端侧部署经常选 CPU/NPU 的原因。
3. **Netron 可视化**：用 `netron`（`pip install netron && netron mnist_cnn.onnx`）打开模型，找到 `Conv` 节点的属性（kernel_shape、strides、pads），与代码中的参数对应。
4. **融合收益估算**：思考为什么 `Conv + Relu` 融合能省一次全局内存写读——结合你在矩阵乘法篇学到的访存开销知识，估算一个 64×64 特征图融合后能省多少字节的读写。

### 里程碑自检

- [ ] 能解释 ONNX 为什么被称为"模型界的 ELF"
- [ ] 能说出 ONNX Runtime 的 Execution Provider 机制（GPU 优先、CPU 兜底）
- [ ] 能画出"框架 → ONNX → 推理引擎 → cuBLAS/cuDNN/自定义 kernel → GPU"的链路
- [ ] 能说出 TensorRT 的四种优化手段（融合/精度/kernel 选择/Tensor Core）
- [ ] 能解释自定义 CUDA kernel 在整条链路中的位置

## 7. 小结

本节打通了"模型 → 中间表示 → 推理引擎 → 算子"的完整部署链路：

- **ONNX** 是框架无关的计算图 IR，让"一次导出、处处部署"成为可能，等价于编译器里的 LLVM IR/ELF；
- **ONNX Runtime** 是通用推理引擎，用 Execution Provider 切换后端，GPU 后端调用 cuBLAS/cuDNN 执行算子；
- **TensorRT** 是 NVIDIA GPU 专用引擎，通过层融合、FP16/INT8、kernel 自动选择榨干硬件性能；
- 链路末端站着两类 CUDA 程序：厂商库 kernel 与自定义 kernel——**后者就是算子开发者的主战场**。

至此，"深度学习框架实战"阶段完成：你能亲手训练模型、导出通用格式、看懂部署链路。链路末端站着的两类 CUDA 程序——厂商库 kernel 与自定义 kernel——正是性能方法论要服务的对象：用 Roofline 模型量化"一个算子到底卡在算力还是带宽"，用算子融合思维重新设计 kernel，你会带着"真实模型长什么样"的认知去看性能优化，而不是对着空洞的 benchmark 数字。

> 🏷️ 标签：#ONNX #ONNXRuntime #TensorRT #模型部署 #推理引擎 #CUDA
