# AI 编译器导论：TVM、MLIR 与 Triton 如何自动生成 CUDA Kernel

> 系列：CUDA 高性能算子实战 · NPU-14
> 前置：NPU-13（算子融合与卷积 kernel）、NPU-11（ONNX/部署生态）
> 配套环境：Python 3.10+；Triton 示例需要 GPU（RTX 5060 Ti）与 `pip install triton torch`；TVM/MLIR 部分以理解为主

## 0. 本节目标

前几节我们亲手做了两件事：把卷积映射成 GEMM、把多个算子融合成一个 kernel。这些技能很值钱，但有个现实问题——**模型结构日新月异，硬件平台层出不穷，不可能每个算子都靠人手写**。

这正是 **AI 编译器** 的用武之地：它把"算子长什么样"和"硬件怎么跑最快"分开，自动完成**算子融合、内存规划、kernel 生成、自动调优**。本节回答三个问题：

1. AI 编译器和传统编译器是什么关系？（嵌入式工程师有天然优势理解它）
2. 它内部的三层结构是什么？（IR → 图优化 → 代码生成）
3. 三大主流方案 TVM / MLIR / Triton 各自怎么工作、怎么选？

读完你会明白：你手写的 tiled sgemm、融合 kernel，正是 AI 编译器自动生成的东西的"人工原型"——理解了它，算子开发岗位 JD 上的高频关键词（IR、图优化、代码生成、自动调优）就不再是黑话。

## 1. 为什么需要 AI 编译器

### 1.1 手工优化的不可持续

深度学习生态面临两个爆炸：

- **算子爆炸**：Transformer、Mamba、各种新注意力机制……每个月都有新算子；
- **硬件爆炸**：NVIDIA、AMD、ARM、各种 NPU，每个硬件的内存层级、指令集、Tensor 指令都不同。

如果每个算子在每个硬件上都要手写 kernel，工作量是"算子数 × 硬件数"——指数级的不可持续。**必须让计算机自动生成 kernel。**

### 1.2 嵌入式类比：从汇编到编译器

嵌入式工程师都经历过这个演进：早期单片机开发写汇编，每个硬件指令集不同，换个芯片全部重写；后来有了 C 编译器，**一份 C 代码编译到任何架构**——编译器负责指令选择、寄存器分配、优化。

AI 编译器就是深度学习领域的"通用编译器"：

```
传统编译器：   C 源码 ──▶ 前端/IR ──▶ 中端优化 ──▶ 后端/汇编 ──▶ 机器码
AI 编译器：   模型计算图 ──▶ IR ──▶ 图优化/融合 ──▶ 代码生成 ──▶ CUDA kernel
```

你手写的 CUDA kernel 相当于"手写汇编"——性能上限最高但成本高；AI 编译器相当于"高级语言 + 编译器"——自动生成接近手写的性能，成本低得多。

## 2. AI 编译器的三层结构

几乎所有 AI 编译器都遵循三层结构（和 GCC/LLVM 同构）：

### 第一层：IR（中间表示）

**定义 1（IR, Intermediate Representation）**：统一的、与框架无关的算子描述语言。ONNX 是一种图级 IR（描述"算子怎么连"）；TVM 的 Relay/TIR、MLIR 的各种 dialect、Triton 的 Python DSL 都是不同抽象级别的 IR。

IR 是编译器的"公共语言"：前端（框架导出）把模型翻译成 IR，后端（硬件代码生成）从 IR 生成 kernel。有了 IR，"算子数 + 硬件数"的爆炸被解耦成"前端数 + 后端数"。

### 第二层：图优化

**定义 2（图优化, Graph Optimization）**：在 IR 层面对计算图做等价变换，减少计算或访存。最重要的优化就是**算子融合**（算子融合篇手写过的 Conv+ReLU 融合，在这里由编译器自动完成）：

- 算子融合：`Conv + BN + ReLU` → 一个算子；
- 常量折叠：训练时不变的常量提前算好；
- 布局转换：NCHW ↔ NHWC 自动插入（框架实战篇提到两个框架的布局差异，编译器在这里统一处理）；
- 死代码消除：训练专用的算子（如反向传播相关）在推理图里直接删掉。

嵌入式类比：这相当于 GCC 的 `-O2` 优化 pass——开发者写清晰的源码，编译器自动做内联、常量折叠、循环优化。

### 第三层：代码生成

**定义 3（代码生成, Code Generation）**：把优化后的 IR 翻译成目标硬件的可执行 kernel，包括：

- **tiling/调度**：决定 tile 大小、线程块形状、共享内存用量（你手写的选择，编译器自动搜）；
- **指令选择**：选普通 FMA 还是 Tensor Core 指令（`mma`/`wmma`），是否向量化加载；
- **自动调优**：生成几十种配置，实际跑一遍选最优（和你在 Nsight 里手动调参做的事一样，编译器自动做）。

【图1：AI 编译器三层结构】

```text
┌─────────────────────────────────────────────────────────┐
│  前端：模型导入                                            │
│  PyTorch/TF 模型 ──▶ 图级 IR（ONNX / Relay / linalg）     │
├─────────────────────────────────────────────────────────┤
│  中端：图优化                                             │
│  算子融合 · 常量折叠 · 布局转换 · 死代码消除                │
├─────────────────────────────────────────────────────────┤
│  后端：代码生成                                           │
│  tiling/调度 → 指令选择（FMA/Tensor Core）→ 自动调优       │
│  ──▶ CUDA kernel / 其他硬件 kernel                        │
└─────────────────────────────────────────────────────────┘
```

> 图1 生图 prompt：三层堆叠架构图，白色背景。顶部绿色块"前端：模型导入（PyTorch/TF → 图级 IR）"，中间蓝色块"中端：图优化（算子融合/常量折叠/布局转换）"，底部橙色块"后端：代码生成（tiling/指令选择/自动调优 → CUDA kernel）"，三个箭头从上往下，右侧竖排中文标注"对应 GCC：前端/中端/后端"。扁平信息图，比例 16:9。

## 3. TVM：端到端的深度学习编译器

**定义 4（TVM）**：Apache 开源项目，最早由华盛顿大学陈天奇团队发起，是**端到端**的深度学习编译器：从框架模型一路编译到 CUDA、OpenCL、ARM、甚至 NPU 代码，是嵌入式 AI 部署的重要工具。

TVM 的核心组件：

| 组件 | 作用 | 对应三层 |
|:---|:---|:---|
| Relay IR | 图级 IR，描述算子连接 | 第一层 |
| TE / TIR | 张量表达式 / 底层 IR，描述算子的循环结构 | 第一层 |
| Pass 优化 | 算子融合、布局转换等 | 第二层 |
| AutoTVM / Ansor | 自动搜索最优调度（tile、向量化等） | 第三层 |
| Codegen | 生成 CUDA / OpenCL / 汇编 | 第三层 |

TVM 的经典工作流：`Relay 图 → 图优化 → 按算子调度 → 自动调优 → 代码生成`。它特别适合"要把模型部署到非 NVIDIA 硬件（ARM、NPU）"的场景——这和端侧部署的目标高度契合。

一段最小 TVM 示例（API 随版本演进，以官方文档为准）：

```python
import tvm
from tvm import te

n = 1024
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute((n,), lambda i: A[i] + B[i], name="C")   # 描述算子

s = te.create_schedule(C.op)
# 自动调度：向量化 + 并行化（相当于手写 kernel 的优化选择）
s[C].vectorize(C.op.axis[0])
```

**嵌入式工程师视角**：TVM 就像一个"深度学习版的 GCC 交叉编译工具链"——`aarch64-linux-gnu-gcc` 把 C 编译成 ARM 机器码，TVM 把模型编译成目标硬件 kernel，还能针对不同硬件自动调优（相当于 `-mcpu` 自动探测）。

## 4. MLIR：多级 IR 基础设施

**定义 5（MLIR, Multi-Level Intermediate Representation）**：LLVM 社区推出的编译器基础设施，核心思想是**多级 IR（dialect）**：用不同抽象级别的"方言"描述从高层算子到底层指令的整个编译过程，各级之间逐层 lower（下降）。

MLIR 的关键概念：

- **Dialect（方言）**：一组特定抽象级别的 IR 指令集。`tosa`/`linalg` 描述高层张量算子，`gpu` 描述 GPU kernel，`llvm` 对接 LLVM IR；
- **Lowering（下降）**：把高层方言逐步翻译成低层方言：`linalg → gpu → llvm`；
- **Pass 管道**：一系列优化 pass 串起来，编译器可以自由组合。

为什么 MLIR 重要？因为它把"编译器的组件"标准化了：算子融合、布局转换、循环优化都可以写成通用 pass，在不同硬件后端复用。**它是 LLVM 在 AI 领域的延伸**——如果你懂 LLVM 的结构（前端→IR→后端），MLIR 几乎不用重新学方法论。

MLIR 命令行示例（`mlir-opt` 执行 pass 管道，类似 `gcc -O2`）：

```bash
# 把 linalg 方言（高层张量算子）下降并优化成 gpu 方言
mlir-opt --convert-linalg-to-loops --convert-scf-to-cf \
         --convert-cf-to-llvm input.mlir
```

**嵌入式类比**：MLIR 的 dialect 就像"分层的中间语言"——从可读的 C 到汇编到机器码，每一层都有明确的语义，优化 pass 在最适合的层级做。你写汇编时做的寄存器分配，编译器在低层 dialect 上自动完成。

## 5. Triton：用 Python 写 GPU kernel

### 5.1 是什么

**定义 6（Triton）**：OpenAI 开源的 GPU 编程语言与编译器。它让你**用类似 Python 的 DSL 写 kernel**，编译器自动完成 tiling、共享内存分配、向量化、Tensor Core 映射。

嵌入式类比：手写 CUDA 像写汇编，Triton 像写 C——你描述"对一块数据做什么"，编译器负责"怎么分块、怎么搬、用什么指令"。**表达力大幅提升，性能接近手写**。

Triton 的核心抽象是 **program + block**：每个 program 处理一个数据块（tile），块内用 `tl.arange` 描述并行维度，编译器自动展开成线程。

### 5.2 可运行示例：向量加法

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                       # 当前 program 编号
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                  # 边界处理
    x = tl.load(x_ptr + offsets, mask=mask)      # 加载一块
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)  # 存回

n = 1024 * 1024
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
out = torch.empty_like(x)

grid = (triton.cdiv(n, 1024),)                   # 1024 个 program
add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)  # 启动 kernel

assert torch.allclose(out, x + y)
print("Triton 向量加法正确 ✔")
```

### 5.3 可运行示例：矩阵乘法（自动 tiling）

```python
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # 每个 program 负责一个 (BM, BN) 的输出 tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    a_ptrs = a_ptr + rm[:, None] * sa_m + rk[None, :] * sa_k
    b_ptrs = b_ptr + rk[:, None] * sb_k + rn[None, :] * sb_n

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):                    # K 维分块循环
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)                      # 自动映射到 Tensor Core
        a_ptrs += BK * sa_k
        b_ptrs += BK * sb_k

    c_ptrs = c_ptr + rm[:, None] * sc_m + rn[None, :] * sc_n
    tl.store(c_ptrs, acc)

# 启动（假设 M,N,K 能被 tile 整除）
M, N, K = 1024, 1024, 1024
a = torch.randn(M, K, device='cuda')
b = torch.randn(K, N, device='cuda')
c = torch.empty(M, N, device='cuda')
matmul_kernel[(M // 64, N // 64)](a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
    c.stride(0), c.stride(1), BM=64, BN=64, BK=32)

assert torch.allclose(c, a @ b, atol=1e-2)
print("Triton 矩阵乘法正确 ✔")
```

对比你在矩阵乘法篇手写的 tiled sgemm：**分块大小（BM/BN/BK）、K 循环、共享内存搬运**这些你手动管理的东西，Triton 编译器全部自动处理——你只描述"每个 tile 做什么"。这就是"编译器生成 kernel"的直观感受。

## 6. 三个方案怎么选

| | TVM | MLIR | Triton |
|:---|:---|:---|:---|
| 定位 | 端到端深度学习编译器 | 编译器基础设施（多级 IR） | GPU kernel 编程语言 |
| 抽象级别 | 图级 → kernel 级 | 多级 dialect，可自定义 | kernel 级（块级并行） |
| 硬件支持 | 广（CUDA/OpenCL/ARM/NPU） | 广（通过 LLVM 后端） | 主要 NVIDIA GPU |
| 学习成本 | 高（概念多） | 高（面向编译器开发者） | 低（像写 Python） |
| 适用场景 | 部署到多硬件、自动调优 | 构建新编译器/研究 | 快速写高性能 CUDA kernel |

**对算子开发者的建议**：想快速验证新算子性能，用 **Triton**（几天上手）；想把模型部署到端侧多硬件，研究 **TVM**；想深入编译器原理甚至做编译器开发，学 **MLIR**。三条路线都通往同一个终点：**理解算子如何被自动映射到硬件**。

## 7. 练习与里程碑

### 练习

1. **跑通 Triton**：安装 `triton` 后运行 5.2 和 5.3 的代码，把 BM/BN 改成 32 对比性能（`torch.cuda.Event` 计时），观察 tiling 大小对性能的影响。
2. **自动调优初体验**：在 Triton 里给 `matmul_kernel` 换不同的 BM/BN/BK 组合（64×64×32、128×128×16...），记录最快配置——体会"编译器替你做 tiling 搜索"。
3. **对照手写**：把 5.3 的 Triton matmul 与你手写的 tiled sgemm（矩阵乘法优化篇）性能对比，看编译器自动生成的差距有多大。
4. **理解融合**：在 Triton 里给 matmul 的 epilogue 加一行 `acc = tl.maximum(acc, 0.0)`（ReLU），体会"高层 DSL 里融合就是一行"。

### 里程碑自检

- [ ] 能画出 AI 编译器的三层结构并对应到 GCC 的三层
- [ ] 能说出图优化阶段最重要的优化是算子融合
- [ ] 能解释 MLIR 的 dialect 概念
- [ ] 能运行 Triton 的向量加法和矩阵乘法示例
- [ ] 能说清 TVM / MLIR / Triton 的定位区别

## 8. 小结

AI 编译器把"手写 kernel"升级为"自动生成 kernel"：

- **三层结构**：IR（公共语言）→ 图优化（算子融合/布局转换）→ 代码生成（tiling/指令选择/自动调优），与 GCC/LLVM 同构；
- **TVM**：端到端编译器，自动调优 + 多硬件部署；
- **MLIR**：多级 dialect 的编译器基础设施，面向编译器开发者；
- **Triton**：Python DSL 写 kernel，编译器自动 tiling/向量化/Tensor Core 映射，上手最快。

对你来说，前几节的"手写 GEMM、手写融合"是理解这些编译器的钥匙——**你知道编译器在自动做什么，才能真正用好它、并且知道它生成的东西为什么可能不如手写**。到这里，"编译器视野"阶段完成：从并行基础、算子优化、框架实战到方法论与编译器，你已经具备算子开发的全链路认知。这些沉淀最终汇聚成一个可展示的作品：一个纯 CUDA 手写的端侧推理算子库，以及一份能写进简历的性能报告。

> 🏷️ 标签：#AI编译器 #TVM #MLIR #Triton #代码生成 #自动调优
