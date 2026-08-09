# 算子融合与 Kernel 设计：从 GEMM 到卷积的访存优化实践

> 系列：CUDA 高性能算子实战 · NPU-13
> 前置：NPU-05（矩阵乘法 tiled 优化）、NPU-12（Roofline 性能模型）
> 配套环境：RTX 5060 Ti、CUDA Toolkit 13.x（代码用 nvcc 编译运行）

## 0. 本节目标

Roofline 模型告诉我们：**访存密集的算子，天花板是带宽，而减少访存最有效的手段是"算子融合"**。但前面几节我们只优化过 GEMM——深度学习里最常用的另一个算子**卷积（Conv）**还没有亲手写过。

本节完成两件事：

1. **手写卷积**：从朴素直接卷积出发，走通 im2col（把卷积变成矩阵乘法）和隐式 GEMM（不展开、用索引）两条主流路线，理解为什么卷积优化的终点是"复用 GEMM 的 tiling 功夫"；
2. **算子融合实战**：把 `Conv + ReLU`、`GEMM + ReLU` 融合进一个 kernel，用 Roofline 的思路量化融合省了多少访存，并给出可编译的融合 kernel。

这两项能力（把新算子映射成 GEMM、把多个算子合成一个 kernel）是算子开发者的日常，也是后面 AI 编译器自动做融合时要理解的手工原型。

## 1. 卷积：深度学习的第一访存大户

### 1.1 回顾：滑动窗口滤波

卷积在概念上就是"滑动窗口加权求和"（深度学习基础篇讲过）：一个 K×K 的核在输入特征图上滑动，每个位置做一次窗口内加权求和。设：

- 输入：`(C_in, H, W)`，输出：`(C_out, H_out, W_out)`
- 核：`(C_out, C_in, K, K)`，步长 stride，填充 pad

输出尺寸公式（与嵌入式图像处理一致）：

```
H_out = (H + 2*pad - K) / stride + 1
```

### 1.2 计算量与访存量

- 计算量：每个输出元素要算 `C_in × K × K` 次乘加 → 总 `2 × C_out × H_out × W_out × C_in × K × K` FLOP；
- 访存量（朴素实现）：输入每个窗口读一遍，相邻窗口**高度重叠**，数据重复读取严重。

关键观察：**卷积的数据复用性极强**——同一个输入像素会被附近多个输出窗口反复使用。朴素实现没有利用复用，每次重复读显存，这就是它慢的原因。优化的本质就是"把数据搬进片上存储（共享内存/寄存器）再复用"——和 GEMM 的 tiling 一模一样。

## 2. 三种卷积实现路线

### 2.1 路线一：朴素直接卷积

最简单的实现：每个输出元素一个线程，直接循环累加。

```cuda
// 朴素卷积：单样本、NCHW 布局
// in:   (C_in, H, W)
// w:    (C_out, C_in, K, K)
// b:    (C_out,) 或 NULL
// out:  (C_out, H_out, W_out)
__global__ void conv2d_naive(
    const float* in, const float* w, const float* b, float* out,
    int C_in, int H, int W, int C_out, int K, int pad, int stride)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;
    if (oh >= H_out || ow >= W_out) return;

    float acc = (b ? b[oc] : 0.0f);
    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int ih = oh * stride - pad + kh;
                int iw = ow * stride - pad + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    acc += in[ic * H * W + ih * W + iw]
                         * w[oc * C_in * K * K + ic * K * K + kh * K + kw];
                }
            }
        }
    }
    out[oc * H_out * W_out + oh * W_out + ow] = acc;
}
```

问题：每个线程从全局内存重复读窗口，相邻线程的窗口高度重叠，访存效率极低。**它是对的，但很慢**——正好作为性能对比的"下限"。

### 2.2 路线二：im2col——把卷积变成 GEMM

**定义 1（im2col, Image to Column）**：把输入按滑动窗口展开成一个"列矩阵"，使卷积变成一次矩阵乘法。

具体做法：

1. 对输入做展开：每个输出位置对应的 K×K×C_in 窗口，拉平成一列 → 得到列矩阵 `col`，形状 `(C_in×K×K, H_out×W_out)`；
2. 把权重重排成矩阵 `W_mat`，形状 `(C_out, C_in×K×K)`；
3. 卷积结果 = `W_mat × col`，形状 `(C_out, H_out×W_out)`——这就是一个标准 GEMM！

【图1：im2col 展开示意】

```text
输入 (C_in=1, H=5, W=5)              列矩阵 col (9, 9)
┌─┬─┬─┬─┬─┐      3×3 窗口滑动        ┌─────────┬─────────┬───┐
│1│2│3│4│5│      stride=1            │1 2 3    │2 3 4    │...│
├─┼─┼─┼─┼─┤                          │4 5 6    │5 6 7    │   │
│6│7│8│9│0│  ──────────────▶         │7 8 9    │8 9 0    │   │
├─┼─┼─┼─┼─┤   每个窗口拉成一列        │2 3 4    │3 4 5    │   │
│1│2│3│4│5│                          │5 6 7    │6 7 8    │   │
├─┼─┼─┼─┼─┤                          │8 9 0    │9 0 1    │   │
│6│7│8│9│0│                          │3 4 5    │4 5 6    │   │
└─┴─┴─┴─┴─┘                          │6 7 8    │7 8 9    │   │
                                     │9 0 1    │0 1 2    │   │
                                     └─────────┴─────────┴───┘
```

展开代码（host 端，演示思路；生产环境通常直接在 kernel 里用索引，不真正展开）：

```cpp
// im2col：把 (C_in,H,W) 输入展开成 (C_in*K*K, H_out*W_out) 列矩阵
void im2col(const float* in, float* col,
            int C_in, int H, int W, int K, int pad, int stride) {
    int H_out = (H + 2*pad - K) / stride + 1;
    int W_out = (W + 2*pad - K) / stride + 1;
    int rows = C_in * K * K;          // 每列长度
    for (int c = 0; c < rows; c++) {
        int ic  = c / (K * K);
        int kh  = (c / K) % K;
        int kw  = c % K;
        for (int p = 0; p < H_out * W_out; p++) {
            int oh = p / W_out, ow = p % W_out;
            int ih = oh * stride - pad + kh;
            int iw = ow * stride - pad + kw;
            col[c * (H_out * W_out) + p] =
                (ih >= 0 && ih < H && iw >= 0 && iw < W)
                ? in[ic * H * W + ih * W + iw] : 0.0f;   // 边界填 0
        }
    }
}
```

**收益**：展开后直接调用优化好的 GEMM（cuBLAS 或你在矩阵乘法篇写的 tiled sgemm），吃满计算。**代价**：内存膨胀——列矩阵是输入的 `K×K` 倍（3×3 卷积就是 9 倍）。对深层网络，展开矩阵可能比权重还大几十倍，这在大模型推理中不可接受。

### 2.3 路线三：隐式 GEMM——不展开的 GEMM

**定义 2（隐式 GEMM, Implicit GEMM）**：逻辑上按 im2col 的 GEMM 分块执行，但**不实际展开矩阵**——需要某个窗口数据时，直接按"输出位置 → 输入坐标"的映射从原图读取，把数据加载进共享内存/寄存器后按 GEMM 的 tiling 方式复用。

核心思想：GEMM 的 tile 循环里，`A` 的 tile 就是"一组滑动窗口的展开行"——这些行在原图里就是重叠的窗口区域，加载一次进共享内存，可以被多个输出 tile 复用。这正好解决朴素卷积的重复访存问题，又没有 im2col 的内存膨胀。

```
隐式 GEMM 视角：把 (C_out, C_in*K*K) × (C_in*K*K, H_out*W_out) 分块
  每次加载的 A tile  ← 按索引从原图取窗口数据（不展开）
  每次加载的 B tile  ← 同样按索引取
  共享内存里做 tiled 乘加，和 GEMM 完全一样
```

cuDNN 等库的卷积内核大多基于隐式 GEMM 思路（还有 Winograd 等其他路线），因为它兼顾了"GEMM 的计算效率"和"im2col 没有的内存开销"。

### 2.4 三条路线对比

| 路线 | 访存效率 | 内存开销 | 计算效率 | 实现复杂度 |
|:---|:---|:---|:---|:---|
| 朴素直接卷积 | 低（重复读窗口） | 低 | 低 | 低 |
| im2col + GEMM | 中（展开后 GEMM 好） | 高（K×K 倍） | 高（复用 GEMM） | 中 |
| 隐式 GEMM | 高（共享内存复用） | 低 | 高 | 高 |

**工程结论**：小模型/原型用 im2col 快速验证；生产级推理用隐式 GEMM；朴素实现只作为正确性基准。

## 3. 算子融合：减少中间张量

### 3.1 为什么融合

Roofline 告诉我们：逐元素算子（ReLU、偏置加、BN 缩放）都是**访存密集**——AI 只有 0.08 左右。如果模型是 `Conv → Bias → BN → ReLU` 四个 kernel 顺序执行：

- 每个 kernel 都要把结果写回全局内存，下一个 kernel 再读出来；
- 中间张量 `Bias` 和 `BN` 的结果被写了 3 次、读了 2 次——**纯粹浪费带宽**。

嵌入式类比：这就像你把一次数据处理拆成四个函数，每个函数都把结果存回 RAM，下一个函数再从 RAM 读——而不是在寄存器/局部变量里直接传递。显然应该合并。

### 3.2 融合的收益量化

对 `C_out = 32, H_out = W_out = 28` 的特征图（float32）：

- 一个中间张量大小：`32 × 28 × 28 × 4 B ≈ 100 KB`；
- 不融合：Bias 结果写 100KB、BN 读 100KB 写 100KB、ReLU 读 100KB 写 100KB → 额外搬运约 **500 KB**；
- 融合成一个 kernel：Bias+BN+ReLU 全部在**寄存器/共享内存里完成**，额外搬运 **0 字节**。

对一个 10 层的 CNN，这种融合能省下几 MB 的显存搬运——在 448 GB/s 带宽下，就是几毫秒级的延迟收益。所以**推理引擎（ONNX Runtime / TensorRT）的图优化阶段第一件事就是做算子融合**——你在部署生态篇看到的"层融合"，底层就是这里的手工 kernel 技巧。

### 3.3 融合示例 1：GEMM + ReLU（epilogue 融合）

在矩阵乘法篇的 tiled sgemm 里，每个线程算完 `acc` 后直接写回。融合 ReLU 只需要在**写回之前**加一行：

```cuda
// 在 tiled sgemm 的 epilogue 里融合 ReLU（只改写回部分）
// 假设每个线程已算完 acc（寄存器中的累加结果）
acc = fmaxf(acc, 0.0f);            // ReLU：max(0, x)
// 再执行原来的写回
// C[row + i * ldc] = acc;
```

就这一行，省掉了"ReLU kernel 读一次 + 写一次"的整个中间张量往返。这就是**融合的本质：把相邻算子的计算折叠进前一个 kernel 的 epilogue（或后一个的 prologue）**。

### 3.4 融合示例 2：Conv + ReLU 完整 kernel

在 2.1 节朴素卷积的基础上融合 ReLU——把写回从 `out[...] = acc` 改成 `out[...] = fmaxf(acc, 0.0f)`：

```cuda
// 融合 Conv + ReLU：只改最后一行，省掉一个 ReLU kernel 的整张特征图往返
__global__ void conv2d_relu_fused(
    const float* in, const float* w, const float* b, float* out,
    int C_in, int H, int W, int C_out, int K, int pad, int stride)
{
    // ... 与 conv2d_naive 完全相同的窗口累加 ...
    float acc = (b ? b[oc] : 0.0f);
    // ... 三重循环累加 acc ...

    out[oc * H_out * W_out + oh * W_out + ow] = fmaxf(acc, 0.0f);  // ← ReLU 融合
}
```

编译与运行：

```bash
nvcc -O3 -arch=sm_120 fused_conv.cu -o fused_conv
./fused_conv
```

## 4. 融合的工程边界

融合不是越多越好，有三个约束：

1. **数据依赖**：融合的算子必须是**逐元素或可原地计算**的（ReLU、Bias、BN 缩放、Scale），或者输出形状一致。涉及重排（如 Transpose）的算子融合收益会打折；
2. **寄存器压力**：融合越多，单个线程要保存的中间值越多，可能降低占用率，反而变慢——需要实测权衡；
3. **与 tiling 的配合**：融合通常发生在"计算完一个 tile 的 epilogue"，而不是整个输出。这也是为什么 TensorRT 等引擎的融合优化器要考虑内存布局和 tile 边界。

嵌入式类比：寄存器/共享内存是"片上资源"，全局内存是"片外存储"。融合 = 尽量在片上把事干完，减少片外往返；但片上资源有限，贪多会 spill（寄存器溢出到局部内存），反而更慢。**和 MCU 上寄存器分配溢出一个道理。**

## 5. 练习与里程碑

### 练习

1. **跑通朴素卷积**：把 2.1 节 kernel 补全（加 `main` 和随机初始化），用 `nvcc` 编译运行，与 CPU 参考实现对比正确性。
2. **量化融合收益**：算一个 `C_out=64, H=W=32, K=3` 的 `Conv + ReLU`，融合后省多少字节访存？（提示：64×32×32×4×2 = 512KB，读写各一次）
3. **融合 GEMM+ReLU**：拿矩阵乘法篇的 tiled sgemm，加一行 epilogue ReLU，对比融合前后运行时间（用 ncu 看 dram 读写字节是否下降约一个特征图）。
4. **思考 im2col 边界**：3×3 卷积展开 9 倍，为什么深层网络里通常只在浅层或小模型用 im2col？（提示：中间特征图的内存压力）

### 里程碑自检

- [ ] 能画出 im2col 的展开过程并说明内存膨胀原因
- [ ] 能解释隐式 GEMM 如何既复用 GEMM 的 tiling 又不展开矩阵
- [ ] 能说出算子融合省掉的访存来自哪里（中间张量的写+读）
- [ ] 能写出融合 ReLU 到 GEMM/卷积的修改位置（epilogue）
- [ ] 能说出融合的三个工程边界（依赖/寄存器/布局）

## 6. 小结

本节完成了从 GEMM 到卷积的 kernel 设计闭环：

- **卷积三条路线**：朴素（正确性基准）→ im2col（复用 GEMM，但内存膨胀 K×K 倍）→ 隐式 GEMM（索引映射 + 共享内存复用，生产首选）；
- **算子融合**：把相邻算子的计算折叠进前一个 kernel 的 epilogue，省掉中间张量的整次写读——这是 Roofline 结论的直接应用，也是推理引擎图优化的核心手段。

到这里，我们手工完成了一个算子开发者最核心的两类工作：**把新算子映射成高性能 kernel，把多个算子融合成更少的 kernel**。但手工优化每个算子不可持续——模型结构日新月异、硬件平台多样，业界正把这两件事交给**编译器自动完成**：给定一个高层描述，自动生成融合好的 CUDA kernel。这正是 AI 编译器（TVM / MLIR / Triton）在做的事。

> 🏷️ 标签：#卷积 #im2col #隐式GEMM #算子融合 #kernel设计 #访存优化
