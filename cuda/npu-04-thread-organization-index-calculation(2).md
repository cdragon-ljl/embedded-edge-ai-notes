# 线程组织与索引计算：CUDA 双层三维坐标系统

> 系列：从单片机到 NPU · NPU-04
> 前置：NPU-02（首个 CUDA 程序）、NPU-03（GPU 存储层次）
> 配套环境：RTX 5060 Ti（sm_120，deviceQuery 实测为准）

## 0. 本节目标

本节解决两个问题：**GPU 如何组织数万个线程**，以及**每个线程如何确定自己负责的数据区间**。前者是 CUDA 编程模型的骨架（Grid–Block–Thread 两层结构），后者是编写一切 kernel 的前提（索引公式）。

读完本节，读者应能：
1. 准确描述 CUDA 的线程层次结构与四个内置变量的语义；
2. 从定义推导一维、二维、三维索引公式，并完成行主序矩阵到线程坐标的映射；
3. 熟悉硬件资源上限，避免写出无法启动的 kernel。

## 1. 线程层次结构的正式定义

CUDA 将线程组织为两级结构，每级最多三维：

```
Grid（网格）＝ 若干 Block 的三维排列
 ├── Block（线程块）＝ 若干 Thread 的三维排列
 │    ├── Thread（线程）：kernel 中可并行执行的最小单元
 │    └── Block 内线程通过共享内存（shared memory）与 __syncthreads() 协作
 └── Block 之间无共享内存，仅能通过全局内存通信
```

**定义 1（Grid）**：一次 kernel 启动所创建的全部线程构成一个 Grid。Grid 按三维索引 `(x, y, z)` 排列若干 Block，各维尺寸由启动配置 `gridDim` 指定。

**定义 2（Block）**：Grid 内的线程分组单元。Block 内线程共享同一块共享内存，并可通过屏障同步（`__syncthreads()`）对齐执行进度。Block 是 GPU 调度到流处理器（Streaming Multiprocessor, SM）上的基本单位——同一 Block 必然驻留在同一 SM 上执行，这是共享内存与同步可行的硬件前提。

**定义 3（Thread）**：执行 kernel 代码的最小并发单元。每个线程拥有独立的寄存器、程序计数器与私有局部变量。

**嵌入式工程类比**：可将 Block 类比为可互相通信的"处理器核组"（如多核 MCU 上的一个核，或 DSP 的一个任务组），Grid 类比为整个处理器阵列。核组内成员可共享片上存储并同步；核组之间只能通过片外总线（全局内存）通信。这一映射决定了后续所有访存优化的方向。

需要强调的是，**Block 内线程共享同一 SM 是 CUDA 性能模型的核心约束**：共享内存带宽远高于全局内存，但容量有限（详见 NPU-03 与 NPU-05 的 tiling 优化）。

【图1：CUDA 两层三维线程结构示意图】

```text
Grid（由 blockIdx 索引，三维）
┌───────────────────────────────────────────┐
│ Block(0,0)  Block(1,0)  Block(2,0)  ...   │
│ ┌────────┐  ┌────────┐  ┌────────┐        │
│ │ . . .  │  │ . . .  │  │ . . .  │        │
│ │ . T .  │  │ . T .  │  │ . T .  │        │
│ │ . . .  │  │ . . .  │  │ . . .  │        │
│ └────────┘  └────────┘  └────────┘        │
│ Block(0,1)  Block(1,1)  Block(2,1)  ...   │
│ ...                                        │
└───────────────────────────────────────────┘
   blockIdx.x → 向右；blockIdx.y → 向下

Block 内部（由 threadIdx 索引，三维）
┌────────────────┐
│ (0,0) (1,0) (2,0) ...   ← threadIdx.x 向右
│ (0,1) (1,1) (2,1) ...   ← threadIdx.y 向下
│ ...                     ← threadIdx.z 向里
└────────────────┘
```

> 图1 生图 prompt（AI 生图工具用）：示意图，白底科技风。左侧画一个大矩形（Grid），内含 3×2 个小方块（Block），方块间以浅灰网格线分隔，标注 "Grid · blockIdx(x,y)"；取其中一个方块沿虚线放大至右侧，内部画 4×4 个圆点（Thread），标注 "Block · threadIdx(x,y)"。蓝色主色 #1565C0。比例 16:9，文字中文。

## 2. 内置变量及其语义

CUDA 在 kernel 内部提供四个只读内置变量（均为 `dim3` 或整型，无需声明即可使用），其语义如下：

| 变量 | 类型 | 语义 |
|:---|:---|:---|
| `threadIdx` | `dim3`（uint3） | 当前线程在其所在 Block 内的三维坐标，各分量 `x/y/z` 从 0 起 |
| `blockIdx` | `dim3`（uint3） | 当前线程所在 Block 在 Grid 内的三维坐标，各分量从 0 起 |
| `blockDim` | `dim3` | 当前 Block 各维的线程数（由启动配置决定，对所有线程一致） |
| `gridDim` | `dim3` | 当前 Grid 各维的 Block 数（由启动配置决定） |

两点说明：
1. **未指定的维度默认为 1**。启动配置 `dim3 block(16, 16)` 实际为 `(16, 16, 1)`，Block 共 256 线程。
2. `blockDim` 与 `gridDim` 是**同 Block/同 Grid 内所有线程共享的常量**；`threadIdx` 与 `blockIdx` 是**每个线程各自持有**的坐标。二者的区别是理解索引公式的关键。

## 3. 一维索引公式的推导

NPU-02 的 vectorAdd 中，每个线程通过一行代码定位自己的元素：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

该公式可由定义直接导出。设 Grid 为 `gridDim.x` 个 Block 的一维排列，每个 Block 含 `blockDim.x` 个线程，则**线性化（linearization）**规则为：线程的全局编号 = 其所在 Block 之前所有 Block 的线程总数 + 该线程在 Block 内的偏移，即

```text
i = blockIdx.x × blockDim.x + threadIdx.x        （式 3-1）
```

以 1,048,576 个元素、Block 大小 256 为例：Grid 含 4096 个 Block（`gridDim.x = 4096`）。线程 `(blockIdx.x=1, threadIdx.x=0)` 的全局编号为 `1×256+0 = 256`，即第 257 个元素。

**嵌入式工程类比**：等价于一条流水线上"工段号 × 每工段工位数 + 工位内编号"的工件寻址方式。该映射是后续所有维度推广的基础。

## 4. 二维映射：矩阵与行主序

### 4.1 行主序存储

C 语言中，`M×N` 矩阵按**行主序（row-major）**连续存储：第 `i` 行第 `j` 列元素 `A[i][j]` 的线性地址为

```text
addr(A[i][j]) = base + (i × N + j) × sizeof(element)        （式 4-1）
```

即先存储第 0 行的 N 个元素，再存储第 1 行，依此类推。

**嵌入式工程类比**：LCD 帧缓冲的寻址 `offset = y × width + x` 即行主序。若读者写过显示驱动，对此应已相当熟悉：`y` 对应行索引 `i`，`x` 对应列索引 `j`，`width` 对应列数 `N`。

```text
A[0][0] A[0][1] ... A[0][N-1] │ A[1][0] A[1][1] ... A[1][N-1] │ ...
偏移   0      1    ...  N-1   │  N      N+1    ...  2N-1      │ ...
       ├───── 第0行 N 个 ─────┤ ├───── 第1行 N 个 ────────────┤
```

### 4.2 线程坐标到矩阵坐标的映射

设矩阵大小为 `M×N`，目标为"每个线程计算一个元素"。映射分两步：

**第一步，确定 Block 形状**：使用二维 Block，如 `dim3 block(16, 16)`（每 Block 256 线程）。将 `threadIdx.x` 分配给列方向、`threadIdx.y` 分配给行方向。**选择依据**：CUDA 约定 `x` 是最内层（变化最快）维度，与行主序中"列是最内层下标"的规律一致（见式 4-1：`j` 变化 1 时地址变化 1，`i` 变化 1 时地址变化 N）。保持"最内层对齐"可使同一 Block 内相邻线程访问相邻地址，是访存合并（coalescing）的前提（NPU-05 将给出量化分析）。

**第二步，确定 Grid 形状**：Grid 须覆盖整个矩阵，各维 Block 数向上取整：

```cuda
dim3 grid((N + block.x - 1) / block.x,   // 列方向所需 Block 数
          (M + block.y - 1) / block.y);  // 行方向所需 Block 数
```

**第三步，推导索引公式**。沿用式 3-1，对每一维应用同一线性化规则：

```text
row = blockIdx.y × blockDim.y + threadIdx.y        （式 4-2a）
col = blockIdx.x × blockDim.x + threadIdx.x        （式 4-2b）
```

由于 Grid 可能超出矩阵边界（向上取整所致），kernel 内必须做边界检查：

```cuda
if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] + B[row * N + col];
}
```

**公式 4-2 可推广至任意维度：每一维的索引均为 `blockIdx × blockDim + threadIdx`。** 这是 CUDA 索引计算的统一规律，后续三维映射（5.2 节）与 NPU-05 的矩阵乘法均直接复用。

### 4.3 为什么列方向用 x、行方向用 y

初学者常混淆该对应关系，此处给出明确理由。CUDA 的线程坐标与存储布局存在隐含约定：
- 存储层面，行主序下**相邻内存地址对应列下标 `j` 的变化**（式 4-1）；
- 线程层面，**`threadIdx.x` 是变化最快的维度**（Block 内连续编号的线程沿 x 方向排列）。

因此将 `col` 与 `threadIdx.x` 绑定，可保证 Block 内相邻线程访问相邻地址，实现访存合并。若颠倒（`col` 绑 `threadIdx.y`），Block 内相邻线程将访问地址相差 N×4 字节的元素，内存带宽利用率将急剧下降。该结论将在 NPU-05 的 naive 矩阵乘法中通过实测验证。

【图2：线程坐标与数据坐标对齐示意】

```text
Block(16×16) 与 16×16 数据子块一一对应
数据子块（行主序连续存放）：
 row0: [c0][c1][c2] ... [c15]    地址连续
 row1: [c0][c1][c2] ... [c15]    ↑ threadIdx.x 变化 → 地址连续（合并访问）
 ...
 线程映射：threadIdx.y → row，threadIdx.x → col
```

> 图2 生图 prompt：示意图，白底。左侧画一个 4×4 的 Block 网格，每个格子内标注 `(tx,ty)`；右侧画同一大小的矩阵子块，每格标注地址偏移 `0,1,2,...,15,16,...`；两者之间画同色对应箭头，x 方向箭头标注"threadIdx.x → col → 地址连续"，y 方向标注"threadIdx.y → row"。蓝色主色 #1565C0。比例 16:9。

## 5. 三维扩展

### 5.1 定义

三维数据（体素、视频帧、特征图 `W×H×C`）可直接用三维 Block 映射。启动配置与索引公式如下：

```cuda
dim3 block(8, 8, 8);   // 每 Block 512 线程
dim3 grid(ceilDiv(W, 8), ceilDiv(H, 8), ceilDiv(C, 8));

int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

其中 `ceilDiv(a,b) = (a + b - 1) / b`。

### 5.2 讨论

三维映射并无新原理，仅是式 3-1 的逐维复用。实际工程中需注意两点：
1. 特征图常用 `C×H×W`（通道在前）布局（如 NCHW），此时应将最内层维度与 `threadIdx.x` 对齐，以维持访存合并；
2. 三维 Block 总线程数仍受 1024 上限约束（第 6 节），故 z 维通常取小值（如 4~8）。

## 6. 硬件资源约束

线程规模并非任意设定。下表为 CUDA 官方文档（CUDA C++ Programming Guide）规定的统一上限，适用于所有 CUDA 设备：

| 维度 | Block 各维线程数上限 | Grid 各维 Block 数上限 |
|:---|:---:|:---:|
| x | 1024 | 2,147,483,647（2³¹−1） |
| y | 1024 | 65,535 |
| z | 64 | 65,535 |
| Block 总线程数 | 1024 | — |
| 单 SM 驻留线程上限 | 2048（5060 Ti，以 deviceQuery 实测为准） | — |

工程指导：
1. **Block 总线程数不得超过 1024**。常用取 128 / 256 / 512，足以隐藏访存延迟且资源占用适中；
2. **Grid 的 x 维上限极大（约 21 亿）**，一般无需担心；y/z 维上限为 65,535，超大任务应将最大维度放至 x；
3. 超限时 kernel 启动失败，返回 `cudaErrorInvalidValue`。建议在 host 端打印 grid/block 配置便于排查。

**嵌入式工程类比**：如同 MCU 的中断优先级与 DMA 通道数量，线程规模上限是硬件的"资源预算"。不需要背诵具体数值，但必须知道如何查询（`deviceQuery` 样例程序）并在设计中留出余量。

## 7. 完整示例：二维矩阵加法

将 NPU-02 的向量加法扩展为二维矩阵加法，完整代码如下：

```cuda
// matrixAdd2d.cu —— 二维 Block/Grid 矩阵加法（每个线程一个元素）
#include <cstdio>
#include <cuda_runtime.h>

#define M 1024
#define N 1024

__global__ void matrixAdd(const float *A, const float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 式 4-2a
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 式 4-2b
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

int main() {
    const size_t bytes = M * N * sizeof(float);

    float *hA = new float[M * N];
    float *hB = new float[M * N];
    float *hC = new float[M * N];
    for (int i = 0; i < M * N; i++) { hA[i] = 1.0f; hB[i] = 2.0f; }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);                       // 每 Block 16×16 = 256 线程
    dim3 grid((N + block.x - 1) / block.x,    // 4.2 节：各维向上取整
              (M + block.y - 1) / block.y);
    matrixAdd<<<grid, block>>>(dA, dB, dC);

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < M * N; i++) {
        if (hC[i] != 3.0f) { ok = false; break; }
    }
    printf("grid=(%d,%d) block=(%d,%d) → %s\n",
           grid.x, grid.y, block.x, block.y,
           ok ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
```

编译与运行：

```bash
nvcc -arch=sm_120 -O2 -o matrixAdd2d matrixAdd2d.cu && ./matrixAdd2d
# 期望输出：grid=(64,64) block=(16,16) → PASS
```

注意 `dim3 block(16, 16)` 未指定的 z 维默认为 1，Block 实际为 `(16,16,1)`，共 256 线程。该细节在三维编程中会反复出现。

## 8. 练习与里程碑

### 8.1 练习

1. **一维降维**：将示例改写为一维 Block（`dim3 block(256)`），在 kernel 内用 `row = i / N; col = i % N;` 换算。对比两版可读性与正确性，思考二维坐标系的必要性。
2. **Block 形状实验**：将 Block 改为 `(8,8)` 与 `(32,32)` 分别运行。三者均应 PASS。分析 Grid 计算公式如何保证正确性（提示：向上取整与边界检查）。
3. **三维练习**：编写 kernel，对 `32×32×3` 的三通道数据逐元素加 1，使用三维 Block，验证结果。

### 8.2 里程碑

- [ ] 能画出两层三维坐标系，准确说明 `blockIdx/threadIdx/blockDim/gridDim` 的语义与区别；
- [ ] 能从式 3-1 推导二维索引公式（式 4-2），并解释 x 对应列的理由；
- [ ] 能根据矩阵规模估算 Grid/Block 配置，说明边界检查的必要性；
- [ ] matrixAdd2d 编译运行 PASS；
- [ ] 能解释行主序与访存合并的关系。

## 9. 下期预告

坐标系统与存储层次已齐备，下一节进入本系列第一个性能优化实战：**矩阵乘法**。它是神经网络全连接层、注意力机制的 QKV 变换与卷积底层实现的公共算子。NPU-05 将先给出朴素（naive）实现并完成访存量核算，再引入共享内存分块（tiling）优化，量化对比两者的算术强度差异。

**NPU-05 · 矩阵乘法优化：从朴素实现到共享内存分块**

> 🏷️ 线程组织 · 网格映射 · blockIdx · threadIdx · 行主序 · 二维索引
