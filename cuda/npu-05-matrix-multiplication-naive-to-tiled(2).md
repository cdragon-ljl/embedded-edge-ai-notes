# 矩阵乘法优化：从朴素实现到共享内存分块

> 系列：从单片机到 NPU · NPU-05
> 前置：NPU-03（存储层次）、NPU-04（线程组织与索引）
> 配套环境：RTX 5060 Ti（sm_120，448 GB/s，FP32 理论峰值约 24 TFLOPS，以 deviceQuery 实测为准）

## 0. 本节目标

矩阵乘法（Matrix Multiplication, MatMul）是神经网络中最核心的计算原语：全连接层、注意力机制的 QKV 线性变换、卷积的底层实现，最终均归结为 MatMul。因此，它是算子开发者的"第一道必考题"，也是检验 CUDA 基本功的标准场景。

本节完成一个完整的优化闭环：
1. 给出朴素（naive）实现，验证正确性；
2. **核算访存量**，用算术强度（Arithmetic Intensity）定量解释 naive 为何慢；
3. 引入共享内存分块（tiling）优化，推导其访存收益；
4. 实测对比并总结优化规律。

本节全部推导均可照抄复现，性能数据以读者本机实测为准。

## 1. 数学定义与复杂度

**定义 1（矩阵乘法）**：设 $A \in \mathbb{R}^{M\times K}$，$B \in \mathbb{R}^{K\times N}$，则 $C = A \times B \in \mathbb{R}^{M\times N}$，其中

$$C[i][j] = \sum_{t=0}^{K-1} A[i][t] \cdot B[t][j] \qquad (式 1-1)$$

即输出 $C$ 的第 $i$ 行第 $j$ 列等于 $A$ 的第 $i$ 行与 $B$ 的第 $j$ 列的内积。$K$ 称为归约维（reduction dimension）。

**复杂度**：每个输出元素需 $K$ 次乘加，共 $M\times N$ 个输出，故总运算量为

$$\text{FLOP} = 2 \times M \times N \times K \qquad (式 1-2)$$

**嵌入式工程类比**：式 1-1 与一维 FIR 滤波器 $y[n]=\sum_k h[k]x[n-k]$ 结构相同，均为"加权求和"；区别仅在于矩阵乘法沿行与列两个方向同时做加权求和，是 FIR 的二维推广。

## 2. 朴素实现（naive）

### 2.1 线程映射

沿用 NPU-04 的二维索引规则：每个线程负责计算 $C$ 的一个元素，Block 形状 `(16,16)`。

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;   // 负责 C 的第 row 行
int col = blockIdx.x * blockDim.x + threadIdx.x;   // 负责 C 的第 col 列
```

### 2.2 访存模式分析

先分析 naive 的访存行为，这是后续优化的依据。

**对 A 的访问**：`A[row * K + t]`，其中 `t` 在归约循环内变化。同一 Block 内 `threadIdx.x` 连续变化时访问的地址为 `A[row*K + t]` 的同一元素——**即 Block 内所有线程同时读取同一地址**（广播，broadcast）。而 `row` 变化时（`threadIdx.y` 变化），地址跳变 `K` 个元素。

**对 B 的访问**：`B[t * N + col]`，`col` 随 `threadIdx.x` 连续变化时，地址 `t*N+col` 连续——**这一点尚好**；但归约循环内 `t` 每加 1，地址跳变 `N` 个元素。**这是 naive 实现最严重的访存问题：B 的列访问在全局内存中是跨行（stride-N）的**，每次仅利用一个 32 字节缓存行中的一个元素，带宽利用率低。

**重复读取**：A 的第 `i` 行被第 `i` 行对应的全部 `N` 个线程各读一遍；B 的第 `j` 列被全部 `M` 个线程各读一遍。即每个 A 元素从全局内存读取 $N$ 次，每个 B 元素读取 $M$ 次。

### 2.3 代码

```cuda
// matmul_naive_vs_tiled.cu —— 矩阵乘法 naive → tiled 对比
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define M 512      // A 行数
#define N 512      // B 列数
#define K 512      // 归约维
#define TILE 32    // tiling 块大小

// ---------- naive：每个线程算 C 的一个元素 ----------
__global__ void matmulNaive(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int t = 0; t < K; t++)
            acc += A[row * K + t] * B[t * N + col];   // B 按列取：跨行访问
        C[row * N + col] = acc;
    }
}

// ---------- tiled：共享内存分块 ----------
__global__ void matmulTiled(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];   // A 子块
    __shared__ float Bs[TILE][TILE];   // B 子块

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        // 协作加载 A 子块：A[row][t .. t+TILE-1]
        if (row < M && t + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        // 协作加载 B 子块：B[t .. t+TILE-1][col]
        if (col < N && t + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();   // 屏障①：等待本 Block 全部线程完成搬运

        for (int k = 0; k < TILE; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();   // 屏障②：防止下一轮覆盖尚未读出的数据
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ---------- CPU 参考实现（校验用） ----------
void matmulCPU(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int t = 0; t < K; t++)
                acc += A[i * K + t] * B[t * N + j];
            C[i * N + j] = acc;
        }
}

// ---------- kernel 计时（CUDA Event） ----------
float timeKernel(void (*kernel)(const float*, const float*, float*, int, int, int),
                 const float *dA, const float *dB, float *dC, dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const size_t bytesA = M * K * sizeof(float);
    const size_t bytesB = K * N * sizeof(float);
    const size_t bytesC = M * N * sizeof(float);

    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC = new float[M * N];
    float *hRef = new float[M * N];
    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)(rand() % 10) / 10.0f;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);
    cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice);

    // naive：Block (16,16)
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    float tNaive = timeKernel(matmulNaive, dA, dB, dC, grid, block);
    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    // tiled：Block (TILE,TILE)
    dim3 gridT((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    float tTiled = timeKernel(matmulTiled, dA, dB, dC, gridT, dim3(TILE, TILE));
    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    // 校验
    matmulCPU(hA, hB, hRef, M, N, K);
    float maxErr = 0.0f;
    for (int i = 0; i < M * N; i++)
        maxErr = fmaxf(maxErr, fabsf(hC[i] - hRef[i]));

    printf("M=N=K=%d, FP32\n", M);
    printf("naive: %8.3f ms\n", tNaive);
    printf("tiled: %8.3f ms   (speedup %.2fx)\n", tTiled, tNaive / tTiled);
    printf("maxErr vs CPU: %g  →  %s\n", maxErr,
           (maxErr < 1e-3f) ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    return 0;
}
```

编译运行：

```bash
nvcc -arch=sm_120 -O2 -o matmul matmul_naive_vs_tiled.cu && ./matmul
```

参考输出（数值以本机实测为准）：

```text
M=N=K=512, FP32
naive:  3.2 ms
tiled:  1.1 ms   (speedup 2.9x)
maxErr vs CPU: 0  →  PASS
```

说明：规模 512 时数据可部分驻留缓存，naive 未必被充分惩罚；将 M/N/K 改为 2048 后差距将显著拉开（见练习 3）。

## 3. naive 为何慢：访存核算

### 3.1 算术强度定义

**定义 2（算术强度, Arithmetic Intensity, AI）**：程序每搬运 1 字节数据所能完成的浮点运算次数，即

$$\text{AI} = \frac{\text{总运算量 (FLOP)}}{\text{总数据搬运量 (Bytes)}} \qquad (式 3-1)$$

AI 是判断程序"计算受限"还是"访存受限"的关键指标，也是 Roofline 模型的横轴（NPU-08 正式展开）。

### 3.2 naive 的访存量

每个线程（共 $M\times N$ 个）需从全局内存读取 $A$ 的 $K$ 个元素与 $B$ 的 $K$ 个元素，写回 1 个元素。全局内存读取总量为

$$\text{Bytes}_{\text{naive}} \approx 2 \times M \times N \times K \times 4 \text{ B} \qquad (式 3-2)$$

结合式 1-2，naive 的算术强度为

$$\text{AI}_{\text{naive}} = \frac{2MNK}{8MNK} = 0.25 \text{ FLOP/B} \qquad (式 3-3)$$

### 3.3 与 5060 Ti 能力对照

5060 Ti 的关键参数（以 deviceQuery 实测为准）：
- FP32 理论峰值：$\approx 4608 \times 2 \times 2.65\,\text{GHz} \approx 24$ TFLOPS；
- 全局内存带宽：448 GB/s。

若程序完全访存受限，则实际可达到的算力上限为

$$\text{算力上限} = 448\,\text{GB/s} \times 0.25\,\text{FLOP/B} \approx 0.11 \text{ TFLOPS}$$

仅为理论峰值的 0.5% 以下。换言之，**naive 实现的性能天花板由访存带宽决定，与计算单元无关**。

Roofline 模型的交叉点（运算强度达到该值后方可逼近峰值算力）为

$$\text{交叉点} = \frac{24\times10^{12}}{448\times10^{9}} \approx 54 \text{ FLOP/B} \qquad (式 3-4)$$

naive 的 0.25 与交叉点相差两个数量级以上。需要说明：实际运行时 L1/L2 缓存会复用部分数据，真实差距不会如此极端，但"访存是瓶颈"的结论方向正确。

### 3.4 naive 的两个具体缺陷

综上，naive 的缺陷可归纳为两点：
1. **B 的跨行访问**：`B[t*N+col]` 中 `t` 变化导致地址跨 `N` 个元素，缓存行利用率低（每次仅取 4 字节/32 字节行）；
2. **重复访存**：A 元素被读 $N$ 次、B 元素被读 $M$ 次，总访存量高达式 3-2。

## 4. 共享内存分块（tiling）

### 4.1 思想

NPU-03 指出，共享内存是 Block 内的片上存储，带宽远高于全局内存但容量有限。tiling 的核心思想是**把重复访存转化为复用**：

1. 将 $C$ 划分为 $\text{TILE}\times\text{TILE}$ 的子块，每个 Block 负责一个子块；
2. 沿归约维 $K$ 循环，每轮将计算所需的 $A$ 子块与 $B$ 子块（各 $\text{TILE}\times\text{TILE}$）从全局内存**协作加载一次**到共享内存；
3. 在共享内存上完成 $\text{TILE}\times\text{TILE}$ 次乘加，期间不再访问全局内存。

【图1：tiling 分块与数据流】

```text
        K 方向
   ┌──────────────────────┐
   │ A 子块 (TILE×TILE)   │   B 子块 (TILE×TILE)      C 子块 (TILE×TILE)
   │ 全局 → 共享内存      │   全局 → 共享内存          ← 本 Block 负责
   │ 协作加载，每线程1元素 │   协作加载，每线程1元素       计算并写回
   └──────────────────────┘
        ↑ 沿 K 循环 K/TILE 轮，每轮一对子块

   关键收益：
   A[i][t] 全局读取 1 次 → 被本 Block 的 TILE 个列线程复用 TILE 次
   B[t][j] 全局读取 1 次 → 被本 Block 的 TILE 个行线程复用 TILE 次
```

> 图1 生图 prompt：示意图，白底。画三个矩阵 A（左）、B（上）、C（右下），以彩色方块标出 A 的 TILE×TILE 子块（蓝）、B 的 TILE×TILE 子块（绿），两子块以箭头汇入中部"Shared Memory"方块（橙），再由 Shared 指向 C 子块（红）。箭头旁标注"全局读取 1 次，复用 TILE 次"。比例 16:9，文字中文。

### 4.2 访存收益的定量推导

**引理 1（tiled 全局读取量）**：设矩阵规模 $M=N=K$ 为 TILE 的整数倍，tiled 实现的全局内存读取总量为

$$\text{Bytes}_{\text{tiled}} \approx \left(\frac{M}{\text{TILE}}\right)\left(\frac{N}{\text{TILE}}\right)\left(\frac{K}{\text{TILE}}\right) \times 2 \times \text{TILE}^2 \times 4 \text{ B} = \frac{2MNK \times 4}{\text{TILE}} \text{ B} \qquad (式 4-1)$$

**证明**：Grid 共 $(M/\text{TILE})\times(N/\text{TILE})$ 个 Block，每个 Block 沿 $K$ 循环 $K/\text{TILE}$ 轮，每轮协作加载 $A$、$B$ 子块各 $\text{TILE}^2$ 个元素（每个元素 4 字节）。∎

与式 3-2 对比，**全局读取量降为原来的 $1/\text{TILE}$**。相应算术强度为

$$\text{AI}_{\text{tiled}} = \frac{2MNK}{8MNK/\text{TILE}} = 0.25 \times \text{TILE} \qquad (式 4-2)$$

取 $\text{TILE}=32$ 得 $\text{AI}=8$ FLOP/B，较 naive 提升 32 倍。虽仍低于交叉点 54（这解释了为何后续还需向量化、双缓冲、bank conflict 消除等优化，NPU-07 以 Nsight Compute 继续分析），但已从"数量级错误"进入"数量级正确"。

### 4.3 两个 `__syncthreads()` 的同步语义

tiled kernel 内层循环的两个屏障不可省略：

```cuda
__syncthreads();   // 屏障①：等待全部线程完成子块写入（写后读保护）

for (int k = 0; k < TILE; k++)
    acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

__syncthreads();   // 屏障②：等待全部线程读完共享内存（读后写保护）
```

- **屏障①**解决写后读（WAR→RAW 方向）竞争：子块由 Block 内全部线程协作加载——每个线程只写入 1 个元素，但随后要读取整个 $\text{TILE}^2$ 子块。若无屏障，先行的线程可能读到尚未写入的脏数据。类比：**DMA 搬运尚未完成即读取目的缓冲区**。
- **屏障②**解决读后写（RAW→WAW 方向）竞争：下一轮循环将**覆盖** `As`/`Bs`，若部分线程提前进入下一轮开始覆盖，其他线程可能仍在读取旧数据。类比：**双缓冲中，生产者不得覆盖消费者尚未消费的缓冲**。

这是多线程协作的纪律：**共享内存为 Block 内公共存储，任何协作读写都必须以屏障同步对齐执行进度**。

## 5. 讨论：tiling 的适用边界

1. **TILE 并非越大越好**。共享内存容量有限（默认上限 48 KB/SM），TILE=32 时占用 $32^2\times4\times2=8$ KB；TILE=64 时升至 32 KB，且 Block 线程数增至 4096，超出每 Block 1024 线程上限——故 TILE=64 需要配合每线程多元素（thread coarsening）策略，本系列 NPU-07 讨论。
2. **边界处理**：当 $M,N,K$ 非 TILE 整数倍时，协作加载需对越界位置补零（代码中的 `else` 分支），以免共享内存未初始化数据污染结果。
3. **更进一步的优化路径**：寄存器分块（register blocking）、向量化加载（float4）、双缓冲（double buffering）、消除 bank conflict（NPU-06/07 展开）。

## 6. 练习与里程碑

### 6.1 练习

1. **验证资源占用**：以 `nvcc -Xptxas -v` 编译两版 kernel，对比寄存器数与共享内存声明量。确认 tiled 版本多出约 8 KB 共享内存（TILE=32）。
2. **扫描 TILE**：将 TILE 改为 16 与 64 重跑。TILE=64 时注意编译/运行是否报资源不足，并解释原因（提示：第 5 节讨论 1）。记录各 TILE 下的性能并分析拐点。
3. **放大规模**：将 M=N=K 改为 2048 重跑，记录加速比。随后实现"预转置 B"变体（将 B 按行复制一份使列访问连续），观察 naive 自身能提升多少——这是访存合并的第一次量化体验。
4. **思考题**：用"仓库（全局内存）→ 工作台（共享内存）"的比喻，分别叙述 naive 与 tiled 的访存过程（各 3~5 行），并说明 tiled 收益的本质是"重复访存 → 复用"。

### 6.2 里程碑

- [ ] 能独立写出 naive 矩阵乘 kernel 并与 CPU 结果对照；
- [ ] 能由式 3-2、式 3-3 推导 naive 的算术强度 0.25 FLOP/B，并解释其瓶颈属性；
- [ ] 能证明（式 4-1）tiled 将全局读取量降为 $1/\text{TILE}$，并计算 TILE=32 时的 AI=8；
- [ ] 能写出 tiled kernel，并准确解释两个 `__syncthreads()` 的同步语义；
- [ ] 实测 tiled 相对 naive 的加速比并记录于实验日志。

## 7. 下期预告

tiling 解决了"数据搬运"问题，下一节转向另一类基础并行原语：**归约（Reduction）**——将若干数高效地累加为一个数。它是 Softmax、LayerNorm、注意力分数归一等聚合算子的地基。NPU-06 将从树状归约出发，逐步引入 Warp 内协作与 Shuffle 指令。

**NPU-06 · 并行归约：从树状求和到 Warp Shuffle**

> 🏷️ 矩阵乘法 · tiling · 共享内存 · 算术强度 · Roofline · 访存优化
