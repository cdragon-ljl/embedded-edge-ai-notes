# GPU 内存层次：Global / Shared / Register 与嵌入式存储直觉

> 系列：从单片机到 NPU · NPU-03
> 前置：NPU-02（首个 CUDA kernel）
> 配套环境：RTX 5060 Ti（sm_120），内存参数以 deviceQuery 实测为准

## 0. 本节目标

嵌入式工程师接触新芯片的第一步，通常是阅读 datasheet 的**内存映射**：Flash 位置、SRAM 位置、外设寄存器地址、总线拓扑。该认知决定后续所有代码的性能上限。

本节以同样方法建立 RTX 5060 Ti 的**内存层次模型**，回答三个问题：
1. GPU 存在哪些存储层级，其容量、速度、可见性如何；
2. 各层级与嵌入式存储体系（Flash / SRAM / 外部 SDRAM）如何对应；
3. 数据应放置于哪一层级的选型原则。

本节建立的模型是后续全部优化（tiling、访存合并、算子融合）的共同基础。

## 1. 嵌入式存储直觉回顾

以 STM32F407 为例，其内存映射可简化为：

```text
0x08000000  Flash  —— 1MB，慢，掉电不丢（代码、常量表）
0x20000000  SRAM   —— 192KB，快，掉电丢（变量、栈）
0x40000000  外设   —— GPIO / USART / DMA 寄存器
```

工程实践中形成的内存规划原则：

- 只读常量表置于 Flash，利用只读缓存；
- 高频变量置于 SRAM（必要时指定内核 SRAM）；
- 大块数据置于外部 SDRAM，需考虑总线延迟与突发传输；
- 尽量减少外部存储访问，能缓存则缓存。

**这些原则在 GPU 上完整成立**。GPU 存储层次仅是"Flash / SRAM / 外部 SDRAM"三元结构的扩展：层级更多、速度差距更极端、管理方式更显式。

## 2. GPU 内存层次总览

【图1：GPU 内存金字塔（速度与容量）】

```text
          速度最快 · 容量最小
   ┌─────────────────────────────┐
   │       Register 寄存器        │  每线程私有，编译器自动分配
   ├─────────────────────────────┤
   │   Shared Memory 共享内存     │  同一 Block 内线程共享（片上 SRAM）
   ├─────────────────────────────┤
   │  Constant / Texture（只读）  │  独立缓存，适合广播/纹理访问
   ├─────────────────────────────┤
   │    Global Memory 全局内存    │  全体线程共享（片外 GDDR7 显存）
   │ （Local Memory 物理上也在此） │
   └─────────────────────────────┘
          速度最慢 · 容量最大
```

核心记忆模型：**寄存器是"手上的工具"，共享内存是"工作台上的物料"，全局内存是"仓库"**。后续所有优化（tiling、访存合并、算子融合）都围绕同一命题：如何让数据尽量留在"手边"，减少对"仓库"的访问。

> 图1 生图 prompt：金字塔/层级图，深蓝科技底 #0B1B3A。自下而上四层：宽大灰色块"Global Memory（片外 GDDR7，慢，8GB）"、蓝色块"Constant/Texture（只读缓存）"、青色块"Shared Memory（片上 SRAM，快）"、顶层红色小块"Register（最快）"。每层右侧标注容量与速度箭头（向上变快变小）。比例 16:9，文字中文。

## 3. 各存储层级详解

### 3.1 Global Memory（全局内存）

**定义 1（全局内存）**：GPU 的显存，位于片外，容量最大、访问最慢，是 CPU 与 GPU 数据交换的必经通道。5060 Ti 配置为 8 GB GDDR7、约 448 GB/s 带宽、128-bit 位宽。

**嵌入式工程类比**：等价于 MCU 外扩 SDRAM——容量大、需过总线、访问慢、掉电丢数据。

三个关键特性：

1. **可见性**：所有 Block、所有线程均可读写；CPU 经 `cudaMemcpy` 亦可访问（NPU-02 的 `d_a/d_b/d_c` 均属全局内存）；
2. **生命周期**：`cudaMalloc` 申请、`cudaFree` 释放，内容初始为脏数据（同 `malloc`）；
3. **延迟**：访问延迟为数百时钟周期量级（与架构及负载相关，以实测为准）。

性能要点预告：**访存合并（Coalesced Access）**——相邻线程访问相邻地址时，硬件以 128 字节为粒度合并搬运，效率最高；乱序访问则效率骤降。这等价于 DMA 配置中对齐、连续搬运的要求。NPU-05 将给出量化验证。

### 3.2 Shared Memory（共享内存）

**定义 2（共享内存）**：每个 SM 内部的高速存储，由同一 Block 内全部线程共享。它位于计算单元附近而非显存中，访问速度高于全局内存一个数量级以上，容量为每 SM 数十至上百 KB（以 deviceQuery 实测为准），**需显式手动管理**。

**嵌入式工程类比**：等价于 MCU 片上 SRAM（紧耦合 SRAM/TCM）——靠近核心、速度快、容量小、手动规划。

三条规则：

1. **作用域 = Block**：同一 Block 内线程可互相读写；不同 Block 之间完全不共享。可将每个 Block 视为独立小组，共享内存为小组内的"公共白板"；
2. **生命周期 = Block 生命周期**：Block 启动时分配、结束时自动释放，无需（亦不可）手动 free；
3. **声明方式**：kernel 内以 `__shared__` 修饰数组，**每个 Block 各有一份独立副本**。

```cuda
__global__ void demo(float *in) {
    __shared__ float s[256];   // 每个 Block 各有一份 256 float 共享数组
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = in[i];    // 每线程写入自己负责的元素
    __syncthreads();           // 屏障：等待本 Block 全部线程写完
    // 此后本 Block 内任意线程均可读 s[0..255]
}
```

**为何是性能关键**：矩阵乘法等算子的核心优化——**tiling（分块）**——将大矩阵切块，每块自全局内存搬运一次至共享内存，随后在共享内存内反复使用，避免重复访问全局内存。等价于将仓库物料一次性搬至工作台，后续操作全部在台上完成。NPU-05 将给出完整推导。

> 注：共享内存的容量限制与 Bank Conflict（多线程同时访问同一存储体导致的串行化）为两大考点，NPU-05 之后深挖，本节先建立"片上 SRAM"心智模型。

### 3.3 Register（寄存器）

**定义 3（寄存器）**：每个线程私有的最快存储，零延迟，由编译器自动分配。每个 SM 拥有一块寄存器堆（常见规模 65536 个 32 位寄存器，Blackwell 以 deviceQuery 实测为准），在 SM 上同时运行的全部线程间切分。

**嵌入式工程类比**：等价于 Cortex-M 的 R0–R15。C 代码无需（亦不可）手动分配，但代码写法（局部变量数量、循环展开程度）直接影响寄存器分配结果。

核心约束：**每线程寄存器上限**。CUDA 规定每线程最多使用 255 个寄存器（sm_120，以 CUDA 文档为准）。局部变量过多超出上限时，将触发 3.4 节的 Local Memory 溢出。

### 3.4 Local Memory（本地内存）

**定义 4（本地内存）**：并非独立硬件。它是"寄存器不足时，编译器将线程私有变量溢出（spill）至全局内存"的产物。逻辑上线程私有，物理上位于显存，访问速度与全局内存相同。

**嵌入式工程类比**：等价于寄存器溢出至栈——中断函数局部变量过多时，编译器被迫将部分变量压栈保存，访问速度骤降。

触发条件：

- 每线程寄存器用量过高（大数组、大结构体、深度循环展开）；
- **动态数组索引**（`arr[i]` 的 i 非编译期常量）——寄存器无法动态索引，只能置于内存；
- 局部数组/结构体过大，无法装入寄存器。

检测方法：编译时加 `-Xptxas -v`：

```bash
nvcc -arch=sm_120 -Xptxas -v -o demo demo.cu
# 输出类似：
# 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads   ← 健康
# 若 spill stores/loads 非 0，说明存在变量被挤至 Local Memory
```

**优化铁律**：spill 非零即需警惕，意味着热数据已落入"仓库"。该指标是 NPU-05~07 性能优化的常规检查项。

### 3.5 Constant Memory（常量内存）

**定义 5（常量内存）**：一块 64 KB 只读内存，带独立缓存。其最大特性：**当 Block 内所有线程读取同一地址时，硬件广播一次即可全部获得**，效率极高。

**嵌入式工程类比**：等价于 Flash 中的只读查表（正弦表、CRC 表）——只读、靠缓存加速。

用法：以 `__constant__` 修饰全局变量，CPU 侧以 `cudaMemcpyToSymbol` 写入，kernel 内直接读取。本系列后续（归一化算子的系数表等）将使用。

## 4. 数据放置决策表

| 内存 | 容量（5060 Ti） | 速度 | 可见性 | 生命周期 | 嵌入式类比 | 适用数据 |
|:---|:---|:---|:---|:---|:---|:---|
| Register | 每 SM 65536 个 32 位 | 最快（零延迟） | 单线程私有 | kernel 执行期 | Cortex-M 寄存器 | 循环变量、中间计算 |
| Shared | 每 SM 数十~上百 KB | 极快 | 同一 Block | Block 生命周期 | 片上 SRAM / TCM | 分块数据、Block 内协作 |
| Constant | 64 KB | 快（只读+广播） | 全部线程（只读） | 程序生命周期 | Flash 查表 | 权重系数、查表 |
| Local | 借用全局内存 | 慢（落入显存） | 单线程私有 | kernel 执行期 | 寄存器溢出至栈 | 尽量避免 |
| Global | 8 GB | 慢（数百周期） | 全部线程 + CPU | cudaMalloc→cudaFree | 外部 SDRAM | 大规模输入输出 |

选型原则（与 MCU 一致）：**能放片上不放片外，能放寄存器不放共享内存**。GPU 性能优化的本质，即让数据尽量停留于金字塔上层。

## 5. 动手验证

### 5.1 查询本机内存参数

不猜不背，直接查询：

```cuda
// memquery.cu —— 打印 5060 Ti 内存关键参数
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);

    int regsPerSM = 0, sharedPerSM = 0, sharedPerBlock = 0;
    cudaDeviceGetAttribute(&regsPerSM,      cudaDevAttrMaxRegistersPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&sharedPerSM,    cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&sharedPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    printf("GPU: %s\n", p.name);
    printf("SM 数量: %d\n", p.multiProcessorCount);
    printf("全局内存: %.1f GB\n", p.totalGlobalMem / 1e9);
    printf("每 SM 寄存器数: %d\n", regsPerSM);
    printf("每 SM 共享内存: %d bytes\n", sharedPerSM);
    printf("每 Block 共享内存上限(可申请): %d bytes\n", sharedPerBlock);
    return 0;
}
```

> 说明：每线程寄存器上限建议以 `cudaDevAttrMaxRegistersPerBlock` 等属性查询，本段代码重点在于展示每 SM 寄存器堆与共享内存的真实数值——**实测值即后续优化的资源预算**。

编译运行：

```bash
nvcc -arch=sm_120 -o memquery memquery.cu && ./memquery
```

### 5.2 Block 内求和示例

验证"Shared 为 Block 公共存储"的语义：每 Block 将本组线程负责的元素写入共享内存，再由线程 0 汇总。

```cuda
// blocksum.cu —— 用共享内存做 Block 内求和（归约雏形）
#include <cstdio>
#include <cuda_runtime.h>

__global__ void blockSum(const float *in, float *out, int n) {
    __shared__ float s[256];                // 每 Block 一份共享数组

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (i < n) ? in[i] : 0.0f;  // 边界外补 0
    __syncthreads();                         // 等待本 Block 256 线程写毕

    if (threadIdx.x == 0) {                  // 线程 0 汇总本 Block
        float sum = 0.0f;
        for (int k = 0; k < 256; k++) sum += s[k];
        out[blockIdx.x] = sum;               // 每 Block 输出一个部分和
    }
}

int main() {
    const int n = 1 << 20;                   // 1M 元素
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    float *h_in = new float[n];
    float *h_out = new float[blocks];
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;   // 全 1，便于核对

    float *d_in, *d_out;
    cudaMalloc(&d_in,  n * sizeof(float));
    cudaMalloc(&d_out, blocks * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    blockSum<<<blocks, threads>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0;
    for (int b = 0; b < blocks; b++) total += h_out[b];
    printf("total = %.0f (期望 %d)\n", total, n);
    printf("%s\n", (total == (float)n) ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}
```

编译运行：

```bash
nvcc -arch=sm_120 -O2 -o blocksum blocksum.cu && ./blocksum
# 期望输出：total = 1048576 (期望 1048576)  PASS
```

两个核心体会：

1. **`s` 每 Block 一份**：4096 个 Block 各有独立 `s[256]`，互不干扰——"Block 为独立小组"的物理含义；
2. **`__syncthreads()` 为 Block 内集合点**：确保"我写的数据你能读到"。缺少屏障将导致线程 0 读到他人尚未写入的脏数据，即经典的数据竞争。**嵌入式工程类比**：DMA 传输完成中断——必须等待搬运真正完成才能读取数据。

> 此处采用最简单的"线程 0 逐个累加"，仅用于理解语义。高效的树状归约与 Warp Shuffle 为 NPU-06 主题。

【图2：Global 与 Shared 访问路径对比】

```text
        Global Memory 访问（慢）                  Shared Memory 访问（快）
┌──────────────────────────┐   ┌──────────────────────────┐
│  GPU 芯片                 │   │  GPU 芯片                 │
│  ┌────────────┐  ←远→     │   │  ┌────────────┐          │
│  │ SM 计算单元 │ ──总线───▶│   │  │ SM 计算单元 │          │
│  └────────────┘ ←数百周期→ │   │  └────┬───────┘          │
│        每次访问走外部总线   │   │       │ 片上，极近        │
└──────────────────────────┘   │  ┌─────▼─────┐          │
     类比：读外部 SDRAM         │  │ Shared Mem │          │
                               │  └───────────┘          │
                               │  类比：读片上 SRAM       │
                               └──────────────────────────┘
```

> 图2 生图 prompt：对比图，白底。左半：芯片方块（SM）经长弯曲总线连接远处显存方块（GDDR7），箭头标注"数百周期·外部总线"，红色警示色；右半：SM 方块旁紧贴小方块（Shared Memory），短箭头标注"片上·极快"，绿色。两半下方各注"类比：读外部 SDRAM / 读片上 SRAM"。比例 16:9，文字中文。

## 6. 练习与里程碑

### 6.1 练习

1. **记录预算**：运行 5.1 的 memquery，记录每 SM 寄存器数、每 SM 共享内存、每 Block 共享内存上限。此即后续调优的"资源预算表"；
2. **对比实验**：将 5.2 的 `blocksum` 改为不使用共享内存——每线程直接将 `in[i]` 累加至 `out[blockIdx.x]`（先清零 out）。观察结果是否仍正确并解释原因（提示：多线程同时写同一地址 = 数据竞争）；
3. **思考题**：为何称 Shared Memory 为"手动管理的缓存"？结合 MCU 上手动规划 SRAM 缓冲池（DMA 双缓冲、音频环形缓冲）的经验，撰写 3~5 行类比笔记。此点想通，NPU-05 的 tiling 已理解过半。

### 6.2 里程碑

- [ ] 能画出 GPU 内存金字塔，说出各层级的嵌入式类比；
- [ ] 能说明 Global / Shared / Register 的容量量级、速度关系与可见性；
- [ ] 知道 Local Memory 的定义与检测方法（`-Xptxas -v` 观察 spill）；
- [ ] blocksum 运行 PASS，能解释 `__shared__` 与 `__syncthreads()` 的语义；
- [ ] 完成练习 3，建立"Shared = 片上 SRAM"心智模型。

## 7. 下期预告

内存模型已建立，下一节回到线程组织：GPU 如何将 4608 个核心组织为两层三维结构，`blockIdx.x * blockDim.x + threadIdx.x` 在二维、三维网格中如何推广。

**NPU-04 · 线程组织与索引计算：CUDA 双层三维坐标系统**

> 🏷️ GPU 内存 · Global Memory · Shared Memory · 寄存器 · Local Memory · SRAM 类比
