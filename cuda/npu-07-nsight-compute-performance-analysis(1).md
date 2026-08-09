# Nsight Compute 性能分析：从指标到优化闭环

> 系列：从单片机到 NPU · NPU-07
> 前置：NPU-05（矩阵乘法 naive→tiled）、NPU-06（并行归约与 Warp Shuffle）
> 配套环境：RTX 5060 Ti（sm_120）、CUDA Toolkit 13.x、NVIDIA Nsight Compute（ncu）

## 0. 本节目标

前两节完成了矩阵乘法 tiling 与并行归约两个 kernel，但"快了 2.9 倍"只是现象，**为什么快、瓶颈在哪、还能快多少**——这些问题不能靠猜测回答，必须由性能剖析工具给出量化证据。

本节引入 CUDA 性能分析的标准工具 **Nsight Compute（ncu）**，完成一个可复用的优化闭环：
1. 建立"优化必须以测量为前提"的方法论；
2. 掌握 ncu 的基本用法与关键指标语义；
3. 对 05/06 两节的 kernel 做量化诊断，确认瓶颈类型；
4. 给出 Bank Conflict 消除、向量化加载、双缓冲等进阶优化的实测路径。

本节的目标不是罗列工具按钮，而是建立**瓶颈定位的三步法**：确认瓶颈维度 → 定位具体原因 → 实施针对性优化。

## 1. 方法论的先决条件：以测量代替猜测

**嵌入式工程类比**：嵌入式工程师排查时序问题不会靠"感觉程序变快了"，而是用示波器/逻辑分析仪测量信号沿、用性能计数器读取总线占用。**GPU 优化同理**——ncu 就是 GPU 的"示波器"，提供硬件性能计数器的统一视图。

两个常见的错误做法：
1. **凭经验猜测瓶颈**：优化方向与真实瓶颈不符，投入产出比低；
2. **只用 `cudaEvent` 计时**：总耗时只能说明"快慢"，无法回答"为什么"。tiled kernel 耗时 1.1 ms，是计算受限还是访存受限？延迟受限？只有计数器能回答。

工程原则：**先测量，后优化；一次只改一个变量，用数据验证收益**。

## 2. Nsight Compute 的基本使用

### 2.1 工具定位

Nsight Compute 是 NVIDIA 提供的 kernel 级性能剖析器，逐 kernel 报告硬件计数器。它回答的问题包括：
- kernel 实际达到多少算力/带宽利用率；
- 每个 SM 的占用率（Occupancy）是多少；
- warp 在等待什么（访存？同步？计算依赖？）；
- 共享内存是否存在 Bank Conflict。

### 2.2 命令行基本用法

```bash
# 对可执行文件整体剖析（默认输出关键指标摘要）
ncu ./matmul

# 限定剖析指定 kernel，输出全部指标（耗时较长）
ncu --kernel-name matmulNaive --set full ./matmul

# 只取关键 Section，速度更快
ncu --kernel-name matmulTiled \
    --section SpeedOfLight \
    --section Occupancy \
    --section WarpStateStats \
    --section MemoryWorkloadAnalysis ./matmul
```

常用 Section 与对应问题：

| Section | 回答的问题 |
|:---|:---|
| `SpeedOfLight` | kernel 距离理论峰值有多远？计算还是访存受限？ |
| `Occupancy` | SM 上同时驻留的 warp 数与理论最大值的比例？ |
| `WarpStateStats` | warp 因何停顿（访存/同步/依赖）？ |
| `MemoryWorkloadAnalysis` | 全局/共享内存吞吐、Bank Conflict 计数？ |
| `SchedulerStats` | 指令发射效率、warp 调度器利用率？ |

> 注：ncu 默认会重放（replay）kernel 以收集计数器，实际运行时间显著变长属正常现象。对每次 kernel 仅执行一次的程序，建议先以普通方式确认正确性，再单独剖析。

## 3. 关键指标的形式化语义

### 3.1 光速指标（Speed of Light）

**定义 1（SOL 指标）**：ncu 将 kernel 实际利用率与硬件理论峰值之比呈现为两个百分比：

- **Compute（SM）Throughput**：SM 执行单元（FP32/INT/特殊函数）的利用率，高值表示接近计算峰值；
- **Memory Throughput**：全局内存/共享内存/L2 等存储层级带宽的利用率，高值表示接近访存峰值。

**瓶颈判定**：两者中较高者决定 kernel 的瓶颈维度——"计算受限"或"访存受限"。若两者均不高（如均 < 50%），则属于**延迟受限**（warp 数量不足，无法隐藏访存延迟）。

### 3.2 Occupancy

**定义 2（Occupancy）**：单个 SM 上实际驻留的 warp 数与硬件可容纳最大 warp 数之比。

$$\text{Occupancy} = \frac{\text{ActiveWarpsPerSM}}{\text{MaxWarpsPerSM}} \qquad (式 3-1)$$

**为什么重要**：GPU 依赖大量可切换的 warp 隐藏访存延迟（数百周期）。Occupancy 过低时，访存等待期间无其他 warp 可执行，SM 空转——即延迟受限。限制 Occupancy 的资源包括寄存器用量、共享内存用量与 Block 线程数（三者构成"资源预算"，见 NPU-03/04）。

**工程判断**：并非越高越好。高 Occupancy 若导致每个线程可用寄存器过少、发生 Local Memory 溢出（NPU-03），反而有害。正确用法是与 SOL 指标联合解读。

### 3.3 Warp Stall 原因

**定义 3（Stall）**：warp 因等待资源而暂停执行的原因统计。常见 stall 原因：

| Stall 原因 | 含义 | 典型对策 |
|:---|:---|:---|
| `Long Scoreboard` | 等待全局/本地内存返回 | 访存合并、提高 Occupancy、tiling |
| `Short Scoreboard` | 等待共享内存返回 | 消除 Bank Conflict |
| `Barrier` | 等待 `__syncthreads()` 对齐 | 减少同步频率、避免过度同步 |
| `Wait` | 等待固定延迟单元（如 MIO） | 指令调度优化、减少非计算指令 |
| `Not Selected` | warp 就绪但调度器选择其他 warp | 高 Occupancy 下的正常现象 |

**嵌入式工程类比**：等价于 Cortex-M 流水线停顿分析——`Long Scoreboard` 类似等待外部存储器总线，`Barrier` 类似等待 DMA 完成中断。定位 stall 原因即定位性能瓶颈的直接证据。

### 3.4 本系列的环境基线

5060 Ti（sm_120）相关理论值（以 deviceQuery 实测为准）：
- FP32 峰值 ≈ 24 TFLOPS；
- 全局内存带宽 ≈ 448 GB/s；
- 每 SM 最大驻留线程 2048（64 warp），每 SM 最大 Block 数 32。

后续所有诊断结论均与该基线对照。

## 4. 实战诊断：matmul naive vs tiled

### 4.1 诊断流程

对 NPU-05 的 `matmul_naive_vs_tiled.cu`（M=N=K=512）执行：

```bash
nvcc -arch=sm_120 -O2 -o matmul matmul_naive_vs_tiled.cu
ncu --kernel-name matmulNaive --set full ./matmul
ncu --kernel-name matmulTiled  --set full ./matmul
```

### 4.2 典型读数与解读（以 5060 Ti 为例，数值以实测为准）

| 指标 | matmulNaive | matmulTiled | 结论 |
|:---|:---:|:---:|:---|
| Compute (SM) Throughput | ≈ 8% | ≈ 25% | 两者均非计算受限 |
| Memory Throughput | ≈ 85% | ≈ 55% | naive 访存受限 |
| Achieved Occupancy | ≈ 95% | ≈ 90% | 均健康，非延迟受限 |
| Warp Stall（主要） | Long Scoreboard | Short Scoreboard / Wait | 见 4.3 |

**核心结论**：
1. naive 的 Memory Throughput 接近峰值而 Compute 极低 → **访存受限**，与 NPU-05 式 3-3 的算术强度分析（AI=0.25）相互印证；
2. tiled 的 Memory Throughput 显著下降、Compute 上升 → 数据复用生效，瓶颈向计算侧移动；
3. 两者 Occupancy 均健康，**排除了延迟受限**，优化方向锁定在访存路径。

### 4.3 Warp Stall 的进一步定位

naive 的 `Long Scoreboard` 占比高：warp 等待全局内存返回。对应 B 的跨行访问 `B[t*N+col]`（NPU-05 第 2.2 节）——每次仅取 4 字节/32 字节缓存行。

tiled 的 stall 若出现 `Short Scoreboard` 占比较高，则提示**共享内存访问存在 Bank Conflict**（下一节验证）。这是优化进入"精细阶段"的标志：宏观访存问题已解决，剩下的瓶颈在片上存储的访问模式。

【图1：瓶颈定位三步法】

```text
第一步：SOL 指标 → 计算受限 / 访存受限 / 延迟受限？
   │
   ├─ Memory 高、Compute 低 → 访存受限 → 第二步 A
   ├─ Compute 高、Memory 低 → 计算受限 → 优化指令效率
   └─ 两者均低 → 延迟受限 → 提高 Occupancy

第二步 A：访存受限 → MemoryWorkloadAnalysis
   ├─ 全局内存吞吐高 + Long Scoreboard → 访存合并 / tiling / 减少重复读取
   └─ 共享内存吞吐高 + Short Scoreboard → Bank Conflict → 第三步

第三步：Bank Conflict → 调整共享内存布局 / 填充（padding）/ 换访问顺序
```

> 图1 生图 prompt：决策流程图，白底。三个菱形判断框串联，分支用红/绿箭头：Memory 高 → "访存受限"分支，Compute 高 → "计算受限"分支，两者均低 → "延迟受限"分支；每分支末端列出优化手段文字。主色蓝 #1565C0，判断框橙色。比例 16:9，文字中文。

## 5. 进阶优化：三个可实测的手段

### 5.1 消除 Bank Conflict

**背景**：NPU-06 定义 3 指出，同一 warp 同一周期访问同一 Bank 的不同地址会串行化。对 5.2 节归约 kernel，若第一段使用共享内存交错寻址（NPU-06 3.2 节），ncu 的 `MemoryWorkloadAnalysis` 会报告冲突计数。

**验证方法**：

```bash
ncu --kernel-name reduceKernel --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./reduce
```

**对策**：换顺序寻址（NPU-06 3.4 节）；对矩阵分块，可在共享内存数组加 1 元素填充（padding）使相邻行错开 Bank：

```cuda
__shared__ float As[TILE][TILE + 1];   // 每行末尾补 1 个 float
```

填充使 `As[y][x]` 的地址偏移变为 `y*(TILE+1)+x`，相邻行起点 Bank 错开 1，避免同一时刻多行同列元素命中同一 Bank。**代价**：共享内存增加约 3%（TILE=32 时 4 KB → 4.125 KB），换取访存去串行化，通常净收益为正。

**嵌入式工程类比**：等价于在 DMA 描述符链表中插入对齐填充，使各描述符均匀分布到不同存储体/通道，避免总线竞争。

### 5.2 向量化加载（float4）

**背景**：全局内存访存合并以 128 字节为粒度。每线程加载 1 个 float（4 字节）时，一个 warp 恰好覆盖 32×4=128 字节——已属合并访问。但**指令数量**仍可优化：若每线程以 `float4` 一次加载 4 个连续元素，指令数降为 1/4，LSU（Load/Store Unit）压力与指令发射开销同时下降。

```cuda
// 每线程处理 4 个连续元素：将数据视为 float4 数组
const float4 *a4 = reinterpret_cast<const float4*>(A);
const float4 *b4 = reinterpret_cast<const float4*>(B);
float4 va = a4[i];          // 一次 16 字节加载
float4 vb = b4[i];
float4 vc;
vc.x = va.x + vb.x;
vc.y = va.y + vb.y;
vc.z = va.z + vb.z;
vc.w = va.w + vb.w;
```

**前置条件**：数据长度须为 4 的倍数，且指针 16 字节对齐（`cudaMalloc` 默认满足 256 字节对齐）。长度不足时对尾部单独处理。

**嵌入式工程类比**：等价于 DMA 突发传输（burst）——一次请求搬运 4 个字而非 1 个字，减少总线事务次数。

### 5.3 双缓冲（double buffering）

**背景**：tiled kernel 中，每轮归约循环"加载子块 → 同步 → 计算 → 同步"，加载与计算串行。双缓冲让**下一轮的子块加载与当前轮的计算重叠**：

```cuda
// 伪代码结构：A 子块使用两个缓冲区，交替填充/消费
__shared__ float As[2][TILE][TILE];
int cur = 0;
// 预加载第 0 轮
loadTile(As[cur], ...);
__syncthreads();
for (int t = 0; t < K; t += TILE) {
    int nxt = cur ^ 1;
    if (t + TILE < K) loadTile(As[nxt], ...);   // 异步预取下一块
    computeTile(As[cur], ...);                   // 计算当前块
    __syncthreads();                             // 确保计算完成再切换
    cur = nxt;
}
```

**收益**：访存延迟被计算掩盖，Memory Throughput 与 Compute Throughput 可同时保持高位（接近"计算访存重叠"的理想状态）。ncu 的 SOL 指标中两列同时升高即为生效证据。

**嵌入式工程类比**：等价于 DMA 双缓冲（ping-pong）——搬运与处理重叠，消除"等数据"的空闲时间。这与 STM32 外设 DMA 双缓冲的工程实践完全同构。

### 5.4 优化优先级建议

按"投入产出比"排序，先解决宏观问题，再处理微观问题：
1. 确认瓶颈维度（SOL）→ 访存受限则 tiling/合并/复用（NPU-05）；
2. 消除 Bank Conflict（5.1）；
3. 向量化加载减少指令数（5.2）；
4. 双缓冲重叠访存与计算（5.3）；
5. 寄存器分块（register blocking，NPU-13 矩阵乘再展开）。

## 6. 练习与里程碑

### 6.1 练习

1. **基线剖析**：对 NPU-05 的 naive 与 tiled 两版 matmul 运行 `ncu --set full`，记录 SOL、Occupancy、主要 Stall 三项，填入实验日志；
2. **验证填充**：给 tiled matmul 的 `As`/`Bs` 加 padding（`TILE+1`），用 ncu 对比 Bank Conflict 计数与耗时变化；
3. **向量化实验**：将 vectorAdd（NPU-02）改写为 float4 版本（n 为 4 的倍数），对比耗时与 SOL 指标；
4. **归约诊断**：对 NPU-06 的 reduceKernel 运行 ncu，确认其 Stall 主因（预期为 `Short Scoreboard` 或 `Barrier`），并分析原因；
5. **思考题**：为什么说"Occupancy 高≠性能好"？结合资源预算（寄存器/共享内存）分析一个反例场景。

### 6.2 里程碑

- [ ] 能运行 ncu 并导出 SOL、Occupancy、WarpStateStats、MemoryWorkloadAnalysis 四个 Section；
- [ ] 能根据 SOL 双指标判定 kernel 是计算受限、访存受限还是延迟受限；
- [ ] 能解释 Occupancy 与延迟隐藏的关系，以及其与资源预算的权衡；
- [ ] 能读懂 Warp Stall 原因表并定位主要瓶颈；
- [ ] 能实施 Bank Conflict 消除（含 padding）、float4 向量化、双缓冲三种优化，并用 ncu 验证收益。

## 7. 下期预告

至此，阶段二（算子优化基本功）收束：tiling 解决宏观访存、归约与 Shuffle 解决聚合、ncu 提供诊断闭环。从阶段三起，我们将把具体优化经验**抽象为跨硬件通用的理论模型**：NPU-08 引入 **Roofline 模型**——用一张图回答"这个 kernel 在这块硬件上还能快多少"，为后续 GPU→NPU 的思维迁移奠定分析框架。

**NPU-08 · Roofline 性能模型：判断 kernel 瓶颈的通用理论**

> 🏷️ Nsight Compute · ncu · Occupancy · Warp Stall · Bank Conflict · 双缓冲 · 性能剖析
