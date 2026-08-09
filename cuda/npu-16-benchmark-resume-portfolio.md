# 性能报告与简历项目：把算子库写成可展示的作品集

> 系列：CUDA 高性能算子实战 · NPU-16
> 前置：NPU-15（纯 CUDA 推理算子库）
> 配套环境：RTX 5060 Ti、CUDA Toolkit 13.x、NVIDIA Nsight Compute（ncu）、Python 3.10+

## 0. 本节目标

算子库写完了，但"能跑"和"能拿得出手"之间还差两样东西：

1. **一份可信的性能报告**——用科学的方法测出优化前后数据，用 Roofline 讲清楚"为什么快"，让读者（或面试官）一眼看到你的分析能力；
2. **一段能打动人的简历描述**——把项目翻译成"技术栈 + 成果数字 + 方法论"的语言。

本节是系列的收尾篇，也是"作品集化"的最后一公里：**方法论（怎么测）、报告（怎么呈现）、简历（怎么表达）、面试（怎么答）** 四件事一次讲透。

## 1. Benchmark 方法论：测出可信的数据

### 1.1 五个常见错误

| 错误 | 后果 | 正确做法 |
|:---|:---|:---|
| 只跑一次 | 噪声淹没结果 | warmup + 多次取中位数 |
| 不 warmup | 首次调用含初始化开销 | 先跑 5~10 次热身 |
| 用 CPU 计时 | 测不准 GPU 异步执行 | 用 `cudaEvent` / ncu |
| 不同步 | 计时提前结束 | `cudaEventSynchronize` |
| 编译器优化掉计算 | 结果假快 | 使用输出（如累加校验） |

### 1.2 正确计时模板

```cuda
// 正确 benchmark：warmup + 100 次 + 中位数
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>

void benchmark(void (*run)(), int warmup = 10, int reps = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; i++) run();       // 热身：加载/初始化

    std::vector<float> times(reps);
    for (int i = 0; i < reps; i++) {
        cudaEventRecord(start);
        run();                                     // GPU 异步执行
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);                // 必须同步
        cudaEventElapsedTime(&times[i], start, stop);
    }
    std::sort(times.begin(), times.end());
    printf("中位数: %.3f ms   最小: %.3f ms   最大: %.3f ms\n",
           times[reps / 2], times.front(), times.back());
}
```

**为什么取中位数**：最大值可能被时钟降频/系统干扰污染，最小值可能是运气好；中位数代表"典型性能"，是报告的标准口径。

### 1.3 更专业的验证：ncu 指标

`cudaEvent` 测的是端到端时间；要回答"时间花在哪"，用 Nsight Compute 看硬件利用率（性能分析篇的方法）：

```bash
ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed ./bench_conv
```

- `sm__throughput` 高 + 访存低 → 计算受限（用 Roofline 解释：AI 高）；
- `sm__throughput` 低 + 访存高 → 带宽受限（AI 低，优化融合/减少搬运）。

**报告里同时给"时间数据"和"利用率数据"**，说服力完全不同——前者说明结果，后者说明你懂原因。

## 2. 性能报告的结构

一份专业的算子性能报告，建议按以下模板组织（这也是很多公司技术报告的通用结构）：

```text
1. 概述         项目目标、模型、核心结论（3 句话讲完）
2. 环境         硬件（5060 Ti 参数）、软件（CUDA 13.x / nvcc / OS）、编译选项
3. 方法         benchmark 流程（warmup/次数/取中位数）、工具（cudaEvent/ncu）
4. 结果         优化前后性能表格 + 加速比图
5. 分析         Roofline 定位瓶颈、各优化步骤贡献拆解
6. 结论与展望   当前状态、可继续优化的方向（量化/Tensor Core/更大模型）
```

### 2.1 结果表格示例

| 实现 | 平均耗时 (ms) | 加速比 | sm 利用率 | 访存利用率 |
|:---|:---:|:---:|:---:|:---:|
| 朴素卷积 | 0.52 | 1.0× | 8% | 35% |
| im2col + GEMM（16×16 tile） | 0.11 | 4.7× | 42% | 61% |
| 融合 Conv+ReLU（省一次写读） | 0.09 | 5.8× | 47% | 58% |

**注意每个数字都要能从你的实测复现**——报告里标注"RTX 5060 Ti，448 GB/s，数据为 100 次中位数"。

### 2.2 加速比图（可运行代码）

```python
import matplotlib.pyplot as plt

impls   = ['naive', 'im2col+GEMM', 'fused+ReLU']
ms      = [0.52, 0.11, 0.09]
speedup = [ms[0] / m for m in ms]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(impls, speedup, color=['#888888', '#2E86AB', '#E53935'])
for b, s in zip(bars, speedup):
    plt.text(b.get_x() + b.get_width()/2, s + 0.1, f'{s:.1f}×',
             ha='center', fontsize=12)
plt.ylabel('相对朴素实现的加速比')
plt.title('卷积优化前后加速比（RTX 5060 Ti, 448 GB/s）')
plt.ylim(0, max(speedup) * 1.3)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('conv_speedup.png', dpi=150)
```

### 2.3 分析段示例（Roofline 视角）

```text
朴素卷积：重复读取相邻窗口的输入像素，访存效率低，实测 sm 利用率仅 8%，
说明计算单元大量闲置等待数据，属于带宽受限（AI ≈ 0.5 FLOP/B，低于拐点 53.6）。

im2col + GEMM：展开后复用 tiled GEMM 的共享内存分块，数据复用率提升，
sm 利用率升至 42%；代价是展开内存为输入 9 倍（3×3 核），模型变大时需权衡。

融合 Conv+ReLU：ReLU 不再单独读写一次特征图，省掉约 2×C_out×H_out×W_out×4B
的访存，实测进一步提速 ~20%。
```

这段分析展示了三件事：**会测（数据）、会定位（Roofline）、会解释（为什么）**——这正是算子开发岗位要的能力。

## 3. 简历项目描述

### 3.1 一个可复用的写法模板

```text
项目名称：纯 CUDA 端侧推理算子库
技术栈：CUDA C++ / CMake / PyTorch / Nsight Compute

项目描述：
- 手写实现 Conv / GEMM / ReLU / MaxPool / Softmax 等算子（不依赖 cuDNN/cuBLAS），
  完成 MNIST CNN 端到端推理，1000 张测试图与 PyTorch 输出 100% 一致；
- 实现 im2col + tiled GEMM 卷积方案，相对朴素实现加速约 5×（RTX 5060 Ti 实测）；
- 基于 Roofline 模型定位访存瓶颈，完成 Conv+ReLU 算子融合，省去中间特征图读写，
  延迟进一步降低约 20%；
- 建立 CMake 工程与自动化正确性测试，使用 Nsight Compute 量化利用率指标；
- 预留 INT8 对称量化接口，支持扩展为定点推理。
```

**要点拆解**：

| 要素 | 写法 | 作用 |
|:---|:---|:---|
| 技术栈 | 明确列出 CUDA/CMake/PyTorch/ncu | 过关键词筛选 |
| 成果数字 | "加速约 5×""100% 一致""降低 20%" | 可量化才有说服力 |
| 方法论 | Roofline、算子融合、tiled GEMM | 展示工程思维 |
| 工程化 | CMake、自动化测试 | 展示不是 toy demo |

### 3.2 面试高频问题与回答思路

| 问题 | 回答要点 |
|:---|:---|
| 为什么不用 cuBLAS/cuDNN 而要手写？ | 性能特化（融合/布局）、新算子无现成实现、理解原理才能做编译器/算子开发 |
| GEMM 为什么 tiled 后快？ | 数据复用：一次加载进共享内存，被多个线程多次使用，减少全局内存往返（带宽受限 → 计算受限） |
| 怎么判断瓶颈？ | Roofline：算算术强度 vs 拐点；ncu：sm 利用率 vs 访存利用率 |
| 算子融合省了什么？ | 中间张量的整次写+读，本质是减少访存 |
| 量化怎么做？ | 对称量化 scale、非对称 zero_point；INT8 乘累加 + 反量化；校准集统计数值范围 |
| 接下来怎么继续优化？ | 隐式 GEMM（免展开内存）、Tensor Core（FP16/INT8）、更大 tile + 向量化、跨算子融合 |

**答法原则**：先给结论，再给数据/公式支撑，最后说"我在项目里怎么做的"——面试官要的不是背概念，是"你用过、你理解、你能迁移"。

## 4. 作品集检查清单

发布/展示前，逐项确认：

- [ ] 代码可一键构建（CMake），README 写明环境与运行步骤
- [ ] 正确性测试自动运行并输出 PASS/FAIL
- [ ] 性能数据用规范方法测得（warmup + 中位数），可复现
- [ ] 报告含环境、方法、结果表格、Roofline 分析
- [ ] 简历描述含技术栈、量化成果、方法论关键词
- [ ] 能 3 分钟讲清项目（背景 → 做法 → 结果 → 启发）

## 5. 练习与里程碑

### 练习

1. **写报告**：按 2 节模板，为你的算子库写一份完整性能报告（Markdown），附加速比图。
2. **写 README**：给 `mnist_infer` 写一个 README：环境、构建、运行、结果表、截图。
3. **模拟面试**：把 3.2 的问题逐一写成书面答案（各 100~200 字），找人模拟提问。
4. **延伸思考**：如果给你一个更大的模型（如 ResNet18），算子库要加哪些算子、哪些优化优先级最高？（提示：更大通道数 → 卷积用隐式 GEMM；层多 → 融合收益更大；batch 推理 → 提高占用率）

### 里程碑自检

- [ ] 能按规范方法测出可复现的性能数据
- [ ] 能用 Roofline + ncu 解释优化前后的变化
- [ ] 简历描述含 3 个以上量化成果数字
- [ ] 能流畅回答 6 个面试高频问题
- [ ] 作品集四件套齐全：代码库 / 测试 / 性能报告 / 简历描述

## 6. 小结

本节把算子库变成了可展示的作品集：

- **测**：warmup + 中位数 + ncu 验证，数据可信；
- **写**：环境/方法/结果/分析四段式报告，Roofline 解释"为什么快"；
- **说**：简历用"技术栈 + 数字 + 方法论"表达，面试按"结论 + 依据 + 实践"作答。

到这里，本系列从一条完整的路径走完了：**并行基础 → 算子优化 → 框架实战 → 方法论与编译器 → 综合项目**。你从"会写 CUDA 的嵌入式工程师"，成长为一个"能分析、能优化、能交付、能展示"的算子开发者。这条路的关键不是记住多少 API，而是建立了一套可以迁移的思维：**先定位瓶颈（Roofline），再选择手段（tiling/融合/量化），最后用数据证明（benchmark）**——这套方法论，无论以后做 GPU、NPU 还是任何 AI 加速器，都永远适用。

> 🏷️ 标签：#性能报告 #Benchmark #简历项目 #作品集 #面试准备 #Roofline
