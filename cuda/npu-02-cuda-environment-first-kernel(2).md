# CUDA 环境搭建与第一个 Kernel：向量加法

> 系列：从单片机到 NPU · NPU-02
> 前置：NPU-01（CUDA 学习路线与硬件概览）
> 配套环境：Linux（Ubuntu 22.04/24.04 或 WSL2）、RTX 5060 Ti、CUDA Toolkit 13.x

## 0. 本节目标

本节完成两件事：**其一**，建立 CUDA 软件栈的分层认知并完成环境搭建；**其二**，编写并运行第一个 kernel——向量加法，验证 4608 个 CUDA 核心的并行执行。

本节不推荐在 Windows 原生环境进行算子开发（工具链与生态支持较弱），建议使用 Linux 或 WSL2。

## 1. CUDA 软件栈的三层结构

环境问题排查困难，多因混淆软件栈层次。CUDA 软件栈自下而上分为三层：

| 层 | 内容 | 嵌入式类比 |
|:---|:---|:---|
| 第 1 层：显卡驱动 | NVIDIA 驱动（R570+） | 芯片寄存器配置 + 总线驱动 |
| 第 2 层：CUDA Toolkit | `nvcc` 编译器 + 运行时库 + 调试工具 | 交叉编译链 + libc |
| 第 3 层：用户代码 | `.cu` 源文件 → 可执行程序 | main.c → .elf |

关键认知：**`nvcc` 是编译器而非驱动**。`nvidia-smi` 报错通常指向驱动层（第 1 层），与 Toolkit（第 2 层）无关。排查时应先定位问题所在层。

【图1：CUDA 软件栈三层结构】

```text
┌──────────────────────────────────────────┐
│  用户程序（vectorAdd）                    │
│    └─ 调用 CUDA Runtime API（cudaMalloc 等）│
├──────────────────────────────────────────┤
│  CUDA Toolkit（nvcc 编译器 + 运行时库）    │
│    └─ 将 .cu 编译为 GPU 可执行指令        │
├──────────────────────────────────────────┤
│  NVIDIA 显卡驱动（内核态）                │
│    └─ 调度指令至 GPU 硬件执行             │
├──────────────────────────────────────────┤
│  GPU 硬件（GB206 / 4608 核心 / 8GB GDDR7）│
└──────────────────────────────────────────┘
```

> 图1 生图 prompt：分层架构图，深色科技风底 #0D1117。四层横向堆叠：最上层绿色方块"用户程序"，第二层青色方块"CUDA Toolkit（nvcc）"，第三层蓝色方块"NVIDIA 驱动"，最底层深灰芯片图形"GPU 硬件（GB206）"。层间以发光竖线连接，右侧标注"代码 → 编译 → 调度 → 执行"。比例 16:9，文字中文。

## 2. 环境搭建流程

以下步骤以 Ubuntu 22.04/24.04 为例。

### 2.1 硬件可见性确认

```bash
lspci | grep -i nvidia
nvidia-smi
```

正常输出应包含 `RTX 5060 Ti` 且 Driver Version ≥ 570。若 `nvidia-smi` 报 "command not found"，说明驱动未安装，执行 2.2 节。

### 2.2 驱动安装

两种方式任选：

```bash
# 方式 A：系统仓库（推荐）
sudo apt update
sudo apt install nvidia-driver-570   # 包名以系统仓库为准
sudo reboot
```

```bash
# 方式 B：NVIDIA 官方 runfile
# 自 developer.nvidia.com/driver 下载对应驱动
sudo sh NVIDIA-Linux-*.run
```

### 2.3 CUDA Toolkit 安装

自 [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) 按系统选择安装方式。runfile 方式示例：

```bash
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_linux.run
sudo sh cuda_13.0.0_linux.run
```

安装界面注意事项：**若驱动已由 2.2 节装好，取消勾选 Driver 项**，仅安装 Toolkit，避免版本冲突。

版本要求：CUDA 13.x。5060 Ti 属 Blackwell 架构（sm_120），**CUDA 12.x 及以下不识别 sm_120**，编译运行时会报 "no kernel image is available"。

### 2.4 环境变量与验证

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version
```

`nvcc --version` 输出 `release 13.x` 即成功。建议将两条 export 写入 `~/.bashrc`。

### 2.5 deviceQuery 硬件体检

Toolkit 附带 deviceQuery 样例，用于打印 GPU 硬件参数：

```bash
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

关键输出（数值以本机实测为准）：

```text
CUDA Capability Major/Minor version number:    12.0        ← sm_120
Total amount of global memory:                 8192 MBytes ← 8GB GDDR7
(36) Multiprocessors, (128) CUDA Cores/MP:     4608 CUDA Cores
GPU Max Clock rate:                            2647 MHz (2.65 GHz)
Memory Bus Width:                              128-bit
Max Threads per Multi Processor:               2048
Max Threads per Block:                         1024
Warp size:                                     32
Shared Memory per Block:                       49152 bytes
Registers per Multiprocessor:                  65536
```

注意核算关系：**36 SM × 128 CUDA Core/SM = 4608**。部分每 SM 细节（寄存器数、共享内存）因架构而异，一律以本机实测为准。

## 3. 第一个 Kernel：向量加法

### 3.1 任务定义

设数组 $a$、$b$ 各含 $n = 2^{20} = 1{,}048{,}576$ 个 float（各 4 MB），求 $c[i] = a[i] + b[i]$。

**嵌入式工程类比**：等价于对 100 万个传感器读数统一加偏置。MCU 实现为单核串行循环；GPU 实现为每线程处理一个元素、数千核心并行。

### 3.2 完整代码

```cuda
// vectorAdd.cu —— 第一个 CUDA 程序：向量加法
#include <cstdio>
#include <cuda_runtime.h>

// kernel：GPU 上执行的函数，每个线程执行一份
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线性索引
    if (i < n) {                                     // 边界保护
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20;               // 1,048,576 个元素
    const size_t bytes = n * sizeof(float);

    // 1. 主机端分配与初始化
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    // 2. 设备端分配显存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 3. 数据搬运：内存 → 显存
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 4. 启动 kernel：每 block 256 线程，共 blocks 个 block
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;   // 向上取整
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();   // 阻塞等待 GPU 完成（类比 DMA 完成中断）

    // 5. 结果搬运回内存并校验
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) { ok = false; break; }
    }
    printf("%s\n", ok ? "PASS: 所有元素 == 3.0" : "FAIL");

    // 6. 释放资源
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    return 0;
}
```

### 3.3 编译与运行

```bash
nvcc -arch=sm_120 -O2 -o vectorAdd vectorAdd.cu
./vectorAdd
```

预期输出：`PASS: 所有元素 == 3.0`。

`-arch=sm_120` 指定按 Blackwell 指令集生成代码，**不可省略**；否则 nvcc 按默认架构编译，在 5060 Ti 上可能无法运行。

### 3.4 关键语法逐项说明

**`__global__` 修饰符**：声明该函数为 kernel 入口，由 GPU 硬件调度执行而非 `main` 直接调用。约束：返回类型必须为 void。**嵌入式工程类比**：中断服务函数（ISR）——同样由硬件触发、以特殊方式执行。

**`<<<blocks, threads>>>` 启动语法**：指定 Grid 含 `blocks` 个 Block、每 Block 含 `threads` 个线程。本程序为 4096 Block × 256 线程 = 1,048,576 线程，正好每线程处理一个元素。**嵌入式工程类比**：配置一次 DMA 传输（源、目的、长度）后由硬件自动执行。

**内置变量**：`blockDim.x` 为每 Block 线程数（=256）；`blockIdx.x` 为当前线程所属 Block 编号（0~4095）；`threadIdx.x` 为当前线程在 Block 内编号（0~255）。

**索引公式** `i = blockIdx.x * blockDim.x + threadIdx.x`：Block 0 覆盖 `i = 0~255`，Block 1 覆盖 `256~511`，依此类推。这是 CUDA 一切索引计算的基础，NPU-04 将扩展至一维到三维。

**边界保护 `if (i < n)`**：因 `blocks` 向上取整，当 n 非 256 整数倍时最后一组线程越界，必须拦截。**嵌入式工程类比**：DMA 搬运长度超出缓冲区时的长度检查。遗漏该保护是新手最常见错误。

**`cudaDeviceSynchronize()`**：kernel 启动为异步操作，CPU 发出指令后即继续执行；该调用使 CPU 阻塞等待 GPU 完成。**嵌入式工程类比**：DMA 启动后 CPU 不阻塞、等待完成中断。

**`cudaMemcpy` 方向参数**：`cudaMemcpyHostToDevice`（内存→显存）、`cudaMemcpyDeviceToHost`（显存→内存）。方向混淆将导致静默错误或直接报错。

【图2：kernel 启动的线程组织（Grid → Block → Thread）】

```text
<<<blocks=4096, threads=256>>> 启动 vectorAdd
┌─────────────────────────────────────────────────────┐
│ Grid（网格）= 4096 个 Block                          │
│  ┌───────┐ ┌───────┐ ┌───────┐          ┌───────┐  │
│  │block 0│ │block 1│ │block 2│   ...    │block  │  │
│  │ 256线程│ │ 256线程│ │ 256线程│          │ 4095  │  │
│  │ i=0.. │ │i=256..│ │i=512..│          │       │  │
│  │  255  │ │  511  │ │  767  │          │       │  │
│  └───────┘ └───────┘ └───────┘          └───────┘  │
│  每线程计算 c[i] = a[i] + b[i]                       │
│  线程编号 i = blockIdx.x × 256 + threadIdx.x        │
└─────────────────────────────────────────────────────┘
```

> 图2 生图 prompt：示意图，白底科技风。大括号框（Grid）内含 4 个小组方块（Block 0/1/2/3），每方块画 8 个蓝色圆点（Thread），下方标注公式 `i = blockIdx.x × blockDim.x + threadIdx.x`，箭头指向数组 `c[i]`。主色蓝 #1565C0，公式红色。比例 16:9，文字中文。

## 4. 错误检查机制

CUDA API 为异步模型，错误以**错误码**形式返回而非异常或硬件异常。生产代码应使用统一检查宏：

```cuda
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)
```

用法：`CUDA_CHECK(cudaMalloc(...))`；kernel 启动后以 `CUDA_CHECK(cudaGetLastError())` 捕获启动错误。宏体以 `do { } while(0)` 包裹系 C 宏最佳实践（防止 if/else 悬挂），与主文章系列所述一致。

常见错误速查：

| 报错信息 | 含义 | 排查方向 |
|:---|:---|:---|
| `invalid argument` | grid/block 维度超限、指针非法 | 检查 n、threads 与指针合法性 |
| `out of memory` | 显存不足 | 检查 cudaFree；控制单次申请量 |
| `unspecified launch failure` | kernel 内越界访问 | 检查索引计算与边界保护 |
| `no kernel image is available` | 编译架构与硬件不匹配 | 加 `-arch=sm_120` 或升级 CUDA 13.x |

## 5. 练习与里程碑

### 5.1 练习

1. **规模扩展**：将 `n` 由 `1 << 20` 改为 `1 << 24`（约 1677 万元素），重新编译运行。结果仍应 PASS。体会：并行规模扩大 16 倍，kernel 代码零改动。
2. **线程数扫描**：将 `threads` 依次改为 32、512、1024，分别运行。结果均应 PASS。思考：为何 Block 大小不影响正确性（提示：索引公式的线性映射）。
3. **边界保护实验**：删除 `if (i < n)`，将 `n` 改为 1,000,000（非 256 整数倍），运行观察 FAIL 或崩溃。验证边界保护的必要性。
4. **进阶（可选）**：以 CUDA Event（`cudaEvent`）为练习 1 计时，对比 `n=1<<20` 与 `n=1<<24` 的耗时，建立 GPU 吞吐量的初步感性认识（计时 API 将于 NPU-07 正式讲授）。

### 5.2 里程碑

- [ ] 能说清 CUDA 软件栈三层结构，并能定位报错所属层级；
- [ ] `nvidia-smi` 可见 5060 Ti，`nvcc --version` 显示 13.x；
- [ ] deviceQuery 输出 36 SM × 128 = 4608，sm_120；
- [ ] 能手写 vectorAdd，解释 `<<<>>>`、`blockIdx/blockDim/threadIdx`、`if(i<n)` 的语义；
- [ ] 会使用 CUDA_CHECK 宏进行错误检查。

## 6. 下期预告

kernel 已能运行，下一节转入存储层次：`c[i]` 存放于何处？GPU 存在哪些内存区域？为何 Shared Memory 被视为性能关键？

**NPU-03 · GPU 内存层次：Global / Shared / Register 与嵌入式存储直觉**

> 🏷️ CUDA 13 · sm_120 · 环境搭建 · 向量加法 · kernel 启动 · 线程模型
