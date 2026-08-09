# 综合项目：纯 CUDA 手写端侧推理算子库

> 系列：CUDA 高性能算子实战 · NPU-15
> 前置：NPU-05（GEMM）、NPU-09（PyTorch 训练导出）、NPU-13（卷积与融合）
> 配套环境：RTX 5060 Ti、CUDA Toolkit 13.x、CMake ≥ 3.18、Python 3.10+（torch 用于导出权重）

## 0. 本节目标

系列前 14 节分别练了并行基础、算子优化、框架实战、方法论与编译器视野。本节把它们**串成一个完整的作品**：一个纯 CUDA 手写的端侧推理算子库，具备以下特征：

1. **纯 CUDA**：Conv、GEMM、ReLU、MaxPool、融合全部手写，不依赖 cuDNN/cuBLAS；
2. **真实负载**：推理我们亲手训练导出的 MNIST CNN（框架实战篇的模型）；
3. **端到端验证**：C++ 推理结果与 PyTorch 输出逐元素对比，精度一致才算通过；
4. **工程化**：CMake 构建、测试程序、性能对比（naive vs 优化）、量化接口预留。

完成本节，你就拥有一个可以写进简历、可以继续扩展（加算子、加量化、换模型）的算子库骨架。

## 1. 项目目标与验收标准

### 1.1 目标模型

复用前面 PyTorch 实战篇训练好的 MNIST CNN（在 PyTorch 里重新训练只需几分钟）：

```
输入 (1,28,28)
Conv1: (1→32, 3×3, pad=1) + ReLU + MaxPool(2)   → (32,14,14)
Conv2: (32→64, 3×3, pad=1) + ReLU + MaxPool(2)   → (64,7,7)
Flatten → (3136)
FC1: Linear(3136→128) + ReLU
FC2: Linear(128→10) → Softmax → 类别
```

### 1.2 验收标准

| 项 | 标准 |
|:---|:---|
| 正确性 | 1000 张测试图，推理结果与 PyTorch 的 argmax 完全一致（或 ≥ 99.9% 一致） |
| 性能 | 优化版卷积比 naive 版快（记录加速比）；单张推理延迟 < 1 ms |
| 工程 | `cmake .. && make` 一键构建；`./test_accuracy` 输出 PASS/FAIL |

## 2. 项目结构

```text
mnist_infer/
├── CMakeLists.txt
├── include/
│   └── infer_ops.h              # 算子接口声明
├── src/
│   ├── gemm.cu                  # tiled sgemm
│   ├── conv.cu                  # 朴素卷积 + im2col+GEMM 卷积 + 融合 ReLU
│   ├── ops.cu                   # ReLU / MaxPool / Softmax
│   └── inference.cpp            # 推理主循环（加载权重 → 前向 → 输出）
├── tools/
│   └── export_weights.py        # 从 PyTorch 导出权重/输入/期望输出
├── tests/
│   └── test_accuracy.cu         # 与 PyTorch 输出对比
└── bench/
    └── bench_conv.cu            # naive vs 优化卷积性能对比
```

【图1：算子库推理数据流】

```text
 input(1,28,28)
    │
    ▼ conv2d (1→32, 3×3, pad=1)   ── 手写 kernel
    ▼ relu (融合进 conv epilogue)
    ▼ maxpool(2)                    → (32,14,14)
    │
    ▼ conv2d (32→64, 3×3, pad=1)  ── 手写 kernel
    ▼ relu
    ▼ maxpool(2)                    → (64,7,7)
    │
    ▼ flatten → (3136)
    ▼ gemm (3136×128) + relu      ── tiled sgemm
    ▼ gemm (128×10)
    ▼ softmax → argmax → 类别
```

## 3. 权重导出（PyTorch → 二进制）

用 PyTorch 实战篇的模型定义训练/加载权重，然后按固定顺序把张量导出为二进制文件，C++ 侧按同样顺序读取。

```python
# tools/export_weights.py
import struct, torch
from model import CNN          # PyTorch 实战篇定义的 CNN 类

model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth', weights_only=True))
model.eval()

def save_tensor(f, t):
    t = t.detach().cpu().float().numpy()
    f.write(struct.pack('I', t.size))          # 元素个数
    f.write(t.astype('<f4').tobytes())         # float32 小端

# 顺序与 C++ 读取顺序一致：conv1.w, conv1.b, conv2.w, conv2.b, fc1.w, fc1.b, fc2.w, fc2.b
with open('weights.bin', 'wb') as f:
    save_tensor(f, model.features[0].weight)   # (32,1,3,3)
    save_tensor(f, model.features[0].bias)     # (32,)
    save_tensor(f, model.features[3].weight)   # (64,32,3,3)
    save_tensor(f, model.features[3].bias)     # (64,)
    save_tensor(f, model.classifier[1].weight) # (128,3136)
    save_tensor(f, model.classifier[1].bias)   # (128,)
    save_tensor(f, model.classifier[3].weight) # (10,128)
    save_tensor(f, model.classifier[3].bias)   # (10,)

# 同时导出 1000 张测试图输入与 PyTorch 期望输出（供 C++ 对比）
import torchvision, torchvision.transforms as T
ds = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,),(0.3081,))]))
save_tensor(f2, torch.stack([ds[i][0] for i in range(1000)]))      # input.bin
with torch.no_grad():
    logits = model(torch.stack([ds[i][0] for i in range(1000)]))
save_tensor(f3, logits.argmax(1))                                   # expected.bin
```

## 4. 算子实现

### 4.1 接口头文件

```cpp
// include/infer_ops.h
#pragma once
#include <cstdint>
#include <vector>

// 4 维 NCHW 张量（batch=1 简化）
struct Tensor {
    int n, c, h, w;
    std::vector<float> data;
    Tensor(int n_, int c_, int h_, int w_)
        : n(n_), c(c_), h(h_), w(w_), data(n_ * c_ * h_ * w_) {}
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
    size_t size() const { return data.size(); }
};

// 算子：全部纯 CUDA 实现
void gemm(const float* A, const float* B, float* C, int M, int N, int K);
void conv2d_naive(const float* in, const float* w, const float* b, float* out,
                  int C_in, int H, int W, int C_out, int K, int pad, int stride, bool relu);
void conv2d_im2col_gemm(const float* in, const float* w, const float* b, float* out,
                        int C_in, int H, int W, int C_out, int K, int pad, int stride);
void relu_inplace(float* x, int size);
void maxpool2d(const float* in, float* out, int C, int H, int W, int pool);
void softmax(const float* in, float* out, int n_classes);
```

### 4.2 GEMM：tiled sgemm（16×16 共享内存）

```cuda
// src/gemm.cu —— 16×16 tile + 共享内存（简化版，无边界填充但带边界判断）
__global__ void sgemm_16x16(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += 16) {
        // 每个线程加载一个元素到共享内存
        As[threadIdx.y][threadIdx.x] =
            (row < M && k0 + threadIdx.x < K)
                ? A[row * K + k0 + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (k0 + threadIdx.y < K && col < N)
                ? B[(k0 + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < 16; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = acc;
}

void gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    sgemm_16x16<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

这就是你在矩阵乘法篇写的 tiled 思路的精简版：共享内存复用 K 维 tile，`__syncthreads()` 保证 tile 加载完再算。

### 4.3 卷积：朴素版（融合 ReLU）+ im2col+GEMM 版

朴素卷积 kernel（融合 ReLU，见算子融合篇）：

```cuda
// src/conv.cu —— 朴素卷积 + 可选 ReLU 融合（epilogue 一行实现）
__global__ void conv2d_naive_kernel(
    const float* in, const float* w, const float* b, float* out,
    int C_in, int H, int W, int C_out, int K, int pad, int stride, bool relu)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;
    if (oh >= H_out || ow >= W_out) return;

    float acc = (b ? b[oc] : 0.0f);
    for (int ic = 0; ic < C_in; ic++)
        for (int kh = 0; kh < K; kh++)
            for (int kw = 0; kw < K; kw++) {
                int ih = oh * stride - pad + kh;
                int iw = ow * stride - pad + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    acc += in[ic * H * W + ih * W + iw]
                         * w[oc * C_in * K * K + ic * K * K + kh * K + kw];
            }
    if (relu) acc = fmaxf(acc, 0.0f);       // 融合 ReLU
    out[oc * H_out * W_out + oh * W_out + ow] = acc;
}

void conv2d_naive(const float* in, const float* w, const float* b, float* out,
                  int C_in, int H, int W, int C_out, int K, int pad, int stride, bool relu)
{
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;
    dim3 block(16, 16);
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, C_out);
    conv2d_naive_kernel<<<grid, block>>>(in, w, b, out,
        C_in, H, W, C_out, K, pad, stride, relu);
    cudaDeviceSynchronize();
}
```

im2col+GEMM 版：在 host 端展开列矩阵（小模型内存可接受），再调用 `gemm()`：

```cuda
// src/conv.cu —— im2col + GEMM 卷积
void conv2d_im2col_gemm(const float* in, const float* w, const float* b, float* out,
                        int C_in, int H, int W, int C_out, int K, int pad, int stride)
{
    int H_out = (H + 2 * pad - K) / stride + 1;
    int W_out = (W + 2 * pad - K) / stride + 1;
    int rows = C_in * K * K;               // 列矩阵行数
    int cols = H_out * W_out;              // 列矩阵列数

    // host 展开（小模型：rows×cols ≤ 576×196，约 450KB，可接受）
    std::vector<float> col(rows * cols, 0.0f);
    for (int c = 0; c < rows; c++) {
        int ic = c / (K * K), kh = (c / K) % K, kw = c % K;
        for (int p = 0; p < cols; p++) {
            int oh = p / W_out, ow = p % W_out;
            int ih = oh * stride - pad + kh;
            int iw = ow * stride - pad + kw;
            col[c * cols + p] =
                (ih >= 0 && ih < H && iw >= 0 && iw < W)
                ? in[ic * H * W + ih * W + iw] : 0.0f;
        }
    }

    // 权重重排为 (C_out, rows) 后，GEMM：out = W_mat × col
    std::vector<float> wmat(C_out * rows);
    for (int oc = 0; oc < C_out; oc++)
        for (int c = 0; c < rows; c++)
            wmat[oc * rows + c] = w[oc * rows + c];

    gemm(wmat.data(), col.data(), out, C_out, cols, rows);
    // 加 bias（逐元素 kernel，或融合进后续 ReLU）
}
```

**性能对比的意义**：im2col+GEMM 把卷积复用为 tiled GEMM（计算效率高），代价是展开内存；朴素版内存省但重复访存。bench 程序里用 `cudaEvent` 计时对比两者。

### 4.4 其他算子（ops.cu）

```cuda
// src/ops.cu
__global__ void relu_kernel(float* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) x[i] = fmaxf(x[i], 0.0f);
}
void relu_inplace(float* x, int size) {
    int block = 256;
    relu_kernel<<<(size + block - 1) / block, block>>>(x, size);
    cudaDeviceSynchronize();
}

__global__ void maxpool_kernel(const float* in, float* out,
                               int C, int H, int W, int pool) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * (H / pool) * (W / pool);
    if (idx >= total) return;
    int ow = idx % (W / pool);
    int oh = (idx / (W / pool)) % (H / pool);
    int c  = idx / ((W / pool) * (H / pool));
    float m = -1e30f;
    for (int ph = 0; ph < pool; ph++)
        for (int pw = 0; pw < pool; pw++) {
            int ih = oh * pool + ph, iw = ow * pool + pw;
            m = fmaxf(m, in[(c * H + ih) * W + iw]);
        }
    out[idx] = m;
}
void maxpool2d(const float* in, float* out, int C, int H, int W, int pool) {
    int total = C * (H / pool) * (W / pool);
    maxpool_kernel<<<(total + 255) / 256, 256>>>(in, out, C, H, W, pool);
    cudaDeviceSynchronize();
}
```

### 4.5 推理主循环（inference.cpp）

```cpp
// src/inference.cpp —— 加载权重，执行完整前向
#include "infer_ops.h"
#include <cstdio>
#include <vector>
#include <cstring>

static void load_bin(const char* path, std::vector<float>& out) {
    FILE* f = fopen(path, "rb");
    unsigned int n;
    fread(&n, 4, 1, f);
    out.resize(n);
    fread(out.data(), 4, n, f);
    fclose(f);
}

int main() {
    // 1. 加载权重（顺序与 export_weights.py 一致）
    std::vector<float> w1, b1, w2, b2, fc1w, fc1b, fc2w, fc2b;
    load_bin("weights.bin", w1); load_bin("weights.bin", b1); // 简化：实际应分段读
    // ... 建议：把 8 个张量按顺序一次性读入并切分（示例略）

    // 2. 分配张量
    Tensor in(1, 1, 28, 28);
    Tensor t1(1, 32, 28, 28); Tensor p1(1, 32, 14, 14);
    Tensor t2(1, 64, 14, 14); Tensor p2(1, 64, 7, 7);
    std::vector<float> flat(64 * 7 * 7);
    std::vector<float> h1(128), logits(10);

    // 3. 前向（每步注释对应模型结构）
    conv2d_naive(in.ptr(), w1.data(), b1.data(), t1.ptr(),
                 1, 28, 28, 32, 3, 1, 1, /*relu=*/true);   // Conv1+ReLU
    maxpool2d(t1.ptr(), p1.ptr(), 32, 28, 28, 2);           // Pool1
    conv2d_naive(p1.ptr(), w2.data(), b2.data(), t2.ptr(),
                 32, 14, 14, 64, 3, 1, 1, /*relu=*/true);  // Conv2+ReLU
    maxpool2d(t2.ptr(), p2.ptr(), 64, 14, 14, 2);           // Pool2
    // flatten：p2 (64,7,7) -> flat (3136)
    std::memcpy(flat.data(), p2.ptr(), flat.size() * sizeof(float));

    gemm(fc1w.data(), flat.data(), h1.data(), 128, 1, 3136);  // FC1
    relu_inplace(h1.data(), 128);
    gemm(fc2w.data(), h1.data(), logits.data(), 10, 1, 128);  // FC2
    std::vector<float> prob(10);
    softmax(logits.data(), prob.data(), 10);

    int pred = 0;
    for (int i = 1; i < 10; i++) if (prob[i] > prob[pred]) pred = i;
    printf("预测类别: %d\n", pred);
    return 0;
}
```

### 4.6 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(mnist_infer LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(mnist_infer
    src/inference.cpp
    src/gemm.cu
    src/conv.cu
    src/ops.cu)
target_compile_features(mnist_infer PRIVATE cxx_std_17)
target_compile_options(mnist_infer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--arch=sm_120>)

add_executable(test_accuracy tests/test_accuracy.cu
    src/gemm.cu src/conv.cu src/ops.cu)
target_compile_features(test_accuracy PRIVATE cxx_std_17)
target_compile_options(test_accuracy PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--arch=sm_120>)

add_executable(bench_conv bench/bench_conv.cu
    src/gemm.cu src/conv.cu src/ops.cu)
target_compile_features(bench_conv PRIVATE cxx_std_17)
target_compile_options(bench_conv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--arch=sm_120>)
```

## 5. 量化接口（预留 INT8 定点）

端侧推理的核心诉求是省带宽省显存。算子库预留一个"模拟量化"接口，后续可扩展为真正的 INT8 kernel：

```cpp
// include/quant.h（预留）
// 对称量化：q = round(clamp(x / scale, -127, 127))
struct QuantParam { float scale; };

// 模拟量化（fake quant）：先量化再反量化，验证精度损失
inline float fake_quant(float x, float scale) {
    int q = (int)roundf(fmaxf(-127.0f, fminf(127.0f, x / scale)));
    return q * scale;
}
```

**扩展路线**（留给读者，或作为后续优化篇）：

1. 用 `export_weights.py` 输出每个权重的 `scale`；
2. 把 `gemm` 换成 INT8 版本（`int8_t` 乘累加 + 反量化）；
3. 对比 FP32 / 模拟量化 / 真 INT8 的精度与性能。

## 6. 测试与性能对比

### 6.1 正确性测试（tests/test_accuracy.cu）

```cuda
// 读取 input.bin（1000 张图）与 expected.bin，逐张推理并对比 argmax
// 核心逻辑：加载 → 前向（同 inference.cpp）→ argmax 对比
// 输出：Accuracy: 99.9%  PASS/FAIL
```

运行：

```bash
cd mnist_infer && mkdir build && cd build
cmake .. && make -j
./test_accuracy
# 期望输出：Accuracy: 100.0% (1000/1000)  PASS
```

> ⚠️ 说明：如果 PyTorch 导出输入时做了与 C++ 完全一致的预处理（ToTensor + Normalize），两边的 argmax 应当 100% 一致；若出现少量不一致，优先检查权重读取顺序、内存布局（NCHW）和边界填充。

### 6.2 性能对比（bench/bench_conv.cu）

用 `cudaEvent` 分别计时 naive 卷积和 im2col+GEMM 卷积（同样的输入输出，各跑 100 次取平均）：

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
for (int i = 0; i < 100; i++)
    conv2d_naive(...);          // 或 conv2d_im2col_gemm(...)
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms; cudaEventElapsedTime(&ms, start, stop);
printf("平均耗时: %.3f ms\n", ms / 100);
```

预期结果（5060 Ti 上，单张 MNIST）：

| 实现 | 平均耗时 | 加速比 |
|:---|:---|:---|
| 朴素卷积（重复访存） | ~0.5 ms | 1.0× |
| im2col + GEMM（tiled 复用） | ~0.1 ms | ~5× |
| 单张端到端推理 | < 1 ms | — |

数值以本机实测为准——**性能数据必须自己跑出来，这是算子开发的基本素养**。

## 7. 练习与里程碑

### 练习

1. **跑通全流程**：训练 MNIST CNN（PyTorch 实战篇）→ `export_weights.py` 导出 → CMake 构建 → `test_accuracy` 输出 PASS。
2. **加算子**：在算子库里加一个 `avgpool2d`（平均池化），并用 `model.features[2]` 换成 AvgPool 的变体网络验证。
3. **真量化第一步**：给 `gemm` 写一个 INT8 版本（输入 int8、累加 int、输出 float），对比精度。
4. **换模型**：把模型改成 CIFAR-10（3 通道输入），调整卷积参数与权重导出脚本，跑通端到端。

### 里程碑自检

- [ ] 能用 CMake 一键构建算子库
- [ ] 端到端推理与 PyTorch 结果一致（test_accuracy PASS）
- [ ] 能说清朴素卷积与 im2col+GEMM 的优缺点
- [ ] 能测出本机 naive vs 优化卷积的加速比
- [ ] 知道量化接口如何扩展为真 INT8

## 8. 小结

本节交付了一个完整的纯 CUDA 推理算子库：

- **算子**：tiled GEMM、朴素/融合卷积、im2col+GEMM 卷积、ReLU/MaxPool/Softmax，全部手写；
- **验证**：与 PyTorch 端到端对比精度，工程化 CMake 构建；
- **性能**：naive vs 优化卷积实测对比；
- **扩展**：量化接口预留，可加算子、换模型。

这个工程骨架就是你的作品集核心。最后一步，我们把它的性能数据整理成一份专业的性能报告，并把它写成简历上能打动面试官的项目描述——让努力转化为面试中的优势。

> 🏷️ 标签：#综合项目 #CUDA算子库 #推理 #im2col #tiled GEMM #端到端验证
