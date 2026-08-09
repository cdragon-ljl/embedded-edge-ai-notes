# 板端 C 推理第一课：rknn_api 五步流程跑通图像分类

> 系列：RKNN 端侧部署实战：基于 RV1126 的模型转换与部署
> 前置：已完成 INT8 量化模型（mobilenetv2_int8.rknn）
> 配套硬件：RV1126 开发板（本文起需要板子）；交叉编译主机：x86_64 Ubuntu

## 0. 本节目标

模型在 PC 上模拟通过，现在搬到真实硬件。RV1126 的板端 C 推理就是经典的**五步 API**：

```text
rknn_init → rknn_query → rknn_inputs_set → rknn_run → rknn_outputs_get
（初始化）   （查输入输出） （喂数据）        （执行）     （取结果）
```

本节完成三件事：

1. **搭起板端运行环境**：拷贝模型、链接 `librknnmrt.so`、交叉编译；
2. **写一个完整的图像分类 C 程序**：五步 API 全流程 + 输入预处理；
3. **跑通并看懂结果**：CPU/NPU 耗时、top1 类别、常见坑。

## 1. 板端运行环境

### 1.1 板子上要有什么

| 组件 | 位置（SDK 内） | 说明 |
|:---|:---|:---|
| 运行时库 `librknnmrt.so` | `external/rknn-toolkit/rknpu/` | NPU 推理库，拷到板子 `/usr/lib/` 或程序同目录 |
| 头文件 `rknn_api.h` | `external/rknn-toolkit/rknpu/` | C API 声明，编译时需要 |
| 模型 `mobilenetv2_int8.rknn` | 你自己转换的 | 拷到板子任意目录 |

**先验证 NPU 驱动正常**（板子串口/SSH 登录后）：

```bash
# 查看 rknpu 相关内核模块是否加载
ls /dev/rknpu*         # 一代平台通常出现 /dev/rknpu
dmesg | grep -i rknpu  # 看初始化日志
```

> 如果 `/dev/rknpu` 不存在，说明内核没启用 NPU 驱动，先确认 SDK 内核配置（`CONFIG_ROCKCHIP_RKNPU=y`）再刷机。

### 1.2 交叉编译环境

**重要**：RV1126 的四核 Cortex-A7 是 **32 位 ARM**（armv7l），交叉编译工具链用 **arm-linux-gnueabihf**，不是 aarch64：

```bash
# 安装工具链
sudo apt install -y gcc-arm-linux-gnueabihf

# 验证
arm-linux-gnueabihf-gcc --version
```

## 2. 完整 C 程序：五步推理图像分类

创建 `classify.c`（放在 SDK 的 rknpu 目录旁边，方便引用头文件）：

```c
// classify.c —— RV1126 板端图像分类（rknn_api 五步流程）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rknn_api.h"

// ---- 简单读取 RGB 图像（无第三方库，BMP 24bit 为例）----
// 实际项目建议用 stb_image / libjpeg / RKMedia 取帧，此处为演示
static unsigned char *load_image(const char *path, int *w, int *h) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("open image"); return NULL; }
    // ... 省略 BMP 解析细节（实际工程用成熟库）...
    // 返回 RGB888 数据，*w/*h 为宽高
    fclose(fp);
    return data;
}

// ---- top-k 工具：取输出中得分最高的 k 个类别 ----
static void topk(const float *scores, int n, int k) {
    for (int t = 0; t < k; t++) {
        int idx = -1; float maxv = -1.0f;
        for (int i = 0; i < n; i++) {
            if (scores[i] > maxv) { maxv = scores[i]; idx = i; }
        }
        printf("  Top%d: class %d, score %.4f\n", t + 1, idx, maxv);
        scores[idx] = -1.0f;   // 用完后抹掉，找下一个
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("用法: %s <model.rknn> <image>\n", argv[0]);
        return -1;
    }

    /* ============ 第 1 步：rknn_init 初始化 ============ */
    rknn_context ctx;
    int ret = rknn_init(&ctx, argv[1], 0, 0, NULL);
    if (ret < 0) { printf("rknn_init 失败: %d\n", ret); return -1; }
    printf("[1/5] rknn_init OK\n");

    /* ============ 第 2 步：rknn_query 查询输入输出信息 ============ */
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) { printf("rknn_query 失败: %d\n", ret); return -1; }
    printf("[2/5] 输入数=%d, 输出数=%d\n", io_num.n_input, io_num.n_output);

    // 查输入 tensor 属性（shape、类型）
    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    if (ret < 0) { printf("查询输入属性失败\n"); return -1; }
    int in_w = input_attr.dims[2];   // NHWC 布局：N,H,W,C
    int in_h = input_attr.dims[1];
    int in_c = input_attr.dims[3];
    printf("输入尺寸: %dx%dx%d\n", in_w, in_h, in_c);

    /* ============ 第 3 步：rknn_inputs_set 喂数据 ============ */
    int img_w, img_h;
    unsigned char *img = load_image(argv[2], &img_w, &img_h);
    if (!img) return -1;
    // 图像缩放：将原图缩放到模型输入尺寸（此处示意，实际用 RGA/libyuv 高效缩放）
    unsigned char *resized = malloc(in_w * in_h * in_c);
    resize_bilinear(img, img_w, img_h, resized, in_w, in_h, in_c); // 工具函数

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;          // 喂原始 0~255 像素
    inputs[0].fmt = RKNN_TENSOR_NHWC;            // 与模型输入布局一致
    inputs[0].size = in_w * in_h * in_c;
    inputs[0].buf = resized;
    inputs[0].pass_through = 0;                  // 0: 由库做 mean/std 归一化
    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) { printf("rknn_inputs_set 失败: %d\n", ret); return -1; }
    printf("[3/5] rknn_inputs_set OK\n");

    /* ============ 第 4 步：rknn_run 执行推理 ============ */
    ret = rknn_run(ctx, NULL);
    if (ret < 0) { printf("rknn_run 失败: %d\n", ret); return -1; }
    printf("[4/5] rknn_run OK\n");

    /* ============ 第 5 步：rknn_outputs_get 取结果 ============ */
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;                   // 要浮点输出（库做反量化）
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0) { printf("rknn_outputs_get 失败: %d\n", ret); return -1; }
    printf("[5/5] rknn_outputs_get OK\n");

    float *scores = (float *)outputs[0].buf;
    int out_size = 1;
    for (int i = 0; i < io_num.n_output; i++) {
        // 实际应从输出 tensor 属性取 dims 计算；此处单输出模型
        out_size = 1000;   // MobileNetV2 是 1000 类
    }
    printf("分类结果:\n");
    topk(scores, out_size, 5);

    /* ============ 释放资源 ============ */
    rknn_outputs_release(ctx, 1, outputs);
    free(resized);
    free(img);
    rknn_release(ctx);
    printf("done.\n");
    return 0;
}
```

> ⚠️ 代码中 `load_image` 和 `resize_bilinear` 为示意函数（BMP 解析 + 双线性缩放，代码较长未完整贴出）。实际工程请用 **stb_image + libyuv/RGA** 或直接走 RKMedia 取帧（后面摄像头篇会讲）。重点是五步 API 的结构。

## 3. 编译与运行

### 3.1 编译

```bash
# 假设 SDK 的 rknpu 目录下有 rknn_api.h 和 librknnmrt.so
arm-linux-gnueabihf-gcc classify.c -o classify \
    -I<SDK>/external/rknn-toolkit/rknpu/include \
    -L<SDK>/external/rknn-toolkit/rknpu/lib \
    -lrknnmrt -lm
```

### 3.2 部署到板子

```bash
# 拷贝三个文件到板子（scp 或 U 盘）
scp classify mobilenetv2_int8.rknn test.jpg root@<板子IP>:/root/
scp <SDK>/external/rknn-toolkit/rknpu/lib/librknnmrt.so root@<板子IP>:/usr/lib/
```

### 3.3 运行

```bash
# 板子上
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
./classify mobilenetv2_int8.rknn test.jpg
```

**预期输出**：

```text
[1/5] rknn_init OK
[2/5] 输入数=1, 输出数=1
输入尺寸: 224x224x3
[3/5] rknn_inputs_set OK
[4/5] rknn_run OK
[5/5] rknn_outputs_get OK
分类结果:
  Top1: class 281, score 0.9021
  ...
done.
```

类别 281 = 虎斑猫，与测试图一致就成功了。

## 4. 关键点解读

### 4.1 输入数据：只需要 0~255 原始像素

注意第 3 步：`inputs[0].type = RKNN_TENSOR_UINT8`，喂的是**原始 0~255 像素**。mean/std 归一化哪去了？

**在转换时通过 config 写进了模型**（你在 PC 端配的 `mean_values/std_values`），运行时库自动应用。所以板端 C 代码**不需要手动归一化**——这是 RKNN 设计的一大便利。反过来，如果你在板端代码里又手动做了一次归一化（除以 255 再乘 128），结果必错。

### 4.2 pass_through 参数

- `pass_through = 0`：库做 mean/std 归一化和数据布局转换（默认，推荐）；
- `pass_through = 1`：原始数据直通 NPU，不做任何预处理（此时你必须自己完成全部预处理，通常配合零拷贝使用）。

新手一律用 `pass_through = 0`。

### 4.3 输入输出格式与内存

| 项目 | 说明 |
|:---|:---|
| 输入 fmt | `RKNN_TENSOR_NHWC` 或 `NCHW`，以模型实际布局为准（TFLite/ONNX 常见 NHWC） |
| 输出 want_float | `1` = 拿浮点结果（库反量化，方便调试）；`0` = 拿 INT8 原始结果（省一次转换，性能好） |
| 内存 | 基本模式是库内部分配；追求极致性能可用 `rknn_create_mem/rknn_set_io_mem` 零拷贝（一代平台支持情况查官方文档，标注待核实） |

### 4.4 性能观察

板端推理耗时可以用简单计时看：

```c
#include <time.h>
struct timespec t0, t1;
clock_gettime(CLOCK_MONOTONIC, &t0);
rknn_run(ctx, NULL);
clock_gettime(CLOCK_MONOTONIC, &t1);
double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
            (t1.tv_nsec - t0.tv_nsec) / 1e6;
printf("rknn_run 耗时: %.2f ms\n", ms);
```

MobileNetV2 INT8 在 RV1126 上单帧推理通常在 **10~30 ms** 量级（具体与模型、量化、NPU 频率有关，以实测为准）。**注意这只是 NPU 推理时间，不含取图/预处理/后处理**——全链路延迟后面专门有一篇讲。

## 5. 常见问题

| 现象 | 原因 | 处理 |
|:---|:---|:---|
| `rknn_init` 返回 -1/-2 | 库缺失 / 驱动未加载 / 模型是 Toolkit2 转的 | 确认 librknnmrt.so 在 LD_LIBRARY_PATH；`ls /dev/rknpu`；重转一代模型 |
| `rknn_inputs_set` 返回非 0 | 输入尺寸或类型不匹配 | 用 query 打出的 dims 核对；确认 fmt/type |
| 结果全错 | 输入布局/通道顺序问题 | 核对 NHWC/NCHW、RGB/BGR、是否重复归一化 |
| 编译找不到 rknn_api.h | 头文件路径不对 | `-I` 指向 include 目录 |
| 板端报 `cannot open shared object` | 运行时库没拷/没设路径 | `export LD_LIBRARY_PATH`；确认 .so 存在 |

## 6. 练习与里程碑

### 练习

1. **跑通五步**：把分类程序跑通，记录 rknn_run 耗时；
2. **多图测试**：准备 10 张不同类别图片，逐一推理，统计 top1 是否正确；
3. **格式实验**：把输入 fmt 改成 NCHW（数据也要按 NCHW 重排）再跑，观察结果变化，体会布局匹配的重要性；
4. **计时**：把耗时打印加进程序，连续跑 100 帧统计平均耗时与最大耗时。

### 里程碑自检

- [ ] 能说出板端五步 API 的名字和职责
- [ ] 能独立完成交叉编译 → 拷贝 → 板端运行
- [ ] 知道输入只需 0~255 原始像素（mean/std 已编译进模型）
- [ ] 知道 pass_through 0/1 的区别
- [ ] 能测出单次 rknn_run 耗时

## 7. 小结

- **环境**：`librknnmrt.so` + `rknn_api.h`，RV1126 用 arm-linux-gnueabihf 交叉编译（32 位）；
- **五步 API**：`init → query → inputs_set → run → outputs_get`，外加 `outputs_release` 和 `release` 收尾；
- **输入**：喂 0~255 原始像素 + 正确 fmt，归一化由库完成（pass_through=0）；
- **输出**：`want_float=1` 拿浮点方便调试，追求性能用 INT8 直出；
- **验证**：MobileNetV2 INT8 单次推理 10~30 ms 量级（以实测为准）。

C 版本是生产级路线。但调试阶段、快速验证模型效果时，用 C 写一遍太慢——板端 Python 推理（rknn-toolkit-lite）就是为这个场景准备的：同样的模型，Python 三行搞定推理，原型开发效率高一个量级。

> 🏷️ 标签：#RKNN #rknn_api #板端部署 #交叉编译 #C语言
