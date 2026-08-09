# 摄像头 → NPU 全链路：RKMedia 取流 + 推理 + 显示

> 系列：RKNN 端侧部署实战：基于 RV1126 的模型转换与部署
> 前置：已完成 YOLO 检测（图片输入）与板端推理
> 配套硬件：RV1126 开发板 + MIPI/USB 摄像头 + 显示器或网口

## 0. 本节目标

前面所有推理都在**静态图片**上完成。真实产品（IPC、门禁）是**实时视频**：摄像头每秒 25~30 帧进来，每帧都要过一遍"取流 → 预处理 → 推理 → 后处理 → 显示/存储"。

本节打通 RV1126 的**摄像头全链路**：

1. 认识 RKMedia（rk_mpi）媒体框架：VI / VPSS / VO 各管什么；
2. 用代码把 Sensor 图像送到 NPU 推理，再把结果叠加显示；
3. 跑通第一个"实时视频检测"demo（IPC 场景雏形）。

## 1. 全链路总览

【图1：摄像头 → NPU 全链路数据流】

```text
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Sensor   │──▶│  VI      │──▶│  VPSS    │──▶│  推理线程  │
│ (摄像头) │   │ 采集入流  │   │ 缩放/格式 │   │  NPU 执行  │
└──────────┘   └──────────┘   └──────────┘   └────┬─────┘
                                                  │
┌──────────┐   ┌──────────┐   ┌──────────┐        │
│ 显示/编码 │◀──│  VO      │◀──│ 结果叠加  │◀───────┘
│ (HDMI/RTSP)│   │ 输出显示 │   │ (画框)   │
└──────────┘   └──────────┘   └──────────┘
```

> 图1 生图 prompt：横向流水线图，白色背景。从左到右六个节点：Sensor 摄像头（图标）→ VI 采集入流 → VPSS 缩放/格式转换 → 推理线程 NPU 执行（高亮橙色）→ 结果叠加画框 → VO 输出显示/编码。每个节点下方一行小字说明。扁平插画风，比例 16:9，中文标注。

**定义 1（RKMedia）**：瑞芯微的媒体软件框架（Rockit），提供 `rk_mpi_*` 系列 API 管理视频通路。

| 模块 | API 前缀 | 职责 | 嵌入式类比 |
|:---|:---|:---|:---|
| VI | `RK_MPI_VI_*` | Video Input，从 Sensor 采集图像 | 摄像头驱动的"读卡器" |
| VPSS | `RK_MPI_VPSS_*` | Video Post-Processing，缩放/裁剪/格式转换 | 图像预处理流水线（硬件加速） |
| VO | `RK_MPI_VO_*` | Video Output，送显示器/编码器 | 显示驱动的"写屏器" |

**为什么需要 VPSS**：NPU 需要 640×640 的 RGB 输入，而 Sensor 输出的是 1920×1080 的 NV12（YUV 格式）。直接用 CPU 缩放+转格式太慢（1080p 一帧要好几毫秒），VPSS 是**硬件加速**的，几乎不占 CPU。

## 2. 环境准备

RV1126 SDK（Buildroot）中启用 Rockit 相关配置（一般默认开启）：

```bash
# 确认设备节点
ls /dev/video*        # 摄像头设备
ls /dev/rknpu         # NPU

# 确认 RKMedia 库（SDK 构建后位于板端系统）
ls /usr/lib/librk_mpi* 
```

## 3. 全链路代码框架

以"摄像头 → VI → VPSS 缩放 → NPU 推理 → VO 显示"为例，核心流程：

```c
// pipeline.c —— 摄像头→NPU→显示 全链路（核心骨架）
#include <rk_mpi.h>
#include "rknn_api.h"

int main(void) {
    /* ---------- 1. 初始化 RKMedia 通路 ---------- */
    RK_MPI_SYS_Init();   // 系统初始化

    // VI：绑定摄像头 Sensor（参数：设备号/分辨率/帧率，按你的 Sensor 配置）
    VI_CHN_ATTR_S vi_attr = {0};
    vi_attr.pcVideoNode = "mipi0";            // MIPI 摄像头节点名
    vi_attr.stSize.u32Width  = 1920;
    vi_attr.stSize.u32Height = 1080;
    vi_attr.enPixFmt = RK_FMT_YUV420SP;       // NV12
    RK_MPI_VI_SetChnAttr(0, 0, &vi_attr);
    RK_MPI_VI_EnableChn(0, 0);

    // VPSS：缩放 + 格式转换（1080p NV12 → 640x640 RGB）
    VPSS_CHN_ATTR_S vpss_attr = {0};
    vpss_attr.stSize.u32Width  = 640;
    vpss_attr.stSize.u32Height = 640;
    vpss_attr.enPixFmt = RK_FMT_RGB888;
    RK_MPI_VPSS_SetChnAttr(0, 0, &vpss_attr);
    RK_MPI_VPSS_EnableChn(0, 0);
    // 绑定：VI → VPSS
    MPP_CHN_S vi_chn = {RK_ID_VI, 0, 0};
    MPP_CHN_S vpss_chn = {RK_ID_VPSS, 0, 0};
    RK_MPI_SYS_Bind(&vi_chn, &vpss_chn);

    /* ---------- 2. 初始化 NPU ---------- */
    rknn_context ctx;
    rknn_init(&ctx, "yolov5s_int8.rknn", 0, 0, NULL);
    // ... rknn_query 获取输入尺寸（640x640x3） ...

    /* ---------- 3. 主循环：取帧 → 推理 → 显示 ---------- */
    for (;;) {
        // 3.1 从 VPSS 取一帧（阻塞等待）
        VIDEO_FRAME_INFO_S frame;
        if (RK_MPI_VPSS_GetChnFrame(0, 0, &frame, -1) != RK_SUCCESS)
            continue;
        // frame 的 virAddr 指向 640x640 RGB 数据

        // 3.2 喂给 NPU
        rknn_input inputs[1] = {0};
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].size = 640 * 640 * 3;
        inputs[0].buf = (void *)frame.virAddr;
        inputs[0].pass_through = 0;
        rknn_inputs_set(ctx, 1, inputs);
        rknn_run(ctx, NULL);

        // 3.3 取结果 + 后处理（解码/NMS，见检测篇）
        rknn_output outputs[1] = {{0}};
        outputs[0].want_float = 1;
        rknn_outputs_get(ctx, 1, outputs, NULL);
        // ... decode + nms → 得到检测框列表 ...

        // 3.4 画框（在帧数据上叠加矩形，直接写内存）
        // draw_boxes(frame.virAddr, 640, 640, boxes);

        // 3.5 送显示（VO 通道）
        RK_MPI_VO_SendFrame(0, &frame);
        // 或送编码器做 RTSP 推流

        // 3.6 释放帧（重要！不释放会丢帧/内存泄漏）
        RK_MPI_VPSS_ReleaseChnFrame(0, 0, &frame);
    }

    /* ---------- 4. 清理 ---------- */
    rknn_release(ctx);
    RK_MPI_SYS_UnBind(&vi_chn, &vpss_chn);
    RK_MPI_SYS_Exit();
    return 0;
}
```

> ⚠️ 上述为**核心骨架**：完整工程还需 Sensor 驱动配置（dts/media）、VO 显示参数、帧缓冲管理等，均与具体板卡/SDK 版本相关。实际开发时**以 SDK 自带的 sample 为起点修改**（RV1126 SDK 的 `examples/rkmedia` 或 `samples/rk_mpi` 目录下有 VI→VPSS→VO 完整 demo，例如 vi_vpss_vo 示例），在其基础上插入推理即可。

## 4. 工程落地要点

### 4.1 从 SDK 示例改起

不建议从零写 RKMedia 代码。路径建议：

```text
SDK 示例（vi_vpss_vo / vi_vo 等）
   │ 读懂通路建立逻辑
   ▼
修改：VPSS 输出尺寸 = 模型输入（640×640）
   ▼
插入：NPU 推理（五步 API 已有）
   ▼
插入：后处理 + 画框（直接写帧内存）
   ▼
验证：显示器或 RTSP 查看实时检测效果
```

### 4.2 常见工程坑

| 坑 | 表现 | 处理 |
|:---|:---|:---|
| 帧忘记 Release | 内存持续增长，最终丢帧 | 每帧必须 `RK_MPI_VPSS_ReleaseChnFrame` |
| 推理太慢占住取流 | 画面卡顿/队列溢出 | 取流与推理分离到不同线程（下节重点） |
| 格式不匹配 | 推理结果错乱 | VPSS 输出必须与模型输入一致（RGB888/NHWC） |
| 显示花屏 | VO 参数与帧格式不符 | 核对 VO 分辨率/像素格式 |

### 4.3 验证方式：先显示，再推流

调试阶段优先接 HDMI/VO 直接看画面（实时观察检测框是否正确）；稳定后再接编码器走 RTSP 网络推流（IPC 产品的标准形态）。

## 5. 练习与里程碑

### 练习

1. **跑 SDK 示例**：先跑通 SDK 的 vi_vpss_vo 示例，确认摄像头取流 + 显示正常；
2. **插入推理**：在示例的 VPSS 输出后插入 NPU 分类或检测，实现"实时分类"；
3. **叠加画框**：把检测框画到帧内存上，通过 VO 确认叠加效果；
4. **测量延迟**：记录从取帧到显示的耗时，找出当前瓶颈在哪一段。

### 里程碑自检

- [ ] 能画出 VI → VPSS → 推理 → VO 的数据流
- [ ] 能跑通 SDK 摄像头示例并看到画面
- [ ] 能把推理插入取流循环，实时显示检测结果
- [ ] 知道每帧必须 Release 的原因
- [ ] 能说出当前 demo 的瓶颈环节

## 6. 小结

- **链路**：Sensor → VI（采集）→ VPSS（缩放/格式）→ NPU（推理）→ 叠加 → VO（显示/编码）；
- **RKMedia** = `rk_mpi_*` API，VPSS 硬件加速做预处理，别用 CPU 干；
- **工程方法**：从 SDK 示例改起，先跑通显示再插入推理；
- **纪律**：每帧必须 Release，取流与推理分离线程（否则卡顿）。

实时链路通了，但"能跑"和"跑得好"之间还有很大距离——帧率、延迟、CPU 占用、内存都可能不理想。下一节进入性能调优：把这条链路从"demo 级"优化到"产品级"。

> 🏷️ 标签：#RKNN #RKMedia #摄像头 #IPC #全链路 #RV1126
