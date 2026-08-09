# 目标检测实战：YOLO 转换 + 板端后处理（解码 + NMS）

> 系列：RKNN 端侧部署实战：基于 RV1126 的模型转换与部署
> 前置：已完成分类模型的板端推理（C/Python 均可）
> 配套硬件：RV1126 开发板（本节的转换可在 PC，板端验证需要板子）

## 0. 本节目标

分类解决"是什么"，检测解决"**在哪** + 是什么"。IPC 场景（人形检测、周界报警、客流统计）的核心就是目标检测。

本节用 YOLOv5s 走完整检测链路：

1. **模型转换**：YOLOv5 ONNX → INT8 .rknn（检测模型转换的注意点）；
2. **看懂输出**：YOLO 输出的 85 维向量是什么、三个尺度怎么回事；
3. **后处理**：解码（把网格坐标还原成图像坐标）+ 置信度过滤 + **NMS**（去重选框），输出检测框。

## 1. YOLO 检测原理速览（嵌入式直觉）

### 1.1 检测 = 分类 + 定位

**定义 1（目标检测, Object Detection）**：在一张图中找出所有目标，输出每个目标的**类别 + 边界框（x, y, w, h）**。

嵌入式类比：分类像"看门人认人"（这人是谁），检测像"保安扫全场"（哪里有几个人、分别在哪个位置）。YOLO 的做法是把图像切成网格，**每个网格预测中心落在自己格子里的目标**。

### 1.2 网格 + 锚框

【图1：YOLO 网格与锚框】

```text
图像被分成 S×S 网格（如 80×80）
┌──┬──┬──┬──┐
│  │  │  │  │  每个格子预测：
├──┼──┼──┼──┤    · 目标中心是否在本格（置信度）
│  │■ │  │  │    · 相对本格坐标 (x, y)
├──┼──┼──┼──┤    · 宽高 (w, h)（相对锚框缩放）
│  │  │  │  │    · 各类别概率
└──┴──┴──┴──┘
每个格子还有 K 个"预设形状"（锚框），预测的是对锚框的缩放
```

> 图1 生图 prompt：网格示意图，白色背景。一张 4×4 网格，其中一个格子用红色高亮（目标中心），标注"目标中心在本格"；每个格子内画 3 个不同长宽比的虚线矩形（锚框：竖长条、横长条、方形）；右侧注释：每个格子预测 置信度 + 坐标(x,y) + 宽高(w,h) + 类别概率。扁平插画风，比例 16:9，中文标注。

**锚框（anchor）** 是预先定义的一组"典型目标形状"（比如人的框偏竖长、车的框偏扁平）。模型不直接预测绝对宽高，而是预测**相对锚框的缩放量**——这让模型更容易学（不用从零学"像素"这种绝对量）。

### 1.3 为什么有三个尺度

目标大小差异很大：远处的行人只有几十像素，近处的车占满半屏。YOLO 用**三个尺度的特征图**分别检测大小目标：

| 特征图 | 网格数 | 步长 stride | 负责的目标 |
|:---|:---|:---|:---|
| 小尺度层 | 80×80 | 8 | 小目标 |
| 中尺度层 | 40×40 | 16 | 中目标 |
| 大尺度层 | 20×20 | 32 | 大目标 |

## 2. 模型转换

### 2.1 获取 YOLOv5s ONNX

YOLOv5 官方仓库（`https://github.com/ultralytics/yolov5`）可导出 ONNX；也可以直接用瑞芯微官方示例中的 yolov5s onnx（`rknn-toolkit/examples/onnx/yolov5/` 下有转换脚本和说明，社区也有大量现成文件）。导出时注意：**固定输入尺寸 640×640**（动态尺寸转换麻烦且一代支持有限）。

### 2.2 转换脚本

与分类模型几乎相同，只多了**校准集用检测场景图片**（人、车等）：

```python
# convert_yolov5.py —— YOLOv5s → INT8 RKNN
from rknn.api import RKNN

rknn = RKNN(verbose=True)

rknn.config(
    mean_values=[[0, 0, 0]],              # YOLOv5 训练预处理：除以 255（0~1）
    std_values=[[255, 255, 255]],         # 即 x/255，不是 127.5 那套
    target_platform='rv1126',
    quantized_dtype='asymmetric_quantized-8',
    quantized_algorithm='kl_divergence')

rknn.load_onnx(model='yolov5s.onnx')

# 检测模型校准集：放 100~300 张人/车/常见目标图片
ret = rknn.build(do_quantization=True, dataset='dataset_det.txt')
if ret != 0:
    print('build 失败'); exit(ret)

rknn.export_rknn('yolov5s_int8.rknn')
print('✅ YOLOv5s 量化转换完成')
rknn.release()
```

**⚠️ 检测模型的 mean/std 和分类不同**：YOLOv5 官方预处理是像素除以 255（归一化到 0~1），所以 `mean_values=[[0,0,0]]`、`std_values=[[255,255,255]]`。**照抄分类模型的 127.5 那套会出问题**——再次验证那条铁律：预处理必须匹配模型训练时的设定。

## 3. 看懂 YOLO 输出

转换后的模型推理输出 shape 为 `[1, 25200, 85]`：

```text
25200 = (80×80 + 40×40 + 20×20) × 3 锚框 = 8400 × 3
85    = 5 + 80 = [cx, cy, w, h, obj_conf] + 80 个类别得分 (COCO)
```

【图2：输出向量结构】

```text
[1, 25200, 85]
 │
 ├── 前 8400 行：80×80 小尺度层（stride=8）的预测
 ├── 中 8400 行：40×40 中尺度层（stride=16）
 └── 后 8400 行：20×20 大尺度层（stride=32）

每行 85 个值：
┌────┬────┬────┬────┬────────┬───────────────┐
│ cx │ cy │ w  │ h  │ obj_conf│ 80 类得分      │
└────┴────┴────┴────┴────────┴───────────────┘
 格子内  格子内  相对  相对   该格有目标的   每个类别的
 中心x  中心y  锚框  锚框   概率          概率
```

> 图2 生图 prompt：数据布局图，白色背景。上方一个长条矩形标注 "[1, 25200, 85]"，用三种颜色分段标注：蓝（80×80 层）、绿（40×40 层）、橙（20×20 层）；下方展开一行 85 维向量，前 5 格高亮标注 cx/cy/w/h/obj_conf，后 80 格灰色标注"80 类得分"。扁平插画风，比例 16:9，中文标注。

## 4. 后处理：解码 + 置信度过滤 + NMS

### 4.1 为什么要后处理

NPU 输出的是"网格空间"的原始预测，要变成图像上的像素坐标框，需要：

1. **解码**：网格坐标 × stride + 锚框 → 像素坐标；
2. **过滤**：置信度（目标概率 × 类别概率）低于阈值的丢掉；
3. **NMS**：同一目标会有多个框重叠，保留最好的，抑制重复的。

**定义 2（NMS, Non-Maximum Suppression，非极大值抑制）**：在多个重叠的候选框中，保留置信度最高的，删除与它 IoU 超过阈值的其他框。

嵌入式类比：NMS 就像**多路信号去重**——多个传感器同时报警同一事件时，只保留最强信号的那一路，避免重复上报。IoU（交并比）就是两个框的重叠程度：重叠越多，越可能是同一个目标。

### 4.2 Python 后处理完整代码

板端用 lite 推理 + Python 后处理（验证阶段足够用；量产再移植 C）：

```python
# detect.py —— YOLOv5 板端检测（lite + 后处理）
import numpy as np
from PIL import Image, ImageDraw
from rknnlite.api import RKNNLite

# ---- 常量 ----
CLASSES = ('person', 'bicycle', 'car', ...)   # COCO 80 类，完整列表见附录/官方
ANCHORS = {
    8:  [[10,13], [16,30], [33,23]],    # 80×80 层锚框
    16: [[30,61], [62,45], [59,119]],   # 40×40 层锚框
    32: [[116,90], [156,198], [373,326]],  # 20×20 层锚框
}
CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_SIZE = 640

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def decode(pred, stride, anchors):
    """把一个尺度的输出解码成框（xyxy 像素坐标）+ 得分"""
    # pred: [1, 3, H, W, 85] → [H*W*3, 85]
    pred = pred.reshape(3, -1, 85)
    boxes, scores = [], []
    for ai, (aw, ah) in enumerate(anchors):
        p = pred[ai]                      # [H*W, 85]
        h, w = p.shape[0] ** 0.5, p.shape[0] ** 0.5  # 简化示意，见下方说明
        # 实际按网格坐标解码（完整代码见官方示例）：
        # cx = (sigmoid(p[:,0]) + grid_x) * stride
        # cy = (sigmoid(p[:,1]) + grid_y) * stride
        # bw = exp(p[:,2]) * aw ; bh = exp(p[:,3]) * ah
        # 此处为流程示意，省略网格索引计算
        obj = sigmoid(p[:, 4])
        cls = sigmoid(p[:, 5:]).max(axis=1)
        conf = obj * cls
        mask = conf > CONF_THRES
        # ... 拼接有效框到 boxes / scores ...
    return boxes, scores

def nms(boxes, scores, iou_thres=IOU_THRES):
    """NMS：按得分排序，逐个保留，抑制 IoU 过高的框"""
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = compute_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]
    return keep

def compute_iou(box, boxes):
    """计算一个框与一组框的 IoU（交并比）"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / union

# ---- 主流程 ----
rknn = RKNNLite()
rknn.load_rknn('yolov5s_int8.rknn')
rknn.init_runtime()

img = Image.open('test_det.jpg').convert('RGB')
img_resized = img.resize((IMG_SIZE, IMG_SIZE))
input_data = np.array(img_resized).astype(np.uint8)

outputs = rknn.inference(inputs=[input_data])   # 3 个输出（3 个尺度）
boxes_all, scores_all = [], []
for out, stride, anchors in zip(outputs, (8, 16, 32),
                                (ANCHORS[8], ANCHORS[16], ANCHORS[32])):
    boxes, scores = decode(out, stride, anchors)
    boxes_all += boxes
    scores_all += scores

boxes_all = np.array(boxes_all)
scores_all = np.array(scores_all)
keep = nms(boxes_all, scores_all)

# 画框
draw = ImageDraw.Draw(img)
for idx in keep:
    x1, y1, x2, y2 = boxes_all[idx] * (img.width / IMG_SIZE)   # 缩放回原图
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
img.save('result.jpg')
print(f'检测到 {len(keep)} 个目标, 结果保存 result.jpg')

rknn.release()
```

> ⚠️ 上面的 `decode` 是**流程示意**：完整实现需要按网格坐标（grid_x/grid_y）逐点解码，代码较长。瑞芯微官方示例 `rknn-toolkit/examples/onnx/yolov5/` 提供完整可运行的 Python 后处理（`yolov5_postprocess.py`），**建议直接参考官方实现**，核心逻辑与上面一致：sigmoid → 网格解码 → 置信度过滤 → NMS。

## 5. 常见问题

| 现象 | 原因 | 处理 |
|:---|:---|:---|
| 检测框全在图像边缘/中心 | 解码公式错误（grid 索引没加） | 对照官方示例逐行核对解码 |
| 同一目标多个重叠框 | NMS 阈值太松 / 没跑 NMS | 确认 NMS；IOU_THRES 调低（0.4~0.5） |
| 什么都检不到 | CONF_THRES 太高 / 量化掉精度 | 降到 0.15 试试；检查校准集 |
| 小目标漏检 | 量化对检测模型更敏感 | kl_divergence；校准集覆盖小目标场景 |
| 框位置偏 | 原图缩放比例没还原 | 框坐标要 ×(原图宽/输入宽) |

## 6. 练习与里程碑

### 练习

1. **跑通检测**：用官方示例的完整后处理代码，在板子上跑通人/车检测，保存标注图；
2. **调阈值**：分别用 CONF_THRES=0.1 / 0.25 / 0.5 跑同一张图，观察漏检与误检的权衡；
3. **量化对比**：对比浮点 YOLO 与 INT8 YOLO 在同一张图上的检测结果，观察量化对检测精度的影响；
4. **读官方代码**：读懂官方 `yolov5_postprocess.py` 的网格解码部分，画一张解码数据流图。

### 里程碑自检

- [ ] 能解释 25200 和 85 的来历
- [ ] 知道检测模型的 mean/std 与分类模型不同（YOLOv5 用 0/255）
- [ ] 能跑通 YOLO 检测并输出正确检测框
- [ ] 能说清 NMS 的原理和 IoU 的作用
- [ ] 能量化对比浮点/INT8 检测效果

## 7. 小结

- **检测 = 分类 + 定位**：YOLO 把图切网格、用锚框预测目标框，三个尺度覆盖大小目标；
- **输出结构**：`[1, 25200, 85]`（8400 网格 × 3 锚框，85 = 5 框参数 + 80 类）；
- **转换注意**：YOLOv5 预处理是 0~255 那套（mean=0, std=255），别照抄分类；
- **后处理三件套**：解码（网格 → 像素）→ 置信度过滤 → NMS 去重，官方示例代码可参考。

检测跑通后，你的板子已经能"看懂"画面里有什么、在哪。但目前的输入还是**静态图片**——真实产品是**摄像头实时视频**。下一阶段把摄像头接进来：RV1126 的 ISP/RKMedia 取流 → 推理 → 结果显示，组成一条完整的实时链路。

> 🏷️ 标签：#RKNN #YOLO #目标检测 #NMS #后处理 #RV1126
