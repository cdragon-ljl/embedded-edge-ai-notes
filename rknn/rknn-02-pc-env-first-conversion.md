# 基于RV1126的端侧AI开发 · 第2期 环境搭建与Resnet50v2模型转换测试

> 定位：本系列主要学习RKNN 工具链的使用 + 模型部署，从转换、量化、板端推理到性能调优
> 配套硬件：正点原子 RV1126 开发板（已下架，最新为RV1126B）+ IMX415 + 5.5' MIPI LCD
> 参考文档：正点原子官方
> 参考视频：[从ARM到AI视觉：基于RV1126B的嵌入式AI开发](https://space.bilibili.com/519718611/channel/collectiondetail?sid=7928216&spm_id_from=333.788.0.0)

## 0. 本节目标

上一节认识了 RV1126 和 RKNN 生态。本节动手走通第一段流水线：**在 PC 上把一个小型分类模型（ONNX 格式）转换成 `.rknn` 文件，并用 PC 模拟器跑一次推理，看到分类结果**。

完成本节，你就掌握了 RKNN 工具链的四个核心步骤：`config → load → build → export`，外加模拟推理 `init_runtime → inference`。这是整个系列的地基——后面所有内容（量化、精度评估、板端部署）都建立在这条流水线上。

## 1. 环境准备：PC 端转换环境

### 1.1 你需要什么

| 项 | 要求 |
|:---|:---|
| PC | x86_64 架构（rknn-toolkit 1.6.x 官方 wheel 只提供 Linux x86_64） |
| 系统 | Ubuntu 18.04 或 20.04（官方推荐 18.04，本系列文章使用20.04） |
| Python | 3.6（推荐；1.7.5 也提供 cp37 的 wheel，以官方 release 说明为准） |
| 网络 | 能访问 GitHub 下载 wheel 与模型 |

**⚠️ 版本红线**：这里安装的是 `rknn-toolkit`（一代，1.6.x），**不是 `rknn-toolkit2`**——装错包，后面所有 API 都会对不上。

```text
┌──────────────────────── PC（x86_64, Ubuntu）────────────────────────┐
│  rknn-toolkit 1.6.x（Python 包）                                     │
│   ├─ 模型转换：config → load → build → export_rknn  →  model.rknn   │
│   ├─ 模拟推理：init_runtime(target=None) 在 PC 上模拟 NPU 执行       │
│   └─ 量化/精度评估（后续篇）                                          │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ 把 model.rknn 拷贝到板子
┌───────────────────────────▼──────────────────────────RV1126 板端─────┐
│  librknnmrt.so（C 运行时） / rknn-toolkit-lite（Python 运行时）       │
│  真实 NPU 推理                                                       │
└───────────────────────────────────────────────────────────────────────┘
```


### 1.2 安装步骤

上节安装的正点原子SDK中已经包含了rknn-toolkit，版本号1.6.0。
```text
~/RV1126/atk-rv1126-sdk/external/rknn-toolkit$ tree -L 2
.
├── LICENSE
├── doc
├── examples
│   ├── caffe
│   ├── common_function_demos
│   ├── darknet
│   ├── keras
│   ├── mxnet
│   ├── onnx
│   ├── pytorch
│   ├── rknn_convert
│   ├── tensorflow
│   └── tflite
├── packages
│   ├── packages.md5sum
│   ├── required-packages-for-arm64-debian9-python35
│   ├── required-packages-for-win-python36
│   ├── requirements-cpu.txt
│   ├── requirements-gpu.txt
│   ├── rknn_toolkit-1.6.0-cp35-cp35m-linux_aarch64.whl
│   ├── rknn_toolkit-1.6.0-cp35-cp35m-linux_x86_64.whl
│   ├── rknn_toolkit-1.6.0-cp36-cp36m-linux_x86_64.whl
│   ├── rknn_toolkit-1.6.0-cp36-cp36m-macosx_10_15_x86_64.whl
│   ├── rknn_toolkit-1.6.0-cp36-cp36m-win_amd64.whl
│   ├── rknn_toolkit-1.6.0-cp37-cp37m-linux_aarch64.whl
│   └── rknn_toolkit-1.6.0-cp37-cp37m-macosx_10_15_x86_64.whl
├── platform-tools
│   ├── drivers_installer
│   ├── ntp
│   └── update_rk_usb_rule
└── rknn-toolkit-lite
    ├── examples
    └── packages
```
### 1.2.1 安装Anaconda3并创建python3.6虚拟环境：
```
conda create -n rknn160_py36 python=3.6
conda activate rknn160_py36
```
### 1.2.2 安装requirements
```
pip install -r requirements-cpu.txt
pip install rknn_toolkit-1.6.0-cp36-cp36m-linux_x86_64.whl
```
由于电脑环境不同，如安装出错可借助AI辅助安装。

### 1.2.3 验证
```
python
from rknn.api import RKNN
```
如无报错则表示环境安装成功。

## 2 模型转换与PC端测试

### 2.1 模型转换
篇幅问题，这里只展示部分转换脚本代码，基于`rknn_toolkit 1.6.0`，最新版本部分`api`参数已变：
```python
if __name__ == '__main__':
    # ==============================================================
    # 第 0 步：创建 RKNN 对象
    # ==============================================================
    # RKNN 类封装了 rknn-toolkit 的全部功能：
    #   加载模型 -> 配置 -> 构建 -> 导出 -> 初始化 -> 推理
    # 注意：在使用完 rknn 后需要调用 rknn.release() 释放资源。
    rknn = RKNN()

    # ==============================================================
    # 第 1 步：下载 ONNX 模型（仅当本地不存在时执行）
    # ==============================================================
    ...

    # ==============================================================
    # 第 2 步：配置模型参数（最关键的步骤之一）
    # ==============================================================
    # rknn.config 告诉转换器如何在 PC 上模拟 RKNPU 的行为：
    #   mean_values     : 图像三个通道的均值（在推理前减去，与训练时保持一致）
    #   std_values      : 图像三个通道的标准差（除以标准差，ImageNet 默认行）
    #   reorder_channel : 输入图像的通道顺序 '0 1 2' 表示 RGB；
    #                     If模型训练时用 BGR，可改成 '2 1 0'
    #   target_platform : 目标运行平台：
    #                     'rk3399pro'（老一代）、“rv1126”或“rv1109”
    print('--> config model')
    rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.82, 58.82, 58.82]],
                reorder_channel='0 1 2', target_platform='rv1126')
    # 注意：
    #   mean/std 的数值来自 ImageNet 训练集统计，
    #   如果你用的是自己的训练模型，请替换成你自己的均值/方差！
    print('done')    

    # ==============================================================
    # 第 3 步：加载 ONNX 模型
    # ==============================================================
    # rknn.load_onnx 会把 ONNX 模型解析为一个中间表示。
    # 返回值 non-zero 表示加载失败。
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load resnet50v2 failed!')
        exit(ret)                # 失败则退出程序
    print('done')

    # ==============================================================
    # 第 4 步：构建 RKNN 模型（核心转换过程）
    # ==============================================================
    # do_quantization=True 表示做 INT8 量化（降低模型精度、内存占用），
    #                     并且可以大幅提升 NPU 推理速度；
    # dataset 参数指向一个 txt 文件，每行一个图片路径，
    # 这些图片用于量化时估算每个激活值的范围。
    # 注意：如果量化后精度下降太明显，可尝试加大 dataset 中的图片数量，
    #       或关闭量化（do_quantization=False）。
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build resnet50v2 failed!')
        exit(ret)
    print('done')

    # ==============================================================
    # 第 5 步：导出 RKNN 模型（保存到磁盘）
    # ==============================================================
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export resnet50v2.rknn failed!')
        exit(ret)
    print('done')

    # ==============================================================
    # 第 6 步：加载测试图片（必须与 model 输入尺寸一致）
    # ==============================================================
    # OpenCV 读取图默认是 BGR 通道顺序，
    # 而 ImageNet 模型训练时使用的是 RGB 顺序，
    # 因此需要 cv2.cvtColor 转换通道顺序。
    img = cv2.imread('./dog_224x224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ==============================================================
    # 第 7 步：初始化运行环境
    # ==============================================================
    # rknn.init_runtime() 默认在 PC 上跑模拟器（使用 CPU/x86）；
    # 要跑在开发板（如 RV1126）上的话，需要连接上板子，
    # 并指定 target='rv1126' 等参数（需要通过 USB 连接设备）。
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # ==============================================================
    # 第 8 步：模型推理（Inference）
    # ==============================================================
    # rknn.inference 返回一个列表，list 中每个元素是某个输出的结果。
    # 这里只有一个输出（1000 类的 logits），所以直接取 [0]。
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    x = outputs[0]                 # x.shape = (1, 1000)

    # 用 softmax 把 logits 归一化成概率（分布）,
    # 便于直观比较每个类别的置信度：
    #   softmax(x)_i = exp(x_i) / sum(exp(x_j))
    output = np.exp(x)/np.sum(np.exp(x))
    outputs = [output]             # 打包成与接口一致的格式
    show_outputs(outputs)          # 打印 Top-5 结果
    print('done')

    # ==============================================================
    # 第 9 步：释放资源
    # ==============================================================
    rknn.release()
```

### 2.2 PC端测试
在以下路径下包含了一些基础的模型测试用例，这里选择`onnx`下的`resnet50v2`进行环境测试（关于模型知识后续会再做介绍，这里只是测试环境是否OK）：
```shell
(rknn160_py36) cloong@DESKTOP-ND03BR9:~/RV1126/atk-rv1126-sdk/external/rknn-toolkit/examples$ ls
caffe  common_function_demos  darknet  keras  mxnet  onnx  pytorch  rknn_convert  tensorflow  tflite
(rknn160_py36) cloong@DESKTOP-ND03BR9:~/RV1126/atk-rv1126-sdk/external/rknn-toolkit/examples$ cd onnx/
(rknn160_py36) cloong@DESKTOP-ND03BR9:~/RV1126/atk-rv1126-sdk/external/rknn-toolkit/examples/onnx$ ls
resnet50v2
(rknn160_py36) cloong@DESKTOP-ND03BR9:~/RV1126/atk-rv1126-sdk/external/rknn-toolkit/examples/onnx$ cd resnet50v2/
(rknn160_py36) cloong@DESKTOP-ND03BR9:~/RV1126/atk-rv1126-sdk/external/rknn-toolkit/examples/onnx/resnet50v2$ ls
README  dataset.txt  dog_224x224.jpg  resnet50v2.onnx  test.py
(rknn160_py36) cloong@DESKTOP-ND03BR9:~/RV1126/atk-rv1126-sdk/external/rknn-toolkit/examples/onnx/resnet50v2$ python test.py
--> config model
done
--> Loading model
W Please confirm that your onnx opset_version <= 11 (current opset_verison = 12)!!!
done
--> Building model
done
--> Export RKNN model
done
--> Init runtime environment
done
--> Running model
resnet50v2
-----TOP 5-----
[155]: 0.6592655181884766
[154]: 0.29156938195228577
[262]: 0.027164233848452568
[152]: 0.006162853911519051
[204 254]: 0.0029354472644627094

done
(rknn160_py36) cloong@DESKTOP-ND03BR9:~/RV1126/atk-rv1126-sdk/external/rknn-toolkit/examples/onnx/resnet50v2$ ls
README  dataset.txt  dog_224x224.jpg  resnet50v2.onnx  resnet50v2.rknn  test.py
```
从运行结果可以看出，TOP1为[155]，对应`imagenet`标签中的`Shih-Tzu`（西施犬）。
输入图片如下：

![dog_224x224](./images/dog_224x224.jpg)

脚本执行后同时生成`resnet50v2.rknn`模型文件。

## 3. 常见报错与排查

| 报错 | 原因 | 处理 |
|:---|:---|:---|
| `No module named 'rknn'` | 没装好 / wheel 平台不对 | 确认 wheel 是 cp36/cp37 且 Linux x86_64；重装 |
| `ImportError: onnx ...` | onnx/protobuf 版本过新 | 按 1.2 节锁定 numpy/onnx/protobuf 版本 |
| `load_tflite` 失败 | 模型算子不在支持列表 | 换官方模型；或用 ONNX 格式转换 |
| `build` 报 `unsupported op` | 模型里有 NPU 不支持的算子 | 后面转换篇专门讲算子约束与规避 |
| 推理结果全是垃圾 | mean/std 与模型训练预处理不一致 | 核对 config 的 mean/std 与模型说明 |
| 模拟推理很慢 | 正常，PC 模拟比 NPU 慢 | 用小型模型（MobileNet 级别）即可 |

## 4. 小结

本节跑通了 RKNN 转换流水线的第一段：

- **环境安装**：PC（x86_64 Ubuntu）+ rknn-toolkit **1.6.x**（一代，RV1126 专用）；
- **test脚本作用**：    
    1. 下载 ONNX 模型（若本地不存在）
    2. 配置模型转换参数（均值、标准差、通道顺序、目标平台）
    3. 加载 ONNX 模型
    4. 构建（转换）为 RKNN 模型（可包含量化）
    5. 导出 .rknn 文件
    6. 初始化运行环境（模拟器或真实 NPU 设备）
    7. 输入图片进行推理，并输出 Top-5 分类结果

下一步将基于转换生成的`.rknn`文件在开发板端完成模型运行，我们将使用瑞芯微官方提供的`rknn_model_zoo`仓库示例代码，其中包含更多的模型文件与测试程序，也是我们后续文章要学习的主要内容，感兴趣的可以提前了解。

> 🏷️ 标签：#RKNN #模型转换 #TFLite #ONNX #模拟推理 #环境搭建
