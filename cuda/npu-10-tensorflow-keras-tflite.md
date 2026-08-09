# TensorFlow 实战：tf.keras 训练、SavedModel / TFLite 导出与框架对比

> 系列：CUDA 高性能算子实战 · NPU-10
> 前置：NPU-09（PyTorch 训练 CNN 与导出全流程）
> 配套环境：Python 3.10+、pip install tensorflow（CPU 版即可；GPU 版要求见下文）

## 0. 本节目标

上一节我们用 PyTorch 完成了"定义 CNN → 训练 → 导出 ONNX"的闭环。本节学习另一大主流框架 **TensorFlow**，用同样的 MNIST 任务走一遍全流程，然后回答两个问题：

1. TensorFlow 的常用姿势是什么（`tf.keras` 高层 API）？
2. 它和 PyTorch 到底差在哪——从模型定义、训练循环到模型导出？

最后你会得到一份**逐环节对照表**，以后看到任何项目用哪个框架都不会发怵。TensorFlow 生态在**端侧/移动部署**（TFLite）上有很强的存在感，这与我们"算子服务真实负载"的目标直接相关。

## 1. 环境搭建

```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow        # CPU 版，本节实验完全够用
```

验证：

```python
import tensorflow as tf
print("TensorFlow:", tf.__version__)          # 2.x
print("GPU 列表:", tf.config.list_physical_devices('GPU'))
# CPU 版会输出空列表；GPU 版会列出 ['/physical_device:GPU:0']
```

> ⚠️ 版本说明：TensorFlow 的 GPU 支持对 CUDA/cuDNN 版本要求较严格（需要与 TF 版本配套的 CUDA 与 cuDNN，官方提供 `tensorflow[and-cuda]` 一键安装方式）。Blackwell（sm_120）支持从较新版本开始，具体请以 [tensorflow.org/install](https://www.tensorflow.org/install) 的 GPU 指引为准。本节实验全部用 CPU 完成，不影响学习目标；GPU 加速对 MNIST 这种小任务收益也不明显。

## 2. 用 tf.keras 定义 CNN

TensorFlow 的主流开发方式是用 **`tf.keras`**——一个内置的高层 API，模型定义、编译、训练、评估全部封装成简洁调用。

### 2.1 同样的 LeNet 风格 CNN

```python
import tensorflow as tf

# ---------- 1. 数据准备（Keras 自带 MNIST） ----------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 归一化到 [0,1] 并增加通道维： (60000,28,28) -> (60000,28,28,1)
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# ---------- 2. 定义模型（Sequential：一层接一层） ----------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same',
                           activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),          # 28x28 -> 14x14
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2),          # 14x14 -> 7x7
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),  # 10 类概率
])

model.summary()   # 打印各层输出形状和参数量（和 PyTorch 的 print(model) 对应）
```

`model.summary()` 会打印类似：

```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
conv2d_1 (Conv2D)            (None, 14, 14, 64)        18496
max_pooling2d_1 (MaxPooling2D)(None, 7, 7, 64)         0
flatten (Flatten)            (None, 3136)              0
dense (Dense)                (None, 128)               401536
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 421,642
```

注意 `(None, 28, 28, 32)`：第一维 None 表示 batch 大小可变，之后是"高、宽、通道"。这和 PyTorch 的 `(batch, channel, height, width)` 顺序不同——**TensorFlow 是"通道在最后"（NHWC），PyTorch 是"通道在前"（NCHW）**，这是两个框架最经典的差异之一，后面导出模型、对接推理引擎时经常要处理这个布局转换。

## 3. 编译与训练

### 3.1 高级 API：model.fit

```python
# ---------- 3. 编译：指定优化器、损失、评估指标 ----------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # 标签是整数（0~9）时用 sparse 版
    metrics=['accuracy'],
)

# ---------- 4. 训练 ----------
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=5,
    validation_data=(x_test, y_test),   # 每个 epoch 后自动评估
)

# 预期输出（每个 epoch 约 20~60 秒，CPU）：
# Epoch 1/5  loss: 0.1798 - accuracy: 0.9461 - val_accuracy: 0.9774
# Epoch 2/5  loss: 0.0555 - accuracy: 0.9831 - val_accuracy: 0.9852
# Epoch 3/5  loss: 0.0381 - accuracy: 0.9883 - val_accuracy: 0.9886
# Epoch 4/5  loss: 0.0280 - accuracy: 0.9912 - val_accuracy: 0.9895
# Epoch 5/5  loss: 0.0225 - accuracy: 0.9929 - val_accuracy: 0.9902
```

`model.fit` 把上一节 PyTorch 里手写的五步循环（清梯度→前向→损失→反向→更新）全部封装了。如果你想看底层机制，TensorFlow 也提供与 PyTorch 逐行对应的低级写法（见 4.3 节）。

### 3.2 评估与推理

```python
# 评估（对应 PyTorch 的 evaluate()）
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"测试精度: {test_acc:.4f}")

# 单张推理（对应 PyTorch 的 model.eval() + torch.no_grad()）
import numpy as np
sample = x_test[0]                        # (28,28,1)
pred = model.predict(sample[None, ...], verbose=0)   # 加 batch 维 -> (1,28,28,1)
print("预测分布:", np.round(pred[0], 3))
print("预测类别:", int(np.argmax(pred[0])), "真实标签:", int(y_test[0]))
```

## 4. 导出模型：SavedModel 与 TFLite

### 4.1 SavedModel：TensorFlow 的标准导出格式

```python
# 导出为 SavedModel 目录（包含计算图 + 权重 + 签名）
model.save('mnist_tf_savedmodel')
```

导出后是一个**目录**（不是单个文件）：

```text
mnist_tf_savedmodel/
├── saved_model.pb        # 计算图（类似 ONNX 的图描述）
├── variables/            # 权重参数
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── assets/
```

嵌入式类比：SavedModel 像一个完整的"固件包"——`saved_model.pb` 是程序代码（计算图），`variables/` 是烧录的参数表，整个目录可以原样拷贝部署。

### 4.2 TFLite：面向端侧的轻量格式

**定义 1（TFLite, TensorFlow Lite）**：TensorFlow 面向移动端/嵌入式设备的轻量推理格式与运行时。它把 SavedModel 转换成一个**单文件** `.tflite`（体积小、可量化、可在无 Python 环境运行）。

```python
# 基本转换
converter = tf.lite.TFLiteConverter.from_saved_model('mnist_tf_savedmodel')
tflite_model = converter.convert()
with open('mnist_tf.tflite', 'wb') as f:
    f.write(tflite_model)
print("tflite 大小:", len(tflite_model), "bytes")   # 约 40~50 KB

# 带动态范围量化（INT8 权重）：体积更小、推理更快
converter2 = tf.lite.TFLiteConverter.from_saved_model('mnist_tf_savedmodel')
converter2.optimizations = [tf.lite.Optimize.DEFAULT]   # 权重量化为 INT8
tflite_q = converter2.convert()
with open('mnist_tf_quant.tflite', 'wb') as f:
    f.write(tflite_q)
print("量化后大小:", len(tflite_q), "bytes")           # 约 12~15 KB
```

量化后模型只有原来 1/3 左右——这就是端侧部署的关键手段。嵌入式类比：**TFLite + 量化 ≈ 把浮点系数转成定点数烧进 MCU**，用一点点精度损失换体积和速度。

### 4.3 与 PyTorch 训练循环的逐行对比

TensorFlow 也提供"手动训练循环"的低级 API，和 PyTorch 几乎一一对应：

```python
# TensorFlow 低级训练循环（对应 PyTorch 五步）
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:        # 对应 PyTorch 的自动微分
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)  # 对应 loss.backward()
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 对应 optimizer.step()
    return loss
```

| PyTorch | TensorFlow |
|:---|:---|
| `loss.backward()` | `tape.gradient(loss, vars)` |
| `optimizer.step()` | `optimizer.apply_gradients(...)` |
| `optimizer.zero_grad()` | 自动（GradientTape 每次重建） |
| `model.train()` / `model.eval()` | `model(x, training=True/False)` |
| `torch.no_grad()` | `tf.stop_gradient` / 推理模式 |

## 5. PyTorch vs TensorFlow 全流程对比

| 环节 | PyTorch | TensorFlow（tf.keras） |
|:---|:---|:---|
| 张量类型 | `torch.Tensor` | `tf.Tensor` |
| 张量布局 | 通道在前 `(B,C,H,W)` | 通道在后 `(B,H,W,C)` |
| 数据加载 | `DataLoader`（迭代器） | `tf.data.Dataset` / 直接 numpy |
| 定义模型 | `nn.Module` 子类 + `forward()` | `Sequential` / `Model` 子类 |
| 自动微分 | `autograd`（`requires_grad`） | `GradientTape` |
| 训练循环 | 手写五步（灵活） | `model.fit`（封装） / 手写（灵活） |
| 损失函数 | `nn.CrossEntropyLoss()` | `tf.keras.losses.SparseCategoricalCrossentropy()` |
| 保存权重 | `torch.save(state_dict)` | `model.save_weights()` |
| 导出格式 | ONNX（通用） | SavedModel（自家）+ TFLite（端侧） |
| 典型场景 | 研究/论文/灵活实验 | 生产部署/端侧/移动 |

**选择建议**：学术与灵活实验多选 PyTorch（生态广、易调试）；端侧/移动部署多选 TensorFlow（TFLite 成熟）。但对算子开发者来说，两者的**底层算子几乎一致**——Conv、MatMul、Softmax 都是同样的数学，框架差异只是"怎么组织调用"，不改变"算子要服务什么负载"这个事实。

【图1：两个框架导出路径对比】

```text
PyTorch:   model ──torch.onnx.export──▶  mnist_cnn.onnx ──▶ ONNX Runtime/TensorRT
TensorFlow: model ──model.save()──────▶  SavedModel ──▶ TFLite ──▶ 移动/嵌入式设备
                      └──tf2onnx──────▶  ONNX ──▶ 统一推理引擎
```

> 图1 生图 prompt：信息图风格，白色背景。左侧两个输入节点（蓝色圆，标注 PyTorch / TensorFlow），中间两条箭头分别经过标注"torch.onnx.export"和"SavedModel + TFLite"的绿色方块，汇聚到右侧两个输出节点（橙色圆，标注 ONNX 通用格式 / 端侧部署）。下方一条虚线从 TensorFlow 分支到 ONNX，标注"tf2onnx"。扁平插画风，比例 16:9，中文标注。

## 6. 练习与里程碑

### 练习

1. **布局转换**：写一行代码把 PyTorch 的 `(B,C,H,W)` 张量转成 TensorFlow 的 `(B,H,W,C)`（提示：`tensor.permute(0,2,3,1)` 或 `np.transpose`），并验证维度。
2. **量化对比**：对比 `mnist_tf.tflite` 和 `mnist_tf_quant.tflite` 的文件大小，再用 `tf.lite.Interpreter` 加载两个模型各推理一遍测试集，比较精度差多少（通常 < 0.5%）。
3. **tf.data 数据管道**：把 `model.fit` 的输入换成 `tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64).shuffle(1000)`，体验与 DataLoader 类似的流水线。
4. **双向导出**：试试 `pip install tf2onnx`，把 SavedModel 转成 ONNX（`python -m tf2onnx.convert --saved-model mnist_tf_savedmodel --output mnist_tf.onnx`），体会"框架 → 通用格式"的统一路径。

### 里程碑自检

- [ ] 能不看文档用 `tf.keras.Sequential` 定义 CNN 并 `model.fit` 训练
- [ ] 能说出 NHWC 与 NCHW 的区别，以及两个框架各自默认哪种
- [ ] 能说明 SavedModel 目录里 `saved_model.pb` 和 `variables/` 各是什么
- [ ] 能解释 TFLite 量化为端侧部署带来什么收益
- [ ] 能画出两个框架"训练 → 导出 → 部署"的完整路径

## 7. 小结

本节用 TensorFlow 复现了 PyTorch 的全流程，并建立了框架对照：

- **tf.keras** 用 `Sequential + compile + fit` 把训练封装成三行调用，底层原理与 PyTorch 五步循环一致；
- **SavedModel** 是"计算图 + 权重"的目录格式，**TFLite** 是单文件的端侧轻量格式，量化可把模型缩小到 1/3；
- 两个框架的本质差异在**张量布局（NCHW vs NHWC）**和**导出格式**，而底层算子与计算图逻辑完全相同。

到这一步，你已经能用两大主流框架训练并导出模型。接下来要回答的问题是：导出的模型文件怎么真正高效地跑起来？ONNX 中间表示、onnxruntime、TensorRT 这些部署生态组件各扮演什么角色，与 CUDA 有什么关系——这是部署环节的核心内容。

> 🏷️ 标签：#TensorFlow #tf.keras #SavedModel #TFLite #模型量化 #框架对比
