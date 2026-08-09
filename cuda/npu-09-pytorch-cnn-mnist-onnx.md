# PyTorch 实战：从张量到训练 CNN 再到导出 ONNX

> 系列：CUDA 高性能算子实战 · NPU-09
> 前置：NPU-08（张量、神经网络、训练与推理概念）
> 配套环境：Python 3.10+、pip install torch torchvision（CPU 版即可；GPU 版需 CUDA 12.8+ 与 PyTorch ≥ 2.7，详见下文）

## 0. 本节目标

上一节我们用 NumPy 手写了一个两层神经网络，看懂了训练的本质。但真实项目不会手写矩阵乘法和反向传播——我们用深度学习框架。本节以 **PyTorch** 为主线，完成一次完整的实战闭环：

1. 环境搭建（CPU / GPU 两种方式，RTX 5060 Ti 需要 CUDA 12.8+）
2. 张量与自动微分：框架替你做了什么
3. 用 `nn.Module` 定义一个 CNN，在 MNIST 上训练到 99%+ 精度
4. 保存模型权重、导出 **ONNX** 通用格式（为部署生态篇铺路）

全程代码可复制运行。安装 CPU 版 PyTorch 即可完成本节全部实验（MNIST 很小，CPU 训练也就一两分钟）；如果你有 5060 Ti，GPU 版可以顺带验证 `torch.cuda.is_available()`。

## 1. 环境搭建

### 1.1 安装

推荐用虚拟环境隔离依赖：

```bash
python3 -m venv venv
source venv/bin/activate

# 方案 A：CPU 版（本节实验完全够用，约 200MB）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 方案 B：CUDA 12.8 版（RTX 5060 Ti / sm_120 需要 CUDA 12.8+）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ 版本说明：Blackwell 架构（sm_120）需要 PyTorch ≥ 2.7 且配套 CUDA 12.8+ 的预编译包。安装前请以 [pytorch.org/get-started](https://pytorch.org/get-started) 给出的最新命令为准，选择与本机驱动匹配的版本（驱动需 R570+）。写到这里时的稳定路线是 cu128 轮子；如果你的驱动更新到支持 CUDA 13.x，也可选择 cu130 轮子。

### 1.2 验证安装

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("算力:", torch.cuda.get_device_capability(0))   # 5060 Ti 应为 (12, 0)
```

5060 Ti 上应看到 `GPU: NVIDIA GeForce RTX 5060 Ti`、`算力: (12, 0)`。

## 2. 张量与自动微分

### 2.1 Tensor：带设备的多维数组

**定义 1（PyTorch Tensor）**：PyTorch 的核心数据结构，等价于 NumPy 的 `ndarray` + 设备属性（CPU/GPU）+ 自动求梯度能力。

```python
import torch

# 创建张量（注意：和 NumPy 一样支持各种初始化）
a = torch.zeros(2, 3)                 # (2,3) 全 0
b = torch.randn(4, 4)                 # (4,4) 标准正态随机
c = torch.tensor([[1., 2.], [3., 4.]])  # 直接给数据

# 与 NumPy 互转
import numpy as np
arr = np.random.rand(3, 3)
t = torch.from_numpy(arr)             # ndarray -> Tensor（共享内存）
back = t.numpy()                      # Tensor -> ndarray

# 张量可以在 CPU 和 GPU 之间搬运（类比：DMA 搬运）
gpu_t = c.to('cuda')                  # 搬到显存
cpu_t = gpu_t.to('cpu')               # 搬回内存
```

嵌入式类比：`ndarray` 像 `float buffer[64]`，`Tensor` 则是"带搬运引擎的 buffer"——`.to('cuda')` 就像启动一次 DMA 把数据从 SRAM 搬到显存。**数据在哪个设备上，运算就在哪个设备上执行**；CPU 张量和 GPU 张量直接运算会报错，这是新手最常见的坑。

### 2.2 自动微分：autograd

**定义 2（自动微分, Autograd）**：PyTorch 会在前向计算时自动构建计算图，记录每个操作；调用 `backward()` 时按链式法则自动算出所有 `requires_grad=True` 张量的梯度。

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x.pow(2).sum()      # y = x0^2 + x1^2 = 13
y.backward()            # 自动求导
print(x.grad)           # tensor([4., 6.])  即 dy/dx = 2x
```

嵌入式类比：这相当于一个"符号求导编译器"——你只管写前向公式，梯度由框架自动生成。上一节我们手写 `relu_grad`、`dy`、`dW1` 的一堆代码，在 PyTorch 里只需要 `loss.backward()` 一行。**理解原理（链式法则）很重要，但日常开发不需要手写梯度。**

## 3. 定义并训练一个 CNN

### 3.1 网络结构：经典 LeNet 风格 CNN

我们定义一个适合 MNIST（28×28 灰度手写数字）的小型 CNN：

```
输入 (1, 28, 28)
  └─▶ Conv2d(1→32, 3×3, padding=1) ──▶ ReLU ──▶ MaxPool2d(2)   → (32, 14, 14)
  └─▶ Conv2d(32→64, 3×3, padding=1) ──▶ ReLU ──▶ MaxPool2d(2)   → (64, 7, 7)
  └─▶ Flatten ──▶ Linear(64×7×7 → 128) ──▶ ReLU ──▶ Linear(128 → 10)
```

【图1：CNN 数据流图】

```text
(1,28,28)   (32,14,14)   (64,7,7)        (128,)      (10,)
   │            │            │              │           │
   ▼ Conv3x3    ▼ Conv3x3    ▼ Flatten      ▼ Linear    ▼ Linear
   ▼ ReLU       ▼ ReLU       ▼ Linear128    ▼ ReLU      ▼ softmax
   ▼ MaxPool    ▼ MaxPool     ──────────▶   ──────────▶  类别概率
   (特征提取阶段)              (分类阶段)
```

### 3.2 完整训练代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ---------- 1. 数据准备 ----------
# 归一化：MNIST 均值 0.1307、标准差 0.3081（官方推荐值）
transform = transforms.Compose([
    transforms.ToTensor(),                    # PIL -> Tensor，并缩放到 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

# DataLoader：按批搬运数据（类比：DMA 批量搬运，每批 64 张）
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=256, shuffle=False)

# ---------- 2. 定义网络 ----------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # (64,7,7) -> 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),                          # 10 个类别
        )

    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
print(model)
print("参数量:", sum(p.numel() for p in model.parameters()))
# 参数量约 41 万：远小于同规模全连接网络（784x128x10 就有 10 万+），
# 这就是卷积共享权重的参数效率

criterion = nn.CrossEntropyLoss()     # 交叉熵（分类）
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam 优化器

# ---------- 3. 训练一个 epoch ----------
def train_one_epoch():
    model.train()                     # 训练模式（启用 Dropout/BN 等）
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()         # 清空上一步梯度
        outputs = model(images)       # 前向
        loss = criterion(outputs, labels)  # 算损失
        loss.backward()               # 反向传播（自动微分）
        optimizer.step()              # 更新参数
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(train_loader), correct / total

# ---------- 4. 验证 ----------
def evaluate():
    model.eval()                      # 评估模式
    correct, total = 0, 0
    with torch.no_grad():             # 不需要梯度，省显存省时间
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ---------- 5. 训练 5 个 epoch ----------
for epoch in range(5):
    loss, train_acc = train_one_epoch()
    test_acc = evaluate()
    print(f"epoch {epoch+1}: loss={loss:.4f} "
          f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

# 预期输出（CPU 每个 epoch 约 10~30 秒，GPU 只需几秒）：
# epoch 1: loss=0.1832 train_acc=0.9458 test_acc=0.9786
# epoch 2: loss=0.0568 train_acc=0.9829 test_acc=0.9863
# epoch 3: loss=0.0388 train_acc=0.9885 test_acc=0.9890
# epoch 4: loss=0.0280 train_acc=0.9916 test_acc=0.9901
# epoch 5: loss=0.0227 train_acc=0.9934 test_acc=0.9910
# （不同机器/随机种子有细微差异，5 个 epoch 后测试精度应 ≥ 0.985）
```

### 3.3 训练循环五步解读

训练循环里每一批数据都执行固定的五步：

| 步骤 | 代码 | 嵌入式类比 |
|:---|:---|:---|
| 1. 清梯度 | `optimizer.zero_grad()` | 清零累加器 |
| 2. 前向 | `outputs = model(images)` | 信号流经各级处理 |
| 3. 算损失 | `loss = criterion(...)` | 计算误差 |
| 4. 反向 | `loss.backward()` | 误差逐级回传 |
| 5. 更新 | `optimizer.step()` | 调整参数 |

`model.train()` / `model.eval()` 切换的是层的行为（Dropout、BatchNorm 在训练和推理时的统计方式不同）；`torch.no_grad()` 告诉框架"这段不需要梯度"，省去构建计算图的开销——**推理阶段必须用它**，这和"推理只做前向"的结论一致。

## 4. 保存模型与导出 ONNX

### 4.1 保存权重：state_dict

```python
# 保存（推荐：只保存参数，文件小、可移植）
torch.save(model.state_dict(), 'mnist_cnn.pth')

# 加载
model2 = CNN()                      # 必须先重建结构
model2.load_state_dict(torch.load('mnist_cnn.pth', weights_only=True))
model2.to(device)
```

`state_dict` 是"参数名 → 张量"的字典。嵌入式类比：它就像固件里的参数表（权重表）——网络结构是"代码"，state_dict 是"标定数据"，两者配合才能使用。

### 4.2 导出 ONNX：模型界的"通用 ELF"

**定义 3（ONNX, Open Neural Network Exchange）**：一种开放的、与框架无关的神经网络描述格式。ONNX 文件描述了一张"计算图"：节点是算子（Conv、MatMul、Relu...），边是张量，与框架无关。

为什么要导出 ONNX？PyTorch 模型只能在 PyTorch 环境里跑；而部署场景（移动端、端侧 NPU、TensorRT）需要统一的中间格式。嵌入式类比：**ONNX 就像编译流程中的目标文件/ELF**——不同的源码语言（C/C++/Rust）编译成统一的 ELF，再由不同的链接器生成最终可执行文件；不同的训练框架导出成统一的 ONNX，再由不同的推理引擎生成最终部署模型。

```python
# 导出 ONNX（需要形状固定的"假输入" dummy）
model.eval()
dummy = torch.randn(1, 1, 28, 28, device=device)
torch.onnx.export(
    model,                      # 模型
    dummy,                      # 示例输入（决定输入形状）
    'mnist_cnn.onnx',           # 输出文件
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={              # 允许 batch 维度可变
        'input':  {0: 'batch'},
        'output': {0: 'batch'},
    },
    opset_version=17,           # ONNX 算子集版本（17 及以上稳妥）
)
print("已导出 mnist_cnn.onnx")
```

导出成功后，你会在目录下看到 `mnist_cnn.onnx`（几十 KB）。这份文件记录了完整的计算图，后续部署环节会直接查看它的结构并用推理引擎加载它。

### 4.3 从 MNIST 换到 CIFAR-10

想试试真实彩色图像（32×32×3，10 类）？只需改四处：

1. 数据集换成 `torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)`
2. 归一化参数改为 CIFAR-10 官方值：`transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))`
3. 第一层卷积 `nn.Conv2d(1, 32, ...)` → `nn.Conv2d(3, 32, ...)`（3 通道输入）
4. 池化后尺寸：32→16→8，所以 `nn.Linear(64 * 8 * 8, 128)`

CIFAR-10 比 MNIST 难，同样的结构精度约 70%，想上 85% 需要加深网络 + 数据增强（随机裁剪、翻转）——这是后续优化话题。

## 5. 练习与里程碑

### 练习

1. **改结构**：把 `MaxPool2d(2)` 去掉一层，观察参数量和精度变化；再把第一个卷积的通道数 32 改成 64，观察训练时间变化。用 `sum(p.numel() for p in model.parameters())` 确认参数变化。
2. **看计算图**：训练前打印 `model.features` 和 `model.classifier` 的每一层输出形状（可以 forward 里逐层 print），确认 28×28 如何变成 10 维向量。
3. **autograd 验证**：用 `requires_grad=True` 的张量手算 `y = (x*w + b).pow(2).mean()` 对 `w` 的梯度，再和 `y.backward()` 打印的 `w.grad` 对比，验证自动微分正确。
4. **ONNX 检查**：导出后用 Python 打印 `model.features[0].weight.shape`，确认卷积核形状为 `(32, 1, 3, 3)`——"输出通道 × 输入通道 × 核高 × 核宽"。

### 里程碑自检

- [ ] 能解释 `Tensor` 的设备和 `requires_grad` 是什么
- [ ] 能不看文档写出 `train_one_epoch` 的五步循环
- [ ] 能说清 `state_dict` 保存了什么、加载时为什么需要重建模型结构
- [ ] 能解释 ONNX 为什么被称为"模型界的 ELF"
- [ ] 亲手训练出一个测试精度 ≥ 98.5% 的 MNIST CNN，并导出 `mnist_cnn.onnx`

## 6. 小结

本节完成了 PyTorch 的实战闭环：

- **Tensor** = 带设备与自动微分能力的多维数组，`.to('cuda')` 就是 DMA 搬运；
- **autograd** = 框架自动做反向传播，`loss.backward()` 一行代替手写链式法则；
- **CNN** = Conv + ReLU + MaxPool 提取特征、Flatten + Linear 分类，训练循环是固定的五步；
- **保存与导出** = `state_dict` 存参数、ONNX 存通用计算图，为跨平台部署做好准备。

有了这份"亲手训练的模型"，后面就能回答真正的工程问题：一个训练好的模型，如何高效地跑在 GPU 上？推理引擎内部做了什么？我们优化的 CUDA kernel 在整条链路里扮演什么角色？

> 🏷️ 标签：#PyTorch #CNN #MNIST #自动微分 #ONNX导出 #深度学习实战
