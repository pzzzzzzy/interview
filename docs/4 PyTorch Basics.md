# PyTorch深度学习基础 核心知识储备

## 一、什么是PyTorch？

### 基础定义
**PyTorch** 是一个开源的深度学习框架，由Facebook AI Research (FAIR) 开发，基于Python，用于构建和训练神经网络。

**核心特点**:
- **动态计算图**: 灵活易调试
- **Python原生**: 代码简洁直观
- **强大的GPU加速**: 自动利用GPU
- **丰富的生态**: torchvision, torchaudio等

**为什么学PyTorch？**
- NVIDIA等AI公司要求熟悉PyTorch
- 研究和工业界都广泛使用
- 是深度学习的基础工具

---

## 二、Tensor（张量）

### PyTorch安装和环境管理

#### 为什么使用Anaconda？

在安装PyTorch之前，强烈推荐使用**Anaconda**或**Miniconda**来管理Python环境。

**Anaconda的三大优势**:

1. **环境隔离**: 
   - 不会影响系统Python
   - 每个项目可以有独立的Python版本和包
   - 避免"在A项目能跑，在B项目就出错"的问题

2. **管理依赖**: 
   - 自动处理版本冲突
   - 比如项目A需要numpy 1.19，项目B需要numpy 1.21
   - Anaconda可以为每个项目创建独立环境

3. **更好的包管理**: 
   - 特别适合科学计算库（PyTorch、NumPy、Pandas等）
   - 预编译二进制包，安装更快更稳定
   - 自动处理CUDA等底层依赖

**使用Anaconda安装PyTorch**:
```bash
# 1. 创建新环境（Python 3.9）
conda create -n pytorch_env python=3.9

# 2. 激活环境
conda activate pytorch_env

# 3. 安装PyTorch（CPU版本）
conda install pytorch torchvision -c pytorch

# 或者安装GPU版本（需要NVIDIA显卡）
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 验证安装
python -c "import torch; print(torch.__version__)"
```

**不用Anaconda的安装方式（pip）**:
```bash
# 直接用pip安装（可能有依赖问题）
pip install torch torchvision
```

**推荐**: 深度学习项目优先使用Anaconda

---

### 什么是Tensor？

**Tensor** 是PyTorch中的核心数据结构，类似于NumPy的array，但可以在GPU上运行。

**简单理解**:
- 0维Tensor = 标量（一个数字）
- 1维Tensor = 向量（一维数组）
- 2维Tensor = 矩阵
- 3维及以上 = 高维张量

**例子**:
```python
import torch

# 标量 (0维)
scalar = torch.tensor(42)
print(scalar.shape)  # torch.Size([])

# 向量 (1维)
vector = torch.tensor([1, 2, 3])
print(vector.shape)  # torch.Size([3])

# 矩阵 (2维)
matrix = torch.tensor([[1, 2], [3, 4]])
print(matrix.shape)  # torch.Size([2, 2])

# 3维张量 (例如: batch_size × height × width)
tensor_3d = torch.randn(2, 3, 4)
print(tensor_3d.shape)  # torch.Size([2, 3, 4])
```

---

### Tensor创建

**从数据创建**:
```python
# 从列表创建
t1 = torch.tensor([1, 2, 3])

# 从NumPy数组创建
import numpy as np
arr = np.array([1, 2, 3])
t2 = torch.from_numpy(arr)
```

**使用特殊函数创建**:
```python
# 全零张量
zeros = torch.zeros(2, 3)  # 2x3的全零矩阵

# 全一张量
ones = torch.ones(2, 3)

# 随机张量 (0-1均匀分布)
rand = torch.rand(2, 3)

# 随机张量 (标准正态分布)
randn = torch.randn(2, 3)

# 单位矩阵
eye = torch.eye(3)

# 等差序列
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# 线性空间
linspace = torch.linspace(0, 10, 5)  # [0, 2.5, 5, 7.5, 10]
```

---

### Tensor操作

**基本运算**:
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 加法
c = a + b  # 或 torch.add(a, b)

# 减法
c = a - b

# 乘法 (逐元素)
c = a * b

# 除法
c = a / b

# 矩阵乘法
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.mm(A, B)  # 或 A @ B
```

**形状操作**:
```python
x = torch.randn(2, 3, 4)

# 查看形状
print(x.shape)  # torch.Size([2, 3, 4])
print(x.size())  # 同上

# 改变形状 (reshape)
y = x.view(2, 12)  # 变成2x12
z = x.reshape(6, 4)  # 变成6x4

# 添加维度
w = x.unsqueeze(0)  # 在第0维添加，变成[1, 2, 3, 4]

# 删除维度
u = w.squeeze(0)  # 删除第0维，变回[2, 3, 4]

# 转置
A = torch.randn(2, 3)
B = A.t()  # 转置，变成3x2
```

**索引和切片**:
```python
x = torch.randn(3, 4)

# 索引
print(x[0])      # 第一行
print(x[:, 0])   # 第一列
print(x[0, 1])   # 第一行第二列

# 切片
print(x[0:2])    # 前两行
print(x[:, 1:3]) # 所有行的第2-3列
```

---

### GPU加速

**将Tensor移到GPU**:
```python
# 检查GPU是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU可用")
else:
    device = torch.device("cpu")
    print("只能使用CPU")

# 创建Tensor并移到GPU
x = torch.randn(3, 4)
x = x.to(device)  # 移到GPU

# 或者直接在GPU上创建
y = torch.randn(3, 4, device=device)

# 从GPU移回CPU
z = x.cpu()
```

**为什么需要GPU？**
- CPU: 串行计算，快速但单线程
- GPU: 并行计算，数千个核心同时工作
- 深度学习涉及大量矩阵运算，GPU快几十倍甚至上百倍

---

## 三、自动微分 (Autograd)

### 什么是自动微分？

**Autograd** 是PyTorch的核心功能，能够自动计算梯度（导数），这是训练神经网络的关键。

**为什么需要梯度？**
- 神经网络通过梯度下降优化参数
- 梯度告诉我们参数应该朝哪个方向调整
- 手动计算梯度非常复杂，Autograd自动完成

---

### requires_grad

**开启梯度追踪**:
```python
# 创建需要梯度的Tensor
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 进行运算
y = x ** 2  # y = x的平方
z = y.sum()  # z = y的和

# 反向传播（计算梯度）
z.backward()

# 查看梯度
print(x.grad)  # dz/dx = 2x = [2.0, 4.0, 6.0]
```

**工作原理**:
```
x = [1, 2, 3]
    ↓
y = x² = [1, 4, 9]
    ↓
z = sum(y) = 14

反向传播:
dz/dy = [1, 1, 1]  (sum的导数)
dy/dx = 2x = [2, 4, 6]
dz/dx = dz/dy * dy/dx = [2, 4, 6]
```

---

### 计算图

**什么是计算图？**

计算图是记录运算过程的有向图，用于自动计算梯度。

**例子**:
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2       # y = 4
z = y + 3        # z = 7

# 计算图:
# x(2.0) → (平方) → y(4.0) → (加3) → z(7.0)
```

**动态图 vs 静态图**:

| 特性 | PyTorch (动态图) | TensorFlow 1.x (静态图) |
|------|-----------------|----------------------|
| 定义时机 | 运行时定义 | 预先定义 |
| 灵活性 | 高，可以用Python控制流 | 低 |
| 调试 | 容易，可以print | 困难 |
| 性能优化 | 较难 | 容易 |

**PyTorch动态图的优势**:
```python
# 可以使用Python的if语句
if some_condition:
    y = x * 2
else:
    y = x * 3

# 静态图中这样做很麻烦
```

---

### 梯度清零

**为什么要清零？**

PyTorch默认会**累积梯度**，如果不清零，梯度会一直叠加。

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)

# 第一次计算
y = (x ** 2).sum()
y.backward()
print(x.grad)  # [2.0, 4.0]

# 第二次计算（没有清零）
y = (x ** 2).sum()
y.backward()
print(x.grad)  # [4.0, 8.0]  累积了！

# 正确做法：清零
x.grad.zero_()
y = (x ** 2).sum()
y.backward()
print(x.grad)  # [2.0, 4.0]  正确
```

---

## 四、神经网络基础 (nn.Module)

### 什么是nn.Module？

**nn.Module** 是PyTorch中所有神经网络的基类，你的模型都应该继承它。

**最简单的神经网络**:
```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(10, 5)  # 输入10维，输出5维
        self.fc2 = nn.Linear(5, 2)   # 输入5维，输出2维

    def forward(self, x):
        # 定义前向传播
        x = self.fc1(x)
        x = torch.relu(x)  # 激活函数
        x = self.fc2(x)
        return x

# 使用
model = SimpleNet()
input_data = torch.randn(1, 10)  # batch_size=1, 特征=10
output = model(input_data)
print(output.shape)  # torch.Size([1, 2])
```

---

### 常用层

**全连接层 (Linear)**:
```python
# 输入特征: 100, 输出特征: 50
fc = nn.Linear(100, 50)

# 使用
x = torch.randn(32, 100)  # batch_size=32
y = fc(x)  # 输出: [32, 50]
```

**卷积层 (Conv2d)**:
```python
# 输入通道:3 (RGB), 输出通道:64, 卷积核:3x3
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

# 使用
x = torch.randn(32, 3, 224, 224)  # [batch, channels, height, width]
y = conv(x)  # 输出: [32, 64, 224, 224]
```

**池化层 (MaxPool2d)**:
```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2池化

x = torch.randn(32, 64, 224, 224)
y = pool(x)  # 输出: [32, 64, 112, 112]  尺寸减半
```

**Dropout**:
```python
dropout = nn.Dropout(p=0.5)  # 50%的神经元随机失活

x = torch.randn(32, 100)
y = dropout(x)  # 训练时随机置零50%，测试时不变
```

**BatchNorm**:
```python
bn = nn.BatchNorm1d(100)  # 对100维特征做归一化

x = torch.randn(32, 100)
y = bn(x)
```

---

### 激活函数

**为什么需要激活函数？**

没有激活函数，多层网络等价于单层（线性叠加还是线性）。

**常用激活函数**:

**ReLU (最常用)**:
```python
relu = nn.ReLU()
x = torch.tensor([-1.0, 0.0, 1.0])
y = relu(x)  # [0.0, 0.0, 1.0]

# 或直接使用函数
y = torch.relu(x)
```

**公式**: `f(x) = max(0, x)`

**优点**: 计算简单，缓解梯度消失  
**缺点**: 可能出现"神经元死亡"

---

**Sigmoid**:
```python
sigmoid = nn.Sigmoid()
x = torch.tensor([-1.0, 0.0, 1.0])
y = sigmoid(x)  # [0.27, 0.5, 0.73]
```

**公式**: `f(x) = 1 / (1 + e^(-x))`

**输出范围**: (0, 1)  
**用途**: 二分类输出层

---

**Tanh**:
```python
tanh = nn.Tanh()
x = torch.tensor([-1.0, 0.0, 1.0])
y = tanh(x)  # [-0.76, 0, 0.76]
```

**输出范围**: (-1, 1)

---

**Softmax**:
```python
softmax = nn.Softmax(dim=1)
x = torch.randn(2, 3)  # 2个样本，3个类别
y = softmax(x)  # 每行和为1
```

**用途**: 多分类输出层

**Softmax公式详解**:

对于向量 z = [z₁, z₂, ..., zₙ]，Softmax将其转为概率分布：

```
Softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
```

**计算步骤**:
1. 对每个值取指数: e^z₁, e^z₂, ..., e^zₙ
2. 求和: sum = e^z₁ + e^z₂ + ... + e^zₙ
3. 归一化: 每个值除以sum

**例子**:
```python
z = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(z, dim=0)
print(probs)  # tensor([0.6590, 0.2424, 0.0986])
print(probs.sum())  # tensor(1.0000)  和为1

# 手动计算验证:
# e^2.0 = 7.39, e^1.0 = 2.72, e^0.1 = 1.11
# sum = 11.22
# P(y=1) = 7.39 / 11.22 = 0.659
```

**Softmax特点**:
- 输出范围: (0, 1)
- 所有输出和为1（概率分布）
- 保持大小关系（输入大→输出概率大）
- 平滑（不像argmax那样硬选择）

---

## 五、损失函数 (Loss Function)

### 什么是损失函数？

**损失函数**衡量模型预测与真实值的差距，训练的目标是最小化损失。

---

### 常用损失函数

**MSE Loss (均方误差)**:

用于回归问题

```python
criterion = nn.MSELoss()

# 预测值和真实值
pred = torch.tensor([2.5, 3.0, 4.2])
target = torch.tensor([3.0, 3.0, 4.0])

loss = criterion(pred, target)
print(loss)  # 平均平方误差
```

**公式**: `Loss = mean((pred - target)²)`

---

**CrossEntropyLoss (交叉熵)**:

用于分类问题

```python
criterion = nn.CrossEntropyLoss()

# 模型输出 (未经softmax的logits)
logits = torch.randn(3, 5)  # 3个样本，5个类别

# 真实标签 (类别索引)
targets = torch.tensor([1, 0, 4])

loss = criterion(logits, targets)
```

**注意**: CrossEntropyLoss内部包含Softmax，所以模型输出不要加Softmax

---

**BCELoss (二元交叉熵)**:

用于二分类

```python
criterion = nn.BCELoss()

# 预测概率 (需要先经过Sigmoid)
pred = torch.tensor([0.9, 0.3, 0.7])
target = torch.tensor([1.0, 0.0, 1.0])

loss = criterion(pred, target)
```

---

## 六、优化器 (Optimizer)

### 什么是优化器？

**优化器**负责更新模型参数，使损失函数最小化。

---

### 常用优化器

**SGD (随机梯度下降)**:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练步骤
optimizer.zero_grad()  # 清零梯度
loss.backward()        # 计算梯度
optimizer.step()       # 更新参数
```

**更新公式**: `θ = θ - lr * ∇θ`

**lr**: 学习率，控制更新步长

---

**SGD with Momentum**:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

**优势**: 加速收敛，减少震荡

---

**Adam (最常用)**:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001  # Adam的学习率通常较小
)
```

**优势**:
- 自适应学习率
- 收敛快
- 鲁棒性好

**适用**: 大多数情况的首选

---

**AdamW**:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2正则化
)
```

**优势**: Adam + 更好的权重衰减

---

### 学习率调整

**为什么要调整学习率？**

- 开始时学习率大，快速收敛
- 后期学习率小，精细调整

**StepLR**:
```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,  # 每10个epoch
    gamma=0.1      # 学习率乘以0.1
)

# 训练循环中
for epoch in range(100):
    train(...)
    scheduler.step()  # 更新学习率
```

**ReduceLROnPlateau**:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5  # 5个epoch不降就减小学习率
)

# 使用
scheduler.step(val_loss)  # 根据验证损失调整
```

---

## 七、训练循环

### 完整的训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
train_loader = ...  # 数据加载器

# 2. 定义模型
model = MyModel()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 4.1 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 4.2 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        # 4.3 记录损失
        running_loss += loss.item()
    
    # 每个epoch结束
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
```

---

### 训练模式 vs 评估模式

**为什么要区分？**

某些层（Dropout, BatchNorm）在训练和测试时行为不同

```python
# 训练模式
model.train()
# - Dropout生效
# - BatchNorm更新统计量

# 评估模式
model.eval()
# - Dropout不生效
# - BatchNorm使用固定的统计量

# 评估时还要禁用梯度计算
with torch.no_grad():
    output = model(test_data)
```

---

## 八、模型保存和加载

### 保存整个模型

```python
# 保存
torch.save(model, 'model.pth')

# 加载
model = torch.load('model.pth')
```

**缺点**: 依赖模型类定义

---

### 只保存参数 (推荐)

```python
# 保存
torch.save(model.state_dict(), 'model_weights.pth')

# 加载
model = MyModel()  # 先创建模型
model.load_state_dict(torch.load('model_weights.pth'))
```

---

### 保存检查点

**什么是检查点(Checkpoint)？**

检查点是训练过程中的"存档点"，保存了完整的训练状态，包括：
- 模型参数
- 优化器状态（动量等）
- 当前epoch
- 当前损失

**为什么需要检查点？**

1. **防止训练中断**: 服务器断电、程序崩溃时不用从头开始
2. **长时间训练**: 训练几天甚至几周的大模型，可以随时暂停/继续
3. **实验管理**: 可以回到任何训练阶段

**保存检查点**:
```python
# 保存完整训练状态
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_acc': best_acc  # 可以保存任何你需要的信息
}
torch.save(checkpoint, 'checkpoint.pth')
```

**加载检查点并继续训练**:
```python
# 1. 创建模型和优化器
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 2. 加载检查点
checkpoint = torch.load('checkpoint.pth')

# 3. 恢复状态
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
loss = checkpoint['loss']

# 4. 继续训练
model.train()  # 设置为训练模式
for epoch in range(start_epoch, total_epochs):
    # 从start_epoch继续训练
    train_one_epoch(...)
```

**完整的训练循环示例（带检查点）**:
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

start_epoch = 0
best_acc = 0.0

# 如果存在检查点，加载它
import os
if os.path.exists('checkpoint.pth'):
    print("找到检查点，加载中...")
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
    best_acc = checkpoint['best_acc']
    print(f"从epoch {start_epoch}继续训练，当前最佳准确率: {best_acc:.2f}%")

# 训练循环
for epoch in range(start_epoch, 100):
    model.train()
    running_loss = 0.0
    
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # 验证
    val_acc = validate(model, val_loader)
    
    # 如果是最佳模型，保存检查点
    if val_acc > best_acc:
        best_acc = val_acc
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'loss': running_loss / len(train_loader)
        }
        torch.save(checkpoint, 'checkpoint.pth')
        print(f"保存检查点: epoch {epoch}, acc {val_acc:.2f}%")
    
    print(f"Epoch {epoch}: Loss {running_loss:.4f}, Val Acc {val_acc:.2f}%")
```

**检查点的使用场景**:

1. **定期保存**: 每N个epoch保存一次
```python
if epoch % 10 == 0:  # 每10个epoch保存
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
```

2. **保存最佳模型**: 只保存验证集上表现最好的
```python
if val_acc > best_acc:
    torch.save(checkpoint, 'best_model.pth')
```

3. **保存最后N个检查点**: 避免占用太多磁盘
```python
# 只保留最近3个检查点
checkpoints = ['ckpt_1.pth', 'ckpt_2.pth', 'ckpt_3.pth']
torch.save(checkpoint, checkpoints[epoch % 3])
```

**注意事项**:

1. **优化器状态很重要**: 不加载优化器状态，momentum等信息会丢失
2. **设备匹配**: 如果在GPU上保存，在CPU上加载需要特殊处理
```python
# CPU加载GPU保存的模型
checkpoint = torch.load('checkpoint.pth', map_location='cpu')
```
3. **随机数状态**: 如果需要完全复现，还要保存随机数种子
```python
checkpoint['rng_state'] = torch.get_rng_state()
```

---

## 九、PyTorch vs TensorFlow

| 特性 | PyTorch | TensorFlow |
|------|---------|-----------|
| 计算图 | 动态 | 静态(1.x)/动态(2.x) |
| 易用性 | 简单直观 | 较复杂 |
| 调试 | 容易 | 困难(1.x) |
| 部署 | 相对难 | 容易(TFLite, TF Serving) |
| 社区 | 研究界主流 | 工业界也广泛使用 |
| 生态 | torchvision等 | 完整的端到端方案 |

**什么时候用PyTorch？**
- 研究、实验
- 需要灵活性
- Python开发者友好

**什么时候用TensorFlow？**
- 生产部署
- 移动端/嵌入式
- Google生态

---

## 十、常见概念解释

### Batch, Epoch, Iteration

**Batch**: 一次训练使用的样本数
```
例如: 总样本1000个，batch_size=100
→ 每次训练用100个样本
```

**Epoch**: 遍历完整个数据集一次
```
1 epoch = 1000 / 100 = 10个batch
```

**Iteration**: 一次参数更新
```
1 iteration = 1个batch的前向+反向传播
```

---

### 过拟合 (Overfitting)

**现象**: 训练集表现好，测试集表现差

**原因**: 模型记住了训练数据的细节和噪音

**解决方法**:
1. **更多数据**
2. **Dropout**: 随机失活神经元
3. **正则化**: L1/L2惩罚
4. **Early Stopping**: 验证集不再提升就停止
5. **数据增强**: 图像旋转、翻转等

---

### 欠拟合 (Underfitting)

**现象**: 训练集和测试集都表现差

**原因**: 模型太简单，学不到数据的模式

**解决方法**:
1. **增加模型复杂度**: 更多层、更多神经元
2. **训练更久**: 更多epoch
3. **降低正则化强度**
4. **特征工程**: 更好的特征

---

## 十一、面试高频问题及回答

### Q1: PyTorch的核心优势是什么？

**回答框架**:
"PyTorch的核心优势是动态计算图，让代码更灵活易调试。与TensorFlow 1.x的静态图不同，PyTorch在运行时构建计算图，可以使用Python的控制流，调试时可以直接print。另外PyTorch的API设计很Pythonic，学习曲线平缓。这些特点使它成为研究界的首选框架。"

---

### Q2: 什么是计算图？动态图和静态图有什么区别？

**回答框架**:
"计算图记录运算过程，用于自动计算梯度。静态图在运行前定义好，优化空间大但不灵活；动态图在运行时构建，灵活但优化难。PyTorch用动态图，可以用if/for等控制流，调试方便；TensorFlow 1.x用静态图，需要先define再run，部署优化好但开发体验差。TensorFlow 2.x已默认动态图。"

---

### Q3: 什么是Autograd？它是如何工作的？

**回答框架**:
"Autograd是PyTorch的自动微分引擎，能自动计算梯度。工作原理是：对设置了requires_grad=True的Tensor，PyTorch会构建计算图记录所有运算。调用backward()时，从输出开始反向遍历计算图，用链式法则计算每个参数的梯度。这样我们不用手动推导复杂的梯度公式，PyTorch自动完成。"

---

### Q4: 训练神经网络的完整流程是什么？

**回答框架**:
"完整流程是：1) 前向传播 - 输入数据通过模型得到预测；2) 计算损失 - 用损失函数衡量预测与真实的差距；3) 反向传播 - 调用loss.backward()计算梯度；4) 更新参数 - 优化器根据梯度更新参数；5) 清零梯度 - 为下一次迭代准备。这个过程重复多个epoch直到模型收敛。"

---

### Q5: 常用的优化器有哪些？如何选择？

**回答框架**:
"最常用的是Adam，它结合了动量和自适应学习率，收敛快且鲁棒，是多数情况的首选。SGD配合momentum也很常用，训练时间长但最终效果可能更好，适合已有经验的调参。AdamW是Adam的改进版，权重衰减更合理。学习率通常Adam用0.001，SGD用0.01-0.1。实际中先试Adam，不行再试其他。"

---

### Q6: 什么是过拟合？如何解决？

**回答框架**:
"过拟合是模型在训练集表现好但测试集差，说明模型记住了训练数据的噪音而不是真正的模式。解决方法有：增加数据、使用Dropout随机失活神经元、L2正则化惩罚大权重、Early Stopping在验证集不再提升时停止、数据增强增加样本多样性。实际中通常组合使用多种方法。"

---

### Q7: 为什么需要激活函数？

**回答框架**:
"没有激活函数，多层神经网络等价于单层，因为线性函数的组合还是线性的。激活函数引入非线性，让网络能拟合复杂函数。ReLU最常用，计算简单且缓解梯度消失；Sigmoid和Tanh用于特定场景，Sigmoid输出0-1适合概率，Tanh输出-1到1；Softmax用于多分类输出层。"

---

### Q8: Batch Size如何选择？

**回答框架**:
"Batch size是权衡速度和性能的参数。太小（如1-8）梯度噪音大、训练慢但泛化好；太大（如1024+）训练快、显存占用大但可能泛化差。常用32-256。GPU显存允许的话，选择能整除样本数的值。也可以用梯度累积模拟大batch。实际中先试32或64，再根据显存和效果调整。"

---

## 十二、核心概念速记卡

### PyTorch = Tensor + Autograd + nn.Module

**Tensor**: 数据容器，可GPU加速  
**Autograd**: 自动计算梯度  
**nn.Module**: 构建神经网络

---

### 训练流程

**前向 → 损失 → 反向 → 更新 → 清零**

---

### 常用组合

**分类任务**: CrossEntropyLoss + Adam  
**回归任务**: MSELoss + Adam  
**激活函数**: ReLU (隐藏层) + Softmax (输出层)

---

### 防止过拟合

**Dropout + L2正则 + Early Stopping + 数据增强**

---

## 十三、学习检查清单

完成以下自测，确保理解：

- [ ] 能创建和操作Tensor
- [ ] 理解requires_grad和backward的作用
- [ ] 能用nn.Module定义简单神经网络
- [ ] 知道常用的激活函数和使用场景
- [ ] 理解损失函数的作用
- [ ] 能写完整的训练循环
- [ ] 理解过拟合和欠拟合
- [ ] 能保存和加载模型
- [ ] 理解动态图vs静态图
- [ ] 知道如何使用GPU加速

---

## 十四、扩展阅读（可选）

### 推荐资源

**官方资源**:
- PyTorch官方60分钟入门
- PyTorch官方文档
- PyTorch Examples (GitHub)

**深入学习**:
- 《深度学习》(Goodfellow)
- CS231n课程 (Stanford)
- Fast.ai课程

**实践平台**:
- Kaggle竞赛
- PyTorch Tutorials
- Papers with Code

---

## 十五、深度学习任务类型全览

### 有监督学习 vs 无监督学习

#### 核心区别：有没有"标准答案"

**有监督学习 (Supervised Learning)**:
- 训练数据有**标签**（正确答案）
- 目标是**预测标签**
- 损失函数：预测 vs 真实标签

**无监督学习 (Unsupervised Learning)**:
- 训练数据**没有标签**
- 目标是**发现规律/模式**
- 损失函数：重建误差或其他规则

#### 判断方法三步法

**第1步：看训练数据**
```python
# 有监督：数据 = (输入, 标签)配对
train_data = [(图片1, "猫"), (图片2, "狗")]

# 无监督：数据 = 只有输入
train_data = [图片1, 图片2]
```

**第2步：看目标**
- 有监督："告诉我这是什么" → 预测标签
- 无监督："这些数据有什么规律" → 发现模式

**第3步：看损失函数**
```python
# 有监督
loss = criterion(prediction, true_label)  # 和标签比

# 无监督
loss = criterion(reconstructed, original)  # 和原始输入比
```

#### 快速判断表

| 问题 | 答案 | 学习类型 |
|------|------|---------|
| 训练数据有标签吗？ | 有 | 有监督 ✅ |
| 训练数据有标签吗？ | 没有 | 无监督 ✅ |
| 目标是预测某个值？ | 是 | 有监督 ✅ |
| 目标是发现模式/分组？ | 是 | 无监督 ✅ |
| 损失函数用到标签？ | 用 | 有监督 ✅ |

#### 类比理解

**有监督学习 = 有老师**
```
老师: "这是苹果" ← 告诉标准答案
学生: 学会认苹果
```

**无监督学习 = 自己探索**
```
一堆水果，没人告诉你哪个是什么
你自己发现: "红色的好像一类，黄色的好像一类"
```

---

### 1. 监督学习任务

#### 1.1 图像分类 (Image Classification)

**任务**: 给图片打标签

**例子**: 识别猫/狗，识别手写数字

**网络**: CNN (卷积神经网络)

**输入**: 图片 (H×W×C)  
**输出**: 类别标签  
**损失**: CrossEntropyLoss

**应用**: 人脸识别、医疗影像诊断、自动驾驶

---

#### 1.2 目标检测 (Object Detection)

**任务**: 找出图片中物体的位置和类别

**例子**: 在照片中框出所有人和车

**网络**: YOLO, Faster R-CNN, SSD

**输入**: 图片  
**输出**: 边界框 + 类别标签  
**损失**: 分类损失 + 边界框回归损失

**应用**: 自动驾驶、安防监控、无人机

---

#### 1.3 语义分割 (Semantic Segmentation)

**任务**: 给图片每个像素打标签

**例子**: 区分图片中哪些像素是人、哪些是背景

**网络**: U-Net, FCN, DeepLab

**输入**: 图片  
**输出**: 和输入同尺寸的标签图  
**损失**: CrossEntropyLoss (像素级)

**应用**: 医疗影像分割、自动驾驶场景理解

---

#### 1.4 文本分类 (Text Classification)

**任务**: 给文本打标签

**例子**: 情感分析（正面/负面）、垃圾邮件检测

**网络**: RNN, LSTM, Transformer

**输入**: 文本序列  
**输出**: 类别标签  
**损失**: CrossEntropyLoss

**应用**: 舆情分析、内容审核、客服分类

---

#### 1.5 文本生成 (Text Generation) ⭐你想做的

**任务**: 生成新文本（写文章、对话、翻译）

**例子**: GPT写作、机器翻译、诗歌生成

**网络**: RNN, LSTM, Transformer, GPT

**输入**: 提示词/上文  
**输出**: 生成的文本  
**损失**: CrossEntropyLoss (下一个词预测)

**应用**: ChatGPT、机器翻译、自动写作

**特点**: 自回归生成，一个词一个词地生成

---

#### 1.6 序列标注 (Sequence Labeling)

**任务**: 给序列中每个元素打标签

**例子**: 命名实体识别（NER）、词性标注

**网络**: BiLSTM-CRF, BERT

**输入**: 文本序列  
**输出**: 每个词的标签  
**损失**: CRF Loss 或 CrossEntropy

**应用**: 信息抽取、知识图谱构建

---

#### 1.7 回归 (Regression)

**任务**: 预测连续值

**例子**: 房价预测、股票预测、年龄预测

**网络**: 全连接神经网络、CNN、RNN

**输入**: 特征向量或序列  
**输出**: 连续数值  
**损失**: MSELoss, L1Loss

**应用**: 金融预测、推荐系统评分

---

### 2. 无监督学习任务

#### 2.1 聚类 (Clustering)

**任务**: 将数据分组

**例子**: 客户分群、图像聚类

**网络**: AutoEncoder + K-means

**应用**: 数据分析、异常检测

---

#### 2.2 降维 (Dimensionality Reduction)

**任务**: 减少特征维度

**例子**: 数据可视化、特征提取

**网络**: AutoEncoder, VAE

**应用**: 数据压缩、可视化

---

### 3. 生成模型

#### 3.1 图像生成 (Image Generation)

**任务**: 生成新图片

**例子**: 生成人脸、艺术作品

**网络**: GAN, VAE, Diffusion Model

**输入**: 随机噪声或文本描述  
**输出**: 图片  

**应用**: Midjourney、DALL-E、Stable Diffusion

---

#### 3.2 文本生成 (Text Generation)

**任务**: 生成文本（同上1.5，但这里强调无条件生成）

**网络**: GPT, LSTM

**应用**: 创意写作、内容生成

---

#### 3.3 语音合成 (Speech Synthesis)

**任务**: 文字转语音

**网络**: Tacotron, WaveNet

**应用**: 智能音箱、有声读物

---

### 4. 强化学习

#### 4.1 游戏AI

**任务**: 学习玩游戏

**例子**: AlphaGo、Dota2 AI

**网络**: DQN, PPO, A3C

**应用**: 游戏、机器人控制

---

#### 4.2 机器人控制

**任务**: 控制机器人行动

**网络**: Actor-Critic

**应用**: 自动驾驶、工业机器人

---

### 5. 多模态学习

#### 5.1 图文匹配 (Image-Text Matching)

**任务**: 判断图片和文字是否匹配

**网络**: CLIP, ALIGN

**应用**: 图片搜索、内容审核

---

#### 5.2 图像描述生成 (Image Captioning)

**任务**: 为图片生成文字描述

**网络**: CNN + LSTM, Vision Transformer

**输入**: 图片  
**输出**: 描述文本  

**应用**: 辅助视觉障碍者、图片标注

---

## 十六、任务类型对比表

| 类型 | 输入 | 输出 | 网络类型 | 损失函数 | 难度 |
|------|------|------|---------|---------|------|
| 图像分类 | 图片 | 类别 | CNN | CrossEntropy | ⭐ 入门 |
| 目标检测 | 图片 | 框+类别 | YOLO/R-CNN | 混合损失 | ⭐⭐⭐ |
| 语义分割 | 图片 | 像素标签 | U-Net | CrossEntropy | ⭐⭐⭐ |
| 文本分类 | 文本 | 类别 | RNN/LSTM | CrossEntropy | ⭐⭐ |
| **文本生成** | 提示词 | 文本 | LSTM/GPT | CrossEntropy | ⭐⭐⭐ |
| 回归预测 | 特征 | 数值 | 全连接 | MSE | ⭐ 最简单 |
| 图像生成 | 噪声 | 图片 | GAN/VAE | 对抗/重构 | ⭐⭐⭐⭐ |

---

## 十七、文本生成模型详解

### 什么是文本生成？

**文本生成**是根据输入（提示词/上文）生成新文本的任务，属于**序列到序列(Seq2Seq)**问题。

### 工作原理

**自回归生成**: 一个词一个词地生成

```
输入: "今天天气"
↓
模型预测下一个词: "很"
↓
输入变成: "今天天气很"
↓
模型预测下一个词: "好"
↓
输入变成: "今天天气很好"
↓
...继续直到生成结束标记
```

### 核心技术

1. **词嵌入 (Embedding)**: 将词转为向量
2. **循环神经网络 (RNN/LSTM)**: 处理序列
3. **注意力机制 (Attention)**: 关注重要信息
4. **Transformer**: 现代LLM的基础（GPT、BERT）

### 训练目标

**下一个词预测**: 给定前N个词，预测第N+1个词

```
训练数据: "我爱北京天安门"

训练样本:
输入: "我"          → 目标: "爱"
输入: "我爱"        → 目标: "北京"
输入: "我爱北京"    → 目标: "天安门"
```

### 文本生成的应用

1. **写作助手**: 续写文章、写邮件
2. **聊天机器人**: ChatGPT、Claude
3. **机器翻译**: 中文→英文
4. **代码生成**: GitHub Copilot
5. **诗歌创作**: 自动作诗

### 和图像分类的区别

| 特性 | 图像分类 | 文本生成 |
|------|---------|---------|
| 输出类型 | 固定类别 | 可变长度序列 |
| 生成方式 | 一次性 | 逐词生成 |
| 网络类型 | CNN | RNN/LSTM/Transformer |
| 核心挑战 | 识别特征 | 保持连贯性 |

---

**最后更新**: 2026-08-27

**下一步**: 完成理论学习后，动手用PyTorch构建你的第一个神经网络！
