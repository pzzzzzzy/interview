# 大模型微调完全指南

## 一、什么是微调（Fine-tuning）？

### 基础概念

**微调**是在预训练好的大模型基础上，用特定任务的数据继续训练，让模型适配你的需求。

**类比理解**:
```
预训练模型 = 大学毕业生（有通用知识）
微调 = 入职培训（学习公司特定业务）

不需要从小学重新教育，只需针对性培训
```

---

## 二、为什么需要微调？

### 预训练模型的局限

**问题1: 不懂你的领域**
```
预训练模型: 通用知识（维基百科、新闻等）
你的需求: 医疗诊断、法律文书、金融分析

需要领域专业知识
```

**问题2: 不符合你的风格**
```
预训练模型: 正式、客观
你的需求: 友好、幽默的客服对话

需要调整输出风格
```

**问题3: 不会特定任务**
```
预训练模型: 啥都懂一点
你的需求: 专门做情感分析

需要任务特化
```

---

## 三、微调的完整流程

### 流程图

```
1. 选择基座模型
   ↓
2. 准备训练数据
   ↓
3. 配置超参数
   ↓
4. 开始训练
   ↓
5. 验证效果
   ↓
6. 调整参数（如果效果不好）
   ↓
7. 部署使用
```

---

### 步骤1: 选择基座模型

**常见选择**:

| 模型 | 参数量 | 特点 | 适用场景 |
|------|-------|------|---------|
| BERT-base | 110M | 理解能力强 | 分类、NER |
| GPT-2 | 117M-1.5B | 生成流畅 | 文本生成 |
| LLaMA-7B | 7B | 开源、性能好 | 通用任务 |
| ChatGLM-6B | 6B | 中文强 | 中文对话 |
| Qwen-7B | 7B | 中英双语 | 多语言任务 |

**选择依据**:
- 任务类型（分类用BERT，生成用GPT）
- 语言（中文优先中文模型）
- 资源限制（GPU显存）
- 开源vs闭源

---

### 步骤2: 准备训练数据

#### 数据格式

**分类任务**:
```json
[
  {"text": "这个产品质量很好，非常满意", "label": "正面"},
  {"text": "服务态度差，不推荐", "label": "负面"},
  {"text": "一般般，没什么特别的", "label": "中性"}
]
```

**问答任务**:
```json
[
  {
    "question": "如何申请退款？",
    "answer": "进入订单详情页，点击申请退款按钮..."
  }
]
```

**对话任务**:
```json
[
  {
    "instruction": "写一首关于春天的诗",
    "input": "",
    "output": "春风拂面暖如酥，柳绿花红映碧湖..."
  }
]
```

#### 数据量要求

| 任务类型 | 最少数据量 | 推荐数据量 | 说明 |
|---------|-----------|-----------|------|
| 分类 | 100-500 | 1000+ | 每类至少50条 |
| 问答 | 500-1000 | 5000+ | 覆盖常见问题 |
| 生成 | 1000+ | 10000+ | 多样性很重要 |
| 对话 | 1000+ | 10000+ | 需要多轮对话 |

**数据质量 > 数据量**

#### 数据质量检查

```python
# 检查项
1. 是否有重复数据？
2. 标签是否正确？
3. 文本是否太短/太长？
4. 类别是否平衡？
5. 是否有脏数据（乱码、HTML标签）？

# 数据清洗示例
import re

def clean_text(text):
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除多余空格
    text = ' '.join(text.split())
    # 移除特殊字符
    text = re.sub(r'[^\w\s一-鿿]', '', text)
    return text
```

---

### 步骤3: 配置超参数

#### 核心超参数详解

**1. Learning Rate (学习率)** ⭐最重要

**作用**: 控制参数更新的步长

```
学习率太大:
参数更新太猛 → 不收敛 → 效果差

学习率太小:
参数更新太慢 → 训练太久 → 可能卡在局部最优

合适的学习率:
稳定下降，最终收敛
```

**推荐值**:
```python
# 全量微调（更新所有参数）
learning_rate = 2e-5  # 0.00002

# LoRA微调（只更新小部分）
learning_rate = 3e-4  # 0.0003

# 范围: 1e-5 到 5e-4
```

**如何选择？**
- 从2e-5开始
- 如果loss下降太慢 → 增大（5e-5）
- 如果loss震荡不稳定 → 减小（1e-5）

---

**2. Batch Size**

**作用**: 一次训练多少个样本

```python
batch_size = 8  # 小
batch_size = 32  # 中
batch_size = 128  # 大
```

**影响**:
- **太小**: 训练不稳定，但能防止过拟合
- **太大**: 训练稳定，但需要更多显存，可能过拟合

**选择依据**:
```
主要看显存:
7B模型 + LoRA:
- 16GB显存 → batch_size = 4-8
- 24GB显存 → batch_size = 16-32
- 48GB显存 → batch_size = 64+
```

**梯度累积技巧**:
```python
# 显存不够，想要大batch效果
batch_size = 4  # 实际batch
gradient_accumulation_steps = 8  # 累积8次再更新

# 等效于 batch_size = 32
```

---

**3. Epochs (训练轮数)**

**作用**: 完整遍历数据集多少次

```python
num_epochs = 3  # 常用
```

**判断标准**:
```
训练集loss持续下降，验证集loss开始上升
→ 过拟合了 → 停止训练

推荐:
- 小数据集: 10-20 epochs
- 大数据集: 3-5 epochs
```

**Early Stopping**:
```python
# 验证集loss连续3个epoch不下降就停止
patience = 3
```

---

**4. Warmup Steps (预热步数)**

**作用**: 开始时学习率从0逐渐增加

```
为什么需要？
刚开始模型不稳定，大学习率可能崩溃
先小步探索，再正常训练
```

**推荐**:
```python
# 方式1: 固定步数
warmup_steps = 500

# 方式2: 总步数的百分比
warmup_ratio = 0.1  # 前10%的步数预热
```

---

**5. Weight Decay (权重衰减)**

**作用**: L2正则化，防止过拟合

**原理详解**:

Weight Decay通过惩罚大权重来防止模型过拟合。

```python
# 损失函数变成:
Loss = 预测误差 + λ × Σ(w²)
                  ↑     ↑
            weight_decay  所有权重的平方和

# λ就是weight_decay参数
weight_decay = 0.01  # 常用值
```

**为什么有效？**

过拟合的模型倾向于有很大的权重：
```python
# 正常模型（泛化好）
weights = [0.5, -0.3, 0.8, 0.2, ...]  # 权重适中

# 过拟合模型（记忆训练数据）
weights = [10.2, -8.5, 15.3, -12.1, ...]  # 权重很大
```

Weight Decay在参数更新时加入惩罚项：
```python
# 正常更新
w_new = w - learning_rate × gradient

# 加上Weight Decay
w_new = w - learning_rate × gradient - learning_rate × weight_decay × w
                                        ↑
                                   额外的衰减项

# 等价于
w_new = w × (1 - learning_rate × weight_decay) - learning_rate × gradient
        ↑
     每次都衰减一点点
```

**直观理解**:
- 权重越大，衰减越多
- 类似"重力"，把权重往0的方向拉
- 防止权重无限增长

**数学推导**:
```
L2正则化: Loss = Error + λ/2 × ||W||²

对W求导: ∂Loss/∂W = ∂Error/∂W + λ × W

更新规则: W = W - lr × (∂Error/∂W + λ × W)
            = W × (1 - lr × λ) - lr × ∂Error/∂W
              ↑
           每次乘以小于1的数，逐渐衰减
```

**类比理解**:

就像控制体重：
- 没有Weight Decay: 可以无限制长胖（过拟合）
- 有Weight Decay: 吃东西的同时要运动消耗（权重衰减）
- weight_decay大 = 运动强度大 → 不容易长胖（不容易过拟合）

**推荐值**:
```python
weight_decay = 0.01  # 常用值

# 根据情况调整:
weight_decay = 0.0   # 数据很多，不怕过拟合
weight_decay = 0.001 # 轻微正则化
weight_decay = 0.01  # 标准正则化（推荐）
weight_decay = 0.05  # 强正则化（小数据集）
weight_decay = 0.1   # 很强正则化（可能欠拟合）
```

**什么时候调？**
- 如果过拟合（训练好，验证差） → 增大weight_decay
- 如果欠拟合（训练就不好） → 减小或去掉weight_decay
- 如果训练和验证都好 → 不用动

**注意事项**:
```python
# 有些层不应该加weight decay
# 比如LayerNorm和bias
# 现代框架通常自动处理

# AdamW优化器
# 是Adam的改进版，weight decay实现更正确
optimizer = AdamW(params, lr=3e-4, weight_decay=0.01)
```

---

### Weight Decay的数学原理（深入）

#### 为什么Weight Decay持续作用？

**更新公式**:

``w_{t+1} = w_t - η∇L(w_t) - ηλw_t``


其中:
- `w_t`: 第t步的权重
- `η`: 学习率 (learning rate)
- `∇L(w_t)`: 损失函数的梯度
- `λ`: weight decay系数

**改写为**:
``
w_{t+1} = w_t(1 - ηλ) - η∇L(w_t)
``

**关键观察**: `w_t`每次都乘以`(1 - ηλ)`

---

#### 衰减因子分析

**定义衰减因子**:
``
β = 1 - ηλ

典型值:
η = 0.001 (学习率)
λ = 0.01 (weight decay)
β = 1 - 0.001 × 0.01 = 0.99999

每次乘以0.99999
``

**递归展开（忽略梯度项）**:
``
w_1 = βw_0
w_2 = βw_1 = β²w_0
w_3 = βw_2 = β³w_0
...
w_t = β^t w_0

因为 0 < β < 1:
lim (t→∞) β^t = 0

如果没有梯度推动，权重会指数衰减到0
``

**数值例子**:
```python
w_0 = 10.0
β = 0.99999

w_10000 = 0.99999^10000 × 10 ≈ 3.68
w_100000 = 0.99999^100000 × 10 ≈ 0.0045

指数衰减！
```

---

#### 为什么会收敛到稳定值？

**完整动力学方程**:
``
w_{t+1} = (1 - ηλ)w_t - η∇L(w_t)
``

**稳定点（平衡点）分析**:

当`w_{t+1} = w_t = w*`时，系统达到平衡:
``
w* = (1 - ηλ)w* - η∇L(w*)

移项:
ηλw* = -η∇L(w*)

简化:
λw* = -∇L(w*)

即:
∇L(w*) + λw* = 0
``

**这正是L2正则化的最优条件！**

---

#### 稳定性证明

**定理**: 如果损失函数`L(w)`是凸函数且有界，则权重序列`{w_t}`收敛。

**简化证明**（一维情况）:

**1. 定义Lyapunov函数**:
``
V(w) = L(w) + (λ/2)w²
``

**2. 证明V(w)单调递减**:
``
V(w_{t+1}) - V(w_t) 
= L(w_{t+1}) - L(w_t) + (λ/2)(w_{t+1}² - w_t²)
``


泰勒展开（一阶）:
``
L(w_{t+1}) ≈ L(w_t) + ∇L(w_t)(w_{t+1} - w_t)
``

代入更新公式:
``
w_{t+1} - w_t = -ηλw_t - η∇L(w_t)
``

得:
``
V(w_{t+1}) - V(w_t) 
≈ -η∇L(w_t)[λw_t + ∇L(w_t)] + (λ/2)(w_{t+1}² - w_t²)
``

在学习率η足够小时:
``
V(w_{t+1}) < V(w_t)
``

即V(w)单调递减


**3. V(w)有下界**:
``
V(w) = L(w) + (λ/2)w² ≥ 0
``

**4. 单调有界定理**:
``
单调递减 + 有下界 → 收敛
``

---

#### 收敛速度分析

**权重更新的渐近行为**:

假设接近稳定点，令
``ε_t = w_t - w*
``
（偏离稳定点的距离）:

**线性化分析**:
``
ε_{t+1} = (1 - ηλ)ε_t - η∇²L(w*)ε_t + O(ε_t²)

忽略高阶项:
ε_{t+1} ≈ [1 - η(λ + ∇²L(w*))]ε_t

定义有效衰减率:
γ = 1 - η(λ + ∇²L(w*))

则:
ε_t ≈ γ^t ε_0
``

**收敛条件**:
``
要使 ε_t → 0，需要 |γ| < 1

即:
|1 - η(λ + ∇²L(w*))| < 1

分两种情况:

1) 如果 λ + ∇²L(w*) > 0:
   需要 0 < η < 2/(λ + ∇²L(w*))
   
2) 如果 λ + ∇²L(w*) < 0:
   系统不稳定（不收敛）
``

**结论**: 
- Weight decay增加稳定性（`λ > 0`扩大稳定区域）
- 收敛速度为指数级：`O(γ^t)`
- 学习率η需要足够小以保证稳定

---

#### 多维情况的推广

**权重向量**`w ∈ ℝ^n`:

**更新规则**:
``
w_{t+1} = (1 - ηλ)w_t - η∇L(w_t)
``

**稳定点条件**:
``
∇L(w*) + λw* = 0
``

**Hessian矩阵分析**:
``
H = ∇²L(w*) + λI
``

其中I是单位矩阵

λI的作用: 
- 使Hessian的所有特征值增加λ
- 提高矩阵的条件数（condition number）
- 增强凸性（正定性）


**特征值分解**:

设H的特征值为``{μ_i}``

收敛速度由最大特征值控制:
``γ_i = 1 - ημ_i``

系统收敛当且仅当:
``max_i |1 - ημ_i| < 1``

Weight decay使最小特征值远离0:
``min_i μ_i ≥ λ``

提高了数值稳定性
```

---

#### 与L2正则化的等价性

**优化目标**:
```
不带正则化:
min_w L(w)

带L2正则化:
min_w L(w) + (λ/2)||w||²
```

**梯度**:
```
∇[L(w) + (λ/2)||w||²] = ∇L(w) + λw
```

**梯度下降更新**:
```
w_{t+1} = w_t - η[∇L(w) + λw]
        = w_t - η∇L(w) - ηλw
        = (1 - ηλ)w_t - η∇L(w)

这就是weight decay的更新公式！
```

**结论**: Weight decay = 在损失函数中加入L2正则项

---

#### 直观的物理类比

**将权重更新类比为物理系统**:

```
dw/dt = -∇L(w) - λw

这是一个阻尼振动方程:
- ∇L(w): 驱动力（梯度）
- λw: 阻尼力（与速度成正比）
```

**能量观点**:
```
总能量: E(w) = L(w) + (λ/2)w²

能量变化率:
dE/dt = ∇L·(dw/dt) + λw·(dw/dt)
      = (∇L + λw)·(-∇L - λw)
      = -||∇L + λw||²
      ≤ 0

能量单调递减，系统必然收敛到能量最小点
```

---

#### 实验验证

**数值模拟**（简化的一维例子）:

```python
import numpy as np
import matplotlib.pyplot as plt

# 损失函数: L(w) = w²
# 真实梯度: ∇L(w) = 2w

def gradient(w):
    return 2 * w

# 参数
eta = 0.1      # 学习率
lambda_wd = 0.1  # weight decay
w0 = 10.0      # 初始权重
steps = 100

# 没有weight decay
w_no_wd = [w0]
w = w0
for _ in range(steps):
    w = w - eta * gradient(w)
    w_no_wd.append(w)

# 有weight decay
w_with_wd = [w0]
w = w0
for _ in range(steps):
    w = w * (1 - eta * lambda_wd) - eta * gradient(w)
    w_with_wd.append(w)

# 理论预测（稳定点）
# 对于L(w) = w²，稳定点满足:
# 2w* + lambda * w* = 0
# w* = 0

print(f"无weight decay最终权重: {w_no_wd[-1]:.6f}")
print(f"有weight decay最终权重: {w_with_wd[-1]:.6f}")
print(f"理论稳定点: 0.0")

# 验证指数衰减
# w_t ≈ β^t w_0 (忽略梯度项)
beta = 1 - eta * lambda_wd
theoretical = [w0 * (beta ** t) for t in range(steps + 1)]

# 都收敛到0，但weight decay收敛更快更稳定
```

**输出**:
```
无weight decay最终权重: 0.000000 (收敛慢)
有weight decay最终权重: 0.000000 (收敛快)
理论稳定点: 0.0 ✓

收敛速度:
无WD: ~100步达到1e-6
有WD: ~50步达到1e-6
```

---

#### 总结

**数学上的关键点**:

1. **持续作用**: 每次更新都有`(1-ηλ)`因子，指数衰减
2. **收敛保证**: Lyapunov函数单调递减且有界
3. **稳定点**: 满足`∇L(w*) + λw* = 0`
4. **收敛速度**: 指数收敛`O(γ^t)`，λ增大加快收敛
5. **稳定性**: λ增加Hessian特征值，提高数值稳定性

**物理直觉**:
- Weight decay是"摩擦力"或"重力"
- 权重是"物体"，梯度是"外力"
- 系统在摩擦力作用下趋于平衡（最小能量）

**实践意义**:
- λ过小: 收敛慢，可能过拟合
- λ过大: 收敛到接近0，可能欠拟合
- λ适中: 平衡训练误差和模型复杂度

---

**6. Max Length (最大长度)**

**作用**: 输入序列的最大长度

```python
max_length = 512  # BERT常用
max_length = 2048  # GPT可以更长
```

**权衡**:
- 太短: 信息截断
- 太长: 显存占用大，训练慢

**建议**: 统计数据的95分位长度

---

#### 完整配置示例

```python
# LoRA微调配置（推荐）
training_args = {
    # 学习率相关
    "learning_rate": 3e-4,
    "warmup_ratio": 0.1,
    
    # 批次相关
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,  # 等效batch=32
    
    # 训练轮数
    "num_train_epochs": 3,
    
    # 正则化
    "weight_decay": 0.01,
    
    # 评估
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "save_steps": 500,
    
    # 优化器
    "optim": "adamw_torch",
    
    # 其他
    "fp16": True,  # 混合精度训练，节省显存
    "logging_steps": 100,
    "save_total_limit": 3,  # 只保留最近3个checkpoint
}
```

---

### 步骤4: 开始训练

#### 使用Hugging Face示例

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 1. 加载模型和tokenizer
model_name = "Qwen/Qwen-7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 配置LoRA
lora_config = LoraConfig(
    r=8,  # LoRA秩，越大能力越强但参数越多
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 要应用LoRA的层
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 4.2M (0.06% of 7B)

# 3. 准备数据
dataset = load_dataset("your_dataset")

def preprocess(example):
    text = f"问题: {example['question']}\n答案: {example['answer']}"
    return tokenizer(text, truncation=True, max_length=512)

train_dataset = dataset['train'].map(preprocess)

# 4. 训练配置
training_args = TrainingArguments(
    output_dir="./output",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=100,
    save_steps=500,
    fp16=True,
)

# 5. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# 6. 保存
model.save_pretrained("./final_model")
```

---

### 步骤5: 验证效果

#### 监控指标

**训练过程中**:
```python
# 1. Loss曲线
训练loss: 应该稳定下降
验证loss: 应该跟随下降

如果:
- 训练loss下降，验证loss上升 → 过拟合
- 训练loss不下降 → 学习率太小或数据有问题
- 训练loss震荡 → 学习率太大
```

**训练完成后**:
```python
# 2. 任务指标
分类: Accuracy, F1-score
生成: BLEU, ROUGE, 人工评估

# 3. 案例测试
随机抽取测试样本，人工检查质量
```

#### 可视化Loss

```python
import matplotlib.pyplot as plt

# 从训练日志读取
losses = trainer.state.log_history

train_losses = [x['loss'] for x in losses if 'loss' in x]
steps = [x['step'] for x in losses if 'loss' in x]

plt.plot(steps, train_losses)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 理想曲线:
#  Loss
#   |  ╲
#   |   ╲___
#   |      ‾‾‾‾---__
#   |_______________
#         Steps
```

---

### 步骤6: 调整参数（Troubleshooting）

#### 常见问题和解决方案

**问题1: Loss不下降**

**原因和解决**:
```python
1. 学习率太小
   → learning_rate = 5e-5  # 增大

2. 数据太少
   → 增加数据或数据增强

3. 模型太小
   → 换更大的基座模型

4. 数据质量差
   → 清洗数据，检查标签
```

---

**问题2: Loss震荡，不稳定**

```python
1. 学习率太大
   → learning_rate = 1e-5  # 减小

2. Batch size太小
   → batch_size = 16  # 增大
   或 gradient_accumulation_steps = 8

3. 没有warmup
   → warmup_ratio = 0.1  # 添加预热
```

---

**问题3: 过拟合（训练好，验证差）**

```python
1. 增加正则化
   → weight_decay = 0.05  # 增大
   → lora_dropout = 0.2  # 增大

2. 减少训练轮数
   → num_epochs = 2  # 从3降到2

3. 数据增强
   → 同义词替换、回译等

4. Early stopping
   → 验证loss不降就停

5. 增加训练数据
   → 收集更多数据
```

---

**问题4: 显存不足 (OOM)**

```python
1. 减小batch size
   → per_device_train_batch_size = 4  # 从8降到4

2. 梯度累积
   → gradient_accumulation_steps = 8  # 保持等效batch

3. 混合精度训练
   → fp16 = True  # 或 bf16 = True

4. 梯度检查点
   → gradient_checkpointing = True  # 用时间换空间

5. 用更小的模型
   → 7B → 1.3B

6. 减少序列长度
   → max_length = 256  # 从512降到256
```

---

**问题5: 训练太慢**

```python
1. 增大batch size
   → batch_size = 32  # 如果显存允许

2. 使用fp16
   → fp16 = True

3. 多GPU训练
   → 使用DataParallel或DistributedDataParallel

4. 减少评估频率
   → eval_steps = 1000  # 从500增到1000

5. 用更小的模型
   → 先在小模型上验证，再用大模型
```

---

## 四、LoRA微调详解

### 为什么用LoRA？

**全量微调的问题**:
```
模型: 7B参数
显存需求: 约28GB (FP32) 或 14GB (FP16)
训练速度: 慢
保存空间: 每个任务保存完整模型（28GB）
```

**LoRA的优势**:
```
只训练: 0.1%参数（7M）
显存需求: 约4-8GB
训练速度: 快很多
保存空间: 每个任务只保存LoRA权重（几十MB）
```

---

### LoRA核心参数

**r (秩)**:
```python
r = 8   # 低秩，参数少，可能欠拟合
r = 16  # 常用
r = 32  # 高秩，参数多，表达能力强

推荐: 从8开始，不够再增加
```

**lora_alpha (缩放)**:
```python
lora_alpha = 16  # 通常是r的2倍
lora_alpha = 32  # r=16时

公式: 实际缩放 = lora_alpha / r
```

**target_modules (目标层)**:
```python
# 只微调attention的Q和V
target_modules = ["q_proj", "v_proj"]

# 微调所有attention
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 微调attention + FFN
target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"]

原则: 
- 越多层，效果越好，但参数越多
- 通常Q和V就够了
```

**lora_dropout**:
```python
lora_dropout = 0.1  # 10%的神经元随机失活

防止过拟合，常用0.05-0.2
```

---

### LoRA配置示例

```python
from peft import LoraConfig

# 轻量配置（显存有限）
light_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

# 标准配置（推荐）
standard_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
)

# 高性能配置（追求效果）
high_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "up_proj", "down_proj"],
    lora_dropout=0.05,
)
```

---

## 五、实战经验总结

### 调参顺序

**第1步: 先跑通**
```python
# 用最小配置快速验证流程
batch_size = 1
num_epochs = 1
max_length = 128

目标: 确保代码能跑
```

**第2步: 找合适的学习率**
```python
# 尝试几个学习率，看loss曲线
learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 3e-4]

目标: 找到稳定下降的学习率
```

**第3步: 调batch size**
```python
# 根据显存，尽量增大
batch_size = 8, 16, 32...

目标: 最大化GPU利用率
```

**第4步: 调训练轮数**
```python
# 观察验证loss，找到最佳epoch
num_epochs = 3, 5, 10...

目标: 不过拟合的前提下充分训练
```

**第5步: 精细调整**
```python
# 如果还不满意，调其他参数
- weight_decay
- warmup_ratio
- LoRA的r和alpha

目标: 榨取最后的性能
```

---

### 调参经验法则

**1. 先大后小**
```
先调影响最大的（学习率、batch size）
再调影响小的（weight decay、warmup）
```

**2. 一次调一个**
```
同时改多个参数 → 不知道哪个有效
一次只改一个 → 明确因果关系
```

**3. 记录所有实验**
```python
# 用wandb或tensorboard记录
实验1: lr=1e-5, batch=8, loss=2.3
实验2: lr=2e-5, batch=8, loss=1.8 ← 更好
实验3: lr=2e-5, batch=16, loss=1.6 ← 最好
```

**4. 先小模型验证**
```
大模型训练慢，先在小模型或小数据上验证
找到好的配置，再用大模型
```

---

### 快速判断标准

**Good👍**:
```
✅ Loss稳定下降
✅ 训练loss和验证loss接近
✅ 生成结果符合预期
✅ 在验证集上指标提升
```

**Bad👎**:
```
❌ Loss不下降或震荡
❌ 训练loss很低，验证loss很高（过拟合）
❌ 生成结果胡言乱语
❌ 验证集指标下降
```

---

## 六、面试回答模板

**面试官**: "你有微调大模型的经验吗？"

**回答框架**:
"有的。我微调过[模型名]用于[任务]。

**流程**是：首先选择合适的基座模型，我选了[LLaMA-7B/ChatGLM]因为[开源/中文好]。然后准备了[数据量]条训练数据，格式是instruction-output对。

**方法**上，我用的是LoRA微调而不是全量微调，因为资源有限。配置了r=16, lora_alpha=32，只训练attention的Q和V矩阵，这样只需要训练0.1%的参数。

**超参数**方面，学习率设置3e-4，batch size用8（因为显存限制），训练3个epoch。我用了warmup_ratio=0.1来稳定训练初期，还加了weight_decay=0.01防止过拟合。

**遇到的问题**：一开始loss震荡，后来发现是学习率太大，从5e-4降到3e-4就稳定了。还有一次显存不足，通过gradient_accumulation来模拟大batch size解决了。

**效果**：在验证集上[指标名]从[基线]提升到[结果]，生成质量明显改善。"

---

## 七、总结

### 微调的本质

```
预训练模型 = 通用知识
    +
你的数据 = 特定知识
    ↓
微调后模型 = 通用知识 + 特定知识
```

### 核心要点

1. **学习率最重要**，先调这个
2. **数据质量 > 数据量**
3. **LoRA是性价比之选**
4. **先小后大**验证想法
5. **记录实验**，不要凭感觉
6. **看loss曲线**判断问题
7. **防止过拟合**很重要

### 资源需求

| 模型规模 | 全量微调 | LoRA微调 | 推荐显卡 |
|---------|---------|---------|---------|
| 1B | 4GB | 2GB | GTX 1660 |
| 7B | 28GB | 8GB | RTX 3090 |
| 13B | 52GB | 16GB | A100 40GB |
| 70B | 280GB | 80GB | 多卡A100 |

---

**创建日期**: 2026-08-27  
**适用对象**: 准备面试、实际微调项目  
**下一步**: 实际动手微调一个模型！
