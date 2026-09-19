# Encoder-Decoder架构深度解析

## 一、什么是Encoder和Decoder？

### 1.1 基本概念

**Encoder（编码器）**：
- 将输入序列转换为连续的表示（向量）
- 提取输入的语义信息
- 输出是**固定长度或可变长度的向量表示**

**Decoder（解码器）**：
- 根据Encoder的输出和已生成的内容，生成目标序列
- 是一个**自回归**生成过程
- 逐个生成输出token

### 1.2 经典应用场景

```
机器翻译：
输入: "I love AI"  →  [Encoder]  →  编码表示  →  [Decoder]  →  输出: "我爱AI"

文本摘要：
输入: 长文本  →  [Encoder]  →  压缩表示  →  [Decoder]  →  输出: 摘要

问答系统：
输入: 问题+文档  →  [Encoder]  →  理解  →  [Decoder]  →  输出: 答案
```

### 1.3 架构演化历史

```
2014: Seq2Seq (RNN-based)
      ↓
2017: Transformer (Attention is All You Need)
      Encoder-Decoder架构
      ↓
2018: BERT (Encoder-only)
      GPT (Decoder-only)
      ↓
2019-2020: T5, BART (Encoder-Decoder)
           GPT-2, GPT-3 (Decoder-only)
      ↓
2022-2026: 主流LLM都是Decoder-only
           (GPT-4, Claude, LLaMA, Gemini等)
```

## 二、Encoder详解

### 2.1 Encoder的结构

**Transformer Encoder Block**：

```
输入 Token Embeddings + Positional Encoding
    ↓
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention          │ ← 关键：Self-Attention
│  (每个token关注所有其他token)        │
└─────────────────────────────────────┘
    ↓
Add & LayerNorm (残差连接 + 层归一化)
    ↓
┌─────────────────────────────────────┐
│  Feed-Forward Network                │
│  (两层全连接，中间ReLU/GELU)          │
└─────────────────────────────────────┘
    ↓
Add & LayerNorm
    ↓
输出（传给下一个Encoder层或Decoder）
```

**堆叠多层**：
- BERT-base: 12层Encoder
- BERT-large: 24层Encoder
- 每层都执行相同的操作，但参数不同

### 2.2 Encoder中的Self-Attention

**核心特点**：每个token可以看到**所有**其他token（双向注意力）

**计算过程**：

```python
# 输入: X = [batch_size, seq_len, d_model]

# 1. 计算Q, K, V
Q = X @ W_Q  # [batch, seq_len, d_k]
K = X @ W_K  # [batch, seq_len, d_k]
V = X @ W_V  # [batch, seq_len, d_v]

# 注意：Q, K, V都来自同一个输入X（所以叫Self-Attention）

# 2. 计算注意力分数
scores = Q @ K.T / sqrt(d_k)  # [batch, seq_len, seq_len]

# 3. Softmax归一化
attention_weights = softmax(scores, dim=-1)

# 4. 加权求和
output = attention_weights @ V  # [batch, seq_len, d_v]
```

**关键点**：
- **Q, K, V都来自输入X**（Self-Attention的定义）
- **没有Mask**：每个位置都能看到所有位置
- **双向的**：既能看前面的词，也能看后面的词

**示例**：编码"I love AI"

```
Token    关注的词（权重）
I     →  I(0.6), love(0.3), AI(0.1)
love  →  I(0.2), love(0.5), AI(0.3)
AI    →  I(0.1), love(0.3), AI(0.6)
```
每个词都能关注到所有词，获得全局上下文。

### 2.3 Encoder的优势

1. **双向上下文**：理解能力强
2. **并行计算**：可以同时处理所有token
3. **适合理解任务**：分类、实体识别、情感分析

### 2.4 Encoder的代表模型

- **BERT**：预训练语言理解模型
- **RoBERTa**：BERT的改进版
- **ALBERT**：轻量化BERT
- **DeBERTa**：解耦注意力

## 三、Decoder详解

### 3.1 Decoder的结构

**Transformer Decoder Block**：

```
输入 Token Embeddings + Positional Encoding
    ↓
┌─────────────────────────────────────────┐
│  Masked Multi-Head Self-Attention       │ ← 第1个注意力层
│  (只能看到当前和之前的token)             │
│  Q, K, V 都来自Decoder自己的输入         │
└─────────────────────────────────────────┘
    ↓
Add & LayerNorm
    ↓
┌─────────────────────────────────────────┐
│  Cross-Attention                        │ ← 第2个注意力层（不同！）
│  Q来自Decoder上一层输出                  │
│  K, V来自Encoder输出                    │
└─────────────────────────────────────────┘
    ↓
Add & LayerNorm
    ↓
┌─────────────────────────────────────────┐
│  Feed-Forward Network                    │
└─────────────────────────────────────────┘
    ↓
Add & LayerNorm
    ↓
输出（传给下一个Decoder层或输出层）
```

**重要！Decoder有两个不同的注意力层**：
1. **Masked Self-Attention**（第1层）：Q、K、V都来自Decoder自己
2. **Cross-Attention**（第2层）：Q来自Decoder，K/V来自Encoder
3. **Feed-Forward Network**：位置无关的非线性变换

### 3.2 Decoder中的Masked Self-Attention（因果注意力）

**关键：这是Decoder的第1个注意力层**

**核心特点**：只能看到**当前位置及之前**的token（单向注意力）

#### **Masked Self-Attention vs Cross-Attention 对比**

```
Decoder Block内部的数据流：

输入: decoder_input (已生成的序列)
    ↓
┌────────────────────────────────────────────┐
│ 第1层: Masked Self-Attention               │
│                                            │
│ Q = decoder_input @ W_Q  ← 来自decoder     │
│ K = decoder_input @ W_K  ← 来自decoder     │
│ V = decoder_input @ W_V  ← 来自decoder     │
│                                            │
│ 这是Self-Attention! 自己关注自己！          │
└────────────────────────────────────────────┘
    ↓
decoder_hidden (第1层的输出)
    ↓
┌────────────────────────────────────────────┐
│ 第2层: Cross-Attention                     │
│                                            │
│ Q = decoder_hidden @ W_Q  ← 来自decoder    │
│ K = encoder_output @ W_K  ← 来自encoder！   │
│ V = encoder_output @ W_V  ← 来自encoder！   │
│                                            │
│ 这是Cross-Attention! Decoder问Encoder答！  │
└────────────────────────────────────────────┘
    ↓
输出到Feed-Forward层
```

**总结区别**：
- **Masked Self-Attention**: Decoder内部自己关注自己（Q、K、V都是decoder_input）
- **Cross-Attention**: Decoder去查询Encoder（Q来自decoder，K/V来自encoder）

#### **为什么需要Mask？**

```
生成过程是自回归的：
时间步1: 生成 token_1 (只能看<start>)
时间步2: 生成 token_2 (只能看<start>, token_1)
时间步3: 生成 token_3 (只能看<start>, token_1, token_2)
...

训练时为了并行化，一次性输入所有token，但必须用Mask防止"看到未来"！
```

#### **Causal Mask（因果掩码）的实现**

```python
# 输入: X = [batch, seq_len, d_model]
# 例如生成 "I love AI"

# 1. 计算Q, K, V（和Encoder一样）
Q = X @ W_Q  # 来自Decoder的输入
K = X @ W_K  # 来自Decoder的输入
V = X @ W_V  # 来自Decoder的输入

# 注意：Masked Self-Attention中，Q, K, V都来自Decoder自己的输入

# 2. 计算注意力分数
scores = Q @ K.T / sqrt(d_k)  # [batch, seq_len, seq_len]

# 3. 应用Causal Mask
#    创建下三角矩阵，上三角设为-inf
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))

# Mask矩阵示例（seq_len=4）：
# [[0,   -inf, -inf, -inf],   ← 位置0只能看位置0
#  [0,    0,   -inf, -inf],   ← 位置1能看位置0,1
#  [0,    0,    0,   -inf],   ← 位置2能看位置0,1,2
#  [0,    0,    0,    0  ]]   ← 位置3能看位置0,1,2,3

# 4. Softmax（-inf变成0）
attention_weights = softmax(scores, dim=-1)

# 5. 加权求和
output = attention_weights @ V
```

#### **具体示例：生成"I love AI"**

**训练时的Mask效果**：

```
目标序列: <start> I love AI <end>

位置0 (<start>): 只能看 <start>
位置1 (I):       只能看 <start>, I
位置2 (love):    只能看 <start>, I, love
位置3 (AI):      只能看 <start>, I, love, AI
位置4 (<end>):   只能看 <start>, I, love, AI, <end>

注意力矩阵（下三角）：
         <s>  I   love  AI  <e>
<s>      [1   0    0    0   0 ]
I        [x   1    0    0   0 ]
love     [x   x    1    0   0 ]
AI       [x   x    x    1   0 ]
<e>      [x   x    x    x   1 ]

其中x表示有权重（softmax后的值），0表示被mask掉
```

**推理时的自回归过程**：

```
步骤1: 输入<start> → 生成"I"
步骤2: 输入<start> I → 生成"love"
步骤3: 输入<start> I love → 生成"AI"
步骤4: 输入<start> I love AI → 生成<end>
```

#### **QKV来自哪里？**

**Masked Self-Attention中**：
```
Q = Decoder当前层的输入 @ W_Q
K = Decoder当前层的输入 @ W_K
V = Decoder当前层的输入 @ W_V

所有的Q, K, V都来自Decoder自己！
```

### 3.3 Decoder中的Cross-Attention（交叉注意力）

**核心作用**：让Decoder关注Encoder的输出（源序列信息）

#### **Cross-Attention的计算**

```python
# encoder_output: Encoder的输出 [batch, src_len, d_model]
# decoder_hidden: Decoder的Masked Self-Attention输出 [batch, tgt_len, d_model]

# 关键：Q, K, V来自不同的地方！
Q = decoder_hidden @ W_Q     # Query来自Decoder
K = encoder_output @ W_K     # Key来自Encoder
V = encoder_output @ W_V     # Value来自Encoder

# 计算注意力
scores = Q @ K.T / sqrt(d_k)  # [batch, tgt_len, src_len]

# 没有Mask！Decoder可以看Encoder的所有位置
attention_weights = softmax(scores, dim=-1)

output = attention_weights @ V  # [batch, tgt_len, d_model]
```

#### **QKV来自哪里？**

**Cross-Attention中**：
```
Q (Query):  来自Decoder（"我想问什么？"）
K (Key):    来自Encoder（"源序列有什么信息？"）
V (Value):  来自Encoder（"具体的信息内容"）

这就是为什么叫"Cross"（跨）Attention！
```

#### **直观理解Cross-Attention**

**机器翻译示例**：英译中 "I love AI" → "我爱AI"

```
Encoder输出: [I的表示, love的表示, AI的表示]
Decoder生成: "我"

Cross-Attention过程：
生成"我"时:
  Query("我"的隐状态) 去问:
    - "I"有多相关？     → 权重0.7
    - "love"有多相关？  → 权重0.2
    - "AI"有多相关？    → 权重0.1
  
  加权组合Encoder的输出，帮助生成"我"

生成"爱"时:
  Query("爱"的隐状态) 去问:
    - "I"有多相关？     → 权重0.1
    - "love"有多相关？  → 权重0.8
    - "AI"有多相关？    → 权重0.1
  
  主要关注"love"，帮助生成"爱"
```

**注意力可视化**：

```
生成目标     关注源序列的权重
"我"    →   I(0.7), love(0.2), AI(0.1)
"爱"    →   I(0.1), love(0.8), AI(0.1)
"AI"    →   I(0.1), love(0.1), AI(0.8)
```

### 3.4 完整的Decoder流程图

```
机器翻译: "I love AI" → "我爱AI"

┌─────────────────────┐
│   Encoder           │
│   输入: I love AI    │
│   输出: encoder_out  │ ──┐
└─────────────────────┘   │
                          │
                          ├── 提供给Cross-Attention
                          │
┌─────────────────────┐   │
│   Decoder           │   │
├─────────────────────┤   │
│ 输入: <start> 我 爱  │   │
│       ↓             │   │
│ Masked Self-Attn    │   │  (因果注意力，只看左边)
│  Q,K,V来自Decoder   │   │
│       ↓             │   │
│ Cross-Attention  ←──┼───┘  (Q来自Decoder, K,V来自Encoder)
│       ↓             │
│ Feed-Forward        │
│       ↓             │
│ 输出: 我 爱 AI       │
└─────────────────────┘
```

### 3.5 Decoder的代表模型

**Encoder-Decoder架构**：
- **T5**：Text-to-Text Transfer Transformer
- **BART**：去噪自编码器
- **mBART**：多语言BART

**Decoder-only架构**：
- **GPT系列**：GPT-2, GPT-3, GPT-4
- **LLaMA**：Meta的开源模型
- **Claude**：Anthropic的模型
- **PaLM**：Google的大模型

## 四、三种注意力机制对比

### 4.1 对比表格

| 注意力类型 | Q来源 | K来源 | V来源 | Mask | 用途 |
|-----------|-------|-------|-------|------|------|
| **Encoder Self-Attention** | Encoder输入 | Encoder输入 | Encoder输入 | 无 | 理解输入序列 |
| **Decoder Masked Self-Attention** | Decoder输入 | Decoder输入 | Decoder输入 | 因果Mask | 生成时只看已生成部分 |
| **Cross-Attention** | Decoder隐状态 | Encoder输出 | Encoder输出 | 无 | 连接源序列和目标序列 |

### 4.2 详细对比

#### **Encoder Self-Attention**
```python
# 输入序列: "I love AI"
X_encoder = embed("I love AI")

Q = X_encoder @ W_Q  # 所有来自Encoder
K = X_encoder @ W_K
V = X_encoder @ W_V

# 无Mask，双向注意力
attention_scores = softmax(Q @ K.T / sqrt(d_k))
output = attention_scores @ V

# 每个词都能看到所有词
```

#### **Decoder Masked Self-Attention**
```python
# 已生成序列: "<start> 我 爱"
X_decoder = embed("<start> 我 爱")

Q = X_decoder @ W_Q  # 所有来自Decoder
K = X_decoder @ W_K
V = X_decoder @ W_V

# 有因果Mask，单向注意力
scores = Q @ K.T / sqrt(d_k)
mask = causal_mask(seq_len)  # 下三角
scores = scores.masked_fill(mask, -inf)
attention_scores = softmax(scores)
output = attention_scores @ V

# "爱"只能看到"<start>"和"我"，看不到未来
```

#### **Cross-Attention**
```python
# Decoder隐状态: 正在生成"AI"
decoder_hidden = masked_self_attention_output

# Encoder输出: "I love AI"的编码
encoder_output = encoder("I love AI")

Q = decoder_hidden @ W_Q      # 来自Decoder
K = encoder_output @ W_K      # 来自Encoder
V = encoder_output @ W_V      # 来自Encoder

# 无Mask，可以看Encoder的所有位置
attention_scores = softmax(Q @ K.T / sqrt(d_k))
output = attention_scores @ V

# Decoder的每个位置都能关注整个输入序列
```

## 五、为什么现代LLM都是Decoder-only？

### 5.1 历史演变

**早期（2017-2018）**：
- Transformer原论文：Encoder-Decoder架构（机器翻译）
- BERT：Encoder-only（理解任务）
- GPT：Decoder-only（生成任务）

**中期（2019-2020）**：
- T5、BART：Encoder-Decoder（多任务）
- GPT-2、GPT-3：Decoder-only规模扩大

**现在（2022-2026）**：
- **几乎所有LLM都是Decoder-only**：
  - GPT-4、Claude、LLaMA、Gemini、文心一言、通义千问

### 5.2 为什么Decoder-only成为主流？

#### **原因1：统一架构，简化设计**

**Encoder-Decoder的复杂性**：
- 需要设计两个不同的网络
- Cross-Attention增加了复杂度
- 训练和推理都更复杂

**Decoder-only的简洁性**：
- 只有一种组件：Masked Self-Attention + FFN
- 架构统一，易于扩展和优化
- 代码实现更简单

```python
# Encoder-Decoder: 需要维护两个网络
encoder = TransformerEncoder(num_layers=12)
decoder = TransformerDecoder(num_layers=12)  # 还有cross-attention

# Decoder-only: 只需一个网络
model = TransformerDecoder(num_layers=24)  # 统一结构
```

#### **原因2：Decoder-only可以做Encoder的所有事**

**关键发现**：Decoder通过因果注意力也能理解文本

**原理**：
```
Encoder方式理解: "I love AI"
- 同时看到所有词，双向理解

Decoder方式理解: "I love AI"
- 虽然是因果的（I → love → AI）
- 但最后一个token"AI"的表示已经融合了前面所有信息
- 最后一层的最后一个token ≈ Encoder的[CLS] token
```

**实践验证**：
- GPT-3在理解任务（分类、QA）上也表现优秀
- 不需要专门的双向Encoder

#### **原因3：预训练目标更自然**

**Encoder-Decoder的预训练**：
- T5：需要设计特殊的去噪任务（span corruption）
- BART：需要多种噪声策略
- 复杂的预训练设计

**Decoder-only的预训练**：
- **简单的语言建模**：预测下一个token
- 目标非常自然：P(token_t | token_1, ..., token_{t-1})
- 与人类语言的生成过程一致

```python
# Decoder-only预训练（极其简单）
loss = cross_entropy(
    model(text[:-1]),  # 输入: 前n-1个token
    text[1:]           # 目标: 后n-1个token（shift一位）
)

# 就是简单的"预测下一个词"！
```

#### **原因4：In-Context Learning的涌现**

**发现**：Decoder-only模型展现出强大的上下文学习能力

**Few-shot Learning**：
```
输入Prompt:
"翻译任务
英文: Hello → 中文: 你好
英文: Thank you → 中文: 谢谢
英文: I love AI → 中文: "

模型会自动学会翻译，无需Cross-Attention！
```

**原理**：
- Decoder的因果注意力在处理长上下文时
- 自然地学会了"从示例中学习模式"
- 不需要显式的Encoder来"理解"任务

#### **原因5：扩展性更好**

**模型规模扩展**：
```
Encoder-Decoder:
- 需要同时扩展Encoder和Decoder
- Cross-Attention的计算复杂度: O(src_len × tgt_len)
- 内存和计算成本都更高

Decoder-only:
- 只扩展一个网络
- 注意力复杂度: O(seq_len²)
- 可以使用各种优化（Flash Attention等）
```

**实际数据**：
- GPT-3 175B：Decoder-only
- LLaMA 70B：Decoder-only
- Claude：Decoder-only
- 扩展到千亿参数更容易

#### **原因6：生成质量更好**

**实践发现**：
- Decoder-only在生成任务上表现更好
- 生成的文本更流畅、连贯
- Encoder-Decoder有时会产生不一致

**推测原因**：
- Decoder从头到尾都是自回归的
- 训练和推理完全一致
- Encoder-Decoder的Cross-Attention引入了额外的依赖

#### **原因7：长文本处理更自然**

**Decoder-only的优势**：
```
输入: 10000 token的文档 + "总结一下"
处理方式: 
- 整个文档 + 指令作为一个序列
- 用因果注意力逐步处理
- 生成总结

Encoder-Decoder:
- 需要将10000 token送入Encoder
- Encoder输出10000个表示
- Decoder通过Cross-Attention访问
- 内存和计算都更大
```

**长度扩展**：
- Decoder-only更容易扩展到更长上下文
- GPT-4: 32k-128k tokens
- Claude: 200k tokens
- 只需优化一种注意力机制

### 5.3 Encoder-Decoder还有优势吗？

**仍然适用的场景**：

#### **1. 显式的序列到序列转换**
```
语音识别: 音频序列 → 文本序列
  - 输入输出模态不同
  - Encoder处理音频，Decoder生成文本

机器翻译（严格对齐）:
  - 需要精确的源-目标对齐
  - Cross-Attention提供显式对齐信息
```

#### **2. 编辑任务**
```
文本纠错: 错误句子 → 正确句子
  - Encoder编码错误
  - Decoder修正生成

代码修复: 有bug的代码 → 修复后的代码
```

#### **3. 需要显式对齐的任务**
```
神经机器翻译中的对齐:
  - 可以可视化Cross-Attention
  - 看到每个目标词对应的源词

解释性需求:
  - Cross-Attention提供显式的对应关系
```

### 5.4 总结对比

| 维度 | Encoder-Decoder | Decoder-only |
|------|----------------|--------------|
| **架构复杂度** | 高（两个网络+Cross-Attn） | 低（统一结构） |
| **预训练** | 复杂（需要设计任务） | 简单（语言建模） |
| **理解能力** | 强（双向） | 强（通过上下文） |
| **生成能力** | 好 | 更好 |
| **扩展性** | 较难 | 容易 |
| **长文本** | 复杂 | 自然 |
| **In-Context Learning** | 弱 | 强 |
| **代表模型** | T5, BART | GPT, Claude, LLaMA |
| **当前趋势** | 下降 | 主流 |

## 六、因果注意力的深度解析

### 6.1 什么是因果性（Causality）？

**定义**：当前时刻只能依赖过去，不能依赖未来

**在语言模型中**：
```
生成第t个词时，只能看到:
- 第1个词
- 第2个词
- ...
- 第t-1个词
- 第t个词本身

不能看到第t+1, t+2, ... 个词
```

**为什么叫"因果"**：
- 过去是"因"，现在是"果"
- 未来不能影响现在（时间箭头）
- 符合真实世界的因果律

### 6.2 因果注意力的数学实现

#### **完整的计算过程**

```python
import torch
import torch.nn.functional as F

def causal_self_attention(x, W_q, W_k, W_v):
    """
    因果自注意力的完整实现
    
    Args:
        x: 输入 [batch_size, seq_len, d_model]
        W_q, W_k, W_v: 查询、键、值的权重矩阵
    
    Returns:
        output: 注意力输出 [batch_size, seq_len, d_model]
    """
    batch_size, seq_len, d_model = x.shape
    
    # 1. 计算Q, K, V（都来自x）
    Q = x @ W_q  # [batch, seq_len, d_k]
    K = x @ W_k  # [batch, seq_len, d_k]
    V = x @ W_v  # [batch, seq_len, d_v]
    
    d_k = Q.shape[-1]
    
    # 2. 计算注意力分数
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    # scores: [batch, seq_len, seq_len]
    
    # 3. 应用因果掩码
    # 创建下三角矩阵（对角线及以下为True）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    
    # 可视化mask（seq_len=4）：
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    
    # 将上三角（未来位置）设为-inf
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    
    # 4. Softmax（-inf会变成0）
    attention_weights = F.softmax(scores, dim=-1)
    # attention_weights: [batch, seq_len, seq_len]
    
    # 5. 加权求和
    output = attention_weights @ V
    # output: [batch, seq_len, d_v]
    
    return output, attention_weights
```

#### **具体数值示例**

假设输入序列"I love AI"（简化为3个token）：

```python
# 输入嵌入（简化，每个词3维）
x = torch.tensor([
    [1.0, 0.5, 0.2],  # I
    [0.8, 1.0, 0.3],  # love
    [0.3, 0.7, 1.0],  # AI
])  # [3, 3]

# 简化：Q = K = V = x（省略权重矩阵）
Q = K = V = x

# 计算注意力分数
scores = Q @ K.T
# scores = [[1.29, 1.43, 0.77],
#           [1.43, 1.73, 1.01],
#           [0.77, 1.01, 1.58]]

# 应用因果掩码
causal_mask = torch.tril(torch.ones(3, 3))
# [[1, 0, 0],
#  [1, 1, 0],
#  [1, 1, 1]]

scores_masked = scores.masked_fill(causal_mask == 0, float('-inf'))
# scores_masked = [[1.29,  -inf,  -inf],
#                  [1.43,  1.73,  -inf],
#                  [0.77,  1.01,  1.58]]

# Softmax（每行独立）
attention_weights = F.softmax(scores_masked, dim=-1)
# [[1.0,  0.0,  0.0],    ← "I"只能看"I"
#  [0.42, 0.58, 0.0],    ← "love"看"I"和"love"
#  [0.21, 0.26, 0.53]]   ← "AI"看所有三个词

# 最终输出
output = attention_weights @ V
# output[0] = 1.0 * V[0] = [1.0, 0.5, 0.2]  (只包含"I"的信息)
# output[1] = 0.42*V[0] + 0.58*V[1]         (融合"I"和"love")
# output[2] = 0.21*V[0] + 0.26*V[1] + 0.53*V[2]  (融合所有信息)
```

### 6.3 因果注意力的可视化

#### **注意力矩阵可视化**

```
生成序列: The cat sat on the mat

注意力矩阵（每行是一个query，每列是key）:
         The  cat  sat  on   the  mat
The      [1.0  0    0    0    0    0  ]  ← 只看自己
cat      [0.3  0.7  0    0    0    0  ]  ← 看The(30%) cat(70%)
sat      [0.1  0.3  0.6  0    0    0  ]  ← 主要看sat
on       [0.1  0.2  0.3  0.4  0    0  ]  ← 分散注意力
the      [0.2  0.1  0.2  0.3  0.2  0  ]  
mat      [0.1  0.1  0.2  0.2  0.2  0.2]  ← 均匀关注所有

下三角模式：未来全是0
```

#### **与Encoder Self-Attention对比**

```
Encoder Self-Attention（无mask）:
         The  cat  sat  on   the  mat
The      [0.2  0.3  0.1  0.1  0.2  0.1]  ← 可以看到"mat"
cat      [0.3  0.4  0.1  0.1  0.05 0.05] ← 可以看到后面
sat      [0.1  0.3  0.3  0.2  0.05 0.05]
...
全矩阵都有值！双向的！

Decoder Causal Attention（有mask）:
         The  cat  sat  on   the  mat
The      [1.0  0    0    0    0    0  ]
cat      [0.3  0.7  0    0    0    0  ]
sat      [0.1  0.3  0.6  0    0    0  ]
...
严格的下三角！单向的！
```

### 6.4 为什么训练时也要用Causal Mask？

**问题**：训练时我们已经有了完整的目标序列，为什么不能看到未来？

**答案**：为了让训练和推理一致！

#### **如果训练时不用Mask会怎样？**

```python
# 错误的训练方式（无mask）
输入: "The cat sat on the mat"
目标: "cat sat on the mat <end>"

训练时，生成"cat"的位置能看到整句话：
- 模型会学会"作弊"：直接从输入中复制"cat"
- 因为它能看到未来的"sat on the mat"

推理时，生成"cat"的位置只能看到"The":
- 模型懵了！从来没在这种条件下训练过
- 性能崩溃（train-test mismatch）
```

#### **正确的训练方式（有mask）**

```python
输入: "The cat sat on the mat"
目标: "cat sat on the mat <end>"

训练时强制使用Causal Mask:
- 生成"cat"时只能看"The"
- 生成"sat"时只能看"The cat"
- ...

推理时也是一样的条件:
- 生成"cat"时只能看"The"
- 生成"sat"时只能看"The cat"

训练和推理一致！模型表现稳定！
```

### 6.5 因果注意力的实现细节

#### **方法1：Mask矩阵（标准实现）**

```python
def create_causal_mask(seq_len):
    # 方法1: torch.tril
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask == 0  # True的位置会被mask掉

# 方法2: 手动创建
def create_causal_mask_manual(seq_len):
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:  # 未来位置
                mask[i, j] = 1
    return mask.bool()

# 方法3: 向量化
def create_causal_mask_vectorized(seq_len):
    row_indices = torch.arange(seq_len).unsqueeze(1)
    col_indices = torch.arange(seq_len).unsqueeze(0)
    mask = col_indices > row_indices
    return mask
```

#### **方法2：直接置零（不推荐）**

```python
# 不推荐：在softmax之前置零会导致数值不稳定
scores[:, :, upper_triangle] = 0  # 错误！
attention_weights = F.softmax(scores, dim=-1)

# 正确：在softmax之前置为-inf
scores = scores.masked_fill(mask, float('-inf'))
attention_weights = F.softmax(scores, dim=-1)
# softmax(-inf) = 0，数值稳定
```

#### **方法3：注册为buffer（GPT实现）**

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 注册mask为buffer（不是参数，但会被保存）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                  .view(1, 1, config.max_seq_len, config.max_seq_len)
        )
    
    def forward(self, x):
        B, T, C = x.size()
        
        # 使用预先创建的mask
        scores = ...  # 计算注意力分数
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, 
            float('-inf')
        )
        ...
```

### 6.6 因果注意力的变体

#### **1. Sliding Window Attention（滑动窗口）**

```python
# 限制注意力窗口大小（减少计算）
window_size = 512

# 不是看到所有历史，而是只看最近的window_size个token
# 例如位置1000只能看[1000-512, 1000]

mask = torch.tril(torch.ones(seq_len, seq_len))
# 再加一个上限
for i in range(seq_len):
    mask[i, :max(0, i-window_size)] = 0

# Longformer, BigBird使用这种方法处理长文本
```

#### **2. Prefix LM（前缀语言模型）**

```python
# 前半部分可以双向注意力（理解），后半部分因果注意力（生成）

# 输入: "翻译: Hello World →" (前缀)  "你好 世界" (生成部分)
#       <------ 双向 ------>            <---- 因果 ---->

prefix_len = 5
mask = torch.ones(seq_len, seq_len)

# 前缀部分：全1（双向）
mask[:prefix_len, :prefix_len] = 1

# 生成部分对前缀：全1（可以看到整个前缀）
mask[prefix_len:, :prefix_len] = 1

# 生成部分内部：下三角（因果）
mask[prefix_len:, prefix_len:] = torch.tril(
    torch.ones(seq_len - prefix_len, seq_len - prefix_len)
)

# UniLM, GLM使用这种方法
```

## 七、面试高频问题

### 7.1 基础概念题

**Q1: 解释Encoder和Decoder的区别。**

参考答案：
- **Encoder**：使用双向Self-Attention，每个token可以看到所有其他token，适合理解任务（分类、NER）。输出是输入序列的向量表示。
- **Decoder**：使用因果（Masked）Self-Attention，每个token只能看到之前的token，适合生成任务。输出是自回归生成的序列。
- **核心区别**：Encoder是双向的（并行处理），Decoder是单向的（顺序生成）。

**Q2: 什么是Cross-Attention？Q、K、V分别来自哪里？**

参考答案：
Cross-Attention是Decoder中连接Encoder和Decoder的机制：
- **Q (Query)**：来自Decoder的隐状态（"我想问什么？"）
- **K (Key)**：来自Encoder的输出（"源序列有什么？"）
- **V (Value)**：来自Encoder的输出（"具体内容是什么？"）

这样Decoder在生成每个token时，可以关注到输入序列的相关部分。例如翻译时，生成中文"爱"会主要关注英文"love"。

**Q3: 详细解释Decoder中的Masked Self-Attention，为什么需要Mask？**

参考答案：
**为什么需要Mask？**
- Decoder是自回归生成的：生成第t个token时只能依赖前t-1个token
- 如果训练时能看到未来，模型会"作弊"，导致训练和推理不一致

**如何实现？**
- 创建下三角Mask矩阵
- 将上三角（未来位置）的注意力分数设为-inf
- Softmax后这些位置权重变为0

**Q、K、V来源？**
- Q、K、V都来自Decoder自己的输入（Self-Attention）
- 与Cross-Attention不同，那里K、V来自Encoder

**Q4: 为什么现在的大语言模型都是Decoder-only？**

参考答案：
主要原因：
1. **架构简化**：只需一种组件，不需要Cross-Attention
2. **预训练简单**：直接用语言建模（预测下一个词），无需设计复杂任务
3. **统一能力**：Decoder通过因果注意力也能理解文本（最后一个token的表示包含全文信息）
4. **In-Context Learning**：展现出强大的少样本学习能力，不需要专门的Encoder
5. **扩展性好**：更容易扩展到千亿参数和长上下文
6. **生成质量高**：在生成任务上表现更优

实践证明GPT、Claude、LLaMA等Decoder-only模型在理解和生成任务上都很强，不需要Encoder。

### 7.2 技术深度题

**Q5: 对比三种注意力机制：Encoder Self-Attention、Decoder Masked Self-Attention、Cross-Attention。**

参考答案：

| 维度 | Encoder Self-Attn | Decoder Masked Self-Attn | Cross-Attention |
|------|-------------------|--------------------------|-----------------|
| Q来源 | Encoder输入 | Decoder输入 | Decoder隐状态 |
| K来源 | Encoder输入 | Decoder输入 | Encoder输出 |
| V来源 | Encoder输入 | Decoder输入 | Encoder输出 |
| Mask | 无 | 因果Mask（下三角） | 无 |
| 方向性 | 双向 | 单向（只看过去） | 全局（看所有源） |
| 用途 | 理解输入 | 生成时保持因果性 | 连接源和目标 |

**关键区别**：
- Self-Attention: Q、K、V来自同一来源
- Cross-Attention: Q、K、V来自不同来源

**Q6: 详细描述因果注意力的实现过程，包括数学公式和代码。**

参考答案：
```python
# 1. 计算Q, K, V（都来自Decoder输入x）
Q = x @ W_Q  # [batch, seq_len, d_k]
K = x @ W_K
V = x @ W_V

# 2. 计算注意力分数
scores = (Q @ K.T) / sqrt(d_k)  # [batch, seq_len, seq_len]

# 3. 创建因果mask
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
# [[1, 0, 0],    下三角为1
#  [1, 1, 0],
#  [1, 1, 1]]

# 4. 应用mask（上三角设为-inf）
scores = scores.masked_fill(causal_mask == 0, float('-inf'))

# 5. Softmax（-inf变成0）
attn_weights = softmax(scores, dim=-1)

# 6. 加权求和
output = attn_weights @ V
```

**关键**：
- 必须用-inf而不是0，保证softmax数值稳定
- Mask确保位置i只能看到位置≤i的信息
- 训练和推理都要用mask，保持一致性

**Q7: 为什么训练Decoder时也需要Causal Mask？**

参考答案：
**原因：训练和推理的一致性**

**如果训练时不用Mask**：
```
训练：输入完整序列"The cat sat"
- 生成"cat"时能看到"sat"（未来信息）
- 模型学会利用未来信息

推理：逐个生成
- 生成"cat"时看不到"sat"
- 条件不一致，性能下降（train-test mismatch）
```

**用Mask的好处**：
```
训练：强制使用Causal Mask
- 生成"cat"时只能看"The"
- 模型学会在受限信息下生成

推理：自然地只能看历史
- 生成"cat"时只能看"The"
- 条件一致，模型表现稳定
```

这叫做**Teacher Forcing + Causal Masking**，是标准做法。

**Q8: Encoder-Decoder架构在哪些场景下仍然优于Decoder-only？**

参考答案：

**优势场景**：
1. **不同模态的序列转换**：
   - 语音识别（音频→文本）
   - 图像字幕（图像→文本）
   - Encoder处理特殊模态，Decoder生成文本

2. **需要显式对齐的任务**：
   - 机器翻译中的词对齐可视化
   - Cross-Attention提供可解释的对应关系

3. **严格的编辑任务**：
   - 文本纠错：错误句子→正确句子
   - 代码修复：buggy code→fixed code
   - Encoder编码原始，Decoder生成修正

4. **输入输出长度差异大**：
   - 文档摘要：长文档→短摘要
   - Encoder压缩长输入，Decoder生成短输出

**现状**：
- 纯文本任务：Decoder-only主导
- 多模态/特殊任务：Encoder-Decoder仍有价值
- 趋势：Decoder-only通过特殊设计也在侵入这些领域

### 7.3 实践应用题

**Q9: 如何在实际项目中选择Encoder-only、Decoder-only还是Encoder-Decoder？**

参考答案：

**Encoder-only（如BERT）**：
- **任务**：分类、实体识别、情感分析、语义相似度
- **特点**：需要双向理解，不需要生成
- **优势**：理解能力强，训练快
- **缺点**：不能生成文本
- **何时选**：纯理解任务，已有标注数据

**Decoder-only（如GPT）**：
- **任务**：文本生成、对话、翻译、问答、代码生成
- **特点**：统一架构，既能理解也能生成
- **优势**：通用性强，少样本学习
- **缺点**：理解任务可能稍弱于BERT
- **何时选**：需要生成，或需要通用模型

**Encoder-Decoder（如T5）**：
- **任务**：翻译、摘要、多模态转换
- **特点**：显式分离理解和生成
- **优势**：对齐可视化，特定任务可能更优
- **缺点**：架构复杂，训练成本高
- **何时选**：严格的seq2seq任务，需要对齐信息

**2026年实践建议**：
- 默认选Decoder-only（如GPT、LLaMA）
- 除非有特殊需求（多模态、对齐可视化）

**Q10: 实现一个简单的Transformer Decoder，需要注意哪些关键点？**

参考答案：

**关键组件**：
```python
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        # 1. Token Embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # 3. Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads) 
            for _ in range(num_layers)
        ])
        
        # 4. 输出投影
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 5. 注册Causal Mask（关键！）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
        )
    
    def forward(self, idx):
        B, T = idx.shape
        
        # Token + Position embedding
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # 通过各层
        for layer in self.layers:
            x = layer(x, self.causal_mask[:T, :T])
        
        # 输出logits
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Masked Self-Attention
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        
        # Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        # Self-Attention + 残差
        x = x + self.attn(self.ln1(x), mask)
        # FFN + 残差
        x = x + self.ffn(self.ln2(x))
        return x
```

**关键注意点**：
1. **Causal Mask必须正确**：注册为buffer，在attention中使用
2. **残差连接**：每个子层都要加残差
3. **LayerNorm位置**：Pre-LN（先norm再attention）更稳定
4. **权重共享**：token_embed和输出head可以共享权重
5. **位置编码**：可学习或固定的sinusoidal

**Q11: 如何优化Transformer Decoder的推理速度？**

参考答案：

**问题**：自回归生成慢，每生成一个token需要一次前向传播

**优化方法**：

**1. KV Cache（最重要）**
```python
# 问题：每步都重新计算历史token的K, V
# t=1: 计算K_0, V_0
# t=2: 重新计算K_0, V_0, K_1, V_1  ← 浪费！
# t=3: 重新计算K_0, V_0, K_1, V_1, K_2, V_2

# 解决：缓存历史的K, V
class CausalSelfAttentionWithCache(nn.Module):
    def forward(self, x, cache=None):
        Q = x @ self.W_q  # 只计算当前token的Q
        K = x @ self.W_k
        V = x @ self.W_v
        
        if cache is not None:
            # 拼接历史K, V
            K = torch.cat([cache['K'], K], dim=1)
            V = torch.cat([cache['V'], V], dim=1)
        
        # 计算attention
        scores = Q @ K.T / sqrt(d_k)
        # ...
        
        # 更新cache
        new_cache = {'K': K, 'V': V}
        return output, new_cache

# 效果：从O(n²)降到O(n)，速度提升显著
```

**2. 批处理（Batching）**
```python
# 同时生成多个序列
batch_size = 32
# 利用GPU并行性，提升吞吐量
```

**3. 模型优化**
- **量化**：FP32→FP16/INT8
- **剪枝**：移除不重要的权重
- **蒸馏**：用小模型模仿大模型

**4. 注意力优化**
- **Flash Attention**：优化内存访问模式
- **Multi-Query Attention**：多个head共享K, V
- **Grouped-Query Attention**：折中方案

**5. 推理引擎**
- **vLLM**：优化的LLM推理引擎
- **TensorRT-LLM**：NVIDIA的推理加速
- **Text Generation Inference**：HuggingFace的推理服务

**实际效果**：
- 无优化：10 tokens/s
- KV Cache：50 tokens/s
- + 量化 + Flash Attention：200+ tokens/s

## 八、总结与学习建议

### 8.1 核心知识点回顾

**必须掌握**：
1. ✅ Encoder的双向Self-Attention（Q、K、V都来自输入）
2. ✅ Decoder的因果Self-Attention（Masked，只看历史）
3. ✅ Cross-Attention（Q来自Decoder，K/V来自Encoder）
4. ✅ 为什么现代LLM是Decoder-only
5. ✅ 因果Mask的实现和作用

**深入理解**：
- 训练时为什么也需要Causal Mask
- 三种注意力机制的QKV来源和用途
- Encoder-Decoder vs Decoder-only的权衡

### 8.2 面试准备建议

**理论准备**：
- 能清晰画出Encoder和Decoder的架构图
- 能解释Cross-Attention的计算过程
- 理解因果性和自回归生成

**代码准备**：
- 手写Causal Mask的实现
- 实现简单的Decoder Layer
- 了解KV Cache优化

**实践经验**：
- 使用过BERT（Encoder）或GPT（Decoder）
- 了解Transformer库（HuggingFace）
- 知道如何微调和部署模型

### 8.3 推荐学习资源

**必读论文**：
1. Attention Is All You Need（Transformer原论文）
2. BERT: Pre-training of Deep Bidirectional Transformers
3. Language Models are Unsupervised Multitask Learners（GPT-2）
4. Exploring the Limits of Transfer Learning with T5

**代码资源**：
- Annotated Transformer（Harvard NLP）
- nanoGPT（Andrej Karpathy）
- HuggingFace Transformers库

**在线课程**：
- Stanford CS224N
- Fast.ai Practical Deep Learning

---

祝你面试顺利！掌握这些知识，你就能自信地回答关于Encoder-Decoder的所有问题。