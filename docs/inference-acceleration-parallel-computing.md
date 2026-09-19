# 推理加速与并行计算面试指南

## 目录
- [1. 推理加速基础](#1-推理加速基础)
- [2. 并行计算原理](#2-并行计算原理)
- [3. GPU 加速技术](#3-gpu-加速技术)
- [4. 模型优化技术](#4-模型优化技术)
- [5. 分布式推理](#5-分布式推理)
- [6. 工程实践](#6-工程实践)
- [7. 面试高频问题](#7-面试高频问题)

---

## 1. 推理加速基础

### 1.1 什么是推理加速

**推理 (Inference)**: 使用训练好的模型对新数据进行预测的过程。

**推理加速**: 通过各种技术手段降低推理延迟、提高吞吐量、降低资源消耗。

### 1.2 为什么需要推理加速

| 场景 | 需求 | 挑战 |
|------|------|------|
| 在线服务 | 低延迟 (<100ms) | 模型参数量大，计算密集 |
| 批量处理 | 高吞吐量 | 资源利用率低 |
| 边缘设备 | 低功耗、小内存 | 硬件资源受限 |
| 实时应用 | 稳定性能 | 峰值负载波动 |

### 1.3 性能指标

```
延迟 (Latency): 单个请求的响应时间
吞吐量 (Throughput): 单位时间处理的请求数
QPS (Queries Per Second): 每秒查询数
资源利用率: GPU/CPU/内存使用效率
成本: 硬件成本 + 能耗成本
```

### 1.4 推理 vs 训练的区别

| 维度 | 训练 | 推理 |
|------|------|------|
| 计算模式 | 前向 + 反向传播 | 仅前向传播 |
| 批次大小 | 较大 (32-512) | 较小 (1-32) |
| 精度要求 | FP32/FP16 | INT8/INT4 可接受 |
| 延迟敏感 | 不敏感 | 高度敏感 |
| 硬件 | 高端 GPU/TPU | 多样化 (CPU/GPU/边缘) |
| 目标 | 模型收敛 | 低延迟高吞吐 |

---

## 2. 并行计算原理

### 2.1 并行计算层次

#### 2.1.1 指令级并行 (ILP)
```
单核 CPU 通过流水线、超标量等技术同时执行多条指令
- 流水线 (Pipeline)
- 乱序执行 (Out-of-Order Execution)
- 分支预测 (Branch Prediction)
```

#### 2.1.2 数据级并行 (DLP)
```
SIMD (Single Instruction Multiple Data)
- CPU: AVX, AVX-512
- GPU: CUDA Core, Tensor Core
- 向量化运算
```

#### 2.1.3 线程级并行 (TLP)
```
多线程同时执行不同任务
- CPU: 多核并行
- GPU: 数千个线程并发
```

#### 2.1.4 任务级并行
```
多个独立任务并发执行
- 多进程
- 分布式计算
- 流水线并行
```

### 2.2 Amdahl 定律

**核心思想**: 程序加速受串行部分限制

```
加速比 = 1 / (S + P/N)

S: 串行部分占比
P: 可并行部分占比 (S + P = 1)
N: 处理器数量
```

**示例**:
```python
# 假设程序 10% 串行，90% 可并行
S = 0.1
P = 0.9

# 使用 10 个处理器
N = 10
speedup = 1 / (0.1 + 0.9/10) = 5.26x

# 使用无限处理器
# 最大加速比 = 1 / S = 1 / 0.1 = 10x
```

**面试要点**: 强调并行化的瓶颈在于串行部分，不是无限加核心就能无限加速。

### 2.3 Roofline 模型

用于分析计算密集型 vs 访存密集型任务。

```
性能 (FLOPS) = min(峰值算力, 访存带宽 × 计算强度)

计算强度 (Arithmetic Intensity) = FLOPs / Bytes
```

**三个区域**:
1. **访存受限**: 提升带宽、优化数据访问
2. **计算受限**: 提升算力、算法优化
3. **峰值区**: 已达硬件极限

### 2.4 并行计算模式

#### 2.4.1 Map-Reduce
```python
# Map: 并行处理每个元素
results = parallel_map(function, data)

# Reduce: 聚合结果
final = reduce(combine, results)
```

#### 2.4.2 Fork-Join
```python
# Fork: 任务分解
tasks = split_task(big_task)

# Parallel: 并发执行
results = [execute(task) for task in tasks]

# Join: 结果合并
final = merge(results)
```

#### 2.4.3 Pipeline
```
Stage 1    Stage 2    Stage 3
  |          |          |
  v          v          v
[Task A] -> [Task A] -> [Task A]
            [Task B] -> [Task B]
                       [Task C]
```

---

## 3. GPU 加速技术

### 3.1 GPU 架构基础

#### 3.1.1 CUDA 核心结构
```
GPU
├── SM (Streaming Multiprocessor) × N
│   ├── CUDA Core × 64-128
│   ├── Tensor Core × 4-8 (Volta+)
│   ├── Shared Memory (64-128 KB)
│   ├── L1 Cache
│   └── Register File
├── L2 Cache (全局)
└── Global Memory (HBM/GDDR)
```

#### 3.1.2 CUDA 编程模型
```cpp
// 网格 (Grid) -> 块 (Block) -> 线程 (Thread)
__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 启动配置
dim3 blockDim(16, 16);  // 256 个线程/块
dim3 gridDim((N+15)/16, (N+15)/16);
matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
```

### 3.2 内存层次与优化

#### 3.2.1 内存类型对比

| 内存类型 | 延迟 | 带宽 | 大小 | 可见性 |
|---------|------|------|------|--------|
| Register | 1 cycle | 最高 | ~256 KB/SM | 线程私有 |
| Shared Memory | ~30 cycles | TB/s | 64-128 KB/SM | 块内共享 |
| L1 Cache | ~30 cycles | TB/s | 128 KB/SM | 自动管理 |
| L2 Cache | ~200 cycles | TB/s | 6-40 MB | 全局 |
| Global Memory | ~400 cycles | 1-2 TB/s | 16-80 GB | 全局 |

#### 3.2.2 内存访问优化

**合并访问 (Coalesced Access)**:
```cpp
// 好: 连续访问
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    data[i] = ...;  // 相邻线程访问相邻内存
}

// 差: 跨步访问
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    data[i * stride] = ...;  // 浪费带宽
}
```

**使用 Shared Memory**:
```cpp
__global__ void matmul_shared(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // 加载数据到共享内存
    As[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
    Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
    __syncthreads();
    
    // 从共享内存计算
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[ty][k] * Bs[k][tx];
    }
}
```

### 3.3 Tensor Core 加速

**什么是 Tensor Core**: 专门用于矩阵乘法的硬件单元 (Volta/Turing/Ampere/Hopper)

**性能提升**:
- FP16: 8-16× vs CUDA Core
- INT8: 16-32× vs CUDA Core
- FP8 (H100): 2× vs FP16

**WMMA API 示例**:
```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void wmma_matmul(half *A, half *B, float *C, int M, int N, int K) {
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    fill_fragment(c_frag, 0.0f);
    
    // 加载 16×16 矩阵块
    load_matrix_sync(a_frag, A + ..., K);
    load_matrix_sync(b_frag, B + ..., K);
    
    // 矩阵乘加 C = A × B + C
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // 存储结果
    store_matrix_sync(C + ..., c_frag, N, mem_row_major);
}
```

### 3.4 CUDA Streams 与异步执行

**并发执行多个操作**:
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Stream 1: 数据传输 + 计算
cudaMemcpyAsync(d_A1, h_A1, size, H2D, stream1);
kernel1<<<grid, block, 0, stream1>>>(d_A1, d_C1);
cudaMemcpyAsync(h_C1, d_C1, size, D2H, stream1);

// Stream 2: 并发执行
cudaMemcpyAsync(d_A2, h_A2, size, H2D, stream2);
kernel2<<<grid, block, 0, stream2>>>(d_A2, d_C2);
cudaMemcpyAsync(h_C2, d_C2, size, D2H, stream2);

// 同步
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

**时间线**:
```
Stream 1: [H2D1] [Kernel1] [D2H1]
Stream 2:   [H2D2] [Kernel2] [D2H2]
          --------------------------->
          并发执行，节省时间
```

---

## 4. 模型优化技术

### 4.1 量化 (Quantization)

#### 4.1.1 基本原理

将高精度权重/激活映射到低精度表示。

```
FP32 (32 bit) -> FP16 (16 bit) -> INT8 (8 bit) -> INT4 (4 bit)
```

**量化公式**:
```python
# 对称量化
scale = max(abs(x)) / 127
x_int8 = round(x / scale).clip(-128, 127)
x_dequant = x_int8 * scale

# 非对称量化
scale = (max(x) - min(x)) / 255
zero_point = round(-min(x) / scale)
x_uint8 = round(x / scale + zero_point).clip(0, 255)
x_dequant = (x_uint8 - zero_point) * scale
```

#### 4.1.2 量化类型

**训练后量化 (PTQ - Post-Training Quantization)**:
```python
import torch

# FP32 模型
model = MyModel()
model.load_state_dict(torch.load('model.pth'))

# 动态量化 (推理时量化激活)
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 静态量化 (校准数据集)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# 运行校准数据
for data in calibration_loader:
    model(data)
torch.quantization.convert(model, inplace=True)
```

**量化感知训练 (QAT - Quantization-Aware Training)**:
```python
import torch.quantization as quant

model = MyModel()
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model_prepared = quant.prepare_qat(model)

# 训练
for epoch in range(num_epochs):
    train(model_prepared, train_loader)

# 转换为量化模型
model_quantized = quant.convert(model_prepared)
```

#### 4.1.3 性能对比

| 精度 | 内存 | 速度 | 精度损失 |
|------|------|------|---------|
| FP32 | 1× | 1× | 0% |
| FP16 | 0.5× | 2-3× | <0.1% |
| INT8 | 0.25× | 3-4× | 0.5-2% |
| INT4 | 0.125× | 4-8× | 2-5% |

### 4.2 剪枝 (Pruning)

#### 4.2.1 非结构化剪枝

移除个别权重。

```python
import torch.nn.utils.prune as prune

# 移除 30% 权重
prune.l1_unstructured(module, name='weight', amount=0.3)

# 永久移除
prune.remove(module, 'weight')
```

**稀疏矩阵存储 (CSR)**:
```python
# 原始: [0, 3, 0, 0, 5, 0]
# CSR 表示
values = [3, 5]
col_indices = [1, 4]
row_ptr = [0, 2]  # 行起始位置
```

#### 4.2.2 结构化剪枝

移除整个通道/层/注意力头。

```python
# 移除 50% 的卷积通道
prune.ln_structured(
    module, 
    name='weight', 
    amount=0.5, 
    n=2,  # L2 范数
    dim=0  # 输出通道维度
)
```

#### 4.2.3 N:M 结构化稀疏

Ampere GPU (A100) 支持 2:4 稀疏 (每 4 个元素中 2 个非零)。

```
原始: [1, 2, 3, 4, 5, 6, 7, 8]
2:4:  [1, 2, 0, 0, 5, 6, 0, 0]
      └─────┘     └─────┘
       保留2/4     保留2/4
```

**加速**: 2× 理论加速 (减少 50% 计算)

### 4.3 知识蒸馏 (Knowledge Distillation)

#### 4.3.1 基本原理

用大模型 (Teacher) 训练小模型 (Student)。

```python
# Teacher (大模型)
teacher = BigModel()
teacher.eval()

# Student (小模型)
student = SmallModel()

# 蒸馏损失
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # 软标签损失
    soft_loss = nn.KLDivLoss()(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)
    
    # 硬标签损失
    hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss

# 训练
for data, labels in train_loader:
    with torch.no_grad():
        teacher_logits = teacher(data)
    
    student_logits = student(data)
    loss = distillation_loss(student_logits, teacher_logits, labels, T=3, alpha=0.7)
    loss.backward()
    optimizer.step()
```

#### 4.3.2 蒸馏变体

- **Logit Distillation**: 匹配输出分布
- **Feature Distillation**: 匹配中间层特征
- **Attention Distillation**: 匹配注意力权重
- **Self-Distillation**: 模型自蒸馏

### 4.4 算子融合 (Operator Fusion)

#### 4.4.1 基本思想

合并多个操作，减少内存读写。

**示例: GELU 激活**:
```python
# 未融合 (3 次内存读写)
x1 = x * 0.5
x2 = torch.tanh(...)
y = x * x1 * (1 + x2)

# 融合 (1 次内存读写)
y = gelu_fused(x)
```

#### 4.4.2 常见融合模式

```
Conv + BatchNorm + ReLU -> ConvBNReLU (单次 kernel)
Linear + Bias + Activation -> FusedLinear
Softmax(Q @ K.T / sqrt(d)) -> FusedScaledDotProduct
```

**TorchScript 融合**:
```python
@torch.jit.script
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

model = torch.jit.script(model)  # 自动融合
```

### 4.5 低秩分解 (Low-Rank Factorization)

#### 4.5.1 矩阵分解

**SVD 分解**:
```python
import numpy as np

# 原始权重 W: [m, n]
U, S, Vt = np.linalg.svd(W, full_matrices=False)

# 保留前 k 个奇异值
k = 64  # rank
W_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# 分解为两个小矩阵
W1 = U[:, :k] @ np.diag(np.sqrt(S[:k]))  # [m, k]
W2 = np.diag(np.sqrt(S[:k])) @ Vt[:k, :]  # [k, n]
# W ≈ W1 @ W2
```

**参数量**:
```
原始: m × n
分解: m × k + k × n
压缩比: (m × n) / (m × k + k × n)

# 例如: m=1024, n=1024, k=64
原始: 1M 参数
分解: 128K 参数 (8× 压缩)
```

#### 4.5.2 LoRA (Low-Rank Adaptation)

用于大模型微调。

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.requires_grad = False  # 冻结原始权重
        
        # 低秩分解
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 0.01
        
    def forward(self, x):
        # W × x + (A × B) × x
        return self.W(x) + (x @ self.lora_A @ self.lora_B) * self.scaling
```

---

## 5. 分布式推理

### 5.1 模型并行

#### 5.1.1 张量并行 (Tensor Parallelism)

**按维度切分权重**:

```python
# 原始: Y = XW, W: [d, h]
# GPU 0: W1: [d, h/2]
# GPU 1: W2: [d, h/2]

class ColumnParallelLinear(nn.Module):
    def forward(self, x):
        # 每个 GPU 计算一部分输出
        output_parallel = F.linear(x, self.weight)  # [b, h/2]
        return output_parallel  # 不需要通信

class RowParallelLinear(nn.Module):
    def forward(self, x_parallel):
        # 每个 GPU 计算部分结果
        output_parallel = F.linear(x_parallel, self.weight)
        # All-Reduce 聚合结果
        output = torch.distributed.all_reduce(output_parallel)
        return output
```

**Megatron-LM 风格**:
```
Attention:
  Q, K, V: Column Parallel (每个 GPU 算部分头)
  Output: Row Parallel (聚合)

FFN:
  FC1: Column Parallel
  FC2: Row Parallel
```

#### 5.1.2 流水线并行 (Pipeline Parallelism)

**按层切分**:
```
GPU 0: Layer 0-5
GPU 1: Layer 6-11
GPU 2: Layer 12-17
GPU 3: Layer 18-23
```

**GPipe 调度**:
```
微批次 (Micro-batch): 将 batch 切分为多个小 batch

时间步:
t0: GPU0[batch1]
t1: GPU0[batch2] GPU1[batch1]
t2: GPU0[batch3] GPU1[batch2] GPU2[batch1]
t3: GPU0[batch4] GPU1[batch3] GPU2[batch2] GPU3[batch1]
...
```

#### 5.1.3 序列并行 (Sequence Parallelism)

**按序列长度切分**:
```python
# 长序列: seq_len = 8192
# GPU 0: tokens 0-2047
# GPU 1: tokens 2048-4095
# GPU 2: tokens 4096-6143
# GPU 3: tokens 6144-8191

# Attention 需要 All-to-All 通信交换 K/V
```

### 5.2 数据并行

#### 5.2.1 标准数据并行 (DP)

```python
import torch.nn.parallel as parallel

# 每个 GPU 复制完整模型
model = MyModel()
model = parallel.DataParallel(model, device_ids=[0, 1, 2, 3])

# 数据按 batch 维度切分
data = data.cuda()  # [batch, ...]
output = model(data)  # 自动分发到各 GPU
```

#### 5.2.2 分布式数据并行 (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl', world_size=4, rank=rank)

# 每个进程独立模型副本
model = MyModel().cuda(rank)
model = DDP(model, device_ids=[rank])

# 训练
for data, labels in train_loader:
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()  # All-Reduce 梯度
    optimizer.step()
```

### 5.3 推理框架

#### 5.3.1 TensorRT

**NVIDIA 推理优化引擎**:

```python
import tensorrt as trt

# 构建引擎
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)
parser.parse_from_file("model.onnx")

# 优化配置
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # 启用 FP16

# 构建
engine = builder.build_serialized_network(network, config)

# 推理
context = engine.create_execution_context()
context.execute_v2(bindings)
```

**优化技术**:
- 层融合、算子融合
- FP16/INT8 量化
- Kernel Auto-Tuning
- Dynamic Shapes

#### 5.3.2 ONNX Runtime

**跨平台推理**:

```python
import onnxruntime as ort

# 加载模型
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 推理
inputs = {'input': input_data.numpy()}
outputs = session.run(None, inputs)
```

#### 5.3.3 vLLM

**LLM 推理优化框架**:

```python
from vllm import LLM, SamplingParams

# 初始化
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,
    dtype="float16"
)

# 批量推理
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

**核心技术**:
- **PagedAttention**: KV Cache 分页管理
- **Continuous Batching**: 动态批处理
- **Tensor Parallelism**: 多 GPU 并行

### 5.4 通信优化

#### 5.4.1 集合通信操作

**All-Reduce**:
```
每个 GPU 有一个值，所有 GPU 得到总和

GPU 0: 1        GPU 0: 10
GPU 1: 2   ->   GPU 1: 10
GPU 2: 3        GPU 2: 10
GPU 3: 4        GPU 3: 10
```

**All-Gather**:
```
每个 GPU 有一个值，所有 GPU 收集所有值

GPU 0: [1]        GPU 0: [1,2,3,4]
GPU 1: [2]   ->   GPU 1: [1,2,3,4]
GPU 2: [3]        GPU 2: [1,2,3,4]
GPU 3: [4]        GPU 3: [1,2,3,4]
```

**Reduce-Scatter**:
```
分块 All-Reduce，每个 GPU 得到一部分结果

GPU 0: [1,2,3,4]     GPU 0: [10]
GPU 1: [1,2,3,4] ->  GPU 1: [10]
GPU 2: [1,2,3,4]     GPU 2: [10]
GPU 3: [1,2,3,4]     GPU 3: [10]
```

#### 5.4.2 Ring All-Reduce

**带宽最优算法**:
```
时间复杂度: 2(N-1)/N × data_size / bandwidth
N 个 GPU 情况下接近最优
```

#### 5.4.3 NCCL (NVIDIA Collective Communications Library)

```python
import torch.distributed as dist

# 初始化
dist.init_process_group(backend='nccl')

# All-Reduce
tensor = torch.randn(1000, 1000).cuda()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# All-Gather
tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
dist.all_gather(tensor_list, tensor)
```

---

## 6. 工程实践

### 6.1 批处理优化

#### 6.1.1 动态批处理

```python
class DynamicBatcher:
    def __init__(self, max_batch_size=32, max_wait_ms=10):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []
        
    async def add_request(self, request):
        self.queue.append(request)
        
        # 达到批量大小或超时
        if len(self.queue) >= self.max_batch_size:
            return await self.process_batch()
        
        await asyncio.sleep(self.max_wait_ms / 1000)
        if self.queue:
            return await self.process_batch()
    
    async def process_batch(self):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        
        # 批量推理
        inputs = torch.stack([req.input for req in batch])
        outputs = model(inputs)
        
        return outputs
```

#### 6.1.2 Padding 策略

```python
# 变长序列批处理
def collate_fn(batch):
    # 找到最大长度
    max_len = max(len(item) for item in batch)
    
    # Padding 到相同长度
    padded = []
    attention_mask = []
    for item in batch:
        pad_len = max_len - len(item)
        padded.append(item + [PAD_TOKEN] * pad_len)
        attention_mask.append([1] * len(item) + [0] * pad_len)
    
    return torch.tensor(padded), torch.tensor(attention_mask)

# Bucketing: 按长度分组减少 padding
buckets = {
    'short': [],   # 0-128
    'medium': [],  # 129-512
    'long': []     # 513-2048
}
```

### 6.2 KV Cache 优化

#### 6.2.1 KV Cache 基础

**Transformer 自回归生成**:
```python
# 未优化: 每步重算所有历史 token
for t in range(max_len):
    logits = model(input_ids[:, :t+1])  # O(t^2)
    next_token = logits[:, -1].argmax()
    input_ids = torch.cat([input_ids, next_token], dim=1)

# 优化: 缓存 K, V
past_key_values = None
for t in range(max_len):
    logits, past_key_values = model(
        input_ids[:, t:t+1],  # 只输入新 token
        past_key_values=past_key_values  # 使用缓存
    )
    next_token = logits[:, -1].argmax()
```

**内存占用**:
```
每层每个 token: 2 × hidden_size × 2 bytes (K + V, FP16)
GPT-3 (96 层, d=12288): 2 × 96 × 12288 × 2 = 4.7 MB/token
2048 tokens: 9.6 GB
```

#### 6.2.2 PagedAttention (vLLM)

**传统方法**: 预分配连续内存 → 浪费 + 碎片

**PagedAttention**: 分页管理，类似虚拟内存

```python
# KV Cache 分页
page_size = 16  # 每页 16 个 token
num_pages = (seq_len + page_size - 1) // page_size

# 非连续存储
kv_cache = {
    layer_0: [page_0, page_5, page_12, ...],
    layer_1: [page_1, page_8, page_15, ...],
    ...
}

# Attention 计算时按需加载
for page in kv_pages:
    k_page = load_page(page.k_addr)
    v_page = load_page(page.v_addr)
    attention_score += q @ k_page.T
    attention_output += attention_score @ v_page
```

**优势**:
- 内存利用率 >90% (传统 ~20-40%)
- 支持 beam search 等共享场景
- 减少内存碎片

### 6.3 Flash Attention

**问题**: 标准 Attention 需要 O(N²) 内存存储注意力矩阵

**Flash Attention**: IO-aware 算法，减少 HBM 访问

```python
# 标准 Attention (伪代码)
Q, K, V = input_proj(X)  # [N, d]
S = Q @ K.T              # [N, N] 存储到 HBM
P = softmax(S)           # [N, N]
O = P @ V                # [N, d]

# Flash Attention: 分块计算，不存储 S
block_size = 128
for i in range(0, N, block_size):
    Q_block = Q[i:i+block_size]  # 加载到 SRAM
    for j in range(0, N, block_size):
        K_block = K[j:j+block_size]
        V_block = V[j:j+block_size]
        
        # 在 SRAM 中计算
        S_block = Q_block @ K_block.T
        P_block = softmax(S_block)  # 不存储到 HBM
        O_block += P_block @ V_block
    
    O[i:i+block_size] = O_block
```

**性能**:
- 2-4× 训练加速
- 内存: O(N²) → O(N)
- 支持任意序列长度

### 6.4 推理服务部署

#### 6.4.1 Triton Inference Server

```python
# model_repository/my_model/config.pbtxt
name: "my_model"
platform: "pytorch_libtorch"
max_batch_size: 32
dynamic_batching {
  max_queue_delay_microseconds: 100
}
instance_group [{ kind: KIND_GPU, count: 2 }]

input [
  { name: "input", data_type: TYPE_FP32, dims: [-1, 768] }
]
output [
  { name: "output", data_type: TYPE_FP32, dims: [-1, 10] }
]
```

#### 6.4.2 TorchServe

```bash
# 打包模型
torch-model-archiver \
  --model-name my_model \
  --version 1.0 \
  --serialized-file model.pt \
  --handler custom_handler.py

# 启动服务
torchserve --start \
  --model-store model_store \
  --models my_model=my_model.mar \
  --ncs
```

#### 6.4.3 性能监控

```python
import time
import numpy as np

class LatencyTracker:
    def __init__(self):
        self.latencies = []
    
    def track(self, func):
        start = time.perf_counter()
        result = func()
        latency = (time.perf_counter() - start) * 1000  # ms
        self.latencies.append(latency)
        return result
    
    def report(self):
        p50 = np.percentile(self.latencies, 50)
        p95 = np.percentile(self.latencies, 95)
        p99 = np.percentile(self.latencies, 99)
        print(f"P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")
```

---

## 7. 面试高频问题

### Q1: 如何降低模型推理延迟？

**系统回答框架**:
1. **模型层面**
   - 量化: FP16/INT8 降低计算和内存
   - 剪枝: 移除冗余参数
   - 蒸馏: 训练小模型
   - 算子融合: 减少 kernel launch overhead

2. **系统层面**
   - 批处理: 提高吞吐
   - 异步推理: Pipeline 计算和 IO
   - KV Cache: 避免重复计算
   - 专用推理引擎: TensorRT, ONNX Runtime

3. **硬件层面**
   - GPU: 并行计算
   - Tensor Core: 混合精度加速
   - 模型并行: 大模型切分

### Q2: GPU 和 CPU 推理的区别？

| 维度 | CPU | GPU |
|------|-----|-----|
| 核心数 | 8-64 | 数千-上万 |
| 计算模式 | 复杂控制流 | 大规模并行 |
| 适用场景 | Batch=1, 低延迟 | Batch>8, 高吞吐 |
| 内存带宽 | ~100 GB/s | ~1000 GB/s |
| 优势 | 延迟敏感小模型 | 大模型批量推理 |

**选择建议**:
- 小模型 (< 100M 参数) + 单个请求: CPU
- 大模型或批量请求: GPU

### Q3: 解释 Tensor Core 和 CUDA Core 的区别

**CUDA Core**: 通用浮点单元，逐个元素计算

**Tensor Core**: 专用矩阵乘法单元 (4×4×4 或更大)

```
CUDA Core: c = a * b  (标量乘法)
Tensor Core: C = A × B + C  (矩阵乘加，FMA)

1 个 Tensor Core = 64 个 CUDA Core 吞吐 (FP16)
```

**使用条件**:
- 矩阵维度是 8 或 16 的倍数
- 使用 FP16/BF16/TF32/INT8 精度
- 通过 cuBLAS/cuDNN 或 WMMA API 调用

### Q4: 什么是 PagedAttention？优势是什么？

**传统 KV Cache**: 预分配大块连续内存
- 浪费: 实际长度 << 最大长度
- 碎片: 释放后难以复用
- 共享难: Beam search 需要复制

**PagedAttention**: 分页管理 (灵感来自虚拟内存)
- 按需分配: 只分配实际使用的页
- 灵活共享: 不同序列共享相同前缀的页
- 高利用率: >90% vs 传统 20-40%

**核心思想**:
```
Sequence A: [Page 0] -> [Page 5] -> [Page 12]
Sequence B: [Page 0] -> [Page 5] -> [Page 9]
            └─ 共享前缀 ─┘
```

### Q5: INT8 量化为什么能加速推理？

**三个层面的加速**:

1. **计算加速**
   - INT8 乘法比 FP32 快 4× (理论)
   - Tensor Core: INT8 吞吐是 FP16 的 2×

2. **内存加速**
   - 内存占用: 1/4 (8 bit vs 32 bit)
   - 带宽利用: 4× 数据吞吐
   - Cache 命中率提升

3. **能耗降低**
   - INT8 MAC: ~0.2 pJ
   - FP32 MAC: ~3.7 pJ
   - 能耗降低 18×

**实际效果**: 2-4× 端到端加速 (考虑量化/反量化开销)

### Q6: Flash Attention 解决了什么问题？

**问题**: 标准 Attention 的 IO 瓶颈

```
标准 Attention:
1. Q @ K.T -> HBM [O(N²) 写]
2. Softmax -> HBM [O(N²) 读写]
3. Softmax @ V -> HBM [O(N²) 读]

总 IO: O(N²) 次 HBM 访问
HBM 带宽: ~1.5 TB/s
SRAM 带宽: ~19 TB/s (12× faster)
```

**Flash Attention 解决方案**:
- 分块计算: 数据保留在 SRAM
- 在线 Softmax: 增量更新，不存储中间结果
- IO 复杂度: O(N²/M), M = SRAM 大小

**收益**:
- 训练: 2-4× 加速
- 内存: 从 O(N²) 降到 O(N)
- 支持超长序列 (64K+)

### Q7: 模型并行 vs 数据并行？

| 维度 | 数据并行 | 模型并行 |
|------|---------|---------|
| 切分对象 | 数据 | 模型参数 |
| 模型复制 | 每个 GPU 完整模型 | 每个 GPU 部分模型 |
| 通信 | 梯度同步 (All-Reduce) | 激活传递 (P2P/All-Reduce) |
| 适用场景 | 模型小，数据大 | 模型大，单卡放不下 |
| 扩展性 | 近线性 (通信少) | 受限于模型结构 |

**混合并行**: 数据并行 + 张量并行 + 流水线并行
```
例: 8 GPU 训练 GPT-3
- 2-way Pipeline (前/后 12 层)
- 2-way Tensor Parallel (每层切分)
- 2-way Data Parallel (2 个副本)
```

### Q8: 如何选择 Batch Size？

**权衡因素**:

1. **延迟 vs 吞吐**
   - Batch=1: 最低延迟，低吞吐
   - Batch=32: 高吞吐，高延迟

2. **GPU 利用率**
   - 太小: 算力浪费
   - 太大: 内存溢出 / 超时

3. **数学公式**
```
Latency = compute_time(batch) + overhead
Throughput = batch_size / latency

最优 batch: 使 GPU 利用率 >80% 的最小 batch
```

**实践建议**:
- 在线服务: batch=1-8 (延迟敏感)
- 离线批处理: batch=32-128 (吞吐优先)
- 动态批处理: 根据队列长度调整

### Q9: 为什么 Transformer 推理慢？如何优化？

**瓶颈分析**:

1. **计算瓶颈**
   - Attention: O(N² × d) 复杂度
   - FFN: 大量矩阵乘法

2. **内存瓶颈**
   - KV Cache: 每 token 数 MB
   - 自回归生成: 逐 token 计算

3. **通信瓶颈** (多 GPU)
   - 每层需要 All-Reduce
   - 激活传递

**优化方案**:

| 技术 | 针对问题 | 效果 |
|------|---------|------|
| Flash Attention | 内存 IO | 2-4× 加速 |
| KV Cache | 重复计算 | 10-100× 加速 |
| PagedAttention | KV 内存利用 | 2-3× 吞吐 |
| Continuous Batching | 批处理效率 | 2-10× 吞吐 |
| 量化 INT8 | 计算+内存 | 2-4× 加速 |
| 张量并行 | 大模型推理 | 线性扩展 |

### Q10: 设计一个高性能推理系统需要考虑什么？

**系统架构**:

```
Load Balancer
      |
   [API Gateway]
      |
   [Request Queue] <- 动态批处理
      |
   [Model Server Pool]
      |-- Worker 1 (GPU 0)
      |-- Worker 2 (GPU 1)
      |-- Worker N (GPU N)
      |
   [Result Cache]
```

**关键技术点**:

1. **请求调度**
   - 动态批处理
   - 优先级队列
   - 超时控制

2. **模型优化**
   - 量化 (INT8)
   - 编译优化 (TensorRT)
   - 算子融合

3. **内存管理**
   - KV Cache 管理
   - 模型权重共享
   - 显存碎片整理

4. **并发控制**
   - GPU Stream 并发
   - 多进程/多线程
   - 异步 IO

5. **监控告警**
   - 延迟监控 (P50/P95/P99)
   - 吞吐量统计
   - GPU 利用率
   - 错误率

6. **容错机制**
   - 请求重试
   - 模型热备份
   - 降级策略

---

## 8. 实战案例分析

### 案例 1: GPT 模型推理优化

**初始状态**:
- 模型: GPT-2 (1.5B 参数)
- 硬件: 1× A100 (40GB)
- 性能: 5 tokens/s, Batch=1

**优化过程**:

```python
# Step 1: 启用 KV Cache
past_key_values = None
for t in range(max_new_tokens):
    outputs = model(
        input_ids[:, -1:],
        past_key_values=past_key_values,
        use_cache=True
    )
    past_key_values = outputs.past_key_values
# 提升: 5 -> 50 tokens/s (10×)

# Step 2: FP16 混合精度
model = model.half()
# 提升: 50 -> 80 tokens/s (1.6×)

# Step 3: 编译优化
model = torch.compile(model, mode='max-autotune')
# 提升: 80 -> 120 tokens/s (1.5×)

# Step 4: Flash Attention 2
from flash_attn import flash_attn_func
# 替换 Attention 实现
# 提升: 120 -> 180 tokens/s (1.5×)

# Step 5: 批处理
batch_size = 8
# 吞吐: 180 × 8 = 1440 tokens/s
# 单请求延迟略增: 1/180 -> 8/1440 = 5.6ms
```

**最终结果**:
- 单请求: 5 -> 180 tokens/s (36× 加速)
- 批量吞吐: 1440 tokens/s
- 延迟: P95 < 10ms

### 案例 2: BERT 分类服务优化

**场景**: 在线文本分类，QPS 目标 1000

**优化方案**:

```python
# 1. 模型量化
from torch.quantization import quantize_dynamic
model_int8 = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
# 延迟: 15ms -> 8ms

# 2. ONNX Runtime
import onnxruntime as ort
torch.onnx.export(model, dummy_input, "bert.onnx")
session = ort.InferenceSession("bert.onnx", providers=['CUDAExecutionProvider'])
# 延迟: 8ms -> 5ms

# 3. 动态批处理
batcher = DynamicBatcher(max_batch_size=32, max_wait_ms=5)
# 吞吐: 200 -> 1200 QPS

# 4. TensorRT 编译
import tensorrt as trt
# 构建 TensorRT 引擎
# 延迟: 5ms -> 3ms
# 吞吐: 1200 -> 1500 QPS
```

**部署架构**:
```
Nginx (Load Balancer)
    |
    v
Triton Inference Server
    ├── Instance 1 (GPU 0, Batch=32)
    ├── Instance 2 (GPU 1, Batch=32)
    └── Instance 3 (GPU 2, Batch=32)
```

**结果**: P95 延迟 8ms, QPS 1500, GPU 利用率 75%

---

## 9. 快速参考

### 9.1 优化技术选择指南

| 场景 | 推荐技术 | 预期效果 |
|------|---------|---------|
| 在线服务 (Batch=1) | FP16 + 编译优化 + KV Cache | 2-5× |
| 批量推理 (Batch=32) | 动态批处理 + INT8 量化 | 4-10× |
| 大模型 (>10B) | 张量并行 + PagedAttention | 能跑 + 2-3× 吞吐 |
| 边缘设备 | INT8 量化 + 剪枝 + 蒸馏 | 5-10× |
| 超长序列 (>8K) | Flash Attention + 稀疏注意力 | 2-10× |

### 9.2 常用工具链

```
模型训练: PyTorch / TensorFlow
模型转换: ONNX / TorchScript
推理引擎: TensorRT / ONNX Runtime / OpenVINO
量化工具: PyTorch Quantization / TensorRT PTQ/QAT
LLM 推理: vLLM / TGI / DeepSpeed-Inference
服务框架: Triton / TorchServe / Ray Serve
监控: Prometheus + Grafana
```

### 9.3 性能基准参考

**延迟 (ms) - BERT-Base (FP16, A100)**:
```
Batch=1:  2-3 ms
Batch=8:  5-8 ms
Batch=32: 15-25 ms
```

**吞吐 (tokens/s) - GPT-3 7B (FP16, A100)**:
```
Batch=1:  ~100 tokens/s
Batch=8:  ~600 tokens/s
Batch=32: ~1500 tokens/s
```

---

## 10. 学习资源

**论文**:
- Attention Is All You Need (Transformer)
- FlashAttention: Fast and Memory-Efficient Exact Attention
- Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

**开源项目**:
- vLLM: https://github.com/vllm-project/vllm
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- TensorRT: https://github.com/NVIDIA/TensorRT
- DeepSpeed: https://github.com/microsoft/DeepSpeed

**课程**:
- CS149 (Stanford): Parallel Computing
- CS217 (Stanford): Hardware Accelerators for Machine Learning

---

**文档版本**: 1.0.0  
**最后更新**: 2026-09-19  
**作者**: pzzzzzzy  
**用途**: 技术面试准备

祝面试顺利！