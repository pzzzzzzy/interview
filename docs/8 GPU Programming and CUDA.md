# GPU编程与CUDA基础 核心知识储备

## 一、为什么需要GPU？

### CPU vs GPU的根本差异

**CPU (中央处理器)**:
- 设计目标: 通用计算，复杂逻辑
- 核心数: 少（4-64核）
- 特点: 每个核心强大，处理复杂任务
- 适合: 串行任务、复杂控制流

**GPU (图形处理器)**:
- 设计目标: 大规模并行计算
- 核心数: 多（数千到数万个核心）
- 特点: 每个核心简单，但数量多
- 适合: 并行任务、简单重复计算

**类比理解**:
```
CPU = 少数博士 (强大但数量少)
- 能解决复杂问题
- 但一次只能做几件事

GPU = 大量小学生 (简单但数量多)
- 单个能力有限
- 但可以同时做成千上万件简单的事
```

---

### 为什么深度学习需要GPU？

**深度学习的核心操作**: 矩阵运算

**例子**: 矩阵乘法
```
C = A × B

A: 1000 × 2000
B: 2000 × 3000
C: 1000 × 3000

计算量: 1000 × 2000 × 3000 = 60亿次乘法
```

**CPU方式** (串行):
```
for i in range(1000):
    for j in range(3000):
        for k in range(2000):
            C[i][j] += A[i][k] * B[k][j]

时间: 可能需要几秒
```

**GPU方式** (并行):
```
启动 1000×3000 个线程，每个线程计算一个C[i][j]

所有计算同时进行！

时间: 可能只需要几毫秒
```

**加速比**: 100-1000倍！

---

### GPU在AI中的应用

**训练阶段**:
- 前向传播: 大量矩阵乘法
- 反向传播: 梯度计算
- 参数更新: 向量运算

**推理阶段**:
- 实时图像处理
- 大批量预测

**实际数据**:
```
训练一个GPT-3规模的模型:
- CPU: 需要几年
- GPU集群: 几个月
- 成本节省: 百万美元级别
```

---

## 二、GPU架构基础

### 整体架构

```
GPU
├─ 多个 SM (Streaming Multiprocessor)
│  ├─ CUDA核心 (数百个)
│  ├─ 共享内存 (Shared Memory)
│  ├─ 寄存器 (Registers)
│  └─ L1缓存
├─ L2缓存
└─ 全局内存 (DRAM, 几GB-几十GB)
```

**NVIDIA GPU架构演进**:
- Kepler (2012)
- Maxwell (2014)
- Pascal (2016)
- Volta (2017) ← Tensor Core引入
- Turing (2018)
- Ampere (2020)
- Hopper (2022)
- Blackwell (2024)

---

### SM (Streaming Multiprocessor)

**SM是GPU的核心计算单元**

**一个典型的SM包含**:
```
- 64-128个CUDA核心
- 共享内存 (48-96 KB)
- 寄存器文件 (64K个32位寄存器)
- L1缓存
- 调度器 (Warp Scheduler)
```

**例子**: NVIDIA A100 GPU
```
- 108个SM
- 每个SM有64个FP32核心
- 总共: 108 × 64 = 6912个CUDA核心
```

---

### 内存层次结构

**从快到慢，从小到大**:

```
1. 寄存器 (Registers)
   - 最快
   - 每个线程私有
   - 容量: 几KB
   - 延迟: 1 cycle

2. 共享内存 (Shared Memory)
   - 很快
   - Block内线程共享
   - 容量: 48-96 KB
   - 延迟: ~20 cycles

3. L1缓存
   - 快
   - SM内自动管理
   - 容量: 几十KB

4. L2缓存
   - 较快
   - 所有SM共享
   - 容量: 几MB

5. 全局内存 (Global Memory)
   - 慢
   - 所有线程可访问
   - 容量: 几GB-几十GB (如16GB, 40GB, 80GB)
   - 延迟: 200-800 cycles
```

**类比**:
```
寄存器 = 手边的笔记本 (立刻能用)
共享内存 = 办公桌上的文件 (很快拿到)
L1/L2缓存 = 办公室书架 (走几步)
全局内存 = 图书馆 (要走一段路)
```

---

### CPU vs GPU架构对比

```
CPU (Intel Core i9):
┌─────────────────────┐
│  核心1  核心2       │
│  ┌───┐  ┌───┐      │
│  │ALU│  │ALU│      │ ← 复杂的ALU
│  │控制│  │控制│      │ ← 强大的控制单元
│  └───┘  └───┘      │
│  大缓存 (几十MB)     │
└─────────────────────┘

特点: 少而强


GPU (NVIDIA A100):
┌─────────────────────────────────────┐
│ SM1   SM2   SM3   ...   SM108      │
│ ┌──┐  ┌──┐  ┌──┐       ┌──┐       │
│ │64│  │64│  │64│  ...  │64│       │ ← 简单的CUDA核心
│ │核│  │核│  │核│       │核│       │
│ └──┘  └──┘  └──┘       └──┘       │
│ 小共享内存 + 大全局内存             │
└─────────────────────────────────────┘

特点: 多而简单
```

---

## 三、CUDA编程模型

### 什么是CUDA？

**CUDA (Compute Unified Device Architecture)** 是NVIDIA开发的并行计算平台和编程模型。

**CUDA让你能够**:
- 用C/C++编写GPU代码
- 控制GPU的线程和内存
- 实现高性能并行计算

**CUDA程序的结构**:
```
主机代码 (Host Code) → 运行在CPU上
设备代码 (Device Code) → 运行在GPU上
```

---

### 核心概念: Thread, Block, Grid

**CUDA的线程组织是三层结构**:

```
Grid (网格)
├─ Block 0
│  ├─ Thread 0
│  ├─ Thread 1
│  ├─ Thread 2
│  └─ ...
├─ Block 1
│  ├─ Thread 0
│  ├─ Thread 1
│  └─ ...
└─ Block N
   └─ ...
```

**1. Thread (线程)**:
- 最小执行单元
- 执行相同的代码（Kernel函数）
- 每个线程有唯一的ID

**2. Block (线程块)**:
- 一组线程的集合
- 同一个Block内的线程可以:
  - 共享内存（Shared Memory）
  - 同步（__syncthreads()）
- Block内线程数: 通常32的倍数，最多1024个

**3. Grid (网格)**:
- 所有Block的集合
- 一个Kernel启动对应一个Grid

---

### 线程索引计算

**1D索引**:
```cuda
// 每个Block有256个线程
// 总共有4个Block

int idx = blockIdx.x * blockDim.x + threadIdx.x;

例子:
Block 0, Thread 5  → idx = 0*256 + 5 = 5
Block 1, Thread 100 → idx = 1*256 + 100 = 356
Block 3, Thread 200 → idx = 3*256 + 200 = 968
```

**可视化**:
```
Block 0:    [0  1  2  ...  255]
Block 1:    [256  257  ...  511]
Block 2:    [512  513  ...  767]
Block 3:    [768  769  ...  1023]
             ↑
           threadIdx.x = 0, blockIdx.x = 3
           idx = 3*256 + 0 = 768
```

**2D索引**:
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;  // 转为1D索引

用途: 图像处理
```

---

### Kernel函数

**Kernel是运行在GPU上的函数**

**定义**:
```cuda
__global__ void kernel_name(参数) {
    // GPU代码
}

__global__: 这是一个Kernel函数
```

**调用**:
```cuda
// 在CPU代码中调用
kernel_name<<<gridSize, blockSize>>>(参数);
           ↑             ↑
         Block数量    每个Block的线程数
```

**例子**:
```cuda
// 定义Kernel
__global__ void add(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 调用Kernel
int N = 1024;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // 向上取整

add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
```

---

## 四、CUDA编程实战

### Hello World - 向量加法

**完整代码**:

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel函数: 在GPU上运行
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // 计算当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 1000000;  // 100万个元素
    size_t size = N * sizeof(float);
    
    // 1. 在CPU上分配内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 2. 在GPU上分配内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 3. 将数据从CPU复制到GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 4. 启动Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // 5. 将结果从GPU复制回CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 6. 验证结果
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %.0f (expected %.0f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }
    
    // 7. 释放内存
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

**编译和运行**:
```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

---

### CUDA编程流程

```
1. 在CPU上分配内存 (malloc)
   ↓
2. 在GPU上分配内存 (cudaMalloc)
   ↓
3. 将数据从CPU复制到GPU (cudaMemcpy H→D)
   ↓
4. 启动Kernel <<<grid, block>>>
   ↓
5. 等待GPU完成 (cudaDeviceSynchronize, 可选)
   ↓
6. 将结果从GPU复制回CPU (cudaMemcpy D→H)
   ↓
7. 释放GPU内存 (cudaFree)
   ↓
8. 释放CPU内存 (free)
```

---

### 内存操作详解

**cudaMalloc - 在GPU上分配内存**:
```cuda
float *d_a;
cudaMalloc(&d_a, N * sizeof(float));
//          ↑     ↑
//        指针地址  大小(字节)
```

**cudaMemcpy - 复制数据**:
```cuda
// CPU → GPU
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
//         ↑    ↑    ↑     ↑
//        目标  源  大小   方向

// GPU → CPU
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

// GPU → GPU
cudaMemcpy(d_a, d_b, size, cudaMemcpyDeviceToDevice);
```

**cudaFree - 释放GPU内存**:
```cuda
cudaFree(d_a);
```

---

### 性能对比

**CPU版本** (单线程):
```c
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// N = 10,000,000
// 时间: ~50ms
```

**GPU版本**:
```cuda
// 相同的数据量
// 时间: ~2ms

加速比: 50 / 2 = 25倍！
```

---

## 五、Warp和线程执行

### 什么是Warp？

**Warp** 是GPU执行的基本单位

**关键点**:
- 1 Warp = 32个线程
- Warp内的32个线程**同时**执行**相同**的指令
- 这叫做SIMT (Single Instruction, Multiple Threads)

**例子**:
```
Block有256个线程 → 分为8个Warp
- Warp 0: Thread 0-31
- Warp 1: Thread 32-63
- ...
- Warp 7: Thread 224-255
```

---

### Warp分歧 (Warp Divergence)

**问题**: 如果Warp内的线程执行不同的路径，会怎样？

**例子**:
```cuda
__global__ void kernel(int *data) {
    int idx = threadIdx.x;
    
    if (idx % 2 == 0) {
        // 偶数线程: 路径A
        data[idx] = data[idx] * 2;
    } else {
        // 奇数线程: 路径B
        data[idx] = data[idx] + 1;
    }
}
```

**Warp内的线程**:
```
Thread 0, 2, 4, ... → 执行路径A
Thread 1, 3, 5, ... → 执行路径B

Warp需要:
1. 先让偶数线程执行A，奇数线程等待
2. 再让奇数线程执行B，偶数线程等待

结果: 串行执行，性能降低！
```

**避免Warp分歧的建议**:
```cuda
// 不好: 分歧发生在Warp内
if (threadIdx.x % 2 == 0) { ... }

// 好: 整个Warp走同一路径
if (blockIdx.x % 2 == 0) { ... }
```

---

## 六、共享内存优化

### 为什么需要共享内存？

**问题**: 全局内存很慢（200+ cycles延迟）

**解决**: 使用共享内存（~20 cycles延迟）

**使用场景**: Block内的线程需要共享数据

---

### 矩阵乘法优化示例

**朴素版本** (只用全局内存):
```cuda
__global__ void matMulNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
        //     ↑ 访问全局内存     ↑ 访问全局内存
    }
    C[row * N + col] = sum;
}

// 每个元素计算需要2N次全局内存访问
// 非常慢！
```

**优化版本** (使用共享内存):
```cuda
__global__ void matMulShared(float *A, float *B, float *C, int N) {
    // 在Block内共享的内存
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 分块计算
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // 1. 协作加载数据到共享内存
        s_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        
        // 2. 同步：确保所有线程都加载完成
        __syncthreads();
        
        // 3. 从共享内存计算（快！）
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
            //     ↑ 共享内存        ↑ 共享内存
        }
        
        // 4. 同步：确保所有线程都计算完成
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}

// 性能提升: 5-10倍！
```

**关键点**:
- `__shared__`: 声明共享内存
- `__syncthreads()`: Block内线程同步

---

## 七、PyTorch与CUDA

### PyTorch自动使用GPU

**PyTorch已经封装好了CUDA操作**，你不需要写CUDA代码！

```python
import torch

# 检查GPU
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # 'NVIDIA A100'

# 创建tensor并移到GPU
x = torch.randn(1000, 1000)
x = x.cuda()  # 或 x.to('cuda')

# 在GPU上计算
y = torch.randn(1000, 1000).cuda()
z = x @ y  # 矩阵乘法，自动在GPU上运行

# 移回CPU
z_cpu = z.cpu()
```

---

### PyTorch的GPU内存管理

**查看GPU内存**:
```python
print(f"已分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"已缓存: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 清空缓存
torch.cuda.empty_cache()
```

**常见错误**: Out of Memory (OOM)
```python
# 问题: Batch size太大
batch_size = 1024  # 太大！GPU内存不够

# 解决1: 减小batch size
batch_size = 128

# 解决2: 梯度累积
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps  # 缩放损失
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### 自定义CUDA扩展

**如果PyTorch的操作不够用，可以写自定义CUDA Kernel**

```python
# custom_cuda.cu
#include <torch/extension.h>

__global__ void my_kernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2;  // 自定义操作
    }
}

torch::Tensor my_function(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    
    int threads = 256;
    int blocks = (input.size(0) + threads - 1) / threads;
    
    my_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0)
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_function", &my_function);
}
```

**编译和使用**:
```python
from torch.utils.cpp_extension import load

custom_cuda = load(
    name='custom_cuda',
    sources=['custom_cuda.cu'],
    verbose=True
)

# 使用
import torch
x = torch.randn(1000).cuda()
y = custom_cuda.my_function(x)
```

---

## 八、并行计算原理

### 数据并行 vs 任务并行

**数据并行 (Data Parallelism)**:
```
相同的操作，应用到不同的数据

例子: 向量加法
数据: [1, 2, 3, 4, 5, 6, 7, 8]
操作: 每个元素 +1

并行:
Thread 0: 1+1 → 2
Thread 1: 2+1 → 3
Thread 2: 3+1 → 4
...
同时执行！
```

**任务并行 (Task Parallelism)**:
```
不同的操作，同时执行

例子:
任务A: 图像增强
任务B: 视频编码
任务C: 音频处理

并行:
Core 0: 执行任务A
Core 1: 执行任务B
Core 2: 执行任务C
```

**深度学习主要用数据并行**:
- 同一个模型应用到不同的样本
- 完美适合GPU

---

### Amdahl定律

**并行加速的理论极限**

**公式**:
```
加速比 = 1 / [(1-P) + P/N]

P: 可并行部分的比例
N: 处理器数量
```

**例子**:
```
程序: 90%可并行 (P=0.9)，10%必须串行
使用100个核心 (N=100)

加速比 = 1 / [(1-0.9) + 0.9/100]
       = 1 / [0.1 + 0.009]
       = 1 / 0.109
       ≈ 9.17倍

即使有100个核心，只能加速9.17倍！
瓶颈: 10%的串行部分
```

**启示**:
- 并行加速有上限
- 要最大化可并行的部分
- 减少串行瓶颈（如数据传输）

---

### 性能优化策略

**1. 最大化并行度**
```cuda
// 不好: 串行
for (int i = 0; i < N; i++) {
    process(data[i]);
}

// 好: 并行
__global__ void kernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        process(data[idx]);
    }
}
```

**2. 最小化内存传输**
```cuda
// 不好: 频繁传输
for (int i = 0; i < 1000; i++) {
    cudaMemcpy(d_data, h_data, size, H2D);
    kernel<<<...>>>(d_data);
    cudaMemcpy(h_result, d_result, size, D2H);
}

// 好: 一次传输
cudaMemcpy(d_data, h_data, size, H2D);
for (int i = 0; i < 1000; i++) {
    kernel<<<...>>>(d_data);
}
cudaMemcpy(h_result, d_result, size, D2H);
```

**3. 使用共享内存**
```cuda
// 减少全局内存访问
__shared__ float s_data[BLOCK_SIZE];
```

**4. 合并内存访问 (Coalesced Access)**
```cuda
// 好: 连续访问
int idx = threadIdx.x;
float val = data[idx];  // Thread 0→data[0], Thread 1→data[1], ...

// 不好: 跨步访问
int idx = threadIdx.x * stride;
float val = data[idx];  // 不连续
```

---

## 九、GPU编程的挑战

### 1. 调试困难

**问题**:
- GPU代码在设备上运行，难以调试
- 没有print很难看到中间状态
- 崩溃信息不清晰

**解决**:
```cuda
// 使用printf (CUDA支持)
__global__ void kernel(int *data) {
    int idx = threadIdx.x;
    if (idx == 0) {  // 只让一个线程打印
        printf("Block %d, Thread %d: %d\n", blockIdx.x, idx, data[idx]);
    }
}

// 使用cuda-gdb调试器
cuda-gdb ./program

// 检查错误
cudaError_t err = cudaMemcpy(...);
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

---

### 2. 竞态条件 (Race Condition)

**问题**: 多个线程同时修改同一内存

```cuda
__global__ void bad_kernel(int *counter) {
    // 多个线程同时执行:
    int temp = *counter;  // 读
    temp = temp + 1;      // 修改
    *counter = temp;      // 写
    
    // 竞态条件！最终结果不确定
}
```

**解决**: 原子操作
```cuda
__global__ void good_kernel(int *counter) {
    atomicAdd(counter, 1);  // 原子加法，线程安全
}

// 其他原子操作:
// atomicSub, atomicMin, atomicMax, atomicCAS, ...
```

---

### 3. 内存访问模式

**Bank Conflict (共享内存)**:

共享内存分为32个bank，如果同一Warp的线程访问同一bank的不同地址，会串行化。

```cuda
// 不好: Bank conflict
__shared__ float s_data[32][32];
float val = s_data[threadIdx.x][threadIdx.x];  // 冲突！

// 好: 无冲突
float val = s_data[threadIdx.x][0];  // 不同bank
```

---

## 十、实际应用案例

### 案例1: 图像模糊

**CPU版本**:
```c
void blurCPU(unsigned char *input, unsigned char *output, 
             int width, int height) {
    for (int y = 1; y < height-1; y++) {
        for (int x = 1; x < width-1; x++) {
            int sum = 0;
            // 3x3平均
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += input[(y+dy)*width + (x+dx)];
                }
            }
            output[y*width + x] = sum / 9;
        }
    }
}

// 1920×1080图像: ~100ms
```

**GPU版本**:
```cuda
__global__ void blurGPU(unsigned char *input, unsigned char *output,
                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += input[(y+dy)*width + (x+dx)];
            }
        }
        output[y*width + x] = sum / 9;
    }
}

// 调用
dim3 threadsPerBlock(16, 16);
dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);
blurGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

// 1920×1080图像: ~5ms
// 加速比: 20倍！
```

---

### 案例2: 矩阵乘法性能对比

```
矩阵大小: 4096 × 4096

CPU (单核):        ~60秒
CPU (多核, 8核):  ~10秒
GPU (朴素版):      ~2秒
GPU (优化版):      ~0.2秒
cuBLAS (NVIDIA优化): ~0.05秒

最快vs最慢: 1200倍加速！
```

---

## 十一、GPU编程最佳实践

### 1. 何时使用GPU？

**适合GPU**:
✅ 大规模数据并行（数百万个元素）
✅ 计算密集型（大量运算）
✅ 规则的内存访问模式
✅ 数据可以一次性传输

**不适合GPU**:
❌ 小数据量（几百个元素）
❌ 复杂的控制流（大量if/else）
❌ 不规则的内存访问
❌ 频繁的CPU-GPU数据传输

---

### 2. 性能优化Checklist

```
□ 使用足够的线程（至少几千个）
□ 每个Block 128-256个线程（32的倍数）
□ 最小化CPU-GPU数据传输
□ 使用共享内存减少全局内存访问
□ 避免Warp分歧
□ 合并内存访问
□ 使用异步操作和Stream
□ 分析性能（nvprof, Nsight）
```

---

### 3. 常用工具

**性能分析**:
```bash
# nvprof (旧版)
nvprof ./program

# Nsight Systems (推荐)
nsys profile ./program

# Nsight Compute (详细分析)
ncu ./program
```

**库**:
- **cuBLAS**: 矩阵运算
- **cuDNN**: 深度学习原语
- **Thrust**: C++ STL for GPU
- **CUB**: CUDA基础算法

---

## 十二、面试高频问题及回答

### Q1: GPU相比CPU有什么优势？为什么深度学习需要GPU？

**回答框架**:
"GPU的优势是大规模并行计算能力。CPU有少数强大的核心（通常几十个），适合串行和复杂逻辑；GPU有数千个简单核心，适合并行的重复计算。深度学习的核心是矩阵运算，比如一个1000×2000的矩阵乘法涉及几十亿次计算，这些计算是独立的、可以并行。GPU可以同时计算数千个元素，获得100-1000倍的加速。训练大模型时，GPU能把几年的训练时间缩短到几个月，成本节省巨大。"

---

### Q2: 解释CUDA的线程组织：Thread、Block、Grid

**回答框架**:
"CUDA用三层结构组织线程。Thread是最小执行单元，执行相同的Kernel代码。Block是一组线程（通常几百个），同一Block内的线程可以共享内存和同步。Grid是所有Block的集合，一个Kernel启动对应一个Grid。这种层次设计的原因是：Block内线程可以高效协作（共享内存），而不同Block独立执行，可以映射到不同的SM并行。线程的全局索引通常计算为 blockIdx.x * blockDim.x + threadIdx.x。"

---

### Q3: 什么是Warp？什么是Warp Divergence？

**回答框架**:
"Warp是GPU执行的基本单位，1个Warp包含32个线程。Warp内的线程采用SIMT模式，同时执行相同的指令。Warp Divergence指Warp内的线程执行不同的代码路径，比如if-else的不同分支。发生分歧时，GPU必须串行执行各个分支，性能降低。避免方法是让判断基于blockIdx而不是threadIdx，确保整个Warp走同一路径。这是GPU编程性能优化的重要考虑点。"

---

### Q4: CUDA的内存层次是怎样的？如何优化内存访问？

**回答框架**:
"CUDA内存从快到慢是：寄存器（每线程私有，最快）、共享内存（Block内共享，很快）、L1/L2缓存、全局内存（所有线程可访问，最慢）。优化策略：1) 尽量使用寄存器存储临时变量；2) 用共享内存缓存频繁访问的数据，减少全局内存访问；3) 保证合并内存访问，让Warp内的线程访问连续地址；4) 避免Bank Conflict。实际中，把热数据从全局内存加载到共享内存能带来5-10倍性能提升。"

---

### Q5: PyTorch如何使用GPU？

**回答框架**:
"PyTorch使用GPU非常简单。首先检查GPU可用性 torch.cuda.is_available()，然后用 .cuda() 或 .to('cuda') 把tensor移到GPU。模型也需要移到GPU。PyTorch会自动在GPU上执行运算，用户不需要写CUDA代码。注意事项：1) 确保数据和模型在同一设备；2) 注意GPU内存限制，可能需要减小batch size或使用梯度累积；3) 可以用 torch.cuda.empty_cache() 清理缓存。实际项目中99%的情况不需要自己写CUDA，PyTorch已经高度优化。"

---

### Q6: 如何调试CUDA程序？

**回答框架**:
"CUDA调试确实比CPU困难。方法有：1) 在Kernel中使用printf，但只让少数线程打印避免输出过多；2) 检查每个CUDA API的返回值，用cudaGetErrorString获取错误信息；3) 使用cuda-gdb调试器；4) 用Nsight工具进行可视化调试和性能分析；5) 先在CPU上实现并验证算法，再移植到GPU对比结果。常见错误包括索引越界、未同步、内存未正确释放等。养成检查错误的习惯很重要。"

---

### Q7: 什么情况下不应该使用GPU？

**回答框架**:
"GPU不是银弹。不适合GPU的情况：1) 数据量太小（几百个元素），GPU启动开销大于计算收益；2) 算法串行性强，无法并行；3) 复杂的控制流（大量分支），导致Warp分歧；4) 需要频繁CPU-GPU数据传输，传输时间超过计算时间；5) 内存访问不规则，无法合并。这些情况CPU可能更快。选择时要profiling对比，不要想当然认为GPU一定快。"

---

### Q8: Amdahl定律告诉我们什么？

**回答框架**:
"Amdahl定律说明并行加速有理论上限，公式是 加速比 = 1/[(1-P) + P/N]，其中P是可并行比例，N是处理器数。关键洞察是：如果10%的代码必须串行，即使有无限多核心，加速比最多10倍。启示：1) 要最大化可并行部分；2) 消除串行瓶颈（如CPU-GPU传输）比增加核心数更重要；3) 实际加速比通常远小于核心数。这解释了为什么GPU有几千核心但只能加速几十到几百倍。"

---

## 十三、CUDA与Day1-7的联系

### 与前几天学习的关系

**Day4 - PyTorch**:
```python
# PyTorch底层就是CUDA
import torch

x = torch.randn(1000, 1000).cuda()  # ← 调用CUDA
y = torch.randn(1000, 1000).cuda()
z = x @ y  # ← 底层调用cuBLAS (CUDA库)

# PyTorch的autograd也在GPU上运行
loss.backward()  # ← 在GPU上计算梯度
```

**Day5 - Transformer**:
```python
# Transformer的Self-Attention是矩阵运算
# 完美适合GPU并行

# Attention(Q, K, V) = softmax(QK^T / √d) V
# 每个注意力头可以并行计算
# GPU加速: 100+倍
```

**Day6 - 机器学习**:
```python
# XGBoost有GPU版本
import xgboost as xgb

model = xgb.XGBRegressor(tree_method='gpu_hist')  # GPU加速
model.fit(X_train, y_train)

# 大数据集训练: GPU快10-50倍
```

---

### 为什么NVIDIA对AI如此重要？

**历史**:
```
2006: CUDA发布
2012: AlexNet在ImageNet夺冠（用GPU训练）
2015: 深度学习爆发，GPU需求激增
2017: Tensor Core（专为AI优化）
2023: ChatGPT需要数千张A100/H100 GPU
```

**现状**:
- NVIDIA占据AI芯片市场>90%
- 一张H100 GPU售价$30,000+
- 训练GPT-4级模型需要数万张GPU
- NVIDIA市值>$3万亿（2024）

**技术护城河**:
- CUDA生态完善
- cuDNN、TensorRT高度优化
- 所有AI框架都支持CUDA
- 竞争对手（AMD、Intel）生态弱很多

---

## 十四、核心概念速记卡

### GPU vs CPU

**CPU**: 少而强，复杂逻辑  
**GPU**: 多而简单，并行计算

---

### CUDA三层结构

**Grid → Block → Thread**

```
Thread: 最小单元
Block: 可协作的线程组
Grid: 所有Block
```

---

### 内存层次

**快→慢**: 寄存器 > 共享内存 > L1/L2 > 全局内存

---

### Kernel调用

```cuda
kernel<<<blocks, threads>>>(args);
```

---

### 线程索引

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

---

### CUDA编程流程

**malloc → cudaMalloc → cudaMemcpy(H2D) → kernel → cudaMemcpy(D2H) → free**

---

## 十五、学习检查清单

完成以下自测，确保理解：

- [ ] 能解释GPU相比CPU的优势
- [ ] 理解Thread、Block、Grid的概念
- [ ] 知道如何计算线程的全局索引
- [ ] 能写简单的CUDA Kernel函数
- [ ] 理解CUDA的内存层次
- [ ] 知道什么是Warp和Warp Divergence
- [ ] 理解共享内存的作用
- [ ] 知道PyTorch如何使用GPU
- [ ] 能回答面试高频问题
- [ ] 理解GPU在AI中的重要性

---

## 十六、扩展学习资源

### 官方资源
- **NVIDIA CUDA C Programming Guide**（必读）
- **CUDA by Example** (书籍)
- **NVIDIA Deep Learning Institute** (免费课程)

### 在线课程
- Coursera: Introduction to Parallel Programming (Udacity)
- YouTube: CUDA Crash Course

### 实践平台
- **Google Colab**: 免费GPU（T4/V100）
- **Kaggle Notebooks**: 免费GPU
- **NVIDIA NGC**: 优化的容器

### 性能分析工具
- Nsight Systems
- Nsight Compute
- nvprof

---

## 十七、Day8 学习总结

### 今天学到了什么

✅ **GPU架构基础**
- CPU vs GPU的本质区别
- SM、CUDA核心、内存层次

✅ **CUDA编程模型**
- Thread、Block、Grid三层结构
- Kernel函数的编写和调用
- 内存管理（cudaMalloc、cudaMemcpy）

✅ **性能优化**
- Warp和SIMT执行模型
- 共享内存的使用
- 内存访问模式优化

✅ **实际应用**
- 向量加法、矩阵乘法、图像处理
- PyTorch的GPU使用
- 性能对比和加速比

✅ **最佳实践**
- 何时使用GPU
- 调试方法
- 常见陷阱

---

### 与前几天的完整知识体系

```
Day1-3: AI应用层
├─ Agent、RAG、编排

Day4-5: 深度学习层
├─ PyTorch框架
└─ Transformer架构

Day6: 传统机器学习
├─ 分类、回归、聚类
└─ 特征工程

Day7: 强化学习
└─ Q-Learning、DQN、Policy Gradient

Day8: 硬件加速层 ⭐
├─ GPU架构
├─ CUDA编程
└─ 性能优化

完整的AI技术栈！
从应用到硬件，从算法到工程
```

---

### 为什么GPU对AI工程师重要？

**理解层面**:
- 知道为什么深度学习需要GPU
- 理解PyTorch操作的底层原理
- 能够分析性能瓶颈

**实践层面**:
- 优化训练速度
- 处理GPU内存不足
- 选择合适的硬件

**面试层面**:
- NVIDIA等公司必问
- 体现工程深度
- 区分初级和高级工程师

---

### 下一步建议

**Day9-10: 项目整合**
1. 构建端到端的AI系统
2. 结合GPU加速优化
3. 准备面试作品集

**实践建议**:
1. 在Google Colab上运行CUDA代码
2. 分析PyTorch模型的GPU性能
3. 优化一个实际的训练流程

**深入学习**（长期）:
1. 深入学习cuDNN、TensorRT
2. 学习多GPU编程（NCCL）
3. 关注新的GPU架构（Hopper、Blackwell）

---

**最后更新**: 2026-08-29

**恭喜你完成Day8的学习！你现在理解了AI加速的底层原理！** 🚀

**你已经掌握从应用到硬件的完整AI技术栈，准备好面试了！**
