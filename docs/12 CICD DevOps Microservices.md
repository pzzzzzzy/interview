# CI/CD + DevOps + 微服务 核心知识储备

---

## 一、为什么AI工程师需要DevOps知识？

### AI项目中的工程挑战

**传统观念 vs 现实**:

```
传统观念:
AI工程师 = 调模型 + 写代码
→ 在Jupyter Notebook里做实验就够了

现实:
✓ 模型训练需要GPU资源调度
✓ 模型需要部署到生产环境
✓ 需要监控模型性能和漂移
✓ 需要A/B测试和灰度发布
✓ 需要自动化训练pipeline
✓ 需要版本控制（代码+数据+模型）
```

**实际场景**:

```
场景1: 模型部署
- 本地训练好的模型怎么上线？
- 如何保证环境一致性（Docker）？
- 如何做到快速回滚？
- 如何做灰度发布（新模型只给10%用户）？

场景2: 自动化训练
- 数据更新后自动触发训练
- 训练完成后自动评估
- 评估通过后自动部署
- 全程不需要人工介入

场景3: 多模型管理
- 同时维护多个模型版本
- 不同用户使用不同模型
- 监控各版本性能
- 快速切换/回滚

场景4: 大规模服务
- 模型推理QPS达到10000+
- 需要负载均衡
- 需要自动扩缩容
- 需要容错和降级
```

**面试中的考察**:
- Docker和容器化
- CI/CD流程设计
- 微服务架构理解
- Kubernetes基础
- 监控和日志
- 实际项目经验

---

## 二、Docker与容器化

### 2.1 什么是Docker？

#### 核心概念

**问题**: 环境不一致

```
开发环境（你的电脑）:
- Python 3.8
- PyTorch 1.10
- CUDA 11.3

生产环境（服务器）:
- Python 3.9  ← 不同
- PyTorch 1.12  ← 不同
- CUDA 11.7  ← 不同

结果: "在我电脑上能跑啊！" 😱
```

**Docker的解决方案**: 把应用和环境一起打包

```
Docker镜像 = 应用 + 依赖 + 运行时环境

就像一个"集装箱":
- 里面是完整的应用
- 外面标准化接口
- 可以在任何地方运行（只要有Docker）
```

---

#### Docker vs 虚拟机

```
虚拟机 (VM):
┌─────────────────────┐
│   App A   │  App B  │
├───────────┼─────────┤
│  OS 1     │  OS 2   │  ← 每个都有完整的操作系统
├───────────┴─────────┤
│   Hypervisor        │
├─────────────────────┤
│   Host OS           │
├─────────────────────┤
│   Hardware          │
└─────────────────────┘

特点:
- 完整的OS，占用资源多（GB级）
- 启动慢（分钟级）
- 隔离性强

Docker容器:
┌─────────────────────┐
│   App A   │  App B  │
├───────────┼─────────┤
│  Docker Engine      │  ← 共享Host OS内核
├─────────────────────┤
│   Host OS           │
├─────────────────────┤
│   Hardware          │
└─────────────────────┘

特点:
- 共享OS内核，占用资源少（MB级）
- 启动快（秒级）
- 轻量级
```

---

### 2.2 Docker核心概念

#### 镜像 (Image)

**定义**: 只读的模板，包含运行应用所需的一切

```
类比: 镜像 = 类（Class）

特点:
- 只读（不可修改）
- 分层结构（每层可以复用）
- 可以从镜像创建容器
```

**镜像的分层结构**:

```
我的AI应用镜像:
┌──────────────────────┐
│ app.py, model.pth    │ ← 你的应用层
├──────────────────────┤
│ pip install torch    │ ← 依赖层
├──────────────────────┤
│ Python 3.9           │ ← 运行时层
├──────────────────────┤
│ Ubuntu 20.04         │ ← 基础OS层
└──────────────────────┘

优势:
- 每层可以缓存和复用
- 修改代码只需要重建最上层
- 节省存储空间和传输时间
```

---

#### 容器 (Container)

**定义**: 镜像的运行实例

```
类比: 容器 = 对象（Object）

特点:
- 可读写（在镜像基础上添加了可写层）
- 可以启动、停止、删除
- 相互隔离
- 一个镜像可以创建多个容器
```

**容器的生命周期**:

```
创建 → 运行 → 暂停 → 停止 → 删除
  ↓      ↓      ↓      ↓      ↓
 新建  工作中  休眠  已停止  清理
```

---

#### Dockerfile

**定义**: 构建镜像的配方（脚本）

**AI项目的Dockerfile示例**:

```dockerfile
# 基础镜像（包含Python和CUDA）
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装Python
RUN apt-get update && apt-get install -y python3-pip

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "app.py"]
```

**Dockerfile指令详解**:

```dockerfile
FROM：指定基础镜像
  FROM python:3.9
  FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR：设置工作目录
  WORKDIR /app
  # 后续命令都在/app目录下执行

COPY：复制文件到镜像
  COPY requirements.txt .
  COPY . .  # 复制当前目录所有文件

RUN：执行命令（构建时）
  RUN apt-get update
  RUN pip install torch

CMD：容器启动时执行的命令
  CMD ["python", "app.py"]
  # 只能有一个CMD

ENTRYPOINT：容器入口点
  ENTRYPOINT ["python"]
  CMD ["app.py"]
  # 结合使用，灵活性更高

ENV：设置环境变量
  ENV MODEL_PATH=/models/bert.pth
  ENV CUDA_VISIBLE_DEVICES=0

EXPOSE：声明端口
  EXPOSE 8000
  # 只是声明，实际映射需要-p参数

VOLUME：挂载卷
  VOLUME /data
  # 持久化数据
```

---

### 2.3 Docker常用命令

#### 镜像管理

```bash
# 构建镜像
docker build -t my-ai-app:v1.0 .
# -t: 指定镜像名和标签
# .: Dockerfile所在目录

# 列出镜像
docker images
# REPOSITORY    TAG       IMAGE ID       SIZE
# my-ai-app     v1.0      abc123...      2.5GB

# 删除镜像
docker rmi my-ai-app:v1.0

# 从远程仓库拉取镜像
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 推送镜像到远程仓库
docker push myusername/my-ai-app:v1.0

# 查看镜像历史（各层）
docker history my-ai-app:v1.0
```

---

#### 容器管理

```bash
# 运行容器
docker run -d \
  --name my-app \
  -p 8000:8000 \
  -v /data:/app/data \
  --gpus all \
  my-ai-app:v1.0

# 参数说明:
# -d: 后台运行（detached）
# --name: 容器名称
# -p 主机端口:容器端口（端口映射）
# -v 主机路径:容器路径（挂载卷）
# --gpus all: 使用所有GPU
# -e: 设置环境变量，如 -e MODEL_PATH=/models/bert.pth

# 列出运行中的容器
docker ps

# 列出所有容器（包括停止的）
docker ps -a

# 停止容器
docker stop my-app

# 启动已停止的容器
docker start my-app

# 重启容器
docker restart my-app

# 删除容器
docker rm my-app

# 强制删除运行中的容器
docker rm -f my-app

# 查看容器日志
docker logs my-app
docker logs -f my-app  # 实时查看（类似tail -f）

# 进入容器内部（调试用）
docker exec -it my-app /bin/bash
# -i: 交互模式
# -t: 分配伪终端

# 查看容器详细信息
docker inspect my-app

# 查看容器资源使用
docker stats my-app
```

---

#### 实战示例：部署PyTorch模型

**目录结构**:
```
my-model-api/
├── Dockerfile
├── requirements.txt
├── app.py
├── model.pth
└── config.yaml
```

**requirements.txt**:
```
torch==2.0.0
fastapi==0.100.0
uvicorn==0.23.0
transformers==4.30.0
```

**app.py**（简化版）:
```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load('model.pth')
model.eval()

@app.post("/predict")
async def predict(text: str):
    with torch.no_grad():
        result = model(text)
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Dockerfile**:
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
```

**构建和运行**:
```bash
# 1. 构建镜像
docker build -t my-model-api:v1 .

# 2. 运行容器
docker run -d \
  --name model-server \
  -p 8000:8000 \
  --gpus all \
  my-model-api:v1

# 3. 测试
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# 4. 查看日志
docker logs model-server

# 5. 停止并删除
docker stop model-server
docker rm model-server
```

---

### 2.4 Docker Compose

**问题**: 复杂应用需要多个容器协同

```
AI应用架构:
- 模型推理服务（容器1）
- Redis缓存（容器2）
- PostgreSQL数据库（容器3）
- Nginx反向代理（容器4）

手动启动很麻烦:
docker run ... redis
docker run ... postgres
docker run ... model-api
docker run ... nginx
```

**Docker Compose**: 用一个配置文件管理多容器应用

**docker-compose.yml示例**:

```yaml
version: '3.8'

services:
  # 模型API服务
  model-api:
    build: .
    image: my-model-api:v1
    container_name: model-api
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - DB_HOST=postgres
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Redis缓存
  redis:
    image: redis:7-alpine
    container_name: redis
    ports:
      - "6379:6379"

  # PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    container_name: postgres
    environment:
      - POSTGRES_PASSWORD=mypassword
      - POSTGRES_DB=mydb
    volumes:
      - postgres-data:/var/lib/postgresql/data

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    container_name: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - model-api

volumes:
  postgres-data:
```

**Docker Compose命令**:

```bash
# 启动所有服务（后台）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f model-api

# 停止所有服务
docker-compose stop

# 停止并删除所有容器
docker-compose down

# 重新构建镜像
docker-compose build

# 扩展服务（运行多个实例）
docker-compose up -d --scale model-api=3
```

---

## 三、CI/CD持续集成与持续部署

### 3.1 什么是CI/CD？

#### CI (Continuous Integration) - 持续集成

**定义**: 频繁地将代码集成到主分支，并自动测试

**传统方式的问题**:

```
开发者A: 写代码一个月
开发者B: 写代码一个月
开发者C: 写代码一个月

一个月后合并 → 冲突爆炸 💥
→ 花两周解决冲突
→ 集成地狱
```

**CI的方式**:

```
开发者每天提交代码 → 自动构建 → 自动测试

好处:
✓ 早发现问题，早解决
✓ 减少集成冲突
✓ 代码质量有保障
✓ 快速反馈
```

**CI流程**:

```
1. 开发者提交代码 (git push)
   ↓
2. 触发CI Pipeline
   ↓
3. 拉取代码
   ↓
4. 安装依赖
   ↓
5. 运行测试（单元测试、集成测试）
   ↓
6. 代码质量检查（Lint、类型检查）
   ↓
7. 构建（编译、打包、Docker镜像）
   ↓
8. 反馈结果（成功 or 失败）
```

---

#### CD (Continuous Deployment/Delivery) - 持续部署/交付

**两种CD**:

```
Continuous Delivery（持续交付）:
- 代码随时可以部署
- 但需要手动批准才部署到生产

Continuous Deployment（持续部署）:
- 代码自动部署到生产
- 无需人工干预（测试通过即部署）
```

**CD流程**:

```
1. CI通过
   ↓
2. 部署到测试环境
   ↓
3. 自动化测试（端到端测试）
   ↓
4. 部署到预发布环境
   ↓
5. 冒烟测试
   ↓
6. 部署到生产环境
   - 蓝绿部署
   - 金丝雀发布（灰度）
   - 滚动更新
   ↓
7. 监控和告警
```

---

### 3.2 GitHub Actions

**GitHub Actions**: GitHub内置的CI/CD工具

#### 核心概念

```
Workflow（工作流）:
- 自动化流程
- 由一个或多个Job组成
- 用YAML文件定义

Job（任务）:
- Workflow中的一个任务单元
- 可以并行或串行执行
- 在Runner上执行

Step（步骤）:
- Job中的单个操作
- 可以是命令或Action

Action（动作）:
- 可复用的步骤
- 类似函数库
```

---

#### 实战示例：AI模型训练与部署

**目录结构**:
```
.github/
└── workflows/
    ├── ci.yml          # CI流程
    └── deploy.yml      # 部署流程
```

**.github/workflows/ci.yml**:

```yaml
name: CI Pipeline

# 触发条件
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

# 任务
jobs:
  # 任务1: 代码检查和测试
  test:
    runs-on: ubuntu-latest  # 运行环境
    
    steps:
      # 步骤1: 检出代码
      - name: Checkout code
        uses: actions/checkout@v3
      
      # 步骤2: 设置Python环境
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      # 步骤3: 缓存依赖
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
      
      # 步骤4: 安装依赖
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest flake8
      
      # 步骤5: 代码风格检查
      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
      
      # 步骤6: 运行单元测试
      - name: Run tests
        run: |
          pytest tests/ -v --cov=. --cov-report=xml
      
      # 步骤7: 上传覆盖率报告
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  # 任务2: 构建Docker镜像
  build:
    needs: test  # 依赖test任务成功
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: myusername/my-model:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

**.github/workflows/deploy.yml**:

```yaml
name: Deploy to Production

# 触发条件：创建Release时
on:
  release:
    types: [created]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      # 部署到Kubernetes
      - name: Deploy to K8s
        uses: azure/k8s-deploy@v4
        with:
          manifests: |
            k8s/deployment.yaml
            k8s/service.yaml
          images: |
            myusername/my-model:${{ github.sha }}
          kubeconfig: ${{ secrets.KUBE_CONFIG }}
      
      # 通知Slack
      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Deployment to production completed!'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        if: always()
```

---

#### Secrets管理

**问题**: 敏感信息（密码、API Key）不能写在代码里

**解决**: GitHub Secrets

```
设置位置: 
GitHub Repo → Settings → Secrets and variables → Actions

添加Secret:
- DOCKER_USERNAME
- DOCKER_PASSWORD
- KUBE_CONFIG
- SLACK_WEBHOOK

使用:
${{ secrets.DOCKER_USERNAME }}
```

---

### 3.3 GitLab CI/CD

**GitLab CI/CD**: GitLab内置的CI/CD工具

**.gitlab-ci.yml示例**:

```yaml
# 定义阶段
stages:
  - test
  - build
  - deploy

# 定义变量
variables:
  DOCKER_IMAGE: registry.gitlab.com/mygroup/my-model
  KUBE_NAMESPACE: production

# 模板（可复用）
.test_template: &test_template
  image: python:3.9
  before_script:
    - pip install -r requirements.txt

# 任务：单元测试
test:unit:
  <<: *test_template
  stage: test
  script:
    - pytest tests/unit/ -v
  coverage: '/TOTAL.*\s+(\d+%)$/'

# 任务：集成测试
test:integration:
  <<: *test_template
  stage: test
  script:
    - pytest tests/integration/ -v
  only:
    - main
    - develop

# 任务：构建Docker镜像
build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $DOCKER_IMAGE:$CI_COMMIT_SHA .
    - docker push $DOCKER_IMAGE:$CI_COMMIT_SHA
  only:
    - main

# 任务：部署到Staging
deploy:staging:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context staging
    - kubectl set image deployment/my-model my-model=$DOCKER_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/my-model
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

# 任务：部署到Production
deploy:production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context production
    - kubectl set image deployment/my-model my-model=$DOCKER_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/my-model
  environment:
    name: production
    url: https://example.com
  when: manual  # 需要手动触发
  only:
    - main
```

---

### 3.4 CI/CD最佳实践

#### 1. 快速反馈

```
✓ 保持CI Pipeline快速（< 10分钟）
✓ 并行运行测试
✓ 使用缓存加速构建
✓ 失败立即通知

❌ 不要: 20分钟的Pipeline（开发者会失去耐心）
```

---

#### 2. 自动化测试

```
测试金字塔:

    /\
   /UI\       ← 少量（慢、脆弱）
  /────\
 /集成测\    ← 适量（中速）
/────────\
/单元测试 \  ← 大量（快、稳定）
──────────

✓ 单元测试: 快速、覆盖率高
✓ 集成测试: 验证组件协作
✓ 端到端测试: 验证用户流程

目标: 
- 单元测试覆盖率 > 80%
- 关键路径有集成测试
- 核心功能有端到端测试
```

---

#### 3. 环境一致性

```
开发环境 = 测试环境 = 生产环境

✓ 使用Docker保证环境一致
✓ 使用相同的配置管理工具
✓ Infrastructure as Code (IaC)

❌ 不要: 生产环境有的东西，测试环境没有
```

---

#### 4. 分支策略

**Git Flow**:

```
main（生产）
  ↓
develop（开发）
  ↓
feature/xxx（功能分支）

工作流程:
1. 从develop创建feature分支
2. 开发完成，PR到develop
3. CI自动测试
4. 合并到develop，部署到测试环境
5. 测试通过，PR到main
6. 部署到生产环境
```

**Trunk-Based Development**:

```
main（主干）
  ↓
short-lived branches（短期分支）

工作流程:
1. 从main创建短期分支（< 2天）
2. 小步提交，频繁合并
3. 使用Feature Flag控制功能发布
4. 持续部署
```

---

#### 5. 部署策略

**蓝绿部署**:

```
蓝色环境（旧版本，当前服务）
绿色环境（新版本，待切换）

流程:
1. 绿色环境部署新版本
2. 测试绿色环境
3. 切换流量：蓝色 → 绿色
4. 观察监控
5. 如果有问题，切回蓝色（快速回滚）

优点: 零停机，快速回滚
缺点: 需要双倍资源
```

**金丝雀发布（灰度发布）**:

```
旧版本: 90%流量
新版本: 10%流量

流程:
1. 新版本部署到10%服务器
2. 观察指标（错误率、延迟）
3. 逐步增加：10% → 25% → 50% → 100%
4. 发现问题随时回滚

优点: 渐进式，风险小
缺点: 实现复杂，需要流量控制
```

**滚动更新**:

```
10台服务器，每次更新2台

流程:
1. 停止2台旧版本
2. 启动2台新版本
3. 等待健康检查通过
4. 重复步骤1-3
5. 直到所有服务器更新完成

优点: 无需额外资源
缺点: 更新慢，两个版本同时存在
```

---

## 四、微服务架构

### 4.1 单体 vs 微服务

#### 单体架构 (Monolithic)

**特点**: 所有功能在一个应用里

```
┌─────────────────────────────┐
│        单体应用              │
│  ┌────────────────────────┐ │
│  │  用户管理               │ │
│  ├────────────────────────┤ │
│  │  订单处理               │ │
│  ├────────────────────────┤ │
│  │  支付系统               │ │
│  ├────────────────────────┤ │
│  │  推荐算法               │ │
│  └────────────────────────┘ │
│                             │
│  共享数据库                  │
└─────────────────────────────┘

优点:
✓ 开发简单（一个代码库）
✓ 部署简单（一个部署包）
✓ 测试简单（一起测）

缺点:
❌ 难以扩展（必须整体扩展）
❌ 技术栈锁定（全部用同一语言）
❌ 维护困难（改一处可能影响全局）
❌ 部署风险大（一处出错全崩）
```

---

#### 微服务架构 (Microservices)

**特点**: 拆分成多个独立的小服务

```
┌──────────┐   ┌──────────┐   ┌──────────┐
│ 用户服务  │   │ 订单服务  │   │ 支付服务  │
│ (Python) │   │ (Java)   │   │ (Go)     │
│  DB1     │   │  DB2     │   │  DB3     │
└──────────┘   └──────────┘   └──────────┘
     ↓              ↓              ↓
     └──────────────┴──────────────┘
              API Gateway
                  ↓
              负载均衡
                  ↓
                用户

优点:
✓ 独立扩展（只扩展需要的服务）
✓ 技术自由（每个服务选合适的技术）
✓ 团队自治（不同团队负责不同服务）
✓ 故障隔离（一个服务挂了不影响其他）
✓ 快速部署（只部署变更的服务）

缺点:
❌ 复杂度高（分布式系统）
❌ 运维成本高（多个服务要管理）
❌ 调试困难（跨服务追踪）
❌ 数据一致性（分布式事务）
```

---

### 4.2 微服务核心概念

#### 服务拆分原则

**1. 单一职责原则**

```
✓ 每个服务只做一件事
✓ 服务边界清晰

例子:
- 用户服务: 只管用户注册、登录、资料
- 订单服务: 只管订单创建、查询、取消
- 支付服务: 只管支付处理

❌ 不要: 用户服务里写订单逻辑
```

**2. 按业务能力拆分**

```
电商系统拆分:
- 商品目录服务
- 购物车服务
- 订单服务
- 支付服务
- 物流服务
- 用户服务

AI平台拆分:
- 模型训练服务
- 模型推理服务
- 数据预处理服务
- 特征工程服务
- 模型管理服务
```

**3. 避免过度拆分**

```
❌ 太细: 每个API一个服务（运维噩梦）
✓ 合理: 一个业务领域一个服务
```

---

#### 服务间通信

**1. 同步通信 - REST API**

```
服务A → HTTP请求 → 服务B
服务A ← HTTP响应 ← 服务B

优点:
✓ 简单直观
✓ 实时响应

缺点:
❌ 服务B挂了，服务A也受影响
❌ 级联失败风险
```

**REST API示例**:

```python
# 服务A：订单服务
import requests

def create_order(user_id, product_id):
    # 调用用户服务检查用户
    user_response = requests.get(f'http://user-service/users/{user_id}')
    if user_response.status_code != 200:
        return {"error": "User not found"}
    
    # 调用商品服务检查库存
    product_response = requests.get(f'http://product-service/products/{product_id}')
    if product_response.json()['stock'] <= 0:
        return {"error": "Out of stock"}
    
    # 创建订单
    order = {"user_id": user_id, "product_id": product_id}
    return order
```

---

**2. 异步通信 - 消息队列**

```
服务A → 发送消息 → 消息队列 → 服务B订阅

优点:
✓ 解耦（服务A不需要知道服务B）
✓ 异步处理
✓ 服务B挂了，消息不丢失

缺点:
❌ 复杂度高
❌ 调试困难
```

**消息队列示例（RabbitMQ）**:

```python
# 服务A：订单服务（生产者）
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='orders')

# 发送消息
channel.basic_publish(
    exchange='',
    routing_key='orders',
    body='{"order_id": 123, "user_id": 456}'
)

# 服务B：通知服务（消费者）
def callback(ch, method, properties, body):
    print(f"Received order: {body}")
    # 发送邮件通知
    send_email(body)

channel.basic_consume(queue='orders', on_message_callback=callback, auto_ack=True)
channel.start_consuming()
```

---

#### API Gateway

**作用**: 所有外部请求的统一入口

```
客户端（Web/Mobile）
      ↓
  API Gateway  ← 统一入口
      ↓
  ┌───┴────┬────────┬────────┐
  ↓        ↓        ↓        ↓
用户服务  订单服务  商品服务  支付服务
```

**API Gateway的功能**:

```
1. 路由
   /api/users/* → 用户服务
   /api/orders/* → 订单服务

2. 认证/鉴权
   检查JWT Token
   验证权限

3. 负载均衡
   请求分发到多个实例

4. 限流
   防止单个用户发送过多请求

5. 监控和日志
   记录所有API调用

6. 协议转换
   HTTP → gRPC
```

**常用API Gateway**:
- Kong
- Nginx
- AWS API Gateway
- Traefik

---

#### 服务发现

**问题**: 微服务的IP和端口是动态的

```
手动配置（不可行）:
用户服务IP: 192.168.1.10:8001
订单服务IP: 192.168.1.11:8002

问题:
- 服务重启，IP可能变
- 自动扩缩容，实例数量变化
- 手动维护配置文件太麻烦
```

**服务发现**: 自动注册和查找服务

```
服务注册中心（如Consul、Etcd）
       ↓
   注册 & 心跳
       ↓
  ┌────┴────┐
用户服务 订单服务

流程:
1. 服务启动时，向注册中心注册（IP、端口）
2. 定期发送心跳（证明活着）
3. 服务A要调用服务B，向注册中心查询B的地址
4. 注册中心返回B的地址列表
5. 服务A选一个地址调用
```

---

#### 负载均衡

**作用**: 将请求分发到多个实例

```
       负载均衡器
          ↓
   ┌──────┼──────┐
   ↓      ↓      ↓
实例1   实例2   实例3
```

**负载均衡算法**:

```
1. 轮询（Round Robin）
   请求1 → 实例1
   请求2 → 实例2
   请求3 → 实例3
   请求4 → 实例1
   ...

2. 随机（Random）
   随机选一个实例

3. 最少连接（Least Connections）
   选择当前连接数最少的实例

4. 加权（Weighted）
   性能好的实例分配更多请求
   实例1（权重2）: 40%
   实例2（权重2）: 40%
   实例3（权重1）: 20%

5. IP Hash
   根据客户端IP哈希，同一客户端总是访问同一实例
   （有状态服务需要）
```

---

#### 熔断器 (Circuit Breaker)

**问题**: 级联失败

```
用户 → 服务A → 服务B → 服务C
                    ↓
                  挂了

结果:
- 服务A一直等待服务B响应
- 服务A的线程被占满
- 服务A也挂了
- 整个系统崩溃
```

**熔断器**: 快速失败，防止级联

```
状态机:
┌────────┐  失败率高  ┌────────┐
│ 关闭   │─────────→│  打开   │
│(正常)  │          │(熔断)   │
└────────┘          └────────┘
    ↑                   ↓
    │                超时后
    │                   ↓
    │               ┌────────┐
    └───────────────│ 半开   │
       成功率高      │(尝试)  │
                    └────────┘

关闭状态: 正常调用
打开状态: 直接返回错误，不调用下游服务
半开状态: 尝试少量请求，观察是否恢复
```

**Python示例（pybreaker）**:

```python
from pybreaker import CircuitBreaker

# 创建熔断器
breaker = CircuitBreaker(
    fail_max=5,  # 5次失败后打开
    timeout_duration=60  # 60秒后进入半开状态
)

@breaker
def call_service_b():
    response = requests.get('http://service-b/api')
    return response.json()

# 调用
try:
    result = call_service_b()
except CircuitBreakerError:
    # 熔断器打开，快速失败
    return {"error": "Service B is down"}
```

---

### 4.3 AI场景的微服务实践

#### AI推理服务架构

```
客户端
  ↓
API Gateway
  ↓
┌─────────┬──────────┬──────────┬──────────┐
│  模型A  │  模型B   │  模型C   │ 特征服务 │
│ (文本)  │ (图像)   │ (推荐)   │         │
└─────────┴──────────┴──────────┴──────────┘
     ↓         ↓         ↓          ↓
  缓存服务（Redis）
     ↓
  数据库（PostgreSQL）
     ↓
  消息队列（RabbitMQ）
     ↓
  离线训练服务
```

**服务拆分**:

```
1. 特征服务
   - 输入: user_id, context
   - 输出: feature_vector
   - 作用: 统一的特征提取

2. 模型推理服务（多个）
   - 输入: feature_vector
   - 输出: prediction
   - 作用: 模型预测
   - 独立扩展（热门模型多实例）

3. 缓存服务
   - 缓存特征和预测结果
   - 减少重复计算

4. A/B测试服务
   - 流量分配
   - 实验管理

5. 监控服务
   - 模型性能监控
   - 预测延迟监控
```

---

#### 模型服务化示例

**使用FastAPI部署模型**:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# 加载模型
model = torch.load('model.pth')
model.eval()

class PredictRequest(BaseModel):
    text: str
    user_id: int

class PredictResponse(BaseModel):
    prediction: float
    model_version: str

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # 预处理
    input_tensor = preprocess(request.text)
    
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
    
    return PredictResponse(
        prediction=output.item(),
        model_version="v1.0"
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Docker化**:

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Kubernetes部署**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-service
spec:
  replicas: 3  # 3个实例
  selector:
    matchLabels:
      app: model-service
  template:
    metadata:
      labels:
        app: model-service
    spec:
      containers:
      - name: model-service
        image: myregistry/model-service:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 五、Kubernetes (K8s) 基础

### 5.1 什么是Kubernetes？

**定义**: 容器编排平台

**问题**: Docker只管单个容器，生产环境需要管理成百上千个容器

```
手动管理的问题:
- 100个容器，手动启动？
- 容器挂了，谁来重启？
- 流量增加，如何自动扩容？
- 如何做滚动更新？
- 如何做服务发现和负载均衡？
```

**Kubernetes的解决方案**: 自动化管理容器

```
功能:
✓ 自动部署和扩缩容
✓ 自我修复（容器挂了自动重启）
✓ 服务发现和负载均衡
✓ 滚动更新和回滚
✓ 配置和密钥管理
✓ 存储编排
```

---

### 5.2 Kubernetes核心概念

#### Pod

**定义**: K8s的最小部署单元

```
Pod = 一个或多个容器 + 共享存储/网络

通常一个Pod = 一个容器（主容器）+ 可选的sidecar容器

例子:
Pod
├─ 主容器: 应用
└─ Sidecar: 日志收集器

Pod内容器:
- 共享网络（localhost通信）
- 共享存储卷
- 一起调度（同一节点）
```

---

#### Deployment

**定义**: 管理Pod的控制器

```
Deployment = 声明式管理Pod
- 期望状态: 3个Pod
- 实际状态: 2个Pod
→ K8s自动创建1个Pod

功能:
✓ 副本管理（replicas）
✓ 滚动更新
✓ 回滚
✓ 暂停和恢复
```

**示例**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3  # 期望3个Pod
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.0
        ports:
        - containerPort: 8000
```

---

#### Service

**定义**: 为Pod提供稳定的网络访问

```
问题:
- Pod的IP是动态的（重启后变化）
- 多个Pod需要负载均衡

Service:
- 提供稳定的虚拟IP（ClusterIP）
- 自动负载均衡到后端Pod
- 通过selector选择Pod
```

**Service类型**:

```
ClusterIP（默认）:
- 集群内部访问
- 虚拟IP，只在集群内有效

NodePort:
- 通过节点IP+端口访问
- 端口范围: 30000-32767

LoadBalancer:
- 云平台提供的负载均衡器
- 分配外部IP

ExternalName:
- 映射到外部DNS
```

**示例**:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  selector:
    app: my-app  # 选择匹配的Pod
  ports:
  - port: 80  # Service端口
    targetPort: 8000  # Pod端口
  type: LoadBalancer
```

---

#### ConfigMap 和 Secret

**ConfigMap**: 存储配置

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgresql://localhost:5432/mydb"
  log_level: "INFO"
```

**Secret**: 存储敏感信息

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
data:
  password: cGFzc3dvcmQxMjM=  # base64编码
```

**在Pod中使用**:

```yaml
spec:
  containers:
  - name: my-app
    image: my-app:v1
    env:
    - name: DATABASE_URL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: database_url
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: app-secret
          key: password
```

---

### 5.3 kubectl常用命令

```bash
# 获取资源
kubectl get pods  # 列出Pod
kubectl get deployments
kubectl get services
kubectl get nodes

# 查看详情
kubectl describe pod my-app-abc123
kubectl logs my-app-abc123  # 查看日志
kubectl logs -f my-app-abc123  # 实时查看

# 创建资源
kubectl apply -f deployment.yaml
kubectl create deployment my-app --image=my-app:v1

# 删除资源
kubectl delete pod my-app-abc123
kubectl delete -f deployment.yaml

# 扩缩容
kubectl scale deployment my-app --replicas=5

# 滚动更新
kubectl set image deployment/my-app my-app=my-app:v2
kubectl rollout status deployment/my-app

# 回滚
kubectl rollout undo deployment/my-app

# 进入容器
kubectl exec -it my-app-abc123 -- /bin/bash

# 端口转发（本地访问Pod）
kubectl port-forward pod/my-app-abc123 8000:8000
```

---

## 六、监控和日志

### 6.1 监控指标

**四个黄金信号**:

```
1. 延迟（Latency）
   - 请求响应时间
   - P50, P95, P99

2. 流量（Traffic）
   - 每秒请求数（QPS）
   - 每秒事务数（TPS）

3. 错误（Errors）
   - 错误率
   - 5xx错误数

4. 饱和度（Saturation）
   - CPU使用率
   - 内存使用率
   - 磁盘IO
```

---

### 6.2 Prometheus + Grafana

**Prometheus**: 指标收集和存储

**工作原理**:

```
应用 → 暴露/metrics端点 → Prometheus拉取 → 时序数据库

查询:
PromQL查询语言
```

**Python应用暴露指标**:

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI

app = FastAPI()

# 定义指标
REQUEST_COUNT = Counter('http_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'Request latency')

@app.post("/predict")
@REQUEST_LATENCY.time()
async def predict(text: str):
    REQUEST_COUNT.inc()
    # 业务逻辑
    return {"result": "..."}

@app.get("/metrics")
async def metrics():
    return generate_latest()
```

**Grafana**: 可视化

```
Prometheus数据源 → Grafana Dashboard → 图表

常见图表:
- QPS趋势图
- P99延迟图
- 错误率图
- CPU/内存使用率
```

---

### 6.3 日志管理

**ELK Stack**: Elasticsearch + Logstash + Kibana

```
应用 → 输出日志 → Logstash收集 → Elasticsearch存储 → Kibana查询

优势:
- 集中式日志管理
- 全文搜索
- 日志分析
```

**结构化日志**:

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "prediction_made",
    user_id=123,
    model_version="v1.0",
    latency_ms=50,
    prediction=0.95
)

# 输出JSON格式:
# {"event": "prediction_made", "user_id": 123, "model_version": "v1.0", ...}
```

---

## 七、面试高频问题

### Q1: Docker和虚拟机的区别？

**回答框架**:
```
核心区别在于隔离级别:

虚拟机:
- 完整的操作系统
- 强隔离（Hypervisor层）
- 占用资源多（GB级）
- 启动慢（分钟级）

Docker:
- 共享宿主机内核
- 轻量级隔离（namespace、cgroup）
- 占用资源少（MB级）
- 启动快（秒级）

选择:
- 需要不同OS: 虚拟机
- 需要轻量级、快速部署: Docker
```

---

### Q2: CI/CD的好处是什么？

**回答框架**:
```
CI/CD解决的核心问题是"快速、可靠地交付软件"

好处:
1. 早发现问题
   - 每次提交都测试
   - Bug在引入时就发现

2. 快速迭代
   - 自动化部署，分钟级上线
   - 小步快跑，持续改进

3. 降低风险
   - 小批量变更，问题影响小
   - 快速回滚

4. 提高质量
   - 自动化测试保证质量
   - 代码审查集成到流程

实际项目经验:
- 实现了自动化训练pipeline
- 代码提交后自动训练、评估、部署
- 部署频率从每月一次到每天多次
```

---

### Q3: 什么是微服务？优缺点是什么？

**回答框架**:
```
微服务是将应用拆分成多个小的、独立的服务

优点:
1. 独立扩展：热门服务多实例
2. 技术自由：每个服务选合适的技术栈
3. 团队自治：不同团队负责不同服务
4. 故障隔离：一个服务挂了不影响全局

缺点:
1. 复杂度高：分布式系统固有的复杂性
2. 运维成本：多个服务要监控、部署
3. 调用链复杂：跨服务调试困难
4. 数据一致性：分布式事务难处理

何时使用:
- 大型应用、多团队
- 需要独立扩展不同模块
- 技术栈需要灵活性

何时不用:
- 小团队、小项目
- 初期MVP阶段（先单体，再拆分）
```

---

### Q4: Kubernetes的作用是什么？

**回答框架**:
```
Kubernetes是容器编排平台，自动化管理容器

核心功能:
1. 自动部署：声明期望状态，K8s自动实现
2. 自动扩缩容：根据负载动态调整实例数
3. 自我修复：容器挂了自动重启
4. 服务发现：自动管理服务网络
5. 滚动更新：零停机更新

使用场景:
- 大规模容器管理（几十上百个服务）
- 需要高可用
- 需要自动扩缩容

AI项目中的应用:
- 部署模型推理服务
- 根据QPS自动扩容
- GPU资源调度
```

---

### Q5: 如何保证微服务的高可用？

**回答框架**:
```
高可用需要多层保障:

1. 多实例
   - 至少3个实例
   - 分布在不同节点/机房

2. 健康检查
   - Liveness：容器是否活着
   - Readiness：是否准备好接收流量

3. 熔断器
   - 防止级联失败
   - 快速失败，不占用资源

4. 限流和降级
   - 限流：保护系统不过载
   - 降级：核心功能优先

5. 监控告警
   - 实时监控关键指标
   - 异常自动告警

6. 灰度发布
   - 渐进式发布新版本
   - 快速回滚

实践经验:
- 设置99.9%的SLA目标
- 每周演练故障恢复
- 自动化回滚机制
```

---

## 八、总结

### 核心知识点回顾

✅ **Docker**:
- 镜像、容器、Dockerfile
- 常用命令
- Docker Compose多容器管理

✅ **CI/CD**:
- 持续集成和持续部署的流程
- GitHub Actions / GitLab CI
- 部署策略（蓝绿、金丝雀、滚动）

✅ **微服务**:
- 单体 vs 微服务
- 服务拆分原则
- 服务通信（REST、消息队列）
- API Gateway、服务发现、熔断器

✅ **Kubernetes**:
- Pod、Deployment、Service
- 常用命令
- 自动扩缩容

✅ **监控和日志**:
- 监控指标（四个黄金信号）
- Prometheus + Grafana
- 日志管理（ELK）

---

### 为什么这些对AI工程师重要？

```
1. 模型部署
   - Docker保证环境一致性
   - K8s自动管理服务

2. 自动化
   - CI/CD自动化训练和部署
   - 减少人工介入

3. 规模化
   - 微服务架构支撑大规模应用
   - 独立扩展热门模型

4. 可靠性
   - 监控和告警快速发现问题
   - 熔断和降级保证系统稳定

5. 团队协作
   - 标准化的工具和流程
   - 提高开发效率
```

---

**恭喜你完成Day 12的学习！**

你现在掌握了完整的AI工程技能栈：
- ✅ AI应用开发（Agent、RAG）
- ✅ 深度学习（PyTorch、Transformer）
- ✅ 分布式训练
- ✅ 系统基础（OS、网络、存储）
- ✅ 数据库
- ✅ DevOps和微服务 ✨

**你已经准备好成为全栈AI工程师了！** 🚀
