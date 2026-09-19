# RAG架构与向量数据库 核心知识储备

## 一、什么是RAG？

### 基础定义
**RAG (Retrieval-Augmented Generation，检索增强生成)** 是一种结合了信息检索和文本生成的AI技术，让大语言模型能够访问外部知识库，生成更准确、更及时的回答。

**核心思想**: 先检索相关信息，再基于检索结果生成回答

**简单理解**:
- 传统LLM: 只靠"记忆"（训练数据）回答问题
- RAG: 先"查资料"（检索），再根据资料回答问题

**类比**: 
- 传统LLM = 闭卷考试（只能靠记忆）
- RAG = 开卷考试（可以查阅资料）

---

## 二、为什么需要RAG？

### LLM的三大局限

**1. 知识截止日期**
- GPT-4的知识截止于2023年4月
- 不知道最新发生的事
- **RAG解决**: 从最新文档中检索信息

**2. 幻觉问题 (Hallucination)**
- LLM可能编造不存在的信息
- 特别是对不熟悉的领域
- **RAG解决**: 基于真实文档回答，可验证

**3. 领域知识不足**
- 通用模型对专业领域了解有限
- 企业内部知识、专业文献
- **RAG解决**: 接入专业知识库

### RAG vs Fine-tuning

| 特性 | RAG | Fine-tuning |
|------|-----|-------------|
| 知识更新 | 实时（更新文档即可） | 需要重新训练 |
| 成本 | 低（只需存储文档） | 高（需要GPU训练） |
| 可解释性 | 高（可以看到引用来源） | 低（黑盒） |
| 适用场景 | 知识密集型任务 | 改变模型行为/风格 |
| 技术门槛 | 较低 | 较高 |

**结论**: RAG适合需要访问大量外部知识的场景，Fine-tuning适合改变模型的输出风格或特定能力。

---

## 三、RAG的工作流程

### 完整流程图

```
离线阶段（建立知识库）:
    文档集合
       ↓
    文档切片 (Chunking)
       ↓
    生成向量 (Embedding)
       ↓
    存入向量数据库


在线阶段（回答问题）:
    用户提问
       ↓
    问题向量化 (Embedding)
       ↓
    向量相似度搜索
       ↓
    检索Top-K相关文档
       ↓
    构建提示词 (问题+检索内容)
       ↓
    LLM生成回答
       ↓
    返回给用户
```

### 详细步骤解析

#### 离线阶段（一次性准备）

**步骤1: 收集文档**
```
知识来源:
- PDF文件
- Word文档
- 网页内容
- 数据库记录
- API数据
```

**步骤2: 文档切片 (Chunking)**

**为什么需要切片？**
- 文档太长，LLM无法一次处理全部
- 提高检索精度（找到最相关的段落）
- 降低成本（只发送相关部分给LLM）

**切片策略**:

1. **固定大小切片**
```python
chunk_size = 500  # 每块500字
overlap = 50      # 重叠50字（保持上下文连贯）

text = "很长的文档内容..."
chunks = split_text(text, chunk_size, overlap)
```

2. **按段落切片**
```python
# 按\n\n分割
chunks = text.split("\n\n")
```

3. **按语义切片**
```python
# 保持完整的句子或段落
# 使用NLP技术识别语义边界
```

**切片参数选择**:
- **Chunk Size**: 通常200-1000字
  - 太小: 上下文不足
  - 太大: 检索不精确，成本高
- **Overlap**: 通常10-20%
  - 避免重要信息被切断

**步骤3: 向量化 (Embedding)**

**什么是Embedding？**
- 将文本转换为数字向量（一串数字）
- 语义相似的文本，向量也相似

**例子**:
```
"猫是动物" → [0.2, 0.8, 0.1, 0.5, ...]
"狗是动物" → [0.3, 0.7, 0.2, 0.4, ...]  (相似！)
"天气很好" → [0.9, 0.1, 0.8, 0.2, ...]  (不相似)
```

**常用Embedding模型**:
- **OpenAI Embeddings**: text-embedding-ada-002
- **开源模型**: sentence-transformers, BGE, m3e
- **多语言**: multilingual-e5

**向量维度**:
- OpenAI: 1536维
- 开源模型: 通常384-768维
- 维度越高，表达能力越强，但存储成本也越高

**步骤4: 存入向量数据库**
```python
# 伪代码
for chunk in chunks:
    embedding = embedding_model.encode(chunk)
    vector_db.insert(
        id=chunk_id,
        vector=embedding,
        metadata={"text": chunk, "source": "doc1.pdf"}
    )
```

#### 在线阶段（每次查询）

**步骤1: 用户提问**
```
用户: "RAG的优势是什么？"
```

**步骤2: 问题向量化**
```python
question = "RAG的优势是什么？"
question_embedding = embedding_model.encode(question)
# 得到问题的向量表示
```

**步骤3: 向量相似度搜索**

**相似度计算方法**:

1. **余弦相似度 (Cosine Similarity)** - 最常用
```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
范围: -1 到 1（越接近1越相似）
```

2. **欧氏距离 (Euclidean Distance)**
```
distance = √Σ(ai - bi)²
距离越小越相似
```

3. **点积 (Dot Product)**
```
similarity = A · B
```

**步骤4: 检索Top-K文档**
```python
# 找到最相似的3个文档片段
results = vector_db.search(
    query_vector=question_embedding,
    top_k=3
)

# 返回:
# [
#   {"text": "RAG的优势包括...", "score": 0.89},
#   {"text": "相比Fine-tuning...", "score": 0.85},
#   {"text": "RAG可以实时更新...", "score": 0.82}
# ]
```

**K值选择**:
- 太小（k=1-2）: 可能遗漏重要信息
- 太大（k>10）: 引入噪音，增加成本
- 常用值: k=3-5

**步骤5: 构建Prompt**
```python
prompt = f"""
基于以下参考资料回答问题：

【参考资料1】
{result1.text}

【参考资料2】
{result2.text}

【参考资料3】
{result3.text}

问题: {question}

要求:
1. 基于参考资料回答
2. 如果资料中没有相关信息，明确说明
3. 引用具体的参考资料编号
"""
```

**步骤6: LLM生成回答**
```python
response = llm.generate(prompt)
```

**步骤7: 返回结果**
```
回答: RAG的优势主要包括：
1. 知识实时更新（参考资料1）
2. 降低幻觉问题（参考资料2）
3. 成本低于Fine-tuning（参考资料3）

来源:
- doc1.pdf, 第3页
- doc2.pdf, 第7页
```

---

## 四、向量数据库 (Vector Database)

### 什么是向量数据库？

**定义**: 专门用于存储和检索向量（高维数组）的数据库，能高效地进行相似度搜索。

**传统数据库 vs 向量数据库**:

| 特性 | 传统数据库 | 向量数据库 |
|------|-----------|-----------|
| 存储内容 | 结构化数据（文字、数字） | 向量（数字数组） |
| 查询方式 | 精确匹配（WHERE id=1） | 相似度搜索 |
| 索引结构 | B树、哈希 | HNSW、IVF |
| 典型应用 | 业务数据 | AI、推荐系统 |

### 常见向量数据库

**1. Pinecone**
- **类型**: 云服务（托管）
- **优势**: 易用，无需管理基础设施
- **劣势**: 收费，数据在第三方
- **适用**: 快速原型，小型项目

**2. Weaviate**
- **类型**: 开源+云服务
- **优势**: 功能丰富，支持多种数据类型
- **特点**: 集成多种AI模型
- **适用**: 企业级应用

**3. Milvus**
- **类型**: 开源
- **优势**: 高性能，可扩展
- **特点**: 支持GPU加速
- **适用**: 大规模部署

**4. ChromaDB**
- **类型**: 开源
- **优势**: 轻量级，易于使用
- **特点**: Python友好
- **适用**: 开发测试，中小项目

**5. Qdrant**
- **类型**: 开源
- **优势**: 性能好，功能完整
- **特点**: Rust编写，内存效率高
- **适用**: 生产环境

**6. FAISS** (Facebook AI Similarity Search)
- **类型**: 库（不是完整数据库）
- **优势**: 超快，Meta开发
- **特点**: 只提供检索功能
- **适用**: 本地部署，研究

### 向量数据库的核心技术

#### 1. 索引算法

**为什么需要索引？**
- 暴力搜索：计算查询向量与所有向量的相似度 → O(n)，太慢
- 索引：快速找到近似最相似的向量 → O(log n)或更快

**主要索引算法**:

**HNSW (Hierarchical Navigable Small World)**
- **原理**: 构建多层图结构
- **优势**: 查询速度极快
- **劣势**: 内存占用大
- **适用**: 内存充足的场景

```
层3:  节点A ←→ 节点B
        ↓        ↓
层2:  节点A ←→ 节点C ←→ 节点B
        ↓        ↓        ↓
层1:  所有节点相连（密集）
```

**IVF (Inverted File Index)**
- **原理**: 将向量聚类，只在相关簇中搜索
- **优势**: 内存友好
- **劣势**: 准确率略低
- **适用**: 大规模数据

**LSH (Locality-Sensitive Hashing)**
- **原理**: 用哈希函数将相似向量映射到同一个桶
- **优势**: 快速
- **劣势**: 准确率较低

#### 2. 相似度计算

**余弦相似度**:
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 例子
vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 4, 6])
similarity = cosine_similarity(vec1, vec2)
# 结果接近1（非常相似）
```

**欧氏距离**:
```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
```

#### 3. 过滤 (Filtering)

**场景**: 不仅要相似，还要满足条件

**例子**:
```python
# 找到相似的文档，且文档类型是"技术文档"
results = vector_db.search(
    query_vector=embedding,
    filter={"type": "技术文档", "date": {"$gte": "2024-01-01"}},
    top_k=5
)
```

**实现方式**:
- **Pre-filtering**: 先过滤再搜索（更准确）
- **Post-filtering**: 先搜索再过滤（更快）

---

## 五、RAG的核心组件详解

### 1. Embedding模型选择

**评估标准**:
- **准确性**: 能否准确捕捉语义
- **速度**: 向量化速度
- **维度**: 向量维度大小
- **语言支持**: 中英文、多语言
- **成本**: 是否开源，API费用

**OpenAI Embeddings**:
```python
import openai

response = openai.Embedding.create(
    input="这是要向量化的文本",
    model="text-embedding-ada-002"
)
embedding = response['data'][0]['embedding']
# 1536维向量
```

**开源Embedding (Sentence Transformers)**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("这是要向量化的文本")
# 384维向量
```

**中文Embedding推荐**:
- **BGE**: BAAI开发，中文效果好
- **m3e**: Moka开发，中文特化
- **text2vec**: 简单易用

### 2. 文档加载器 (Document Loader)

**不同格式的处理**:

```python
# PDF
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Word
from langchain.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("document.docx")

# 网页
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")

# Markdown
from langchain.document_loaders import UnstructuredMarkdownLoader
loader = UnstructuredMarkdownLoader("README.md")
```

### 3. 文本分割器 (Text Splitter)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块500字符
    chunk_overlap=50,    # 重叠50字符
    separators=["\n\n", "\n", "。", ".", " ", ""]  # 优先级分隔符
)

chunks = splitter.split_text(long_text)
```

**分割策略**:
- 优先按段落分割
- 保持句子完整
- 重叠部分保持上下文

### 4. 检索策略优化

**基础检索**: 只用向量相似度

**混合检索 (Hybrid Search)**:
```
最终得分 = α × 向量相似度 + (1-α) × 关键词匹配度

向量搜索: 语义相似（"猫" 和 "宠物"）
关键词搜索: 精确匹配（特定术语、代码）
```

**重排序 (Reranking)**:
```
初检: 向量搜索 → Top-50
重排: 更精确的模型 → Top-5
送入LLM
```

**为什么重排？**
- 初检快但不够准
- 重排慢但更准
- 两阶段平衡速度和准确性

**元数据过滤**:
```python
# 只搜索特定来源的文档
results = db.search(
    query=embedding,
    filter={"source": "official_docs", "year": 2024},
    top_k=5
)
```

---

## 六、RAG的评估指标

### 如何评估RAG系统？

#### 1. 检索质量

**Precision@K (精确率)**:
```
检索的K个结果中，相关的有多少个

Precision@5 = 相关文档数 / 5
```

**Recall@K (召回率)**:
```
所有相关文档中，检索到了多少个

Recall@5 = 检索到的相关文档 / 总相关文档数
```

**MRR (Mean Reciprocal Rank)**:
```
第一个相关结果的排名倒数

例: 第一个相关结果排第3 → MRR = 1/3
```

#### 2. 生成质量

**准确性 (Faithfulness)**:
- 生成的答案是否基于检索到的文档
- 是否有幻觉

**相关性 (Relevance)**:
- 答案是否回答了问题
- 是否包含无关信息

**完整性 (Completeness)**:
- 是否完整回答了问题
- 关键信息是否遗漏

#### 3. 端到端评估

**人工评估**:
- 招募评估者
- 设计评分标准
- 成本高但最准确

**自动评估**:
```python
# 使用GPT-4评估答案质量
evaluation_prompt = f"""
问题: {question}
参考资料: {context}
答案: {answer}

评估以下方面（1-5分）:
1. 准确性: 答案是否基于参考资料
2. 相关性: 答案是否回答了问题
3. 完整性: 答案是否完整
"""
score = gpt4.evaluate(evaluation_prompt)
```

---

## 七、RAG的常见问题和优化

### 问题1: 检索不准确

**症状**: 检索到的文档与问题无关

**原因**:
- Embedding模型不够好
- Chunk切得不合理
- 问题表述不清

**解决方案**:
1. 更换更好的Embedding模型
2. 调整Chunk大小和overlap
3. 对用户问题进行改写
```python
# 问题改写
original = "它是啥？"
rewritten = "RAG技术是什么？"  # 补充上下文
```

### 问题2: 上下文太长

**症状**: 检索到太多文档，超过LLM的context限制

**解决方案**:
1. 减少Top-K
2. 使用重排序筛选
3. 总结检索结果后再送入LLM
```python
# 先总结检索结果
summary = summarize(retrieved_docs)
prompt = f"基于以下摘要回答: {summary}\n问题: {question}"
```

### 问题3: 答案有幻觉

**症状**: LLM编造信息，不基于检索结果

**解决方案**:
1. 在Prompt中强调"仅基于参考资料"
2. 让LLM引用来源
3. 后处理检查答案是否在检索文档中
```python
prompt = """
严格基于以下资料回答，如果资料中没有，回答"资料中未提及"

参考资料: {context}
问题: {question}
"""
```

### 问题4: 成本太高

**症状**: API调用费用过高

**解决方案**:
1. 使用开源Embedding模型
2. 缓存常见问题的结果
3. 本地部署向量数据库
4. 优化Chunk大小减少token

---

## 八、RAG的进阶技术

### 1. Multi-Query RAG

**问题**: 单一问题可能检索不全

**方案**: 生成多个相关问题，分别检索，合并结果

```python
# 原问题
question = "RAG的优势是什么？"

# 生成相关问题
related_questions = [
    "RAG相比传统方法有什么好处？",
    "为什么要使用RAG？",
    "RAG解决了什么问题？"
]

# 分别检索
all_results = []
for q in related_questions:
    results = retrieve(q)
    all_results.extend(results)

# 去重合并
unique_results = deduplicate(all_results)
```

### 2. Self-Query RAG

**问题**: 用户问题包含过滤条件

**方案**: LLM自动提取过滤条件

```python
# 用户问题
question = "2024年关于AI的技术文档"

# LLM提取
extraction = llm.extract_metadata(question)
# {
#   "query": "AI技术文档",
#   "filter": {"year": 2024, "type": "技术文档"}
# }

# 应用过滤检索
results = db.search(query, filter=extraction["filter"])
```

### 3. Parent-Child Chunking

**问题**: 小chunk上下文不足，大chunk检索不精确

**方案**: 用小chunk检索，返回大chunk（父级）

```
文档
  ↓ 切分
大Chunk（段落）
  ↓ 再切分
小Chunk（句子） ← 用于检索

检索时:
1. 用小chunk的向量搜索
2. 找到匹配的小chunk
3. 返回它所属的大chunk给LLM
```

### 4. Hypothetical Document Embeddings (HyDE)

**问题**: 问题和答案的向量差异大

**方案**: 先让LLM生成假设答案，用假设答案检索

```python
# 用户问题
question = "RAG的优势？"

# 生成假设答案
hypothetical_answer = llm.generate(
    f"请回答: {question}"
)
# "RAG的优势包括知识更新快、成本低..."

# 用假设答案的向量检索（更准确！）
embedding = embed(hypothetical_answer)
results = db.search(embedding)
```

---

## 九、面试高频问题及回答

### Q1: 什么是RAG？

**回答框架**:
"RAG是检索增强生成，结合了信息检索和文本生成。工作流程是：先将文档切片并向量化存入数据库，当用户提问时，检索最相关的文档片段，然后把这些片段和问题一起送给LLM生成回答。RAG的核心优势是让LLM能访问外部知识，解决了知识截止日期和幻觉问题，相比Fine-tuning成本更低，知识更新也更容易。"

### Q2: RAG和Fine-tuning有什么区别？

**回答框架**:
"主要区别在于知识的存储方式和更新成本。RAG是将知识存在外部数据库，通过检索获取；Fine-tuning是将知识融入模型参数。RAG的优势是知识更新简单（只需更新文档），成本低，可解释性强（能看到引用来源）。Fine-tuning适合改变模型的输出风格或行为，RAG适合需要访问大量动态知识的场景。实际项目中，两者可以结合使用。"

### Q3: 向量数据库是什么？为什么需要？

**回答框架**:
"向量数据库专门用于存储和检索高维向量，支持相似度搜索。与传统数据库的精确匹配不同，向量数据库能找到'最相似'的结果。在RAG中，我们将文本转换为向量存储，查询时通过计算向量相似度找到语义相关的文档。常用的有ChromaDB、Pinecone、Milvus等。向量数据库使用特殊的索引算法如HNSW来加速搜索，比暴力计算快得多。"

### Q4: Embedding是什么？

**回答框架**:
"Embedding是将文本转换为数字向量的过程。语义相似的文本会被转换为相近的向量。比如'猫'和'宠物'的向量会很接近，但和'天气'就很远。Embedding捕捉了文本的语义信息，这样我们就能用数学方法（如余弦相似度）来衡量文本的相似程度。常用的Embedding模型有OpenAI的text-embedding-ada-002和开源的sentence-transformers。"

### Q5: RAG的文档切片策略有哪些？

**回答框架**:
"主要有三种策略：固定大小切片（如每500字一块，重叠50字）、按段落切片、按语义切片。关键参数是chunk_size和overlap。Chunk太小会丢失上下文，太大会降低检索精度。Overlap保证重要信息不会被切断。我通常会根据文档特点选择，技术文档可能用较大chunk保持完整性，聊天记录可能用较小chunk提高精度。需要在准确性和成本之间平衡。"

### Q6: 如何评估RAG系统的好坏？

**回答框架**:
"我会从两个层面评估：检索质量和生成质量。检索质量看Precision（检索结果的准确率）和Recall（是否遗漏相关文档）。生成质量看Faithfulness（是否基于检索文档）、Relevance（是否回答问题）、Completeness（是否完整）。实际评估可以用人工打分，或者用GPT-4等强模型自动评估。还要考虑响应速度和成本。"

### Q7: RAG有哪些常见问题和优化方法？

**回答框架**:
"常见问题包括：检索不准（可以用更好的Embedding模型、调整chunk大小、问题改写）、上下文太长（减少Top-K、使用重排序）、答案有幻觉（在Prompt中强调仅基于资料、要求引用来源）。进阶优化包括Multi-Query（生成多个相关问题检索）、混合检索（结合向量和关键词）、重排序（两阶段检索）等。选择哪种方法要看具体场景和资源。"

### Q8: 你做过RAG项目吗？

**回答框架** (学完后可以这样说):
"是的，我实现过一个基于RAG的文档问答系统。使用ChromaDB作为向量数据库，sentence-transformers做Embedding，文档按500字切片，50字重叠。实现了完整的离线建库和在线检索流程。在优化过程中，我发现调整chunk大小和使用重排序能显著提升准确率。这个项目让我深入理解了RAG的核心原理，特别是Embedding和相似度检索的细节。"

---

## 十、核心概念速记卡

### RAG = Retrieval + Generation

**Retrieval (检索)**: 从知识库找相关信息  
**Generation (生成)**: 基于检索结果回答问题

### RAG工作流

**离线**: 文档 → 切片 → 向量化 → 存储  
**在线**: 问题 → 向量化 → 检索 → 生成回答

### 向量相似度

**余弦相似度**: 最常用，范围0-1  
**欧氏距离**: 距离越小越相似  
**原理**: 用数学方法衡量文本语义相似度

### Chunk大小选择

**太小**: 上下文不足  
**太大**: 检索不精确，成本高  
**推荐**: 200-1000字，重叠10-20%

### Top-K选择

**K=1-2**: 可能遗漏信息  
**K=3-5**: 常用，平衡  
**K>10**: 引入噪音

---

## 十一、学习检查清单

完成以下自测，确保理解：

- [ ] 能解释RAG是什么，为什么需要RAG
- [ ] 能说出RAG的完整工作流程（离线+在线）
- [ ] 理解Embedding的作用和原理
- [ ] 知道文档切片的策略和参数选择
- [ ] 了解向量数据库的作用和常见选择
- [ ] 能说出至少3种相似度计算方法
- [ ] 理解RAG vs Fine-tuning的区别
- [ ] 知道RAG的常见问题和优化方法
- [ ] 能评估RAG系统的质量
- [ ] 了解至少2种RAG的进阶技术

---

## 十二、扩展阅读（可选）

### 推荐资源

**官方文档**:
- LangChain RAG教程
- OpenAI Embeddings文档
- ChromaDB快速开始指南

**深入学习**:
- RAG论文: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- 向量数据库对比
- Embedding模型对比研究

**实践教程**:
- LangChain RAG示例
- Building RAG from Scratch
- Advanced RAG Techniques

---

**最后更新**: 2026-08-27

**下一步**: 完成理论学习后，动手构建你的第一个RAG系统！
