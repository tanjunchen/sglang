# SGLang 项目结构分析文档

## 1. 业务目标 (Business Objectives)

SGLang 是一个高性能的大语言模型（LLM）和视觉语言模型（VLM）服务框架，旨在为企业和研究机构提供低延迟、高吞吐的推理服务。项目的主要业务目标包括：

- **极致性能**: 通过 RadixAttention、零开销调度器等创新技术，实现比同类框架（如 vLLM、TensorRT-LLM）更快的推理速度
- **广泛兼容性**: 支持 NVIDIA、AMD、Intel、Google TPU、华为昇腾等多种硬件平台
- **生产就绪**: 已被 xAI、LinkedIn、Cursor 等大型企业采用，每天处理数万亿 tokens
- **生态整合**: 兼容 OpenAI API 和 HuggingFace 模型生态，降低迁移成本
- **灵活编程**: 提供直观的前端语言，支持复杂的 LLM 应用开发模式

## 2. 核心功能 (Core Features)

### 2.1 后端运行时 (Backend Runtime)
- **RadixAttention**: 基于基数树的智能前缀缓存，自动复用共享前缀
- **零开销调度器**: CPU 调度器开销极低，最大化 GPU 利用率
- **Prefill-Decode 分离**: 将预填充和解码阶段分离到不同设备，提升整体吞吐量
- **投机解码**: 通过草稿模型加速生成过程
- **连续批处理**: 动态批处理请求，提高资源利用率
- **结构化输出**: 支持 JSON、Regex 等结构化生成
- **量化支持**: FP4/FP8/INT4/AWQ/GPTQ 等多种量化方式
- **多 LoRA 批处理**: 同时服务多个 LoRA 适配器

### 2.2 模型支持
- **生成模型**: Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral 等
- **嵌入模型**: e5-mistral、gte、mcdse 等
- **奖励模型**: Skywork 等
- **视觉语言模型**: LLaVA、Qwen-VL 等

### 2.3 硬件支持
- **NVIDIA**: GB200/B300/H100/A100/Spark 等全系列 GPU
- **AMD**: MI355/MI300 系列 GPU
- **Intel**: Xeon CPU
- **Google**: TPU
- **华为**: 昇腾 NPU

### 2.4 前端语言
- **链式调用**: 支持生成调用的链式组合
- **高级提示**: 支持复杂提示工程技术
- **控制流**: 支持条件、循环等控制结构
- **多模态输入**: 支持图像、视频等多种输入
- **并行执行**: 支持请求并行化
- **外部交互**: 支持调用外部 API 和工具

## 3. 整体架构 (Architecture)

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    前端语言层 (Frontend)                     │
│  sglang.lang.* - 用户API、提示管理、后端抽象                 │
├─────────────────────────────────────────────────────────────┤
│                    服务入口层 (Entrypoints)                  │
│  HTTP/gRPC服务器、OpenAI兼容API、引擎封装                    │
├─────────────────────────────────────────────────────────────┤
│                    运行时核心层 (SRT Core)                   │
│  调度器、内存管理、模型执行、采样、投机解码                  │
├─────────────────────────────────────────────────────────────┤
│                    模型层 (Models)                           │
│  各种LLM/VLM模型实现、层定义、量化                           │
├─────────────────────────────────────────────────────────────┤
│                    内核层 (Kernels)                          │
│  sgl-kernel - C++/CUDA高性能算子实现                        │
├─────────────────────────────────────────────────────────────┤
│                    硬件抽象层                                │
│  多硬件后端支持（NVIDIA、AMD、Intel、TPU等）                │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件

#### 3.2.1 前端组件 (`python/sglang/lang/`)
- **`api.py`**: 核心 API 定义（`gen`、`select`、`image` 等）
- **`backend/`**: 后端抽象（OpenAI、Anthropic、VertexAI 等）
- **`ir/`**: 中间表示，将前端调用转换为执行计划
- **`interpreter.py`**: 解释器，执行 IR 并管理状态

#### 3.2.2 运行时核心 (`python/sglang/srt/`)
- **`entrypoints/`**: 服务入口
  - `http_server.py`: HTTP API 服务器
  - `openai_api_adapter.py`: OpenAI API 兼容层
  - `engine.py`: 引擎封装
- **`scheduler.py`**: 核心调度器，管理请求生命周期
- **`mem_cache/`**: 内存管理
  - `radix_cache.py`: 基数树缓存实现
  - `hierarchical_cache.py`: 分层缓存
  - `chunk_cache.py`: 分块缓存
- **`model_executor/`**: 模型执行引擎
  - `model_runner.py`: 模型运行器
  - `cuda_graph_runner.py`: CUDA Graph 优化
- **`layers/`**: 模型层实现
  - `attention/`: 注意力机制（FlashAttention、 MLA 等）
  - `moe/`: 混合专家层
  - `quantization/`: 量化层
- **`models/`**: 模型定义
  - `llama.py`: Llama 模型
  - `deepseek.py`: DeepSeek 模型（支持 MLA）
  - `qwen.py`: Qwen 模型
- **`sampling/`**: 采样逻辑
- **`speculative/`**: 投机解码实现

#### 3.2.3 内核层 (`sgl-kernel/`)
- **`csrc/`**: C++/CUDA 源码
  - `attention/`: 注意力算子
  - `moe/`: MoE 算子
  - `quantization/`: 量化算子
- **`python/`**: Python 绑定

#### 3.2.4 路由器 (`sgl-router/`)
- **Rust 实现**: 高性能请求路由
- **负载均衡**: 支持多种负载均衡策略

## 4. 数据实体 (Data Entities)

### 4.1 核心数据结构

#### 4.1.1 请求相关
```python
# 请求对象
class GenerateReqInput:
    text: str                    # 输入文本
    input_ids: List[int]         # 输入token ID
    sampling_params: SamplingParams  # 采样参数
    rid: str                     # 请求ID
    stream: bool                 # 是否流式输出

# 采样参数
class SamplingParams:
    temperature: float          # 温度
    top_p: float                # top-p采样
    top_k: int                  # top-k采样
    max_new_tokens: int         # 最大生成token数
    stop: List[str]             # 停止字符串
```

#### 4.1.2 调度相关
```python
# 运行中的请求
class Req:
    req_id: str                 # 请求ID
    input_ids: torch.Tensor     # 输入token
    prefix_indices: List[int]   # 前缀缓存索引
    sampling_params: SamplingParams
    token_ids: List[int]        # 已生成token

# 批次
class Batch:
    reqs: List[Req]             # 批次中的请求
    req_pool_indices: List[int] # 请求池索引
    seq_lens: List[int]         # 序列长度
```

#### 4.1.3 缓存相关
```python
# 基数树节点
class TreeNode:
    children: Dict[int, TreeNode]  # 子节点
    value: Optional[torch.Tensor]  # 缓存的值
    lock_ref: int                  # 锁引用计数
    last_access_time: float        # 最后访问时间

# 内存块
class MemoryBlock:
    block_id: int               # 块ID
    ref_count: int              # 引用计数
    last_access_time: float     # 最后访问时间
```

### 4.2 配置实体
```python
# 服务器参数（4200+行，配置中心）
class ServerArgs:
    model_path: str             # 模型路径
    tokenizer_path: str         # 分词器路径
    load_format: str            # 加载格式
    device: str                 # 设备类型
    tp_size: int                # 张量并行大小
    dp_size: int                # 数据并行大小
    mem_fraction_static: float  # 静态内存比例
    # ... 100+ 配置项
```

## 5. 业务流程 (Business Processes)

### 5.1 请求处理流程

```
用户请求
  ↓
HTTP/OpenAI API 适配层
  ↓
调度器接收请求
  ↓
前缀缓存查找（RadixAttention）
  ├─ 命中缓存 → 复用 KV cache
  └─ 未命中 → 分配新缓存
  ↓
加入请求池（ReqPool）
  ↓
批次构建（连续批处理）
  ↓
模型执行
  ├─ Prefill 阶段（并行计算）
  └─ Decode 阶段（自回归生成）
  ↓
采样（Sampling）
  ↓
更新缓存
  ↓
返回结果（流式/非流式）
```

### 5.2 缓存管理流程

```
新请求到达
  ↓
解析输入token
  ↓
在基数树中查找最长匹配前缀
  ├─ 找到匹配节点 → 增加引用计数
  └─ 未找到 → 创建新节点
  ↓
分配物理内存块
  ↓
更新访问时间
  ↓
定期清理（LRU策略）
  ├─ 减少引用计数为0的节点
  └─ 回收内存块
```

### 5.3 调度流程

```
主循环
  ↓
收集新到达请求
  ↓
尝试合并到现有批次
  ├─ 可以合并 → 加入当前批次
  └─ 无法合并 → 创建新批次
  ↓
执行批次
  ├─ 判断是否可以抢占
  ├─ 执行 prefill/decode
  └─ 更新请求状态
  ↓
检查完成的请求
  ↓
返回结果
  ↓
清理资源
```

### 5.4 投机解码流程

```
主模型（大模型）
  ↓
草稿模型（小模型）生成 K 个候选token
  ↓
并行验证
  ├─ 全部接受 → 跳过 K 步
  ├─ 部分接受 → 跳过 M 步（M<K）
  └─ 全部拒绝 → 正常生成1步
  ↓
更新 KV cache
  ↓
继续生成
```

## 6. 关键设计模式

### 6.1 LazyImport 模式
```python
# 延迟导入，减少启动时间和内存占用
Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
```

### 6.2 配置中心模式
`ServerArgs` 类（4200+行）集中管理所有配置，支持从命令行、环境变量、配置文件加载。

### 6.3 分层缓存模式
- **L1**: RadixAttention 前缀缓存（逻辑层）
- **L2**: 物理内存块管理（物理层）
- **L3**: 分层缓存（跨设备）

### 6.4 多硬件抽象
通过统一的接口抽象不同硬件后端，常见路径优先（NVIDIA）。

## 7. 性能优化策略

### 7.1 计算优化
- **CUDA Graph**: 捕获静态计算图，减少 Python 开销
- **算子融合**: 融合多个算子为一个（如 FlashAttention）
- **量化**: 使用 FP8/INT4 等低精度计算

### 7.2 内存优化
- **PagedAttention**: 分页管理 KV cache
- **前缀复用**: 通过 RadixAttention 最大化缓存命中率
- **内存池**: 预分配内存池，减少分配开销

### 7.3 调度优化
- **连续批处理**: 动态调整批次，提高吞吐量
- **PD 分离**: 预填充和解码分离，避免互相干扰
- **抢占机制**: 支持请求抢占，保证公平性

## 8. 扩展性设计

### 8.1 模型扩展
- **统一接口**: 所有模型继承 `nn.Module` 并实现统一接口
- **模块化设计**: 注意力层、MLP 层等可插拔
- **配置驱动**: 通过配置文件定义模型结构

### 8.2 硬件扩展
- **后端注册**: 新硬件通过注册机制接入
- **算子抽象**: 统一算子接口，硬件特定实现
- **内存管理**: 抽象内存分配器，支持不同硬件

### 8.3 功能扩展
- **采样器**: 可插拔的采样策略
- **量化方案**: 支持多种量化算法
- **投机解码**: 支持多种草稿模型

## 9. 部署架构

### 9.1 单节点部署
```
单GPU/多GPU
  ↓
SGLang Runtime
  ↓
模型权重
  ↓
KV Cache (GPU显存)
```

### 9.2 分布式部署
```
Load Balancer (sgl-router)
  ↓
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Worker 1 │  │ Worker 2 │  │ Worker 3 │
│ (GPU)    │  │ (GPU)    │  │ (GPU)    │
└──────────┘  └──────────┘  └──────────┘
  ↓              ↓              ↓
分布式 KV Cache 管理
```

### 9.3 大规模部署（Expert Parallelism）
```
Coordinator
  ↓
┌─────────────────────────────────────┐
│ 多个节点，每个节点多个GPU            │
│ 模型分片 + 专家并行                 │
└─────────────────────────────────────┘
```

## 10. 监控与可观测性

### 10.1 指标收集
- **吞吐量**: tokens/second
- **延迟**: TTFT（首个token时间）、TPOT（每token时间）
- **缓存命中率**: RadixAttention 缓存命中情况
- **GPU利用率**: 显存使用、计算利用率

### 10.2 日志系统
- **结构化日志**: JSON 格式日志
- **分级日志**: DEBUG、INFO、WARNING、ERROR
- **请求追踪**: 请求ID贯穿整个处理流程

## 11. 安全与可靠性

### 11.1 安全机制
- **请求验证**: 输入验证和清洗
- **资源限制**: 最大序列长度、最大批大小
- **访问控制**: API 密钥验证（可配置）

### 11.2 可靠性
- **优雅降级**: 硬件故障时自动切换
- **重试机制**: 自动重试失败请求
- **健康检查**: 定期健康检查端点

## 12. 项目结构总结

SGLang 是一个设计精良、性能极致的 LLM 服务框架，其核心优势在于：

1. **创新的缓存机制**: RadixAttention 实现了智能前缀复用
2. **极致的性能优化**: 从调度器到算子全方位优化
3. **广泛的硬件支持**: 统一抽象支持多种硬件平台
4. **生产级质量**: 被多家大型企业验证的可靠性
5. **活跃的社区**: 快速迭代，持续创新

项目代码结构清晰，模块化程度高，便于扩展和维护。通过分层设计和抽象，成功平衡了性能、灵活性和可维护性。

---

*本文档基于 SGLang v0.5.5 版本分析生成*
*最后更新: 2025-12-08*