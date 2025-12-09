# 深入解析 SGLang 请求处理路径：从输入到输出的完整旅程

## 引言

在大型语言模型（LLM）服务系统中，一个请求从用户输入到最终输出的处理路径涉及多个复杂的组件和优化技术。SGLang 作为高性能的 LLM 服务框架，通过创新的架构设计和多项优化技术，实现了低延迟、高吞吐的推理服务。本文将深入剖析一个请求在 SGLang 中的完整处理路径，揭示其内部工作机制和性能优化策略。

## 1. 请求入口层：接收与解析

### 1.1 HTTP 服务器启动

当 SGLang 服务启动时，首先初始化 HTTP 服务器：

```python
# sglang/srt/entrypoints/http_server.py
class HTTPServer:
    def __init__(self, server_args: ServerArgs):
        # 初始化 FastAPI 应用
        self.app = FastAPI()
        # 注册路由
        self.register_routes()
        # 启动 Uvicorn 服务器
        self.start_server()
```

**关键步骤：**
1. **配置加载**：从 `ServerArgs` 读取 100+ 配置项（模型路径、并行策略、量化设置等）
2. **中间件注册**：添加 API 密钥认证、Prometheus 监控、CORS 处理
3. **路由注册**：注册 `/generate`, `/v1/completions`, `/v1/chat/completions` 等端点
4. **服务器启动**：使用 Uvicorn 启动异步 HTTP 服务器

### 1.2 请求接收与验证

当用户发送请求时，HTTP 服务器首先进行验证：

```python
# 请求示例
{
  "text": "What is the capital of France?",
  "sampling_params": {
    "temperature": 0.7,
    "max_new_tokens": 100
  }
}

# 验证流程
1. 认证检查：验证 API 密钥（如果启用）
2. 参数验证：检查必需字段和参数类型
3. 内容清洗：过滤非法字符和过长的输入
4. 速率限制：检查用户配额
```

**性能优化：**
- **异步处理**：使用 FastAPI 的异步特性，避免阻塞
- **请求批处理**：自动合并短时间内的多个请求
- **连接复用**：支持 HTTP/2 和连接池

## 2. API 适配层：OpenAI 兼容

### 2.1 OpenAI API 适配器

SGLang 提供与 OpenAI API 兼容的接口，方便用户迁移：

```python
# sglang/srt/entrypoints/openai_api_adapter.py
def v1_completions(tokenizer_manager, raw_request: Request):
    # 转换 OpenAI 格式到内部格式
    request = convert_openai_to_sglang(raw_request)
    
    # 处理请求
    result = tokenizer_manager.generate_request(request)
    
    # 转换回 OpenAI 格式
    return convert_sglang_to_openai(result)
```

**格式转换：**
```python
# OpenAI 格式 → SGLang 格式
{
  "model": "gpt-3.5-turbo",
  "prompt": "Hello",
  "max_tokens": 100
}
↓
{
  "text": "Hello",
  "sampling_params": {
    "max_new_tokens": 100
  }
}
```

### 2.2 Tokenizer 管理

在处理请求前，需要将文本转换为 token ID：

```python
# sglang/srt/tokenizer_manager.py
class TokenizerManager:
    def encode(self, text: str) -> List[int]:
        # 使用 HuggingFace tokenizer
        return self.tokenizer.encode(text)
    
    def generate_request(self, request):
        # 编码输入文本
        input_ids = self.encode(request.text)
        
        # 创建生成请求
        return GenerateReqInput(
            input_ids=input_ids,
            sampling_params=request.sampling_params,
            rid=random_uuid(),
            stream=request.stream
        )
```

**关键特性：**
- **缓存复用**：使用 LRU 缓存加速频繁出现的文本编码
- **并发安全**：支持多线程同时调用 tokenizer
- **特殊 token 处理**：正确处理 BOS、EOS、PAD token

## 3. 调度器层：请求管理与调度

### 3.1 调度器初始化

调度器是 SGLang 的核心组件，负责管理请求生命周期：

```python
# sglang/srt/scheduler.py
class Scheduler:
    def __init__(self, server_args: ServerArgs):
        # 初始化请求池
        self.req_pool = ReqPool(server_args.max_num_reqs)
        
        # 初始化内存管理器
        self.memory_manager = MemoryManager(server_args)
        
        # 初始化基数树缓存
        self.radix_cache = RadixCache()
        
        # 初始化批次管理器
        self.batch_manager = BatchManager()
```

### 3.2 请求接收与处理

当请求到达调度器时，执行以下流程：

```python
def add_request(self, req: Request):
    # 1. 分配请求 ID
    req.req_id = self.req_pool.allocate()
    
    # 2. 前缀缓存查找
    prefix_indices = self.radix_cache.match_prefix(req.input_ids)
    
    if prefix_indices:
        # 命中缓存，复用 KV cache
        req.prefix_indices = prefix_indices
        req.kv_cache_allocated = True
    else:
        # 未命中，分配新缓存
        self.memory_manager.allocate_kv_cache(req)
    
    # 3. 加入等待队列
    self.waiting_queue.append(req)
    
    # 4. 触发调度
    self.schedule()
```

**性能优化：**
- **RadixAttention**：通过基数树实现智能前缀复用，命中率可达 80%+
- **零拷贝**：缓存命中时直接引用已有 KV cache，避免数据复制
- **异步调度**：使用事件驱动模型，避免忙等待

### 3.3 批次构建与连续批处理

SGLang 采用连续批处理（Continuous Batching）技术，动态调整批次：

```python
def build_batch(self) -> Batch:
    # 1. 收集可运行请求
    runnable_reqs = self.collect_runnable_reqs()
    
    # 2. 尝试合并到现有批次
    if self.current_batch:
        merged_batch = self.try_merge_batch(self.current_batch, runnable_reqs)
        if merged_batch:
            return merged_batch
    
    # 3. 创建新批次
    return Batch(runnable_reqs)

def collect_runnable_reqs(self) -> List[Request]:
    runnable = []
    for req in self.waiting_queue:
        # 检查是否可以运行
        if self.can_run(req):
            runnable.append(req)
            # 控制批次大小
            if len(runnable) >= self.max_batch_size:
                break
    return runnable
```

**调度策略：**
- **先来先服务（FCFS）**：保证公平性
- **抢占机制**：长序列请求可被短序列抢占
- **优先级调度**：支持用户定义优先级
- **资源感知**：考虑 GPU 显存和计算资源

## 4. 内存管理层：KV Cache 与前缀缓存

### 4.1 RadixAttention：智能前缀缓存

RadixAttention 是 SGLang 的核心创新之一，通过基数树自动复用共享前缀：

```python
# sglang/srt/mem_cache/radix_cache.py
class RadixCache:
    def __init__(self):
        # 根节点
        self.root = TreeNode()
        # 最近使用队列
        self.lru_queue = LRUQueue()
    
    def match_prefix(self, token_ids: List[int]) -> List[int]:
        # 在基数树中查找最长匹配前缀
        node = self.root
        matched_indices = []
        
        for token_id in token_ids:
            if token_id in node.children:
                node = node.children[token_id]
                matched_indices.append(node.value)
                self.lru_queue.touch(node)  # 更新访问时间
            else:
                break
        
        return matched_indices
    
    def insert(self, token_ids: List[int], kv_cache: torch.Tensor):
        # 插入新前缀到基数树
        node = self.root
        
        for token_id in token_ids:
            if token_id not in node.children:
                # 创建新节点
                new_node = TreeNode(token_id, kv_cache)
                node.children[token_id] = new_node
            
            node = node.children[token_id]
            self.lru_queue.touch(node)
```

**优势：**
- **自动复用**：无需手动配置，自动识别共享前缀
- **高效存储**：相同前缀只存储一份 KV cache
- **快速查找**：基数树查找复杂度 O(L)，L 为序列长度

### 4.2 分页内存管理

SGLang 使用分页内存管理 KV cache，类似操作系统的虚拟内存：

```python
# sglang/srt/mem_cache/memory_manager.py
class MemoryManager:
    def __init__(self, server_args: ServerArgs):
        # 分配物理内存块
        self.blocks = self.allocate_blocks(
            num_blocks=server_args.max_num_blocks,
            block_size=server_args.page_size,
            dtype=server_args.dtype
        )
        
        # 空闲块列表
        self.free_blocks = list(range(num_blocks))
        
        # 块引用计数
        self.block_ref_count = [0] * num_blocks
    
    def allocate_kv_cache(self, req: Request):
        # 计算所需块数
        num_tokens = len(req.input_ids) + req.sampling_params.max_new_tokens
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        
        # 分配物理块
        if len(self.free_blocks) >= num_blocks:
            block_ids = self.free_blocks[:num_blocks]
            self.free_blocks = self.free_blocks[num_blocks:]
            
            # 更新引用计数
            for block_id in block_ids:
                self.block_ref_count[block_id] += 1
            
            req.block_table = block_ids
        else:
            # 触发内存回收
            self.evict_blocks()
    
    def evict_blocks(self):
        # LRU 策略回收
        lru_block = self.lru_queue.get_lru_block()
        if self.block_ref_count[lru_block] == 0:
            self.free_blocks.append(lru_block)
```

**性能优化：**
- **动态分配**：根据请求长度动态分配内存块
- **碎片整理**：定期整理内存碎片
- **预分配**：批量预分配减少分配开销

## 5. 模型执行层：前向传播

### 5.1 模型执行器

模型执行器负责运行模型前向传播：

```python
# sglang/srt/model_executor/model_runner.py
class ModelRunner:
    def __init__(self, server_args: ServerArgs):
        # 加载模型权重
        self.model = self.load_model(server_args.model_path)
        
        # 初始化 CUDA Graph
        self.cuda_graph_runner = CudaGraphRunner(self.model)
        
        # 初始化注意力后端
        self.attention_backend = self.init_attention_backend()
    
    def forward(self, batch: Batch) -> torch.Tensor:
        # 1. 准备输入
        input_ids = batch.input_ids.to(self.device)
        positions = batch.positions.to(self.device)
        kv_cache = batch.kv_cache
        
        # 2. 执行模型
        if batch.is_cuda_graphable():
            # 使用 CUDA Graph 加速
            output = self.cuda_graph_runner.run(input_ids, positions, kv_cache)
        else:
            # 标准前向传播
            output = self.model(input_ids, positions, kv_cache)
        
        return output
```

### 5.2 注意力机制

根据模型架构和硬件平台，选择最优注意力实现：

```python
# sglang/srt/layers/attention/selection.py
def select_attention_backend(server_args: ServerArgs, model_config):
    # 1. 检查 DeepSeek MLA
    if is_deepseek_mla(model_config):
        if is_sm90_supported() or is_sm100_supported():
            return "cutlass_mla"  # 使用 CUTLASS MLA 内核
        else:
            return "triton_mla"  # 使用 Triton MLA 内核
    
    # 2. 检查 FlashInfer 可用性
    if is_flashinfer_available():
        return "flashinfer"
    
    # 3. 检查 FlashAttention
    if is_flashattention_available():
        return "flash_attention"
    
    # 4. 回退到原生实现
    return "native"
```

**DeepSeek MLA 优化：**
```python
# sglang/srt/layers/attention/cutlass_mla.py
def cutlass_mla_decode(q_nope, q_pe, kv_cache, seq_lens, page_table):
    # MLA 将 KV 压缩为低维表示
    # q_nope: [batch, num_heads, latent_dim]
    # q_pe: [batch, num_heads, rope_dim]
    # kv_cache: [batch, seq_len, latent_dim + rope_dim]
    
    # 调用 CUTLASS MLA 内核
    output = torch.ops.sgl_kernel.cutlass_mla_decode(
        q_nope, q_pe, kv_cache, seq_lens, page_table,
        workspace, sm_scale, num_kv_splits
    )
    
    return output
```

**性能优化：**
- **CUDA Graph**：捕获静态计算图，减少 Python 开销
- **算子融合**：融合多个算子为一个，减少内存访问
- **精度选择**：根据硬件自动选择 FP16/BF16/FP8

### 5.3 MoE 层执行

对于混合专家模型（如 DeepSeek），执行以下流程：

```python
# sglang/srt/layers/moe/moe_runner.py
class MoeRunner:
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. 路由计算
        gating_output = self.gate(hidden_states)
        
        # 2. Top-k 选择
        topk_weights, topk_ids = self.topk_softmax(gating_output)
        
        # 3. MoE 对齐
        sorted_token_ids, expert_ids = self.moe_align(topk_ids)
        
        # 4. 分组矩阵乘法
        if self.use_fp8:
            # FP8 量化分组 GEMM
            expert_output = self.fp8_grouped_gemm(
                hidden_states, self.expert_weights,
                sorted_token_ids, expert_ids, topk_weights
            )
        elif self.use_fp4:
            # FP4 量化分组 GEMM (Blackwell)
            expert_output = self.fp4_grouped_gemm(
                hidden_states, self.expert_weights,
                sorted_token_ids, expert_ids, topk_weights
            )
        else:
            # BF16 分组 GEMM
            expert_output = self.bf16_grouped_gemm(
                hidden_states, self.expert_weights,
                sorted_token_ids, expert_ids, topk_weights
            )
        
        # 5. 结果合并
        output = self.moe_sum(expert_output, topk_weights)
        
        return output
```

**性能优化：**
- **FP8/FP4 量化**：减少内存带宽和计算量
- **分组 GEMM**：一次性计算所有专家的输出
- **异步执行**：计算与数据传输重叠

## 6. 采样层：生成 token

### 6.1 采样策略

根据采样参数选择不同的采样策略：

```python
# sglang/srt/sampling/sampling_batch_info.py
class SamplingBatchInfo:
    def from_sampling_params(self, sampling_params_list: List[SamplingParams]):
        # 解析采样参数
        for params in sampling_params_list:
            if params.temperature == 0:
                # 贪心采样
                self.sampling_methods.append("greedy")
            elif params.top_k == 1:
                # Top-1 采样
                self.sampling_methods.append("top1")
            elif params.top_p < 1.0:
                # Top-p 采样
                self.sampling_methods.append("top_p")
            elif params.top_k > 0:
                # Top-k 采样
                self.sampling_methods.append("top_k")
            else:
                # 多项式采样
                self.sampling_methods.append("multinomial")
```

### 6.2 采样内核

使用高性能采样内核生成 token：

```python
# sglang/srt/sampling/sampling_kernel.py
def sampling_kernel(
    probs: torch.Tensor,
    sampling_method: str,
    top_k: int,
    top_p: float,
    temperature: float
) -> torch.Tensor:
    # 根据采样方法选择内核
    if sampling_method == "greedy":
        # 贪心采样：选择概率最高的 token
        return torch.argmax(probs, dim=-1)
    
    elif sampling_method == "top_k":
        # Top-k 采样：从 top-k 个 token 中采样
        return torch.ops.sgl_kernel.top_k_sampling(
            probs, top_k, temperature
        )
    
    elif sampling_method == "top_p":
        # Top-p 采样：从累积概率达到 p 的 token 中采样
        return torch.ops.sgl_kernel.top_p_sampling(
            probs, top_p, temperature
        )
    
    elif sampling_method == "top_k_top_p":
        # Top-k + Top-p 组合采样
        return torch.ops.sgl_kernel.top_k_top_p_sampling(
            probs, top_k, top_p, temperature
        )
```

**性能优化：**
- **并行采样**：批量处理多个请求的采样
- **硬件加速**：使用 GPU 并行计算采样
- **确定性**：支持可复现的随机采样

## 7. KV Cache 更新层：状态管理

### 7.1 KV Cache 存储

生成新 token 后，更新 KV cache：

```python
# sglang/srt/mem_cache/kv_cache.py
def store_kv_cache(
    self,
    layer_id: int,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    req: Request
):
    # 1. 定位存储位置
    block_table = req.block_table
    seq_len = len(req.token_ids)
    
    # 2. 计算块内偏移
    block_id = block_table[seq_len // self.block_size]
    block_offset = seq_len % self.block_size
    
    # 3. 存储新 token 的 KV
    self.k_cache[block_id, block_offset] = new_k
    self.v_cache[block_id, block_offset] = new_v
    
    # 4. 更新序列长度
    req.token_ids.append(req.new_token)
```

### 7.2 前缀缓存更新

更新基数树缓存：

```python
# sglang/srt/mem_cache/radix_cache.py
def update_prefix_cache(self, req: Request):
    # 1. 构建完整 token 序列
    full_tokens = req.prefix_tokens + req.new_tokens
    
    # 2. 插入到基数树
    node = self.root
    for token_id in full_tokens:
        if token_id not in node.children:
            # 创建新节点
            new_node = TreeNode(token_id, req.kv_cache)
            node.children[token_id] = new_node
        
        node = node.children[token_id]
        node.last_access_time = time.time()
    
    # 3. 更新 LRU
    self.lru_queue.touch(node)
```

## 8. 响应返回层：输出格式化

### 8.1 结果格式化

将生成的 token 转换为文本并格式化响应：

```python
# sglang/srt/entrypoints/openai_api_adapter.py
def generate_response(self, req: Request) -> Dict:
    # 1. Token 解码
    output_text = self.tokenizer.decode(req.token_ids)
    
    # 2. 构建响应
    response = {
        "id": f"cmpl-{req.rid}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": self.model_name,
        "choices": [
            {
                "index": 0,
                "text": output_text,
                "logprobs": None,
                "finish_reason": req.finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": len(req.input_ids),
            "completion_tokens": len(req.token_ids) - len(req.input_ids),
            "total_tokens": len(req.token_ids)
        }
    }
    
    return response
```

### 8.2 流式输出

对于流式请求，逐步返回生成的 token：

```python
# sglang/srt/entrypoints/http_server.py
async def generate_stream(request: Request):
    # 1. 创建生成器
    generator = tokenizer_manager.generate_request(request)
    
    # 2. 逐步返回结果
    async for token in generator:
        # 格式化 SSE 响应
        sse_data = f"data: {json.dumps(token)}\n\n"
        yield sse_data
        
        # 检查是否完成
        if token.finish_reason:
            yield "data: [DONE]\n\n"
            break
```

**性能优化：**
- **异步生成**：使用 Python 生成器避免内存占用
- **批量编码**：批量解码 token 减少开销
- **连接保持**：保持 HTTP 连接活跃

## 9. 完整请求处理路径总结

### 9.1 时序图

```
时间 →
─────────────────────────────────────────────────────────────

用户
  │ 发送 HTTP 请求
  │───────────────────────────────────────────────────────────────→
  │                                                      HTTP Server
  │                                                      1. 验证请求
  │                                                      2. 解析参数
  │                                                      3. 路由分发
  │←───────────────────────────────────────────────────────────────
  │ 返回接收确认
  │
  │                                                      Tokenizer Manager
  │                                                      1. 编码文本 → token_ids
  │                                                      2. 创建生成请求
  │                                                      3. 发送到调度器
  │←───────────────────────────────────────────────────────────────
  │ 请求已排队
  │
  │                                                      Scheduler
  │                                                      1. RadixAttention 前缀查找
  │                                                      2. 分配请求 ID
  │                                                      3. 加入等待队列
  │                                                      4. 构建批次
  │←───────────────────────────────────────────────────────────────
  │ 请求已调度
  │
  │                                                      Model Runner
  │                                                      1. 准备输入 (input_ids, positions)
  │                                                      2. 执行注意力层
  │                                                      3. 执行 MLP/MoE 层
  │                                                      4. 采样生成 token
  │                                                      5. 更新 KV cache
  │←───────────────────────────────────────────────────────────────
  │ 生成 token
  │
  │                                                      Tokenizer Manager
  │                                                      1. 解码 token → 文本
  │                                                      2. 格式化响应
  │                                                      3. 返回结果
  │←───────────────────────────────────────────────────────────────
  │ 返回最终结果
  │
  │                                                      HTTP Server
  │                                                      1. 封装 HTTP 响应
  │                                                      2. 发送给客户端
  │←───────────────────────────────────────────────────────────────
  │ 接收响应
```

### 9.2 关键路径性能分析

| 阶段 | 主要操作 | 耗时占比 | 优化技术 |
|------|----------|----------|----------|
| 请求接收 | HTTP 解析、验证 | 1-2% | 异步 IO、连接复用 |
| Token 编码 | Tokenizer 编码 | 2-5% | 缓存、批量处理 |
| 调度 | 前缀查找、批次构建 | 3-8% | RadixAttention、连续批处理 |
| 模型执行 | 注意力、MLP、采样 | 70-80% | CUDA Graph、算子融合、量化 |
| KV 更新 | Cache 存储 | 5-10% | 分页内存、异步拷贝 |
| Token 解码 | Tokenizer 解码 | 1-2% | 批量解码 |
| 响应返回 | HTTP 封装 | 1-2% | 流式输出 |

**总延迟分解：**
- **首 token 时间（TTFT）**：主要由模型执行决定，占 70-80%
- **每 token 时间（TPOT）**：主要由采样和 KV 更新决定，占 20-30%

### 9.3 吞吐量优化

**Batching 策略：**
- **动态批次**：根据请求到达速率自动调整批次大小
- **最优批次**：寻找吞吐量和延迟的最佳平衡点
- **抢占机制**：长序列请求可被短序列抢占，提高整体吞吐量

**内存优化：**
- **前缀复用**：RadixAttention 减少 50-80% 的 KV cache 占用
- **分页管理**：避免内存碎片，提高利用率
- **量化压缩**：FP8/FP4 量化减少 50-75% 的内存占用

**计算优化：**
- **CUDA Graph**：减少 10-20% 的计算开销
- **算子融合**：减少 15-30% 的内存访问
- **投机解码**：提升 2-3 倍的生成速度

## 10. 总结

SGLang 的请求处理路径体现了现代 LLM 服务系统的设计精髓：

1. **全栈优化**：从 HTTP 层到内核层，每个环节都经过深度优化
2. **智能缓存**：RadixAttention 实现了自动、高效的前缀复用
3. **动态调度**：连续批处理和抢占机制平衡了延迟和吞吐量
4. **硬件适配**：支持多种硬件平台，自动选择最优内核
5. **量化加速**：FP8/FP4 量化显著降低内存和计算需求

通过这些创新技术，SGLang 实现了比传统框架（如 vLLM、TensorRT-LLM）更高的性能，成为生产环境 LLM 服务的首选方案。

---

*本文基于 SGLang v0.5.5 版本分析*
*最后更新：2025-12-08*