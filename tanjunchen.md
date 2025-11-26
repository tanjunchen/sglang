# 介绍

sglang 不是一个简单的工具库，而是一个用于与 LLM 交互的领域特定语言（DSL）和运行时引擎。

总结来说，SGLang 解决的核心问题是：通过一门专门的语言和一个高性能的运行时，让开发者能够更轻松、更高效地编写和执行复杂的大语言模型交互程序。

# 一个请求的完整生命周期

结合架构图，让我们追踪一个请求的旅程：
1. 编译期 (Compilation Time)：h

    * 用户定义 @sgl.function。
    * 前端解析该函数，构建出初始的 IR 执行图。

2. 运行期 (Runtime)：

    * 用户程序调用该函数并传入参数（如 translation(text="Hello, world!")）。
    * 前端与运行时协作，将参数绑定到 IR 图上，形成一个具体的、可执行的实例。
    * 运行时调度器开始执行这个图实例：
        * a. 遇到 APPEND_TEXT 节点，Runtime 调用 RadixAttention 引擎处理该文本。引擎查询前缀缓存，若命中则直接获取 KV Cache，若未命中则派发计算并缓存结果。
        * b. 遇到 GENERATE 节点，调度器会等待一批请求的同类节点，然后将这个批量生成的任务提交给后端（如 vLLM）。
        * c. 后端执行 LLM 推理，生成 tokens，返回给 Runtime。
        * d. Runtime 更新内部状态 s，并沿着图的边执行到下一个节点。

    * 此过程持续进行，直到整个 IR 图执行完毕。
    * 最终结果返回给用户程序，并可能以流式方式输出。


# 源码解析

一个请求的一生如下所示：

从http server 进来后，传给tokenizer，然后传给schedule 进程。请求先放到waiting queue，随后被scheduler 取出，通过PrefillAdder 构建一个scheduleBatch，作为running batch 进行推理（forward & sample）。如果run_batch完请求结束，发给detokinizer，随后回到tokenizer，从http server 出去。



如果run_batch 后请求没有结束，则进行下一轮推理，这里有几个判断。
首先，之前的请求是不是chunked prefill 请求，且prefill 还没有做完，如果是，扔回waiting queue（一切需要prefill的请求，都进waitqueue，作为prefill 请求的总生产者）。
然后，看看waiting queue里有没有新item，有的话，接下来作为mix_running （如果支持mix infer）或者 处理 extend/prefill 请求的batch（上一个请求的decode 被延后）。
如果接下来要做的是decode，判断是否接下来是jump forward请求，如果是，扔回waiting queue（需要prefill），否则进行decode的推理。如果oom，需要撤回当前batch并后续重新build batch，也会扔回waitingqueue。


