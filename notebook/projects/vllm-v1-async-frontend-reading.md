# vLLM V1 Async Frontend 源码分析

> 源码路径: `vllm/v1/engine/async_llm.py`, `core_client.py`, `detokenizer.py`
> 核心类: AsyncLLM, EngineCoreClient, RequestOutputCollector, FastIncrementalDetokenizer

## 1. 双进程架构

```
Frontend Process (AsyncLLM)          Engine Core Process
├── generate()                       ├── Input Thread (ZMQ→queue)
│   └── EngineCoreClient             ├── Output Thread (queue→ZMQ)
│       └── ZMQ DEALER ────────────► └── Main Thread (busy loop)
├── output_handler (asyncio Task)        │
│   └── ZMQ PULL ◄──────────────────────┘
├── InputProcessor
├── OutputProcessor
└── Detokenizer
```

## 2. 请求流程

1. **接收**: `AsyncLLM.generate()` 创建 RequestOutputCollector
2. **处理**: `InputProcessor` 将输入转为 `EngineCoreRequest` (msgpack序列化)
3. **分发**: `EngineCoreClient` 通过 ZMQ DEALER 发送到核心进程
4. **执行**: EngineCore schedule→execute→sample→update
5. **返回**: output_thread → ZMQ PUSH → output_handler
6. **解码**: Detokenizer 增量解码 token ID → 文本
7. **流式**: yield 每个 token 的 delta output

## 3. output_handler 后台任务

- **持久运行**: 在第一个请求时启动, 持续到进程结束
- **分批处理**: 大量输出分块处理, 避免阻塞事件循环
- **错误传播**: 捕获异常并传播到前端

## 4. ZMQ IPC 协议

| Socket类型 | 用途 | 方向 |
|-----------|------|------|
| ROUTER | 输入(前端) | 多引擎路由 |
| DEALER | 输入(核心) | 连接ROUTER |
| PUSH | 输出(核心) | 推送结果 |
| PULL | 输出(前端) | 拉取结果 |
| XSUB/XPUB | 统计 | 发布订阅 |

**序列化**: msgspec/msgpack 二进制格式, 比 JSON 快 5-10x

## 5. 流式输出

```
Request → RequestOutputCollector (per-request)
    ↑ output_handler 调用 put()
    ↓ generate() 中 await collector.get()
    → yield delta output (增量文本)
```

**Delta 合并**: 生产者快于消费者时自动合并输出, 避免冗余传输

## 6. RequestOutputCollector

- `put()`: 非阻塞存储输出
- `get()`: asyncio.Event 等待输出
- `aggregate`: DELTA 模式自动合并增量
- 错误传播: 存储 Exception 并在 get() 时 raise

## 7. Detokenizer 流水线

- **FastIncrementalDetokenizer**: 基于 tokenizers 库的高性能解码
- **增量解码**: `decode_next(token_id)` 每次处理一个 token
- **特殊 token 处理**: 控制前后空格
- **停止词检测**: 检查是否匹配停止条件
- **缓冲区管理**: 支持停止词排除

## 8. 数据并行支持

- 每个引擎有唯一 identity (2 bytes)
- 支持负载均衡和请求路由
- DP Coordinator 协调多实例
- Stats publish (XPUB) 用于负载均衡决策
