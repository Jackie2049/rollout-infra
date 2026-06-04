# rollout-infra — AI Infra 工程师成长之路

> 从零开始，系统学习 AI 基础设施，成为一名强悍的 AI Infra 工程师

## 仓库运作规则

## 工作日志

- [2026-06-04](diary/2026-06-04.md) — vLLM 安装、推理性能估算工具、LoRA/Tokenizer 笔记
- [2026-06-03](diary/2026-06-03.md) — GPU 基准测试、vLLM 安装、HuggingFace 推理实验、笔记体系搭建

## 规划与路线

- [学习路线图](notebook/roadmap.md) — Phase 1-5 学习计划，从分布式训练到 vLLM 开源贡献
- [知识地图](notebook/knowledge-map.md) — AI Infra 8 大领域技能树，标注掌握程度
- [GPU 实验计划](notebook/gpu-experiment-plan.md) — A16 GPU 上的实验清单和优先级

## 分布式训练

- [分布式训练总览](notebook/fundamentals/distributed-training.md) — DP/TP/PP/SP 演进逻辑，通信原语图解
- [ZeRO 优化器](notebook/fundamentals/zero-optimizer.md) — ZeRO 1/2/3 原理、显存公式、适用场景
- [FSDP 全分片数据并行](notebook/fundamentals/fsdp.md) — FlattenParameter、Sharding 策略、Prefetching、verl FSDP 实践
- [混合精度训练](notebook/fundamentals/mixed-precision.md) — FP16/BF16/FP8、Loss Scaling、Tensor Core 利用
- [梯度检查点](notebook/fundamentals/gradient-checkpointing.md) — 激活重计算策略，用计算换显存
- [NCCL 通信库](notebook/fundamentals/nccl.md) — Ring/Tree AllReduce、调优参数、网络拓扑
- [RDMA 高性能网络](notebook/fundamentals/rdma-networking.md) — InfiniBand/RoCE/iWARP、GPUDirect RDMA、NCCL 调优、实战排错
- [CUDA 编程基础](notebook/fundamentals/cuda-programming-basics.md) — Grid/Block/Thread、内存层次、Coalesced Access、Occupancy、Roofline Model
- [通信与计算重叠](notebook/fundamentals/comm-compute-overlap.md) — DDP Bucket Overlap、1F1B、ZeRO-3 Prefetch
- [MoE 混合专家架构](notebook/fundamentals/moe.md) — 稀疏激活、Top-K 路由、负载均衡、Expert Parallelism、DeepSeek-V3
- [序列并行](notebook/fundamentals/sequence-parallelism.md) — DeepSpeed Ulysses、Ring Attention、USP 统一框架
- [Ray 分布式框架](notebook/fundamentals/ray-framework.md) — Actor/Task 模型、Placement Group、RLHF 调度
- [分布式 Checkpoint](notebook/fundamentals/checkpoint.md) — 分片保存、异步 IO、mmap 加速加载
- [DataLoader 优化](notebook/fundamentals/dataloader.md) — Prefetch、mmap、WebDataset 流式加载

## GPU 编程与优化

- [CUDA 编程基础](notebook/fundamentals/cuda-basics.md) — 线程层次、内存模型、流、Tensor Core
- [Triton 编程](notebook/fundamentals/triton-programming.md) — OpenAI Triton：kernel、autotune、GEMM、FlashAttention
- [torch.compile 与 Inductor](notebook/fundamentals/torch-compile.md) — Dynamo 图捕获、Triton 代码生成、CUDA Graphs 集成
- [FlashAttention](notebook/fundamentals/flash-attention.md) — 1/2/3 深度解析，tiling、online softmax、IO 复杂度
- [Attention Kernel 优化](notebook/fundamentals/attention-kernel-optimization.md) — FlashAttention-3/4、FlashInfer、FlashMLA 对比与选型
- [RoPE 旋转位置编码](notebook/fundamentals/rope.md) — 数学推导、四种实现、五种长度扩展方法、推理优化
- [KV Cache 深度解析](notebook/fundamentals/kv-cache.md) — 显存估算、PagedAttention、GQA/MQA、Prefix Caching、量化
- [KV Cache 量化](notebook/fundamentals/kv-cache-quantization.md) — FP8/INT8/INT4/2-bit 量化对比、KIVI/KVQuant、vLL8 实践、组合优化
- [Multi-head Latent Attention](notebook/fundamentals/mla.md) — DeepSeek-V2 MLA 低秩 KV 压缩、解耦 RoPE、矩阵吸收、93.3% KV Cache 减少
- [Prefix Caching 深度对比](notebook/fundamentals/prefix-caching.md) — vLLM Hash vs SGLang Radix vs verl Group、收益量化
- [GPU 显存优化](notebook/fundamentals/gpu-memory-optimization.md) — 训练显存估算公式、OOM 排查、ZeRO、碎片化
- [GPU 内存分配器](notebook/fundamentals/gpu-memory-allocator.md) — PyTorch Caching Allocator、vLLM Paged Memory、碎片化分析
- [CUDA Graphs](notebook/fundamentals/cuda-graphs.md) — 推理场景 kernel launch 开销优化，录制重放机制
- [GPU 性能分析](notebook/fundamentals/gpu-profiling.md) — 四层方法：nvidia-smi → nsys → ncu → ncu detail

## RL 训练基础设施

- [RLHF/GRPO 训练基础设施](notebook/fundamentals/rlhf-training-infra.md) — PPO 四模型架构、GRPO 无 Critic、verl/OpenRLHF 框架、Prefix Sharing
- [DPO 直接偏好优化](notebook/fundamentals/dpo.md) — 跳过 RM 和 RL 的对齐方法、IPO/KTO/SimPO 变体
- [LoRA 与参数高效微调](notebook/fundamentals/lora-peft.md) — 低秩分解原理、QLoRA、多 LoRA 服务、RLHF 中的应用

## 推理与部署

- [Speculative Decoding](notebook/fundamentals/speculative-decoding.md) — Draft-Verify、Medusa、Eagle 投机采样
- [Continuous Batching](notebook/fundamentals/continuous-batching.md) — Static→Continuous 演进、Chunked Prefill、调度策略
- [Tokenizer 与推理管线](notebook/fundamentals/tokenizer.md) — BPE、词表大小影响、压缩率与推理成本、特殊 Tokens
- [推理量化](notebook/fundamentals/quantization.md) — GPTQ/AWQ/SmoothQuant/FP8 量化方法对比
- [FP8 量化与训练](notebook/fundamentals/fp8-quantization.md) — E4M3/E5M2 格式、缩放策略、Transformer Engine、vLLM FP8 实现
- [模型服务架构](notebook/fundamentals/model-serving.md) — API 网关、负载均衡、HPA 自动扩缩容
- [长上下文 LLM 推理服务](notebook/fundamentals/long-context-serving.md) — KV Cache 显存压力、Chunked Prefill、Sliding Window、服务策略
- [Chunked Prefill 深度解析](notebook/fundamentals/chunked-prefill.md) — vLLM V1 统一调度模型、Chunk Size 三层约束、Partial Prefill 处理、Prefix Caching 组合
- [Disaggregated Serving](notebook/fundamentals/disaggregated-serving.md) — Prefill/Decode 分离、KV Transfer、Splitwise/Mooncake 架构
- [LLM 采样方法](notebook/fundamentals/sampling-methods.md) — Temperature/Top-K/Top-P/Min-P/Beam Search/重复惩罚

## 集群与基础设施

- [K8s GPU 调度](notebook/fundamentals/k8s-gpu-scheduling.md) — Device Plugin、MIG、Gang Scheduling
- [Slurm 集群管理](notebook/fundamentals/slurm.md) — sbatch 提交、NCCL 配置、抢占式调度
- [分布式文件系统](notebook/fundamentals/distributed-fs.md) — Lustre 条带化、IO 优化、并行读写

## 开源项目源码阅读

- [vLLM V1 架构](notebook/projects/vllm-architecture.md) — V1 整体架构、Scheduler/Executor 分离、混合推理
- [vLLM KV Transfer Connector](notebook/projects/vllm-kv-transfer-connector.md) — NIXL 连接器架构、P/D 分离、ZMQ 握手、TP 聚合
- [vLLM V1 Scheduler 源码阅读](notebook/projects/vllm-v1-scheduler-reading.md) — 调度循环、抢占策略、KV Cache 管理、Prefix Caching 集成
- [vLLM 投机解码源码阅读](notebook/projects/vllm-spec-decode-reading.md) — Draft-Verify、Eagle/Medusa/Ngram proposer、Rejection Sampling
- [vLLM Prefix Caching V1](notebook/projects/vllm-prefix-caching-v1.md) — Hash chain、block 生命周期、四种 attention 缓存策略
- [vLLM 开源贡献计划](notebook/projects/vllm-contribution-plan.md) — 11 个 good first issue 筛选和分析
- [vLLM 贡献准备](notebook/projects/vllm-contribution-prep.md) — FlashInfer 后端分析、开发环境搭建
- [Megatron-LM 张量并行](notebook/projects/megatron-tp-reading.md) — ColumnParallelLinear/RowParallelLinear 源码
- [Megatron-LM TP 源码深度阅读](notebook/projects/megatron-tp-source-reading.md) — 通信原语 autograd、TP 数据流、权重分发、SP 扩展、3D 并行交互
- [Megatron-LM 流水线并行](notebook/projects/megatron-pp-reading.md) — 1F1B 调度、气泡分析、通信 overlap
- [verl 架构](notebook/projects/verl-architecture.md) — 三级前缀缓存（系统级+进程级+请求级）
- [vLLM MLA Backend 源码阅读](notebook/projects/vllm-mla-backend-reading.md) — FlashMLA/FlashInfer/Triton/Cutlass MLA 后端、压缩数据流、KV Cache 格式、Backend 选择
- [verl PrefixGrouper](notebook/projects/verl-prefix-grouper.md) — RL 训练中 prompt 复用的分组与调度
- [vLLM V1 KV Cache Manager 源码阅读](notebook/projects/vllm-v1-kv-cache-manager-reading.md) — Block 分配/释放、6 种 Attention Manager、Prefix Caching、Coordinator 架构
- [vLLM V1 Executor 源码阅读](notebook/projects/vllm-v1-executor-reading.md) — Executor→Worker→ModelRunner→AttentionBackend 完整执行路径、CUDA Graph、20+ Attention Backend
- [vLLM MoE Serving 源码阅读](notebook/projects/vllm-moe-serving-reading.md) — 模块化 MoE kernel 架构、8+ All-to-All 后端、Router/Experts/Prepare-Finalize 分离
- [verl 源码架构阅读](notebook/projects/verl-source-reading.md) — HybridFlow RL 训练框架、PPO/GRPO 循环、FSDP+混合推理引擎、Prefix Grouper
- [vLLM 量化推理管线源码阅读](notebook/projects/vllm-quantization-pipeline-reading.md) — 量化方法注册表、Kernel 选择逻辑、离线/在线量化、QuantKey 系统
- [vLLM 权重加载管线源码阅读](notebook/projects/vllm-weight-loading-reading.md) — safetensors 到 GPU 全流程、TP/PP/EP 分片、weight_loader 机制
- [vLLM P/D 分离架构源码阅读](notebook/projects/vllm-pd-disaggregation-reading.md) — KV Connector 抽象、NIXL 实现、异步传输、异构 TP、部署配置
- [vLLM CUDA Graph 源码阅读](notebook/projects/vllm-cuda-graph-reading.md) — FULL/PIECEWISE 模式、捕获回放流程、Attention 兼容、Breakable Graph
- [vLLM V1 架构全景图](notebook/projects/vllm-v1-architecture-map.md) — 请求生命周期、四层架构、模块依赖关系、已学清单
- [vLLM Sampling Pipeline 源码阅读](notebook/projects/vllm-sampling-pipeline-reading.md) — Logits 到 Token 完整管线、Top-K/P 实现、Penalty、CUDA Graph 兼容
- [vLLM Data Parallel Serving 源码阅读](notebook/projects/vllm-data-parallel-reading.md) — DP vs TP 扩展、MoE EP、Wave 协调、DBO、Elastic EP
- [vLLM V1 可观测性与指标体系 源码阅读](notebook/projects/vllm-observability-reading.md) — Stats 采集、Prometheus 指标、Logging、告警建议
- [vLLM LoRA Serving 源码阅读](notebook/projects/vllm-lora-serving-reading.md) — Punica Multi-LoRA、批量 LoRA kernel、MoE+LoRA、权重管理
- [GRPO 实战指南](notebook/projects/grpo-practical-guide.md) — verl 框架 GRPO 训练完整流程、数据格式、配置参数、奖励函数
- [SGLang 架构深度分析](notebook/projects/sglang-architecture.md) — RadixAttention 基数树、DSL 前端、FlashInfer 集成、vs vLLM 对比
- [分布式训练排错指南](notebook/projects/troubleshooting-guide.md) — NCCL 超时、OOM、梯度异常等常见问题

## GPU 实验记录

- [A16 基准测试](notebook/experiments/gpu-a16-benchmark.md) — GEMM 14.64 TFLOPS、HBM 76 GB/s、混合精度 3.9x 加速
- [vLLM GPT-2 推理](notebook/experiments/vllm-a16-gpt2-benchmark.md) — vLLM 0.6.4+cu118, Batch=16 达 1875 tok/s, Flash Attention 后端
- [GPU Profiling 深度实验](notebook/experiments/gpu-profiling-a16.md) — GEMM/带宽/Attention/Kernel Launch 五类操作 profiling，A16 约为 A100 的 1/10 性能

## 工具脚本

| 工具 | 说明 | 状态 |
|------|------|------|
| [training_recipe.py](tools/training_recipe.py) | 输入模型规格和 GPU 配置，生成 Megatron/DeepSpeed/DDP 训练启动命令 | 已验证 |
| [gpu_benchmark.py](tools/gpu_benchmark.py) | GPU 环境验证 + GEMM TFLOPS / HBM 带宽 / 混合精度 / Attention 性能测试 | 已验证 |
| [training_config_guide.py](tools/training_config_guide.py) | 根据模型大小和硬件推荐并行策略、batch size、梯度累积 | 已验证 |
| [ddp_train_demo.py](tools/ddp_train_demo.py) | PyTorch DDP 完整训练示例（多进程、梯度同步、checkpoint） | 已验证 |
| [comm_benchmark.py](tools/comm_benchmark.py) | 集合通信性能测试（AllReduce/AllGather/ReduceScatter 延迟和带宽） | 已验证 |
| [ai_infra_cheatsheet.py](tools/ai_infra_cheatsheet.py) | AI Infra 常用命令速查表（nvidia-smi/nccl/pytorch/triton） | 已验证 |
| [inference_estimator.py](tools/inference_estimator.py) | LLM 推理性能估算（显存/KV Cache/吞吐/延迟） | 已验证 |
| [memory_estimator.py](tools/memory_estimator.py) | 训练显存估算（参数/梯度/优化器/激活值逐项计算） | |
| [scaling_simulator.py](tools/scaling_simulator.py) | 多卡扩展效率模拟（通信占比、线性度、最优卡数） | |
| [gpu_monitor.py](tools/gpu_monitor.py) | GPU 实时监控（利用率/显存/温度/功耗） | |
| [triton_examples.py](tools/triton_examples.py) | Triton kernel 示例集（vector_add/GEMM/softmax/flash_attn） | |
| [ddp_cpu_demo.py](tools/ddp_cpu_demo.py) | PyTorch DDP CPU 模式 demo（无 GPU 也可跑） | |
| [collective_ops_viz.py](tools/collective_ops_viz.py) | 集合通信原语可视化（AllReduce/AllGather/Broadcast 数据流） | |
| [vllm_experiment.py](tools/vllm_experiment.py) | vLLM 推理 benchmark（不同 batch/seq 配置、Prefix Caching 测试） | 待 GPU |
| [moe_router_demo.py](tools/moe_router_demo.py) | MoE Router 模拟器（Top-K/Capacity 路由、负载均衡分析、Dense vs MoE 对比） | 已验证 |
| [expert_parallelism_sim.py](tools/expert_parallelism_sim.py) | Expert Parallelism 模拟（All-to-All 通信、EP vs TP vs DP 对比、DeepSeek-V3 EP） | 已验证 |
| [gpu_profile_experiment.py](tools/gpu_profile_experiment.py) | GPU profiling（GEMM/带宽/Attention/Kernel Launch/Reduction 五类操作） | 已验证 |
| [cuda_graph_demo.py](tools/cuda_graph_demo.py) | CUDA Graph 实验（eager vs graph 对比、batch/kernel 数量 scaling） | 已验证 |
| [flash_attention_bench.py](tools/flash_attention_bench.py) | Flash Attention 微基准（SDPA vs Naive、显存对比、序列长度/batch scaling） | 已验证 |
| [inference_latency_breakdown.py](tools/inference_latency_breakdown.py) | 推理延迟分解（逐操作耗时、GPT-2/LLaMA decode step profiling） | 已验证 |
| [quantization_bench.py](tools/quantization_bench.py) | 量化推理 benchmark（FP32/FP16/BF16/INT8 性能+精度+内存对比） | 已验证 |
| [pytorch_inference_bench.py](tools/pytorch_inference_bench.py) | PyTorch 原生推理 benchmark（KV Cache 加速比、Prefill/Decode 延迟、Batch 扩展效率） | 已验证 |
| [fp8_simulation.py](tools/fp8_simulation.py) | FP8 量化模拟（E4M3/E5M2 格式、缩放策略、Outlier 影响、误差传播） | 已验证 |
| [fsdp_memory_sim.py](tools/fsdp_memory_sim.py) | FSDP 内存模拟（DDP vs FSDP 对比、Sharding 策略、Wrapping 分析、Prefetching） | 已验证 |
| [sampling_simulator.py](tools/sampling_simulator.py) | LLM 采样模拟（Temperature/Top-K/Top-P/Min-P/Beam Search/重复惩罚） | 已验证 |
| [memory_allocator_sim.py](tools/memory_allocator_sim.py) | GPU 内存分配器模拟（Naive vs Caching vs Paged, 碎片化分析） | 已验证 |
| [prefix_caching_sim.py](tools/prefix_caching_sim.py) | Prefix Caching 模拟（Hash/Trie/Group 策略对比, RL 场景, 收益量化） | 已验证 |
| [continuous_batching_sim.py](tools/continuous_batching_sim.py) | Continuous Batching 模拟（Static vs Continuous, 调度策略, KV Cache 压力, Chunked Prefill） | 已验证 |
| [speculative_decoding_sim.py](tools/speculative_decoding_sim.py) | Speculative Decoding 模拟（Draft-Verify、质量影响、K 最优化、批量扩展） | 已验证 |
| [disaggregated_serving_sim.py](tools/disaggregated_serving_sim.py) | Disaggregated Serving 模拟（Monolithic vs P/D 分离、KV Transfer、实例比例优化） | 已验证 |
| [flash_attention_sim.py](tools/flash_attention_sim.py) | FlashAttention 算法模拟（tiling、online softmax、IO 分析、block size 影响） | 已验证 |
| [sequence_parallel_sim.py](tools/sequence_parallel_sim.py) | 序列并行模拟（Ulysses vs Ring Attention vs USP、通信量、扩展效率） | 已验证 |
| [gemm_roofline.py](tools/gemm_roofline.py) | GEMM Roofline 分析（Prefill vs Decode 瓶颈、算术强度、GPU 对比） | 已验证 |
| [tensor_parallel_sim.py](tools/tensor_parallel_sim.py) | 张量并行通信分析（TP 效率、通信占比、模型规模、推荐策略） | 已验证 |
| [profiling_guide.py](tools/profiling_guide.py) | LLM 推理性能诊断（Prefill/Decode 分解、KV 压力、吞吐估算、瓶颈诊断） | 已验证 |
| [pipeline_parallel_sim.py](tools/pipeline_parallel_sim.py) | 流水线并行分析（GPipe vs 1F1B 气泡、micro-batch 影响、3D 并行策略推荐） | 已验证 |
| [training_optimizer_sim.py](tools/training_optimizer_sim.py) | 训练优化模拟（梯度累积、混合精度、梯度检查点、综合显存估算） | 已验证 |
| [collective_comm_sim.py](tools/collective_comm_sim.py) | 集合通信深度分析（Ring/Tree AllReduce、AllGather/ReduceScatter、通信重叠） | 已验证 |
| [nccl_tuning_sim.py](tools/nccl_tuning_sim.py) | NCCL 调优参数分析（算法/协议选择、通道调优、跨节点拓扑、速查表） | 已验证 |
| [checkpoint_perf_sim.py](tools/checkpoint_perf_sim.py) | Checkpoint 性能分析（大小估算、保存策略、加载恢复、存储推荐） | 已验证 |
| [dataloader_perf_sim.py](tools/dataloader_perf_sim.py) | DataLoader 优化分析（加载瓶颈、Prefetch 预取、存储对比、调优推荐） | 已验证 |
| [ray_schedule_sim.py](tools/ray_schedule_sim.py) | Ray 分布式调度模拟（Actor/Task 调度、Placement Group、RLHF PPO） | 已验证 |
| [long_context_serving_sim.py](tools/long_context_serving_sim.py) | 长上下文推理模拟（KV Cache 压力、Chunked Prefill、Sliding Window、Prefix Caching、服务策略） | 已验证 |
| [inference_planner.py](tools/inference_planner.py) | LLM 推理端到端规划器（单卡部署、多卡并行策略、Batch 优化、成本效率） | 已验证 |
| [kv_cache_offload_sim.py](tools/kv_cache_offload_sim.py) | KV Cache 分层存储模拟（GPU/CPU/SSD 三层容量、Offload 性能、服务策略、成本优化） | 已验证 |
| [chunked_prefill_sim.py](tools/chunked_prefill_sim.py) | Chunked Prefill 模拟（无 chunking vs chunked、chunk size 影响、并发调度、Prefix Caching 组合） | 已验证 |
| [kv_cache_quant_sim.py](tools/kv_cache_quant_sim.py) | KV Cache 量化模拟（FP16/FP8/INT4/2-bit 对比、Batch 影响、组合优化、成本效率） | 已验证 |
| [serving_framework_compare.py](tools/serving_framework_compare.py) | LLM Serving 框架对比（vLLM/SGLang/TRT-LLM 吞吐、KV 共享、结构化生成、扩展效率） | 已验证 |
| [inference_cost_calculator.py](tools/inference_cost_calculator.py) | LLM 推理部署成本计算器（GPU 选型、量化策略、场景成本、能效分析） | 已验证 |
| [cuda_kernel_simulator.py](tools/cuda_kernel_simulator.py) | CUDA Kernel 执行模拟器（Grid/Block、内存层次、Occupancy、Roofline 分析） | 已验证 |
| [cuda_stream_simulator.py](tools/cuda_stream_simulator.py) | CUDA Stream 并发模型模拟器（通信计算重叠、TP overlap、vLLM 多 Stream） | 已验证 |
| [nccl_tuning_cheatsheet.py](tools/nccl_tuning_cheatsheet.py) | NCCL 调优速查表（算法选择、通道调优、多节点配置、问题排查） | 已验证 |
| [llm_latency_estimator.py](tools/llm_latency_estimator.py) | LLM 推理延迟估算器（Prefill/Decode 分解、TP 影响、场景估算、GPU 对比） | 已验证 |
| [model_compression_analyzer.py](tools/model_compression_analyzer.py) | LLM 模型压缩策略分析器（蒸馏/剪枝/量化/综合策略对比、生产部署选型） | 已验证 |
| [dp_serving_simulator.py](tools/dp_serving_simulator.py) | Data Parallel Serving 模拟器（DP vs TP 扩展、MoE EP、Wave 协调、DBO、生产选型） | 已验证 |

### 日志记录规范

- **文件**: `diary/YYYY-MM-DD.md`，每天一个文件
- **颗粒度**: 每完成一件事（学一个知识点、做一个实验、写一个工具）就记录一条
- **格式**: `HH:MM — 标题` + 2-3 句话，包含做了什么、为什么做、得出了什么结论或思考
- **顺序**: 按时间正序排列（最早的在最前面）
- **要求**: 不要太简略（不能只列名字），也不要太冗长；要能看到进展、结论和思考

### 笔记记录规范

- **基础知识**: `notebook/fundamentals/` — 每篇包含目标、核心概念、实践记录、参考资料
- **项目源码**: `notebook/projects/` — 源码阅读和分析笔记
- **实验记录**: `notebook/experiments/` — GPU 实验数据和结论
- **规划文档**: `notebook/roadmap.md`, `notebook/knowledge-map.md`

### 工具沉淀规范

- 每个工具独立可运行（有 `if __name__ == "__main__"` 入口）
- 包含 docstring 说明用法
- 验证状态标注在上方的目录树中

### Git 管理规范

- 每完成一个有意义的任务就 commit + push
- commit message 简要说明变更内容
- 所有变更通过 git 管理，不手动管理文件

### 环境信息

- **本地 conda**: `ai-infra` (Python 3.11, macOS)
- **GPU 服务器**: matpool.com A16 15GB, `myconda` 环境 (Python 3.9, PyTorch 2.0+cu117)
- **GPU 连接**: `source /root/miniconda3/bin/activate myconda`（必须激活，不能用系统 Python）
- **镜像源**: pip 用清华/阿里云镜像（中国大陆网络）

## 统计

| 类别 | 数量 |
|------|------|
| 基础知识笔记 | 45 篇 |
| 项目源码阅读 | 29 篇 |
| 实验记录 | 3 篇 |
| 实用工具脚本 | 56 个 |
| **总计** | **132 项** |

## GitHub

https://github.com/Jackie2049/rollout-infra
