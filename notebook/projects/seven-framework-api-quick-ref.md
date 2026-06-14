# 7框架API速查表: DeepSpeed / Megatron-LM / vLLM / verl / MindIE / rLLM / PyTorch

> 2026-06-15 | 贴身顾问实用工具 — 关键API、配置模式、一行命令
> 详细笔记见各框架 notebook/projects/ 下的阅读文件

## 总览

| 框架 | 类型 | 一行启动 | RTX 4090最优配置 |
|------|------|----------|------------------|
| DeepSpeed | ZeRO distributed tra | `deepspeed.initialize(model, optimizer, config=conf` | 7B ZeRO-2+CPU Adam+BF16 → 有效; ZeRO-3慢(of |
| Megatron-LM | 3D parallel training | `mcore.initialize_tensor_parallel()` | TP>1无意义(PCIe瓶颈); 单GPU用; MoE EP=1; PP不适合单 |
| vLLM | General inference se | `vllm serve model_name --tensor-parallel-size 1` | 7B INT4+INT8KV+GQA-8+FlashInfer → 4,791  |
| verl | RL training engine — | `hybrid_engine=True, actor_rollout_ref.model.path=.` | 7B GRPO单GPU可行; CPU Adam省50% opt内存; prefi |
| MindIE | Huawei Ascend infere | `mindie-service --config config.yaml` | MindIE不支持RTX 4090(昇腾专用); 用vLLM替代; 910B是企 |
| rLLM | Agentic RL orchestra | `rllm eval gsm8k` | tinker backend单GPU可用; 需vLLM/SGLang推理后端;  |
| PyTorch | Foundation framework | `model = torch.compile(model, mode='reduce-overhead` | FSDP2+BF16+compile = 有效; custom_op注册fuse |

---

## DeepSpeed

**定位**: ZeRO distributed training — offload optimizer states to CPU/NVMe

**内存公式**: `ZeRO-1: 4Ψ+(12+K)Ψ/N | ZeRO-3: (16+K)Ψ/N | ZeRO-Infinity: ≈2Ψ_layer(只持当前层!)`

**RTX 4090建议**: 7B ZeRO-2+CPU Adam+BF16 → 有效; ZeRO-3慢(offload开销); 单GPU用verl更好

| 用途 | API/配置 | 说明 |
|------|----------|------|
| 初始化 | `deepspeed.initialize(model, optimizer, config=config)` | 必须先初始化再训练 |
| ZeRO-1配置 | `{"zero_optimization": {"stage": 1}}` | 只分片optimizer states，通信=DDP |
| ZeRO-2配置 | `{"zero_optimization": {"stage": 2}}` | 分片optimizer+gradients, 通信=ReduceScatter |
| ZeRO-3配置 | `{"zero_optimization": {"stage": 3}}` | 分片optimizer+grad+params, 通信=AllGather+ReduceScatter |
| CPU Offload | `{"zero_optimization": {"offload_optimizer": {"device": "cpu"...` | ZeRO-2/3 offload optimizer到CPU |
| NVMe Offload | `{"zero_optimization": {"offload_optimizer": {"device": "nvme...` | ZeRO-Infinity: offload到NVMe SSD → 支撑任意大模型! |
| NVMe参数Offload | `{"zero_optimization": {"offload_param": {"device": "nvme"}}}` | 参数分片也offload到NVMe → GPU≈0 |
| 混合精度 | `{"bf16": {"enabled": true}}` | BF16唯一正确训练精度 |
| 梯度检查点 | `{"activation_checkpointing": {"partition_activations": true}...` | ZeRO-3配合activation partition |
| Prefetch配置 | `param_persistent_threshold=100000` | 小参数常驻GPU(ds_persist) → 省AllGather |
| AllGather Coalesced | `自动: 多参数合并1次AllGather → O(1)通信` | PartitionedParameterCoordinator自动处理 |
| 训练step | `engine.step()` | 替代optimizer.step() |

---

## Megatron-LM

**定位**: 3D parallel training (TP+PP+DP) — NVIDIA industrial standard

**内存公式**: `TP: Ψ/TP_size per GPU | PP: layers/stages | SP: activation/TP_size`

**RTX 4090建议**: TP>1无意义(PCIe瓶颈); 单GPU用; MoE EP=1; PP不适合单GPU

| 用途 | API/配置 | 说明 |
|------|----------|------|
| 初始化TP | `mcore.initialize_tensor_parallel()` | 设置TP进程组 |
| ColumnParallel | `ColumnParallelLinear(hidden, output, config)` | 列切分: output沿TP分片 |
| RowParallel | `RowParallelLinear(output, hidden, config)` | 行切分: input沿TP分片→AllReduce |
| 1F1B调度 | `--pipeline-model-parallel-size 4` | 1 forward 1 backward交错 |
| Sequence Parallel | `--sequence-parallel` | activation沿seq切分, 省TP倍内存 |
| MoE Layer | `MoELayer(config, submodules)` | 3种dispatcher: allgather/alltoall/flex |
| Expert Parallel | `--expert-model-parallel-size 8` | 64 experts / 8 EP = 8 local experts |
| Latent MoE | `--moe-latent-size 256` | hidden→latent→expert (DeepSeek-V3风格) |
| Shared Expert | `--moe-shared-expert-intermediate-size 256` | 共享expert+overlap(side stream) |
| CUDA Graph | `--use-cuda-graph` | 固定batch/seq shape → 消除launch overhead |
| FSDP2集成 | `TorchFullyShardedDataParallel` | PyTorch FSDP2兼容 |
| 梯度检查点 | `--recompute-granularity selective --recompute-method block` | selective: 只recomputeattention/MoE |

---

## vLLM

**定位**: General inference serving — PagedAttention + Continuous Batching

**内存公式**: `KV Cache = 2 * num_layers * num_heads * head_dim * seq_len * batch_size * dtype_bytes / block_size`

**RTX 4090建议**: 7B INT4+INT8KV+GQA-8+FlashInfer → 4,791 tok/s; EAGLE → 9,088 tok/s

| 用途 | API/配置 | 说明 |
|------|----------|------|
| 启动服务 | `vllm serve model_name --tensor-parallel-size 1` | OpenAI兼容API服务 |
| 离线推理 | `llm = LLM(model=model_name); llm.generate(prompts)` | 批量生成 |
| Async引擎 | `engine = AsyncLLMEngine.from_engine_args(args)` | 异步推理引擎 |
| PagedAttention | `--gpu-memory-utilization 0.9` | KV Cache block管理, 90% GPU内存 |
| Prefix Caching | `--enable-prefix-caching` | hash-based block级prefix sharing |
| 量化 | `--quantization awq_marlin` | INT4 AWQ量化推理 |
| KV量化 | `--kv-cache-dtype fp8_e5m2` | FP8 KV Cache → 省50%内存 |
| Speculative | `--speculative-config model=small_model` | speculative decoding |
| PD分离 | `--kv-transfer-config connector_type=nixl` | prefill-decode disaggregation |
| LoRA服务 | `--enable-lora` | 动态加载LoRA adapter |
| Chunked Prefill | `--max-num-batch-tokens-per-prompt 2048` | 长序列分块prefill |
| 采样参数 | `SamplingParams(max_tokens=100, temperature=0.7)` | per-request采样配置 |

---

## verl

**定位**: RL training engine — GRPO/PPO with Ray distributed

**内存公式**: `7B GRPO+BF16 = 20.04GB fits RTX 4090; LoRA+CPU Adam → 进一步省内存`

**RTX 4090建议**: 7B GRPO单GPU可行; CPU Adam省50% opt内存; prefix grouper省30% KV

| 用途 | API/配置 | 说明 |
|------|----------|------|
| 配置 | `hybrid_engine=True, actor_rollout_ref.model.path=...` | OmegaConf配置 |
| ActorWorker | `ActorRolloutRefWorker(config)` | 训练+rollout混合worker |
| GRPO训练 | `--algorithm adv_estimator=grpo --rollout.n=8` | 8轨迹per task→group advantage |
| PPO训练 | `--algorithm adv_estimator=gae --critic.model.path=...` | 需要critic模型 |
| Rollout引擎 | `rollout_mode=async --rollout.name=vllm` | 异步rollout via vLLM |
| Prefix Grouper | `--actor.use-prefix-grouper=True` | prefix-aware batch grouping |
| CPU Adam | `--actor.optimizer.type=cpu_adam` | offload Adam到CPU→省GPU内存 |
| LoRA训练 | `--actor.model.lora.rank=16` | LoRA adapter训练 |
| FSDP2 | `--actor.strategy=fsdp2` | PyTorch FSDP2分片训练 |
| Megatron策略 | `--actor.strategy=megatron` | Megatron-LM backend |
| Reward函数 | `reward_fn(task, response) → float` | 用户提供reward函数 |
| Checkpoint | `--trainer.save_freq=100` | 每100步保存checkpoint |

---

## MindIE

**定位**: Huawei Ascend inference — CANN+ATB+HCCL (Ascend NPU专用)

**内存公式**: `Ascend 910B: FP16=320 TFLOPS, 64GB HBM; RTX 4090: FP16=82.6 TFLOPS, 24GB`

**RTX 4090建议**: MindIE不支持RTX 4090(昇腾专用); 用vLLM替代; 910B是企业级国产替代

| 用途 | API/配置 | 说明 |
|------|----------|------|
| 启动服务 | `mindie-service --config config.yaml` | OpenAI兼容API |
| 模型配置 | `model_name: qwen3-8b, world_size: 1` | yaml配置文件 |
| ATB算子 | `ATB::FlashAttention / ATB::RMSNorm` | 昇腾专用融合算子 |
| HCCL通信 | `HCCL_AllReduce / HCCL_AllGather` | 昇腾集合通信(类似NCCL) |
| 量化 | `quantize: w8a8 / w4a16` | INT8/INT4量化 |
| 多模型并发 | `multiple_models: true` | 同时服务多个模型 |
| PD分离(roadmap) | `--disaggregated-serving` | 实验性prefill-decode分离 |
| FP8(910C) | `quantize: w8a8_fp8` | Ascend 910C FP8开发中 |
| vLLM-Ascend替代 | `pip install vllm-ascend` | 开源社区替代方案, 2.2K stars |
| openMind替代 | `pip install openmind` | MindIE开源子集 |
| CANN版本 | `CANN 8.0+` | 昇腾驱动/runtime |
| API兼容 | `/v1/chat/completions, /v1/completions` | OpenAI API格式 |

---

## rLLM

**定位**: Agentic RL orchestration — any harness, any backend, any benchmark

**内存公式**: `GRPO = (r-mean)/(std+ε); RLOO = N/(N-1)*(r-mean); REINFORCE++ = centered/batch_std`

**RTX 4090建议**: tinker backend单GPU可用; 需vLLM/SGLang推理后端; same code eval+train

| 用途 | API/配置 | 说明 |
|------|----------|------|
| CLI eval | `rllm eval gsm8k` | 60+内置benchmark即开即用 |
| CLI train | `rllm train gsm8k` | RL训练(自动eval+train) |
| 定义rollout | `@rllm.rollout
def solve(task, config) → Episode` | 任意agent代码 |
| 定义evaluator | `@rllm.evaluator
def score(task, episode) → EvalOutput` | 任意reward逻辑 |
| 训练器 | `AgentTrainer(workflow_class, backend='verl/tinker/fireworks'...` | 一行切换backend |
| Model Gateway | `rllm-model-gateway --worker url --model model_name` | 透明捕获token IDs+logprobs |
| Drift-free多轮 | `--cumulative-token-mode --model Qwen3-8B` | 消除multi-turn tokenization drift |
| Sandbox | `--sandbox docker/daytona/modal/local` | 沙箱环境选择 |
| GRPO配置 | `--algorithm estimator=grpo --rollout.n=8` | GRPO advantage计算 |
| RLOO配置 | `--algorithm estimator=rloo --rollout.n=4` | Leave-one-out baseline |
| Advantage自定义 | `register_rllm_adv_estimator('my_algo', fn)` | 注册自定义advantage计算 |
| TrajectoryGroup | `TrajectoryGroup(trajectories, group_id='task:agent')` | 同组对比→advantage |

---

## PyTorch

**定位**: Foundation framework — distributed, compile, custom ops, FSDP2

**内存公式**: `FSDP2: (16+K)Ψ/N (per-param DTensor); vs FSDP1: flat→需transform`

**RTX 4090建议**: FSDP2+BF16+compile = 有效; custom_op注册fused kernel→不graph break

| 用途 | API/配置 | 说明 |
|------|----------|------|
| torch.compile | `model = torch.compile(model, mode='reduce-overhead')` | JIT编译→Triton kernel |
| FSDP2 | `from torch.distributed._tensor import DTensor, Shard` | per-parameter分片 |
| FSDP2包裹 | `FSDP(model, device_mesh, shard_params=[Shard(0)])` | DTensor-based分片 |
| ProcessGroup NCCL | `dist.init_process_group(backend='nccl')` | NCCL通信后端 |
| AllReduce | `dist.all_reduce(tensor, op=dist.ReduceOp.SUM)` | 梯度同步 |
| custom_op注册 | `@custom_op('my::op')
def my_op(x) → Tensor` | 纯Python注册+compile兼容 |
| custom_op CUDA | `@my_op.register_kernel('CUDA')
def my_op_cuda(x)` | 注册CUDA实现 |
| custom_op fake | `@my_op.register_fake
def my_op_fake(x)` | shape推理(torch.compile) |
| custom_op autograd | `@my_op.register_autograd
def my_op_autograd(x)` | 自定义backward |
| DTensor reshard | `dtensor = DTensor.from_local(local, mesh, [Shard(0)])` | 分片变换 |
| 混合精度 | `torch.amp.autocast('cuda', dtype=torch.bfloat16)` | BF16训练 |
| CUDA Graph | `torch.cuda.make_graphed_callables(fn, sample_inputs)` | 消除launch overhead |

---

## 跨框架通用模式

### 1. 分布式训练通用配置

```python
# 所有框架: 初始化分布式
import torch.distributed as dist
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
```

### 2. BF16训练通用模式

```python
# DeepSpeed: bf16.enabled=True in config
# Megatron: --bf16 flag
# PyTorch: torch.amp.autocast('cuda', dtype=torch.bfloat16)
# verl: actor_rollout_ref.model.dtype=bfloat16
```

### 3. 梯度检查点通用模式

```python
# PyTorch: torch.utils.checkpoint.checkpoint(fn, *args)
# DeepSpeed: activation_checkpointing.partition_activations=True
# Megatron: --recompute-granularity selective
# verl: actor.use_gradient_checkpointing=True
```

### 4. 推理服务通用模式

```python
# vLLM: vllm serve model_name
# SGLang: python -m sglang.launch_server --model-path model_name
# MindIE: mindie-service --config config.yaml
# rLLM gateway: rllm-model-gateway --worker url
```

### 5. RL训练通用模式

```python
# verl: RayPPOTrainer(config) → train()
# rLLM: AgentTrainer(workflow_class, backend='verl') → train()
# GRPO: adv = (r - mean) / (std + ε)  ← verl/rLLM/Megatron共享
```