# verl Checkpoint Management 源码级深度阅读

> 2026-06-15 | 源码: verl/utils/checkpoint/ + verl/checkpoint_engine/ + verl/model_merger/ + verl/trainer/ppo/ray_trainer.py
> 核心: 2层架构(CheckpointManager→磁盘持久化+CheckpointEngineManager→实时权重同步); FSDP每rank独立shard+HF export(rank0 gather); Megatron dist_ckpt+async save; CheckpointEngineManager 5后端(naive/NCCL/NIXL/Kimi/Mooncake/HCCL); FSDPModelMerger离线合并→HF; GRPO只存actor+dataloader→省critic

## 1. Checkpoint 2层架构

```
★ ★ verl checkpoint = 2层系统!

Layer 1: CheckpointManager → 磁盘持久化
  → BaseCheckpointManager → abstract
  → FSDPCheckpointManager → FSDP1/FSDP2 per-rank shard + HF export
  → MegatronCheckpointManager → dist_ckpt + mbridge HF export

Layer 2: CheckpointEngineManager → 实时权重同步
  → 训练→rollout GPU 权重传输
  → 5后端: naive/CUDA IPC/NCCL/NIXL/Mooncake/HCCL
  → 不是磁盘checkpoint → 是live weight sync!

★ 关键: CheckpointManager = 保存到磁盘 → 恢复训练 → 持久化
  CheckpointEngineManager = 训练→rollout权重同步 → 不写磁盘 → 实时!
```

## 2. FSDPCheckpointManager — Per-Rank Sharded Save/Load

```
fsdp_checkpoint_manager.py (57-395):

★ ★ Save (220-395): 每rank独立保存

每rank保存3个文件:
  model_world_size_{W}_rank_{R}.pt → sharded model state dict
  optim_world_size_{W}_rank_{R}.pt → sharded optimizer state dict
  extra_state_world_size_{W}_rank_{R}.pt → lr_scheduler + RNG(cpu/numpy/random/cuda)

FSDP1 context:
  with ShardedStateDictConfig(offload_to_cpu=True):
    → FlatParameter shards → torch.save per rank

FSDP2 context:
  with nullcontext(): → DTensor local shards → torch.save per rank
  → FSDP2 state_dict()直接返回DTensor local shard → 不需要ShardedStateDictConfig!

★ ★ HF Export (287-395): rank 0 only + optional

if "hf_model" in checkpoint_save_contents:
  → rank 0: get_fsdp_full_state_dict(model, offload_to_cpu=True, rank0_only=True)
  → FSDP1: FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
  → FSDP2: get_model_state_dict(model, full_state_dict=True, ...)
  → init_empty_weights() → AutoModelForCausalLM.from_config() → 空骨架
  → save_model.save_pretrained(hf_local_path, state_dict=state_dict) → safetensors!

rank 0 always saves:
  huggingface/ → model_config + tokenizer/processor + generation_config

★ LoRA: lora_train_meta.json → {r, lora_alpha, task_type} → 下游恢复PEFT config

★ Load (138-218): 每rank独立加载
  torch.load(model shard) → model.load_state_dict()
  torch.load(optim shard) → optimizer.load_state_dict()
  torch.load(extra shard) → lr_scheduler + RNG恢复
  ShardedStateDictConfig(offload_to_cpu=True) → CPU加载→避免GPU OOM!
```

## 3. MegatronCheckpointManager — Dist Checkpoint + Async Save

```
megatron_checkpoint_manager.py (115-1298):

★ ★ V2 Split Layout (3个sibling目录):

local_path/
  ckpt_contents.json → manifest(最后写入→signals completion)
  transformer_config.json → Megatron TransformerConfig
  model/
    huggingface/ → HF weights via mbridge(可选)
    dist_ckpt/ → Megatron sharded model weights
  optimizer/dist_ckpt/ → optimizer + lr_scheduler shards
  extra/dist_ckpt/ → rng_state shards

★ ★ Async Save:
  → Megatron.AsyncCallsQueue → async_save_request返回
  → finalize callback(manifest写入+HDFS上传+tracker更新)→attached to last request
  → 不阻塞训练! → 大模型checkpoint保存极慢 → async → 继续训练!

★ ★ Save (1169-1298):
  → FullyParallelSaveStrategyWrapper → 多rank并行写
  → _build_model_sharded_state_dict(metadata):
    → model.sharded_state_dict(metadata=model_metadata) → per VP rank
  → Megatron-FSDP: model.state_dict_for_save_checkpoint() → DTensor-aware
  → mbridge HF export: bridge.save_weights() / bridge.save_hf_weights()

★ Load (856-985):
  → dist_checkpointing.load() → FullyParallelLoadStrategyWrapper
  → Megatron-FSDP: load_fsdp_dtensor_checkpoint
  → Legacy v1 layout detection: _raise_for_old_layout() → reject!
```

## 4. CheckpointEngineManager — 实时权重同步

```
checkpoint_engine/base.py (345-515):

★ ★ 架构: Trainer → [NCCL/NIXL/RDMA] → Rollout → ServerAdapter

update_weights() (470-514) 核心流程:
  1. if backend=="naive"(colocated): trainer.update_weights() → 零拷贝generator!
  2. else:
     a. abort_replicas() → 取消所有in-flight rollout请求
     b. 创建临时RayWorkerGroup → 所有replica workers
     c. release_kv_cache_replicas() → 释放KV cache内存(weights留在原位)
     d. build_process_group() → prepare+build_topology+init_process_group
     e. trainer.update_weights() + rollout.update_weights() → NCCL/NIXL send/receive
     f. finalize() → 所有workers
     g. resume_kv_cache_replicas()
     h. resume_generation_replicas()

★ ★ 5+后端实现:
  → ColocatedCheckpointEngine("naive") → 同进程generator零拷贝
  → NCCLCheckpointEngine → NCCL collective send/receive+bucketing
  → NIXLCheckpointEngine → RDMA(SM90/NVLink必需)
  → KIMICheckpointEngine → Kimi专用
  → MooncakeCheckpointEngine → Mooncake
  → HCCLCheckpointEngine → Ascend HCCL

★ ★ Weight提供: engine.get_per_tensor_param() → yields (name, tensor)

FSDP engine实现(transformer_impl.py:794+):
  → LoRA: collect_lora_params() or collect_merged_lora_params() → merge/unmerge
  → FSDP2: model.named_parameters() → param.full_tensor().detach().cpu()
  → Bucketing: split_weight_chunks() / merge_weight_chunks() → 大tensor分bucket

★ ★ vs rLLM Tinker: rLLM用save_checkpoint→new SamplingClient → 磁盘中转!
  verl用CheckpointEngineManager → 实时通信 → 不写磁盘 → 更快!
  但: naive backend = 同进程generator → 等价于rLLM in-process → 极快!
```

## 5. Training Loop Checkpoint Integration

```
ray_trainer.py (974-1108):

★ ★ _save_checkpoint() (974-1041):
  1. actor_rollout_wg.save_checkpoint(actor_local_path, ...)
  2. critic_wg.save_checkpoint(critic_local_path, ...) → only if use_critic!
  3. torch.save(self.train_dataloader.state_dict(), data.pt)
  4. latest_checkpointed_iteration.txt → atomic write

★ ★ GRPO只保存:
  → actor checkpoint + dataloader state → 无critic → 省50%存储!
  → PPO保存: actor + critic + dataloader → 3倍存储!

★ ★ _load_checkpoint() (1043-1108):
  1. 读取latest_checkpointed_iteration.txt → step number
  2. global_steps = int(folder.split("global_step_")[-1])
  3. actor_rollout_wg.load_checkpoint(actor_path)
  4. critic_wg.load_checkpoint(critic_path) → only if use_critic
  5. torch.load(data.pt) → dataloader.load_state_dict()
  6. checkpoint_manager.update_weights(global_steps) → sync到rollout replicas

★ resume_mode 3种:
  → "disable" → 不恢复
  → "auto" → 找最新checkpoint
  → "resume_path" → 指定路径
```

## 6. FSDP2 / DTensor Checkpoint

```
fsdp_utils.py (422-471):

★ ★ Version Detection:
  fsdp_version(model):
    isinstance(model, FSDP) → 1 (FSDP1)
    isinstance(model, FSDPModule) → 2 (FSDP2)
    else → 0

★ ★ FSDP1: ShardedStateDictConfig → FlatParameter shards
★ ★ FSDP2: nullcontext() → DTensor local shards → state_dict()直接返回!

get_fsdp_full_state_dict() (438-471):
  → FSDP1: FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
  → FSDP2: get_model_state_dict(model, full_state_dict=True, ...)

★ ★ FSDP2 Sharded Save/Load to CPU (1044-1142):
  fsdp2_sharded_save_to_cpu():
    → iterate all parameters → _local_tensor.detach().cpu() + DTensorSpec
  fsdp2_sharded_load_from_cpu():
    → verify device_mesh consistency → copy _local_tensor back from CPU
```

## 7. FSDPModelMerger — 离线合并→HF格式

```
model_merger/fsdp_model_merger.py (35-):

★ ★ 合并FSDP shards → 标准HF格式:
  1. 读取fsdp_config.json → world_size
  2. 加载rank-0 state dict → 提取DTensor/device_mesh信息
  3. ThreadPoolExecutor并行加载所有rank shards
  4. 合并DTensor shards: Shard(N) → torch.cat(dim=N)
  5. save_pretrained() → HF format(safetensors+config.json+tokenizer)

★ CLI:
  python -m verl.model_merger merge --backend fsdp --local_dir <path> --target_dir <path>
  python -m verl.model_merger merge --backend megatron ...

★ LoRA: lora_train_meta.json → 重建PEFT config → 分离adapter和base weights
```

## 8. 关键设计洞察

```
1. 2层架构 → 磁盘持久化+实时同步 → 分离关注点!
   → CheckpointManager → 磁盘 → 恢复训练 → 离线合并 → 下游使用
   → CheckpointEngineManager → 实时 → trainer→rollout → 不写磁盘
   → 分离 = 不同需求不同路径 → 磁盘慢但持久 → 实时快但临时 → 设计正确!

2. Per-rank sharded → 不gather完整模型 → 省内存!
   → 每rank保存自己的shard → torch.save → 不需要AllGather
   → Load: ShardedStateDictConfig(offload_to_cpu=True) → CPU加载 → GPU OOM安全!
   → vs consolidated: rank0 gather → 大模型=70GB+ → OOM风险 → sharded更安全!

3. HF export → rank 0 only → 生产部署桥梁!
   → get_fsdp_full_state_dict → rank0_only=True → 只1个rank gather → 省通信
   → init_empty_weights → 空骨架 → 不占GPU内存 → 填入state_dict → save_pretrained
   → safetensors格式 → vLLM/SGLang直接加载 → 生产部署无缝!

4. Megatron async save → 不阻塞训练 → 大模型关键!
   → AsyncCallsQueue → async_save_request → finalize callback → 继续训练
   → ckpt_contents.json最后写入 → completion signal → 检测是否完整
   → vs FSDP: torch.save阻塞 → 大模型=分钟级 → 训练暂停 → 性能损失!

5. CheckpointEngineManager → vs rLLM Tinker → 两种weight sync范式!
   → rLLM: save_checkpoint → disk → new SamplingClient → 磁盘中转 → 简单但慢
   → verl naive: generator零拷贝 → 同进程 → 等价rLLM in-process → 极快
   → verl NCCL: collective send/receive → 跨GPU → 不写磁盘 → 比磁盘快!
   → verl NIXL: RDMA → 跨节点 → 极快 → 需SM90/NVLink → RTX 4090不能用!

6. GRPO checkpoint → 省critic → 50%存储节省!
   → PPO: actor + critic + dataloader → 3×
   → GRPO: actor + dataloader → 2× → 无critic → 省1整套模型存储!
   → 这与GRPO省50%内存/compute一致 → 全系统优化!

7. FSDPModelMerger → 离线工具 → 训练→推理桥梁!
   → 训练产出FSDP shards → 不直接可用于推理 → 需合并
   → merger: ThreadPoolExecutor + torch.cat → 离线合并 → HF format → vLLM直接用
   → 这是训练→推理的完整pipeline → 训练checkpoint → merge → vLLM serve → 闭环!

8. LoRA checkpoint → lora_train_meta.json → adapter独立管理!
   → 训练: collect_lora_params() → 只保存adapter → 极小(~2.6GB for 7B r=16)
   → 合并: lora_train_meta.json → PEFT config → merge adapter into base → or keep separate
   → 部署: LoRA merge → vLLM serve → or vLLM dynamic LoRA loading → 两种路径!
```

---

Sources:
- verl/utils/checkpoint/checkpoint_manager.py (BaseCheckpointManager)
- verl/utils/checkpoint/fsdp_checkpoint_manager.py (FSDPCheckpointManager)
- verl/utils/checkpoint/megatron_checkpoint_manager.py (MegatronCheckpointManager)
- verl/utils/checkpoint/checkpoint_handler.py (CheckpointHandler for SFT)
- verl/checkpoint_engine/base.py (CheckpointEngineManager)
- verl/utils/fsdp_utils.py (FSDP1/FSDP2 state dict helpers)
- verl/trainer/ppo/ray_trainer.py (training loop checkpoint)
- verl/workers/engine/fsdp/transformer_impl.py (FSDP engine save/load)
- verl/model_merger/fsdp_model_merger.py (FSDP shard consolidation)
- Background agent research (verl checkpoint system)
