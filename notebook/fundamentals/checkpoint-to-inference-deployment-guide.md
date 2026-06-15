# 7框架 Checkpoint → 推理部署 完整路径指南

> 2026-06-15 | 实战: 训练checkpoint → 推理模型部署 → 生产级serving
> 核心: ZeRO-3→universal ckpt→HF→vLLM; FSDP2→ModelMerger→HF→vLLM/SGLang; Megatron→TRT-LLM→TensorRT; verl→FSDPModelMerger→HF→vLLM; MindIE→ATB→Ascend NPU; rLLM→Tinker→HF→vLLM; ★ 所有路径最终→HF格式→vLLM/SGLang/TRT-LLM serving

## 1. 跨框架Checkpoint→推理 总览

```
★ ★ ★ 所有训练框架 → 最终 → HF格式 → 推理框架serving!

训练框架                Checkpoint格式            接出路径               推理框架
─────────────────────────────────────────────────────────────────────────────
DeepSpeed ZeRO-3      → per-rank shard           → universal ckpt        → vLLM/SGLang
DeepSpeed ZeRO-2      → per-rank shard           → HF直接                → vLLM/SGLang
FSDP2                 → per-rank DTensor shard   → FSDPModelMerger       → vLLM/SGLang
FSDP1                 → FlatParameter shard      → FullStateDict gather  → vLLM/SGLang
Megatron              → dist_ckpt                → TRT-LLM export        → TensorRT
Megatron              → dist_ckpt                → mbridge HF             → vLLM/SGLang
verl(FSDP backend)    → per-rank DTensor shard   → FSDPModelMerger       → vLLM/SGLang
verl(Megatron backend)→ dist_ckpt                → mbridge HF             → vLLM/SGLang
rLLM Tinker           → HF direct                → 直接加载              → vLLM/SGLang
MindIE                → ATB format               → 直接                  → MindIE Serving

★ ★ 关键洞察: HF格式是训练→推理的桥梁!
  → 所有框架最终需要把checkpoint转成HF格式 → model.safetensors + config.json + tokenizer
  → vLLM/SGLang直接加载HF格式 → 不需要框架特定格式
  → TRT-LLM另一条路径: HF→TRT-LLM build→TensorRT engine → 更高性能
```

## 2. DeepSpeed ZeRO-3 → HF → vLLM/SGLang

```
★ ★ 3种路径 → universal checkpoint最推荐!

路径A: Universal Checkpoint (推荐, DeepSpeed内置)

Step 1: 训练 → ZeRO-3 per-rank shards
  → 每rank: zero_pp_rank_{R}_mp_rank_{M}_model_states.pt
  → 每rank: zero_pp_rank_{R}_mp_rank_{M}_optimizer_states.pt

Step 2: 转换 → Universal Checkpoint
  → python zero_to_fp32.py . output_dir/
  → → 合并所有rank shards → 单个FP32 model.pt → 可直接加载!

  → ★ 或: deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint()
  → → 在Python中合并 → 不需要磁盘中间文件 → 更快!

Step 3: HF格式 → save_pretrained
  → model.load_state_dict(fp32_state_dict)
  → model.save_pretrained("hf_model_dir")
  → → 生成: model.safetensors + config.json + tokenizer.json

Step 4: vLLM serving
  → python -m vllm.entrypoints.openai.api_server --model hf_model_dir
  → → 或: INT4量化 → vLLM FP8/INT4 serving → 4,791 tok/s (7B RTX 4090)

路径B: Universal Checkpoint (新API, 推荐)

  → DeepSpeed 0.15+ → zero_checkpoint_to_hf_path()
  → → 自动合并 → 自动保存HF → 一步完成!
  → → 支持: ZeRO-2/ZeRO-3/BF16/FP32

路径C: ZeRO-2 → HF直接 (更简单)

  → ZeRO-2: 参数不分片 → model.state_dict() → 直接save_pretrained!
  → → 不需要universal checkpoint → 简单!
  → → ZeRO-2+LoRA: merge LoRA → save_pretrained → vLLM直接serving

★ ★ LoRA处理:
  → 训练: LoRA adapters → adapter_model.safetensors
  → 合并: peft_model.merge_and_unload() → 合入base → save_pretrained
  → 或: vLLM动态LoRA加载 → 不merge → 多adapter serving → 更灵活!
  → → vLLM: --enable-lora → adapter加载 → 极快!

★ ★ RTX 4090实战:
  → ZeRO-3单GPU不可用(3Ψ通信) → 只能ZeRO-2+LoRA或CPU offload
  → ZeRO-2+LoRA → merge → save_pretrained → vLLM INT4 → 4,791 tok/s
  → 或: universal checkpoint → HF → INT4量化 → vLLM → 极快
```

## 3. FSDP2 → FSDPModelMerger → HF → vLLM/SGLang

```
★ ★ FSDP2 → DTensor per-rank shards → 合并 → HF → 推理!

Step 1: 训练 → FSDP2 per-rank shards
  → 每rank: model_world_size_{W}_rank_{R}.pt → DTensor local shard
  → 每rank: optim_world_size_{W}_rank_{R}.pt → optimizer shard
  → huggingface/ → config.json + tokenizer (rank 0 always saves)

Step 2: 合并 → FSDPModelMerger (离线工具)
  → python -m verl.model_merger merge --backend fsdp --local_dir <ckpt_path> --target_dir <hf_path>
  → → 读取fsdp_config.json → world_size
  → → ThreadPoolExecutor并行加载所有rank shards
  → → DTensor shards → torch.cat on shard dim → 合并
  → → save_pretrained() → HF format (safetensors + config.json)

  ★ 或: PyTorch native合并
  → with FullStateDictConfig(offload_to_cpu=True, rank0_only=True):
  →   full_state_dict = model.state_dict()
  → → rank 0 gather → full params → CPU → save_pretrained

  ★ 或: FSDP2新API
  → get_model_state_dict(model, full_state_dict=True, ...) → 全参数gather
  → → 一步完成 → 不需要离线合并!

Step 3: vLLM serving
  → python -m vllm.entrypoints.openai.api_server --model <hf_path>
  → → INT4量化 → 更高吞吐

★ ★ vs ZeRO-3 checkpoint合并:
  → ZeRO-3: universal checkpoint → 所有rank合并 → 单个FP32 model.pt → 大文件!
  → FSDP2: per-rank shards → DTensor合并 → safetensors → 更高效!
  → ★ FSDP2 DTensor state dict更自然 → 不需要特殊工具 → PyTorch原生支持!

★ ★ LoRA处理:
  → verl GRPO训练: lora_train_meta.json → 记录LoRA config
  → FSDPModelMerger: 自动读取 → 分离adapter → 或merge into base
  → → 选择: merge(推理最优) 或 keep separate(vLLM dynamic LoRA)

★ ★ RTX 4090实战:
  → 单GPU FSDP2不适用(24GB不够7B BF16 FSDP2 peak ~25.9GB)
  → → LoRA FSDP2 → peak更低 → 可行!
  → → 但: LoRA+ZeRO-2+CPU Adam更简单 → 推荐!
  → 多GPU NVLink → FSDP2+compile → save → merge → vLLM → 最优路径!
```

## 4. Megatron → TRT-LLM Export / mbridge HF → TensorRT / vLLM

```
★ ★ 2条推理路径 → TRT-LLM(生产最优) / mbridge HF(开发调试)

路径A: TRT-LLM Export → TensorRT推理 (推荐生产路径)

Step 1: 训练 → Megatron dist_ckpt
  → model/dist_ckpt/ → sharded model weights
  → optimizer/dist_ckpt/ → sharded optimizer

Step 2: TRT-LLM Export
  → TRTLLMHelper → get_trtllm_pretrained_config_and_model_weights()
  → → TransformerConfig → PretrainedConfig(GPT/LLaMA/Mixtral/Gemma)
  → → MoE config: moe_num_experts + moe_top_k + moe_normalization_mode
  → TRTLLMEngineBuilder → build_and_save_engine()
  → → PluginConfig: paged_kv_cache + remove_input_padding + gpt_attention_plugin + gemm_plugin
  → → BuildConfig → model class → load weights → build engine → save

Step 3: TensorRT Serving
  → trtllm-engine → TensorRT runtime → 最高吞吐!
  → → paged KV cache → continuous batching → 生产级!

路径B: mbridge HF Export → vLLM/SGLang (开发调试)

Step 1: 训练 → Megatron dist_ckpt

Step 2: mbridge HF Export
  → bridge.save_weights() → HF safetensors
  → → 或: bridge.save_hf_adapter() → LoRA adapter单独保存
  → → mbridge: Megatron→HF权重映射 → 自动转换!

Step 3: vLLM/SGLang Serving
  → python -m vllm.entrypoints.openai.api_server --model <hf_path>

★ ★ Megatron MoE → 推理特殊考虑:
  → MoE模型 → MoE config + expert weights → TRT-LLM支持
  → → 或: vLLM MoE serving → expert parallelism → 多GPU
  → → vLLM: --moe-configuration + expert weights → 自动EP

★ ★ RTX 4090实战:
  → Megatron TP>1 PCIe不可用 → 单GPU训练 → checkpoint
  → → mbridge HF → vLLM INT4 → 单GPU推理 → 最优!
  → → TRT-LLM也支持单GPU → 但vLLM INT4更成熟 → 推荐
  → 多GPU集群 → TRT-LLM + TP + MoE EP → 最优生产路径!
```

## 5. verl → FSDP/Megatron Checkpoint → 推理

```
★ ★ verl使用FSDP或Megatron backend → checkpoint格式由backend决定!

verl FSDP backend → FSDP per-rank shards → 同FSDP2路径!
  → FSDPCheckpointManager → save → per-rank DTensor shards + HF config
  → → FSDPModelMerger merge → HF format → vLLM/SGLang
  → → 或: verl内置HF export → "hf_model" in checkpoint_save_contents → rank0 gather

verl Megatron backend → Megatron dist_ckpt → 同Megatron路径!
  → MegatronCheckpointManager → save → dist_ckpt + mbridge HF
  → → TRT-LLM export → TensorRT serving
  → → 或: mbridge HF → vLLM/SGLang

★ ★ GRPO训练 → 推理路径:
  → verl GRPO → actor checkpoint only (no critic!)
  → → actor = target model → 直接用于推理 → 不需要额外转换!
  → → merge LoRA → save_pretrained → vLLM INT4 → 极快!
  → ★ GRPO训练的actor就是推理模型 → 训练→推理无缝衔接!

★ ★ RTX 4090实战:
  → verl GRPO + HYBRID + LoRA → actor checkpoint
  → → TinkerBackend: LoRA merge → save_pretrained → HF
  → → vLLM INT4 + INT8KV → 4,791 tok/s → 极快!
  → → 或: EAGLE spec decode → 9,088 tok/s → 最优!
```

## 6. rLLM Tinker → HF → vLLM/SGLang

```
★ ★ rLLM最简单 → Tinker in-process → HF直接保存!

Step 1: Tinker训练 → save_checkpoint → new SamplingClient
  → save_pretrained() → HF format直接 → 不需要合并!
  → → LoRA: adapter_model.safetensors → 可merge或keep separate

Step 2: vLLM/SGLang Serving
  → python -m vllm.entrypoints.openai.api_server --model <hf_path>

★ ★ rLLM优势: 无分布式 → 无shard合并 → 最简单的训练→推理路径!
  → 单GPU RTX 4090 → Tinker GRPO → save_pretrained → vLLM → 最简路径!
  → → 不需要FSDPModelMerger/universal checkpoint/mbridge → 直接!

★ ★ rLLM vs verl训练→推理路径对比:
  → rLLM: save_checkpoint → new SamplingClient → in-process → 极简
  → verl(HYBRID): CheckpointEngineManager(naive) → generator → in-process → 也极简
  → verl(COLOCATED/STANDALONE): NCCL/NIXL → 跨GPU → 需要合并
  → ★ 单GPU → 两者都极简 → 多GPU → verl需要合并
```

## 7. MindIE → Ascend NPU Serving (RTX 4090不适用)

```
★ ★ MindIE = Ascend专用 → NVIDIA GPU不适用 → 但了解路径!

Step 1: MindIE训练 → ATB格式checkpoint

Step 2: MindIE Serving → 直接 → 不需要转换!
  → MindIE-LLM → 加载ATB格式 → Ascend NPU serving
  → → 或: vLLM-Ascend → 加载HF格式 → Ascend NPU serving

★ ★ RTX 4090: MindIE不适用(NVIDIA GPU) → 用vLLM/SGLang替代!
  → 开源替代: vLLM-Ascend(Ascend NPU) / openMind(多硬件)
  → NVIDIA GPU → vLLM INT4 → 最优路径!
```

## 8. 量化 → 推理加速关键

```
★ ★ 所有路径 → 量化是推理加速的最后一步 → 极关键!

量化方法 → vLLM支持:

| 方法 | 精度 | 7B模型大小 | RTX 4090吞吐 | 推荐度 |
|------|------|-----------|-------------|--------|
| BF16 | 16bit | 14GB | 1,088 tok/s | 基准 |
| FP8(E4M3) | 8bit | 7GB | ~3,000 tok/s | ✓ |
| INT4(GPTQ/AWQ) | 4bit | ~3.5GB | 4,791 tok/s | ★★★ |
| INT4+EAGLE spec | 4bit | ~4.5GB | 9,088 tok/s | ★★★★ |
| INT4+INT8KV+GQA | 4bit+8bitKV | ~4GB | 4,791 tok/s | ★★★ |

量化流程:
  → HF格式模型 → 量化工具(GPTQ/AWQ/FP8) → 量化模型
  → → vLLM直接加载量化模型 → --quantization gptq/awq/fp8
  → → INT8 KV cache → --kv-cache-dtype int8 → 省KV内存!

★ ★ 推荐量化路径:
  → 训练checkpoint → HF合并 → GPTQ INT4量化 → vLLM serving → 4,791 tok/s
  → → 或: EAGLE spec decode → 9,088 tok/s → 最优!
  → → 或: FP8 E4M3 → 3,000 tok/s → 更简单但吞吐低
  → ★ INT4是RTX 4090推理最优 → decode memory-bound → INT4减少weight read → 极快!
```

## 9. 实战决策树 — RTX 4090训练→推理

```
★ ★ RTX 4090训练→推理最优路径决策树:

训练方式 → Checkpoint → 接出 → 量化 → 推理
─────────────────────────────────────────────────────────────────

LoRA ZeRO-2+CPU Adam → universal ckpt → HF → INT4 → vLLM → 4,791 tok/s
  ★ 推荐! 最简单最稳定!

LoRA FSDP2 → FSDPModelMerger → HF → INT4 → vLLM → 4,791 tok/s
  ★ 需2+GPU训练 → 单GPU推理 → NVLink推荐

verl GRPO HYBRID+LoRA → actor ckpt → LoRA merge → HF → INT4 → vLLM → 4,791 tok/s
  ★★ RL训练最优路径! GRPO训练的actor=推理模型!

rLLM Tinker GRPO+LoRA → save_pretrained → HF → INT4 → vLLM → 4,791 tok/s
  ★★ 最简单! 无分布式 → 直接保存HF → 无合并步骤!

Megatron TP=1 → mbridge HF → INT4 → vLLM → 4,791 tok/s
  ✓ 可行但不推荐 → Megatron单GPU不如vLLM/DeepSpeed训练

★ ★ 推荐组合:
  → GRPO训练: verl HYBRID+LoRA 或 rLLM Tinker+LoRA → 最简单RL训练
  → 接出: LoRA merge → save_pretrained → HF → 最快
  → 量化: GPTQ INT4 → vLLM FP8 KV cache → 最优推理
  → Spec decode: EAGLE → 9,088 tok/s → 极快!

★ ★ 关键: GRPO训练的actor model = 推理model → 不需要额外转换!
  → vs PPO: actor≠critic → actor单独转换 → 但PPO RTX 4090不可行(28GB)
  → → GRPO是RTX 4090唯一可行的RL训练 → 也是最简单的训练→推理路径!
```

## 10. 关键设计洞察

```
1. HF格式 = 训练→推理的统一桥梁 → 所有框架最终都转成HF!
   → ZeRO-3: universal ckpt → FP32 → HF
   → FSDP2: DTensor shards → merge → HF
   → Megatron: dist_ckpt → mbridge/TRT-LLM → HF/TensorRT
   → rLLM: 直接HF → 最简单
   → ★ HF格式 = AI infra的"HTTP" → 标准协议 → 所有框架都支持!

2. GRPO训练→推理 = 最简单的路径 → actor model直接使用!
   → PPO: 需要从actor+critic checkpoint提取actor → 复杂
   → GRPO: actor checkpoint = 推理模型 → 不需要转换 → 极简!
   → ★ 这是GRPO的隐藏优势 → 不仅省compute/memory → 还省部署复杂度!

3. 量化是推理加速最后一步 → 所有路径都需要!
   → BF16推理 → 1,088 tok/s → 不够
   → INT4推理 → 4,791 tok/s → 4.4x加速!
   → INT4+EAGLE → 9,088 tok/s → 8.3x加速!
   → ★ 训练用BF16 → 推理用INT4 → 精度保证 → 吞吐极大提升!

4. FSDPModelMerger = verl训练→推理的桥梁 → 离线合并 → HF → vLLM!
   → verl保存FSDP shards → 不能直接推理 → 需合并
   → merger: ThreadPoolExecutor+torch.cat → 离线 → HF → vLLM加载
   → ★ 这是训练→推理的关键一步 → 不合并 → 推理框架无法加载!

5. TRT-LLM = Megatron生产推理路径 → 不走vLLM → 走TensorRT!
   → Megatron定位 = 训练框架 → 推理靠TRT-LLM → 不是vLLM!
   → TRT-LLM: paged KV cache + remove_input_padding + gemm_plugin → 生产级
   → ★ 训练和推理用不同框架 → Megatron训练 → TRT-LLM推理 → 最优!

6. Universal Checkpoint = DeepSpeed ZeRO-3 → HF的关键工具!
   → zero_to_fp32.py → 合并所有rank shards → FP32 model → HF
   → 或: DeepSpeed 0.15+ → zero_checkpoint_to_hf_path() → 一步完成!
   → ★ 这是ZeRO-3推理部署的必经之路 → 没有它 → ZeRO-3 checkpoint不可用!

7. LoRA merge = 训练→推理的最后优化 → merge或dynamic加载!
   → merge: peft_model.merge_and_unload() → 合入base → 最快推理
   → dynamic: vLLM --enable-lora → 多adapter → 更灵活 → 但稍慢
   → ★ merge后推理 = 无LoRA overhead → 和base model一样快!

8. RTX 4090最优路径 = verl/rLLM GRPO + LoRA → INT4 vLLM → 极简极快!
   → 训练: GRPO+LoRA → 17GB ✓ → 单GPU可行
   → 接出: LoRA merge → HF → 最简单
   → 推理: INT4+INT8KV → vLLM → 4,791 tok/s → 或+EAGLE→9,088
   → ★ 这是一条端到端路径 → 训练→推理 → 从checkpoint到serving → 最优!
```

---

Sources:
- verl/utils/checkpoint/fsdp_checkpoint_manager.py (FSDP save/load)
- verl/model_merger/fsdp_model_merger.py (FSDP shard consolidation)
- Megatron-LM/megatron/core/export/trtllm/ (TRT-LLM export)
- DeepSpeed zero_to_fp32.py (universal checkpoint)
- vLLM quantization docs (GPTQ/AWQ/FP8)
- rLLM TinkerBackend (save_checkpoint → new SamplingClient)
- All prior 7-framework source readings
