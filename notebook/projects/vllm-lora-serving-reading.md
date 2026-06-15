# vLLM V1 LoRA Serving 源码级深度阅读

> 2026-06-15 (updated) | 源码: vllm/lora/ + vllm/v1/worker/lora_model_runner_mixin.py + vllm/v1/cudagraph_dispatcher.py
> 核心: Punica SGMV(Segmented Grouped Matrix Vector)→per-token LoRA mapping→lora_a/b_stacked[max_loras,1,rank,input/output]→Triton shrink+expand→★ 不merge→全动态多tenant→CUDA graph: LoRA buffers固定地址→replay前copy_()内容→★ LoRA+prefix caching不兼容(不同adapter→KV不同)→RTX 4090: max_loras=1-2+rank=16→~64MB

## 1. LoRA Request Lifecycle

```
★ ★ 从请求到执行的完整流程:

LoRARequest (msgspec.Struct, request.py:8-73):
  lora_name: str → 用于equality/hash
  lora_int_id: int → 必须>0 → 全局唯一adapter ID
  lora_path: str → 本地路径或HF repo ID
  base_model_name: str | None
  is_3d_lora_weight: bool → MoE 3D vs 2D layout

生命周期:
  1. LoRARequest附加到EngineRequest → 传给scheduler
  2. Scheduler(scheduler.py:589-602): len(scheduled_loras)==max_loras → 拒绝→回waiting queue
  3. InputBatch(gpu_input_batch.py:468-479): request_lora_mapping[req_index]=lora_id
  4. make_lora_inputs()(976-999):
     → prompt_lora_mapping = req_lora_mapping.repeat(num_sampled_tokens) → per-token!
     → token_lora_mapping = req_lora_mapping.repeat(num_scheduled_tokens) → per-token!
  5. LoRAModelRunnerMixin.set_active_loras()(73-91): → LoRAMapping → lora_manager.set_active_adapters()
  6. WorkerLoRAManager._apply_adapters()(194-212): diff→load→activate
```

## 2. Dynamic LoRA Loading — 7阶段Pipeline

```
★ ★ 7阶段从磁盘到GPU:

Stage 1: Path resolution (utils.py:303-357):
  → get_adapter_absolute_path() → 绝对路径/HF Hub snapshot_download()

Stage 2: PEFT config (peft_helper.py:80-112):
  → PEFTHelper.from_local_dir() → adapter_config.json → r/lora_alpha/target_modules
  → scaling = lora_alpha / r (or lora_alpha / sqrt(r) for rsLoRA)

Stage 3: Validation (peft_helper.py:114-128):
  → r <= max_lora_rank / modules_to_save=None / bias="none" / use_dora不支持

Stage 4: Tensor loading (lora_model.py:166-306):
  → from_local_checkpoint() → adapter_model.safetensors(优先) / adapter_model.bin(legacy)
  → lazy-load per key → EP模型可跳过remote expert keys

Stage 5: Weight construction (lora_model.py:117-164):
  → from_lora_tensors() → parse_fine_tuned_lora_name() → 分离lora_A/lora_B
  → LoRALayerWeights per module → move to device+dtype+pin_memory

Stage 6: ★ Packed module merging (model_manager.py:724-774):
  → _create_merged_loras_inplace() → gate_up_proj=gate_proj+up_proj
  → PackedLoRALayerWeights.pack() → scaling merge进lora_b → optimize()
  → ★ 不是base weight merge → 是sub-module LoRA打包!

Stage 7: GPU activation (model_manager.py:285-324):
  → activate_adapter() → lora_index_to_id[]找空slot
  → module.set_lora(index, lora_a, lora_b) → copy到预分配GPU buffers
  → 无LoRA weights → module.reset_lora(index) → 清零slot
```

## 3. Multi-LoRA Serving — Punica SGMV

```
★ ★ ★ vLLM = Punica SGMV → 多tenant → 同batch不同adapter → 极妙!

架构: 两层cache
  → CPU cache(_registered_adapters): 所有加载LoRAModel → max_cpu_loras → LRU eviction
  → GPU cache(_active_adapters + lora_index_to_id[]): GPU上活跃adapter → max_loras slots

Slot allocation:
  → lora_index_to_id: list[int | None] → length=max_loras → 每slot一个adapter
  → +1 slot(index=max_loras) → reserved for "no LoRA"(-1)

★ ★ LoRAMapping (layers/utils.py):
  index_mapping: tuple[int, ...] → per-token LoRA int_id
  prompt_mapping: tuple[int, ...] → per-prompt LoRA int_id (for logits)
  is_prefill: bool
  type: LoRAMappingType (LANGUAGE/TOWER/CONNECTOR)

★ ★ Index conversion (punica_wrapper/utils.py:54-152):
  → convert_mapping() → lora_int_id → slot index → lora_index_to_id.index()
  → no LoRA → slot -1

★ ★ Multimodal LoRA routing:
  → punica_wrapper_mapping: dict → module prefix → separate PunicaWrapper
  → Language model → LLM wrapper / Vision → tower wrapper / Connector → connector wrapper
  → 每wrapper独立管理token→LoRA mapping → 多模态LoRA!

API请求流程:
  POST /v1/load_lora_adapter → {lora_name, lora_path}
  POST /v1/completions → {model: lora_name} → 使用lora_name作为model标识
  POST /v1/unload_lora_adapter → {lora_name} → 卸载adapter
```

## 4. LoRA Weight GPU Memory Layout

```
★ ★ ★ 预分配固定GPU buffers → 地址不变 → CUDA graph replay安全!

base_linear.py (128-149):
  lora_a_stacked = tuple(
    torch.zeros(max_loras, 1, lora_a_out_size, self.input_size, dtype=lora_dtype)
    for _ in range(n_slices)
  )
  lora_b_stacked = tuple(
    torch.zeros(max_loras, 1, lora_b_out_size, max_lora_rank, dtype=lora_dtype)
    for _ in range(n_slices)
  )

★ ★ Buffer维度:
  lora_a_stacked[slice][slot, 1, rank, input_size] → A矩阵: input→rank
  lora_b_stacked[slice][slot, 1, output_size, rank] → B矩阵: rank→output
  → max_loras slots → 每slot一个adapter的权重 → 固定大小!
  → 1维度 = placeholder for per-layer stacking(未来)

★ ★ Scaling optimization: LoRALayerWeights.optimize()
  → lora_b *= (lora_alpha / r) → scaling=1 → runtime不需要scaling → 极快!

★ LoRALayerWeights vs PackedLoRALayerWeights:
  → LoRALayerWeights: 单(A,B) pair → [rank, input] + [output, rank]
  → PackedLoRALayerWeights: packed modules → .pack() → scaling merge进sub-lora_b
  → PackedLoRALayerWeights.pack_moe(): stack成(num_experts, rank, input/output) → MoE LoRA!

★ ★ Kernel metadata (LoRAKernelMeta):
  → token_lora_mapping → per-token slot index
  → token_indices_sorted_by_lora_ids → tokens按LoRA ID排序
  → active_lora_ids → 本batch出现的LoRA IDs
  → num_tokens_per_lora → 每LoRA token数 → CSR-style
  → lora_token_start_loc → cumsum → 每LoRA起始偏移
  → no_lora_flag_cpu → CPU bool → kernel early-exit
  → num_active_loras_cpu → CPU int → torch.compile动态值
```

## 5. LoRA + CUDA Graph — 固定地址策略

```
★ ★ ★ LoRA weights OUTSIDE CUDA graph → 地址固定 → 内容动态 → 极妙!

关键洞察:
  → lora_a/b_stacked = 预分配GPU buffers → 固定地址 → CUDA graph capture记录地址
  → replay前: set_lora() / reset_lora() → .copy_() → 修改内容 → 地址不变!
  → ★ CUDA graph = 固定地址 + 动态内容 → replay安全 → 设计精妙!

CUDA graph capture策略 (cudagraph_dispatcher.py:115-134):

  specialize_active_lora=False (默认):
    → 只capture 1个case: num_active_loras = max_loras + 1
    → +1 = "no LoRA" slot

  specialize_active_lora=True:
    → Powers of 2 up to max_loras, plus max_loras + 1
    → max_loras=8 → num_active_loras=[1, 2, 4, 8, 9]
    → 每count一个CUDA graph → Triton kernel grid dimension(axis=2 for lora_idx)

BatchDescriptor: has_lora: bool + num_active_loras: int → cudagraph dispatch key的一部分

★ Warmup (lora_model_runner_mixin.py:93-131):
  → maybe_setup_dummy_loras() → zero-filled LoRARequests → lora_warmup_rank=min(max_lora_rank, 8)
  → 分配GPU buffer slots → CUDA graph capture看到正确memory layout → 不crash!
```

## 6. LoRA Forward — shrink + expand Triton Kernels

```
★ ★ ★ per LoRA layer forward:
  1. Base linear: output = base_layer.quant_method.apply(base_layer, x, bias) → base output
  2. LoRA: punica_wrapper.add_lora_linear(output, x, lora_a_stacked, lora_b_stacked, 1.0, output_slices)
  3. ★ ★ Inside add_lora_linear:
     → add_shrink(buffer, x, lora_a_stacked, scale) → x @ lora_A → [tokens, rank]
     → add_expand(output, buffer, lora_b_stacked, output_slices) → buffer @ lora_B → [tokens, output]
     → shrink: tokens按LoRA ID排序 → batched GEMM per adapter → 极快!
     → expand: 同样batched → 加到base output → in-place → 无额外内存!

★ ★ ★ vLLM V1不merge LoRA到base weights → 全动态!
  → optimize()只merge scaling进lora_b → 不是base weight merge
  → _create_merged_loras_inplace() → packed format → 不是base merge
  → ★ 为什么不merge: Punica多tenant → 同batch不同adapter → merge只能一个 → 不兼容!
  → ★ 动态LoRA = 多tenant最优 → shrink+expand → 每request独立adapter → 极灵活!

Triton Kernels:
  lora_shrink: [tokens, input_dim] × A[rank, input_dim] → [tokens, rank] → 降维
  lora_expand: [tokens, rank] × B[output_dim, rank] → [tokens, output_dim] → 升维
  fused_moe_lora: MoE + LoRA 融合 → expert forward + LoRA增量
  ★ 同一Triton kernel处理不同LoRA的所有token → batched → 极快!
```

## 7. LoRA + Speculative Decoding

```
★ ★ Draft model共享target LoRA → 不需要单独LoRA!

1. Batch structure: 1 + num_speculative_tokens → prompt_mapping_meta维度
2. ★ LoRA applies to both draft and target: 共享lora_a/b_stacked → 同一adapter
3. CUDA graph: uniform_decode_query_len = 1 + num_speculative_tokens
4. ★ Draft model(EAGLE/MTP)没有自己的LoRA → 基于target+LoRA hidden states → draft也受LoRA影响!

★ ★ RTX 4090: LoRA+EAGLE → adapter同时影响target和draft → 一致性保证!
```

## 8. LoRA + Prefix Caching — ★ 不兼容!

```
★ ★ ★ 关键发现: LoRA和prefix caching本质上不兼容!

原因: KV cache值依赖prefill时应用的LoRA adapter
  → Request A用LoRA-1 → KV values ≠ Request B用LoRA-2 → 同prefix但KV不同!
  → prefix caching hash = chained hash(parent + curr + extra_keys)
  → ★ hash不包含LoRA ID → 可能跨adapter共享block → 错误结果!

★ ★ 正确使用prefix caching + LoRA:
  1. LoRA不应用于prefix部分(常见: LoRA只target特定modules → system prompt无LoRA)
  2. 所有共享prefix的requests用同一LoRA adapter → GRPO rollout_n=8 → 同adapter!
  3. prefix部分来自embedding/lm_head → 可能无LoRA → 安全

★ ★ GRPO训练 + LoRA + prefix caching = 兼容!
  → rollout_n=8 → 所有copies用同一LoRA → 同adapter → prefix可共享 → SGLang 7×省prefill!
  → ★ 但: 多tenant serving → 不同LoRA → prefix caching不兼容 → 必须注意!

★ ★ vLLM vs SGLang:
  → vLLM: prefix hash不含LoRA ID → 跨adapter共享 → 可能错误!
  → SGLang: radix tree → 每adapter独立KV → 不跨adapter共享 → 更安全!
```

## 9. LoRA Memory Budget — RTX 4090最优配置

```
★ ★ 7B模型 LoRA内存计算:

Per layer (max_loras=4, rank=16, bf16):
  → lora_a: 4 * 1 * 16 * 4096 * 2 = 524KB
  → lora_b: 4 * 1 * 4096 * 16 * 2 = 524KB
  → Attention (q/k/v/o): ~1MB * 32 = ~32MB
  → MLP (gate_up/down): ~3MB * 32 = ~96MB
  → Total: ~128MB for 4 concurrent LoRAs at rank 16!

★ ★ RTX 4090最优配置:
  max_loras = 1-2 → 有限GPU → 单tenant或少adapter
  max_lora_rank = 8-16 → 低rank省内存 → rank 16=128MB(4 adapters)
  max_cpu_loras = 4-8 → 更多CPU cache → 避免重复加载
  lora_dtype = bfloat16 → RTX 4090最优dtype
  fully_sharded_loras = False → 无TP → 单GPU
  specialize_active_lora = False → 少slot → specialization不值得

★ ★ 7B INT4 + LoRA内存:
  → INT4 base: ~3.5GB
  → INT8 KV: ~5K blocks
  → LoRA GPU buffers: ~64MB (max_loras=2, rank=16)
  → 可用KV cache: ~18-19GB → ✓ 完全可行!

★ ★ Scheduler overflow: len(scheduled_loras)==max_loras → 拒绝 → 回waiting queue
  → LRUCacheWorkerLoRAManager: CPU cache超容量 → eviction → 最旧adapter释放

LoRA推理开销: ~2-5% (rank=16 vs base model) → 极小!

CLI示例:
  vllm serve meta-llama/Llama-3.1-8B \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank 16 \
    --lora-dtype bfloat16 \
    --max-cpu-loras 8
```

## 10. 关键设计洞察

```
1. Punica SGMV → 多tenant LoRA → 同batch不同adapter → vLLM独有!
   → shrink: 按LoRA ID排序 → batched GEMM → 每adapter独立 → 极快!
   → expand: batched → 加到base → in-place → 无额外内存
   → ★ 这是vLLM LoRA的核心算法 → 多tenant推理 → SGLang无此机制!

2. 固定GPU buffers + 动态内容 → CUDA graph兼容!
   → lora_a/b_stacked: max_loras slots → 固定大小 → 固定地址
   → set_lora()/reset_lora(): .copy_() → 修改内容 → 地址不变 → graph replay安全
   → ★ 与persistent CUDA graph buffers设计一致 → 地址固定+内容动态 → vLLM核心pattern!

3. 不merge → 全动态 → 多tenant最优但单adapter有overhead!
   → merge: 1次 → 之后无额外compute → 单adapter更快
   → dynamic: 每步shrink+expand → 有compute overhead → 但多adapter极灵活
   → ★ 单adapter → merge更快; 多adapter → dynamic更灵活
   → ★ verl/rLLM LoRA merge → save_pretrained → vLLM INT4 → 最优单adapter路径!

4. LoRA+prefix caching不兼容 → 不同adapter→KV不同 → 共享block=错误!
   → GRPO rollout_n=8 → 同adapter → prefix可共享 → 兼容!
   → 多tenant → 不同adapter → prefix不兼容 → 必须注意!
   → ★ vLLM hash不含LoRA ID → SGLang radix tree更安全 → 每adapter独立KV!

5. CPU两层cache → LRU eviction → adapter加载→卸载→重载 → 热adapter缓存!
   → CPU: max_cpu_loras → 所有加载 → LRU eviction → 热adapter保留CPU
   → GPU: max_loras slots → 固定预分配 → copy_() → cold adapter eviction
   → ★ pin_lora() → 热adapter → 常驻GPU → 不evict → 单tenant最优!

6. Scaling merge进lora_b → runtime不需要 → kernel更简单!
   → optimize(): lora_b *= (lora_alpha / r) → scaling=1 → 1次
   → runtime: shrink+expand → scale=1 → 无额外乘法 → 极快!
   → ★ pre-activation optimization → 不是base merge → scaling消除 → kernel简化!

7. Dummy LoRA warmup → CUDA graph capture → 正确memory layout → 不crash!
   → zero-filled adapters → lora_warmup_rank=min(max_lora_rank, 8) → 小但够
   → 分配GPU slots → capture看到正确layout → 所有slot有buffer → 不OOM!
   → ★ 与GPUModelRunner warmup一致 → 先warmup再capture → 防止crash!

8. RTX 4090最优: merge LoRA→INT4→vLLM → 不用dynamic LoRA!
   → 单GPU单adapter → dynamic LoRA overhead不必要 → merge后更快
   → verl/rLLM GRPO+LoRA → merge → save_pretrained → HF → INT4 → vLLM → 4,791 tok/s
   → ★ 这是RTX 4090最优路径 → merge+INT4 → 不需要dynamic multi-LoRA!
   → 但: 多adapter serving → dynamic LoRA → max_loras=2 → ~64MB → 可行!

9. LoRA+MoE → FusedMoEWithLoRA → Triton MoE backend → SM89 OK
   → 2D/3D融合格式 → enable_mixed_moe_lora_format → gate_up+down LoRA融合
   → ★ MoE+LoRA = 同一模型双重微调 → RTX 4090 MoE LoRA serving可行!

10. LoRA推理开销 ~2-5% → 极小 → 不影响throughput!
    → rank=16 → shrink+expand → memory-bound decode → LoRA不影响weight read
    → ★ INT4 base + LoRA → 4,791 tok/s → LoRA overhead ~2-5% → ~4,600 tok/s → 仍极快!
```

---

Sources:
- vllm/lora/request.py (LoRARequest)
- vllm/lora/worker_manager.py (WorkerLoRAManager + LRUCache)
- vllm/lora/lora_model.py (LoRAModel + from_local_checkpoint)
- vllm/lora/peft_helper.py (PEFTHelper)
- vllm/lora/lora_weights.py (LoRALayerWeights + PackedLoRALayerWeights)
- vllm/lora/model_manager.py (LoRAModelManager + LRUCacheLoRAModelManager)
- vllm/lora/layers/base_linear.py (GPU buffer allocation + set_lora/reset_lora)
- vllm/lora/layers/ (All LoRA layer types)
- vllm/lora/layers/utils.py (LoRAMapping)
- vllm/lora/ops/triton_ops/ (lora_shrink/lora_expand/fused_moe_lora kernels)
- vllm/lora/ops/triton_ops/lora_kernel_metadata.py (LoRAKernelMeta)
- vllm/lora/punica_wrapper/ (PunicaWrapper + convert_mapping)
- vllm/v1/worker/lora_model_runner_mixin.py (LoRAModelRunnerMixin)
- vllm/v1/cudagraph_dispatcher.py (LoRA + CUDA graph)
- Punica论文: Chen et al., "Punica: Multi-Tenant LoRA Serving" (2023)
- Background agent research (vLLM LoRA serving internals)
