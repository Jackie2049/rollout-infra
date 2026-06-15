# vLLM V1 GRPO Rollout Scheduler Interaction (2026-06-15)

> ★★★ vLLM作为GRPO rollout引擎时的调度器行为 → RTX 4090 24GB内存约束下的关键分析
> 源码: vllm/v1/core/sched/scheduler.py + kv_cache_manager.py + verl rollout worker

---

## 1. GRPO rollout_n=8 调度行为

### 1.1 verl如何提交请求

```
verl GRPO rollout phase:
  → repeat_interleave(prompt, n=8) → 8个expanded请求
  → 一次性提交到vLLM作为batched generation call
  → vLLM连续批处理 → 同时处理8个请求
```

### 1.2 V1 Scheduler admission逻辑

```
★★★ schedule() 两约束:
  1. max_num_running_reqs (= max_num_seqs) → 硬上限concurrent序列数
  2. token_budget (= max_num_scheduled_tokens) → 每step总token预算

  调度循环:
  1. 先处理RUNNING (decode) → 每request分配1 token
  2. 再从WAITING queue中admit → 消耗prompt token count从budget
  3. rollout_n=8 → 8个expanded请求一起入waiting queue → 依次admit

  ★★★ chunked prefill: 不需要 upfront全部KV → allocate_slots支持partial allocation
  → 8个请求可以incrementally admit → 即使full KV budget不能一次满足

  ★★★ full_sequence_must_fit (scheduler_reserve_full_isl):
  → 可以强制: 整个sequence长度必须可行才admit → 防止over-admission
```

---

## 2. Prefix Caching与共享System Prompt

### 2.1 Hash-based prefix matching

```
★★★ get_computed_blocks(request):
  → coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
  → GRPO rollout: 8个请求共享system prompt → 相同block_hashes → 全部命中!

  ★★★ Reference counting:
  → 共享prefix blocks用ref_cnt → 第1个request计算system prompt → blocks cached+hashed
  → 后续7个请求通过hash查找找到blocks → ref_cnt += 1
  → blocks不被free直到所有referencing requests完成

  ★★★ get_num_common_prefix_blocks:
  → scheduler计算所有running requests的最长common prefix (scheduler.py:957-964)
  → 8个GRPO请求共享system prompt → 识别shared prefix length
  → → cascade attention → shared prefix KV只计算一次 → 8个序列共享!
```

### 2.2 内存节省

```
System prompt = 256 tokens = 16 blocks (block_size=16)

  Without prefix caching: 8 × 256 × KV_per_token
  With prefix caching: 1 × 256 × KV_per_token → 省7×!

  Qwen2-7B (GQA, INT8 KV): 省约98MB (7 × 256 × 14MB/seq_prefix)
  Llama-2-7B (MHA): 省约459MB (7 × 256 × 256KB/token)
```

### 2.3 ★★★ Prefix caching必须显式启用!

```
★★★ V1中APC是OPTIONAL → 必须设置:
  --prefix-caching-no-always 或 cache_config.enable_prefix_caching=True

  ★★★★★ Without it: 8个rollout请求各自独立计算system prompt →
    7× compute waste + 7× memory waste → GRPO on RTX 4090必须启用!
```

---

## 3. Preemption行为 (★★★★★ RTX 4090关键)

### 3.1 V1 Retraction机制

```python
★★★★★ _preempt_request (scheduler.py:1056):
  self.kv_cache_manager.free(request)        # ★★★ 释放ALL KV cache blocks
  self.encoder_cache_manager.free(request)    # 释放encoder cache
  self._inflight_prefills.discard(request)    # 从inflight set移除
  request.status = RequestStatus.PREEMPTED
  request.num_computed_tokens = 0             # ★★★★ 重置为0 → 完全从头来过!
  request.num_preemptions += 1
  self.waiting.prepend_request(request)       # 放回waiting queue前端 → 优先重新admit
```

★★★★★ **V1使用retraction → KV cache完全丢弃 → 无CPU swap → 无KV snapshot保存**

### 3.2 Eviction策略

```
★★★ FCFS (默认): self.running.pop() → 淘汰最近admitted的request
★★★ PRIORITY: max(self.running, key=priority+arrival_time) → 淘汰最高priority值
```

### 3.3 GRPO Rollout Preemption场景

```
★★★★★ 当RTX 4090内存紧张:
  → scheduler无法allocate_slots → while True循环 → preempt request → free blocks → retry

  GRPO request #5被preempted:
  → 丢失所有progress (KV cache freed, computed_tokens reset to 0)
  → ★★★ 但如果prefix caching启用 → 其他group成员仍持有system prompt blocks (ref_cnt保护)
  → request #5重新admit → 通过hash查找找到prefix blocks → 只需重新计算unique portion!

  ★★★★★ 如果没有prefix caching → request #5完全从头来过 → 包括system prompt prefill → 2×慢!
```

---

## 4. LoRA Adapter调度

### 4.1 LoRA ID tracking

```python
★★★ scheduler.py (580-588, 617-630, 912-913):
  scheduled_loras: set[int] = set()
  if self.lora_config:
    scheduled_loras = set(
      req.lora_request.lora_int_id
      for req in scheduled_running_reqs
      if req.lora_request and req.lora_request.lora_int_id > 0
    )
    assert len(scheduled_loras) <= self.lora_config.max_loras
```

### 4.2 Admission gating

```python
★★★★ LoRA admission cap:
  if len(scheduled_loras) == self.lora_config.max_loras
    and request.lora_request.lora_int_id not in scheduled_loras:
    # ★★★ 超出max_loras → skip request → defer to later step
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

### 4.3 GRPO weight sync (★★★★★ 关键!)

```
★★★★★ verl GRPO weight sync循环:
  1. Rollout phase: vLLM生成8 responses (with LoRA if used)
  2. Training phase: training workers更新policy (LoRA) weights
  3. Weight sync: verl调用 update_weights → push updated weights → CUDA IPC/ZMQ
  4. ★★★★ Cache reset: reset_prefix_cache(reset_running_requests=True)
     → preempt ALL running requests + clear ENTIRE prefix cache
     → ★★★ 因为model weights变了 → cached KV blocks now stale → 必须清理!

  ★★★★ LoRA adapter weights在weight sync期间in-place更新
  → 无单独"deactivation" → adapter直接replaced

  ★★★★★ Multi-LoRA GRPO: max_loras限制 → requests超出limit被deferred
```

---

## 5. RTX 4090 内存预算分析

### 5.1 Core formula

```
KV_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_size_bytes
total_available_tokens = (usable_memory - model_weights) / KV_per_token
```

### 5.2 Budget for 7B models (24GB, gpu_memory_utilization=0.9)

| Model | KV_heads | KV/token | Model wt | Available KV | Total tokens |
|-------|----------|----------|----------|--------------|--------------|
| Llama-2-7B (MHA, fp16) | 32 | 256KB | ~14GB | ~7.6GB | ~29,600 |
| ★★★ Qwen2-7B (GQA, fp16) | 4 | 32KB | ~14GB | ~7.6GB | ~236,800 |
| Llama-3-8B (GQA, fp16) | 8 | 64KB | ~16GB | ~5.6GB | ~87,500 |
| ★★★★ Any 7B (INT4) | varies | same | ~4GB | ~17.6GB | 4-8× more |

### 5.3 rollout_n=8 + max_model_len=1024

```
Without prefix caching: 8 × 1024 = 8,192 tokens required
  Llama-2-7B (MHA): 8,192 < 29,600 → feasible but ~3 prompt groups only
  ★★★ Qwen2-7B (GQA): 8,192 < 236,800 → easily feasible → ~28 prompt groups!

With prefix caching (shared 256-token system prompt):
  256 (shared) + 8 × 768 (unique) = 6,400 tokens → 省约12% KV memory
```

### 5.4 ★★★★★ Preemption trigger analysis

```
★★★★★ Llama-2-7B (MHA) + rollout_n=8 + max_model_len=2048:
  1 group = 16,384 tokens → fits in 29,600 ✓
  2 groups (16 requests) = 32,768 tokens → exceeds 29,600 → ★★★ PREEMPTION!
  → scheduler preempt requests → recompute → thrashing pattern → disaster!

★★★★★ Qwen2-7B (GQA) + rollout_n=8 + max_model_len=2048:
  1 group = 16,384 tokens → fits in 236,800 → ★★★ comfortable!
  14 groups concurrently → ★★★★ GQA是RTX 4090关键优化!

★★★★★ INT4 weights: available KV → 17.6GB → 4-8× more tokens → 即使MHA也可行!
```

### 5.5 ★★★ Recommended GRPO configuration for RTX 4090

```yaml
★★★★★ Qwen2-7B (GQA) — 推荐配置:
  model: Qwen/Qwen2-7B-Instruct
  gpu_memory_utilization: 0.90
  max_model_len: 1024
  max_num_seqs: 48  # ~6 prompts × rollout_n=8
  enable_prefix_caching: True  # ★★★ 必须! GRPO共享system prompt
  kv_cache_dtype: int8  # ★★★ SM89可行 → 省一半内存

★★★★★ INT4+INT8KV (最优路径):
  model: Qwen2-5-7B-Instruct-GPTQ-INT4
  gpu_memory_utilization: 0.90
  max_model_len: 4096  # INT4→更多KV空间
  max_num_seqs: 89  # 大量concurrent requests
  enable_prefix_caching: True
  kv_cache_dtype: int8  # INT8 KV → FlashInfer SM89支持
```

---

## 6. 关键洞察

1. ★★★★★ **Preemption = retraction** → KV cache完全丢弃 → 无CPU swap → 从头来过 → prefix caching可缓解
2. ★★★★★ **Prefix caching必须显式启用** → V1 APC是optional → GRPO必须设置enable_prefix_caching=True
3. ★★★★★ **LoRA weight sync = reset all caches** → 训练后update_weights → reset_prefix_cache → 全部KV失效
4. ★★★★ **GQA是关键** → MHA 32 heads → 紧张; GQA 4 heads → 舒适 → Qwen2-7B GQA-8最佳
5. ★★★★ **INT4扩展KV budget 4-8×** → 即使MHA也可行 → INT4+INT8KV=RTX 4090最优
6. ★★★ **rollout_n=8 admission** → FCFS queue → sequential if constraints allow → chunked prefill helps
7. ★★★ **Preempted requests prepend** → 优先重新admit → 但竞争新请求 → 可能thrashing

## 参考资料

- vLLM scheduler.py: schedule(), _preempt_request(), LoRA tracking, prefix caching
- vLLM kv_cache_manager.py: get_computed_blocks(), allocate_slots(), hash-based prefix caching
- vLLM request_queue.py: SchedulingPolicy (FCFS, PRIORITY)
- verl vllm_rollout.py: weight sync + cache reset
- ★★★ SM89 compatibility: `tools/sm89_compatibility_checker.py`
- ★★★ KV cache cost: `tools/sm89_kv_cache_cost_analyzer.py`
- ★★★ Prefix hash collision: `notebook/projects/vllm-prefix-cache-hash-collision-reading.md`
