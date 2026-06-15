# verl SGLang PD Disaggregation — Prefill-Decode分离架构深度分析

> 2026-06-16 | verl PR #6117 (merged) | SGLang PD | DisaggregationConfig | NIXL/Mooncake
> ★★★★★ SGLang PD merged → 1P:ND topology → -6.8% step time on H100 → 生产可用!
> ★★★★★ vLLM PD (#6243 open) → trails colocated by +10-19% → 不成熟!
> ★★★ RTX 4090: PD需要NVLink/RDMA → 不可行 → 但SGLang deterministic inference有价值!

## 1. ★★★★★ SGLang PD Architecture — 1P:ND topology

```
★★★★★★★ SGLang PD Architecture (sglang_pd_replica.py):

  → 1 Prefill server + N Decode servers per replica → 1P:ND topology
  → Asymmetric TP: prefill_tp != decode_tp → 可配置不同并行度!
  → Per-rank role assignment: 每个Ray worker根据rollout_rank分配prefill/decode角色
  → Bootstrap coordination: Prefill server绑定bootstrap port → decode servers连接KV transfer
  → PD peer linkage: set_pd_peer → prefill连接decode peers via Ray actor handles

★★★★★★★ PD dispatch flow (async_sglang_server.py generate method):
  When self._disaggregation_role == "prefill" and decode peers exist:
    1. Mint random bootstrap_room (63-bit random integer) → KV transfer session
    2. Pick random decode peer → avoid systematic skew → heavy-tailed RL prompt lengths
    3. Two concurrent calls:
       → Local prefill → KV computation + NIXL/Mooncake push
       → Remote decode → KV reception + token generation
    4. Return only decode output → prefill only materializes KV

★★★★★★★ DisaggregationConfig (verl/workers/config/disaggregation.py):
  enabled: bool = False
  prefill_replicas: int = 1 (目前只允许1)
  decode_replicas: int = 1 (可以为N)
  decode_tensor_model_parallel_size: Optional[int] = None
  transfer_backend: str = "nixl" (allowed: nixl, mooncake, ascend, mori, fake)
  bootstrap_port: Optional[int] = None
  ib_device: Optional[str] = None
```

## 2. ★★★★★ SGLang PD Benchmarks — vs Colocated

```
★★★★★★★ SGLang PD Benchmarks (PR #6117, Qwen2.5-7B, GSM8K):

| Setup | Step wall (s) | Delta vs coloc |
|-------|--------------|----------------|
| Single-node 1x8 H100, coloc 2xTP=1 | 12.24 ±0.2 | -- |
| ★★★★★ Single-node 1x8 H100, PD 1P+3D | 11.41 ±0.2 | ★★★★★ -6.8%! |
| Multi-node 2x8 H100, coloc | 12.37 | -- |
| ★★★★★ Multi-node 2x8 H100, PD | 11.78 ±0.02 | ★★★★ -4.75%! |

★★★★★★★ Key insight: PD wins = decode-pressure dependent!
  → ~20 sequences per decode server → SGLang single-loop scheduler bottleneck
  → → PD's separated schedulers pay off → prefill/decode独立 → no contention!

★★★★★★★ vLLM PD Benchmarks (PR #6243, still open!):

| Setup | Backend | coloc (s/step) | PD (s/step) | Delta |
|-------|---------|---------------|-------------|-------|
| 4-card 0.6B | NIXL | 13.98 | 16.42 | ★★ +17.49% (worse!) |
| 8-card 7B | NIXL | 17.86 | 21.22 | ★★ +18.82% (worse!) |
| 8-card 7B | Mooncake | 18.16 | 20.13 | ★★★ +10.83% (less worse) |

★★★★★★★ SGLang PD > vLLM PD:
  → SGLang: -6.8% → faster than colocated → production-ready!
  → vLLM: +10-19% → slower than colocated → experimental!
  → → ★★★★★ SGLang PD = verl PD唯一推荐!
```

## 3. ★★★★★ Transfer Backend — NIXL vs Mooncake vs Ascend

```
★★★★★★★ 5种KV transfer backend:

| Backend | GPU | NVLink/RDMA | RTX 4090 | 备注 |
|---------|-----|-------------|----------|------|
| NIXL | NVIDIA | ✓ NVLink | ✗ 无NVLink | NVIDIA native → 最高throughput |
| Mooncake | NVIDIA | ✓ RDMA | ✗ 无RDMA | 月之暗面贡献 → ~5% slower than NIXL |
| Ascend | Huawei | ✓ HCCL | ✗ NPU-only | Ascend专用 → HCCL transfer |
| Mori | ? | ? | ✗ | 实验性 |
| Fake | CPU | N/A | ✗ | Testing only |

★★★★★★★ RTX 4090限制:
  → No NVLink → NIXL不可用 → 需要NVLink for inter-GPU KV transfer
  → No RDMA → Mooncake不可用 → 需要RDMA for remote KV transfer
  → → ★★★★★ RTX 4090 PD disaggregation = ✗✗✗ 不可行!
  → → → 与DeepEP分析一致 → RTX 4090无NVLink/RDMA → only colocated or in-process

★★★★★★★ Mooncake vs NIXL:
  → Mooncake → TCP connection → +5% overhead vs NIXL → 但更portable
  → NIXL → NVLink direct → 最高throughput → 但需要NVIDIA hardware
  → → Mooncake有EADDRNOTAVAIL bug → TCP pool disabled → exhaust 5-tuple → PR #6243 noting
```

## 4. ★★★★★ SGLang Deterministic Inference — SM89替代方案

```
★★★★★★★ SGLang deterministic inference → SM89 batch invariance替代方案!

  → --enable-deterministic-inference → batch-invariant operators → Thinking Machines Lab
  → ★★★★★ 不需要enforce_eager → 不需要禁用compile → deterministic by design!
  → sampling_seed → reproducible non-greedy sampling → GRPO critical!
  → Supported backends: FlashInfer, FA3, Triton
  → ★★★ FlashInfer不支持radix cache in deterministic mode → tradeoff!

★★★★★★★ vs vLLM SM89 batch invariance:

| 方面 | vLLM + enforce_eager | SGLang + deterministic_inference |
|------|---------------------|----------------------------------|
| Compile | ✗ 禁用 | ✓ enabled |
| CUDA graphs | ✗ 禁用 | ✓ enabled |
| Throughput | -10-15% | ★★★★★ no penalty! |
| Batch invariance | ✓ (by not compiling) | ✓ (by deterministic ops) |
| Radix cache | ✓ | ★★★ limited (FlashInfer不支持) |
| Spec decode | ✗ | ★★★★★ supported |

★★★★★★★ RTX 4090 impact:
  → SGLang deterministic inference → potential SM89 solution → no throughput loss!
  → 但verl SGLang rollout = 实验性 → 不是default → 需要更多testing
  → → ★★★★★ SGLang deterministic inference = SM89最有吸引力的替代方案!
  → → → ★★★★★ 但需要verl SGLang rollout support → not fully async mode yet (#5474)
```

## 5. ★★★★★ SGLang Server-Level RL Optimizations

```
★★★★★★★ 5大SGLang server-level RL optimization:

1. ★★★★★ Fine-Grained Engine Sleep/Wake Up:
   → POST /release_memory_occupation → tags=["kv_cache", "weights"] or ["kv_cache"]
   → POST /resume_memory_occupation → virtual memory addresses preserved
   → No disk I/O → no CUDA graph recapture → per RL step
   → ★★★★★ LoRA mode: only release KV cache → keep base weights → adapter-only sync

2. ★★★★★ Three Weight Update Strategies:
   → From disk (POST /update_weights_from_disk) → checkpoint → elastic scaling
   → From tensor (POST /update_weights_from_tensor) → colocated → in-memory → fastest
   → From distributed (POST /update_weights_from_distributed) → NCCL/IB broadcast → disaggregated

3. ★★★★★ Generation Pause/Resume:
   → POST /pause_generation → modes: abort/retract/in_place
   → POST /continue_generation → resume after weight update
   → ★★★★★ retract → move running→waiting → KV flushed/recomputed → APRIL paper pattern

4. ★★★★★ Deterministic Inference:
   → --enable-deterministic-inference → batch-invariant ops → SM89 compatible!
   → sampling_seed → reproducible → GRPO critical

5. ★★★★★ SGLang Model Gateway (Router):
   → Rust-based → high-performance → cache-aware load balancing
   → Routes requests → servers with highest prefix match → radix tree in router
   → ★★★★★ PD disaggregation support → separate prefill/decode routing
   → Circuit breaker + retries + rate limiting → production-grade
   → ★★★ NOT yet integrated into verl → issue #5674 asks for this
```

## 6. ★★★★★ verl Rollout Backend Abstraction

```
★★★★★★★ 3种rollout backend → pluggable:

| Backend | Status | PD Support | KV Cache | Weight Sync | RTX 4090 |
|---------|--------|-----------|----------|-------------|----------|
| vllm | ★★★★★ default | ★★ experimental (#6243 open) | PagedAttention | IPC ZMQ | ✓ enforce_eager |
| sglang | ★★★★ available | ★★★★★ merged (#6117) | RadixAttention | HTTP update_weights | ★★★ deterministic inference |
| trtllm | ★★★ experimental | ✗ none | TRT-LLM own | custom | ★★★ INT8 inference |

★★★★★★★ Backend切换: actor_rollout_ref.rollout.name=sglang/vllm/trtllm

★★★★★★★ Key backend differences:
  → vLLM: IPC-based ZMQ + CUDA handle → memory_saver → V1 sleep/wake_up
  → SGLang: HTTP-based update_weights_from_tensor → tag-based memory release/resume
  → → SGLang sleep_level=1 → only KV cache → keep base weights → LoRA adapter mode
  → → ★★★★★ SGLang LoRA: LoadLoRAAdapterFromTensorsReqInput → separate from base weights

★★★★★★★ SGLang clone OOM bug (#6733):
  → get_named_tensor_buckets → unconditional .clone() → doubles peak GPU memory
  → PR #6738 → skip redundant clone → fix OOM during SGLang weight sync
  → ★★★★★ RTX 4090: clone OOM → 24GB tight → MUST wait for fix!
```

## 7. ★★★★★ RTX 4090可行性评估

```
★★★★★★★ RTX 4090 PD disaggregation = ✗✗✗ 不可行!

Reasons:
  → No NVLink → NIXL不可用 → KV transfer无法跨GPU
  → No RDMA → Mooncake不可用 → remote KV transfer impossible
  → → PD需要inter-GPU KV transfer → RTX 4090单GPU → no transfer needed → no PD!

★★★★★★★ RTX 4090可行路径:

Colocated mode (vLLM):
  → enforce_eager=True → SM89 batch invariance → throughput -10-15%
  → → ★★★★★ 当前推荐 → but throughput受限

Colocated mode (SGLang):
  → --enable-deterministic-inference → SM89 batch invariance → no throughput loss!
  → → ★★★★★★ 如果SGLang deterministic inference在SM89上work → RTX 4090最大提升!
  → → → ★★★★★★ 但需要测试 → SGLang deterministic inference在SM89上 → 未验证!

In-process (rLLM Tinker):
  → 不用vLLM/SGLang → 不受SM89 bug影响 → in-process → 最简单
  → → ★★★★★★ RTX 4090 GRPO #1 → 不需要考虑PD/batch invariance!

★★★★★★★ RTX 4090 SGLang探索建议:
  → 1. 测试SGLang deterministic inference on SM89 → 是否真的batch invariant?
  → 2. 如果YES → verl + SGLang rollout + deterministic → RTX 4090 throughput +10-15%!
  → 3. ★★★★★★ 这是SGLang最大的RTX 4090潜在价值!
```

## 参考
- verl PR #6117: SGLang PD disaggregated rollout (merged)
- verl PR #6243: vLLM PD disaggregated rollout (open, trails colocated)
- sglang_pd_replica.py: SGLangPDReplica, DisaggregationConfig, 1P:ND topology
- async_sglang_server.py: SGLangHttpServer, SGLangPDReplica, generate method
- SGLang deterministic inference docs: docs/advanced_features/deterministic_inference.md
- SGLang Model Gateway: docs/advanced_features/sgl_model_gateway.md
- SGLang for RL: docs/advanced_features/sglang_for_rl.md
- vLLM #39096: SM89 batch invariance bug
- 相关笔记: verl-v080-latest-developments-2026-06-reading.md, vllm-sm89-batch-invariance-bug-reading.md, sglang-radix-attention.md
