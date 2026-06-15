# vLLM V1 Scheduler Watermark Optimization 深度阅读

> 2026-06-16 | 源码: vllm/v1/core/scheduler.py + PR #44594
> 核心: watermark=0.05 → 抢占82%减少 → ITL p99 -56% → throughput +5.1%
> ★★★★★ 这是vLLM V1 scheduler最重要的生产优化之一

## 1. V1 Scheduler schedule() 两阶段设计

```
★★★★★ V1 scheduler schedule() = 两阶段:

Phase 1: schedule_running (处理正在运行的请求)
  → 计算每个running request需要的KV blocks
  → 检查可用blocks是否足够 → 不够 → 需要抢占(preempt)!
  → 抢占策略: retraction → 完全丢弃KV → ref_cnt保护shared prefix blocks

Phase 2: schedule_waiting (处理等待中的请求)
  → 从waiting queue选择请求 → 分配KV blocks → 开始运行
  → admission gate: 新请求必须有足够blocks才能开始 → 防thrashing
```

## 2. ★★★★★ Watermark PR #44594 — 核心优化

### 2.1 问题: 默认watermark=0.05太保守?

```
★★★★ PR #44594发现: 默认watermark=0.05 → 过度保守 → 太多抢占!

watermark机制:
  watermark_blocks = int(total_blocks * watermark) → 预留blocks
  可用blocks = free_blocks - watermark_blocks → "有效"可用blocks

  → 当free_blocks < watermark_blocks → 认为"空间不足" → 开始抢占running requests!
  → watermark=0.05 → 24GB RTX 4090约5000 blocks → 预留250 blocks → 过度!

★★★★★★ 核心洞察: watermark的设计初衷是防止thrashing → 但过度保守 → 反而降低throughput!
```

### 2.2 优化: watermark=0.05 → 仍然0.05 但改变了抢占判断逻辑

```
★★★★★ PR #44594的关键修改:

之前:
  preempt条件 = free_blocks < watermark_blocks → 太激进 → 小波动也触发抢占

现在 (PR #44594):
  preempt条件改为更精细的判断 → 不是简单free_blocks < watermark_blocks
  → 考虑实际需求 → 只在确实无法满足running请求时才抢占
  → 减少不必要的抢占 → 82%减少!

★★★★★★ 效果:
  - Preemptions: -82%
  - ITL (inter-token latency) p99: -56%
  - Throughput: +5.1%
  - 对RTX 4090影响: ★★★★★ 直接正面 → 更少抢占 → 更少KV重算 → 更高throughput
```

## 3. ★★★★★ 4层Thrashing预防机制

```
★★★★★ V1 scheduler的4层thrashing预防:

Layer 1: Admission Gate (准入门)
  → 新请求需要足够blocks才能开始 → 否则留在waiting queue
  → 防止: 新请求抢走running请求的blocks → thrashing

Layer 2: Watermark (水位线)
  → 预留一定比例blocks → 不让running请求用满所有空间
  → 防止: KV cache用满 → 每个新请求都触发抢占 → thrashing循环

Layer 3: Prefix Protection (前缀保护)
  → ref_cnt机制 → shared prefix blocks不被抢占
  → 防止: 抢占shared prefix → 所有依赖它的请求都受影响 → 连锁抢占

Layer 4: Self-Detection (自检测)
  → 检测请求是否在"thrashing状态" → 反复被抢占又重新调度
  → 防止: 单个请求反复被抢占 → 浪费compute → 延迟飙升
```

## 4. Preemption = Retraction (完全KV丢弃)

```
★★★★ V1 preemption = retraction → 完全丢弃KV → 不是swap-out!

为什么选择retraction而非swap-out:
  1. Swap-out需要CPU内存 → RTX 4090只有24GB GPU → CPU swap延迟高
  2. Retraction = 直接丢弃 → 下次重新prefill → 更简单更可靠
  3. Prefix blocks有ref_cnt保护 → 共享前缀不会被丢弃 → 只丢弃unique部分

★★★★ RTX 4090影响:
  → Retraction = KV重算 → 计算开销 → 但比thrashing更好!
  → Watermark优化 → 减少retraction → 减少KV重算 → throughput ↑
  → INT8 KV → KV blocks更小 → 可用blocks更多 → 抢占更少 → 更好!
```

## 5. Prefix Caching Hash Collision

```
★★★★ Prefix cache hash collision → Issue #44701:

BlockHashWithGroupId = hash(parent_block_hash + current_token_ids + extra_keys)

extra_keys = (*lora_keys, cache_salt)
  → LoRA name 和 cache_salt 都是bare strings → 无domain separation → collision!

★★★★ Collision条件:
  LoRA name = cache_salt → extra_keys完全相同 → 但语义不同 → silent KV corruption

★★★★ Fix: PR #44706 → domain-tag → ("lora", name) + ("salt", value) → 数学保证安全

详见: vllm-prefix-cache-hash-collision-reading.md
```

## 6. ★★★★ RTX 4090 Scheduler优化建议

### 6.1 Watermark配置

```
★★★★ RTX 4090 scheduler配置建议:

INT4 + INT8 KV (最常用路径):
  → ~5000 KV blocks可用
  → watermark=0.05 → 预留250 blocks → 可能过度
  → PR #44594优化后 → 250 blocks预留更合理 → 减少不必要抢占

★★★★★ 推荐配置:
  --block-size 16 (FlashInfer默认)
  --kv-cache-dtype int8_per_token_head
  --gpu-memory-utilization 0.90
  --enable-prefix-caching
  --enforce-eager (for spec decode / GRPO on SM89)

  # 不需要手动调整watermark → PR #44594已优化默认值
```

### 6.2 对GRPO训练的影响

```
★★★★★ GRPO on RTX 4090 → scheduler优化直接影响throughput:

GRPO rollout_n=8 → 8个并发请求 → KV blocks压力大
  → 没有watermark优化 → 可能频繁抢占 → rollout延迟飙升
  → watermark优化 → 抢占减少82% → rollout更稳定 → throughput +5.1%

★★★★★ 组合优化:
  INT4 weights (3.5GB) + INT8 KV + prefix caching + watermark优化
  → 最优RTX 4090 GRPO serving配置!
```

## 7. 关键洞察总结

1. ★★★★★ **Watermark PR #44594**: 抢占-82%, ITL p99 -56%, throughput +5.1% → RTX 4090直接正面
2. ★★★★★ **4层Thrashing预防**: admission gate + watermark + prefix protection + self-detection → 全面保护
3. ★★★★ **Preemption = Retraction**: 完全KV丢弃 → 但prefix blocks有ref_cnt保护 → shared prefix不被丢
4. ★★★★ **INT8 KV + Watermark**: INT8 KV blocks更小 → 可用blocks更多 → 抢占更少 → 更好!
5. ★★★★★ **RTX 4090 GRPO最优**: INT4 + INT8 KV + prefix caching + watermark优化 → 最稳定throughput

## 参考

- PR #44594: https://github.com/vllm-project/vllm/pull/44594
- Issue #44701: Prefix cache hash collision
- vLLM V1 Scheduler: `vllm/v1/core/scheduler.py`
- 相关笔记: vllm-v1-scheduler-reading.md, vllm-v1-preemption-source-reading.md, vllm-prefix-cache-hash-collision-reading.md
- 工具: sm89_batch_invariance_diagnostic.py, sm89_compatibility_checker.py
