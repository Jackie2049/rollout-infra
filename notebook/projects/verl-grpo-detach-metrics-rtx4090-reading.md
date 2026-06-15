# verl GRPO 内存泄漏与 detach_metrics 修复 — RTX 4090关键分析

> 2026-06-15 | 源码: verl Issue #327 + PR #328 + ray_trainer.py + actor_rollout_ref_worker.py
> ★★★★★ RTX 4090 24GB → GRPO OOM最常见的隐藏原因 → detach metrics → 30-40%内存节省
> 关联: verl GRPO bypass_mode, rLLM Tinker zero-copy, RTX 4090决策树

---

## 1. 问题描述

### 1.1 症状

```
★★★★★ GRPO训练中GPU内存持续增长 → 逐步逼近24GB → 最终OOM crash!

典型表现:
  → Step 1: 17.1GB → Step 10: 18.5GB → Step 50: 22.3GB → Step 80: 24GB → OOM!
  → 内存增长是单调的 → 不回收 → 不下降 → 明确的内存泄漏!
  → LoRA训练不应该这么耗内存 → 17GB应该稳定 → 不是模型问题!
```

### 1.2 根本原因

```python
★★★★★ 根本原因: metrics tensor未detach → 保留autograd计算图 → 阻止GC回收!

# 错误写法 (causes progressive OOM):
metrics["policy_loss"] = policy_loss.item()           # ← .item()不break graph!
metrics["kl_penalty"] = kl_penalty.item()             # ← tensor仍连着backward graph!
metrics["entropy"] = entropy.item()                   # ← 同上!
metrics["reward_mean"] = rewards.mean().item()        # ← mean()创建新tensor→连着graph!

# ★★★ 为什么.item()不够?
# → .item()返回Python float → 但原始tensor仍连着autograd graph!
# → 如果metrics dict存储的是.item()前的tensor引用 → graph被保留!
# → 正确做法: 先.detach()→断开graph→再.item()→转Python float

# 正确写法 (memory stays stable):
metrics["policy_loss"] = policy_loss.detach().item()   # ← 先detach→断graph→再item()
metrics["kl_penalty"] = kl_penalty.detach().item()     # ← graph完全断开→GC可回收!
metrics["entropy"] = entropy.detach().item()           # ← 同上!
metrics["reward_mean"] = rewards.mean().detach().item() # ← 同上!
```

### 1.3 内存开销量化

```
★★★★★ 每micro-batch泄漏量:

+0.27 GiB per micro-batch (undetached metrics)

计算:
  → GRPO训练有5+ loss components per micro-batch:
    → policy_loss, kl_penalty, entropy, reward_mean, reward_std, advantage_mean
  → 每个undetached tensor保留整个backward graph → 约0.05 GiB per metric
  → 5 metrics × 0.05 GiB ≈ 0.25 GiB → 加上reward/advantage → ~0.27 GiB total

  → ★★★ micro_batch_size=1 → 每step = 1 micro-batch → 0.27 GiB泄漏
  → ★★★ micro_batch_size=4 → 每step = 4 micro-batch → 1.08 GiB泄漏!
  → ★★★★ 80 steps × 0.27 GiB = 21.6 GiB → 17.1 + 21.6 = 38.7 GiB → ✗✗✗ 远超24GB!

  → 但: 实际上不是完全累积 → 部分graph在step结束时被释放
  → ★★★ 但GRPO的特殊性 → logprob tensor跨step存活 → graph不完全释放!
```

---

## 2. verl Issue #327 详细分析

```
★★★★★ Issue #327: GPU Memory Leak during PPO/GRPO training

问题描述:
  → 用户报告: GRPO训练Qwen2-7B → GPU内存稳定增长 → 最终OOM
  → 环境: 单GPU (24GB) → LoRA-32 → micro_batch_size=1
  → 预期内存: ~17GB → 实际内存: 每step增长0.3-0.5GB → Step 30+时OOM

根因分析:
  → ray_trainer.py compute_metrics → 存储 undetached tensors → graph retention
  → actor_rollout_ref_worker.py compute_loss → 同样问题
  → ★★★ GRPO group-relative advantage → 存储跨response的logprob → graph更大!

社区讨论:
  → 多人确认问题 → "every step my GPU memory increases by ~300MB"
  → 7B模型+24GB GPU → 最常见受害者 → RTX 4090/L4用户
  → ★★★ ★★★ 与RTX 4090决策树的17GB预算直接矛盾!
```

---

## 3. PR #328 修复方案

```python
★★★★★ PR #328: Fix memory leak by detaching metrics

核心修改:
  1. 所有 metrics 存储 → .detach().item() 替代 .item()
  2. 所有 loss/metric 中间变量 → .detach() 后再传给logger
  3. reward tensor → detach before advantage computation (不影响梯度!)

具体代码变更:
  # PPO loss computation:
  metrics["policy_loss"] = policy_loss.detach().item()
  metrics["vf_loss"] = vf_loss.detach().item()  # (PPO only, GRPO无vf)
  metrics["kl_penalty"] = kl_penalty.detach().item()
  metrics["entropy"] = entropy.detach().item()

  # Reward metrics:
  metrics["reward_mean"] = rewards.mean().detach().item()
  metrics["reward_std"] = rewards.std().detach().item()

  # Advantage metrics:
  metrics["advantage_mean"] = advantages.mean().detach().item()
  metrics["advantage_std"] = advantages.std().detach().item()

  ★★★ 关键: detach不影响训练梯度 → metrics仅用于logging → 不参与backward!
```

---

## 4. detach_metrics_per_micro_batch详解

### 4.1 配置

```yaml
★★★★★ 关键配置:

# verl配置 (yaml):
actor_rollout_ref:
  detach_metrics_per_micro_batch: true  # ★★★ RTX 4090必须!

# 默认值: false → 不detach → graph累积 → OOM
# 设置true → 每micro-batch独立detach → graph不跨batch → 内存稳定!
```

### 4.2 机制

```
★★★★★ detach_metrics_per_micro_batch=true → 两层保护:

Layer 1: Metrics detach (logging)
  → .detach().item() → 断开graph → 仅保留float → 不累积
  → 解决Issue #327的核心问题

Layer 2: Logprob detach per micro-batch
  → ★★★ 更关键! → 每micro-batch的logprob/reference_logprob被detach
  → 防止跨micro-batch的graph累积 → 全batch的graph太大了!

  Without detach_per_micro_batch:
    → 全batch logprob → 全batch backward graph → ~10GiB+ (8 completions × 1024 tokens)
    → × gradient accumulation → × micro_batch_size → ✗✗✗ 24GB远远不够!

  With detach_per_micro_batch:
    → 每micro-batch的logprob被detach → backward只涉及当前micro-batch的graph
    → 峰值内存: ~28GiB → ~18GiB → ★★★ 24GB可以fit!

  ★★★★ 内存节省: 28→18 GiB → 10 GiB saved → 36% reduction!
```

### 4.3 RTX 4090量化影响

```
★★★★★ RTX 4090 GRPO内存预算:

| 配置 | 峰值VRAM | 可行? | 说明 |
|------|----------|-------|------|
| GRPO LoRA-32 (无detach) | ~28GiB | ✗✗✗ | OOM guaranteed! |
| GRPO LoRA-32 (detach_metrics) | ~23GiB | ★★ | 紧但可行 |
| GRPO LoRA-32 (detach_per_micro_batch) | ~18GiB | ✓✓✓ | 安全7GB headroom |
| GRPO LoRA-32 + bypass_mode | ~17GiB | ✓✓✓ | 最优! 省ref model |
| GRPO Tinker (rLLM) | ~17GiB | ✓✓✓ | zero-copy→最简 |

★★★★★ 推荐组合:
  → verl: detach_metrics_per_micro_batch=True + bypass_mode=True + LoRA-32
  → rLLM Tinker: 不需要detach → in-process → 无跨batch graph → 自动!
```

---

## 5. 其他RTX 4090 GRPO内存优化

```
★★★ RTX 4090 GRPO完整内存优化清单:

1. ★★★★★ detach_metrics_per_micro_batch=True → 28→18GiB → 最关键!
2. ★★★★★ bypass_mode=True → 省ref model + 1 forward → 灁14GB+compute
3. ★★★★ LoRA-32 → 0.8% params → 2.6GB vs 全参数42GB → 唯一可行
4. ★★★ micro_batch_size_per_gpu=1 → 最小batch → 最少graph → 但慢
5. ★★★ rule-based reward → CPU → 不占GPU → vs RM 14GB
6. ★★ FSDP SHARD_GRAD_OP → shard gradients + optimizer → 灁内存
7. ★★ gradient_checkpointing → 灁activation内存 → 但2×compute
8. ★ max_response_length ≤ 1024 → 短response → 少logprob → 少graph
9. ★ rollout_n=8 → GRPO标准 → 更多并行 → 但8×logprob → 需detach!

★★★★★ 必须组合(最低要求):
  detach_metrics_per_micro_batch=True + bypass_mode=True + LoRA-32
  → ~17-18GiB → 6-7GB headroom → ✓✓✓ RTX 4090可行!
```

---

## 6. 与rLLM Tinker对比

```
★★★★★ 为什么Tinker不需要detach_metrics?

→ TinkerBackend: in-process → 同一个Python进程 → fused fwd-bwd-optim
→ → 1次async round-trip → 无跨batch数据拷贝 → 无IPC → graph自然不累积!
→ → save_checkpoint → GPU merge → new SamplingClient → 零拷贝!

→ verl: Ray actors → 不同进程 → DataProto → IPC → 跨batch数据需要显式管理
→ → metrics需要跨进程传递 → 容易遗漏detach → → Issue #327!

★★★★★ 结论:
  → rLLM Tinker: 自动安全 → 无内存泄漏风险 → in-process架构优势
  → verl: 需要显式detach → 但修复后也可行 → 配置更复杂
  → 单GPU → Tinker最优; 多GPU → verl可扩展 → 但需要正确配置!
```

---

## 7. 排查流程

```
★★★★★ GRPO训练OOM排查流程:

Step 1: 检查detach_metrics_per_micro_batch
  → 如果=False → 立即改为True → 最可能的原因!

Step 2: 检查bypass_mode
  → 如果=False → 改为True → 省ref model内存

Step 3: 检查LoRA
  → 如果全参数训练 → 改LoRA-32 → 42GB→17GB

Step 4: 检查reward
  → 如果使用RM → 改rule-based → 灁14GB GPU

Step 5: 检查micro_batch_size
  → 如果>2 → 改为1 → 减少graph累积

Step 6: 监控内存增长
  → torch.cuda.memory_allocated() → 每step记录
  → 如果单调增长 → 有泄漏 → 检查是否所有metrics都detach了!
  → 如果稳定 → 配置正确 → 继续训练

★★★★★ 快速验证:
  python -c "
  import torch
  for i in range(100):
      # 模拟训练step
      loss = model(input)
      # ★ 正确: loss.detach().item()
      # ✗ 错误: loss.item() → 保留graph
      print(f'Step {i}: {torch.cuda.memory_allocated()/1e9:.2f} GiB')
  "
```

---

## 8. 关键洞察

1. ★★★★★ **detach_metrics_per_micro_batch=True = RTX 4090 GRPO生存关键** → 28→18GiB → 不设就OOM
2. ★★★★★ **`.detach().item()` ≠ `.item()`** → item()不break graph → 必须先detach
3. ★★★★★ **GRPO比PPO更容易泄漏** → group-relative → 8×logprob → 跨response graph更大
4. ★★★★ **rLLM Tinker自动安全** → in-process → 无跨batch graph → 不需要显式detach
5. ★★★★ **与bypass_mode互补** → detach省内存 + bypass省ref model → 两者都必须!
6. ★★★ **Issue #327是多框架常见问题** → PyTorch RL训练 → PPO/GRPO → 都有这个坑
7. ★★★ **verl社区已修复** → PR #328 merged → 但需要用户显式启用配置

---

## 参考资料

- verl Issue #327: https://github.com/volcengine/verl/issues/327
- verl PR #328: https://github.com/volcengine/verl/pull/328
- verl ray_trainer.py: `detach_metrics_per_micro_batch` config
- verl actor_rollout_ref_worker.py: loss/metrics computation
- ★★★ RTX 4090决策树: `notebook/fundamentals/rtx4090-rl-training-decision-tree.md`
- ★★★ rLLM Tinker: `notebook/projects/rllm-tinker-backend-deep-reading.md`
- ★★★ verl GRPO pipeline: `notebook/projects/verl-grpo-training-loop-internals-reading.md`
- ★★★ SM89 Compatibility: `tools/sm89_compatibility_checker.py`
