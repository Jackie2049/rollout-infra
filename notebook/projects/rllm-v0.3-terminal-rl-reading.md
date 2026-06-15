# rLLM v0.3 Terminal-RL + AgentFlow Redesign 深度阅读

> 2026-06-15 | 源码: rllm-org/rLLM, PR #632/#635/#636/#637/#638/#641/#644/#647/#649/#650
> 核心: Terminus-2 agent harness / AgentFlow 3-phase redesign(engine owns sandbox) / snapshot+warm-pool acceleration / pass@k evaluation
> ★ ★ rLLM从纯文本RL→沙盒化agent RL→Terminal-RL→全新的RL范式!

## 1. Terminal-RL: Terminus-2 Harness — PR #647

```
★ ★ ★ Terminus-2 = Harbor的agent evaluation harness → rLLM沙盒集成

Terminus-2:
  → 在rLLM sandbox中运行Harbor的Terminus-2
  → agent harness → 完整的agent评估环境
  → ★ 沙盒化 → 安全隔离 → agent可执行代码/访问文件/运行命令

意义:
  → 之前rLLM主要做text RL(数学/代码/对话)
  → Terminal-RL → agent在真实terminal环境中完成任务 → 更复杂的RL!
  → ★ 这是RL训练范式的重大扩展 → 从文本生成→terminal交互→multi-step agent!

★ RTX 4090: 沙盒化推理 → 同样适用 → 但需要更多compute → multi-step agent rollout更长
```

## 2. AgentFlow 3-Phase Redesign — PR #636/#637/#638

```
★ ★ ★ AgentFlow = agent执行流管理 → 3-phase redesign

Phase 0 (PR #636): One env-requirement join + always-hooks engine
  → ★ 旧设计: needs_sandbox判断 → 3个地方3种逻辑 → 不一致!
  → 新设计: 每个component声明needs_env → one pure function joins declarations
  → needs_env = flow.needs_env ∨ env-style verifier(sandbox-sh)
  → ★ ★ always-hooks engine → hooks总是执行 → 不需要条件判断 → 简化!

Phase 1 (PR #637): CLI installs baked into snapshots
  → 沙盒snapshot → 包含所有CLI工具 → 无需动态安装 → 极快启动!
  → ★ warm-pool → 预构建snapshot → training/eval直接pop → 无冷启动
  → vs vLLM sleep/wake → 类似理念 → 预分配资源 → 快速恢复

Phase 2 (PR #638): Engine owns sandbox; flows become plain programs
  → ★ ★ ★ 核心变化: engine管理sandbox → flow只写逻辑 → 不管理资源!
  → 之前: flow→创建sandbox→运行→清理 → flow拥有sandbox
  → 现在: engine→创建sandbox→flow运行→engine清理 → ★ engine拥有生命周期!
  → ★ ★ 这是类似vLLM scheduler的变化 → centralized lifecycle management!

★ 与rLLM TinkerBackend的关系:
  → TinkerBackend = in-process → 无sandbox → 单GPU最快
  → AgentFlow = sandboxed → engine管理 → 更安全但稍慢
  → ★ 两者可以共存 → Tinker for pure text RL / AgentFlow for terminal RL
```

## 3. Snapshot + Warm-Pool Acceleration — PR #632/#635

```
★ ★ ★ Snapshot + Warm-Pool = rLLM训练加速核心!

Snapshot (PR #632):
  → 预构建沙盒snapshot → 包含CLI工具+环境+dataset
  → 训练时直接从snapshot启动 → 无冷启动延迟
  → ★ 类似vLLM CUDA graph warmup → 预计算+预分配 → 极快replay!

Warm-Pool (PR #632):
  → 预热的sandbox队列 → pop直接使用 → 无创建延迟
  → liveness-checked → pop返回活跃sandbox → 无dead handoff
  → ★ 类似vLLM persistent GPU buffers → 预分配 → 固定地址 → 快速恢复

Train on separate benchmarks (PR #635):
  → rllm train → separate train/val benchmarks
  → ★ 训练和评估用不同benchmark → 更好泛化 → 防止overfitting!
  → 训练sandbox → warm-pool → 评估sandbox → warm-pool → 两者独立

★ RTX 4090影响:
  → snapshot构建 → CPU+网络 → 不需要GPU → 可在本地构建!
  → warm-pool → 沙盒运行 → CPU → 不需要GPU → 可在本地运行!
  → ★ ★ RL训练 → 需要GPU → TinkerBackend → 7B INT4 → 可行!
  → 全流程: 本地warm-pool+snapshot → GPU rollout+训练 → TinkerBackend → 单GPU最优!
```

## 4. pass@k Evaluation — PR #649

```
★ ★ ★ pass@k = unbiased estimator → 多次尝试→pass率!

Implementation:
  → rllm eval --attempts N → 每个task→N rollout → unbiased pass@k estimate
  → 公式: pass@k = 1 - C(c,n) / C(c+k,n+k) → c=correct, n=total, k=attempts
  → ★ 这是Kolmogorov的unbiased estimator → 数学保证!

Modal fixes surfaced by running TB2 89×2 E2E:
  → chunked base64 uploads → Modal argv 64 KiB cap
  → RLLM_MODAL_SANDBOX_TIMEOUT_S → 配置超时
  → right-sized snapshot-build sandbox lifetime → 不浪费资源

★ 与GRPO的关系:
  → pass@k → 多次尝试 → best-of-k → 推理scaling!
  → GRPO rollout_n → 同prompt多response → 等价于pass@n!
  → ★ ★ pass@k evaluation和GRPO训练是同一系统的两面!
  → rLLM eval --attempts → 和 rllm train rollout_n → 完美对齐!
```

## 5. rllm-swesmith Integration — PR #641

```
★ ★ ★ rllm-swesmith = SWE-bench微训练数据集集成!

Loader/eval/oracle:
  → honor task and dataset declarations for harbor-style dirs
  → Dockerfile FROM → docker_image → 自动解析!
  → WORKDIR → workdir → 自动映射!
  → dataset.toml default_sandbox → 自动配置!
  → oracle solve.sh → 从declared workdir运行 → 标准化!

★ 意义:
  → SWE-bench → 代码修复 → 真实任务 → agent RL最佳benchmark
  → rllm-swesmith → 微训练集 → 快速迭代 → 验证agent RL方法
  → ★ 这是从math/code RL → software engineering RL → 更实用!

★ RTX 4090:
  → Qwen3-8B sandboxed training → rllm-swesmith → 单GPU可行
  → ★ ★ agent RL + math RL → 两条路径 → 都用TinkerBackend → 最优!
```

## 6. 关键洞察

```
★ ★ ★ 6个关键洞察:

1. AgentFlow = centralized lifecycle management:
   → engine owns sandbox → flow only writes logic → 分离关注点
   → ★ ★ 类似vLLM scheduler管理KV cache → model runner只做forward → 分离!
   → 这是AI infra系统设计的共同pattern → centralized resource management!

2. Terminal-RL = RL训练范式的重大扩展:
   → 从text generation RL → terminal interaction RL → multi-step agent RL
   → ★ ★ 这是2025-2026趋势 → agent RL → inference-time compute → 多步推理
   → rLLM走在最前面 → Terminus-2 harness → 完整沙盒化agent评估

3. pass@k和GRPO rollout_n是同一系统两面:
   → eval: pass@k → unbiased estimator → 多次尝试→pass率
   → train: rollout_n → 多response → group-relative → best-of-N
   → ★ ★ ★ 评估和训练完美对齐 → rLLM设计的一致性!

4. Snapshot+warm-pool = RL训练加速关键:
   → 预构建snapshot → 无冷启动 → warm-pool pop → 无创建延迟
   → ★ 类似CUDA graph warmup → 预计算+预分配 → amortize overhead
   → RL训练每步都需要sandbox → warm-pool → 30-60s→<1s → 极大加速!

5. rLLM 0.3 = 从研究框架→生产框架:
   → 之前: 纯text RL → math/code → 研究导向
   → 现在: Terminal-RL + sandboxed training + warm-pool → 生产级!
   → ★ ★ rLLM正在成为agent RL的go-to framework → RTX 4090最优路径!

6. RTX 4090最优agent RL路径:
   → rllm train → TinkerBackend → GRPO+LoRA-32+bypass_mode → 单GPU最快
   → rllm eval → pass@k → warm-pool → 本地CPU → 不需要GPU
   → ★ ★ 训练用GPU+评估用CPU → 最大化GPU利用率 → 最优路径!
```

## 参考资料

- PR #650: Terminal-RL: Terminus-2 harness, AgentFlow redesign, warm-pool training (9 commits, 71 files, +3673/-1413)
- PR #636: AgentFlow phase 0 — one env-requirement join + always-hooks engine
- PR #637: AgentFlow phase 1 — CLI installs baked into snapshots
- PR #638: AgentFlow phase 2 — engine owns sandbox; flows become plain programs
- PR #632: Snapshot + warm-pool acceleration for rllm train
- PR #635: Train sandboxed agents on separate local train/val benchmarks
- PR #641: rllm-swesmith integration
- PR #644: Warm-queue / sandboxed-training hardening
- PR #647: Terminus-2 agent harness
- PR #649: pass@k via --attempts
- 相关: [rLLM Architecture](rllm-architecture-reading.md), [TinkerBackend Deep Reading](rllm-tinker-backend-deep-reading.md)
- Terminus-2: Harbor's agent evaluation harness

---

## v0.3.0-pre Release Update (2026-04-30)

★★★★ rLLM v0.3.0-pre release新增功能:
- ★★★★★ **Fully Async Trainer** (#394): 完全异步训练 → rollout和training完全解耦 → 类似verl的async模式
- ★★★★ **On Policy Distillation** (#356): 在策略蒸馏 → 新的训练范式
- ★★★ **Geo3K Tinker VLM** (#357): 视觉语言模型Tinker训练 → 多模态RL
- ★★★ **Custom Metric for SDK** (#362): SDK自定义metric → 更灵活的reward
- ★★★ **Verifiers + Prime Intellect Integration** (#367): 环境验证框架 → agent RL
- ★★★★ **Advantage Computation Fix** (#375): Tinker advantage计算bug修复 → 正确的GAE
- ★★★ **Tinker Rollout Prompt IDs Fix** (#390): rollout prompt ids修复 → 正确的sequence tracking
- ★★★ **Unified Trainer Enhancements** (#383/#385): 统一训练器进一步改进

★★★ 与verl对比: rLLM v0.3.0-pre的Fully Async Trainer → 类似verl的AsyncRolloutManager → 但rLLM是in-process → 零拷贝更高效
