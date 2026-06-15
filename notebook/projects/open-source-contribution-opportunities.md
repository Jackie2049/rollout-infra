# AI 开源贡献实战指南 (2026-06-15 更新)

> 基于 142 项深度阅读笔记 + RTX 4090 (SM89) 实战经验的 7 框架贡献分析
> 核心原则: 用已有知识深度撬动高价值贡献, 避免从零学习新领域

---

## 一、技能画像与框架匹配矩阵

### 1.1 核心知识域 vs 框架需求对齐

| 知识域 | 深度 | 对应框架 | 匹配度 |
|--------|------|----------|--------|
| **prefix-sharing / KV cache** | 源码级 (vLLM BlockPool/SGLang RadixAttention/verl DataProto) | vLLM, verl, SGLang | **极高** |
| **INT4 quantization + SM89** | 实战级 (Marlin/Triton fallback/FP8限制/GQA-8) | vLLM | **极高** |
| **CUDA Graph** | 源码级 (CUDAGraphDispatcher/BCG/MRV2/capture lifecycle) | vLLM, Megatron | **高** |
| **LoRA serving** | 源码级 (Punica SGMV/packed merge/prefix collision) | vLLM, verl | **高** |
| **GRPO + RL training** | 实战级 (bypass_mode/ref_in_actor/rule-based RM/outcome reward) | verl, rLLM, Megatron | **高** |
| **ZeRO-3 / FSDP2** | 源码级 (分区算法/overlap/prefetch+backpressure) | DeepSpeed, PyTorch | **中** |
| **torch.compile stack** | 全栈 (Dynamo/AOTAutograd/FX/Inductor/compile e2e) | vLLM, PyTorch | **中** |
| **MoE EP + inference** | 源码级 (DeepEP/FlexTokenDispatcher/RoutedExpertsCapturer) | vLLM, Megatron | **中** |

### 1.2 RTX 4090 特殊优势

RTX 4090 (SM89/Ada) 在开源社区是**稀缺视角**:
- 大多数 NVIDIA 工程师用 H100/H800 (SM90), consumer GPU bugs 常被忽视
- SM89 限制 (FP8 E5M2不可用/TMA不可用/NVLS不可用) 直接影响量化路径和 CUDA graph
- 单 GPU 场景 (PCIe scaling灾难) 是消费级硬件独特约束, 多数测试在大集群

**关键洞察**: SM89 bug 报告和修复是**低竞争高价值**的贡献类型 — 大厂不关注但影响大量用户

---

## 二、框架贡献生态分析

### 2.1 活跃度与贡献接受度

| 框架 | Stars | Open Issues | 周 PR 合并量 | Good First Issue 数 | 外部贡献者比例 | 适合度 |
|------|-------|-------------|-------------|--------------------|---------------|--------|
| **vLLM** | 82,873 | 5,425 | 30+ | 30+ | 高 (50%+) | **A+** |
| **verl** | 21,966 | 2,061 | 30+ | 10+ | 高 (70%+) | **A** |
| **rLLM** | 5,615 | 113 | 10+ | 无标签但全部欢迎 | 极高 (90%+) | **A+** |
| **Megatron-LM** | 16,703 | 897 | 5-10 | 无标签 | 低 (NVIDIA主导) | **B** |
| **DeepSpeed** | 42,517 | 1,299 | <5 | 无标签 | 极低 (Microsoft主导) | **C** |
| **PyTorch** | 100,760 | 18,377 | 50+ | 30+ | 中 (Meta主导核心) | **B-** |

**优先级排序**: vLLM > rLLM > verl > Megatron > PyTorch > DeepSpeed

理由:
- vLLM: 最活跃, good-first-issue最多, 与SM89/prefix/quantization完全对齐
- rLLM: 小社区但极友好, Tinker/GRPO完美匹配, 几乎所有PR都被接受
- verl: GRPO核心框架, 好问题多但需Ray集群测试
- Megatron: NVIDIA主导, 需多GPU环境, 但CUDA graph/inference方向有空间
- PyTorch: compile stack问题多但review周期长(3-6个月)
- DeepSpeed: 几乎停转, PR merge率极低, 不推荐投入时间

---

## 三、Tier 1: 最高价值贡献 (独特专业知识 + RTX 4090)

### 3.1 vLLM: SM89 FP8 KV Cache Crash 修复

| Issue | 标题 | 匹配原因 |
|-------|------|----------|
| [#44879](https://github.com/vllm-project/vllm/issues/44879) | compressed-tensors kv_cache_scheme overrides kv_cache_dtype to fp8 during MTP speculative decoding, causing CUDA crash on SM89 (L4/RTX4090) | **直接是 RTX 4090 bug**, FP8+MTP+SM89 三重交叉 |
| [#45038](https://github.com/vllm-project/vllm/issues/45038) | Guard compressed-tensors FP8 KV cache override on SM90+ | 44879的修复, 需要SM89验证 |

**为什么最高价值**:
- 这两个issue明确提到 **RTX 4090** — SM89社区的痛点
- 我们的 INT4/FP8/SM89 知识深度足以理解根因 (kv_cache_scheme无条件覆盖→FlashInfer FP8 kernel需要SM90)
- 修复方向明确: 添加 `has_device_capability(90)` guard, 回退到 bf16/fp16 KV cache
- 可测试: RTX 4090 本地可直接验证

**行动**:
1. 在 #44879 评论, 分享 SM89 FP8 限制的系统性理解 (来自 quantization-inference-training-crosspoint-reading.md)
2. 在 #45038 review, 提出SM89下的替代路径 (INT8 KV cache 而非 FP8)
3. 如果 #45038 未合并, 提出自己的SM89-safe PR

### 3.2 vLLM: LoRA + Prefix Cache Collision

| Issue | 标题 | 匹配原因 |
|-------|------|----------|
| [#44701](https://github.com/vllm-project/vllm/issues/44701) | V1 prefix-cache extra-key domain collision between LoRA name and cache_salt | **LoRA+prefix cache 安全性** — 我们 vLLM LoRA Serving 源码阅读直接覆盖 |
| [#44706](https://github.com/vllm-project/vllm/issues/44706) | fix(kv_cache): domain-tag LoRA and cache_salt prefix-cache keys to prevent collision | 44701的修复PR |

**为什么最高价值**:
- 我们的 vLLM LoRA Serving 源码阅读已详细分析了 **LoRA+prefix caching 不兼容**问题
- SGLang radix tree hash不含 LoRA ID → vLLM 同样存在 → 这是系统性架构缺陷
- domain collision: `lora_name="adapter_A"` 和 `cache_salt="adapter_A"` 产生相同 hash → 错误共享 KV
- 修复方向: 给 extra_keys 加 domain tag (如 `lora:adapter_A`, `salt:xxx`)

**行动**:
1. 在 #44701 评论, 引用 LoRA Serving 源码阅读中的 prefix collision 分析
2. 提出 domain-tag prefix 方案, 参考 SGLang radix tree 的做法
3. 为 GRPO rollout_n=8 同adapter场景 (prefix可共享) 提出专门讨论

### 3.3 vLLM: LoRA Stale Prefix-Cache Blocks

| Issue | 标题 | 匹配原因 |
|-------|------|----------|
| [#42125](https://github.com/vllm-project/vllm/issues/42125) | Runtime LoRA same-name reload can reuse stale prefix-cache blocks from previous adapter version | **LoRA热替换+prefix cache** — 生产级问题 |

**为什么高价值**:
- LoRA adapter 热替换是 multi-tenancy serving核心场景
- vLLM 的 LoRA activate/deactivate 不清理对应 prefix-cache block hash
- 同名adapter reload后, 新adapter可能复用旧adapter的KV → silent correctness bug
- 我们的 LoRA Serving 源码阅读覆盖了 GPU buffer lora_a/b_stacked 和 activate 流程

### 3.4 verl: Prefix-Tree Shared Attention for GRPO

| Issue | 标题 | 匹配原因 |
|-------|------|----------|
| [#6401](https://github.com/volcengine/verl/issues/6401) | [RFC] Prefix-Tree Shared Attention for Multi-Turn RL Training | **完美匹配 prefix-sharing 项目经验** |
| [#6689](https://github.com/volcengine/verl/pull/6689) | feat: prefix-tree MAGI attention for verl SFT and RL (已merge) | 参考实现 |

**为什么最高价值**:
- GRPO rollout_n=8 → 同prompt 8个response共享prefix → 当前独立计算→7倍浪费
- 我们的 prefix-sharing 项目直接解决这个问题
- Magi Attention 已merge (#6689), 但仅覆盖 SFT/RL forward, 不覆盖 verl 的 GRPO 完整流程
- **RTX 4090 单GPU场景**: prefix共享 = 省prefill compute = 唯一可行的scaling路径

---

## 四、Tier 2: 高价值 Good First Issues (明确范围 + 知识对齐)

### 4.1 vLLM Quantization Pipeline

| Issue | 标题 | 难度 | 知识对齐 | 评论数 | 状态 |
|-------|------|------|----------|--------|------|
| [#32268](https://github.com/vllm-project/vllm/issues/32268) | Refactor Int8ScaledMMLinearLayerConfig to use QuantKey | 低 | 量化管线 (INT4/INT8 配置) | 8 | open |
| [#37753](https://github.com/vllm-project/vllm/issues/37753) | Unify MoE "Oracles" with Class Structure | 低 | MoE router/oracle (4种oracle → class) | 8 | open |
| [#38066](https://github.com/vllm-project/vllm/issues/38066) | fix W4A8-INT activation quantization and int4 support in Marlin | 中 | W4A8量化 (INT4 activation path) | 0 | open |
| [#45306](https://github.com/vllm-project/vllm/issues/45306) | Support modelopt_mixed on Ampere (SM80/SM86) | 中 | SM89相关 (consumer GPU量化支持) | 2 | open |

**推荐**: #32268 (QuantKey refactor)
- 范围极明确: 把 boolean config fields 替换为 QuantKey objects
- 纯代码重构, 不需新kernel
- 可以学习量化配置的完整类型系统
- **后续**: 修完后可顺势处理 #38066 (W4A8-INT bug) 和 #45306 (SM80/SM86 support → 类似SM89)

### 4.2 vLLM KV Cache & Prefix

| Issue | 标题 | 难度 | 知识对齐 | 评论数 | 状态 |
|-------|------|------|----------|--------|------|
| [#32335](https://github.com/vllm-project/vllm/issues/32335) | Extract KV-Cache update from all attention backends | 中 | KV cache extraction (1/9 backend remaining) | 51 | open |
| [#33267](https://github.com/vllm-project/vllm/issues/33267) | Remove attention layer name from unified_kv_cache_update | 低 | CUDA graph + torch.compile | 10 | open |
| [#43204](https://github.com/vllm-project/vllm/issues/43204) | Simplify UnitaryKVCacheCoordinator hash_block_size assert | 极低 | KV cache coordinator cleanup | 0 | open |
| [#44237](https://github.com/vllm-project/vllm/issues/44237) | Fix linear host RSS growth with prefix caching | 中 | prefix cache内存泄漏 | 6 | open |
| [#43789](https://github.com/vllm-project/vllm/issues/43789) | Add MessagePack prefix block hash algorithms | 中 | prefix block hash性能优化 | 1 | open |

**推荐**: #32335 (KV-Cache extraction — 最后1个backend)
- 51评论说明活跃, 已完成8/9 backend (只剩 AiterFlashAttention)
- 我们的 KV cache 源码阅读覆盖了所有backend的内部结构
- 实现模式明确: 参考已完成的FlashAttention/FlashInfer/Triton extraction
- **RTX 4090 相关**: TritonAttention 已extracted, 可测试AiterFlashAttention对应

**推荐**: #33267 (remove layer name from kv_cache_update)
- 直接解决 torch.compile 分区重复编译问题
- 我们的 compile stack 知识完全覆盖此问题
- 修改模式明确: 参考 #32805 和 #33184 的做法
- CUDA graph友好 → 直接影响SM89上的cold start时间

**推荐**: #43204 (simplify hash_block_size assert)
- 极小范围, 纯cleanup PR
- 了解 KV cache coordinator 内部结构的好入口
- 零风险修改, 极高合并概率

### 4.3 vLLM Speculative Decoding + CUDA Graph

| Issue | 标题 | 难度 | 知识对齐 | 评论数 |
|-------|------|------|----------|--------|
| [#34880](https://github.com/vllm-project/vllm/issues/34880) | Enables Eagle drafter support for FULL CUDA Graph mode | 高 | EAGLE+CUDA graph (47评论, 活跃) | 47 |
| [#41732](https://github.com/vllm-project/vllm/issues/41732) | enable lora for cuda graph in MRV2 | 中 | LoRA+CUDA graph+MRV2 | 4 |
| [#41785](https://github.com/vllm-project/vllm/issues/41785) | Overlap LoRA weight H2D copies with compute via side CUDA stream | 中 | LoRA PCIe H2D overlap | 4 |
| [#44879](https://github.com/vllm-project/vllm/issues/44879) | FP8 KV crash on SM89 with MTP spec decode | 中 | SM89+spec decode | 0 |

**推荐**: #41785 (LoRA H2D overlap)
- PCIe DMA和SMs独立 → 可以overlap → RTX 4090 PCIe瓶颈下特别有价值
- 我们的 LoRA Serving 源码阅读覆盖了 activate_adapter 流程
- 275行修改, 独立特性, 可自测

### 4.4 vLLM torch.compile

| Issue | 标题 | 难度 | 知识对齐 | 评论数 |
|-------|------|------|----------|--------|
| [#39479](https://github.com/vllm-project/vllm/issues/39479) | torch.compile config hashing refactor follow-ups | 低 | compile stack 全栈理解 | 16 |
| [#29139](https://github.com/vllm-project/vllm/issues/29139) | Optimize collectives in TP MoE using torch.compile pass | 高 | MoE+compile+SP | 35 |
| [#19817](https://github.com/vllm-project/vllm/issues/19817) | CustomOp cleanup | 中 | CustomOp dispatch + compile | 7 |

**推荐**: #39479 (config hashing refactor)
- **9个具体TODO** — 可逐个完成, 每个都是独立小PR
- 我们完整阅读了 compile stack (Dynamo/AOTAutograd/FX/Inductor/compile e2e)
- `compute_hash` → `compile_factors` 重命名是纯重构
- 入门友好, 可以逐步深入 compile config 系统

### 4.5 verl GRPO + Training

| Issue | 标题 | 难度 | 知识对齐 | 评论数 |
|-------|------|------|----------|--------|
| [#907](https://github.com/volcengine/verl/issues/907) | Tests for gradient accumulation | 低 | GRPO训练数学验证 | 0 |
| [#1345](https://github.com/volcengine/verl/issues/1345) | Unify & Complete the docstring format | 低 | 了解代码结构的入口 | 2 |
| [#229](https://github.com/volcengine/verl/issues/229) | Support Generative Reward Model (GenRM) | 中 | GenRM (rule-based RM的扩展) | 17 |
| [#1014](https://github.com/volcengine/verl/issues/1014) | Documentation about Resolving OOM | 低 | 单GPU内存优化指南 | 6 |
| [#1508](https://github.com/volcengine/verl/issues/1508) | Better metric reduction utility | 中 | verl metric体系 | 3 |

**推荐**: #1014 (OOM documentation)
- **完美RTX 4090视角**: 单GPU内存优化是我们实战经验的核心
- 可以贡献 LoRA+CPU_Adam / GRPO+bypass_mode / INT4推理 等配置指南
- 纯文档贡献, 零代码风险, 高接受率
- 引用我们的 RTX 4090 RL Quick Reference 和 memory 表

**推荐**: #907 (gradient accumulation tests)
- 数学验证: 不含梯度累积 vs 含梯度累积 → loss应该相等
- 我们的 GRPO training loop 源码阅读覆盖了 mini-batch 处理
- 可以在 RTX 4090 本地测试 (单GPU足够)

### 4.6 rLLM Tinker + GRPO

| Issue | 标题 | 难度 | 知识对齐 | 评论数 |
|-------|------|------|----------|--------|
| [#489](https://github.com/rllm-org/rllm/issues/489) | replace bypass_render_with_parser with TinkerChatTemplateParser | 中 | bypass_mode + Tinker架构 | 0 |
| [#605](https://github.com/rllm-org/rllm/issues/605) | GRPO groups by trajectory.uid instead of prompt/agent key | 低 | GRPO分组逻辑 | 0 |
| [#477](https://github.com/rllm-org/rllm/issues/477) | Add strict DPO objective plumbing | 中 | DPO vs GRPO objective | 3 |
| [#475](https://github.com/rllm-org/rllm/issues/475) | Add early-finalize continuation for truncated reasoning rollouts | 中 | reasoning budget 管理 | 3 |

**推荐**: #605 (GRPO grouping logic)
- 直接影响 GRPO advantage 计算: trajectory-level vs prompt-level grouping
- 我们的 GRPO training loop 源码阅读已深入分析了 rollout_n grouping
- 小社区 → 极高PR接受率
- 可以用 RTX 4090 本地验证

---

## 五、Tier 3: 入门贡献 (快速建立贡献记录)

### 5.1 极小范围 PR (1-3天可完成)

| Issue | 标题 | 范围 | 预估时间 |
|-------|------|------|----------|
| vLLM [#43204](https://github.com/vllm-project/vllm/issues/43204) | Simplify UnitaryKVCacheCoordinator hash_block_size assert | 删除一个dead code条件 | 1小时 |
| vLLM [#44931](https://github.com/vllm-project/vllm/issues/44931) | Fix 'douby' -> 'doubly' typo in prefix caching diagram | 1个文档typo | 15分钟 |
| verl [#1345](https://github.com/volcengine/verl/issues/1345) | Unify docstring format (部分) | 统一1-2个文件的docstring | 2-3小时 |
| vLLM [#32268](https://github.com/vllm-project/vllm/issues/32268) | QuantKey refactor | 替换boolean → QuantKey | 3-4小时 |
| Megatron [#4950](https://github.com/NVIDIA/Megatron-LM/issues/4950) | Add full model cuda graph for MTP inference | 标记: complexity:low | 1-2天 |

### 5.2 文档贡献 (零代码风险)

| Issue | 标题 | 内容方向 |
|-------|------|----------|
| verl [#1014](https://github.com/volcengine/verl/issues/1014) | OOM resolution documentation | 单GPU内存优化配置指南 |
| vLLM [#41230](https://github.com/vllm-project/vllm/issues/41230) + [PR #45494](https://github.com/vllm-project/vllm/pull/45494) | NIXL KV connector metrics docs | **我们已有open PR**, 继续推进 |
| verl [#1345](https://github.com/volcengine/verl/issues/1345) | Docstring format unification | 逐文件推进 |

---

## 六、Tier 4: 战略性长期贡献 (深度知识优势)

### 6.1 vLLM Prefix-Cache-Aware Routing for DP

| Issue | 标题 | 深度 |
|-------|------|------|
| [#43483](https://github.com/vllm-project/vllm/issues/43483) | Add prefix-cache-aware routing for internal DP load balancing | 高 |
| [#45261](https://github.com/vllm-project/vllm/issues/45261) | Report prefix-cache-reused blocks in full report mode | 中 |
| [#44882](https://github.com/vllm-project/vllm/issues/44882) | Avoid duplicate KV block allocation for shared external-prefix hits | 高 |

**战略价值**: prefix-aware routing 是 vLLM PD 分离和 multi-instance 的核心需求
- #43483 已有 PR (abligail), 但merge conflict → 可协助review并贡献SM89测试
- #45261 (prefix-cache-reused blocks reporting) 是 #43483 的前置依赖
- #44882 (shared external-prefix dedup) 直接对应我们的 prefix-sharing 项目经验

### 6.2 vLLM INT4 Triton Fallback 生态建设

已merge的 #43731 开启了 Triton INT4 fallback 路径, 但生态仍不完整:

| 需要方向 | 说明 |
|----------|------|
| Triton INT4 KV cache quantization | #40835 (PR已提交但stale, 可fork并rebase) |
| Triton INT4 MoE support | 当前Marlin MoE仍需SM90门槛 → 需Triton fallback |
| SM89 benchmarking | consumer GPU INT4性能基准测试 → 公开数据稀缺 |

### 6.3 verl RTX 4090 单GPU GRPO 配置指南

| Issue | 方向 |
|-------|------|
| [#1014](https://github.com/volcengine/verl/issues/1014) | 写一份 "GRPO on Single Consumer GPU" 操作指南 |
| 无对应issue | 提新issue: "verl GRPO bypass_mode on RTX 4090 performance report" |

---

## 七、跨框架贡献桥梁 (独特视角)

### 7.1 MoE Routing Replay (Megatron → verl 桥梁)

| Issue | 标题 | 框架 |
|-------|------|------|
| [Megatron #4256](https://github.com/NVIDIA/Megatron-LM/issues/4256) | Add training-side support for Rollout Routing Replay (R3) | Megatron |

**桥接分析**:
- R3解决: 推理引擎(vLLM/SGLang)和训练引擎(Megatron) MoE路由不一致 (~10% router disagree)
- 我们的 MoE Serving Patterns 源码阅读分析了 RoutedExpertsCapturer = RL+推理桥梁
- 但 Capturer与 KV connector不兼容 → GRPO不能PD分离
- 可贡献: 在verl侧提出 "routing replay + LoRA weight sync" 的SM89配置

### 7.2 DeepSpeed LoRA + ZeRO-3

| Issue | 标题 |
|-------|------|
| [#7669](https://github.com/deepspeedai/DeepSpeed/issues/7669) | Support for modules_to_save with LoRA under ZeRO-3 |

**注意**: DeepSpeed 目前近乎停转, 此issue 0评论, 不推荐投入大量时间。但如果想了解 ZeRO-3 内部机制, 这是好学习目标。

---

## 八、已有贡献状态与跟进计划

### 8.1 已有贡献

| PR | 状态 | 下一步 |
|----|------|--------|
| vLLM [#45157](https://github.com/vllm-project/vllm/pull/45157) (NIXL KV connector metrics) | CLOSED | 已学习DCO流程 |
| vLLM [#45494](https://github.com/vllm-project/vllm/pull/45494) (NIXL metrics docstring) | OPEN (2评论) | 回复评论, 推进merge |
| vLLM fork PR #7 (Top-n-sigma logits processor) | 完成 | vectorized 10-66x性能 |

### 8.2 跟进优先级

1. **本周**: 推进 #45494 merge (回复reviewer评论)
2. **本周**: 在 #44879/#45038 评论 (SM89 FP8 KV expertise)
3. **本周**: 在 #44701 评论 (LoRA+prefix cache collision)
4. **下周**: 提交 #43204 (极小cleanup PR, 建立贡献记录)
5. **2周内**: 提交 #32268 (QuantKey refactor)
6. **持续**: #39479 的9个TODO逐个提交

---

## 九、贡献策略建议

### 9.1 渐进式贡献路径

```
Phase 1 (1-2周): 极小PR建立信任
  → #43204 (cleanup), #44931 (typo), #45494 (doc推进)

Phase 2 (2-4周): Good First Issue 扩展
  → #32268 (QuantKey), #39479 (compile hashing), #907 (verl tests)

Phase 3 (1-2月): 知识域深度贡献
  → #44701 (LoRA+prefix collision), #44879 (SM89 FP8 fix)
  → #32335 (KV cache extraction last backend)

Phase 4 (持续): 战略性长期贡献
  → #43483 (prefix-aware routing), #44882 (shared prefix dedup)
  → verl #6401 (prefix-tree GRPO), rLLM #605 (GRPO grouping)
```

### 9.2 RTX 4090 测试优势

每个PR都可以在 RTX 4090 本地验证, 这是多数外部贡献者做不到的:
- SM89-specific bugs: 只有RTX 4090/L4用户能复现和验证
- INT4+INT8KV配置: 只有消费级GPU需要此路径
- 单GPU场景: 8x4090 PCIe集群 ≠ H100 NVLink集群, 行为不同
- CUDA graph + LoRA: 小GPU内存下的capture行为与大GPU不同

### 9.3 避免的领域

| 领域 | 原因 |
|------|------|
| DeepSpeed 核心开发 | 几乎停转, PR merge率极低 |
| PyTorch core dispatcher/dynamo | review周期3-6月, 需Meta内部共识 |
| Megatron PP/TP 核心改动 | NVIDIA主导, 需多GPU环境 |
| FP8 训练/通信 kernel | SM89不支持, 无法本地验证 |
| NVLS/TMA kernel | SM89不支持, 无法测试 |
| H100/H800 特有优化 | 无法本地验证 |

---

## 十、每框架贡献接触点详解

### 10.1 vLLM (主战场)

**社区入口**:
- Slack: https://slack.vllm.ai → #pr-reviews, #feat-kv-cache, #feat-quantization, #feat-lora
- Good First Issue: 30+ 开放, 每周新增
- DCO要求: 必须git commit加 `-s` 签名

**最对齐的5个issue** (按价值排序):
1. #44879 — SM89 FP8 crash (RTX 4090直接bug, 最高优先)
2. #44701 — LoRA+prefix collision (源码级知识覆盖)
3. #33267 — Remove layer name from kv_cache_update (compile stack覆盖)
4. #32335 — KV-Cache extraction last backend (KV cache源码覆盖)
5. #39479 — compile hashing refactor (9个独立TODO)

### 10.2 rLLM (最快回报)

**社区入口**:
- GitHub: rllm-org/rllm (5.6K stars, 113 open issues)
- 几乎所有PR都被快速merge
- 创始人 kylemontgomery1 活跃review

**最对齐的3个issue**:
1. #605 — GRPO grouping逻辑 (trajectory vs prompt)
2. #489 — bypass_render_with_parser → TinkerChatTemplateParser (bypass_mode知识)
3. #477 — DPO objective plumbing (DPO vs GRPO)

**RTX 4090 完美匹配**: TinkerBackend = in-process + LoRA auto-init + zero-copy weight sync + GRPO → 单GPU最优路径

### 10.3 verl (GRPO深度)

**社区入口**:
- GitHub: volcengine/verl (21.9K stars)
- Good First Issue: 10+ (label: good first issue, call for contribution)
- 中文社区活跃, 可中文交流

**最对齐的5个issue**:
1. #1014 — OOM documentation (单GPU内存指南)
2. #907 — gradient accumulation tests (GRPO数学验证)
3. #1345 — docstring unification (了解代码结构)
4. #6401 — prefix-tree shared attention RFC (prefix-sharing专业知识)
5. #229 — GenRM support (rule-based RM扩展)

### 10.4 Megatron-LM (有限空间)

**社区入口**:
- GitHub: NVIDIA/Megatron-LM (16.7K stars)
- 无good-first-issue标签, 但有 complexity:low/medium 标签
- NVIDIA主导, 外部贡献需Design Doc

**可贡献方向** (仅SM89相关):
1. #4950 — MTP inference CUDA graph (complexity:low)
2. #5197 — free MoE activations in recompute (社区请求)
3. #4997 — Q2 roadmap评论 (提SM89需求)

### 10.5 PyTorch (compile stack)

**社区入口**:
- GitHub: pytorch/pytorch (100K stars)
- Good First Issue: 30+, 但review慢
- compile/dynamo方向: oncall:pt2

**可贡献方向** (仅compile stack):
1. #133443 — Deduplicate auto_functionalized and triton_kernel_wrapper_functional (good first issue)
2. #135859 — bmm/topk out variants recompilation (good first issue)
3. #131941 — Schematize nn_module_stack serialization (good first issue, export方向)

### 10.6 DeepSpeed (不推荐)

**现状**: Microsoft主导, 近乎停转, PR merge率极低, 大量陈年issue未处理
**唯一价值**: 如果想学习ZeRO-3内部机制, 可读issue但不要投入开发时间

---

## 十一、快速参考: 本周可执行的行动清单

| 序号 | 行动 | 时间 | 预期结果 |
|------|------|------|----------|
| 1 | 回复 vLLM #45494 reviewer评论, 推进merge | 30分钟 | PR推进 |
| 2 | 在 vLLM #44879 评论SM89 FP8限制分析 | 20分钟 | 建立SM89专家形象 |
| 3 | 在 vLLM #44701 评论LoRA+prefix collision分析 | 20分钟 | 建立prefix-sharing专家形象 |
| 4 | 提交 vLLM #43204 cleanup PR | 1小时 | 第一个cleanup merge |
| 5 | 在 verl #6401 评论prefix-tree方案 | 30分钟 | RFC讨论参与 |
| 6 | Fork vLLM #40835 (INT4 KV Triton, 当前stale) 并rebase | 2-3小时 | 活化INT4 KV路径 |

---

## 附录: Issue 紧急度评分标准

- **紧急度**: 0评论无PR → 最高机会; 多评论有活跃PR → 需竞争
- **知识匹配**: 源码级 > 实战级 > 概念级 > 无
- **RTX 4090可测性**: 本地可验证 > 需远程集群 > 无法验证
- **PR接受概率**: 小框架(rLLM) > 大框架活跃(vLLM/verl) > 大框架保守(PyTorch/Megatron) > 停转(DeepSpeed)

综合评分 = 知识匹配 x RTX4090可测性 x 接受概率 / 竞争度
