# prefix-0501 项目深度分析

> 2026-06-07 | 从rollout-infra学习视角分析prefix-0501项目架构与关键挑战

## 一、项目定位

**prefix-0501** 是一个面向 **Agentic RL训练** 的前缀共享(prefix sharing)系统，目标是消除micro-batch内多条轨迹间公共前缀的冗余计算。

**关键区别**: Prefix Sharing ≠ Prefix Caching
- Prefix Caching (vLLM/SGLang): **推理时**，单条轨迹内的时序KV复用 → 请求间独立
- Prefix Sharing (本项目): **训练时**，多条轨迹间的空间KV复用 → 共同前缀的完整计算结果可共享

**场景**: GRPO n_samples, step模式多步展开, tree模式多分支 → 同prompt多条response → 公共prefix的KV/activation只需计算1次

## 二、架构设计 (4层)

```
dependency/ (verl/Megatron源码快照, 最小侵入修改)
  ├── try/except可选导入 → prefix-sharing不可用时自动回退
integrations/ (框架适配层, thin)
  ├── verl_mcore.py: batch准备+logprob恢复
  ├── context.py: ContextVar线程安全运行时上下文
  ├── megatron_runtime.py: attention拦截+RoPE+KV注入
  ├── megatron_attention.py: patch安装/卸载
  ├── verl_attention.py: Qwen3.6新架构patch
core/ (框架无关核心语义层)
  ├── config.py: Phase-1约束验证(PP=1/CP=1/no rope fusion)
  ├── planner.py: PrefixSharingPlan执行计划
  ├── prefix_detector.py: TriePrefixDetector前缀检测
  ├── model_spec.py: ModelSpec模型架构规格
  ├── batch_trim.py: batch裁剪工具
  ├── logprob.py: Prefix-Last Restore logprob恢复
backends/ (硬件执行层)
  ├── torch_ref.py: PyTorch参考实现(autograd完整)
  ├── packed_layout.py: PackedBatchLayout THD格式
  ├── flash_atten_gpu/npu: CUDA/CANN优化后端
  ├── factory.py: 后端工厂(按config选择)
```

**设计原则**: core→backends→integrations分层, 框架无关核心语义, thin适配层

## 三、核心数据流

```
输入micro-batch [B, L]
  → TriePrefixDetector.detect(): 找出provider/reuser复用关系
  → PrefixSharingPlanner.plan(): 计算裁剪长度+cu_seqlens+restore位置
  → trim_inputs(): 移除reuser的prefix tokens, 只保留suffix
  → prefix_sharing_runtime_context(): 激活ContextVar运行时上下文
  → model forward (Megatron SelfAttention拦截)
    → maybe_run_prefix_sharing_attention():
      → RoPE(用packed_position_ids保持绝对位置)
      → backend.build_kv(): provider存KV, reuser加载并拼接provider prefix KV
      → backend.attention(): flash-attn风格计算(packed THD格式)
  → restore_suffix_first_log_probs_from_prefix(): prefix-last位置恢复logprob
```

**核心创新**: One-Forward + KV Injection + Prefix-Last Restore
1. Reuser只forward suffix部分(省计算)
2. Provider的prefix KV注入到reuser的KV路径(保持attention语义)
3. Prefix-last位置独立恢复logprob(确保训练精度等价)
4. 全程不detach()(保持autograd梯度回传)

## 四、Qwen3.6-27B HybridAttention挑战

### 4.1 模型架构
- 64层: 16层full_attention (每4层1个) + 48层GatedDeltaNet
- GQA 24:4 (24 query heads, 4 KV heads, head_dim=256)
- Partial RoPE: 只对前64/256维应用(25%)
- Output gate: attn_output × sigmoid(gate) BEFORE o_proj
- q_norm/k_norm: per-head RMSNorm (每head独立归一化)

### 4.2 约束
- **TP≤4**: GQA 4 KV heads → max TP=4 (每GPU 1 KV head)
- **PP=1**: Phase-1不支持pipeline parallel
- **CP=1**: Phase-1不支持context parallel
- **27B bf16**: ~54GB weights, TP=4每GPU ~7GB + activations需gradient checkpointing

### 4.3 DeltaNet层无KV Cache
- 48层DeltaNet使用recurrent state(conv1d + chunk_gated_delta_rule)
- prefix-sharing对这些层只能复用recurrent/conv state, 不复用KV
- 全注意力层(16层=25%)是prefix-sharing的主要收益来源
- **PrefixDeltanetStore**: 专门存储DeltaNet activation state

### 4.4 权重适配修复链
- apex fallback (RMSNorm)
- head_dim=256 (不是hidden_size/num_heads=213.33)
- q_proj gather_output (fused query+gate, output=12288)
- conv1d dtype (bf16 input vs float32 weight)
- layer_type路由修复 ((layer_idx+1)%interval==0 而非 layer_idx%4==0)

## 五、与rollout-infra学习成果的映射

| 学习成果 | prefix-0501关联 |
|---------|----------------|
| **GQA KV Cache BW实测** | GQA 24:4 → KV维度小4x → prefix KV注入BW更低 |
| **Prefix Caching线性关系** | time_savings≈compute_savings → prefix-sharing的理论基础 |
| **L2 cache超高BW** | 小prefix KV → L2命中 → 注入开销极低 |
| **Block alignment浪费** | prefix-sharing是token级(无block对齐限制) |
| **verl PrefixGrouper分析** | flat结构 → prefix-0501的tree/packed更灵活 |
| **Magi Attention** | 同为prefix tree → 但prefix-0501面向训练(保留autograd) |
| **FlashAttention decode更慢** | RTX 4090 Q=1时FlashAttn无IO收益 → prefix-sharing用packed THD可能不同 |
| **CUDA Graph RTX 4090** | launch 8us → Megatron forward中每层8us → PS减少层数也减少launch |

## 六、待做事项 (优先级排序)

### P0: E2E PS精度对齐测试
- 在RTX 4090 TP=4上测试: prefix-sharing开启前后logits/logprobs差异
- 数学等价性验证: reuser suffix logprobs ≈ 独立forward logprobs
- 这是整个项目可信度的基石

### P1: GRPO训练端到端
- verl GRPO pipeline + prefix-sharing → 实际训练吞吐测量
- n_samples=4/8 prefix-sharing savings实测
- 与RTX 4090 benchmark理论值对比

### P2: DeltaNet state PS
- PrefixDeltanetStore实现 → 48层activation复用
- 需要理解GatedDeltaNet内部state结构(conv state + recurrent state)
- 参考实现: `torch_chunk_gated_delta_rule`

### P3: verl #6401贡献
- prefix-0501的核心代码可以作为verl prefix-tree shared attention的参考
- Phase 1: FSDP兼容fix → prefix-0501已有Megatron integration
- Phase 2: tree utilities → prefix-0501已有TriePrefixDetector

## 七、务实推进路线 (Option B + 当前进展)

当前已完成Option C(Qwen3.5→text-only适配), 最务实推进:

1. **立即**: 在GPU服务器运行E2E精度测试脚本
2. **本周**: 准备GRPO训练配置(YAML)
3. **下周**: 运行GRPO+PS训练, 测量实际savings
4. **长期**: DeltaNet PS + verl贡献

Sources:
- prefix-0501项目: ~/workspace/project/prefix-proj/prefix-0501_claude-loop/
- 项目文档: docs/overview.md, docs/concepts.md, docs/pending-items.md
- 核心代码: prefix-sharing/prefix_sharing/core/, integrations/, backends/
- 项目日志: diary/2026-06-07.md