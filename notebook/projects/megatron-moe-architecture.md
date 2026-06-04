# Megatron-LM MoE 架构分析

> 目录: `megatron/core/transformer/moe/` (13 文件)
> 分析日期: 2026-06-04

## 架构总览

```
MoE Layer
  ├── Router:           token → expert 分配
  │   ├── Top-K gating
  │   ├── Aux loss (load balancing)
  │   └── Token dropping (capacity overflow)
  ├── Token Dispatcher: Dispatch → Compute → Combine
  │   ├── Dispatch:     permute + all-to-all
  │   ├── Experts:      grouped GEMM (per-expert)
  │   └── Combine:      all-to-all + unpermute
  └── Shared Experts:   (optional) 所有 token 共享
```

## Router — Top-K Gating

```python
class Router(MegatronModule):
    def forward(self, hidden_states):
        # 1. Linear gate: [S, H] → [S, E]
        logits = F.linear(hidden_states, self.weight)  # router weight

        # 2. Optional bias/routing strategies
        logits = apply_biased_logits(logits, biased_experts)
        logits = apply_random_logits(logits, random_mode)

        # 3. Top-K selection
        scores, indices = topk_routing_with_score_function(logits, top_k)

        # 4. Auxiliary load balancing loss
        aux_loss = switch_load_balancing_loss_func(
            scores, indices, num_experts
        )

        # 5. Token dropping (capacity overflow)
        scores, indices = apply_router_token_dropping(
            scores, indices, capacity_factor
        )

        return scores, indices, aux_loss
```

### Load Balancing Strategies

| 策略 | 描述 |
|------|------|
| **Aux Loss** | Switch Transformer 风格: `E * sum(f_i * P_i)` |
| **Z-Loss** | Logit magnitude regularizer: `log(sum(exp(z)))^2` |
| **Sinkhorn** | Optimal transport rebalancing |
| **Token Dropping** | Capacity factor overflow → drop excess tokens |

### Capacity Factor

```python
# Capacity = (tokens / num_experts) * capacity_factor * top_k
# CF=1.0: 刚好够, CF=1.25: 25% buffer
# 超量 token 被 drop (score=0, routing failed)
```

---

## Token Dispatcher — All-to-All Pipeline

```
Dispatch Phase:
  1. Permute:    排序 tokens by target expert
  2. All-to-All: 跨 EP rank 发送 tokens
  3. Pad:        对齐到 expert group 大小

Compute Phase:
  4. Expert:     grouped GEMM [group_size, H] × [H, 4H] → [group_size, 4H]
                 SiLU gate: output × gate_proj * activation(up_proj)
                 Output:    [group_size, 2H] × [2H, H] → [group_size, H]

Combine Phase:
  5. All-to-All: 跨 EP rank 接收 results
  6. Unpermute:  恢复原始 token 顺序
```

### Notation

```
H: hidden size
B: micro batch size
S: sequence length
E: num experts
K: top-k
TP: tensor parallel size
EP: expert parallel size

num_local_tokens: S/TP * B
num_global_tokens: num_local_tokens × TP × EP
```

### Dispatch 代码

```python
def token_permutation(tokens, indices, num_experts, capacity):
    # 1. Sort tokens by target expert
    sorted_indices = indices.argsort()

    # 2. All-to-All across EP ranks
    tokens = all_to_all(tokens, group=ep_group)

    # 3. Group tokens per expert for batched GEMM
    # tokens_per_expert[e] has shape [capacity, H]
    return tokens_per_expert
```

---

## Expert Computation

```python
class GroupedMLP:
    """
    将所有 expert 的 tokens 打包为一个 batch:

    Rather than:  for e in experts: out += expert[e](tokens[e])
    Do:           out = fused_gemm(concat(tokens[e] for e), all_expert_weights)

    关键优化:
    - 单次大 GEMM 替代多次小 GEMM
    - 利用 VC (Variable Batched) GEMM
    - 可选 token permutation for better L2 locality
    """
```

---

## Expert Parallelism (EP)

### EP 配置

```
EP=1, E=8:  所有 expert 在每个 rank
EP=2, E=8:  每个 rank 4 个 expert
EP=8, E=8:  每个 rank 1 个 expert (最大通信)

通信: Dispatch (all-to-all) + Combine (all-to-all)
      2 × All-to-All / MoE layer
```

### Fused All-to-All (DeepEP)

```python
from megatron.core.transformer.moe.fused_a2a import (
    fused_dispatch,    # permute + all-to-all + pad (fused)
    fused_combine,     # all-to-all + unpermute + scale (fused)
    hybrid_ep_dispatch,  # TP+EP hybrid
    hybrid_ep_combine,
    set_deepep_num_sms,  # SM allocation for DeepEP
)
```

**DeepEP 优化**:
- NVLink RDMA 代替 host staging
- SM 控制 (预留 SM 给通信 kernel)
- 异步 dispatch/combine 与 compute 重叠

### TP+EP Hybrid

```
TP before Router: [S/TP, H] each rank
EP after Router:  token → expert (cross-rank)
Combine:          expert output → original order
TP after Combine: normal TP (RowParallel linear)

通信: TP AllReduce + EP All-to-All (双通信)
```

---

## Shared Experts

```python
class SharedExpertMLP:
    """
    DeepSeek-V2/V3 风格:

    output = shared_expert(tokens) + sum(routed_expert_i(token) for token in tokens_i)

    优点:
    - 共享专家捕获 common knowledge
    - 路由专家处理 specialization
    - 稳定训练 (共享专家保证所有 token 有 base quality)
    """
```

---

## Aux Loss Auto-Scaler

```python
class MoEAuxLossAutoScaler:
    """
    自动调节 aux loss 权重:

    loss = main_loss + scale * aux_loss

    scale 动态调整:
    - 如果 aux_loss 太小 (load 不均衡) → 增大 scale
    - 如果 aux_loss 太大 (影响 main loss) → 减小 scale
    """
```

---

## 代码位置速查

| 文件 | 内容 |
|------|------|
| `router.py` | Top-K gating, aux loss, token dropping |
| `token_dispatcher.py` | Dispatch + Combine via all-to-all |
| `experts.py` | GroupedMLP expert computation |
| `moe_layer.py` | MoE Layer wrapper |
| `fused_a2a.py` | Fused All-to-All (DeepEP backend) |
| `shared_experts.py` | Shared expert MLP |
| `moe_utils.py` | Routing utilities, capacity calc |
| `router_replay.py` | Router replay for inference |
| `token_dispatcher_inference.py` | Inference-optimized dispatcher |
