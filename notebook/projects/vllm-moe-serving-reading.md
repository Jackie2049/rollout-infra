# vLLM MoE Serving 架构源码阅读

> 深入理解 vLLM 的 MoE 模块化 kernel 架构：从 Router 到 Expert Parallelism

## 1. 核心文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `fused_moe/layer.py` | ~350 | FusedMoE 层定义（PluggableLayer）、权重管理 |
| `fused_moe/modular_kernel.py` | ~500 | **核心**: 模块化 MoE kernel 抽象（Router→Prepare→Experts→Finalize） |
| `fused_moe/fused_moe_modular_method.py` | ~400 | 模块化方法实现，组合 Prepare/Experts/Finalize |
| `fused_moe/runner/moe_runner.py` | ~500 | MoE Runner，执行路由+分发+计算+合并 |
| `fused_moe/config.py` | ~300 | MoE 配置（量化、并行、路由方法） |
| `fused_moe/router/` | ~800 | 路由器实现（TopK/GroupedTopK/Bias/Factory） |
| `fused_moe/prepare_finalize/` | ~2476 | **8+ All-to-All 后端**（DeepEP/FlashInfer/NIXL/MoRI 等） |
| `fused_moe/experts/` | ~2000 | Expert 计算实现（Triton/CUTLASS/DeepGEMM/Marlin 等） |

## 2. 模块化 Kernel 架构

### 2.1 流水线设计

```
[Router] → [Prepare] → [Experts] → [Finalize]

Router:    计算 topk_ids, topk_weights (路由决策)
Prepare:   量化 + All-to-All Dispatch + Permute (分发 tokens 到专家)
Experts:   matmul + activation + matmul (专家计算)
Finalize:  All-to-All Combine + weight application + reduce (收集结果)
```

### 2.2 核心抽象 (modular_kernel.py)

```python
class FusedMoEPrepareAndFinalizeModular(ABC):
    """Prepare (量化+分发) 和 Finalize (合并+归约) 的抽象基类"""

    def prepare(self, a1, topk_weights, topk_ids, ...) -> PrepareResultType:
        """量化输入 + All-to-All Dispatch"""
        # 返回: (量化后的输入, scales, expert_token_metadata, ...)

    def finalize(self, output, fused_expert_output, topk_weights, topk_ids, ...):
        """All-to-All Combine + 权重应用 + 归约"""

class FusedMoEExperts(ABC):
    """Expert 计算的抽象基类"""

    def apply(self, a1q, a1q_scale, expert_tokens_metadata, ...) -> torch.Tensor:
        """执行 expert matmul"""

class FusedMoEModularKernel:
    """组合 Prepare+Experts+Finalize 的完整 MoE kernel"""
```

**设计要点**:
- Prepare/Finalize 和 Experts 可以独立选择（"混搭"）
- 避免组合爆炸：N 个 Prepare × M 个 Expert = N×M 种组合，而不是 N×M 种实现
- `PrepareResultType` 定义统一的数据交换格式

## 3. All-to-All 后端 (prepare_finalize/)

### 3.1 后端列表

| 后端 | 文件 | 行数 | 适用场景 |
|------|------|------|----------|
| **NoDPEP** | `no_dp_ep.py` | 141 | 无 EP，仅 TP |
| **NaiveDPEP** | `naive_dp_ep.py` | 301 | 简单 DP+EP |
| **Batched** | `batched.py` | 171 | 批量处理 |
| **DeepEP-HT** | `deepep_ht.py` | 437 | DeepSeek EP，高吞吐 (Prefill) |
| **DeepEP-LL** | `deepep_ll.py` | 448 | DeepSeek EP，低延迟 (Decode) |
| **FlashInfer NVLink 1-sided** | `flashinfer_nvlink_one_sided.py` | 168 | FlashInfer All-to-All |
| **FlashInfer NVLink 2-sided** | `flashinfer_nvlink_two_sided.py` | 239 | FlashInfer 双向 |
| **MoRI** | `mori.py` | 124 | MoRI All-to-All |
| **NIXL EP** | `nixl_ep.py` | 418 | NIXL-based EP |

### 3.2 NoDPEP (no_dp_ep.py)

最简单的实现，不使用 All-to-All：

```python
class MoEPrepareAndFinalizeNoDPEPModular:
    def prepare(self, a1, topk_weights, topk_ids, ...):
        # 仅量化，不分发
        a1q, a1q_scale = _quantize_input(a1, quant_config)
        return a1q, a1q_scale, None, None, None

    def finalize(self, output, fused_expert_output, ...):
        # 仅权重应用和归约
        weight_and_reduce_impl.apply(output, fused_expert_output, ...)
```

**适用场景**: TP 并行（所有 GPU 有所有 expert 的权重副本），不需要 All-to-All。

### 3.3 DeepEP (deepep_ht.py / deepep_ll.py)

DeepSeek 开源的高性能 All-to-All 库：
- **HT (High Throughput)**: 用于 Prefill，批量发送 token
- **LL (Low Latency)**: 用于 Decode，单 token 发送，使用 warp specialization

### 3.4 激活格式

```python
class FusedMoEActivationFormat(Enum):
    Standard = "standard"              # (num_tokens, hidden_dim)
    BatchedExperts = "batched_experts"  # (num_experts, max_tokens_per_expert, hidden_dim)
```

不同的 Prepare/Finalize 实现产生不同格式的激活，Expert kernel 需要适配。

## 4. Expert 计算后端 (experts/)

| 后端 | 文件 | 说明 |
|------|------|------|
| **Triton MoE** | `triton_moe.py` | 纯 Triton 实现，fallback |
| **CUTLASS MoE** | `cutlass_moe.py` | CUTLASS 高性能 |
| **DeepGEMM MoE** | `deep_gemm_moe.py` | DeepSeek DeepGEMM |
| **Marlin MoE** | `marlin_moe.py` | Marlin 量化 MoE |
| **FlashInfer MoE** | `flashinfer_b12x_moe.py` | FlashInfer batched |
| **TRT-LLM MoE** | `trtllm_fp8_moe.py` | TensorRT-LLM kernel |
| **AITER MoE** | `rocm_aiter_moe.py` | ROCm AMD GPU |

## 5. 路由器 (router/)

| 路由器 | 说明 |
|--------|------|
| `fused_topk_router.py` | 标准 Top-K 路由 |
| `grouped_topk_router.py` | 分组 Top-K (DeepSeek) |
| `fused_topk_bias_router.py` | 带 bias 的 Top-K (负载均衡) |
| `routing_simulator_router.py` | 路由模拟 (调试) |
| `zero_expert_router.py` | 零路由 (所有 token 到 expert 0) |

## 6. EP vs TP 选择

| 策略 | 通信 | 显存 | 适用 |
|------|------|------|------|
| **TP** | AllReduce (每层) | 每卡存全部 expert | 小 MoE (8-16 experts) |
| **EP** | All-to-All (每层) | 每卡存部分 expert | 大 MoE (64-256 experts) |
| **DP+EP** | All-to-All + DP 同步 | DP 组间共享 | 多节点大规模 |

vLLM 的设计支持所有三种策略的混搭。

## 7. 与 MoE 笔记的关联

本源码阅读与 `notebook/fundamentals/moe.md` 的理论知识对应：
- 笔记中的 "Expert Parallelism" → `prepare_finalize/` 的 All-to-All 实现
- 笔记中的 "Top-K 路由" → `router/` 的路由器实现
- 笔记中的 "负载均衡" → `fused_topk_bias_router.py` 的 bias-based 路由
- 模拟器 `tools/expert_parallelism_sim.py` 中的 All-to-All 通信分析

## 参考资料

- 源码路径: `vllm-latest/vllm/model_executor/layers/fused_moe/`
- 相关笔记: [MoE 架构](../fundamentals/moe.md), [vLLM V1 Executor](vllm-v1-executor-reading.md)
- 相关工具: `tools/moe_router_demo.py`, `tools/expert_parallelism_sim.py`
