# vLLM V1 MoE Serving & Expert Parallelism Deep Dive

> 2026-06-07 | 源码阅读: DPEngineCoreProc + FusedMoE layer + DeepEP + MoE架构对比

## 一、MoE架构: 从Dense到Sparse

### Dense vs MoE 计算对比

```
Dense模型 (如LLaMA-70B):
  每个token经过所有参数 → 计算∝总参数量
  70B模型: 每token 70B参数 × 6 FLOPS/param = 420 GFLOPS/token

MoE模型 (如DeepSeek-V3):
  每个token只经过部分参数 → 计算∝激活参数量
  671B总参数 / 37B激活参数 = 18x稀疏比
  每token: 37B × 6 = 222 GFLOPS/token (比Dense 70B仅多3x!)
  但存储671B → 5x内存开销

关键: MoE用少量计算获得大模型容量 → 推理成本≈37B而非671B
```

### DeepSeek-V3 MoE结构

```
每层: 256 routed experts + 1 shared expert
  shared expert: 所有token必经 (像Dense层)
  routed experts: Top-K=6 选择 (每token选6个expert)

路由: group-wise routing
  256 experts分成8组(每组32)
  每token每组选4个 → 8×4=32候选 → 再Top-6
  目的: 限制跨节点通信(每组在同一节点)

辅助loss: auxiliary loss for load balancing
  β×Σ(f_i×P_i) → f_i=expert i频率, P_i=router概率
  防止所有token涌向少数"热门"expert
```

## 二、Expert Parallelism (EP)

### EP原理

```
假设: 256 experts, 8 GPU
  每GPU放32 experts (256/8)
  token路由: [token→expert_id] → expert在不同GPU上 → 需跨GPU传输!

  通信模式: All-to-All
    Dispatch: 每GPU将自己的token发给对应的expert所在GPU
    Combine:  每GPU从expert所在GPU收集处理结果

  示例 (8 GPU, 256 experts, 128 tokens):
    GPU 0有128 tokens → 路由决定 → 16去GPU0(expert0-31), 16去GPU1, ...
    Dispatch: GPU0发送16×6=96 tokens到其他7个GPU
    Combine:  GPU0从其他7个GPU接收96个处理结果+本地16结果
```

### EP通信瓶颈

```
All-to-All通信量:
  每个token: hidden_size × dtype_size
  7B模型: hidden=4096, FP16=2B → 每token 8KB
  B=128: 128×8KB = 1MB per GPU
  8 GPU All-to-All: 总通信 = 8×1MB×7 = 56MB

NVLink: 160 GB/s bidirectional → 56MB/160 = 0.35ms ← 快!
PCIe:  20 GB/s bidirectional → 56MB/20 = 2.8ms ← 慢8x!
Ethernet: 1.6 GB/s (400 Gbps IB) → 56MB/1.6 = 35ms ← 灾难!

结论: EP必须NVLink! PCIe RTX 4090完全不可行(实测AllReduce 3-7.5GB/s)
```

### EP vs TP vs DP 对比

| 方法 | 通信模式 | 每token通信量 | 适用模型 | NVLink必需? |
|------|---------|-------------|---------|------------|
| DP | AllReduce(梯度) | ∝模型大小 | 所有 | 否(但慢) |
| TP | AllReduce(每层) | ∝激活大小 | Dense | 是(<5%占比) |
| EP | All-to-All(token) | ∝hidden×batch | **MoE** | **是**(主要瓶颈!) |
| TP+EP | AllReduce+All-to-All | 两者混合 | 大MoE | 是 |

## 三、DeepEP: DeepSeek开源EP通信库 (2025-02)

### 核心架构

```
DeepEP提供两种kernel模式:

1. Normal Mode (Training/低延迟):
   - NVLink + RDMA(IB/RoCE) 双路径
   - Dispatch: token→expert分配 → 跨节点RDMA发送
   - Combine:  expert结果 → RDMA回收
   - 优化: 多channel并行 + hook buffer管理
   - 适用: 训练(小batch+低延迟优先)

2. High-Throughput Mode (Serving):
   - **NVLink-only forwarding** (不用IB!)
   - 前提: 所有expert在同一节点(EP=NVLink范围内的GPU数)
   - 不需要跨节点 → 通信仅NVLink → 160 GB/s → 极快
   - 适用: 大batch推理(吞吐优先)
```

### Dispatch + Combine 流程

```
Dispatch (发送token到expert):
  1. Router计算: 每token→expert_id (GPU上计算)
  2. Group tokens by destination GPU
  3. TMA/cp.async拷贝到send buffer
  4. All-to-All dispatch: 每GPU发送给其他GPU
  5. 接收端: 从recv buffer读出 → 按expert组织

Combine (收集expert结果):
  1. Expert计算完成 (各GPU本地计算)
  2. 按源GPU组织结果
  3. All-to-All combine: 每GPU发送结果回源GPU
  4. 源GPU: 按token顺序重组 → 加权求和(top-k权重)

关键: Dispatch和Combine各一次All-to-All → 每层2次A2A!
```

### DeepEP优化技术

```
- **FP8通信**: token数据用FP8传输 → 带宽需求减半
  但expert计算仍用FP16/BF16 → dispatch时FP8编码, combine时FP16还原

- **不对称发送**: MoE的token分布不均匀 → expert3可能收到50token而expert7只10
  DeepEP支持不对称buffer → 非padding浪费带宽(vs NCCL对称假设浪费30-50%)

- **Hook buffer**: 发送/接收buffer管理 → 避免频繁malloc/free

- **Channel并行**: NVLink多channel同时传输 → 充分利用160GB/s双向带宽
```

## 四、vLLM V1 MoE Serving实现

### DPEngineCoreProc: 数据并行核心

```python
# vllm/v1/engine/core.py:1676
class DPEngineCoreProc(EngineCoreProc):
    """MoE专用数据并行EngineCore"""

    # 断言: 只用于MoE模型
    assert vllm_config.model_config.is_moe

    # Wave协调: DP ranks按"wave"同步
    self.step_counter = 0  # 每32步all-reduce一次
    self.current_wave = 0

    # 两阶段暂停:
    # Phase1: pending_pause=True → 继续dummy stepping到all-reduce共识点
    # Phase2: 所有rank同意暂停 → ignore_start_dp_wave=True → 停止
```

### 核心busy loop

```
run_busy_loop():
  while not shutdown:
    1. _process_input_queue() → 处理请求
    2. _maybe_publish_request_counts() → 发布负载统计(DP负载均衡)
    3. _process_engine_step() → 执行模型forward
    4. 如果无请求 → execute_dummy_batch() → 空forward(DP同步需要)
    5. _has_global_unfinished_reqs() → All-Reduce检查(每32步)
       → sync_dp_state(dp_group, has_unfinished, pending_pause)
    6. 如果all-reduce说所有rank完成 → 暂停, 通知client
```

**关键**: 每32步才做一次All-Reduce → 减少通信开销(从每步→每32步)!

### Wave机制

```
Wave = 一批请求的完整处理周期

DP rank 0 (Coordinator):
  接收所有请求 → 分配wave编号 → 广播START_DP_WAVE给所有rank

DP rank 1-N:
  收到START_DP_WAVE → 开始stepping → 处理分配的请求
  → 32步后all-reduce检查 → wave完成 → 通知coordinator

好处: 所有DP rank同步处理同一批请求 → 负载均衡
```

### 两阶段暂停协议

```
场景: 一个rank要暂停(无新请求), 但其他rank还在处理

Phase1: pending_pause=True
  → 继续stepping(发dummy batch) → 等所有rank到all-reduce共识点

Phase2: pause_consensus=True (all-reduce确认所有rank同意)
  → ignore_start_dp_wave=True → 不接受新wave → 安全暂停

原因: 不能直接停 → 可能导致其他rank的all-reduce永远等不到这个rank
```

### Elastic EP: 动态扩缩容

```python
# vllm/v1/engine/core.py:1922
def reinitialize_distributed(self, reconfig_request):
    # 动态改变DP size!
    new_dp_size = reconfig_request.new_data_parallel_size

    # ElasticEPScalingState:
    # "removing": 当前rank要退出 → raise SystemExit
    # "existing": 重新配置后继续
    # "scale_up": 新rank加入

    self.eep_scaling_state = ElasticEPScalingState(
        worker_type="removing" if is_shutdown else "existing",
        scale_type="scale_down" if is_scale_down else "scale_up",
    )
```

**Elastic EP**: 根据负载动态增减DP rank → 低负载时少GPU(省钱) → 高负载时加GPU(保吞吐)

## 五、FusedMoE Layer实现

```python
# vllm/model_executor/layers/fused_moe/layer.py
class FusedMoE(PluggableLayer):
    # 参数: num_experts, top_k, hidden_size, intermediate_size
    # 并行配置: FusedMoEParallelConfig(tp, ep, dp, pcp, sp)

    # 关键属性:
    global_num_experts = num_experts + num_redundant_experts
    logical_num_experts = num_experts

    # 专家放置策略:
    expert_placement_strategy → 决定expert如何分布到GPU

    # EPLB (Expert Parallel Load Balancer):
    eplb_state → 动态负载均衡(热expert复制到多GPU)

    # 并行:
    tp_size, ep_size, dp_size, pcp_size (Pipeline Consistent Parallel)

    # 执行:
    # 1. Router: 计算top-k expert + 权重
    # 2. MoERunner: 执行expert计算
    # 3. SharedExperts: 所有token必经的shared expert (DeepSeek-V3)
```

### FusedMoEParallelConfig

```
并行配置:
  tp_size: Tensor Parallel (权重切分)
  ep_size: Expert Parallel (expert分布)
  dp_size: Data Parallel (请求复制)
  pcp_size: Pipeline Consistent Parallel (跨层专家一致性)
  sp_size: Sequence Parallel (序列切分)

组合: TP×EP×DP×PCP×SP = 总GPU数
典型: DeepSeek-V3 on 8×H100 = TP=1×EP=8×DP=1 (或 DP=2×EP=4)
```

### Expert Placement Strategy

```
ExpertPlacementStrategy:
  - 均匀分布: 256 experts → 8 GPU → 每GPU 32
  - EPLB动态: 热expert复制到多GPU → 负载均衡
  - Redundant experts: 添加冗余expert → 更灵活分布

冗余expert (DeepSeek-V3):
  num_redundant_experts: 额外的expert副本
  256 logical experts + redundant = global_num_experts
  EPLB可以将logical expert映射到不同GPU上的冗余副本
```

## 六、MoE推理Pipeline完整流程

```
一个token的MoE推理路径:

1. Router计算:
   hidden_states → gate network → logits
   logits → Top-K selection → expert_ids + weights

2. Dispatch (All-to-All):
   按expert_id分组 → 发到对应GPU
   (EP: 每GPU负责一部分expert)

3. Expert计算:
   每GPU收到分配的tokens → 各expert独立计算
   expert_i: gate_up_proj(hidden) → activation → down_proj
   (batched GEMM: 同一expert的所有token一起计算)

4. Combine (All-to-All):
   expert结果 → 按源GPU分组 → 发回原GPU
   原GPU: 按token顺序重组 → 加权求和(top-k weights)

5. Shared Expert (如果存在):
   所有token: hidden → shared_expert → 加到结果上
   (不需要All-to-All! 所有token本地计算)

6. 输出:
   output = Σ(top_k_weight_i × expert_i_output) + shared_expert_output
```

## 七、RTX 4090 MoE Serving可行性

| 特性 | RTX 4090 (PCIe) | H100 (NVLink) |
|------|-----------------|---------------|
| EP可行性 | **不可行** | **可行** |
| All-to-All BW | 3-7.5 GB/s (实测) | 160 GB/s (NVLink) |
| 8GPU EP延迟 | ~7.5ms | ~0.35ms |
| 通信占比 | >75% | <5% |
| 单GPU推理 | **可行** (小MoE) | 可行 |

**RTX 4090只能做单GPU MoE推理**:
- 全256 experts在同一24GB GPU → 671B模型不可能
- 8B MoE(如Mixtral-8×7B): 每expert ~1GB → 8 expert+共享=约13GB → fits 24GB
- 但无法EP → 只能单GPU → 吞吐低

## 八、实用结论

1. **EP是MoE serving的核心挑战**: All-to-All是瓶颈, 必须NVLink
2. **DeepEP是关键基础设施**: 不对称buffer + FP8通信 + NVLink-only高吞吐模式
3. **vLLM DP MoE**: DPEngineCoreProc(32步All-Reduce + Wave + 两阶段暂停 + Elastic EP)
4. **RTX 4090不适合EP**: PCIe All-to-All太慢 → 只能单GPU推理
5. **EPLB很重要**: 热expert负载不均 → 需动态重分布或冗余expert
6. **Shared Expert不需要A2A**: 所有token本地计算 → 减少总通信量
7. **MoE推理成本≈激活参数**: 671B但只37B active → 比Dense 70B仅多3x计算
8. **batched GEMM是关键**: 同expert多token → batched matmul → 利用Tensor Core

Sources:
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
- [vLLM V1 Source Code](https://github.com/vllm-project/vllm)
- [vLLM MoE Documentation](https://docs.vllm.ai/en/latest/features/moe.html)