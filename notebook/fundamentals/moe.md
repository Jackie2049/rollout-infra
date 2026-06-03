# MoE (Mixture of Experts) 架构深度解析

> 稀疏激活、Top-K 路由、Expert Parallelism——MoE 如何用少量计算激活大量参数

## 1. MoE 基础概念

### 1.1 Dense vs Sparse：为什么需要 MoE

```
Dense 模型:
  输入 x → FFN(所有参数) → 输出 y
  每个token激活所有参数 → 计算量大，扩展困难

MoE 模型:
  输入 x → Router → 选择 top-K 个 expert → 只激活部分参数
  总参数很多，但每次只激活一小部分 → 用参数量换质量
```

### 1.2 核心数学

```
MoE Layer 输出:
  y = Σ_{i∈TopK} g_i(x) · E_i(x)

其中:
  g_i(x) = gate 网络输出的第 i 个权重 (softmax 归一化)
  E_i(x) = 第 i 个 expert 的输出 (本质是 FFN)
  TopK = gate 值最大的 K 个 expert 的索引

Router:
  gate_logits = W_gate @ x        # [hidden] → [num_experts]
  gate_probs = softmax(gate_logits)
  top_k_indices = argsort(-gate_probs)[:K]
  top_k_gates = gate_probs[top_k_indices] / sum(gate_probs[top_k_indices])
```

### 1.3 参数效率对比

```
| 模型 | 总参数 | 活跃参数 | 等效Dense | 训练成本 |
|------|--------|----------|-----------|----------|
| LLaMA-2 70B | 70B | 70B | 70B | 1x |
| Mixtral 8x7B | 46.7B | 12.9B | ~13B | ~0.2x |
| DeepSeek-V3 | 671B | 37B | ~37B | ~0.05x |

关键洞察: MoE用 3-18x 的总参数量，但只激活等价Dense模型大小的参数
→ 模型质量接近大Dense模型，计算成本接近小Dense模型
```

### 1.4 MoE 层在 Transformer 中的位置

```
标准 Transformer 层:
  x → Attention → Add&Norm → FFN → Add&Norm → y
                              ↑
                          Dense: 一个大 FFN

MoE Transformer 层:
  x → Attention → Add&Norm → MoE Layer → Add&Norm → y
                              ↑
                    Router + N 个小 FFN (Experts)
                    每个 token 只选 top-K 个 expert
```

## 2. 路由机制 (Routing)

### 2.1 Top-K 路由（标准方法）

```
算法:
1. 计算 gate logits: g = x @ W_gate  (W_gate: [hidden, num_experts])
2. Softmax: p = softmax(g)
3. 选择 top-K: indices = argsort(-p)[:K]
4. 提取权重: weights = p[indices]
5. 归一化: weights = weights / sum(weights)
6. 计算输出: y = Σ_i weights[i] * Expert[indices[i]](x)

典型配置:
  K=1: 最稀疏，计算量最少，但可能信息损失
  K=2: 平衡点，Mixtral 使用
  K=4-8: 更多的计算，更好的质量 (DBRX用K=4, DeepSeek-V3用K=8)
```

### 2.2 Expert Choice 路由

```
传统 Top-K: token 选 expert (token perspective)
  问题: 某些 expert 可能被太多 token 选，负载不均

Expert Choice: expert 选 token (expert perspective)
1. 计算所有 token-expert 亲和度矩阵 S ∈ [T, E]
2. 每个expert独立选择亲和度最高的 T*K/E 个token
3. 自然达到完美负载均衡!

优点:
  - 无需 auxiliary loss
  - 完美负载均衡
  - 某些token可能被多个expert处理(更多计算)
  - 某些token可能不被任何expert选中(可以skip)

缺点:
  - 不适用于因果语言模型的训练(token不能被future expert处理)
  - 实现更复杂
```

### 2.3 Hash 路由

```
简单哈希路由:
  expert_index = hash(token_id) % num_experts

优点:
  - 零计算开销
  - 确定性 (推理时无随机性)
  - 天然负载均衡

缺点:
  - 没有学到token-expert亲和度 → 质量差
  - 只用于推理优化或极低延迟场景

改进: Switch Hash — 训练时学习hash函数参数
```

### 2.4 DeepSeek-V3 Group-Limited 路由

```
DeepSeek-V3: 256个routing experts + 1个shared expert, top-8

Group-Limited 路由:
1. 将256个expert分成8组，每组32个
2. 对每个token:
   a. 计算gate logits: g = x @ W_gate  (输出256维)
   b. 每组内选择top-1: 8个候选expert
   c. 全局从8个候选中选top-8 (但通常全选)
3. 加上shared expert (所有token都经过)

设计理念:
  - 细粒度expert: 256个小expert >> 8个大expert
  - 组内选择降低计算: 不需要对256维做完整排序
  - shared expert捕捉通用知识，routing expert捕捉专业知识
```

## 3. 负载均衡 (Load Balancing)

### 3.1 路由崩塌问题

```
问题: 训练初期，gate 网络倾向于把所有 token 送给少数 expert
  → 大部分 expert 不被使用 (参数浪费)
  → GPU 利用率低 (EP 场景下某些 GPU 空闲)

原因:
  - 初始时 gate 参数随机
  - 某些 expert 碰巧得到更多 token → 更好的梯度 → 更强的gate分数
  - 正反馈循环 → 路由崩塌

严重性:
  8个expert中可能只有2-3个被频繁使用
  其余expert几乎不参与计算 → 等效于小Dense模型
```

### 3.2 Auxiliary Loss（辅助损失）

```
最常用的负载均衡方法:

L_aux = α · N · Σ_{i=1}^{N} f_i · P_i

其中:
  N = num_experts
  f_i = 被 expert i 处理的 token 比例 = count_i / (T × K)
  P_i = expert i 的平均 gate 概率 = mean(gate_probs[:, i])
  α = 辅助损失权重 (典型值 0.01)

为什么用 f_i · P_i 而不是 f_i²?
  f_i: 不可微分 (离散的选择)
  P_i: 可微分 (连续的概率)
  f_i · P_i: 让 f_i 和 P_i 都趋向均匀

理想状态 (完美均衡):
  f_i = 1/N, P_i = 1/N → L_aux = α · N · N × (1/N)² = α

典型值: L_aux ≈ 0.01 × 8 = 0.08 (8 experts)
```

### 3.3 DeepSeek-V3: Auxiliary-Loss-Free

```
问题: Auxiliary loss 会影响主任务训练质量
  α 太大 → 负载均衡好但模型质量差
  α 太小 → 负载不均

DeepSeek-V3 方案: 用可学习的 bias 替代 auxiliary loss

每个 expert 有一个 bias b_i:
  gate_logits_biased = gate_logits + b_i
  → bias 大的 expert 被更多 token 选择

训练时:
  - 动态监控每个 expert 的负载
  - 如果 expert i 过载 → 减小 b_i
  - 如果 expert i 欠载 → 增大 b_i
  - 更新率: b_i += γ × (target_load - actual_load)

优势:
  - 不影响主损失函数
  - 调整更直接 (不经过梯度反传)
  - 模型质量显著提升 (DeepSeek报告)
```

### 3.4 容量因子 (Capacity Factor)

```
每个 expert 的最大容量:
  capacity = CF × (num_tokens × top_k / num_experts)

CF = 1.0: 严格均衡 (可能drop token)
CF = 1.25: 留25%余量 (常见设置)
CF = 2.0: 大余量 (几乎不drop，但浪费显存)

Token drop 处理:
  当 expert 达到容量上限:
    1. 优先处理 gate 分数高的 token
    2. 被drop的token: 分配给负载最低的expert 或 丢弃
    3. 丢弃的token不参与该层的计算 (相当于identity)

Drop rate 影响:
  <1%: 基本无影响
  1-5%: 轻微质量下降
  >5%: 明显质量损失
```

## 4. 关键 MoE 模型

### 4.1 Mixtral 8x7B

```
架构:
  基于 LLaMA 架构, 每 2 层用 MoE 替代 FFN
  8个expert, top-2 routing

参数分析:
  总参数: 46.7B (但只有 12.9B 活跃/token)
  每层 MoE:
    8 × FFN(hidden=4096, intermediate=14336) ≈ 3.7B
    Shared: attention + embedding ≈ 0.5B

与 LLaMA-2 70B 对比:
  | 指标 | Mixtral 8x7B | LLaMA-2 70B |
  |------|-------------|-------------|
  | 总参数 | 46.7B | 70B |
  | 活跃参数 | 12.9B | 70B |
  | 推理FLOPs | ~13B | 70B |
  | 性能 | 接近70B | 基准 |
  | 显存 | ~90GB (FP16) | ~140GB (FP16) |

关键发现: 12.9B 活跃参数的 MoE 达到接近 70B Dense 的质量
→ MoE 的参数效率非常高
```

### 4.2 DeepSeek-V2

```
架构创新: 细粒度 Expert + Shared Expert
  64 个 routing expert + 2 个 shared expert
  top-6 routing + top-2 shared = 每token经过8个expert

细粒度设计:
  传统MoE: 8个大expert (每个 expert 是完整 FFN)
  DeepSeek-V2: 64个小expert (每个 expert 是更小的 FFN)
  → 路由更灵活: 64选6 vs 8选2

参数:
  总参数: 236B
  活跃参数: 21B
  每层 expert: 64 × FFN(hidden=5120, intermediate=1536)

MLA (Multi-head Latent Attention):
  配合 MoE 使用，KV Cache 压缩到原来的 1/70
  → 推理显存大幅降低
```

### 4.3 DeepSeek-V3

```
架构:
  256 个 routing expert + 1 个 shared expert
  top-8 routing (Group-Limited)
  Auxiliary-Loss-Free 负载均衡

参数:
  总参数: 671B (0.67T!)
  活跃参数: 37B
  激活率: 5.5% (每token只激活 37B/671B)

训练:
  14.8T tokens
  2048 × H800 GPU (使用 HAI-LLM 框架)
  FP8 混合精度训练 (业界首次在超大规模验证 FP8 MoE)
  训练成本: $5.6M (极低!)

创新点:
  1. Auxiliary-Loss-Free: 用 bias 调整替代辅助损失
  2. Group-Limited 路由: 分组选择降低排序开销
  3. Shared Expert: 1个共享expert捕捉通用知识
  4. FP8 MoE: expert计算用 FP8 精度
  5. Multi-Token Prediction: 辅助训练目标
```

### 4.4 DBRX

```
Databricks 开源 MoE 模型:
  16 个 expert, top-4 routing
  总参数: 132B, 活跃参数: 36B

特点:
  - 16选4 比 8选2 路由更灵活
  - 在多个 benchmark 上超过 LLaMA-2 70B
  - 开源，可商用

教训:
  expert 数量和 top-k 的选择影响模型质量
  更多 expert + 更大 top-k → 更灵活但计算量增加
```

## 5. Expert Parallelism (EP)

### 5.1 EP 与其他并行策略的关系

```
四种并行维度的正交组合:

  DP (数据并行): 切 batch → 每个 GPU 持有完整模型副本
  TP (张量并行): 切 hidden_dim → 每个 GPU 持有模型的一部分
  PP (流水线并行): 切 layers → 每个 GPU 持有部分层
  EP (专家并行): 切 experts → 每个 GPU 持有部分 expert

MoE 新增的 EP:
  每个 GPU 持有 N/E 个 expert (N=总expert数, E=EP并行度)
  token 需要发送到持有对应 expert 的 GPU
  → 需要 All-to-All 通信
```

### 5.2 All-to-All 通信详解

```
EP 的三阶段流程 (4 GPU, 8 experts 的例子):

初始状态:
  GPU 0: 持有 Expert 0,1, 接收 batch 的 1/4 token
  GPU 1: 持有 Expert 2,3
  GPU 2: 持有 Expert 4,5
  GPU 3: 持有 Expert 6,7

Stage 1: Dispatch (All-to-All)
  GPU 0 的 tokens 根据路由结果分组:
    → 发给 GPU 0: 需要 Expert 0,1 的 tokens
    → 发给 GPU 1: 需要 Expert 2,3 的 tokens
    → 发给 GPU 2: 需要 Expert 4,5 的 tokens
    → 发给 GPU 3: 需要 Expert 6,7 的 tokens
  所有 GPU 同时发送 → All-to-All

Stage 2: Compute
  每个 GPU:
    接收来自所有 GPU 的 tokens
    用本地 expert 计算输出
    (这是纯计算，无通信)

Stage 3: Gather (All-to-All)
  计算结果发回原始 GPU:
    GPU 0 把 Expert 0,1 的输出发回各自的源 GPU
  所有 GPU 同时接收 → All-to-All

每层 MoE 需要 2 次 All-to-All!
```

### 5.3 EP 通信量分析

```
All-to-All 通信量 (per layer):
  Dispatch: 每个GPU发送 (T/E) × H × dtype_bytes 到每个目标GPU
  Gather:   同上 (结果发回)
  总通信量: 2 × T × H × dtype_bytes
  (与 EP 并行度 E 无关!)

对比:
  TP (per layer): 2 × T × H / P × dtype_bytes (AllReduce)
  EP (per layer): 2 × T × H × dtype_bytes (All-to-All)

EP 通信量 = TP 通信量 × P (EP的通信量更大!)
但: EP 的计算量也更大 (expert 计算)

通信/计算比:
  EP compute = T × K × (2 × H × FFN_hidden × 2) FLOPs
  EP comm    = 2 × T × H × 2 bytes (FP16)
  Ratio = 4 × K × FFN_hidden / (2 × H) × (bandwidth / FLOPS)

当 expert 计算量大时，通信占比低 → EP 高效
```

### 5.4 EP + TP 组合

```
大规模训练常用: EP + TP 组合

示例 (256 GPU, DeepSeek-V3):
  TP = 4 (张量并行, GPU 0-3 做 TP)
  EP = 64 (专家并行, 64 组 TP 组)
  每个 TP 组: 持有 256/64 = 4 个 expert

通信模式:
  TP 组内: AllReduce (NVLink, 高带宽)
  EP 组间: All-to-All (IB/RoCE, 较低带宽)

GPU 分配:
  [TP组0: GPU0-3, expert 0-3] ←All-to-All→ [TP组1: GPU4-7, expert 4-7] ...
  ↑ TP组内 AllReduce                                    ↑
  |_______________________________________________________|
```

## 6. 显存分析

### 6.1 Expert 参数存储

```
每个 Expert 的参数:
  FFN: 2 × hidden × intermediate × dtype_bytes
  (两个线性层: [hidden→intermediate] + [intermediate→hidden])

Mixtral 8x7B (hidden=4096, intermediate=14336):
  每层 expert 参数: 2 × 4096 × 14336 × 2 = ~235 MB (FP16)
  8个 expert: ~1.88 GB/层
  32层: ~60 GB (仅 MoE 层的 expert 参数)
  + Attention + Embedding: ~30 GB
  总计: ~90 GB (FP16)

对比 Dense LLaMA-2 70B:
  每层 FFN: 2 × 8192 × 28672 × 2 = ~940 MB
  80层: ~75 GB (FFN)
  + Attention + Embedding: ~65 GB
  总计: ~140 GB (FP16)

MoE 显存更大 (90GB vs 140GB) 但活跃计算量更小 (13B vs 70B)
→ 显存挑战: 需要更多GPU来存储所有expert参数
```

### 6.2 显存优化策略

```
1. Expert 参数量化:
   FP16 → FP8: 显存减半, DeepSeek-V3 验证可行
   FP16 → INT4: 显存 1/4, 推理质量损失小

2. Expert Offloading:
   不活跃的 expert 放 CPU 内存
   需要时异步加载到 GPU (需要提前预测路由结果)
   适用于推理场景 (batch小, 只用少量expert)

3. Shared Expert:
   所有 token 都经过 shared expert
   只需存储一份 (不需要 EP 切分)
   减少重复的通用知识存储

4. 细粒度 Expert (DeepSeek 方案):
   更多更小的 expert → 每个expert参数少
   灵活路由 → 更少的冗余参数
   但需要更快的 All-to-All (更多小消息)
```

## 7. vLLM 中的 MoE 实现

### 7.1 FusedMoE Kernel

```
传统实现 (naive):
  for token in tokens:
    for expert_id in top_k_experts[token]:
      output[token] += gate_weight * expert_ffn(input[token])

  问题: 大量小 GEMM, kernel launch 开销巨大
  8 expert × top-2 = 16 次 GEMM per token → 极低效

FusedMoE (vLLM):
  将同一 expert 的所有 token 拼成大 batch
  一次 GEMM 处理一个 expert 的所有 token

  Step 1: 按expert分组 token → token_permutation
  Step 2: 对每个expert做一次 batched GEMM
  Step 3: 按原始顺序 unpermutate 输出

  使用 Triton kernel 实现，避免 Python 循环
  → 减少 kernel launch 开销, 提高GPU利用率
```

### 7.2 Triton MoE Kernel

```
vLLM 的 MoE Triton kernel 优化:

1. Gate + Dispatch 融合:
   softmax → top-k → sort by expert → 在一个 kernel 中完成

2. Expert GEMM:
   使用 grouped GEMM (Triton)
   每个 expert 独立的 GEMM, 但共享 kernel launch
   自动处理不同 expert 的 batch 大小

3. Gather + Weighted Sum:
   unpermutate + 乘以 gate weight + sum
   融合在一个 kernel 中

性能: 比纯 PyTorch 实现 快 2-5x
```

### 7.3 Shared Expert 处理

```
vLLM 对 shared expert 的优化:

Shared Expert (不参与路由):
  所有 token 都经过 shared expert
  → 可以用标准 GEMM (不需要 permutation)
  → 天然高效 (大 batch, 无 All-to-All)

输出组合:
  output = routing_output + shared_expert_output
  简单的逐元素加法

显存: shared expert 只存一份, 不需要 EP 切分
```

### 7.4 Expert Parallel Load Balancing (EPLB)

```
vLLM 的运行时 expert 负载均衡:

问题: 预训练的模型 expert 使用频率不均
  某些 expert 被频繁使用 → 对应 GPU 过载
  某些 expert 很少被使用 → 对应 GPU 空闲

EPLB 方案:
  1. 统计每个 expert 的使用频率
  2. 将热门 expert 分配到不同 GPU
  3. 搭配冷门 expert 平衡负载

  类似于 bin-packing 问题
  可以在推理启动时做一次性优化
  也可以在运行时动态调整 (更复杂)
```

## 8. 训练与推理优化

### 8.1 通信与计算重叠

```
EP 的通信计算 overlap:

Pipeline:
  All-to-All Dispatch(第1层) → Compute(第1层) → All-to-All Gather(第1层)
                              ↓ overlap
                              All-to-All Dispatch(第2层)

关键: 第N层的Gather和第N+1层的Dispatch可以overlap
  因为 Gather 之后的数据可以直接用于下一层的路由计算

实现:
  使用 NCCL 的异步通信 (isend/irecv)
  在通信的同时做 expert 计算
  需要 2x 的显存 (当前层 + 通信缓冲区)
```

### 8.2 EP 策略选择

```
| 规模 | GPU数 | 推荐策略 | 原因 |
|------|-------|----------|------|
| 小 | 2-4 | DP | Expert不多, EP无意义 |
| 中 | 8-32 | DP + EP | EP减少显存, DP处理batch |
| 大 | 64-256 | DP + EP + TP | TP加速单expert, EP分布多expert |
| 超大 | 256+ | DP + EP + TP + PP | 全部并行策略组合 |

DeepSeek-V3 (2048 GPU):
  DP = 8
  EP = 64 (256 expert / 4 expert per GPU group)
  TP = 4
  PP = 1 (未使用, DeepSeek倾向用长序列+EP代替PP)
```

### 8.3 细粒度 Expert 的设计理念

```
传统: 8个大expert
  每个expert: FFN(4096→14336→4096)
  路由: 8选2, 组合数 C(8,2)=28
  灵活性: 有限

DeepSeek: 256个小expert
  每个expert: FFN(7168→2048→7168) (更小)
  路由: 256选8, 组合数 C(256,8) ≈ 4.2 × 10^13
  灵活性: 极高!

为什么更好?
  1. 更多选择 → 更精确的 token-expert 匹配
  2. 每个expert更小 → 更容易load balance
  3. 组合多样性 → 更强的表达能力
  4. 类似于 "更多的小 specialists > 几个大 generalists"
```

### 8.4 推理优化

```
1. Expert 缓存 (Expert Caching):
   缓存最近使用的 expert 参数在 GPU
   冷门 expert offload 到 CPU
   需要预测: 哪些 expert 会被下一步使用

2. 动态 Batch:
   MoE 的 batch 中不同 token 走不同 expert
   需要更智能的调度: 把使用相同 expert 的 token 排在一起

3. FP8 MoE:
   Expert 计算用 FP8 (显存减半, 带宽减半)
   Attention 和 Router 仍用 FP16 (精度敏感)
   DeepSeek-V3 验证: 质量损失可忽略

4. 推测解码 + MoE:
   Draft model: Dense 小模型 (快速)
   Verify model: MoE 大模型 (精确)
   MoE 的 decode 速度较慢 → 推测解码收益更大
```

## 9. 关键要点

1. **MoE 用参数量换质量** — 总参数远大于活跃参数，以接近大 Dense 模型的质量，用小 Dense 模型的计算成本

2. **路由是 MoE 的核心** — Top-K 路由是标准方法，Expert Choice 和 Group-Limited 是重要改进

3. **负载均衡是关键挑战** — 路由崩塌会严重降低 MoE 效果，Auxiliary Loss 是标准方案，DeepSeek 的 bias 方法更优

4. **EP 是 MoE 独有的并行维度** — 需要 All-to-All 通信，通信量较大但可以与计算 overlap

5. **细粒度 Expert + Shared Expert 是趋势** — DeepSeek 系列验证了更多更小 expert + 1 个共享 expert 的设计优于少量大 expert

6. **vLLM 通过 FusedMoE kernel 优化推理** — 将 gate + dispatch + GEMM + gather 融合，避免大量小 kernel launch

## 参考

- 论文: [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (2024)
- 论文: [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model](https://arxiv.org/abs/2405.04434)
- 论文: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- 论文: [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) (2022)
- 论文: [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) (2022)
- 论文: [DBRX: An Open, General-Purpose LLM](https://www.databricks.com/blog/announcing-dbrx) (2024)
- 博客: [Mixture of Experts Explained](https://huggingface.co/blog/moe) (HuggingFace)
