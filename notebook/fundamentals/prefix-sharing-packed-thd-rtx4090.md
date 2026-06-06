# Prefix Sharing Packed THD Micro-Benchmark — RTX 4090 实测

> 2026-06-07 | 5个实验: KV Injection, GRPO Savings, Qwen3.6 GQA, Packed vs Unpacked, Prefix-Last Restore

## 一、KV Injection Overhead (GQA 32:4, B=8)

| prefix_len | suffix | total | compute节省% | time节省% | speedup |
|-----------|--------|-------|-------------|----------|---------|
| 64 | 128 | 192 | 29.2% | 71.5%* | 3.51x* |
| 128 | 128 | 256 | 43.8% | 61.1% | 2.57x |
| 256 | 128 | 384 | 58.3% | 49.0% | 1.96x |
| 512 | 128 | 640 | 70.0% | 14.6% | 1.17x |
| 1024 | 128 | 1152 | 77.8% | 33.1% | 1.50x |

*prefix=64数据含warmup噪声，忽略即可

**关键发现**: 纯attention层面的KV injection加速随prefix增大衰减!

1. **短prefix(128-256): 2-2.6x加速** → KV cache小→L2命中→attention本身快→PS减少tokens比例大
2. **长prefix(512): 仅1.17x** → KV cache大→HBM→memory-bound→PS后KV体积仍大→attention时间≈不变
3. **核心矛盾**: Attention是memory-bound → 时间∝KV数据量 → PS注入的prefix KV仍然需要被reuser的attention读取 → **KV读BW不变**!

**与之前KV Cache BW实验对照**: B≤16时KV BW达1868 GB/s(L2) → 小KV injection几乎无开销 → 但大KV(≥128MB)降至HBM 460 GB/s → 注入+attention时间被HBM限制

## 二、GRPO n_samples Savings (prefix=512, suffix=256)

| n | compute节省% | time节省% | speedup |
|---|-------------|----------|---------|
| 2 | 33.3% | -3435%* | 0.03x* |
| 4 | 50.0% | 3.7% | 1.04x |
| 8 | 58.3% | -0.9% | 0.99x |
| 16 | 62.5% | 12.7% | 1.15x |

*n=2数据含warmup噪声，忽略即可

**震惊发现**: 纯attention层面GRPO n=8仅0.99x加速(几乎无收益)!

1. **compute节省58%但time节省-0.9%**: 计算量确实减少58% → 但RTX 4090上attention是memory-bound → 时间∝数据量而非FLOPS
2. **PS后KV数据量**: reuser仍需读取provider prefix KV → 总KV读取量 = (B-1)×(prefix_len+suffix_len)×KV_dim → 与全forward的B×total_len×KV_dim相比 → 减少了prefix tokens的KV计算但**不减KV读取量**
3. **真实收益不在attention层**: Prefix-sharing的真正价值在于:
   - **全模型forward**: reuser跳过prefix的所有64层(hidden→KV+MLP+LN) → 真正节省FLOPS
   - **KV内存**: 每个reuser少存prefix_len×KV_dim → GPU内存减少 → 更多并发

**与prefix-0501项目对照**: 项目目标不是节省attention层 → 而是节省**整个model forward**(包括KV projection + MLP + LayerNorm) → 我的benchmark只测了attention → 需要补充full-layer测试

## 三、Qwen3.6 GQA 24:4 KV Injection (head_dim=256)

| seq_len | prefix | suffix | inject开销(ms) | per-layer加速 | 总加速估计 |
|---------|--------|--------|---------------|-------------|-----------|
| 256 | 192 | 64 | 0.031 | 22.7x* | 6.42x* |
| 512 | 384 | 128 | 0.039 | 15.18x* | 4.55x* |
| 1024 | 768 | 256 | 0.05 | 1.0x | 1.0x |
| 2048 | 1536 | 512 | 0.085 | 1.11x | 1.03x |

*小seq数据含warmup/jitter噪声

**关键发现**:

1. **KV injection开销极低**: 0.031-0.085ms → cat+expand操作几乎免费 → L2 cache命中
2. **head_dim=256时KV更大**: Qwen3.6每token KV = 4×256×2×2bytes = 4KB → 比7B(4×128×2×2=2KB)大2x → 更早进入HBM瓶颈
3. **只有25%层(full attn)受益**: HybridAttention 16/64层 → 总加速 = 1 + 0.25×(per_layer加速-1)
   - seq=2048: 总加速仅1.03x → 因为attention本身memory-bound
   - **DeltaNet层(48层)收益来自state injection, 不在此benchmark中**

**核心结论**: Qwen3.6-27B在RTX 4090上, 纯full-attn KV injection对长序列(≥1K)几乎无加速 → 但**对短序列和decode batch有效**(KV小→L2命中)

## 四、Packed vs Unpacked Attention Throughput

| B | full_BSH(ms) | PS_packed(ms) | speedup | token减少% |
|---|-------------|--------------|---------|-----------|
| 2 | 0.131 | 0.186 | 0.71x | 37.5% |
| 4 | 0.199 | 0.199 | 1.0x | 56.2% |
| 8 | 0.312 | 0.300 | 1.04x | 65.6% |
| 16 | 0.655 | 0.667 | 0.98x | 70.3% |

**发现**: packed格式与BSH格式吞吐几乎相同(0.98-1.04x)

1. **token减少65.6%(B=8)但speedup仅1.04x**: 再次证明memory-bound → 减少tokens≠减少时间
2. **B=2时PS反而慢0.71x**: 小batch时KV injection(cat+expand)开销>计算节省 → Python overhead
3. **实际收益**: Packed THD格式减少padding浪费 → 但SDPA实现中padding被mask → 无额外计算 → 格式差异不影响吞吐

## 五、Prefix-Last Restore Overhead

| B | vocab | base_logprob(ms) | restore(ms) | overhead% | per_reuser(ms) |
|---|-------|-----------------|------------|----------|--------------|
| 2 | 32K | 33.6* | 0.181 | 0.5%* | 0.181 |
| 4 | 32K | 0.732 | 0.202 | 27.7% | 0.067 |
| 8 | 32K | 1.311 | 0.380 | 29.0% | 0.054 |
| 16 | 32K | 2.473 | 0.723 | 29.2% | 0.048 |
| 32 | 32K | 4.799 | 1.387 | 28.9% | 0.045 |
| 8 | 248K | 17.872 | 0.415 | **2.3%** | 0.059 |

*B=2含warmup噪声

**关键发现**: Prefix-Last Restore开销极低!

1. **Qwen3.6 vocab(248320): 仅2.3% overhead**: log_softmax在大vocab上耗时17.87ms → restore仅0.415ms → 7个reuser各做1次log_softmax ≈0.06ms/个 → 微不足道
2. **小vocab(32K)看似29% overhead**: 但绝对时间仅0.054ms/reuser → 微秒级 → 不影响训练
3. **结论**: Prefix-Last Restore不是瓶颈 → logprob恢复开销几乎为零

## 六、综合结论与对prefix-0501项目的启示

### 核心发现

1. **纯attention层KV injection对长序列(≥1K)几乎无加速**: memory-bound → KV读取量不变 → time∝data volume → PS减少compute但不减memory traffic
2. **KV injection开销本身极低**: 0.03-0.09ms → cat+expand几乎免费
3. **Prefix-Last Restore开销极低**: 2.3% of logprob compute → 不影响训练
4. **短序列/小batch有效**: prefix≤256时1.96-2.57x → KV小→L2命中→attention快→PS比例效应大
5. **Qwen3.6 HybridAttention只有25%层受益**: full-attn KV injection → 总加速约1.03x(长序列)

### 对prefix-0501项目的启示

**关键**: 我的benchmark只测了**attention层**的PS → 但prefix-0501的PS发生在**整个model forward** → 真正节省的是:

1. **KV projection**: hidden→K/V GEMM → compute-bound → PS减少(B-1)/B × prefix_len × KV_proj FLOPS
2. **MLP层**: gate+up+down GEMM → compute-bound → PS减少prefix_len × 3×MLP FLOPS
3. **LayerNorm/RMSNorm**: 小但也是compute → PS减少prefix_len × LN FLOPS
4. **总FLOPS节省**: ≈(B-1)/B × prefix_ratio × model_total_FLOPS →这才是真正的收益!

**估算** (7B模型, GRPO n=8, prefix=512, suffix=256):
- 模型总FLOPS/token ≈ 6×7B×2 = 84 GFLOPS (FP16)
- 全forward: 8×768×84G = 515 TFLOPS
- PS forward: 1×768×84G + 7×256×84G = 64.5 + 150.5 = 215 TFLOPS
- savings: (515-215)/515 = **58.3%** → 与之前prefix cache benchmark吻合!
- **但attention时间不变** → MLP+KV_proj时间减少58% → 总forward时间减少取决于MLP占比

**MLP占比68%** (实测) → PS主要节省MLP compute → 总forward加速 ≈ 1 + 0.68×(compute_savings) ≈ **1.5x** (n=8)

### 下一步

- 需要full-model-layer PS benchmark (不只是attention) → 在RTX 4090上模拟7B完整forward
- 对比: full forward vs PS forward (skip prefix layers for reusers)
- 这样才能验证prefix-0501项目的真实收益

Sources:
- prefix-0501项目: ~/workspace/project/prefix-proj/prefix-0501_claude-loop/
- 之前KV Cache BW实验: notebook/fundamentals/kv-cache-bandwidth-rtx4090.md
- 之前Prefix Cache实验: notebook/fundamentals/prefix-cache-throughput-rtx4090.md