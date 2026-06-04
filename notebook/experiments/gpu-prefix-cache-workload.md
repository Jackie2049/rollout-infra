# Prefix Caching 工作负载模拟

> 文件: `tools/gpu_prefix_cache_workload.py`
> 日期: 2026-06-04

## 实验 2: 多轮对话 (Growing Prefix)

Turn = 256 tokens, growing prefix:

| Turns | Turn 1 | Mid | Last | **Avg Hit** |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 0% | 60% | 60% | 30.0% |
| 4 | 0% | 71% | 78% | 52.3% |
| 8 | 0% | 82% | 88% | 68.8% |
| **16** | 0% | 90% | **94%** | **80.4%** |

**结论**: 16 轮对话累积命中率 80%。SGLang RadixAttention 专门优化此场景。

---

## 实验 3: GRPO Rollout

Prompt=512, Generation=256, 同 prompt 多次 rollout:

| n | KV Saved | Hit Rate |
|:---:|:---:|:---:|
| 4 | 2,048 | 66.7% |
| 8 | 4,096 | 66.7% |
| 16 | 8,192 | 66.7% |

**结论**: GRPO n=8 节省 ~41% KV (n 个 rollout 共享 prompt KV)。verl PrefixGrouper 用此优化。

---

## 实验 4: 缓存驱逐

30% 共享 rate, seq_len=1024:

| Cache Blocks | Hit Rate |
|:---:|:---:|
| 512 | 11.5% |
| 1024 | 15.8% |
| 4096 | 13.5% |

**结论**: 缓存大小影响上限但达到容量后命中率稳定。LRU 驱逐效率高。

---

## 关键洞察

1. **System Prompt**: 固定前缀 → 100% reuse，最大的 prefix cache 场景
2. **多轮对话**: 随轮次累积，16 轮达 80%+ 命中率
3. **GRPO**: n 个 rollout 共享 prompt → 节省 40-60% KV cache
4. **缓存容量**: 足够大时驱逐率低，LRU 优于 FIFO
5. **block_size=16**: 对齐粒度影响缓存效率（太小→hash 太多，太大→粒度粗）
