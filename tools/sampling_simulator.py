"""LLM 采样方法模拟器 — Temperature / Top-K / Top-P / Beam Search / Penalty

模拟 LLM 推理中的各种采样策略:
1. Temperature 缩放: 观察温度对概率分布的影响
2. Top-K 采样: 只保留概率最高的 K 个 token
3. Top-P (Nucleus) 采样: 只保留累积概率达到 P 的 token
4. 重复惩罚: repetition / frequency / presence penalty
5. Beam Search vs Sampling: 对比不同解码策略
6. Min-P 采样: 动态阈值过滤

使用方法:
    python sampling_simulator.py   # CPU 可运行
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TokenLogit:
    """Token 及其 logit。"""
    token_id: int
    token: str
    logit: float


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """带 temperature 的 softmax。"""
    scaled = logits / max(temperature, 1e-8)
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / exp_scaled.sum()


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Top-K 过滤: 只保留概率最高的 K 个，其余设为 0。"""
    if k >= len(probs):
        return probs
    indices = np.argsort(probs)[::-1]
    filtered = np.zeros_like(probs)
    filtered[indices[:k]] = probs[indices[:k]]
    # 重新归一化
    total = filtered.sum()
    if total > 0:
        filtered /= total
    return filtered


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Top-P (Nucleus) 过滤: 累积概率达到 P 后截断。"""
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cumulative = np.cumsum(sorted_probs)
    # 找到累积概率刚超过 p 的位置
    cutoff = np.searchsorted(cumulative, p) + 1
    cutoff = min(cutoff, len(probs))

    filtered = np.zeros_like(probs)
    filtered[sorted_indices[:cutoff]] = probs[sorted_indices[:cutoff]]

    total = filtered.sum()
    if total > 0:
        filtered /= total
    return filtered


def min_p_filter(probs: np.ndarray, min_p: float) -> np.ndarray:
    """Min-P 过滤: 只保留概率 >= max_prob * min_p 的 token。"""
    max_prob = probs.max()
    threshold = max_prob * min_p
    filtered = np.where(probs >= threshold, probs, 0.0)
    total = filtered.sum()
    if total > 0:
        filtered /= total
    return filtered


def repetition_penalty(logits: np.ndarray, token_ids: List[int],
                       penalty: float = 1.0) -> np.ndarray:
    """Repetition penalty: 对已出现的 token 施加惩罚。

    logit > 0: logit /= penalty
    logit < 0: logit *= penalty
    """
    if penalty == 1.0:
        return logits
    modified = logits.copy()
    for tid in token_ids:
        if modified[tid] > 0:
            modified[tid] /= penalty
        else:
            modified[tid] *= penalty
    return modified


def frequency_presence_penalty(logits: np.ndarray, token_ids: List[int],
                                freq_penalty: float = 0.0,
                                pres_penalty: float = 0.0) -> np.ndarray:
    """Frequency + Presence penalty (OpenAI 风格)。

    frequency: 惩罚量与 token 出现次数成正比
    presence: 只要出现过就惩罚固定量
    """
    from collections import Counter
    counts = Counter(token_ids)
    modified = logits.copy()
    for tid, count in counts.items():
        modified[tid] -= freq_penalty * count + pres_penalty
    return modified


def sample_token(probs: np.ndarray, rng: np.random.Generator) -> int:
    """从概率分布中采样一个 token。"""
    return rng.choice(len(probs), p=probs)


def greedy_decode(probs: np.ndarray) -> int:
    """贪心解码: 选择概率最高的 token。"""
    return np.argmax(probs)


def beam_search_step(beam_scores: np.ndarray, beam_tokens: np.ndarray,
                     beam_width: int) -> np.ndarray:
    """Beam Search 一步: 从所有候选中选出 top beam_width 个。"""
    # beam_scores: (beam_width, vocab_size)
    # 展平后选 top-k
    flat_scores = beam_scores.flatten()
    top_indices = np.argsort(flat_scores)[::-1][:beam_width]
    return top_indices


# ============================================================
# 模拟实验
# ============================================================

def experiment_temperature():
    """实验 1: Temperature 对概率分布的影响。"""
    print("\n" + "=" * 60)
    print("实验 1: Temperature 缩放效果")
    print("=" * 60)

    # 模拟一个 10-token 的 logits 分布
    tokens = ["the", "a", "this", "that", "some", "every", "any", "no", "all", "his"]
    logits = np.array([3.0, 2.5, 1.8, 1.2, 0.8, 0.3, -0.2, -0.5, -1.0, -1.5])

    print(f"\n  原始 logits: {dict(zip(tokens, logits.round(1)))}")

    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    print(f"\n  {'Token':<8} ", end="")
    for t in temperatures:
        print(f"{'T='+str(t):>7}", end="")
    print()
    print(f"  {'-'*8} " + "-" * (8 * len(temperatures)))

    for i, token in enumerate(tokens):
        print(f"  {token:<8} ", end="")
        for t in temperatures:
            probs = softmax(logits, t)
            print(f" {probs[i]*100:>6.1f}%", end="")
        print()

    # Shannon entropy 随温度变化
    print(f"\n  Temperature → Shannon Entropy:")
    print(f"  {'T':>6} {'Entropy':>10} {'Top-1':>8} {'Top-3':>8} {'有效token':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

    for t in temperatures:
        probs = softmax(logits, t)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        top1 = np.max(probs)
        top3 = np.sum(np.sort(probs)[-3:])
        effective = np.sum(probs > 0.01)
        print(f"  {t:>6.1f} {entropy:>10.3f} {top1*100:>7.1f}% {top3*100:>7.1f}% {effective:>10}")

    print("""
关键洞察:
  - T→0: 分布趋近 one-hot (贪心)，entropy → 0
  - T=1: 原始分布，不做缩放
  - T→∞: 分布趋近均匀，entropy → log(vocab_size)
  - 实用范围: T=0.3-1.0 (太低太确定性，太高太随机)
  - 创意任务: T=0.7-1.0; 事实性任务: T=0.1-0.3
    """)


def experiment_top_k_top_p():
    """实验 2: Top-K vs Top-P 过滤对比。"""
    print("=" * 60)
    print("实验 2: Top-K vs Top-P (Nucleus) 采样过滤")
    print("=" * 60)

    tokens = ["the", "a", "this", "that", "some", "every", "any", "no", "all", "his"]
    logits = np.array([3.0, 2.5, 1.8, 1.2, 0.8, 0.3, -0.2, -0.5, -1.0, -1.5])
    probs = softmax(logits, 0.7)

    # Top-K 过滤
    print(f"\n  Top-K 过滤 (T=0.7):")
    print(f"  {'Token':<8} {'原始':>8} ", end="")
    for k in [1, 3, 5, 7]:
        print(f"{'K='+str(k):>8}", end="")
    print()
    print(f"  {'-'*8} {'-'*8} " + "-" * 32)

    for i, token in enumerate(tokens):
        print(f"  {token:<8} {probs[i]*100:>7.1f}% ", end="")
        for k in [1, 3, 5, 7]:
            filtered = top_k_filter(probs, k)
            print(f" {filtered[i]*100:>6.1f}%", end="")
        print()

    # Top-P 过滤
    print(f"\n  Top-P (Nucleus) 过滤 (T=0.7):")
    print(f"  {'Token':<8} {'原始':>8} ", end="")
    for p in [0.3, 0.5, 0.7, 0.9]:
        print(f"{'P='+str(p):>8}", end="")
    print()
    print(f"  {'-'*8} {'-'*8} " + "-" * 32)

    for i, token in enumerate(tokens):
        print(f"  {token:<8} {probs[i]*100:>7.1f}% ", end="")
        for p in [0.3, 0.5, 0.7, 0.9]:
            filtered = top_p_filter(probs, p)
            print(f" {filtered[i]*100:>6.1f}%", end="")
        print()

    # 对比: 保留的 token 数
    print(f"\n  保留 token 数对比:")
    print(f"  {'方法':<15} {'保留数':>8} {'Top-1':>8} {'Entropy':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*10}")

    for k in [1, 2, 3, 5, 8, 10]:
        filtered = top_k_filter(probs, k)
        n_keep = np.sum(filtered > 0)
        entropy = -np.sum(filtered * np.log2(filtered + 1e-10))
        print(f"  Top-K={k:<9} {n_keep:>8} {filtered.max()*100:>7.1f}% {entropy:>10.3f}")

    print()
    for p in [0.2, 0.4, 0.6, 0.8, 0.95, 1.0]:
        filtered = top_p_filter(probs, p)
        n_keep = np.sum(filtered > 0)
        entropy = -np.sum(filtered * np.log2(filtered + 1e-10))
        print(f"  Top-P={p:<8.1f} {n_keep:>8} {filtered.max()*100:>7.1f}% {entropy:>10.3f}")

    print("""
关键洞察:
  - Top-K: 固定保留 K 个 token，简单但不够灵活
  - Top-P: 自适应保留，概率集中时少保留，分散时多保留
  - Top-P=0.9 通常比 Top-K=5 效果好（自适应阈值）
  - 推荐组合: Temperature=0.7 + Top-P=0.9
    """)


def experiment_min_p():
    """实验 3: Min-P 动态阈值过滤。"""
    print("=" * 60)
    print("实验 3: Min-P 动态阈值过滤")
    print("=" * 60)

    # 两种分布: 集中 vs 分散
    tokens = [f"t{i}" for i in range(10)]

    # 集中分布
    concentrated = np.array([0.50, 0.20, 0.10, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01])
    # 分散分布
    spread = np.array([0.15, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07])

    print(f"\n  分布 1 (集中): Top-1={concentrated.max()*100:.0f}%")
    print(f"  分布 2 (分散): Top-1={spread.max()*100:.0f}%")

    print(f"\n  {'Min-P':>8} {'集中保留':>8} {'分散保留':>8} {'集中Top-1':>10} {'分散Top-1':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for mp in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
        f1 = min_p_filter(concentrated, mp)
        f2 = min_p_filter(spread, mp)
        print(f"  {mp:>8.2f} {np.sum(f1>0):>8} {np.sum(f2>0):>8} "
              f"{f1.max()*100:>9.1f}% {f2.max()*100:>9.1f}%")

    print("""
关键洞察:
  - Min-P 阈值 = max_prob * min_p_factor
  - 集中分布时阈值高，保留少量 token
  - 分散分布时阈值低，保留更多 token
  - 比 Top-K 更灵活（自适应），比 Top-P 更直观
  - 推荐值: Min-P=0.05-0.1
    """)


def experiment_penalties():
    """实验 4: 重复惩罚机制对比。"""
    print("=" * 60)
    print("实验 4: 重复惩罚机制 (Repetition/Frequency/Presence)")
    print("=" * 60)

    tokens = ["the", "a", "is", "was", "and", "to", "of", "in", "it", "for"]
    vocab_size = len(tokens)
    logits = np.array([3.0, 2.5, 1.8, 1.2, 0.8, 0.3, -0.2, -0.5, -1.0, -1.5])

    # 模拟已生成的 token: "the the a the is"
    generated = [0, 0, 1, 0, 2]  # token IDs
    gen_tokens = [tokens[i] for i in generated]
    print(f"\n  已生成: {' '.join(gen_tokens)}")
    print(f"  'the' 出现 3 次, 'a' 出现 1 次, 'is' 出现 1 次")

    # Repetition Penalty
    print(f"\n  Repetition Penalty (logits 缩放):")
    print(f"  {'Token':<8} {'原始':>8} ", end="")
    for rp in [1.0, 1.2, 1.5, 2.0, 3.0]:
        print(f"{'RP='+str(rp):>8}", end="")
    print()
    print(f"  {'-'*8} {'-'*8} " + "-" * 40)

    for i, token in enumerate(tokens):
        print(f"  {token:<8} {logits[i]:>8.2f} ", end="")
        for rp in [1.0, 1.2, 1.5, 2.0, 3.0]:
            mod = repetition_penalty(logits.copy(), generated, rp)
            print(f" {mod[i]:>7.2f}", end="")
        print()

    # Frequency + Presence Penalty
    print(f"\n  Frequency + Presence Penalty (logits 减法):")
    print(f"  {'Penalty':>15} {'the(3x)':>10} {'a(1x)':>10} {'is(1x)':>10} {'was(0x)':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    configs = [
        ("freq=0.0, pres=0.0", 0.0, 0.0),
        ("freq=0.5, pres=0.0", 0.5, 0.0),
        ("freq=1.0, pres=0.0", 1.0, 0.0),
        ("freq=0.0, pres=1.0", 0.0, 1.0),
        ("freq=0.5, pres=0.5", 0.5, 0.5),
        ("freq=1.0, pres=1.0", 1.0, 1.0),
    ]

    for name, fp, pp in configs:
        mod = frequency_presence_penalty(logits.copy(), generated, fp, pp)
        print(f"  {name:>15} {mod[0]:>10.2f} {mod[1]:>10.2f} {mod[2]:>10.2f} {mod[3]:>10.2f}")

    print("""
关键洞察:
  - Repetition Penalty: 乘法惩罚 (logits 缩放)，>1 惩罚，值越大越强
    - logit > 0: logit /= penalty (缩小正 logit)
    - logit < 0: logit *= penalty (放大负 logit)
  - Frequency Penalty: 减法惩罚，与出现次数成正比
  - Presence Penalty: 减法惩罚，只要出现过就惩罚固定量
  - 推荐: RP=1.1-1.3 或 freq=0.2-0.6 + pres=0.2-0.6
  - 过高的惩罚会导致输出不连贯
    """)


def experiment_sampling_vs_greedy():
    """实验 5: Sampling vs Greedy vs Beam Search 对比。"""
    print("=" * 60)
    print("实验 5: 解码策略对比 (Greedy / Sampling / Beam Search)")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # 模拟 vocab=20, 生成 20 个 token
    vocab_size = 20
    seq_len = 20
    words = [f"w{i}" for i in range(vocab_size)]

    # 每步的 logits (模拟 LLM 输出)
    np.random.seed(42)
    all_logits = rng.standard_normal((seq_len, vocab_size))

    # 不同策略生成
    def generate_greedy(all_logits):
        return [np.argmax(softmax(l)) for l in all_logits]

    def generate_sampling(all_logits, temperature=1.0, top_k=0, top_p=1.0):
        tokens = []
        for logits in all_logits:
            probs = softmax(logits, temperature)
            if top_k > 0:
                probs = top_k_filter(probs, top_k)
            if top_p < 1.0:
                probs = top_p_filter(probs, top_p)
            tokens.append(rng.choice(len(probs), p=probs))
        return tokens

    def generate_beam(all_logits, beam_width=3):
        """简化版 beam search。"""
        # 初始化
        beams = [([], 0.0)]  # (token_list, cumulative_log_prob)

        for step, logits in enumerate(all_logits):
            new_beams = []
            for seq, score in beams:
                log_probs = logits - np.max(logits)  # 数值稳定
                log_probs = log_probs - np.log(np.exp(log_probs).sum())  # log_softmax

                for token_id in range(len(logits)):
                    new_seq = seq + [token_id]
                    new_score = score + log_probs[token_id]
                    new_beams.append((new_seq, new_score))

            # 保留 top beam_width
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        return beams[0][0]  # 返回最佳 beam

    greedy_tokens = generate_greedy(all_logits)
    sampling_t07 = generate_sampling(all_logits, 0.7)
    sampling_t10 = generate_sampling(all_logits, 1.0)
    sampling_t07_topk5 = generate_sampling(all_logits, 0.7, top_k=5)
    sampling_t07_topp09 = generate_sampling(all_logits, 0.7, top_p=0.9)
    beam3_tokens = generate_beam(all_logits, 3)
    beam5_tokens = generate_beam(all_logits, 5)

    strategies = [
        ("Greedy", greedy_tokens),
        ("T=0.7", sampling_t07),
        ("T=1.0", sampling_t10),
        ("T=0.7+K=5", sampling_t07_topk5),
        ("T=0.7+P=0.9", sampling_t07_topp09),
        ("Beam=3", beam3_tokens),
        ("Beam=5", beam5_tokens),
    ]

    print(f"\n  生成的 token 序列:")
    print(f"  {'策略':<15} {'序列':>50} {'多样性':>8}")
    print(f"  {'-'*15} {'-'*50} {'-'*8}")

    for name, tokens_list in strategies:
        seq_str = " ".join(words[t] for t in tokens_list)
        unique = len(set(tokens_list))
        print(f"  {name:<15} {seq_str:>50} {unique:>8}/{seq_len}")

    # 多样性分析: 多次采样的差异
    print(f"\n  多次采样多样性 (T=0.7, 20次):")
    diversity_counts = {}
    for _ in range(20):
        all_logits_rnd = rng.standard_normal((seq_len, vocab_size))
        tokens_gen = generate_sampling(all_logits_rnd, 0.7)
        for t in tokens_gen:
            diversity_counts[t] = diversity_counts.get(t, 0) + 1

    print(f"  不同 token 使用数: {len(diversity_counts)}/{vocab_size}")
    top_tokens = sorted(diversity_counts.items(), key=lambda x: -x[1])[:5]
    print(f"  Top-5 使用: {', '.join(f'{words[t]}({c}次)' for t, c in top_tokens)}")

    print("""
关键洞察:
  - Greedy: 总选最高概率 token，输出确定性，但容易重复和无趣
  - Sampling: 引入随机性，输出更多样，T 控制随机程度
  - Beam Search: 保留多条路径，选最优序列
    - 优点: 全局更优（考虑未来 token）
    - 缺点: 仍倾向高频模式，多样性不足
  - 实际推理: 通常用 Temperature + Top-P 组合
  - Beam Search 更适合翻译/摘要等确定性任务
    """)


def experiment_sampling_distribution():
    """实验 6: 采样分布可视化。"""
    print("=" * 60)
    print("实验 6: 采样分布分析 (采样 1000 次统计)")
    print("=" * 60)

    tokens = ["the", "a", "this", "that", "some"]
    logits = np.array([3.0, 2.5, 1.8, 1.2, 0.5])

    rng = np.random.default_rng(42)
    n_samples = 10000

    configs = [
        ("Greedy", 0.01, 0, 1.0),
        ("T=0.3", 0.3, 0, 1.0),
        ("T=0.7", 0.7, 0, 1.0),
        ("T=1.0", 1.0, 0, 1.0),
        ("T=0.7+K=3", 0.7, 3, 1.0),
        ("T=0.7+P=0.9", 0.7, 0, 0.9),
        ("T=1.5", 1.5, 0, 1.0),
    ]

    print(f"\n  {'策略':<15} ", end="")
    for token in tokens:
        print(f" {token:>8}", end="")
    print(f"  {'Entropy':>8}")
    print(f"  {'-'*15} " + "-" * (5 * 9) + "  " + "-" * 8)

    for name, temp, k, p in configs:
        probs = softmax(logits, temp)
        if k > 0:
            probs = top_k_filter(probs, k)
        if p < 1.0:
            probs = top_p_filter(probs, p)

        # 采样统计
        samples = rng.choice(len(probs), size=n_samples, p=probs)
        counts = np.array([np.sum(samples == i) for i in range(len(tokens))])
        freqs = counts / n_samples

        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        print(f"  {name:<15} ", end="")
        for i in range(len(tokens)):
            bar = "█" * int(freqs[i] * 30)
            print(f" {freqs[i]*100:>6.1f}%", end="")
        print(f"  {entropy:>7.2f}")

    print(f"""
关键洞察:
  - T=0.3 (低温度): 93%+ 集中在 "the"，非常确定性
  - T=0.7 (中温度): 分布适中，"the" 和 "a" 共享大部分概率
  - T=1.0 (默认): 保持模型原始分布
  - T=1.5 (高温度): 分布更均匀，适合创意生成
  - Top-K/P 在 Temperature 基础上进一步过滤低概率 token
    """)


def main():
    print("=" * 60)
    print("LLM 采样方法模拟器")
    print("=" * 60)

    experiment_temperature()
    experiment_top_k_top_p()
    experiment_min_p()
    experiment_penalties()
    experiment_sampling_vs_greedy()
    experiment_sampling_distribution()

    # 总结
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
LLM 采样方法核心知识:

1. Temperature:
   - 控制概率分布的"锐度"
   - T→0: 贪心 (确定性), T=1: 原始分布, T→∞: 均匀 (随机)
   - 推荐: 事实性 T=0.1-0.3, 创意 T=0.7-1.0

2. Top-K:
   - 只保留概率最高的 K 个 token
   - 简单但不灵活 (分布集中时浪费, 分散时不够)
   - 通常 K=40-100

3. Top-P (Nucleus Sampling):
   - 自适应保留累积概率达 P 的 token
   - 比固定 K 更灵活
   - 推荐 P=0.9-0.95

4. Min-P:
   - 动态阈值: 只保留 prob >= max_prob × min_p 的 token
   - 比固定 K 或 P 更优的自适应策略
   - 推荐 min_p=0.05-0.1

5. 重复惩罚:
   - Repetition Penalty: 乘法缩放 logits (RP=1.1-1.3)
   - Frequency Penalty: 按出现次数线性惩罚
   - Presence Penalty: 出现过就惩罚
   - 过高惩罚导致不连贯

6. Beam Search:
   - 保留多条候选路径，选全局最优
   - 适合翻译/摘要等确定性任务
   - 对话/创意场景更适合 sampling

7. 常用组合:
   - Chat: T=0.7 + Top-P=0.9
   - Code: T=0.2 + Top-P=0.95
   - Creative: T=1.0 + Top-P=0.9 + Min-P=0.05
   - Translation: Beam=4-5
    """)


if __name__ == "__main__":
    main()
