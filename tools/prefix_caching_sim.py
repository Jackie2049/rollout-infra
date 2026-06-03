"""Prefix Caching 模拟器 — Block 共享效率与策略对比

模拟不同 Prefix Caching 策略的 KV Cache 共享效率:
1. No Caching: 每个请求独立计算所有 KV Cache
2. Hash-based (vLLM): Block 级哈希链，自动检测共享前缀
3. Trie-based (SGLang): Radix tree，变长前缀匹配
4. Group-based (verl): 请求分组，同 prompt 共享 KV Cache

使用方法:
    python prefix_caching_sim.py   # CPU 可运行
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import hashlib


@dataclass
class Request:
    """推理请求。"""
    req_id: str
    prompt_tokens: List[int]      # System prompt tokens (可共享)
    input_tokens: List[int]       # User input tokens (不共享)
    @property
    def total_tokens(self):
        return len(self.prompt_tokens) + len(self.input_tokens)


# ============================================================
# 无缓存基线
# ============================================================

class NoCaching:
    """无缓存: 每个请求独立计算所有 KV Cache。"""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.total_blocks_allocated = 0
        self.blocks_per_request: Dict[str, int] = {}

    def process_request(self, req: Request) -> int:
        """处理请求，返回分配的 block 数。"""
        n_blocks = (req.total_tokens + self.block_size - 1) // self.block_size
        self.total_blocks_allocated += n_blocks
        self.blocks_per_request[req.req_id] = n_blocks
        return n_blocks

    def free_request(self, req_id: str):
        """释放请求。"""
        pass  # 无缓存，直接释放


# ============================================================
# Hash-based Prefix Caching (vLLM V1)
# ============================================================

class HashBasedCaching:
    """vLLM 风格的哈希链 Prefix Caching:

    Block 级哈希链: hash = H(parent_hash, block_tokens, extra_keys)
    相同 hash 的 block 可以直接复用，不需要重新计算。
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.cache: Dict[str, int] = {}  # hash → block_id
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_blocks_allocated = 0
        self.blocks_per_request: Dict[str, int] = {}
        self.cached_blocks = 0

    def _compute_block_hash(self, parent_hash: str, tokens: List[int]) -> str:
        """计算 block hash (链式)。"""
        data = f"{parent_hash}:{','.join(map(str, tokens))}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def process_request(self, req: Request) -> Tuple[int, int]:
        """处理请求，返回 (新分配 blocks, 命中 blocks)。"""
        all_tokens = req.prompt_tokens + req.input_tokens
        new_blocks = 0
        hit_blocks = 0
        parent_hash = "ROOT"

        for i in range(0, len(all_tokens), self.block_size):
            block_tokens = all_tokens[i:i + self.block_size]
            # 补齐不满的 block
            if len(block_tokens) < self.block_size:
                block_tokens = block_tokens + [0] * (self.block_size - len(block_tokens))

            block_hash = self._compute_block_hash(parent_hash, block_tokens)

            if block_hash in self.cache:
                # Cache hit: 复用已有 block
                hit_blocks += 1
                self.cache_hits += 1
            else:
                # Cache miss: 新计算并存入 cache
                new_blocks += 1
                self.cache_misses += 1
                block_id = len(self.cache)
                self.cache[block_hash] = block_id
                self.cached_blocks += 1

            parent_hash = block_hash

        self.total_blocks_allocated += new_blocks
        self.blocks_per_request[req.req_id] = new_blocks
        return new_blocks, hit_blocks

    def free_request(self, req_id: str):
        """释放请求 (blocks 留在 cache 中)。"""
        pass


# ============================================================
# Trie-based Prefix Caching (SGLang RadixAttention)
# ============================================================

class TrieNode:
    """Radix tree 节点。"""
    def __init__(self):
        self.children: Dict[int, 'TrieNode'] = {}  # token_id → child
        self.block_count: int = 0  # 该节点对应的 block 数
        self.ref_count: int = 0    # 引用计数
        self.value: Optional[int] = None  # block_id


class RadixTreeCaching:
    """SGLang 风格的 Radix Tree Prefix Caching:

    用 Radix Tree 存储 token 序列到 block 的映射。
    支持变长前缀匹配，精确到 token 级别。
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.root = TrieNode()
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_blocks_allocated = 0
        self.blocks_per_request: Dict[str, int] = {}
        self.tree_nodes = 1  # root

    def process_request(self, req: Request) -> Tuple[int, int]:
        """处理请求，用 Radix Tree 查找最长前缀匹配。"""
        all_tokens = req.prompt_tokens + req.input_tokens
        new_blocks = 0
        hit_blocks = 0

        # 逐 block 匹配
        node = self.root
        node.ref_count += 1

        for i in range(0, len(all_tokens), self.block_size):
            block_tokens = all_tokens[i:i + self.block_size]
            if len(block_tokens) < self.block_size:
                block_tokens = block_tokens + [0] * (self.block_size - len(block_tokens))

            # 用第一个 token 作为 key 查找子节点
            # (简化: 实际 RadixAttention 用完整 token 序列匹配)
            first_token = block_tokens[0]

            if first_token in node.children and self._block_matches(node.children[first_token], block_tokens):
                # 前缀匹配
                hit_blocks += 1
                self.cache_hits += 1
                node = node.children[first_token]
                node.ref_count += 1
            else:
                # 不匹配: 创建新节点
                new_node = TrieNode()
                new_node.block_count = 1
                new_node.value = self.tree_nodes
                node.children[first_token] = new_node
                self.tree_nodes += 1

                new_blocks += 1
                self.cache_misses += 1
                node = new_node
                node.ref_count += 1

        self.total_blocks_allocated += new_blocks
        self.blocks_per_request[req.req_id] = new_blocks
        return new_blocks, hit_blocks

    def _block_matches(self, node: TrieNode, tokens: List[int]) -> bool:
        """检查节点是否匹配 token 序列。"""
        # 简化: 总是返回 True (假设 hash 冲突可忽略)
        return node.value is not None

    def free_request(self, req_id: str):
        pass


# ============================================================
# Group-based Prefix Caching (verl)
# ============================================================

class GroupBasedCaching:
    """verl 风格的分组 Prefix Caching:

    RL 训练中同一 prompt 会在多个 PPO step 中重复使用。
    将相同 prompt 的请求分组，组内第一个请求计算 KV Cache，
    后续请求直接复用。
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.prompt_groups: Dict[str, List[str]] = defaultdict(list)  # prompt_hash → req_ids
        self.prompt_cache: Dict[str, int] = {}  # prompt_hash → cached blocks
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_blocks_allocated = 0
        self.blocks_per_request: Dict[str, int] = {}

    def process_request(self, req: Request, prompt_hash: Optional[str] = None) -> Tuple[int, int]:
        """处理请求，同 prompt 分组共享。"""
        if prompt_hash is None:
            prompt_hash = hashlib.md5(str(req.prompt_tokens).encode()).hexdigest()[:12]

        prompt_blocks = (len(req.prompt_tokens) + self.block_size - 1) // self.block_size
        input_blocks = (len(req.input_tokens) + self.block_size - 1) // self.block_size

        if prompt_hash in self.prompt_cache:
            # Prompt KV Cache 已缓存: 只需计算 input 部分
            self.cache_hits += prompt_blocks
            self.cache_misses += input_blocks
            new_blocks = input_blocks
            hit_blocks = prompt_blocks
        else:
            # 首次遇到: 计算全部
            self.prompt_cache[prompt_hash] = prompt_blocks
            self.cache_misses += prompt_blocks + input_blocks
            new_blocks = prompt_blocks + input_blocks
            hit_blocks = 0

        self.prompt_groups[prompt_hash].append(req.req_id)
        self.total_blocks_allocated += new_blocks
        self.blocks_per_request[req.req_id] = new_blocks
        return new_blocks, hit_blocks

    def free_request(self, req_id: str):
        pass


# ============================================================
# 实验函数
# ============================================================

def generate_requests(n_requests: int, n_unique_prompts: int,
                      prompt_len: int = 256, input_len: int = 128,
                      seed: int = 42) -> List[Request]:
    """生成模拟请求，部分请求共享 prompt。"""
    rng = np.random.default_rng(seed)
    vocab_size = 32000

    # 生成 n_unique_prompts 个不同的 system prompt
    prompts = []
    for _ in range(n_unique_prompts):
        prompt = rng.choice(vocab_size, size=prompt_len).tolist()
        prompts.append(prompt)

    requests = []
    for i in range(n_requests):
        prompt_idx = rng.integers(0, n_unique_prompts)
        input_tokens = rng.choice(vocab_size, size=input_len).tolist()
        req = Request(
            req_id=f"req_{i}",
            prompt_tokens=prompts[prompt_idx],
            input_tokens=input_tokens,
        )
        requests.append((req, hashlib.md5(str(prompts[prompt_idx]).encode()).hexdigest()[:12]))

    return requests


def experiment_caching_comparison():
    """实验 1: 四种策略对比。"""
    print("=" * 60)
    print("实验 1: Prefix Caching 策略对比")
    print("=" * 60)

    block_size = 16
    n_requests = 100
    n_unique_prompts = 10  # 10 个不同 prompt，100 个请求
    prompt_len = 256       # 256 tokens prompt
    input_len = 128        # 128 tokens input

    requests = generate_requests(n_requests, n_unique_prompts, prompt_len, input_len)

    # 配置各策略
    no_cache = NoCaching(block_size)
    hash_cache = HashBasedCaching(block_size)
    trie_cache = RadixTreeCaching(block_size)
    group_cache = GroupBasedCaching(block_size)

    total_blocks_all = 0

    for req, prompt_hash in requests:
        total_blocks_all += (req.total_tokens + block_size - 1) // block_size
        no_cache.process_request(req)
        hash_cache.process_request(req)
        trie_cache.process_request(req)
        group_cache.process_request(req, prompt_hash)

    print(f"\n  配置: {n_requests} 请求, {n_unique_prompts} 个不同 prompt")
    print(f"  Prompt: {prompt_len} tokens, Input: {input_len} tokens")
    print(f"  Block size: {block_size} tokens")

    print(f"\n  {'策略':<20} {'总 Block':>10} {'节省':>8} {'命中率':>8} {'每请求 Block':>14}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*14}")

    baseline = no_cache.total_blocks_allocated
    strategies = [
        ("No Caching", no_cache.total_blocks_allocated, 0, 0),
        ("Hash-based (vLLM)", hash_cache.total_blocks_allocated,
         hash_cache.cache_hits, hash_cache.cache_hits + hash_cache.cache_misses),
        ("Trie-based (SGLang)", trie_cache.total_blocks_allocated,
         trie_cache.cache_hits, trie_cache.cache_hits + trie_cache.cache_misses),
        ("Group-based (verl)", group_cache.total_blocks_allocated,
         group_cache.cache_hits, group_cache.cache_hits + group_cache.cache_misses),
    ]

    for name, total, hits, attempts in strategies:
        saving = (1 - total / baseline) * 100 if baseline > 0 else 0
        hit_rate = hits / attempts * 100 if attempts > 0 else 0
        avg = total / n_requests
        print(f"  {name:<20} {total:>10} {saving:>7.1f}% {hit_rate:>7.1f}% {avg:>14.1f}")


def experiment_prompt_reuse_scaling():
    """实验 2: Prompt 复用率对缓存效率的影响。"""
    print("\n" + "=" * 60)
    print("实验 2: Prompt 复用率对缓存效率的影响")
    print("=" * 60)

    block_size = 16
    n_requests = 200
    prompt_len = 256
    input_len = 128

    print(f"\n  配置: {n_requests} 请求, prompt={prompt_len}t, input={input_len}t")
    print(f"  {'Unique Prompts':>15} {'No Cache':>10} {'Hash-based':>12} {'节省':>8} {'命中率':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")

    for n_unique in [1, 2, 5, 10, 20, 50, 100, 200]:
        requests = generate_requests(n_requests, n_unique, prompt_len, input_len, seed=42)

        no_cache = NoCaching(block_size)
        hash_cache = HashBasedCaching(block_size)

        for req, ph in requests:
            no_cache.process_request(req)
            hash_cache.process_request(req)

        baseline = no_cache.total_blocks_allocated
        saving = (1 - hash_cache.total_blocks_allocated / baseline) * 100
        hit_rate = hash_cache.cache_hits / (hash_cache.cache_hits + hash_cache.cache_misses) * 100

        print(f"  {n_unique:>15} {baseline:>10} {hash_cache.total_blocks_allocated:>12} "
              f"{saving:>7.1f}% {hit_rate:>7.1f}%")


def experiment_rl_training_scenario():
    """实验 3: RL 训练场景 (同 prompt 多次 rollout)。"""
    print("\n" + "=" * 60)
    print("实验 3: RL 训练场景模拟 (GRPO/PPO)")
    print("=" * 60)

    block_size = 16
    rng = np.random.default_rng(42)
    vocab_size = 32000

    # 模拟 GRPO: 每个问题生成 8 个 response
    n_prompts = 20
    responses_per_prompt = 8
    prompt_len = 512
    response_len = 256

    # 生成 prompts
    prompts = [rng.choice(vocab_size, size=prompt_len).tolist() for _ in range(n_prompts)]

    # 三级缓存 (verl 风格)
    no_cache = NoCaching(block_size)
    hash_cache = HashBasedCaching(block_size)
    group_cache = GroupBasedCaching(block_size)

    for prompt_idx, prompt in enumerate(prompts):
        prompt_hash = hashlib.md5(str(prompt).encode()).hexdigest()[:12]
        for resp_idx in range(responses_per_prompt):
            response = rng.choice(vocab_size, size=response_len).tolist()
            req = Request(
                req_id=f"p{prompt_idx}_r{resp_idx}",
                prompt_tokens=prompt,
                input_tokens=response,
            )

            no_cache.process_request(req)
            hash_cache.process_request(req)
            group_cache.process_request(req, prompt_hash)

    baseline = no_cache.total_blocks_allocated

    print(f"\n  配置: {n_prompts} prompts × {responses_per_prompt} responses")
    print(f"  Prompt: {prompt_len}t, Response: {response_len}t")

    print(f"\n  {'策略':<20} {'总 Block':>10} {'节省':>8} {'说明':<30}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*30}")
    print(f"  {'No Caching':<20} {baseline:>10} {'0%':>8} {'每次重新计算 prompt':<30}")
    print(f"  {'Hash-based':<20} {hash_cache.total_blocks_allocated:>10} "
          f"{(1-hash_cache.total_blocks_allocated/baseline)*100:>7.1f}% "
          f"{'自动检测同 prompt':<30}")
    print(f"  {'Group-based':<20} {group_cache.total_blocks_allocated:>10} "
          f"{(1-group_cache.total_blocks_allocated/baseline)*100:>7.1f}% "
          f"{'显式分组，最精确':<30}")

    print(f"""
关键洞察:
  - RL 训练中 prompt 复用率极高 (同一 prompt × 8 responses)
  - Prefix Caching 节省 {(1-group_cache.total_blocks_allocated/baseline)*100:.0f}% 的 KV Cache 计算
  - Group-based 最精确: 知道哪些请求共享 prompt
  - Hash-based 最通用: 不需要显式分组信息
  - 实际 verl: 三级缓存 (系统级 + 进程级 + 请求级)
    """)


def experiment_prompt_length_impact():
    """实验 4: Prompt 长度对缓存收益的影响。"""
    print("=" * 60)
    print("实验 4: Prompt 长度对缓存收益的影响")
    print("=" * 60)

    block_size = 16
    n_requests = 50
    n_unique_prompts = 5
    input_len = 128

    print(f"\n  配置: {n_requests} 请求, {n_unique_prompts} 个不同 prompt, input={input_len}t")
    print(f"  {'Prompt长度':>12} {'No Cache':>10} {'With Cache':>12} {'节省':>8} {'命中率':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")

    for prompt_len in [32, 64, 128, 256, 512, 1024, 2048]:
        requests = generate_requests(n_requests, n_unique_prompts, prompt_len, input_len, seed=42)

        no_cache = NoCaching(block_size)
        hash_cache = HashBasedCaching(block_size)

        for req, ph in requests:
            no_cache.process_request(req)
            hash_cache.process_request(req)

        baseline = no_cache.total_blocks_allocated
        saving = (1 - hash_cache.total_blocks_allocated / baseline) * 100
        hit_rate = hash_cache.cache_hits / (hash_cache.cache_hits + hash_cache.cache_misses) * 100

        print(f"  {prompt_len:>12} {baseline:>10} {hash_cache.total_blocks_allocated:>12} "
              f"{saving:>7.1f}% {hit_rate:>7.1f}%")


def main():
    print("=" * 60)
    print("Prefix Caching 模拟器 — Block 共享效率对比")
    print("=" * 60)

    experiment_caching_comparison()
    experiment_prompt_reuse_scaling()
    experiment_rl_training_scenario()
    experiment_prompt_length_impact()

    # 总结
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
Prefix Caching 核心知识:

1. 核心思想:
   如果多个请求共享相同的前缀 token，它们的 KV Cache 可以复用
   → 省去重复计算，减少延迟和显存

2. 实现策略:

   Hash-based (vLLM):
   - Block 级哈希链: hash = H(parent_hash, block_tokens)
   - 自动检测共享前缀，无需用户标注
   - 简单高效，但粒度固定为 block 大小

   Trie-based (SGLang RadixAttention):
   - Radix tree 存储前缀 → block 映射
   - 支持变长前缀匹配
   - 更灵活但树维护开销更大

   Group-based (verl):
   - 显式分组: 同 prompt 的请求归为一组
   - 第一人计算，后续复用
   - RL 训练中最精确 (知道 prompt 分组)

3. 收益因素:
   - Prompt 复用率: 越高收益越大
   - Prompt 长度: 越长收益越大
   - Block 大小: 越小匹配粒度越细

4. 典型场景:
   - RL 训练: GRPO/PPO 同 prompt × N responses (节省 50%+)
   - 多轮对话: 相同 system prompt (节省 30-50%)
   - RAG: 相同文档前缀 (节省 60-80%)
   - API 服务: 不同用户但共享 prompt template (节省 20-40%)
    """)


if __name__ == "__main__":
    main()
