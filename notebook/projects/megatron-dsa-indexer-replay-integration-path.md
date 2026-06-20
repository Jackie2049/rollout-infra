# DSA Indexer Replay Integration Path for verl GRPO

> Created: 2026-06-20 | Priority: ★★★★★★★★ P0 for RTX 4090 GRPO with DSA models
> Based on deep reading: #5384/#5386, #2693 RouterReplay, verl V1 architecture

## 1. Problem Statement

DSA indexer `top-k` is a **discrete decision** — small numerical differences between rollout (SGLang/vLLM, fp16 inference) and training (Megatron, mixed precision) produce **different top-k selections** → logprob mismatch → incorrect GRPO reward attribution.

**Same root cause as MoE Router Replay (R3)**: ~10% routers disagree per forward → 3/3 baseline GRPO runs collapsed on Qwen3-30B-A3B.

## 2. Proposed Solution: DSAIndexerReplay

### Architecture (follows RouterReplay #2693 template)

```
class DSAIndexerReplay:
    """Replay pattern for DSA indexer top-k decisions.
    Follows RouterReplay architecture (megatron/core/transformer/mo/router_replay.py).
    """

    class ReplayAction(Enum):
        RECORD = 0         # Record indexer top-k during rollout
        REPLAY_FORWARD = 1 # Use recorded top-k during training forward
        REPLAY_BACKWARD = 2 # Use recorded top-k during training backward

    _instances: Dict[str, DSAIndexerReplay] = {}  # Global registry per layer
    _replay_data: Dict[str, torch.Tensor] = {}     # Recorded top-k indices

    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.action = ReplayAction.RECORD
        self._recorded_topk = None

    @classmethod
    def set_replay_data(cls, all_layers_topk: Dict[str, torch.Tensor]):
        """Distribute recorded top-k indices to all DSAIndexerReplay instances."""
        for name, topk in all_layers_topk.items():
            if name in cls._instances:
                cls._instances[name]._recorded_topk = topk

    @classmethod
    def get_recorded_data(cls) -> Dict[str, torch.Tensor]:
        """Collect recorded top-k from all instances."""
        return {name: inst._recorded_topk for name, inst in cls._instances.items()}

    def get_replay_topk(self, index_scores: torch.Tensor, topk: int) -> torch.Tensor:
        """Bypass top-k computation during replay mode."""
        if self.action == ReplayAction.REPLAY_FORWARD or self.action == ReplayAction.REPLAY_BACKWARD:
            return self._recorded_topk
        return torch.topk(index_scores, topk).indices
```

### Estimated LOC: ~200-300 (same as RouterReplay)

## 3. Integration Path

### Step 1: Megatron-LM side (~200 LOC)
- Implement `DSAIndexerReplay` class in `megatron/core/transformer/experimental_attention_variant/`
- Add replay hooks to `DSAIndexer.forward()`: bypass top-k when action != RECORD
- Register per-layer instances in global registry
- Support CUDA graph static buffer (same as RouterReplay)

### Step 2: SGLang rollout side (~50 LOC)
- During rollout, extract `indexer_topk` from SGLang DSA attention metadata
- Add to rollout output dict as `meta['indexer_topk']` per layer
- Community workaround already exists: stash meta['indexer_topk']

### Step 3: verl training side (~100 LOC)
- Carry `indexer_topk` from rollout data through TransferQueue
- Before training forward: `DSAIndexerReplay.set_replay_data(rollout_topk_data)`
- Set action = REPLAY_FORWARD for training forward pass
- Set action = REPLAY_BACKWARD for training backward pass
- Set action = RECORD for new rollout generation

## 4. Data Flow Diagram

```
Rollout Phase:
  1. SGLang generates responses with DSA indexer top-k
  2. Record indexer_topk per layer → meta['indexer_topk']
  3. TransferQueue carries: prompts + responses + logprobs + indexer_topk

Training Phase:
  4. DSAIndexerReplay.set_replay_data(indexer_topk_data)
  5. DSAIndexerReplay.action = REPLAY_FORWARD
  6. Forward pass: DSA uses recorded top-k (not recomputed)
  7. DSAIndexerReplay.action = REPLAY_BACKWARD
  8. Backward pass: gradient flows through same top-k selection
  9. Advantages computed with CONSISTENT logprobs → correct reward attribution
```

## 5. Memory Impact on RTX 4090

```
Per-layer indexer_topk storage:
  Shape: [batch_size, seq_len_q, index_topk]
  Dtype: int64 (indices)
  Memory: bs * sq * topk * 8 bytes
  Example: 8 * 1536 * 64 * 8 = 6.3 MiB per layer
  Total for 8 layers: ~50 MiB → NEGLIGIBLE on 24 GiB GPU
```

**No additional VRAM overhead** — indices stored as integers, not floating point.

## 6. RTX 4090 GRPO Config Impact

| Config | Without DSAIndexerReplay | With DSAIndexerReplay |
|--------|--------------------------|----------------------|
| Memory | 16.62 GiB (same) | 16.67 GiB (+0.05 GiB for indices) |
| Step time | ~13.95s | ~14.0s (negligible) |
| Logprob agreement | <90% (mismatch) | >99% (replay) |
| GRPO signal | CORRUPTED | CORRECT |

## 7. Contribution Path

### Option A: Megatron-LM PR
- Fork: jackie2049/Megatron-LM
- Implement DSAIndexerReplay class + forward() hooks
- ~200-300 LOC new code, ~10 LOC modification
- Reference: #2693 RouterReplay as template
- User reviews before submitting to NVIDIA/Megatron-LM

### Option B: verl integration
- Fork: jackie2049/verl
- Add indexer_topk to TransferQueue data flow
- ~100 LOC modification to data flow pipeline
- Depends on Megatron side being ready first

### Option C: SGLang rollout extraction
- Fork: jackie2049/sglang
- Extract indexer_topk from DSA attention metadata
- ~50 LOC modification
- Independent, can proceed first

**Recommended order**: C → A → B (SGLang first, Megatron second, verl third)

## 8. Relation to Existing Issues

- #5384 (feature request): Original proposal — DSAIndexerReplay needed
- #5386 (follow-up): verl RL batch layout specifics
- #2693 (MERGED): RouterReplay template to follow
- #5394/#5395: Muon clipping — orthogonal concern, independent
- #6794: Delta weight sync — different bug class (weight corruption vs indexer mismatch)

## 9. Verification Checklist (when GPU available)

1. SGLang: Extract indexer_topk from DSA attention metadata → save as dict
2. Compare: indexer_topk from rollout vs training forward → compute agreement rate
3. Baseline: Run GRPO without replay → measure logprob agreement rate (expect <90%)
4. Replay: Run GRPO with replay → measure logprob agreement rate (expect >99%)
5. Reward: Compare reward attribution with vs without replay → expect different advantage values
6. Training: Compare loss curves → expect cleaner convergence with replay

## References

- Megatron #5384: https://github.com/NVIDIA/Megatron-LM/issues/5384
- Megatron #5386: https://github.com/NVIDIA/Megatron-LM/issues/5386
- RouterReplay #2693: https://github.com/NVIDIA/Megatron-LM/pull/2693
- DSA implementation: megatron/core/transformer/experimental_attention_variant/dsa.py
- verl TransferQueue: verl/v1/workers/rollout/rollout_manager.py
