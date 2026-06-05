# verl #6401 Technical Comment Draft

> Draft of a technical comment to post on github.com/volcengine/verl/issues/6401

## Comment (English)

Thanks for the great RFC! I've been studying verl's existing PrefixGrouper implementation and have some analysis that might be helpful for this RFC.

### Current PrefixGrouper Architecture

The existing `prefix_grouper` (external package, integrated via `verl/trainer/ppo/prefix_grouper_utils.py`) works as follows:

1. **Flat grouping**: Groups sequences by `uid` (same uid = same prompt prefix)
2. **Concatenation**: `[prefix | suffix1 | suffix2 | ... | suffixN]` into a flat layout
3. **Monkey-patch**: Intercepts attention calls via `verl/models/transformers/monkey_patch.py` → delegates to `prefix_grouper.forward()`
4. **Decomposition**: Splits into prefix attention + suffix attention
5. **Recovery**: `split_output()` restores per-suffix logits

Performance (Qwen3-4B, 4×H800): 1.26x @ 4K, 1.56-1.70x @ 8K context.

### Gap to RFC #6401

| Aspect | Current | RFC #6401 |
|--------|---------|-----------|
| Grouping | Flat (same uid) | Tree (arbitrary depth) |
| Attention backend | FA2/FA3/SDPA | Magi Attention (sparse mask) |
| Prefix detection | uid-based (implicit) | hash-based prefix_segments |
| Backend support | FSDP only | Megatron + FSDP |
| Cross batch | No | Future: cache-based |

### Potential Contribution Path

I've been working on a `prefix-sharing` project that implements:
- `TriePrefixDetector`: Tree-based prefix detection using tries
- `PrefixSharingConfig`: Full configuration dataclass
- KV injection + recovery logic

These could potentially contribute to the tree-based prefix detection module needed for this RFC.

### Questions for the RFC

1. **Prefix tree depth**: What's the expected tree depth in typical GRPO workloads? For n=8 responses, the tree is quite flat (1 prefix + 8 suffixes). Multi-turn would increase depth, but how deep in practice?

2. **FSDP integration**: The current `FSDPEngineWithLMHead.forward_step()` doesn't pass `prefix_grouper` kwargs — is this already addressed in the RFC design?

3. **Magi Attention backend**: Is there an open-source implementation available, or is the plan to implement from scratch?

Looking forward to contributing to this effort!
