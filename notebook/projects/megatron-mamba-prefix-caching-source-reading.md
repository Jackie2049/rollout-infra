# Megatron Mamba Prefix Caching Architecture — Source-Level Analysis

> 2026-06-16 | NVIDIA/Megatron-LM | #5348 + #5309 + #5274 + #5188 + #5122
> ★★★★★★★★ MambaSlotAllocator = dual GPU state tensors (conv_states + ssm_states) + block-to-slot mapping + hash-to-block prefix caching + intermediate state extraction for CUDA graph compatibility
> ★★★★★★★★ SSM state caching = 10-100x smaller than KV cache for same context → hybrid Mamba-Transformer models can achieve MUCH longer context on RTX 4090!
> ★★★★★★★★ Generic SSM interface (#5274) → supports arbitrary SSM variants (GatedDeltaNet, RWKV) beyond Mamba

## 1. ★★★★★★★★ MambaSlotAllocator — Dual GPU State + Block Mapping

```
★★★★★★★★★ megatron/core/inference/contexts/mamba_slot_allocator.py

  __init__(context, max_slots, num_mamba_layers, conv_states_shape, ssm_states_shape,
            conv_states_dtype, ssm_states_dtype):

    # Block <-> slot mappings (CPU bookkeeping)
    block_to_slot = torch.full((num_blocks,), -1, int32, cpu)    # KV block → Mamba slot
    slot_to_block = torch.full((max_slots,), -1, int32, cpu)     # Mamba slot → KV block

    # Free slot pool (stack, CPU)
    free_slots = torch.arange(max_slots, int32, cpu)
    free_count = max_slots

    # ★★★★★★★★ DUAL GPU state tensors — THE CORE DATA STRUCTURE!
    conv_states = torch.zeros(
        (num_mamba_layers, max_slots) + conv_states_shape,   # shape: [L, S, D_conv, ...]
        dtype=conv_states_dtype, device=gpu_device
    )
    ssm_states = torch.zeros(
        (num_mamba_layers, max_slots) + ssm_states_shape,    # shape: [L, S, D_inner, ...]
        dtype=ssm_states_dtype, device=gpu_device
    )

    # ★★★★★★★★ Prefix cache hash mapping — reuse SSM state for shared prefixes!
    hash_to_block_id: Dict[int, int] = {}

    # ★★★★★★★★ Intermediate state extraction — CUDA graph compatible
    # 3 candidates per request: KV divergence boundary, last block-aligned, penultimate
    MAX_INTERMEDIATE_OFFSETS_PER_REQUEST = 3

    intermediate_offsets_cpu = torch.zeros((max_requests, 3), int32, cpu)
    intermediate_counts_cpu = torch.zeros((max_requests,), int32, cpu)
    intermediate_offsets_gpu = torch.zeros((max_requests, 3), int32, gpu)
    intermediate_counts_gpu = torch.zeros((max_requests,), int32, gpu)

    # ★★★★★★★★ Pre-allocated output buffers for CUDA graph extraction (GPU)
    intermediate_ssm_out = torch.zeros(
        (num_mamba_layers, max_intermediate_count) + ssm_states_shape,
        dtype=ssm_states_dtype, device=gpu_device
    )
    intermediate_conv_out = torch.zeros(
        (num_mamba_layers, max_intermediate_count) + conv_states_shape,
        dtype=conv_states_dtype, device=gpu_device
    )

  ★★★★★★★★ Key insight: Mamba has TWO state types:
    1. conv_states: Conv1d state (short, d_conv=4 tokens → ~4*d_inner bytes)
    2. ssm_states: SSM hidden state (recurrent state → ~d_inner*2 bytes for A/B/DT)
    → ★★★★★★★★ Together: ~2*d_inner + d_conv*d_inner bytes per layer per slot
    → → ★★★★★★★★ VS KV cache: ~2*d_head*num_heads*seq_len bytes per layer per slot
    → → → For seq_len=4096: SSM state ≈ 10KB vs KV cache ≈ 128KB per layer → 12x smaller!
```

### 1.1 Slot Allocation — Deduplication + LRU Eviction

```
★★★★★★★★★ allocate_slots_batch(block_ids) → 5-phase pipeline:

  Phase 1: Batch lookup existing slots (1 GPU sync)
    → block_to_slot[bid_tensor].tolist()
    → Deduplication: same block_id → same slot (SSM state shared for prefix!)

  Phase 2: Identify new blocks needing allocation (deduplicated)
    → seen_new = {} → prevents duplicate slot allocation for shared prefix blocks
    → ★★★★★★★★ This is the PREFIX REUSE mechanism — shared prefix → shared Mamba state!

  Phase 3: Get slots from free pool, evicting if necessary
    → from_free = min(num_new, free_count) → take from free pool first
    → need_evict = num_new - from_free → LRU eviction for remaining
    → ★★★★★ _evict_lru_slots_batch(need_evict) → LRU eviction from existing slots

  Phase 4: Batch GPU writes for new mappings
    → block_to_slot[new_bids] = new_slots
    → slot_to_block[new_slots] = new_bids

  Phase 5: Build result mapping
    → existing slots reused for deduplication → new slots for fresh blocks
```

## 2. ★★★★★★★★ MambaMetadata — CUDA Graph Compatible Batch Metadata

```
★★★★★★★★★ megatron/core/inference/contexts/attention_context/mamba_metadata.py (~752 lines)

  __init__(max_requests, max_tokens, mamba_chunk_size=128, d_conv=0):

    # ★★★★★★★★ Dual decode/prefill batch index buffers
    batch_indices_decode_buffer = torch.full((max_requests,), -1, int64, gpu)
      → int64 for selective_state_update kernel direct indexing
    batch_indices_prefill_buffer = torch.full((max_requests,), -1, int32, gpu)

    # ★★★★★★★★ Prefill metadata (varlen SSM kernel)
    seq_idx_buffer = torch.full((1, max_tokens), -1, int32, gpu)        # token→request
    cu_seqlens_buffer = torch.zeros((max_requests+1,), int32, gpu)      # cumulative lengths

    # ★★★★★★★★ SSM chunk metadata — chunk_size subdivision for efficient prefill
    cu_chunk_seqlens_buffer = torch.zeros(max_chunks+1, int32, gpu)     # chunk boundaries
    last_chunk_indices_buffer = torch.zeros(max_requests, int32, gpu)   # last chunk per seq
    seq_idx_for_varlen_buffer = torch.zeros(max_chunks, int32, gpu)     # chunk→request mapping

    # ★★★★★★★★ Conv1d per-token metadata
    conv_seq_idx_buffer = torch.zeros(max_tokens, int32, gpu)           # token→request
    conv_seq_start_buffer = torch.zeros(max_tokens, int32, gpu)         # token→seq start

    # ★★★★★★★★ Decode/prefill count buffer (2-element)
    device_decode_prefill_buffer = torch.zeros((2,), int32, gpu)        # [decode, prefill]

    # ★★★★★★★★ Slot management (CPU for bookkeeping)
    request_to_mamba_state_idx = torch.full((max_requests,), -1, int32, cpu)
    mamba_state_free_slots = torch.arange(max_requests, int32, cpu)
    mamba_state_free_slot_count = max_requests

    # ★★★★★★★★ Coalesced H2D transfer path (CPU pinned + shared GPU views)
    _cpu_bufs = None   # pinned CPU views from DynamicInferenceContext
    _gpu_view = None   # shared GPU views (single H2D transfer!)

  ★★★★★★★★ Two update paths:

    1. Legacy update() — standalone GPU buffers → individual H2D transfers
      → Used by unit tests → NOT production path

    2. ★★★★★★★★ Coalesced compute_cpu_metadata() + load_from_cpu()
      → compute_cpu_metadata(): all Python-side computation → pinned CPU views → no GPU sync!
      → → Single coalesced H2D in transfer_bookkeeping_to_gpu()
      → → load_from_cpu(): slice shared GPU views → intermediate metadata from GPU
      → → ★★★★★★★★ ONE H2D transfer for ALL Mamba metadata! → minimal CPU→GPU overhead!

  ★★★★★★★★ Chunk processing:
    mamba_chunk_size = 128 (default)
    → Each prefill sequence subdivided into chunks of at most 128 tokens
    → cu_chunk_seqlens = cumulative chunk boundaries
    → last_chunk_indices = index of last chunk per sequence
    → → ★★★★★★★★ Enables varlen SSM kernel → efficient batched prefill!
    → → → Chunk = unit of intermediate state extraction → prefix cache granularity!
```

### 2.1 ★★★★★★★★ Intermediate State Extraction — CUDA Graph Compatible

```
★★★★★★★★★ _update_intermediate_metadata() — vectorized GPU computation:

  Input: intermediate_offsets_gpu [prefill_count, 3] + intermediate_counts_gpu [prefill_count]

  Computation (ALL on GPU, CUDA graph compatible):
    1. cu_seqlens → seq_lens → num_chunks → cum_chunks (cumulative chunk counts)
    2. seq_starts + offsets → chunk_indices = cum_chunks + offsets // chunk_size - 1
    3. seq_starts + offsets → abs_positions = seq_starts + offsets
    4. Validity mask: j_indices < intermediate_counts → filter valid entries
    5. Scatter into fixed-size buffers (CUDA graph replay safe)

  ★★★★★★★★ 3 intermediate offset candidates per request:
    1. KV divergence boundary → first position where KV diverges from Mamba state
    2. Last block-aligned boundary → last position that aligns with KV block boundary
    3. Penultimate block boundary → second-to-last KV block boundary
    → ★★★★★ These are extraction points where SSM state can be saved for prefix reuse!

  ★★★★★★★★ CUDA graph compatibility:
    → All buffers pre-allocated to fixed maximum size
    → No data-dependent control flow in forward pass
    → Safe default values for unused slots (chunk_idx=0, abs_pos=d_conv)
    → → ★★★★★★★★ Forward pass has ZERO .item() calls → fully CUDA graph compatible!
```

## 3. ★★★★★★★★ RTX 4090 Memory Comparison: SSM State vs KV Cache

```
★★★★★★★★★ Memory per layer per request:

  SSM state (Mamba):
    conv_states: d_conv * d_inner * dtype_bytes
      → d_conv=4, d_inner=2560, dtype=bf16 → 4*2560*2 = 20,480 bytes ≈ 20KB
    ssm_states: d_inner * d_state * dtype_bytes  (A/B/DT matrices)
      → d_inner=2560, d_state=16, dtype=bf16 → 2560*16*2 = 81,920 bytes ≈ 82KB
    → Total per layer per request: ~102KB ≈ 0.1MB

  KV cache (Transformer attention):
    KV: 2 * num_heads * d_head * seq_len * dtype_bytes
      → num_heads=32, d_head=80, seq_len=4096, dtype=bf16 → 2*32*80*4096*2 = 4,194,304 bytes ≈ 4MB
    → Total per layer per request: ~4MB

  ★★★★★★★★ Ratio: SSM state / KV cache = 0.1MB / 4MB ≈ 40x SMALLER at seq_len=4096!

  ★★★★★★★★ Hybrid model (e.g., 24 Mamba + 8 Attention layers):
    8B model, hybrid Mamba-Transformer:
      → 24 Mamba layers × 0.1MB × batch_size = 2.4MB × batch_size
      → 8 Attention layers × 4MB × batch_size = 32MB × batch_size
      → Total: 34.4MB × batch_size

    vs Pure Transformer (32 Attention layers):
      → 32 × 4MB × batch_size = 128MB × batch_size

    ★★★★★★★★ Hybrid = 34.4MB vs Pure = 128MB → 3.7x savings per request!

  ★★★★★★★★ RTX 4090 (24GB) practical:
    8B pure Transformer: max_seqs at seq_len=4096 → 24GB / (128MB + model) ≈ ~170 seqs
    8B hybrid Mamba-Transformer: max_seqs → 24GB / (34.4MB + model) ≈ ~600 seqs!
    → ★★★★★★★★ 3.5x MORE concurrent requests → MUCH higher throughput for hybrid models!

  ★★★★★★★★ SSM state + KV cache for prefix reuse:
    → SSM state for shared prefix → hash_to_block_id → deduplication
    → KV cache for shared prefix → existing block reuse
    → → ★★★★★★★★ Both cached for prefix reuse → hybrid prefix caching = dual-layer!
    → → → Same block_id → same Mamba slot → deduplication → saves memory!
```

## 4. Recent PR Developments (June 2026)

### #5348 Mamba Prefix Caching Memory Safety (June 15)
```
★★★★★★★ Expand memory safety check to include scratch space buffers
  → Previously: only checked main buffer size → OOMs when scratch exceeded buffer!
  → Now: scratch space buffers included in total → prevents OOMs
  → Warning when hybrid models use prefix caching without mamba-prefix-caching-buffer-size
  → ★★★★★ RTX 4090: safer hybrid inference → no OOM from scratch space → explicit buffer sizing
```

### #5309 SSM States Dtype Configurable (June 11, Approved!)
```
★★★★★★★ Add --mamba-training-ssm-states-dtype argument
  → Configurable dtype for SSM states during training
  → bf16 states → 50% memory savings → RTX 4090 training benefit!
  → fp32 states → numerical stability → important for long sequences
  → ★★★★★★★★ RTX 4090: bf16 states + LoRA → fits MoE+SSM hybrid in 24GB!
```

### #5274 Generic SSM Interface (June 10)
```
★★★★★★★ Refactor Mamba dynamic inference → generic SSM interface
  → Support arbitrary SSM variants beyond Mamba (GatedDeltaNet, RWKV, etc.)
  → DynamicInferenceContext handles SSM states generically
  → ★★★★★★★★ Enables future hybrid architectures: Mamba + GatedDeltaNet + Attention
  → → RTX 4090: more SSM options → more efficient hybrid models possible!
```

### #5188 Heterogeneous KV/Mamba Reshard (June 5)
```
★★★★★ Disag MR3: heterogeneous KV/Mamba reshard planners
  → For disaggregated inference (prefill/decode split)
  → Different reshard strategies for KV cache vs Mamba state
  → ★★★★ RTX 4090: not applicable (single GPU) → but shows architecture direction
```

## 5. ★★★★★★★★ RTX 4090 Hybrid Model Recommendations

```
★★★★★★★★★ RTX 4090 hybrid Mamba-Transformer configuration:

  Model choices:
    → Jamba-1.5-mini (12B, 24 Mamba + 8 Attention) → 5.4B active → fits 24GB with INT4!
    → Zamba2-7B (7B, hybrid) → BF16 fits ~14GB → room for KV+SSM cache
    → Custom: Qwen2.5 backbone + Mamba layers → experimental

  Inference config:
    → INT4 quantization for weights → model ~3-4GB
    → bf16 SSM states + bf16 KV cache → per-request ~34MB
    → max_seqs = (24GB - 4GB) / 34MB ≈ 588 concurrent requests!
    → → ★★★★★★★★ vs pure Transformer INT4: (24-4)/4 ≈ 5 seqs only at seq_len=4096!
    → → → Wait, that's wrong — for seq_len=4096 with INT4 weights:
    → → → Hybrid: model=4GB, per-req KV+SSM=34.4MB → 20GB/34.4MB ≈ 580 seqs
    → → → Pure Transformer: model=4GB, per-req KV=128MB → 20GB/128MB ≈ 156 seqs (seq_len=4096 with GQA)
    → → → ★★★★★★★★ Still 3.7x improvement for hybrid models!

  Training config:
    → bf16 SSM states (via #5309) → 50% SSM state memory savings
    → LoRA rank=32 → trainable params only ~0.6GB
    → ZeRO-2 + CPU_Adam → optimizer on CPU → GPU freed for model + states
    → ★★★★★★★★ Hybrid MoE + Mamba + LoRA → new RTX 4090 training frontier!

  ★★★★★★★★ IMPORTANT: Megatron Mamba prefix caching = INFERENCE feature
    → Not applicable for training (training doesn't use prefix caching)
    → But SSM state dtype (#5309) = TRAINING feature → configurable precision
    → → RTX 4090: train with bf16 states → infer with prefix caching → full lifecycle!
```

## Key Source Files

- `megatron/core/inference/contexts/mamba_slot_allocator.py`: MambaSlotAllocator (dual state + block mapping + prefix hash + intermediate extraction + LRU eviction)
- `megatron/core/inference/contexts/attention_context/mamba_metadata.py`: MambaMetadata (~752 lines, CUDA graph compatible batch metadata + coalesced H2D)
- `megatron/core/inference/contexts/dynamic_context.py`: DynamicInferenceContext (parent context owning both KV and Mamba allocators)
- `megatron/core/ssm/mamba_layer.py`: MambaLayer (forward pass consuming MambaMetadata)
- `megatron/core/ssm/mamba_mixer.py`: MambaMixer (SSM mixer with conv1d + selective_state_update)

## Related Notes

- [Megatron v0.17 Latest Reading](megatron-v0.17-latest-reading.md) — general v0.17 developments
- [Megatron MoE Architecture](megatron-moe-architecture.md) — MoE + EP architecture
- [Megatron Lite Reading](megatron-lite-reading.md) — Lite + LoRA + verl GRPO integration
- [Quantile Balancing MoE Routing](megatron-quantile-balancing-moe-routing-source-reading.md) — QB routing #5349
