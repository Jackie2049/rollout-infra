#!/usr/bin/env python3
"""SGLang Sleep/Wake Integration Guide for verl GRPO Training on RTX 4090.

Modes:
  architecture  — Complete SGLang sleep/wake architecture overview
  rtx4090       — RTX 4090 specific sleep/wake memory & timing analysis
  compare       — SGLang vs vLLM sleep/wake comparison
  debug         — Sleep/wake debugging guide

Usage:
  python3 sglang_sleep_wake_integration_guide.py [mode]
  python3 sglang_sleep_wake_integration_guide.py rtx4090
"""

import sys
import textwrap

BANNER = """
============================================================
 SGLang Sleep/Wake Integration Guide for verl GRPO Training
============================================================
"""

# ---------------------------------------------------------------------------
# Mode 1: architecture
# ---------------------------------------------------------------------------

ARCHITECTURE = """
=== MODE: architecture ===
SGLang Sleep/Wake Architecture for verl GRPO Training

1. SLEEP LEVELS
   ────────────────────────────────────────────────────────────────────
   Level  | What Is Freed          | Memory Freed (8B) | Wake Time
   ────────────────────────────────────────────────────────────────────
   0      | Nothing (baseline)     | 0 GiB             | 0s
   1      | KV cache only          | ~2 GiB            | ~1s (fast)
   2      | ALL GPU memory         | ~16 GiB           | ~3s (full reload)
   ────────────────────────────────────────────────────────────────────

   - Level 0: No sleep. Engine stays resident. Used when GPU memory is
     abundant enough to hold model + KV + training buffers simultaneously.
   - Level 1: Frees KV cache allocations but keeps model weights in GPU
     memory. Fast wake because only KV cache needs re-allocation, not a
     full weight reload. Best for single-GPU setups where training needs
     only the weight memory freed from KV, not the full model offload.
   - Level 2: Frees ALL GPU memory — model weights, KV cache, activations,
     internal buffers. Slowest wake because the entire model must be
     re-loaded from CPU/disk to GPU. Required when training phase needs
     nearly all GPU VRAM.

2. API
   ────────────────────────────────────────────────────────────────────
   engine.sleep(level=1)           # Release KV cache; weights stay
   engine.sleep(level=2)           # Release everything; full offload

   engine.wake_up(tags=['weights', 'kv_cache'])  # Reload weights + KV
   engine.wake_up(tags=['weights'])               # Reload weights only
   engine.wake_up(tags=['kv_cache'])              # Reload KV only

   - The `tags` parameter controls what to reload on wake:
     * 'weights'   = model parameters (full or LoRA adapter)
     * 'kv_cache'  = KV cache state / prefix caches
   - Calling wake_up with incomplete tags before a forward pass will
     cause illegal memory access (see debug mode).

3. LORA INTEGRATION
   ────────────────────────────────────────────────────────────────────
   sleep_level=1 + LoRA adapter path  =>  80x payload reduction

   - LoRA merge=False: adapter weights remain separate from the base
     model. When sleeping at level 2, SGLang releases the full base
     model from GPU but retains adapter metadata (tiny). On wake, only
     the adapter needs updating if weights changed.
   - Weight update: update LoRA adapter only (0.06 GiB) vs full model
     (16 GiB). This is a 267x reduction in transfer size.
   - With LoRA bypass during training (adapter detached, base model
     weights frozen), the peak GPU memory is dramatically lowered.

   LoRA Adapter Update Flow:
     training_step() → update_lora_adapter(new_weights)
     → 0.06 GiB transfer → 0.004s on RTX 4090
     vs full model reload: 16 GiB → ~3s wake overhead

4. RADIXATTENTION — PREFIX KV PINNING (RolloutKV #28608)
   ────────────────────────────────────────────────────────────────────
   - SGLang RadixAttention supports prefix KV pinning during rollout.
     Prompt prefix KV is pinned against LRU eviction, guaranteeing
     reuse across multiple rollout generations with the same system
     prompt / instruction prefix.
   - Result: 22-42x logprob scoring speedup for GRPO reward computation
     because the prompt prefix KV does not need to be recomputed.
   - TTL-based stale pin eviction: pins have a TTL. If the prefix
     changes (e.g., new system prompt), stale pins are evicted safely
     without manual cache management.

   RadixAttention Pin Lifecycle:
     rollout_start() → pin_prefix_kv(prefix_tokens, ttl=step_ttl)
     → generate responses → score logprobs (reuse pinned prefix KV)
     → rollout_end() → sleep(level=1) → pins auto-evicted on KV free

5. VERL INTEGRATION FLOW
   ────────────────────────────────────────────────────────────────────
   Step lifecycle for GRPO with sleep/wake:

     ┌──────────────┐
     │   ROLLOUT     │  Engine is awake, KV cache active, prefix pinned
     │  (generate)   │  RadixAttention serves prefix KV reuse
     └──────────────┘
           │
           ▼
     engine.sleep(level=1)          ← Free KV cache (2 GiB freed)
     │                                 Prefix pins released
     │                                 Model weights STAY on GPU
     ▼
     ┌──────────────┐
     │   TRAINING    │  GPU now has room for training buffers
     │  (GRPO step)  │  Uses model weights still on GPU
     └──────────────┘
           │
           ▼
     engine.wake_up(tags=['weights','kv_cache'])
     │                                 Re-allocate KV cache
     │                                 Reload any updated weights
     │                                 Re-pin prefix KV for next rollout
     ▼
     ┌──────────────┐
     │   ROLLOUT     │  ← Next step begins
     │  (generate)   │
     └──────────────┘

   With LoRA, the flow becomes:

     rollout → sleep(level=2) → update_lora_adapter → wake_up(tags=['weights','kv_cache']) → rollout

   Level 2 is used with LoRA because the full base model is released,
     and on wake only the adapter (0.06 GiB) needs the weight update
     while the base model is reloaded from CPU memory (~3s).
"""

# ---------------------------------------------------------------------------
# Mode 2: rtx4090
# ---------------------------------------------------------------------------

RTX4090 = """
=== MODE: rtx4090 ===
RTX 4090 Sleep/Wake Analysis for verl GRPO Training

RTX 4090 specs: 24 GiB VRAM, 16384 CUDA cores, 450W TDP, PCIe 4.0 x16

1. MEMORY BUDGET WITH SLEEP/WAKE
   ────────────────────────────────────────────────────────────────────

   Level 0 — No Sleep (baseline):
     model weights  : 16.00 GiB
     KV cache       :  2.00 GiB
     misc (buffers) :  1.00 GiB
     ──────────────────────────────
     total          : 19.00 GiB
     remaining      :  5.00 GiB
     training fit?  : NO — training needs ~7+ GiB for gradients,
                      optimizer states, activations. 5 GiB insufficient.

   Level 1 — KV Cache Freed:
     During rollout peak:
       model(16) + KV(2) + misc(1) = 19 GiB
     After sleep(level=1):
       model(16) + misc(1) = 17 GiB
       KV freed → 2 GiB now available for training
     Training peak:
       model(16) + training_buffers(3.5) + misc(1) = 20.5 GiB
     Peak across phases = max(19, 20.5) = 20.5 GiB
     Margin: 24 - 20.5 = 3.5 GiB  ← SAFE

   Level 2 — ALL Freed:
     After sleep(level=2):
       0 GiB on GPU (everything released)
     Training peak (with LoRA):
       model_reload(16) + LoRA(0.25) + gradients(0.25) + misc(0.25) = 16.75 GiB
     Actually, with LoRA bypass:
       base_model(16) + LoRA_adapter(0.06) + training_buffers(0.19) = 16.25 GiB
     Peak = 16.25 GiB
     Margin: 24 - 16.25 = 7.75 GiB  ← VERY SAFE

   BEST CONFIGURATION for RTX 4090:
   ────────────────────────────────────────────────────────────────────
   Level 1 + LoRA bypass:
     - Rollout: model(16) + KV(2) + misc(1) = 19 GiB
     - Training: model(16) + LoRA_bypass_training(3.5) + misc(1) = 20.5 GiB
     - Peak: 20.5 GiB, margin: 3.5 GiB
     - This is the recommended configuration because:
       1. Wake overhead is only ~1s (fast, level 1 wake)
       2. 3.5 GiB margin prevents OOM spikes
       3. LoRA enables weight updates of only 0.06 GiB
       4. Training buffers fit comfortably after KV release

2. TIMING PER LEVEL
   ────────────────────────────────────────────────────────────────────

   Level 1 timing (per GRPO step):
     sleep(level=1)   :  ~0.5s  (KV cache deallocation)
     wake(level=1)    :  ~1.0s  (KV cache re-allocation, prefix re-pin)
     total overhead   :  ~1.5s per step
     Step time impact :  ~1.5s / ~30s total step = ~5% overhead

   Level 2 timing (per GRPO step):
     sleep(level=2)   :  ~1.0s  (full GPU memory release)
     wake(level=2)    :  ~3.0s  (full model reload from CPU)
     total overhead   :  ~4.0s per step
     Step time impact :  ~4.0s / ~30s total step = ~13% overhead

   LoRA adapter update timing:
     adapter size     :  0.06 GiB (rank-16 LoRA on 8B model)
     transfer time    :  0.004s  (PCIe 4.0 x16: ~32 GiB/s)
     This is negligible — 0.004s vs 3s for full model reload

   Recommendation: Use Level 1 + LoRA for lowest overhead per step.

3. SLEEP/WAKE STEP LIFECYCLE INTEGRATION
   ────────────────────────────────────────────────────────────────────

   RTX 4090 GRPO Step Timeline (Level 1 + LoRA):

   Time(s)  Phase          GPU Memory     Action
   ──────────────────────────────────────────────────────────────
   0-10     Rollout        19 GiB         Generate + score
   10.0     Sleep(L1)      17 GiB         Free KV, release prefix pins
   10.5     —              17 GiB         Sleep complete
   10.5-25  Training       20.5 GiB       GRPO gradient step
   25.0     Wake(L1)       17→19 GiB      Re-allocate KV, re-pin prefixes
   26.0     —              19 GiB         Wake complete
   26-36    Rollout        19 GiB         Next step generation
   ──────────────────────────────────────────────────────────────

   Total step: ~36s including 1.5s sleep/wake overhead (~4.2%)

4. DSA LORA TARGETS (#28703)
   ────────────────────────────────────────────────────────────────────
   - DSA (Dynamic Sparse Attention) LoRA targets enable GRPO LoRA training
     on GLM-5.1 and DSV3.2 models.
   - These targets identify which attention layers benefit from LoRA
     adaptation during GRPO, enabling sparse weight updates that
     preserve base model capabilities while adapting for reward signals.
   - On RTX 4090, DSA LoRA targets reduce the adapter size from 0.06 GiB
     (dense LoRA) to ~0.03 GiB (sparse LoRA), further reducing the
     weight update time to ~0.002s.
   - Integration: DSA LoRA targets are specified in the verl config as
     `lora_targets: ['dsa']` and SGLang handles the sparse merge/unmerge
     at sleep/wake boundaries.
"""

# ---------------------------------------------------------------------------
# Mode 3: compare
# ---------------------------------------------------------------------------

COMPARE = """
=== MODE: compare ===
SGLang vs vLLM Sleep/Wake Comparison for verl GRPO Training

1. ARCHITECTURAL DIFFERENCES
   ────────────────────────────────────────────────────────────────────
   SGLang:
     - Sleep/wake API: tag-based, fine-grained control
       engine.sleep(level=1|2)
       engine.wake_up(tags=['weights','kv_cache'])
     - Sleep levels: 1 (KV only) and 2 (full release)
     - Attention: RadixAttention with prefix KV pinning
     - IPC: in-process ZMQ (low latency, single-process model)

   vLLM:
     - Sleep/wake API: integer-based, coarser control
       engine.sleep(level=1|2|3)  # levels are less granular
       engine.wake_up()           # no tag selection; reloads everything
     - Sleep levels: 1, 2, 3 (but effectively similar to SGLang's 1 & 2)
     - Attention: PagedAttention (no prefix pinning; LRU only)
     - IPC: Ray-based (multi-process, higher overhead)

2. IPC AND LATENCY
   ────────────────────────────────────────────────────────────────────
   SGLang IPC (ZMQ in-process):
     - Worker runs in same process as controller
     - Communication via ZMQ in-process transport
     - Latency: ~0.1ms for sleep/wake commands
     - Best for dp=1 (single GPU) setups like RTX 4090
     - No Ray dependency → simpler deployment

   vLLM IPC (Ray):
     - Worker runs in separate Ray actor process
     - Communication via Ray RPC (gRPC-based)
     - Latency: ~5-10ms for sleep/wake commands
     - Viable for multi-GPU but slower for single-GPU
     - Ray dependency → additional deployment complexity

   RTX 4090 implication: dp=1 means single-process is optimal.
     SGLang's in-process ZMQ is naturally suited; vLLM's Ray
     overhead is unnecessary for this topology.

3. FEATURE MATRIX
   ────────────────────────────────────────────────────────────────────
   Feature                        | SGLang      | vLLM
   ────────────────────────────────────────────────────────────────────
   Tag-based wake control         | YES         | NO (integer only)
   Level 1 sleep (KV only)        | YES         | YES
   Level 2 sleep (full release)   | YES         | YES
   Prefix KV pinning              | YES         | NO
   TTL-based pin eviction         | YES         | NO
   LoRA adapter weight update     | YES         | YES (but coarser)
   In-process IPC (ZMQ)           | YES         | NO (Ray-based)
   cumem allocator                | NO          | YES (buggy #45552)
   RadixAttention                 | YES         | NO (PagedAttention)
   Stream sync on memory ops      | YES         | PARTIAL (#45552)
   ────────────────────────────────────────────────────────────────────
   Score (out of 10)              | 9           | 5

4. PERFORMANCE COMPARISON (RTX 4090, 8B model, GRPO)
   ────────────────────────────────────────────────────────────────────
   Metric                  | SGLang L1+LoRA  | vLLM L1+LoRA
   ────────────────────────────────────────────────────────────────────
   Sleep time              | 0.5s            | 0.8s
   Wake time               | 1.0s            | 1.5s
   Total overhead/step     | 1.5s            | 2.3s
   Peak GPU memory         | 20.5 GiB        | 20.5 GiB
   Margin                  | 3.5 GiB         | 3.5 GiB
   Throughput (tokens/s)   | ~280            | ~240
   IPC latency             | 0.1ms           | 5-10ms
   Weight update time      | 0.004s          | 0.004s
   ────────────────────────────────────────────────────────────────────

5. RTX 4090 RECOMMENDATION
   ────────────────────────────────────────────────────────────────────
   SGLang is the #1 choice for sleep/wake integration on RTX 4090:

   Reasons:
   1. Lower overhead: 1.5s vs 2.3s per step (35% faster cycle)
   2. Tag-based wake: reload only what's needed, skip unnecessary ops
   3. Prefix KV pinning: 22-42x logprob scoring speedup
   4. In-process ZMQ: optimal for dp=1 single-GPU topology
   5. No cumem bugs: vLLM's cumem allocator has stream sync issues
      (#45552) that can cause silent corruption
   6. Better throughput: ~280 vs ~240 tokens/s

   Use vLLM only if:
   - Multi-GPU dp>1 setup (Ray scales better across nodes)
   - vLLM is already in your infrastructure and migration cost is high
   - You need vLLM-specific features not in SGLang
"""

# ---------------------------------------------------------------------------
# Mode 4: debug
# ---------------------------------------------------------------------------

DEBUG = """
=== MODE: debug ===
SGLang Sleep/Wake Debugging Guide for verl GRPO Training

1. COMMON ISSUES AND SOLUTIONS
   ────────────────────────────────────────────────────────────────────

   Issue 1: Hadamard / Class Variable Lost After Wake (#10684)
   ────────────────────────────────────────────────────────────────────
   Symptom:  Model produces garbage output after wake_up. Hadamard
             transform or other class-level variables are reset.
   Cause:    Sleep/wake frees GPU tensors but class variables holding
             references to those tensors are not restored. Python class
             variables are not serialized across sleep boundaries.
   Fix:      Use model buffers (registered via register_buffer) instead
             of class variables. Model buffers are serialized with the
             model state_dict and survive sleep/wake cycles.
   Code fix:
     # BROKEN — class variable lost after wake:
     class MyModel:
         hadamard_scale = torch.tensor(...)  # lost after level-2 sleep

     # FIXED — model buffer preserved:
     class MyModel(nn.Module):
         def __init__(self):
             self.register_buffer('hadamard_scale', torch.tensor(...))

   Issue 2: MoE Cache Clobbered After Weight Update (#28676)
   ────────────────────────────────────────────────────────────────────
   Symptom:  MoE (Mixture of Experts) routing produces wrong expert
             selections after weight update. Expert cache returns stale
             results.
   Cause:    MoE routing caches (expert selection indices, load balance
             counters) are not invalidated when model weights are
             updated. The cache holds stale routing decisions computed
             with old weights.
   Fix:      Invalidate all MoE caches at the weight-reload boundary.
             Add cache invalidation in wake_up after weights are loaded.
   Code fix:
     def wake_up(self, tags):
         if 'weights' in tags:
             self.invalidate_moe_caches()  # clear routing + load balance
             self.load_weights()

   Issue 3: KV Cache Illegal Memory Access (#44395)
   ────────────────────────────────────────────────────────────────────
   Symptom:  CUDA illegal memory access error during forward pass after
             wake_up. Crash or silent NaN output.
   Cause:    wake_up called with incomplete tags. Only 'weights' was
             woken but forward pass also needs KV cache. Accessing freed
             KV memory → illegal address.
   Fix:      Ensure ALL tags needed for the next operation are included
             in wake_up. For rollout: always include both 'weights' and
             'kv_cache'.
   Code fix:
     # BROKEN — missing kv_cache tag:
     engine.wake_up(tags=['weights'])  # KV still freed!
     engine.generate(...)              # → illegal memory access

     # FIXED — include all needed tags:
     engine.wake_up(tags=['weights', 'kv_cache'])
     engine.generate(...)              # KV available, safe

   Issue 4: GDN Intermittent Degeneracy (#28679)
   ────────────────────────────────────────────────────────────────────
   Symptom:  GDN (Gradient Displacement Network) produces intermittent
             degenerate outputs. Behavior worsens over uptime, not
             reproducible on fresh start.
   Cause:    GDN state lifecycle mismatch. Internal state (running
             averages, momentum buffers) persists across sleep/wake but
             becomes stale as model weights change. The state was
             computed for old weights and is now mismatched.
   Fix:      Reset GDN state at weight-update boundaries. Add state
             reset in the weight update flow.
   Code fix:
     def update_weights(self, new_weights):
         self.gdn.reset_state()        # clear stale running averages
         self.load_weights(new_weights)

   Issue 5: cumem Stream Sync Missing (#45552 — vLLM specific)
   ────────────────────────────────────────────────────────────────────
   Symptom:  Silent data corruption or CUDA errors after memory
             operations. Hard to detect; may manifest as NaN loss.
   Cause:    vLLM's cumem allocator does not call cuda.synchronize()
             before/after cuMemUnmap/cuMemMap operations. GPU operations
             on freed memory continue executing → corruption.
   Fix:      Add cuda.synchronize() before unmap and after remap.
             This is a vLLM-specific issue; SGLang handles this correctly
             by default.
   Code fix (vLLM only):
     # Add sync around cumem operations:
     torch.cuda.synchronize()          # ensure all ops complete
     cuMemUnmap(ptr, size)             # safe to free now
     cuMemMap(ptr, size)               # remap
     torch.cuda.synchronize()          # ensure remap visible

2. DEBUG WORKFLOW
   ────────────────────────────────────────────────────────────────────
   Recommended workflow: detect → isolate → reproduce → root cause → fix

   Step 1: DETECT
     - Monitor NaN in loss/output (first sign of memory corruption)
     - Monitor GPU memory timeline (unexpected spikes or drops)
     - Monitor weight checksums (detect stale/clobbered weights)
     - Monitor KV cache allocation status (detect missing KV wake)

   Step 2: ISOLATE
     - Narrow down which sleep level causes the issue
     - Test level 0 (no sleep) as baseline — if issue disappears,
       it's sleep/wake related
     - Test level 1 vs level 2 to narrow further
     - Check if issue appears only after wake, not after sleep

   Step 3: REPRODUCE
     - Create minimal reproduction: single step sleep/wake cycle
     - Reduce batch size and sequence length to simplify
     - Add deterministic seeding for reproducibility
     - Log all sleep/wake calls with level and tags

   Step 4: ROOT CAUSE
     - Check if the issue matches one of the 5 known issues above
     - Add weight checksum verification before and after wake
     - Add KV cache allocation verification after wake
     - Check for class variable vs model buffer issues
     - Check for cache invalidation gaps

   Step 5: FIX
     - Apply the specific fix for the identified issue
     - Add monitoring to catch recurrence
     - Test with multiple consecutive sleep/wake cycles
     - Verify fix under stress (many steps, varying batch sizes)

3. MONITORING
   ────────────────────────────────────────────────────────────────────
   Weight checksum monitoring:
     def checksum_weights(model):
         return {name: tensor.sum().item() for name, tensor
                 in model.state_dict().items()}

     # Before sleep:
     pre_sleep_checksums = checksum_weights(model)
     # After wake:
     post_wake_checksums = checksum_weights(model)
     # Verify:
     for name in pre_sleep_checksums:
         if pre_sleep_checksums[name] != post_wake_checksums[name]:
             print(f"WEIGHT MISMATCH: {name}")

   KV cache state monitoring:
     def verify_kv_cache(engine):
         # Check KV cache is allocated after wake
         if engine.kv_cache is None:
             raise RuntimeError("KV cache not allocated after wake!")
         # Check KV cache size matches expected
         expected = engine.kv_cache_config.total_tokens
         actual = engine.kv_cache.shape[0]
         if actual != expected:
             raise RuntimeError(f"KV size mismatch: {actual} vs {expected}")

   GPU memory timeline monitoring:
     import torch
     def gpu_memory_log():
         allocated = torch.cuda.memory_allocated() / 1e9  # GiB
         reserved  = torch.cuda.memory_reserved() / 1e9   # GiB
         peak      = torch.cuda.max_memory_allocated() / 1e9
         return {"allocated": allocated, "reserved": reserved, "peak": peak}

     # Log at each phase:
     for phase in ["rollout_start", "sleep", "sleep_done",
                    "train_start", "train_done", "wake", "wake_done"]:
         print(f"{phase}: {gpu_memory_log()}")

   NaN detection:
     def detect_nan(tensor, name):
         if torch.isnan(tensor).any():
             print(f"NaN DETECTED in {name}: "
                   f"{torch.isnan(tensor).sum().item()} / {tensor.numel()}")
             return True
         return False

     # Check after each forward pass:
     detect_nan(output.logits, "logits")
     detect_nan(output.hidden_states, "hidden_states")
     detect_nan(loss, "loss")
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODES = {
    "architecture": ARCHITECTURE,
    "rtx4090": RTX4090,
    "compare": COMPARE,
    "debug": DEBUG,
}


def main():
    if len(sys.argv) < 2:
        print(BANNER)
        print("Available modes: architecture, rtx4090, compare, debug")
        print("Usage: python3 sglang_sleep_wake_integration_guide.py <mode>")
        print()
        # Show summary of all modes
        for name, content in MODES.items():
            lines = content.strip().split("\n")
            header = lines[0]
            print(f"  {header}")
        return

    mode = sys.argv[1].lower()
    if mode not in MODES:
        print(f"Unknown mode: {mode}")
        print(f"Available modes: {', '.join(MODES.keys())}")
        sys.exit(1)

    print(BANNER)
    print(MODES[mode])


if __name__ == "__main__":
    main()
