#!/usr/bin/env python3
"""AI Infra Interview Preparation Guide -- 7-Framework Knowledge Base

Modes:
  quiz      -- Random quiz questions from the 7-framework KB (50 Qs, 5 categories)
  deep      -- Deep technical interview questions (20 Qs, multi-framework synthesis)
  scenario  -- Scenario-based interview questions (10 Qs, practical problem-solving)
  rtx4090   -- RTX 4090 specific interview questions (15 Qs, deployment focus)

Usage:
  python3 ai_infra_interview_prep_guide.py quiz [--count N]
  python3 ai_infra_interview_prep_guide.py deep [--count N]
  python3 ai_infra_interview_prep_guide.py scenario [--count N]
  python3 ai_infra_interview_prep_guide.py rtx4090 [--count N]
"""

import argparse
import random
import sys
import textwrap

# -- ANSI helpers --

BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
ORANGE  = "\033[38;5;208m"
PINK    = "\033[38;5;213m"

def c(color, text):
    return f"{color}{text}{RESET}"

def difficulty_bar(d):
    bars = "|" * d + "." * (5 - d)
    colors = [GREEN, GREEN, YELLOW, ORANGE, RED]
    return c(colors[min(d - 1, 4)], bars)

def box(title, body, color=CYAN):
    width = 76
    sep = color + "+" + "-" * (width - 2) + "+" + RESET
    inner = color + "|" + RESET
    lines = textwrap.wrap(body, width - 4)
    out = [sep, f"{inner} {c(BOLD + color, title)}"]
    for l in lines:
        out.append(f"{inner} {l}")
    out.append(sep)
    return "\n".join(out)


# -- QUESTION BANKS --

QUIZ_QUESTIONS = {
    "Distributed Training": [
        {"q": "What are the four main parallelism strategies for distributed LLM training?",
         "a": "Tensor Parallelism (TP) -- shard individual ops across GPUs; Pipeline Parallelism (PP) -- shard layers into stages; FSDP/ZeRO -- shard optimizer states, grads, params across data-parallel ranks; Sequence Parallelism (SP) -- shard sequence dimension for attention/layernorm.",
         "ref": "FSDP2, DeepSpeed ZeRO, Megatron-LM", "diff": 1},
        {"q": "Why does FSDP2 use per-parameter sharding instead of per-module sharding?",
         "a": "Per-module sharding (FSDP1) forces entire modules to be gathered at once, wasting memory. Per-parameter sharding (FSDP2) allows fine-grained gather/scatter -- only the parameters needed for the current op are materialized, reducing peak memory significantly.",
         "ref": "FSDP2 torch.distributed._shard spec", "diff": 2},
        {"q": "Explain the difference between ZeRO-1, ZeRO-2, and ZeRO-3.",
         "a": "ZeRO-1 shards optimizer states (4x memory reduction for optimizer). ZeRO-2 shards optimizer states + gradients (8x reduction). ZeRO-3 shards optimizer states + gradients + parameters (dp times reduction). ZeRO-3 requires all-gather for forward/backward, adding communication overhead.",
         "ref": "DeepSpeed ZeRO stages", "diff": 2},
        {"q": "Why is ZeRO-3 broken for PEFT LoRA training on a single GPU?",
         "a": "ZeRO-3 partitions parameters across ranks, then all-gathers them for computation. With LoRA, only the LoRA adapters need gradients, but ZeRO-3 still all-gathers the entire base model every step, wasting bandwidth and memory. ZeRO-2 is the right choice -- it keeps parameters local and only shards optimizer states/gradients for the LoRA params.",
         "ref": "DeepSpeed ZeRO-3 + LoRA conflict", "diff": 3},
        {"q": "What is the communication volume for FSDP2 forward+backward on a model with P parameters and dp=N?",
         "a": "Forward: all-gather P/N per param group, total approx P. Backward: all-gather P/N again for grad computation, then reduce-scatter gradients P/N. Total comm = 2P (all-gather) + P/N (reduce-scatter) approx 2P+P/N bytes. For dp=1, comm=0 (no sharding needed).",
         "ref": "FSDP2 comm analysis", "diff": 3},
        {"q": "How does Pipeline Parallelism introduce bubble overhead, and what mitigations exist?",
         "a": "PP splits layers into stages; each stage must wait for the previous stage's output (forward) and next stage's gradient (backward). This creates idle bubble time. Mitigations: 1) Interleaved 1F1B scheduling; 2) Virtual stages -- each GPU holds multiple stage chunks; 3) Gradient accumulation with larger micro-batch counts.",
         "ref": "Megatron-LM PP scheduler", "diff": 3},
        {"q": "What is Sequence Parallelism and how does it interact with TP?",
         "a": "SP shards the sequence dimension of operations that do not require full-sequence input (LayerNorm, Dropout, residual add). With TP, these ops are redundant across ranks. SP makes each rank compute only seq_len/tp_size tokens, then all-reduce for attention. SP reduces activation memory by tp_size.",
         "ref": "Megatron-LM SP design paper", "diff": 4},
        {"q": "Explain why overlap_comm=False is mandatory on dp=1 in DeepSpeed ZeRO.",
         "a": "On dp=1, ZeRO has no data-parallel peers to communicate with. The overlap_comm feature launches NCCL communication asynchronously alongside computation. On dp=1, NCCL AllReduce is identity (returns input unchanged), but it still allocates NCCL buffers and launches kernel rendezvous, which can corrupt memory or produce NaN due to race conditions on unused communication streams.",
         "ref": "DeepSpeed overlap_comm bug on dp=1", "diff": 4},
        {"q": "How does FSDP2's forward-backward hook mechanism differ from FSDP1?",
         "a": "FSDP1 registers pre-forward hooks on FSDP modules to all-gather parameters and post-backward hooks to reduce-scatter gradients. Entire modules are gathered/scattered. FSDP2 uses per-parameter sharding; each parameter has its own gather/scatter lifecycle tied to the specific computation that needs it. This eliminates module-level all-gather waste and enables fine-grained prefetching.",
         "ref": "FSDP2 per-parameter vs FSDP1 per-module", "diff": 4},
        {"q": "Why is NCCL AllReduce = identity on dp=1? What are the implications?",
         "a": "NCCL AllReduce on 1 rank has no other ranks to reduce with, so it returns the input tensor unchanged -- mathematically identity. Implications: 1) Still allocates NCCL buffers; 2) Launches GPU kernel that rendezvous with itself; 3) If overlap_comm=True, this runs on a separate CUDA stream, creating race conditions; 4) Can silently corrupt tensor values or produce NaN.",
         "ref": "NCCL single-rank behavior, DeepSpeed overlap_comm", "diff": 5},
    ],
    "Inference Engine": [
        {"q": "What is the KV cache and why is it central to LLM inference?",
         "a": "The KV cache stores Key and Value projections from all previous tokens, so each new decode step computes attention against cached KV vectors instead of recomputing for all prior tokens. Without KV cache, decoding is O(n^2) per step; with it, each step is O(n) in memory reads but O(1) in compute.",
         "ref": "vLLM PagedAttention, SGLang RadixCache", "diff": 1},
        {"q": "Explain PagedAttention in vLLM and how it solves KV cache fragmentation.",
         "a": "Traditional KV cache allocates contiguous memory per sequence, causing fragmentation. PagedAttention divides KV cache into fixed-size blocks (pages), similar to virtual memory paging. Each sequence's KV blocks are tracked in a block table, allowing non-contiguous physical storage. This eliminates fragmentation, enables prefix sharing, and allows dynamic preemption/reallocation.",
         "ref": "vLLM PagedAttention paper", "diff": 2},
        {"q": "What is CUDA Graph capture and why does it accelerate LLM inference?",
         "a": "CUDA Graph records GPU kernel launches into a single executable graph. On replay, the entire graph submits in one operation, eliminating CPU-side kernel launch overhead (~5-50us each). For decode with many small kernels (GEMV, attention, sampling), cudagraph can reduce step latency 30-50%. Caveat: requires fixed tensor shapes.",
         "ref": "vLLM cudagraph, SGLang cudagraph", "diff": 2},
        {"q": "How does SGLang's RadixCache differ from vLLM's PagedAttention for prefix caching?",
         "a": "vLLM uses a block table with copy-on-write for prefix sharing. SGLang RadixCache uses a radix tree where nodes represent shared prefix segments; nodes are reference-counted and evicted when ref=0. RadixCache enables efficient tree-level prefix matching -- finding longest matching prefix node and reusing KV blocks directly, avoiding copy-on-write overhead.",
         "ref": "SGLang RadixCache design", "diff": 3},
        {"q": "Explain the vLLM encoder cache stale state bug after RLHF weight update.",
         "a": "vLLM's encoder cache (for multimodal models) persists stale KV entries computed with old weights after training updates model weights. New decode steps read stale encoder KV, producing incorrect attention outputs. Fix: force encoder cache eviction on weight update via a cache flush event.",
         "ref": "vLLM encoder cache bug in RLHF loop", "diff": 3},
        {"q": "What is chunked prefill and why is it important for long-context serving?",
         "a": "Chunked prefill splits long prompt prefill computation into smaller chunks interleaved with decode steps. Without it, a long prompt monopolizes the GPU for its entire prefill duration (10s for 100k tokens), blocking all decode steps. Chunked prefill maintains decode throughput while gradually processing long prefills.",
         "ref": "vLLM chunked prefill, SGLang chunked prefill", "diff": 3},
        {"q": "Compare SGLang tag-based sleep/wake vs vLLM integer-based weight update mechanism.",
         "a": "SGLang uses string tags (step_5, epoch_2) for weight version identification. Workers flush caches on tag mismatch. Tags are human-readable and debuggable. vLLM uses integer counters incremented on each update. Integer is faster but less debuggable. For RTX 4090 GRPO, tag-based is better -- you can trace which step caused cache clobber.",
         "ref": "SGLang sleep/wake, vLLM weight update sync", "diff": 4},
        {"q": "Why does cudagraph on DSV4 MoE models require special handling on RTX 4090?",
         "a": "CUDA Graph requires fixed tensor shapes at capture time. DSV4 MoE has dynamic expert routing producing variable-size intermediate tensors. On SM89, dispatched expert GEMM shapes change per-step, invalidating captured graph. Solutions: 1) Pad expert outputs to max size; 2) Grouped GEMM with fixed expert counts; 3) Disable cudagraph for MoE layers only.",
         "ref": "DSV4 MoE cudagraph, SM89 compatibility", "diff": 4},
        {"q": "How does vLLM's scheduling policy affect throughput and latency tradeoffs?",
         "a": "vLLM uses priority-based scheduler preferring decode over prefill. Handles preemption by evicting longest sequences when KV cache is full (recompute later). Maximizes throughput but can increase TTFT under high load. SGLang similar but RadixCache prefix sharing reduces preemption frequency.",
         "ref": "vLLM scheduler, SGLang scheduler", "diff": 4},
        {"q": "Explain the KV cache memory budget calculation for a 7B model on RTX 4090.",
         "a": "RTX 4090 has 24 GiB VRAM. 7B bf16 weights = 14 GiB. Activation ~0.5 GiB. Remaining for KV ~9.5 GiB. KV per token = 2*32*4096*2*32/32 = 512 bytes/token (Llama-7B). Capacity = 9.5 GiB/512 bytes ~19.5M tokens ~610 sequences at 32k context.",
         "ref": "KV cache budget math", "diff": 5},
    ],
    "RL Training": [
        {"q": "What is GRPO (Group Relative Policy Optimization) and how does it differ from PPO?",
         "a": "GRPO eliminates the value/critic network required by PPO. Instead, it uses group-level statistics: generate G responses per prompt, compute rewards, then normalize rewards within the group as advantages (mean=0, std=1). Policy gradient uses these normalized rewards with PPO-like clipping. Reduces memory (no critic), compute, and hyperparameters.",
         "ref": "DeepSeek GRPO paper, verl GRPO implementation", "diff": 1},
        {"q": "Why does gs=1 (group_size=1) degenerate to REINFORCE? Prove it.",
         "a": "With gs=1, G=1 sample. mean = r_1, std = 0. advantage = (r_1 - r_1)/0 = undefined. Path A: set advantage=0 -> zero gradient, training collapses. Path B: skip normalization, advantage = r_1, which is exactly REINFORCE (no baseline, high variance). Either way, training fails.",
         "ref": "GRPO gs=1 degeneration proof", "diff": 2},
        {"q": "Explain the verl TransferQueue and how it decouples rollout from training.",
         "a": "verl V1 uses producer-consumer architecture: rollout engine pushes trajectories (prompt, response, reward, log_prob) to TransferQueue (Ray-based shared memory buffer). Training engine pulls from queue. Decoupling allows rollout and training to run at different rates -- rollout batches groups while training processes sequentially.",
         "ref": "verl V1 architecture, TransferQueue", "diff": 3},
        {"q": "What is the sleep/wake mechanism and why is it critical for GRPO on single-GPU?",
         "a": "On single-GPU, the same GPU must run both rollout and training. SLEEP = unload inference engine (free KV, weights from inference allocator) for training. WAKE = reload weights, rebuild KV for next rollout. Without sleep/wake, inference memory persists during training causing OOM. SGLang tag-based flushes on mismatch; vLLM integer-based increments counter.",
         "ref": "SGLang sleep/wake, verl single-GPU GRPO", "diff": 3},
        {"q": "How does LoRA+bypass reduce memory from 90 GiB to 22.9 GiB for GRPO on RTX 4090?",
         "a": "Full PPO: 4 models = 56 GiB. GRPO (no critic): 42 GiB. Still > 24 GiB. LoRA+bypass: freeze base (14 GiB, shared between policy and reference), train LoRA adapters (r=32, 0.5 GiB). Reference = base + frozen LoRA copy (0 GiB extra). Rule-based reward (0 GiB). Result: 14+0.5+activations ~22.9 GiB.",
         "ref": "LoRA+bypass memory math, GRPO advantage", "diff": 3},
        {"q": "Explain the GRPO advantage: A_i = (r_i - mean(r)) / std(r). What happens with small group sizes?",
         "a": "With gs=G: advantage normalized from G samples. Small G (G=4) produces noisy mean/std estimates. Single outlier reward skews entire group. Clipping partially mitigates but gradient variance is O(1/G). Minimum: G>=4 for stability, G>=8 recommended, G>=16 for low variance.",
         "ref": "GRPO variance analysis", "diff": 3},
        {"q": "What is the rLLM GRPO advantage zero-gradient bug and how do you fix it?",
         "a": "When all rewards in a group are identical, std=0 -> advantage NaN or zero. rLLM set advantage=0 causing zero gradient and training collapse. Fix: when std<eps, either (1) use advantage = reward - mean (skip normalization), (2) skip gradient step, or (3) increase gs for reward diversity.",
         "ref": "rLLM zero-gradient bug", "diff": 4},
        {"q": "How does PPO's clipping mechanism work and why does GRPO adopt it?",
         "a": "PPO clips ratio r(t) = pi_new/pi_old to [1-eps,1+eps] before multiplying by advantage: min(r*A, clip(r)*A). Prevents large policy updates. GRPO adopts same clipping with group-normalized advantages instead of critic-estimated. Clip prevents single high-reward sample from causing excessive policy shift with noisy group advantages.",
         "ref": "PPO clipping, GRPO clipping", "diff": 4},
        {"q": "Explain the verl V1 vs V2 architecture differences for GRPO training.",
         "a": "verl V1: producer-consumer TransferQueue, decoupled but requires synchronization. verl V2: monolithic actor-worker, same GPU for rollout and training, no queue, sleep/wake handles memory transition. V2 simpler but requires single-GPU memory management.",
         "ref": "verl V1 vs V2 architecture", "diff": 4},
        {"q": "Why is GRPO training prone to NaN after just a few steps on RTX 4090?",
         "a": "6 causes: 1) overlap_comm=True on dp=1 -> NCCL race; 2) bf16 gradient overflow (advantage * log_prob exceeds +/-65504); 3) LoRA scaling amplifies gradients; 4) reward=0 for all -> std=0 -> NaN; 5) DeepSpeed zero3 optimizer bug; 6) SGLang stale KV -> corrupted logits -> corrupted rewards. Each has specific MUST DO fix.",
         "ref": "GRPO NaN debugging, 7-framework common pitfalls", "diff": 5},
    ],
    "GPU/Driver": [
        {"q": "What is a CUDA stream and why are multiple streams useful?",
         "a": "CUDA stream = sequence of GPU commands executed in order. Different streams can run concurrently. Default stream (0) blocks all others. Multiple streams enable: compute-communication overlap (NCCL stream 7 + compute stream 0), kernel concurrency, memory transfer overlap. DeepSpeed uses stream 7 for overlap_comm.",
         "ref": "CUDA streams, DeepSpeed overlap_comm", "diff": 1},
        {"q": "Explain the RTX 4090 memory hierarchy: L2 cache, HBM, host RAM.",
         "a": "RTX 4090 SM89: L2 = 72 MiB (fast, on-chip); HBM = 24 GiB GDDR6X (~1 TB/s); Host RAM = 32-64 GiB DDR5 (~50 GB/s PCIe 4.0 x16). Data flow: L2->HBM->Host. LLM: weights in HBM, active layer in L2, KV cache in HBM, offloaded tensors in host. Sequential decode with small KV hits L2 heavily.",
         "ref": "RTX 4090 memory specs, SM89 architecture", "diff": 2},
        {"q": "What is SM89 (Ada Lovelace) and how does it differ from SM80 (Ampere)?",
         "a": "SM89 RTX 4090 vs SM80 A100: 1) FP8 hardware support (new Tensor Core mode); 2) DP4A int8 dot product; 3) 72 MiB L2 vs 40 MiB; 4) 128 SMs vs 108; 5) 24 GiB GDDR6X vs 40/80 GiB HBM2e; 6) Consumer GPU lacks native bf16 Tensor Core (FP32 emulation for some ops). FP8 enables FP8 KV cache and training on 4090.",
         "ref": "SM89 vs SM80, Ada Lovelace specs", "diff": 2},
        {"q": "Explain FP8 (E4M3/E5M2) format and its significance for RTX 4090 training.",
         "a": "FP8 E4M3: 4 exp + 3 mantissa, range +/-448, for forward activations/weights. FP8 E5M2: 5 exp + 2 mantissa, range +/-57344, for backward gradients. SM89 Tensor Cores native FP8 = 2x throughput vs bf16. FP8 training: forward E4M3, backward E5M2, master weights bf32. 2x faster GEMM, 50% memory reduction.",
         "ref": "FP8 format, NVIDIA Transformer Engine", "diff": 3},
        {"q": "What is the CUDA stream race condition in DeepSpeed overlap_comm on dp=1?",
         "a": "DeepSpeed launches NCCL AllReduce on stream 7 concurrently with gradient computation on stream 0. On dp=1, AllReduce is identity -- reads gradient, processes, writes same values back. NCCL kernel rendezvous creates timing dependency. If NCCL reads before stream 0 finishes writing, or writes back while stream 0 modifies, tensor corrupted. Classic multi-stream race on shared memory.",
         "ref": "DeepSpeed #8080, CUDA stream race", "diff": 4},
        {"q": "How does CUDA memory allocator work and why can it cause fragmentation?",
         "a": "PyTorch caching allocator: freed blocks go to pool instead of cudaFree(). Allocation searches pool for block >= size, splits remainder. Over time, different-sized allocations create fragmented pool where total free > requested but no single block large enough. Symptom: OOM despite memory_reserved() showing free space. Fix: torch.cuda.empty_cache().",
         "ref": "PyTorch CUDA caching allocator", "diff": 3},
        {"q": "Explain the NCCL buffer allocation problem on single-rank AllReduce.",
         "a": "NCCL allocates internal buffers (temp reduction + output) = 2x tensor size on single rank. Also registers CUDA IPC memory and rendezvous counters. On dp=1, buffers consume VRAM unnecessarily, NCCL kernel on separate stream races with compute, buffers persist across steps. Fix: disable NCCL on dp=1 (overlap_comm=False).",
         "ref": "NCCL single-rank overhead", "diff": 4},
        {"q": "Why does bf16 have limited range +/-65504 and how does this cause NaN in GRPO?",
         "a": "bf16: 8 exponent bits + 7 mantissa, range +/-65504, ~3 decimal digit precision. GRPO loss: ratio = exp(log_prob_new - log_prob_old) can be very large. ratio * advantage overflows bf16 -> inf. inf * 0 (from clipping) = NaN. Fix: compute ratio in fp32: torch.exp(log_ratio.float()), clip, then cast to bf16.",
         "ref": "bf16 overflow in PPO/GRPO", "diff": 4},
        {"q": "What CUDA profiling tools are essential for debugging GRPO on RTX 4090?",
         "a": "1) nsys -- timeline profiling, kernel order, stream concurrency; 2) ncu -- per-kernel profiling, occupancy, throughput, bandwidth; 3) torch.profiler -- PyTorch-level with TensorBoard export; 4) torch.cuda.memory_summary() -- allocator snapshot; 5) nvidia-smi dmon -- real-time GPU monitoring. For NaN: torch.autograd.detect_anomaly() locates exact op.",
         "ref": "CUDA profiling tools, GRPO debugging", "diff": 4},
        {"q": "Explain the RTX 4090 PCIe 4.0 x16 bandwidth limitation for GRPO training.",
         "a": "PCIe 4.0 x16 ~32 GB/s bidirectional (practical ~25 GB/s). Weight sync between inference and training: 7B bf16 = 14 GiB transfer takes ~0.56s. Unavoidable overhead. LoRA: only adapters (0.5 GiB ~20ms), but base still reloaded. PCIe bandwidth is fundamental bottleneck for single-GPU RL training.",
         "ref": "PCIe bandwidth, weight reload timing", "diff": 5},
    ],
    "Framework Bugs": [
        {"q": "What is the State Lifecycle Mismatch pattern family? Give 3 examples.",
         "a": "Inference engine state (KV, encoder, cudagraph) has different lifecycle than training state (weights, optimizer). When training updates weights but inference retains stale state, mismatch produces incorrect outputs. Examples: 1) vLLM KV cache with old weights persists; 2) SGLang MoE expert cache stale after LoRA update; 3) cudagraph replayed with new weights.",
         "ref": "State Lifecycle Mismatch pattern", "diff": 2},
        {"q": "Explain the Silent Corruption pattern family and why it is 500,500x more damaging than loud failures.",
         "a": "Silent Corruption: bugs producing incorrect but valid (not NaN/inf/OOM) results. Training continues on corrupted data. Loud failures crash immediately. Silent corruption persists ~500 steps undetected, 500 tokens * 2 (policy+reward) per step = 500,000 corrupted decisions. With cascading effects: ~500,500x more damaging.",
         "ref": "Silent corruption pattern, damage analysis", "diff": 3},
        {"q": "What is MUST DO Rule #1 for GRPO training on any framework?",
         "a": "MUST DO #1: overlap_comm=False when dp=1. Applies to DeepSpeed, FSDP, any NCCL framework. On dp=1, NCCL AllReduce is identity, introduces race conditions on separate CUDA streams. Eliminates NCCL stream entirely, preventing silent gradient corruption and NaN. Most common bug -- 4 of 7 frameworks have this.",
         "ref": "MUST DO overlap_comm=False on dp=1", "diff": 2},
        {"q": "What is MUST DO Rule #2 for weight updates in RL training?",
         "a": "MUST DO #2: Flush all inference engine caches (KV, encoder, cudagraph, MoE routing) after every weight update. Prevents State Lifecycle Mismatch. In verl: update_weight() triggers flush. In SGLang: sleep/wake handles this. In vLLM: manually invalidate caches. Skip this = stale caches = silent corruption.",
         "ref": "MUST DO cache flush after weight update", "diff": 2},
        {"q": "Explain the DSV4 systematic instability across 4 frameworks. What is the universal fix?",
         "a": "DSV4 MoE dynamic routing (variable top_k) breaks: 1) vLLM cudagraph (fixed shapes required); 2) SGLang MoE cache (static routing assumed); 3) DeepSpeed ZeRO-3 (static partitioning can not handle variable); 4) FSDP2 (deterministic gather/scatter). Universal fix: pad to max_k, grouped GEMM fixed counts, disable cudagraph for MoE, invalidate routing cache on update, use ZeRO-2.",
         "ref": "DSV4 cross-framework instability", "diff": 3},
        {"q": "What is MUST DO Rule #3 for GRPO advantage computation?",
         "a": "MUST DO #3: Guard std=0 in advantage normalization. When all rewards identical, std=0, (r-mean)/std=NaN. Fix: if std<1e-8, set advantages=0 and skip gradient step, or use unnormalized (r-mean). Prevents NaN propagation. Second most common NaN source after overlap_comm.",
         "ref": "MUST DO std=0 guard in GRPO", "diff": 3},
        {"q": "What is MUST DO Rule #4 for bf16 GRPO loss computation?",
         "a": "MUST DO #4: Compute clipping ratio in fp32 not bf16. bf16 range +/-65504; ratio exp(log_prob_new-log_prob_old) can exceed this. bf16 ratio -> inf -> inf*clipped_advantage = NaN. Fix: ratio = torch.exp(log_ratio.float()).clamp(1-eps,1+eps), then cast to bf16. Framework-independent.",
         "ref": "MUST DO fp32 ratio computation", "diff": 3},
        {"q": "Give 3 examples of the Initialization Ordering Bug pattern family.",
         "a": "Components initialized in wrong order causing stale references. 1) DeepSpeed ZeRO-3 initializes optimizer before parameters all-gathered -> wrong parameter shapes in optimizer; 2) SGLang tokenizer cache before weights loaded -> mismatched token-to-id; 3) verl TransferQueue before rollout ready -> empty queue pulls produce None tensors.",
         "ref": "Initialization Ordering Bug pattern", "diff": 3},
        {"q": "What is the Double Optimization bug in DeepSpeed ZeRO-3 with LoRA?",
         "a": "ZeRO-3 partitions all parameters including frozen base model. Optimizer creates states for ALL parameters, not just LoRA. optimizer.step() updates all partitions, silently modifying frozen weights. Violates LoRA contract (base must remain unchanged). Fix: ZeRO-2 (keeps params local) or exclude frozen from ZeRO partitioning.",
         "ref": "ZeRO-3 LoRA double optimization bug", "diff": 4},
        {"q": "List all 6 MUST DO rules for safe GRPO training across all 7 frameworks.",
         "a": "#1 overlap_comm=False on dp=1 (NCCL race). #2 Flush inference caches after weight update (state mismatch). #3 Guard std=0 in advantage (NaN from identical rewards). #4 Compute ratio in fp32 (bf16 overflow). #5 ZeRO-2 not ZeRO-3 for LoRA (double optimization). #6 ulimit nofile>=65536 (NCCL socket exhaustion).",
         "ref": "7-framework MUST DO rules", "diff": 5},
    ],
}


DEEP_QUESTIONS = [
    {"q": "Why does gs=1 degenerate to REINFORCE? Prove it mathematically.",
     "a": "GRPO advantage: A_i = (r_i - mu)/sigma, mu=mean(r_G), sigma=std(r_G). With gs=1, G=1 sample. mu=r_1, sigma=0. A_1=(r_1-r_1)/0 undefined.\n\nPath A: sigma<eps => A=0. Loss=-0*log pi=0. Gradient=0. Training collapses (zero-gradient).\n\nPath B: skip normalization, A=r_1. Loss=-r_1*log pi(a|s). This is EXACTLY REINFORCE: gradient=E[r*d/d_theta log pi], no baseline, maximal variance.\n\nVariance comparison:\nVar(REINFORCE)=E[r^2]-E[r]^2 (no reduction)\nVar(PPO)=E[(r-V(s))^2] (much smaller with good critic)\n\nConclusion: gs=1 either zero gradient (collapse) or REINFORCE (worst method). Minimum gs=4; gs>=8 recommended.",
     "ref": "GRPO variance theory, REINFORCE degeneration"},
    {"q": "Explain why overlap_comm=False is mandatory on dp=1 but beneficial on dp>1.",
     "a": "On dp>1: overlap_comm=True enables compute-communication overlap. NCCL reduces-scatters previous layer gradients on stream 7 while stream 0 computes current layer. 30-50% step time reduction.\n\nOn dp=1: NCCL AllReduce(X,1)=X (identity). No data exchange. But NCCL still: allocates 2x buffers, launches kernel rendezvous, runs on stream 7.\n\nRace condition: stream 0 writes gradient G; stream 7 reads/writes G for identity AllReduce. If stream 7 reads before stream 0 finishes writing or writes back during compute -> tensor corrupted.\n\nOn dp>1: NCCL communicates with other GPUs, data flows legitimate and synchronized. Benefits outweigh costs. On dp=1: zero benefits, non-zero costs (race+memory). Hence overlap_comm=False mandatory on dp=1.",
     "ref": "DeepSpeed overlap_comm, NCCL single-rank race"},
    {"q": "How does TransferQueue decouple rollout from training phases in verl V1?",
     "a": "verl V1 uses Ray actor-based TransferQueue between RolloutWorker and TrainingWorker.\n\nRolloutWorker pushes TrajectoryBatch (prompt_ids, response_ids, log_probs, rewards) to bounded TransferQueue (Ray shared memory). TrainingWorker pulls batches.\n\nBenefits: 1) Rollout batches groups before training needs them; 2) Training processes at own pace; 3) Different hardware configs independent; 4) Pipeline efficiency.\n\nPitfalls: 1) Queue overflow (rollout faster than training); 2) Queue starvation; 3) Memory pressure (5 GiB/step host RAM); 4) Stale trajectories (old weights violate on-policy requirement).",
     "ref": "verl V1 TransferQueue, Ray actor pipeline"},
    {"q": "Compare SGLang tag-based sleep/wake vs vLLM integer-based. Which is better for RTX 4090?",
     "a": "SGLang tag-based: string tags (step_42, grpo_epoch_3) identify weight versions. Workers flush caches on tag mismatch. Tags human-readable, debuggable, carry metadata.\n\nvLLM integer-based: global counter increments on each update. Workers check counter parity. Faster comparison, simpler implementation, but no semantic meaning.\n\nFor RTX 4090 GRPO: tag-based superior because: 1) Frequent updates (~44s/step), cache clobber events need debugging; 2) LoRA version tracking needs tags; 3) DSV4 MoE expert cache needs routing version.\n\nConclusion: SGLang tag-based better for RTX 4090 production GRPO.",
     "ref": "SGLang sleep/wake, vLLM weight update, RTX 4090 GRPO"},
    {"q": "Explain the State Lifecycle Mismatch pattern family with 6 concrete examples.",
     "a": "State Lifecycle Mismatch: inference state (longer lifecycle) becomes stale when training state (shorter lifecycle) updates.\n\n6 examples:\n1) KV Cache Stale: vLLM KV from W_old persists after update to W_new -> wrong logits -> wrong rewards -> silent corruption.\n2) Encoder Cache Stale: vision encoder outputs persist after RLHF update -> wrong attention for image prompts.\n3) CUDA Graph Replayed: captured graph with old weight pointers replayed after new weights -> stale memory or freed pointers.\n4) MoE Expert Cache: cached routing persists after LoRA update -> tokens dispatched to wrong experts.\n5) NCCL Buffer Old Gradients: overlap_comm buffers old gradients, identity-processed but can contaminate new via stream race.\n6) Prefix Cache Stale: RadixCache prefix KV from old weights persists -> new requests get stale KV.\n\nUniversal pattern: cache at time T persists beyond weight update at T+1. Fix: invalidate all caches (MUST DO #2).",
     "ref": "State Lifecycle Mismatch, 6 examples across 7 frameworks"},
    {"q": "Why is NCCL AllReduce = identity on dp=1? What are the implications?",
     "a": "NCCL AllReduce(X, 1 rank) = X. Proof: AllGather on 1 rank = [X]. ReduceScatter on 1 rank = X. Result: identity.\n\n5 implications:\n1) Memory waste: NCCL allocates 2x buffers (input+output) of O(param_count). 7B bf16 ~28 GiB unnecessary on 24 GiB GPU -> OOM.\n2) Race condition: identity AllReduce on stream 7, compute on stream 0, both access same gradient tensor -> no sync -> corruption or NaN.\n3) Latency: identity kernel ~1-5ms per step, unnecessary.\n4) Socket exhaustion: NCCL initializes sockets/IPC even for 1 rank, can exhaust ulimit nofile after ~1000 steps.\n5) Corruption cascade: one corrupted gradient -> optimizer -> weights -> inference KV -> reward -> advantage -> next gradient. Hundreds of steps before detection.\n\nConclusion: NCCL on dp=1 has zero benefits, 5 harm categories. MUST DO: overlap_comm=False.",
     "ref": "NCCL identity AllReduce, 5 implications"},
    {"q": "How does LoRA+bypass reduce memory from 90 GiB to 22.9 GiB on RTX 4090?",
     "a": "PPO: 4 models = 56 GiB. GRPO (no critic): 42 GiB. Both > 24 GiB.\n\nLoRA+bypass:\n1) Frozen base model 14 GiB (shared between policy and reference). LoRA adapters r=32 = 0.5 GiB trainable.\n2) Reference bypass: reference = base + frozen LoRA copy. Same weights, 0 GiB extra allocation.\n3) Rule-based reward: 0 GiB (no reward model).\n4) Gradient checkpointing: activations ~2 GiB (only current layer).\n\nTotal: 14 GiB base + 0.5 GiB LoRA + 2 GiB activations + 2 GiB optimizer + 0.4 GiB CUDA ~19 GiB. Peak: ~22.9 GiB during forward+backward.\n\nvs 90 GiB: full PPO 4 models, no LoRA, no checkpointing, ZeRO-3 overhead. LoRA+bypass achieves 4x reduction.",
     "ref": "LoRA+bypass memory math, 90 GiB -> 22.9 GiB"},
    {"q": "Explain the DSV4 systematic instability across 4 frameworks. What is the universal fix?",
     "a": "DSV4 MoE dynamic routing (variable top_k) creates variable-shape tensors, breaking 4 frameworks:\n\n1) vLLM cudagraph: fixed shapes required, dynamic routing invalidates captured graph. Replay reads wrong memory -> corruption/crash.\n2) SGLang MoE cache: assumes static routing, cached routing persists after LoRA update -> wrong expert dispatch.\n3) DeepSpeed ZeRO-3: static partitioning can not handle variable expert counts -> unnecessary or missing all-gathers.\n4) FSDP2: deterministic gather/scatter assumes predictable need, dynamic routing unpredictable.\n\nUniversal 5-point fix:\n1) Pad expert outputs to max_k (deterministic shapes)\n2) Grouped GEMM with fixed expert counts (zero-padded)\n3) Disable cudagraph for MoE layers (capture dense only)\n4) Invalidate MoE routing cache on every weight update\n5) ZeRO-2/FSDP local parameters (no expert sharding)",
     "ref": "DSV4 instability, 4 frameworks, universal 5-point fix"},
    {"q": "Compare ZeRO-2 vs ZeRO-3 for PEFT LoRA training. Why is ZeRO-3 broken?",
     "a": "ZeRO-2: shards optimizer+grads, keeps params LOCAL.\n- Base model: local, frozen, no optimizer states. LoRA: local, trainable, optimizer sharded.\n- Forward: no all-gather (params local). Backward: reduce-scatter LoRA grads only.\n- Memory: 14 GiB base + 0.5 GiB LoRA + small optimizer ~14.5 GiB.\n\nZeRO-3: shards optimizer+grads+PARAMS.\n- Base model: partitioned, all-gather ENTIRE 14 GiB every forward/backward step. LoRA: partitioned (0.5 GiB, negligible).\n- Forward/backward: 2x all-gather entire model per step. Communication ~4*14 GiB.\n\nZeRO-3 broken for LoRA:\n1) Unnecessary communication: all-gathering frozen base model wastes bandwidth.\n2) Double optimization: optimizer states for ALL params including frozen. step() modifies frozen weights -> LoRA contract violated.\n3) Memory paradox: reduces per-GPU param memory (14/N GiB) but needs all-gather buffer (14 GiB temp). Peak > ZeRO-2 on dp=1.\n4) Race condition: all-gather on dp=1 with overlap_comm=True = NCCL identity race.\n\nConclusion: ZeRO-2 correct for LoRA. ZeRO-3 adds overhead and risks corrupting frozen weights.",
     "ref": "ZeRO-2 vs ZeRO-3 for LoRA, 4 reasons ZeRO-3 broken"},
    {"q": "What is the silent corruption pattern family? Why is it 500,500x more damaging than loud failures?",
     "a": "Silent Corruption: bugs producing numerically valid (not NaN/inf/OOM) but semantically wrong outputs. Training continues on corrupted data.\n\n6 pattern members:\n1) Stale KV: old weights projections -> wrong logits -> wrong rewards -> wrong advantages -> wrong gradients.\n2) NCCL race dp=1: gradient partially overwritten, values in valid bf16 range, wrong direction.\n3) MoE routing mismatch: wrong expert dispatch, valid numeric but incorrect semantic.\n4) bf16 precision loss: ratio truncation -> wrong clipping -> gradient 10-50% off.\n5) Double optimization: frozen weights modified -> LoRA on wrong base model.\n6) Gradient accumulation stale: mid-accumulation weight update -> inconsistent gradient average.\n\nWhy 500,500x:\nLoud failure: crashes immediately, 1 step lost.\nSilent: persists ~500 steps undetected. Per step: 500 tokens * 2 (policy+reward) = 1000 corrupted decisions. 500*1000=500,000. With cascading effects (corrupted rewards -> advantages -> gradients -> weight updates): ~500,500x.\n\nCritical: MUST DO rules #1-#6 all address silent corruption.",
     "ref": "Silent corruption, 500,500x damage, 6 pattern members"},
    {"q": "Explain how FSDP2 host RAM grows 5 GiB/step during GRPO training. Diagnose and fix.",
     "a": "Root causes:\n1) Trajectory storage: gs*batch_size trajectories per step. Data ~4 KiB/trajectory. But the 5 GiB/step comes from FSDP2's parameter shard copying to gathered buffers that accumulate in host RAM without gc.\n2) FSDP2 all-gather copies parameters from sharded to contiguous buffer. Multi-step GRPO without gc -> buffers accumulate in CPU tensor cache.\n3) Weight sync serialization: entire model (14 GiB) serialized to CPU before sending to vLLM/SGLang.\n4) Gradient accumulation buffers not freed between micro-steps.\n\nDiagnosis:\n1) Monitor host RAM: ps aux RSS per step\n2) torch.cuda.memory_summary() per step\n3) torch.profiler: look for copy_ and gather CPU allocations\n\nFixes:\n1) torch.cuda.empty_cache() + gc.collect() after each step\n2) FSDP2 use_orig_params=True (avoid param copy overhead)\n3) Free trajectory buffers after optimizer.step()\n4) Ray shared memory for trajectory transfer (no CPU serialization)\n5) torch.utils.checkpoint cleanup hook after backward",
     "ref": "FSDP2 host RAM growth, GRPO memory management"},
    {"q": "Prove that GRPO with gs>=4 has lower variance than PPO with a learned critic.",
     "a": "Claim: GRPO gs=4 has lower effective variance than PPO with learned critic for reward-dominated tasks.\n\nPPO: A_PPO = r - V(s). Var(A_PPO) = Var(r) + Var(V(s)) - 2*Cov(r,V(s)).\nPerfect V(s)=E[r|s]: Var=Var(r)-Var(E[r|s])=E[Var(r|s)]. But V(s) never perfect early.\n\nGRPO: A_GRPO = (r_i - mean(r_G))/std(r_G). Var(A_GRPO) ~ Var(r)/gs.\nFor gs=4: Var(A_GRPO) ~ Var(r)/4.\n\nComparison:\nEarly training: V(s) poorly fitted, Var(V(s)) large. Var(A_PPO) = Var(r)+Var(V(s)) > Var(r) > Var(r)/4 = Var(A_GRPO). GRPO wins.\nAfter critic convergence: V(s)~E[r|s], Var(V(s))->0. Var(A_PPO)->E[Var(r|s)] which can be < Var(r)/4. PPO wins.\n\nGRPO practical advantages:\n1) No critic overhead (memory, compute, hyperparams)\n2) 1/gs variance reduction always\n3) No value function lag\n\ngs=4 minimum proof:\ngs=2: Var=Var(r)/2, but high probability identical rewards -> std=0 risk.\ngs=4: Var=Var(r)/4, probability all identical ~negligible.\n\nConclusion: gs>=4 GRPO practically lower variance than PPO early training.",
     "ref": "GRPO vs PPO variance proof, gs=4 minimum"},
    {"q": "Explain the verl V1->V2 migration and why V2 eliminates TransferQueue for single-GPU.",
     "a": "verl V1: producer-consumer TransferQueue between RolloutWorker and TrainingWorker (separate Ray actors).\n\nProblems on RTX 4090:\n1) TransferQueue stores trajectories in host RAM (5 GiB/step growth)\n2) Two Ray actors -> IPC overhead\n3) Both actors hold model weights -> 2x memory (28 GiB > 24 GiB -> OOM)\n4) Queue overflow/underflow requires manual tuning\n\nverl V2: monolithic ActorWorker running both rollout+training on same GPU.\n1) Single process, single GPU, no Ray IPC, no TransferQueue\n2) Sleep/wake handles memory transition\n3) In-process weight update, no serialization\n4) Trajectories in GPU memory during rollout, CPU for batch\n\nTransferQueue elimination necessary on single-GPU: RTX 4090 24 GiB. V1 needs inference(14)+training(14)=28 GiB > 24 GiB. Two Ray actors have separate CUDA memory pools.\n\nV2 sleep/wake: rollout peak 14+5+0.5=19.5 GiB (fits). Training peak 14+0.5+2+3=19.5 GiB (fits). TransferQueue only useful for multi-GPU decoupled pipelines.",
     "ref": "verl V1->V2 migration, TransferQueue elimination"},
    {"q": "Design the optimal sleep/wake timing for 7B LoRA GRPO on RTX 4090 with SGLang.",
     "a": "Step timing breakdown:\n\nRollout (SGLang inference):\n- Wake: load LoRA adapters (0.5 GiB, ~20ms)\n- Prefill: gs prompts*128 tokens (~2s for gs=8)\n- Decode: gs*512 tokens (~6s)\n- Reward: rule-based (~0.1s CPU)\n- Total: ~8s\n\nSleep:\n- Flush KV cache: free ~5 GiB (~50ms)\n- Flush RadixCache/routing (~20ms)\n- Free inference allocator (~50ms)\n- Total: ~120ms\n\nTraining (FSDP2/DeepSpeed):\n- Forward: gs trajectories, checkpointing (~12s)\n- Backward: LoRA only (~8s)\n- Optimizer step (~1s)\n- Total: ~21s\n\nWake next step:\n- Reload LoRA (~20ms), rebuild caches (~50ms)\n- Total: ~70ms\n\nTotal step: 8+0.12+21+0.07 = ~29.2s\n\nOptimizations:\n1) Prefetch LoRA during backward (save 20ms)\n2) cudagraph for dense layers only (decode 6s->4s)\n3) gs=16 amortizes but more memory\n4) fp8 KV cache (5 GiB->2.5 GiB)\n\nMemory timeline:\nRollout peak: 14+5+0.5=19.5 GiB (fits)\nTraining peak: 14+0.5+2+3=19.5 GiB (fits)\n4.5 GiB headroom for CUDA/NCCL/safety.",
     "ref": "Sleep/wake timing design, 7B GRPO on RTX 4090"},
    {"q": "Explain the CUDA stream synchronization bug family with 3 framework examples.",
     "a": "CUDA Stream Sync Bug: incorrect/missing sync between streams causes memory corruption or race conditions.\n\n3 examples:\n\n1) DeepSpeed overlap_comm (stream 0 vs 7): NCCL AllReduce on stream 7 concurrent with gradient compute on stream 0. On dp=1, identity AllReduce reads/writes gradient immediately, potentially before stream 0 finishes. Missing cudaEventRecord sync.\n\n2) vLLM weight update (inference stream vs default): inference stream reads old weights, cudaMemcpy on default stream copies new weights. If inference still processing, weights overwritten mid-computation. Missing cudaStreamSynchronize before copy.\n\n3) SGLang sleep/wake (KV flush vs decode): sleep frees KV blocks, decode kernel on another stream still reading those blocks. Free causes invalid memory access. Missing cudaStreamSynchronize(inference_stream) before free.\n\nCommon pattern: stream A modifies memory while stream B reads/writes same memory. Fix: cudaEventRecord + cudaStreamWaitEvent or cudaStreamSynchronize. MUST DO: sync inference stream before any memory modification.",
     "ref": "CUDA stream sync bugs, 3 framework examples"},
    {"q": "Explain why gradient checkpointing is mandatory for GRPO on RTX 4090 but optional for SFT.",
     "a": "Checkpointing trades compute for memory: recompute activations during backward from checkpoints instead of storing all.\n\nSFT on RTX 4090:\n- Single forward+backward, full 24 GiB available\n- 7B bf16: 14 GiB weights + 2 GiB optimizer + 4 GiB activations = 20 GiB\n- Fits in 24 GiB. Checkpointing optional (reduces to 1 GiB but +33% compute).\n\nGRPO on RTX 4090:\n- Must share GPU between rollout and training\n- gs trajectories per step (gs=8 = 8x forward passes)\n- Without checkpointing: 8*4 GiB = 32 GiB > 24 GiB -> OOM\n- With checkpointing: 8*1 GiB = 8 GiB + 14 GiB model + 2 GiB optimizer = 24 GiB -> fits\n\nKey difference: GRPO activation memory scales as O(gs). SFT gs=1, GRPO gs>=4.\nMath: GRPO memory = gs * SFT memory. gs=8 means 8x activations.\n\nConclusion: checkpointing mandatory for GRPO because gs amplifies activations beyond RTX 4090 budget.",
     "ref": "Gradient checkpointing, GRPO vs SFT memory"},
    {"q": "Explain the ulimit nofile=65536 requirement and why NCCL fails without it.",
     "a": "ulimit nofile: max open file descriptors per process. Default: 1024. NCCL needs many more.\n\nWhy NCCL opens many fds:\n1) TCP sockets: 2*(N-1) per rank for peer connections\n2) CUDA IPC channels: each requires fd\n3) /dev/shm files: per communication channel\n4) Persistent connections across all steps\n\nOn dp=1: NCCL still initializes ~50-100 fds (listener, IPC, shm, rendezvous).\n\nGRPO exacerbates: frequent step cycles (~44s/step). Combined with vLLM/SGLang (~200 fds) and Ray (~300 fds) and Python (~50 fds): total ~650 fds. Fits in 1024 barely.\n\nProblem: NCCL creates temporary socket bursts during AllReduce rendezvous, briefly pushing above 1024. Triggers socket exhaustion -> NCCL error -> crash.\n\nWith nofile=65536: ample headroom for bursts and all concurrent frameworks.\n\nFix: ulimit -n 65536 before training. Add to ~/.bashrc or training script.",
     "ref": "ulimit nofile, NCCL socket exhaustion, GRPO fd budget"},
    {"q": "Design a monitoring stack for detecting silent corruption in GRPO training.",
     "a": "5-layer monitoring stack:\n\nLayer 1: Gradient Health\n- Gradient norms per param group every step\n- Ratio ||grad_i||/||grad_0||: flag if >10x or <0.01x\n- NaN/inf check (torch.isnan, torch.isinf)\n- Hook on optimizer.step() before applying\n\nLayer 2: Weight Drift\n- Cosine similarity current vs initial every 100 steps\n- Threshold: 0.95 LoRA, 0.99 frozen base\n- Frozen base: ||W_current - W_initial|| must be 0. Nonzero = double-optimization detected\n\nLayer 3: Reward Distribution\n- Stats per group: mean, std, min, max\n- std<eps for >3 consecutive groups: zero-gradient risk\n- Sudden shift (>2*std between steps): stale-cache corruption\n\nLayer 4: Inference Consistency\n- Fixed test prompt after each weight update\n- Output vs reference comparison\n- Logit delta check: fresh KV vs cached KV\n\nLayer 5: Memory/Resources\n- torch.cuda.memory_summary() per step\n- Host RSS: >2 GiB/step = leak detected\n- NCCL buffers on dp=1: overlap_comm active detection\n\nIntegration: all to wandb with threshold alerts. Anomaly triggers detailed state logging.",
     "ref": "Silent corruption monitoring, 5 layers"},
]



SCENARIO_QUESTIONS = [
    {"q": "Your GRPO training produces NaN after 5 steps on RTX 4090. Debug step-by-step.",
     "a": "STEP 1: Check overlap_comm. If True and dp=1 -> NCCL identity race. MUST DO: overlap_comm=False.\nSTEP 2: Check advantage. If all rewards identical -> std=0 -> NaN. MUST DO: std<eps guard, skip gradient step.\nSTEP 3: Check bf16 ratio. If exp(log_diff) in bf16 -> overflow. MUST DO: fp32 ratio computation.\nSTEP 4: Check ZeRO-3+LoRA. All-gathers entire model, corrupts dp=1. MUST DO: switch to ZeRO-2.\nSTEP 5: Check inference caches. Stale KV -> wrong logits -> wrong rewards -> NaN advantage. MUST DO: flush caches on weight update.\nSTEP 6: Check LoRA scaling. alpha/r too large -> gradient overflow. MUST DO: keep alpha/r <= 2.\nSTEP 7: Add detect_anomaly(). torch.autograd.detect_anomaly() pinpoints exact NaN op.\nSTEP 8: Verify CUDA memory. torch.cuda.memory_summary(). Near OOM -> enable checkpointing, reduce batch.\nSTEP 9: Check DSV4 MoE. Dynamic routing + cudagraph = stale graph. MUST DO: disable cudagraph for MoE.\nSTEP 10: System: ulimit -n >=65536, nvidia-smi health, NCCL_DEBUG=INFO.",
     "ref": "NaN debugging workflow, 10 steps, MUST DO rules"},
    {"q": "You need to deploy a 30B-A3B MoE model on RTX 4090 for GRPO. What is your approach?",
     "a": "STEP 1: Memory budget. 30B bf16=60 GiB (exceeds 24 GiB). Only 3B active per token (MoE). FP8 active=3 GiB, full FP8=30 GiB still exceeds.\nSTEP 2: Quantization. Inactive experts in host RAM (offloading). Active experts bf16: 6 GiB. Dense layers: 2 GiB. Total active: 8+5+2=15 GiB fits with offloading.\nSTEP 3: LoRA. r=8 (not 32 -- 30B needs smaller). alpha=16 (ratio=2). Target: attention qkv, shared MLP (not experts). Memory: ~0.2 GiB.\nSTEP 4: Sleep/wake. SGLang tag-based. Sleep: flush KV+inactive expert cache. Wake: reload active from host (~0.5s).\nSTEP 5: Rollout. gs=4 (memory tight, not 8). max_response=256. batch=4 (4*4=16 trajectories).\nSTEP 6: Expert offloading. Host RAM for inactive. Prefetch next experts during compute. Keep top_k=6 in VRAM.\nSTEP 7: Training. ZeRO-2 (NOT ZeRO-3). overlap_comm=False. gradient_checkpointing=True. bf16+fp32 ratio.\nSTEP 8: cudagraph. Dense only. Disable for MoE. Pad expert outputs to fixed max_k.\nSTEP 9: Reward. Rule-based only (can not fit another model). Math verification, format, length penalties.\nSTEP 10: Monitor. Expert offloading latency. VRAM<22 GiB. Host RAM<40 GiB.",
     "ref": "30B-A3B MoE deployment on RTX 4090, 10 steps"},
    {"q": "DeepSpeed overlap_comm=True causes NaN on dp=1. Fix it without disabling overlap_comm.",
     "a": "Root cause: NCCL identity AllReduce on stream 7 races with gradient compute on stream 0.\n\nFIX 1: cudaEvent synchronization. After gradient compute on stream 0: event.record(stream=0), Stream(7).wait_event(event). Ensures stream 7 starts only after stream 0 finishes. Prevents race.\n\nFIX 2: Bypass NCCL on dp=1. if dp_size==1: return tensor (skip NCCL entirely). else: all_reduce(tensor). Functionally equivalent to overlap_comm=False on dp=1 but keeps config enabled for dp>1.\n\nFIX 3: Custom backend. Register custom process group that returns identity on dp=1 without NCCL kernel/stream operations.\n\nFIX 4: Patch DeepSpeed reduce_scatter. Check dp_size, skip NCCL on dp=1: if dp==1: return grads (identity, no NCCL).\n\nFIX 5: Double-buffering. Separate compute_buffer and comm_buffer. Stream 0 writes compute, copy to comm after completion. Stream 7 operates on comm only. No shared memory between streams.\n\nRecommended: FIX 2 (bypass NCCL on dp=1) simplest and most reliable. Keeps overlap_comm=True in config, conditionally skips NCCL when dp=1. Used in DeepSpeed fix #8080.",
     "ref": "overlap_comm NaN fix without disabling, 5 approaches"},
    {"q": "vLLM encoder cache persists stale state after RLHF weight update. Design a solution.",
     "a": "SOLUTION DESIGN:\n\n1) Cache Invalidation Hook: Register callback on weight update that flushes encoder+KV+cudagraph caches. vLLM update_weights() triggers automatically.\n\n2) Version-tagged Cache Entries: Each encoder entry stores weight_version tag. On lookup: if entry.version != current_version, invalidate and recompute. Enables selective invalidation -- only stale entries flushed, not entire cache.\n\n3) Lazy Invalidation: Do not flush proactively. Instead, check version on each cache access. If stale, recompute on-demand. Saves flush cost but adds check overhead per access.\n\n4) Full Flush on Sleep/Wake: During sleep phase (SGLang), flush ALL caches including encoder. During wake, rebuild from scratch. Simplest, most reliable, but highest latency.\n\n5) Differential Cache: Maintain two encoder caches -- current_version and previous_version. After weight update, current becomes previous, new current is empty. New requests populate current; stale requests still in previous for backward compatibility.\n\nRecommended: #2 (version-tagged) for production. Selective invalidation minimizes latency while ensuring correctness. For RTX 4090 GRPO where updates are frequent, #4 (full flush on sleep) is simplest and ensures correctness.",
     "ref": "vLLM encoder cache stale state, 5 solution designs"},
    {"q": "FSDP2 host RAM grows 5 GiB/step during GRPO training. How do you diagnose and fix?",
     "a": "DIAGNOSIS:\n1) Monitor host RAM: ps aux | grep python, track RSS per step. Expected: stable. Actual: +5 GiB/step.\n2) torch.cuda.memory_summary(): VRAM should be stable (FSDP2 properly scatters). If VRAM also grows -> FSDP2 bug.\n3) torch.profiler: profile one step, look for CPU-side copy_/gather allocations accumulating.\n4) gc.get_objects(): count tensor objects on CPU. If growing -> accumulation bug.\n\nROOT CAUSES:\n1) FSDP2 all-gather copies params to contiguous gathered buffer. Without gc.collect() between steps, buffers accumulate in CPU cache.\n2) Trajectory TransferQueue stores gs*batch trajectories in host RAM.\n3) Weight sync serializes entire model (14 GiB) to CPU before sending to vLLM/SGLang.\n4) Reduce-scatter creates CPU-side gradient buffers not freed between micro-steps.\n\nFIXES:\n1) torch.cuda.empty_cache() + gc.collect() after each GRPO step.\n2) FSDP2 use_orig_params=True to avoid parameter copy overhead.\n3) Free trajectory buffers immediately after optimizer.step().\n4) Ray shared memory for trajectory transfer (no CPU serialization).\n5) torch.utils.checkpoint memory cleanup hook after backward.\n6) Monitor with automatic alerting if RSS grows >1 GiB/step.",
     "ref": "FSDP2 host RAM diagnosis and fix, 6 fixes"},
    {"q": "SGLang MoE cache clobbered after weight update on DSV4. Root cause and fix?",
     "a": "ROOT CAUSE:\nSGLang MoE expert routing cache stores per-token expert assignment decisions. After LoRA weight update, routing probabilities change (LoRA modifies attention/query projections that feed the router). But cached routing table persists with old assignments. Tokens dispatched to wrong experts -> incorrect MoE output -> corrupted reward -> corrupted advantage -> corrupted gradient.\n\nThe clobber is silent: MoE output is numerically valid but semantically wrong. No NaN, no OOM. Training continues on corrupted data.\n\nFIX:\n1) Invalidate MoE routing cache on EVERY weight update. Add cache_flush() call to SGLang sleep/wake mechanism.\n2) Version-tagged routing: each routing entry has weight_version. On lookup, check version. If mismatch -> recompute routing.\n3) Disable MoE caching during GRPO: set enable_expert_cache=False. Routing recomputed every decode step (slower but correct).\n4) Full flush on sleep: during SLEEP phase, clear ALL caches including routing table. During WAKE, routing rebuilds from fresh weights.\n\nMUST DO: #1 or #4. For RTX 4090 GRPO, full flush on sleep is simplest. For multi-GPU with longer rollout phases, version-tagged routing (#2) enables selective invalidation.",
     "ref": "SGLang MoE cache clobber, DSV4 root cause and 4 fixes"},
    {"q": "rLLM GRPO advantage computation produces zero gradient. Explain and fix.",
     "a": "ROOT CAUSE:\nrLLM computes A_i = (r_i - mean(r)) / std(r). When all rewards identical (rule-based reward on simple math tasks where all answers correct/incorrect), std=0. rLLM sets advantage=0 when std<eps. Loss = -0 * log_prob = 0. Gradient = 0. No weight update. Training collapses after consecutive zero-gradient steps.\n\nTHREE FIXES:\n\n1) Skip normalization when std<eps:\n   if std < 1e-8:\n     advantages = rewards - mean(rewards)  # unnormalized\n   This produces nonzero advantages (rewards minus their mean). At least one advantage nonzero if rewards differ from mean. Gradient nonzero.\n\n2) Skip gradient step when std<eps:\n   if std < 1e-8:\n     continue  # skip this batch entirely\n   No gradient step this iteration. Move to next batch. Avoids zero gradient but wastes compute on the batch.\n\n3) Increase group size gs:\n   if gs >= 8, probability of all identical rewards << probability at gs=4.\n   More diverse reward samples. std rarely zero.\n\nMUST DO: implement #1 (skip normalization) as primary fix, #3 (increase gs) as preventive measure. #2 (skip step) as fallback.\n\nAdditional: use reward shaping to ensure diversity (format penalties, partial credit). Never use binary 0/1 reward with small gs.",
     "ref": "rLLM zero gradient, 3 fixes, MUST DO"},
    {"q": "Your verl training step takes 80s but should take 44s. Identify and fix the bottleneck.",
     "a": "Expected 44s breakdown (7B LoRA gs=8): rollout 8s + sleep 0.12s + training 21s + wake 0.07s + reward 0.1s + other 14.7s = ~44s.\n\nActual 80s = 36s excess. Bottleneck analysis:\n\n1) Check overlap_comm: if overlap_comm=True on dp=1, NCCL identity adds latency + potential stalls. Fix: overlap_comm=False. Saves ~5-10s.\n\n2) Check weight sync: if serializing full model to CPU before vLLM/SGLang (14 GiB copy), PCIe transfer ~0.56s * 2 (round trip) = ~1.1s. But if copying EVERY parameter instead of LoRA-only: 14 GiB vs 0.5 GiB. Fix: sync only changed params (LoRA adapters). Saves ~10s.\n\n3) Check TransferQueue (V1 only): if queue full, rollout blocks. If queue empty, training blocks. Each block wastes ~5-20s. Fix: V2 architecture eliminates queue. Saves variable.\n\n4) Check gradient checkpointing: if disabled, activation memory may cause gc pressure or OOM recovery. Fix: enable checkpointing. Saves ~5s (avoids OOM recovery).\n\n5) Check cudagraph: if disabled for ALL layers (not just MoE), decode runs without graph optimization. Fix: enable for dense layers. Saves ~2-3s per rollout.\n\n6) Check sleep/wake overhead: if flushing ALL VRAM instead of inference-only, training must reload entire model. Fix: flush only inference allocator, keep training allocator intact. Saves ~10-15s.\n\n7) Check batch construction: if constructing training batch on CPU with full serialization, overhead ~5s. Fix: use GPU-side batch construction with shared memory. Saves ~5s.\n\nMost likely: #6 (full VRAM flush) + #2 (full weight sync) = 25s excess. Fix these two for 80s->55s. Then #1 overlap_comm for 55s->44s.",
     "ref": "verl step timing bottleneck, 7 checks, 80s->44s"},
    {"q": "Design a complete RTX 4090 GRPO training pipeline from scratch. List every config choice.",
     "a": "COMPLETE RTX 4090 GRPO PIPELINE:\n\nMODEL: 7B bf16 (14 GiB)\nLoRA: r=32, alpha=64 (ratio=2), target=[q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj]\nLoRA memory: 0.5 GiB\n\nFRAMEWORK: verl V2 (single-GPU, no TransferQueue)\nRollout: SGLang with tag-based sleep/wake\nTraining: DeepSpeed ZeRO-2\n\nSYSTEM:\nulimit -n 65536\nCUDA_VISIBLE_DEVICES=0\nNCCL_DEBUG=WARN\n\nTRAINING CONFIG:\nDeepSpeed:\n  zero_optimization: stage=2\n  overlap_comm: False (MUST DO #1 dp=1)\n  bf16: {enabled=True}\n  gradient_accumulation_steps: 1\n  gradient_checkpointing: True\n  train_batch_size: 16\n  train_micro_batch_size_per_gpu: 4\n\nGRPO:\n  group_size: 8 (gs=8, stable advantage)\n  max_response_len: 512\n  clip_ratio: 0.2\n  kl_coeff: 0.01\n  advantage_type: group_normalized\n  std_guard: 1e-8 (MUST DO #3)\n  fp32_ratio: True (MUST DO #4)\n\nSGLang:\n  sleep/wake: tag-based\n  cudagraph: True (dense layers only, disabled for MoE)\n  max_running_requests: gs\n  kv_cache_budget: 5 GiB\n  flush_all_caches_on_update: True (MUST DO #2)\n\nREWARD:\n  type: rule-based (math verification, format check)\n  no reward model (0 GiB)\n\nMEMORY BUDGET:\n  Rollout: 14+5+0.5 = 19.5 GiB\n  Training: 14+0.5+2+3 = 19.5 GiB\n  Safety: 4.5 GiB headroom\n\nMONITORING:\n  gradient_norms per step\n  reward stats per group\n  host RSS per step\n  VRAM usage per phase\n  frozen_weight_drift: ||W_base_current - W_base_initial|| (MUST be 0)",
     "ref": "Complete RTX 4090 GRPO pipeline, every config"},
    {"q": "Your DSV4 model crashes with cudagraph on RTX 4090. What 3 things do you check first?",
     "a": "THREE FIRST CHECKS:\n\n1) Dynamic expert routing shapes:\nDSV4 MoE top_k varies per token -> variable-size intermediate tensors. cudagraph captured with fixed shapes can not replay with variable shapes.\nCheck: inspect expert dispatch tensor shapes across steps. If shapes vary -> cudagraph invalidated.\nFix: disable cudagraph for MoE layers. Enable only for dense (attention, shared MLP). Pad expert outputs to max_k for grouped GEMM.\n\n2) SM89 bf16 Tensor Core compatibility:\nRTX 4090 SM89 consumer GPU lacks native bf16 Tensor Core for some operations. cudagraph may capture kernels that fall back to FP32 emulation, producing different numeric results than expected bf16.\nCheck: verify cudagraph ops produce consistent bf16 results across replays. If numeric drift -> SM89 emulation issue.\nFix: use fp16 or fp8 for cudagraph-captured ops, bf16 for non-captured.\n\n3) Memory alignment and padding:\ncudagraph requires all tensors at capture time to match replay time exactly (same addresses, same sizes). DSV4 MoE expert weight pointers may shift after LoRA update (new adapter allocation).\nCheck: verify weight tensor addresses stable across steps. If addresses shift -> cudagraph stale pointers -> crash.\nFix: allocate weight tensors in fixed memory slots. Use cudaMalloc with consistent alignment. After LoRA update, invalidate and recapture cudagraph.\n\nMUST DO after each check: if cudagraph disabled for MoE, verify dense-layer-only cudagraph still captures correctly and replay produces consistent results.",
     "ref": "DSV4 cudagraph crash, 3 first checks on RTX 4090"},
]



RTX4090_QUESTIONS = [
    {"q": "Calculate the memory budget for deploying a 7B model on RTX 4090 for GRPO training.",
     "a": "RTX 4090 = 24 GiB VRAM.\n\n7B bf16 weights: 14 GiB\nLoRA adapters (r=32): 0.5 GiB\nOptimizer states (ZeRO-2, LoRA only): 2 GiB\nActivations (with checkpointing, gs=8): 8 GiB\nCUDA overhead: 0.5 GiB\n\nTraining peak: 14+0.5+2+8+0.5 = 25 GiB -- EXCEEDS 24 GiB!\n\nFix: reduce gs to 4: activations = 4 GiB. Total = 14+0.5+2+4+0.5 = 21 GiB. Fits with 3 GiB headroom.\n\nOr: fp8 activations: 4 GiB -> 2 GiB. Total = 14+0.5+2+2+0.5 = 19 GiB. Fits with 5 GiB headroom.\n\nRollout phase: 14 GiB model + 5 GiB KV cache + 0.5 GiB LoRA = 19.5 GiB. Fits.\n\nKey insight: 7B fits on 4090 only with LoRA+checkpointing+careful gs. Without LoRA: 14+4+4 = 22 GiB barely fits for SFT but NOT for GRPO.",
     "ref": "7B memory budget on RTX 4090"},
    {"q": "What is the best config for 8B model (Llama-3-8B) GRPO on RTX 4090?",
     "a": "8B bf16 = 16 GiB weights. Tighter than 7B.\n\nConfig:\n- LoRA: r=16 (not 32 -- 8B tighter), alpha=32 (ratio=2)\n- LoRA memory: 0.25 GiB\n- ZeRO-2, overlap_comm=False\n- gradient_checkpointing=True\n- gs=4 (not 8 -- memory too tight for gs=8)\n- batch_size=4 (4 prompts * 4 responses = 16 trajectories)\n- max_response_len=256 (shorter to reduce KV)\n- fp8 KV cache during rollout\n- SGLang tag-based sleep/wake\n- cudagraph: dense layers only\n\nMemory:\nRollout: 16+4(fp8 KV)+0.25 = 20.25 GiB\nTraining: 16+0.25+1.5(optimizer)+4(gs=4 checkpointed) = 21.75 GiB\nHeadroom: 2.25 GiB -- tight but works.\n\nCritical: gs=4 minimum, fp8 KV mandatory, shorter responses. 8B is the practical upper limit for single-4090 GRPO.",
     "ref": "8B GRPO best config on RTX 4090"},
    {"q": "What is the best config for 14B model GRPO on RTX 4090?",
     "a": "14B bf16 = 28 GiB weights. EXCEEDS 24 GiB. Can NOT fit in bf16.\n\nConfig:\n- FP8 quantization for weights: 14 GiB (14B * 1 byte)\n- LoRA in bf16 on top of FP8: r=8, alpha=16\n- LoRA memory: ~0.1 GiB\n- ZeRO-2, overlap_comm=False\n- gradient_checkpointing=True\n- gs=4 (absolute minimum, can not do gs=8)\n- batch_size=2 (2*4=8 trajectories)\n- max_response_len=128 (very short)\n- fp8 KV cache\n- Expert offloading not needed (14B dense, not MoE)\n\nMemory:\nRollout: 14(fp8)+4(fp8 KV)+0.1 = 18.1 GiB\nTraining: 14(fp8)+0.1+1(optimizer)+2(gs=4 checkpointed) = 17.1 GiB\n\nProblem: FP8 weights need dequantization for attention compute -> additional overhead. Dequant buffers: ~1 GiB.\n\nTotal training: 18.1 GiB. Fits with 5.9 GiB headroom.\n\n14B is feasible ONLY with FP8 quantization. Performance significantly degraded vs 7B/8B. Not recommended for production -- use 7B/8B instead.",
     "ref": "14B GRPO config on RTX 4090 (FP8 required)"},
    {"q": "What is the best config for 30B-A3B MoE model GRPO on RTX 4090?",
     "a": "30B-A3B: 30B total, 3B active per token. MoE architecture.\n\nConfig:\n- Expert offloading: inactive experts in host RAM\n- Active experts bf16: 6 GiB\n- Dense layers bf16: 2 GiB\n- LoRA: r=8, alpha=16, target=attention+shared MLP\n- LoRA memory: 0.2 GiB\n- ZeRO-2, overlap_comm=False\n- gradient_checkpointing=True\n- gs=4\n- batch_size=4\n- max_response_len=256\n- fp8 KV cache\n- SGLang tag-based sleep/wake\n- cudagraph: dense only, MoE disabled\n- Expert cache: top_k=6 in VRAM, rest host\n\nMemory:\nRollout: 8(active bf16)+2(fp8 KV)+0.2 = 10.2 GiB (sparse, fits easily)\nTraining: 8+0.2+1(optimizer)+4(checkpointed) = 13.2 GiB\n\nAdvantage: MoE means only 3B active params per token, not 30B. Much more VRAM headroom than 14B dense.\n\n30B-A3B is the BEST large model choice for RTX 4090. MoE sparsity makes it feasible where 14B dense barely fits.",
     "ref": "30B-A3B MoE best config on RTX 4090"},
    {"q": "Analyze sleep/wake timing for 7B LoRA GRPO on RTX 4090.",
     "a": "Sleep/wake timing analysis:\n\nStep cycle (gs=8):\n1) WAKE: load LoRA adapters -> 20ms\n2) PREFILL: 8 prompts * 128 tokens -> 2s\n3) DECODE: 8 * 512 tokens -> 6s\n4) REWARD: rule-based -> 0.1s\n5) SLEEP: flush KV(50ms) + caches(20ms) + free allocator(50ms) -> 120ms\n6) FORWARD: 8 trajectories checkpointed -> 12s\n7) BACKWARD: LoRA only -> 8s\n8) OPTIMIZER: LoRA states -> 1s\n\nTotal: 8s+0.12s+21s+0.07s(next wake) = ~29.2s\n\nSleep/wake overhead ratio: (120+70)ms / 29.2s = 0.6%. Negligible.\n\nThe 120ms sleep overhead is the cost of freeing ~5 GiB inference memory. The 70ms wake overhead is reloading 0.5 GiB LoRA adapters and rebuilding caches.\n\nWithout sleep/wake: inference memory persists = 5 GiB wasted -> OOM during training.\n\nWith sleep/wake: memory lifecycle correctly managed, both phases fit in 24 GiB.\n\nOptimization: prefetch LoRA during backward (-20ms), cudagraph dense layers (decode -2s). Optimized total: ~27s.",
     "ref": "Sleep/wake timing analysis, 7B GRPO RTX 4090"},
    {"q": "Which rollout engine is better for RTX 4090 GRPO: SGLang or vLLM?",
     "a": "Comparison for RTX 4090 GRPO:\n\nSGLang advantages:\n1) Tag-based sleep/wake -- debuggable, semantic versioning\n2) RadixCache -- efficient prefix sharing for GRPO (same prompt repeated gs times)\n3) Native LoRA support with hot-swapping\n4) Better cudagraph integration (capture per-batch-size)\n\nvLLM advantages:\n1) More mature PagedAttention implementation\n2) Wider community support and documentation\n3) Better multimodal/vision encoder handling\n4) Continuous batching more battle-tested\n\nFor RTX 4090 GRPO specifically:\n1) Prefix sharing critical: same prompt * gs responses. SGLang RadixCache reuses prompt KV for all gs completions. vLLM needs copy-on-write. SGLang saves ~2 GiB KV memory per group.\n2) Sleep/wake frequency: every ~29s. SGLang tag-based better for debugging frequent cache clobber events.\n3) LoRA hot-swap: SGLang can swap LoRA adapters without full model reload. vLLM requires update_weights() which copies entire model.\n\nConclusion: SGLang is better for RTX 4090 GRPO. Prefix sharing alone saves ~2 GiB, enabling gs=8 where vLLM might need gs=4.",
     "ref": "SGLang vs vLLM for RTX 4090 GRPO"},
    {"q": "Which checkpoint engine is better for RTX 4090 GRPO: naive or NCCL-based?",
     "a": "Checkpoint engine comparison for RTX 4090 GRPO:\n\nNaive checkpoint:\n- Serialize entire model to disk (14 GiB for 7B bf16)\n- Time: 14 GiB / disk_speed. SSD: ~3s. HDD: ~14s.\n- Memory: needs 14 GiB CPU buffer during serialization\n- Simple, reliable, no GPU dependency\n\nNCCL-based checkpoint:\n- Uses NCCL to coordinate checkpoint across ranks\n- On dp=1: NCCL overhead unnecessary (single rank, identity operations)\n- Adds NCCL buffer allocation + rendezvous\n- Same serialization time (no parallelism benefit on 1 rank)\n- Higher memory overhead (NCCL buffers + serialization buffer)\n\nFor RTX 4090 dp=1:\nNaive checkpoint is superior:\n1) No NCCL overhead (identity AllReduce useless)\n2) Lower memory (no NCCL buffers)\n3) Same speed (no parallelism on 1 GPU)\n4) More reliable (no NCCL race condition risk)\n\nFor multi-GPU dp>1:\nNCCL checkpoint may be better:\n1) Coordinated checkpoint across ranks\n2) Async checkpoint possible (overlap with training)\n\nConclusion: naive checkpoint for RTX 4090 (dp=1). NCCL checkpoint for multi-GPU only.",
     "ref": "Checkpoint engine choice, naive vs NCCL"},
    {"q": "Compare gs=4 vs gs=8 for GRPO on RTX 4090. Which is optimal?",
     "a": "gs=4 vs gs=8 comparison:\n\ngs=4:\n- Advantage variance: Var(r)/4 (acceptable)\n- Memory: 4 * checkpointed activations = 4 GiB\n- Total training: 14+0.5+2+4+0.5 = 21 GiB (3 GiB headroom)\n- Rollout: 4 * 512 tokens = 2s decode\n- Training: 4 * forward + backward = ~14s\n- Total step: ~20s\n\ngs=8:\n- Advantage variance: Var(r)/8 (better)\n- Memory: 8 * checkpointed activations = 8 GiB\n- Total training: 14+0.5+2+8+0.5 = 25 GiB (EXCEEDS 24 GiB!)\n- Need fp8 activations: 8 * 2 GiB = 4 GiB. Total = 14+0.5+2+4+0.5 = 21 GiB (fits)\n- Rollout: 8 * 512 tokens = 6s decode\n- Training: 8 * forward + backward = ~21s\n- Total step: ~29s\n\nTrade-off:\ngs=4: faster step (20s), lower variance reduction, more headroom\ngs=8: slower step (29s), better variance reduction, tighter memory\n\nOptimal: gs=4 for safety and speed. gs=8 only with fp8 activations and SGLang prefix sharing (saves ~2 GiB KV).\n\nRecommendation: start with gs=4, upgrade to gs=8 only after verifying memory stability.",
     "ref": "gs=4 vs gs=8 optimization on RTX 4090"},
    {"q": "Compare LoRA rank r=32 vs r=8 for GRPO on RTX 4090.",
     "a": "LoRA rank comparison:\n\nr=32:\n- Parameters: 2*32*4096*32 = 8 MiB/layer * 32 layers ~ 256 MiB + overhead = 0.5 GiB\n- Alpha=64 (ratio=2): safe scaling\n- Expressiveness: high (32 dimensions for adaptation)\n- Optimizer states: 2*0.5 GiB (Adam momentum+variance) = 1 GiB\n- Total LoRA overhead: 0.5+1 = 1.5 GiB\n\nr=8:\n- Parameters: 2*8*4096*32 = 2 MiB/layer * 32 layers ~ 64 MiB + overhead = 0.1 GiB\n- Alpha=16 (ratio=2): safe scaling\n- Expressiveness: limited (8 dimensions)\n- Optimizer states: 2*0.1 GiB = 0.2 GiB\n- Total LoRA overhead: 0.1+0.2 = 0.3 GiB\n\nComparison:\nr=32 saves 1.2 GiB over r=8? NO -- r=32 uses MORE memory (1.5 vs 0.3 GiB).\n\nBut r=32 has better training quality:\n- 4x more adaptation capacity\n- Better reward improvement per step\n- Fewer steps to convergence\n\nFor RTX 4090 7B: r=32 is preferred (1.5 GiB overhead fits in 3 GiB headroom).\nFor RTX 4090 8B: r=16 recommended (0.75 GiB, tighter memory).\nFor RTX 4090 14B FP8: r=8 required (0.3 GiB, minimal overhead).\n\nMUST DO: keep alpha/r ratio at 2 (safe scaling, avoids gradient overflow).",
     "ref": "LoRA rank selection, r=32 vs r=8 on RTX 4090"},
    {"q": "What are the DSV4 deployment considerations on RTX 4090?",
     "a": "DSV4 (DeepSeek-V3 MoE) on RTX 4090 considerations:\n\n1) Dynamic expert routing: top_k varies per token. cudagraph can not handle variable shapes. MUST DO: disable cudagraph for MoE layers, enable for dense only. Pad expert outputs.\n\n2) Expert offloading: 30B total, 3B active. Host RAM stores inactive experts. PCIe bandwidth limits transfer speed. Prefetch experts during compute.\n\n3) Memory budget: 8 GiB active bf16 + 2 GiB fp8 KV + 0.2 GiB LoRA = 10.2 GiB rollout. 13.2 GiB training. Both fit easily (MoE sparsity advantage).\n\n4) Sleep/wake: flush MoE routing cache on every weight update. Stale routing = tokens dispatched to wrong experts = silent corruption.\n\n5) Grouped GEMM: fixed expert batch sizes (pad zeros). Ensures consistent kernel shapes for grouped GEMM dispatch.\n\n6) SM89 compatibility: verify bf16 Tensor Core availability for expert GEMMs. FP8 alternative if bf16 emulation detected.\n\n7) LoRA targeting: attention qkv + shared MLP only. NOT expert MLPs (too many, offloaded anyway).\n\nMUST DO for DSV4 on 4090: disable MoE cudagraph, flush routing cache, pad expert outputs, use ZeRO-2.",
     "ref": "DSV4 deployment considerations on RTX 4090"},
    {"q": "Provide a full step timing breakdown for 7B LoRA GRPO on RTX 4090.",
     "a": "7B LoRA gs=8 GRPO step timing breakdown:\n\nWAKE phase (70ms):\n  Load LoRA adapters: 20ms (0.5 GiB H2D)\n  Rebuild SGLang caches: 50ms\n\nROLLOUT phase (8.1s):\n  Prefill 8 prompts: 2s (batched, 128 tokens each)\n  Decode 8*512 tokens: 6s (batched, SGLang continuous batching)\n  Reward computation: 0.1s (CPU, rule-based)\n\nSLEEP phase (120ms):\n  Flush KV cache: 50ms (5 GiB freed)\n  Flush RadixCache + routing: 20ms\n  Free inference allocator: 50ms\n\nTRAINING phase (21s):\n  Forward 8 trajectories: 12s (checkpointed, LoRA)\n  Backward LoRA only: 8s\n  Optimizer step: 1s (LoRA Adam, small)\n\nTotal step: 29.2s\n\nPer-step throughput: 8*512=4096 tokens/29.2s = 140 tokens/s\n\nComparison without optimizations:\n  No cudagraph: decode 9s, total 33s (30% slower)\n  No checkpointing: OOM (8*4 GiB = 32 GiB > 24 GiB)\n  No prefix sharing (vLLM): KV 7 GiB, total 31s\n  gs=4: total 20s, 4*512=2048/20 = 102 tokens/s\n\nOptimal: gs=8 + cudagraph + checkpointing + SGLang = 140 tokens/s.",
     "ref": "Full step timing breakdown, 7B GRPO RTX 4090"},
    {"q": "What are the OOM prevention strategies for GRPO on RTX 4090?",
     "a": "OOM prevention strategies (7 priority levels):\n\n1) gradient_checkpointing=True: reduces activations from 4 GiB/trajectory to 1 GiB. Most impactful single change. 8x reduction for gs trajectories.\n\n2) LoRA instead of full training: base model frozen, only 0.5 GiB trainable. vs full SFT: all 14 GiB trainable with optimizer states. Saves ~10 GiB.\n\n3) ZeRO-2 not ZeRO-3: ZeRO-2 keeps params local (no all-gather buffers). ZeRO-3 needs 14 GiB all-gather buffer on dp=1. Saves ~14 GiB.\n\n4) Sleep/wake: free inference memory before training. KV cache (~5 GiB) freed during sleep. Without sleep/wake: OOM guaranteed.\n\n5) fp8 KV cache: reduces KV from 5 GiB to 2.5 GiB during rollout. Saves 2.5 GiB.\n\n6) Reduce gs: gs=4 instead of gs=8 halves activation memory. 4 GiB vs 8 GiB. Trade-off: higher variance.\n\n7) Shorter max_response_len: 256 instead of 512 halves KV per trajectory and rollout KV. Saves ~2.5 GiB.\n\nEmergency OOM recovery:\n- torch.cuda.empty_cache() + gc.collect()\n- Reduce batch_size\n- Switch to fp8 activations\n- Disable cudagraph (saves capture buffers)\n- Move optimizer states to CPU (slow but prevents OOM)\n\nMUST DO: always enable #1-#4 before training. #5-#7 for tight budgets.",
     "ref": "OOM prevention strategies, 7 levels, RTX 4090"},
    {"q": "Describe the NaN debugging workflow for GRPO on RTX 4090.",
     "a": "NaN debugging workflow (5 phases):\n\nPHASE 1: Quick checks (30 seconds)\n- grep config for overlap_comm=True + dp=1 -> MUST DO #1\n- grep config for ZeRO stage=3 -> MUST DO #5\n- Check ulimit -n (must >=65536)\n\nPHASE 2: Anomaly detection (1 step)\n- torch.autograd.detect_anomaly() wraps one training step\n- Identifies exact op producing NaN\n- Common ops: ratio computation, advantage division, LoRA scaling\n\nPHASE 3: Root cause mapping\n- If NaN in ratio -> MUST DO #4 (fp32 ratio)\n- If NaN in advantage -> MUST DO #3 (std=0 guard)\n- If NaN in gradients after optimizer -> MUST DO #1 (overlap_comm)\n- If NaN in inference logits -> MUST DO #2 (flush caches)\n- If NaN in LoRA output -> MUST DO alpha/r ratio check\n\nPHASE 4: Validation\n- Apply fix, run 10 steps\n- Monitor gradient norms every step\n- Verify no NaN in: weights, gradients, activations, logits, rewards, advantages\n\nPHASE 5: Prevention\n- Add gradient NaN check hook (torch.isnan)\n- Add reward std check (MUST DO #3)\n- Add frozen weight drift check (||W_base - W_base_init|| == 0)\n- Log all checks to wandb\n\nTime budget: Phase 1-2 = 2 min. Phase 3 = 5 min. Phase 4 = 10 min. Phase 5 = 15 min. Total: ~30 min to debug and fix.",
     "ref": "NaN debugging workflow, 5 phases, RTX 4090"},
    {"q": "How do you monitor host RAM during GRPO training on RTX 4090?",
     "a": "Host RAM monitoring for GRPO:\n\nWHY monitor:\n- FSDP2 all-gather buffers accumulate on CPU\n- TransferQueue trajectories consume host RAM\n- Expert offloading (MoE) stores inactive experts on host\n- Weight serialization copies full model to CPU\n\nTOOLS:\n1) ps aux: track RSS (resident set size) of training process\n2) /proc/self/status: VmRSS, VmSize, VmSwap\n3) torch.cuda.memory_summary(): shows CPU-side allocations\n4) gc.get_objects(): count Python objects on CPU\n\nNORMAL behavior:\n- Initial: ~8 GiB (model loading, framework init)\n- Stable after 5 steps: ~12 GiB (trajectory buffers, caches)\n- Should NOT grow >0.5 GiB/step after stabilization\n\nABNORMAL behavior:\n- Growth >2 GiB/step: memory leak (FSDP2 buffers accumulating)\n- Growth >5 GiB/step: TransferQueue overflow or weight serialization\n- Total >40 GiB: expert offloading for 30B-A3B\n\nALERTS:\n- Yellow: RSS > 20 GiB (approaching host limit)\n- Red: RSS > 30 GiB (swap risk, OOM on 32 GiB host)\n- Critical: RSS growing >1 GiB/step (leak detected)\n\nFIXES for abnormal growth:\n1) gc.collect() + torch.cuda.empty_cache() per step\n2) Free trajectory buffers immediately\n3) FSDP2 use_orig_params=True\n4) Ray shared memory (no CPU serialization)\n\nMUST DO: log RSS every step. Alert if >1 GiB/step growth.",
     "ref": "Host RAM monitoring, GRPO on RTX 4090"},
    {"q": "What ulimit and system preparation is needed for RTX 4090 GRPO training?",
     "a": "System preparation checklist:\n\n1) ulimit -n 65536: NCCL socket headroom. Default 1024 causes socket exhaustion after ~1000 GRPO steps. MUST DO.\n\n2) CUDA_VISIBLE_DEVICES=0: ensure single GPU visible. Prevents accidental multi-GPU initialization.\n\n3) NCCL_DEBUG=WARN: reduce NCCL logging noise. INFO level produces excessive logs that slow training.\n\n4) NCCL_P2P_DISABLE=1 (single GPU): disable P2P access. Not needed on dp=1, prevents NCCL initialization overhead.\n\n5) nvidia-smi checks:\n   - GPU temperature < 85C (throttling risk)\n   - Power limit set correctly (450W for RTX 4090)\n   - No other processes using GPU\n   - Driver version compatible with CUDA 12.x\n\n6) Host RAM: minimum 32 GiB (16 GiB for system + 16 GiB for trajectory buffers/serialization). 64 GiB recommended.\n\n7) Disk: SSD for checkpoints. 14 GiB checkpoint * frequency = disk space needed.\n\n8) Python environment:\n   - PyTorch >= 2.1 with CUDA 12.x\n   - DeepSpeed >= 0.14\n   - vLLM >= 0.23 or SGLang >= 0.3\n   - verl >= 0.1\n   - PEFT >= 0.8\n\n9) Pre-flight check:\n   torch.cuda.is_available() -> True\n   torch.cuda.get_device_name(0) -> 'NVIDIA GeForce RTX 4090'\n   torch.cuda.mem_get_info() -> (free, total)\n\nMUST DO: #1 (ulimit), #2 (single GPU), #5 (nvidia-smi health). Others recommended.",
     "ref": "ulimit and system prep, RTX 4090 GRPO"},
    {"q": "Rank alternative frameworks for RTX 4090 GRPO training.",
     "a": "Framework ranking for RTX 4090 GRPO:\n\n#1: verl V2 + SGLang\n  Best single-GPU architecture. No TransferQueue overhead. Tag-based sleep/wake. Native LoRA hot-swap. Recommended for production.\n\n#2: verl V1 + SGLang\n  Decoupled pipeline with TransferQueue. More complex but enables multi-GPU rollout. Good for experimentation. Single-GPU has queue overhead.\n\n#3: verl V2 + vLLM\n  vLLM integer-based weight sync (less debuggable). PagedAttention (no prefix sharing, uses more KV memory). Works but suboptimal vs SGLang.\n\n#4: DeepSpeed-RLHF + SGLang\n  DeepSpeed training engine (ZeRO-2). More mature training code. SGLang rollout. Good for DeepSpeed users.\n\n#5: rLLM\n  Simpler implementation. Has std=0 bug (needs fix). Good for learning/understanding. Not production-ready without patches.\n\n#6: OpenRLHF + vLLM\n  PPO-focused (has critic model). More memory than GRPO. vLLM rollout. Not recommended for RTX 4090 (critic model exceeds memory).\n\n#7: TRL + vLLM\n  HuggingFace-native. Limited single-GPU optimization. No sleep/wake. Works for SFT/DPO but NOT for GRPO on 4090.\n\nConclusion: verl V2 + SGLang is the clear winner for RTX 4090 GRPO. All others have specific limitations (memory, bugs, architecture).",
     "ref": "Framework ranking for RTX 4090 GRPO"},
    {"q": "Provide a production deployment checklist for RTX 4090 GRPO training.",
     "a": "PRODUCTION DEPLOYMENT CHECKLIST:\n\n[ ] SYSTEM PREP\n  [ ] ulimit -n 65536\n  [ ] CUDA_VISIBLE_DEVICES=0\n  [ ] nvidia-smi health check (temp, power, driver)\n  [ ] Host RAM >= 32 GiB\n  [ ] SSD disk for checkpoints\n\n[ ] FRAMEWORK CONFIG\n  [ ] verl V2 + SGLang selected\n  [ ] DeepSpeed ZeRO-2 (NOT ZeRO-3)\n  [ ] overlap_comm=False (MUST DO #1)\n  [ ] gradient_checkpointing=True\n  [ ] bf16 training with fp32 ratio (MUST DO #4)\n\n[ ] MODEL CONFIG\n  [ ] LoRA r=32, alpha=64 (ratio=2)\n  [ ] Target modules: [q,k,v,o,up,gate,down]\n  [ ] 7B bf16 base model loaded\n\n[ ] GRPO CONFIG\n  [ ] gs=8 (or gs=4 if memory tight)\n  [ ] clip_ratio=0.2\n  [ ] kl_coeff=0.01\n  [ ] std_guard eps=1e-8 (MUST DO #3)\n  [ ] Rule-based reward (no reward model)\n\n[ ] SGLang CONFIG\n  [ ] Tag-based sleep/wake enabled\n  [ ] flush_all_caches_on_update=True (MUST DO #2)\n  [ ] cudagraph: dense layers only\n  [ ] Prefix sharing enabled (RadixCache)\n\n[ ] SAFETY CHECKS\n  [ ] Frozen base model drift monitor (||W-W_init||==0)\n  [ ] Gradient NaN check every step\n  [ ] Reward std check every group\n  [ ] Host RAM monitor (alert if >1 GiB/step)\n  [ ] VRAM monitor (alert if >22 GiB)\n\n[ ] PRE-FLIGHT\n  [ ] Run 3 steps without NaN\n  [ ] Verify gradient norms reasonable (1e-3 to 1e-1)\n  [ ] Verify rewards diverse (std > 0.1)\n  [ ] Verify sleep/wake transitions smooth (<200ms)\n  [ ] Checkpoint save/load works\n\nAll items MUST be checked before production run.",
     "ref": "Production deployment checklist, RTX 4090 GRPO"},
]


# -- MODE HANDLERS --

def run_quiz(count=10):
    """Mode 1: Random quiz questions from 7-framework KB."""
    all_qs = []
    for cat, qs in QUIZ_QUESTIONS.items():
        for q in qs:
            all_qs.append((cat, q))
    
    selected = random.sample(all_qs, min(count, len(all_qs)))
    
    print()
    print(c(BOLD + CYAN, "=" * 76))
    print(c(BOLD + CYAN, "  AI INFRA INTERVIEW PREP -- QUIZ MODE"))
    print(c(BOLD + CYAN, "=" * 76))
    print(c(DIM, f"  {len(selected)} questions randomly selected from {len(all_qs)} total"))
    print(c(DIM, f"  5 categories: Distributed Training, Inference Engine, RL Training, GPU/Driver, Framework Bugs"))
    print()
    
    for i, (cat, q) in enumerate(selected, 1):
        diff = q["diff"]
        bar = difficulty_bar(diff)
        print(c(BOLD, f"  Q{i}: ") + c(YELLOW, f"[{cat}]") + " " + c(BOLD, f"Difficulty: {bar} ({diff}/5)"))
        print(c(BOLD + BLUE, f"  {q['q']}"))
        print()
        print(c(DIM, "  Think about your answer..."))
        input(c(GREEN, "  Press Enter to reveal the answer > "))
        print()
        # Wrap answer for readability
        answer_lines = textwrap.wrap(q["a"], width=72)
        for line in answer_lines:
            print(c(GREEN, f"    {line}"))
        print()
        print(c(DIM + MAGENTA, f"  Reference: {q['ref']}"))
        print()
        print(c(CYAN, "-" * 76))
        print()
    
    print(c(BOLD + GREEN, "  Quiz complete! You answered {0} questions.".format(len(selected))))
    print()

def run_deep(count=10):
    """Mode 2: Deep technical interview questions."""
    selected = random.sample(DEEP_QUESTIONS, min(count, len(DEEP_QUESTIONS)))
    
    print()
    print(c(BOLD + MAGENTA, "=" * 76))
    print(c(BOLD + MAGENTA, "  AI INFRA INTERVIEW PREP -- DEEP MODE"))
    print(c(BOLD + MAGENTA, "=" * 76))
    print(c(DIM, f"  {len(selected)} deep questions requiring multi-framework synthesis"))
    print(c(DIM, f"  Each answer includes mathematical proof and framework references"))
    print()
    
    for i, q in enumerate(selected, 1):
        print(c(BOLD, f"  DEEP Q{i}: "))
        print(c(BOLD + RED, f"  {q['q']}"))
        print()
        print(c(DIM, "  Think deeply... this requires multi-framework understanding."))
        input(c(GREEN, "  Press Enter to reveal the detailed answer > "))
        print()
        # Display detailed answer with paragraph formatting
        paragraphs = q["a"].split("\n\n")
        for para in paragraphs:
            lines = para.split("\n")
            for line in lines:
                wrapped = textwrap.wrap(line, width=72)
                for wl in wrapped:
                    print(c(GREEN, f"    {wl}"))
            print()
        print(c(DIM + MAGENTA, f"  Reference: {q['ref']}"))
        print()
        print(c(MAGENTA, "-" * 76))
        print()
    
    print(c(BOLD + GREEN, "  Deep interview complete! {0} questions answered.".format(len(selected))))
    print()

def run_scenario(count=5):
    """Mode 3: Scenario-based interview questions."""
    selected = random.sample(SCENARIO_QUESTIONS, min(count, len(SCENARIO_QUESTIONS)))
    
    print()
    print(c(BOLD + ORANGE, "=" * 76))
    print(c(BOLD + ORANGE, "  AI INFRA INTERVIEW PREP -- SCENARIO MODE"))
    print(c(BOLD + ORANGE, "=" * 76))
    print(c(DIM, f"  {len(selected)} scenario questions requiring practical problem-solving"))
    print(c(DIM, f"  Each scenario has a step-by-step solution with MUST DO rules"))
    print()
    
    for i, q in enumerate(selected, 1):
        print(c(BOLD, f"  SCENARIO Q{i}: "))
        print(c(BOLD + ORANGE, f"  {q['q']}"))
        print()
        print(c(DIM, "  Work through your solution step-by-step..."))
        input(c(GREEN, "  Press Enter to reveal the solution guide > "))
        print()
        # Display step-by-step solution
        paragraphs = q["a"].split("\n\n")
        for para in paragraphs:
            lines = para.split("\n")
            for line in lines:
                # Highlight MUST DO rules
                if "MUST DO" in line:
                    print(c(BOLD + RED, f"    {line}"))
                elif line.startswith("STEP") or line.startswith("FIX") or line.startswith("PHASE"):
                    print(c(BOLD + YELLOW, f"    {line}"))
                else:
                    wrapped = textwrap.wrap(line, width=72)
                    for wl in wrapped:
                        print(c(GREEN, f"      {wl}"))
            print()
        print(c(DIM + MAGENTA, f"  Reference: {q['ref']}"))
        print()
        print(c(ORANGE, "-" * 76))
        print()
    
    print(c(BOLD + GREEN, "  Scenario interview complete! {0} scenarios solved.".format(len(selected))))
    print()

def run_rtx4090(count=10):
    """Mode 4: RTX 4090 specific interview questions."""
    selected = random.sample(RTX4090_QUESTIONS, min(count, len(RTX4090_QUESTIONS)))
    
    print()
    print(c(BOLD + PINK, "=" * 76))
    print(c(BOLD + PINK, "  AI INFRA INTERVIEW PREP -- RTX 4090 MODE"))
    print(c(BOLD + PINK, "=" * 76))
    print(c(DIM, f"  {len(selected)} questions focused on RTX 4090 deployment"))
    print(c(DIM, f"  Covers: memory budgets, configs, timing, engines, strategies"))
    print()
    
    for i, q in enumerate(selected, 1):
        print(c(BOLD, f"  RTX4090 Q{i}: "))
        print(c(BOLD + PINK, f"  {q['q']}"))
        print()
        print(c(DIM, "  Calculate your answer considering the 24 GiB VRAM constraint..."))
        input(c(GREEN, "  Press Enter to reveal the answer > "))
        print()
        paragraphs = q["a"].split("\n\n")
        for para in paragraphs:
            lines = para.split("\n")
            for line in lines:
                # Highlight memory numbers
                if "GiB" in line or "OOM" in line:
                    print(c(BOLD + YELLOW, f"    {line}"))
                elif "MUST DO" in line:
                    print(c(BOLD + RED, f"    {line}"))
                else:
                    wrapped = textwrap.wrap(line, width=72)
                    for wl in wrapped:
                        print(c(GREEN, f"      {wl}"))
            print()
        print(c(DIM + MAGENTA, f"  Reference: {q['ref']}"))
        print()
        print(c(PINK, "-" * 76))
        print()
    
    print(c(BOLD + GREEN, "  RTX 4090 interview complete! {0} questions answered.".format(len(selected))))
    print()

# -- MAIN --

def main():
    parser = argparse.ArgumentParser(
        description="AI Infra Interview Preparation Guide -- 7-Framework Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Modes:
              quiz      -- Random quiz questions (50 Qs, 5 categories, difficulty 1-5)
              deep      -- Deep technical interview (20 Qs, multi-framework synthesis)
              scenario  -- Scenario-based problem-solving (10 Qs, step-by-step solutions)
              rtx4090   -- RTX 4090 deployment focus (15 Qs, memory/timing/config)

            Examples:
              python3 ai_infra_interview_prep_guide.py quiz
              python3 ai_infra_interview_prep_guide.py quiz --count 20
              python3 ai_infra_interview_prep_guide.py deep --count 5
              python3 ai_infra_interview_prep_guide.py scenario --count 3
              python3 ai_infra_interview_prep_guide.py rtx4090 --count 8
        """)
    )
    parser.add_argument("mode", choices=["quiz", "deep", "scenario", "rtx4090"],
                        help="Interview prep mode")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of questions (default: quiz=10, deep=5, scenario=5, rtx4090=10)")
    
    args = parser.parse_args()
    
    defaults = {"quiz": 10, "deep": 5, "scenario": 5, "rtx4090": 10}
    count = args.count if args.count is not None else defaults[args.mode]
    
    handlers = {
        "quiz": run_quiz,
        "deep": run_deep,
        "scenario": run_scenario,
        "rtx4090": run_rtx4090,
    }
    
    handlers[args.mode](count)

if __name__ == "__main__":
    main()
