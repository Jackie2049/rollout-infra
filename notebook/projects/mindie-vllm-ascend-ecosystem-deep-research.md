# MindIE / vLLM-Ascend Ecosystem Deep Research

> 2026-06-19 | Comprehensive architecture + issue analysis
> ★★★★★★★★ MindIE = Huawei enterprise framework, vLLM-Ascend = open-source community plugin
> ★★★★★★★★ #10684 DSA Hadamard root cause CONFIRMED: class variable lost during sleep/wake
> ★★★★★★★★ #10592 NPUIPC has CRITICAL RCE vulnerability via pickle.loads

---

## 1. Architecture: MindIE vs vLLM-Ascend

| Aspect | MindIE | vLLM-Ascend |
|--------|--------|-------------|
| License | Proprietary (Huawei commercial) | Open-source (Apache 2.0) |
| Target | Enterprise, Huawei Cloud | Developers, community |
| Backend | ATB graph-level (compose fusion) | 5-layer op-level patch on vLLM |
| Optimization | Deep, NPU-specific, proprietary | Moderate, community-maintained |
| MoE | MindIE MoE path (fused) | MC2+EPLB DeepEP-Ascend path |

Both share CANN hardware abstraction layer. vLLM-Ascend as open-source entry point, MindIE as enterprise solution. Mirrors NVIDIA pattern (TensorRT-LLM enterprise vs vLLM community).

**Software stack (bottom to top):**
1. Ascend NPU Hardware (910B4/910C, Da Vinci architecture)
2. CANN 9.0.0 (driver, runtime, AscendCL, AICore operators)
3. torch_npu 2.10.0 (maps torch.cuda → torch.npu)
4. vLLM-Ascend (community plugin)
5. MindIE-Service (optional serving layer)

---

## 2. #10684 DSA Hadamard — Root Cause CONFIRMED

★★★★★★★★★ CRITICAL: Hadamard stored as **CLASS VARIABLE** on `AscendDSACPMetadataBuilder.hadamard` — NOT instance variable, NOT model buffer!

**Why it gets lost during sleep/wake:**
- `CaMemAllocator.sleep()` offloads NPU memory tagged as "weights" to CPU
- `worker.sleep()` saves `model.named_buffers()` (line 221-223)
- Hadamard is a CLASS attribute on the metadata builder, NOT in `named_buffers()`
- After `wake_up()`, the NPU memory backing hadamard tensor is invalidated/zeroed
- ALL downstream DSA attention produces zero output → ALL-ZERO inference

**Fix direction:** (1) Convert hadamard to model buffer for automatic save/restore, (2) Re-compute hadamard after wake_up, (3) Copy before in-place mutation

**Pattern family:** Textbook State Lifecycle Mismatch — identical to SGLang #28676 (MXFP8 cache clobbered), SGLang #28679 (GDN degeneracy), vLLM #44395 (KV cache still asleep)

**Impact:** BLOCKER for verl RLHF on Ascend NPU with DSA models (DSV4, GLM-5.x)

---

## 3. #10592 NPUIPC Weight Transfer — CRITICAL Security Bugs

★★★★★★★★★ Two CRITICAL bugs in NPUIPC:

1. **RCE vulnerability**: `pickle.loads` on HTTP endpoint data = Remote Code Execution. Even with `VLLM_ALLOW_INSECURE_SERIALIZATION` gate, unpickling endpoint is dangerous. Same pattern as SGLang #28582 (RCE).

2. **Device mismatch bug**: `pickle.loads` deserializes `UntypedStorage` with sender's device index baked in. Updating `list_args[6]` only changes tensor's logical device, leaving storage on sender's device → cross-device memory corruption/crash.

**Architecture:** NPUIPCWeightTransferEngine enables zero-copy weight sync via shared NPU memory. Two modes: `ipc_handles` (Ray) and `ipc_handles_pickled` (HTTP+base64+pickle). UUID: `{host_ip}-{physical_chip_id}`.

**verl relevance:** NPUIPC is the Ascend equivalent of verl's ZMQ IPC → verl Ascend integration pathway.

---

## 4. Ascend NPU Hardware — 4 Device Types

| Device | SoC Range | Chip | Key Feature |
|--------|-----------|------|-------------|
| A2 | 220-225 | 910B | 64GB HBM, FP16/BF16/INT8 |
| A3 | 250-255 | 910C | 64-96GB HBM, FP8 support |
| 310P | 200-205 | 310P | Small inference, separate `_310p/` subdirectory |
| A5 | 260 | 950B | MXFP4, FP8 E4M3FN full, unique indexer+QLI path |

**Key hardware vs CUDA differences:**
- HBM bandwidth: 910B ~1.2 TB/s vs H100 ~3.35 TB/s (2-3x more NPUs needed for decode)
- `torch.npu.get_device_properties().uuid` returns EMPTY string (can't use for IPC UUID)
- NPUGraph (Ascend's cudagraph) requires uniform batch sizes
- Float precision differences may cause numerical divergence

---

## 5. NEW Issues Found (June 12-18)

### Critical New Issues:
- **#10724**: DSV4 Flash crash on 2*A2 PD-Mix (8th DSV4 failure, KV cache block pool corruption)
- **#10720**: Qwen3.5-35B-A3B-w8a8-mtp overthinking on 300i duo
- **#10710**: DSV4-Flash-w8a8-mtp prefix cache hit rate = 0%
- **#10700**: GLM5.1 crashes after running without enforce_eager → enforce_eager STILL mandatory on Ascend!

### Critical New PRs:
- **#10733**: Layerwise KV pool with prefill layer reuse (builds on #10077 MERGED)
- **#10727**: MoE async scheduling race condition fix (snapshot mechanism)
- **#10730**: RMSNorm + Dynamic MX quant fusion (2x speedup)
- **#10704**: Drop v0.22.1 compatibility → main now tracks upstream vLLM main only
- **#10694**: DSA-CP TP async allgather for prefill
- **#10697**: Step3P7/Step3P5 with MTP on Ascend

---

## 6. DSA Implementation on Ascend

**Key files:**
- `vllm_ascend/ops/dsa.py` (291 lines) — operator definition with Hadamard
- `vllm_ascend/attention/context_parallel/dsa_cp.py` (1381 lines) — full DSA-CP implementation
- `vllm_ascend/attention/dsa_v1.py` (~3000+ lines, 121KB) — DSA V1 attention backend
- `vllm_ascend/models/deepseek_v4.py` (62KB) — DSV4 model with aiter imports

**DSA components:** wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b, attn_sink, indexer, compressor, swa_cache_layer, indexer_rotary_emb

**Hadamard transform:** `F.linear(x, hadamard)` with `scale = hidden_size**-0.5`. Split into `hadamard_linear` (main stream) + `hadamard_scale` (after aux_stream). **This is the function that corrupts during sleep/wake (#10684).**

**CUDA vs Ascend differences:**
- Hadamard stored as class variable on Ascend vs model parameter on CUDA
- `npu_flash_attention` / `npu_bmm_flash_attention` replaces `scaled_dot_product_attention`
- Custom `AscendDSAMetadataBuilder` for NPU-specific metadata
- Multi-stream overlap uses `torch_npu.npu.Stream()` instead of `torch.cuda.Stream()`

---

## 7. Weight Transfer Comparison

| Mechanism | NPUIPC (Ascend) | CUDA IPC | ZMQ (verl) | POSIX (ZenFlow) |
|-----------|----------------|----------|-------------|-----------------|
| Transport | NPU shared memory | CUDA IPC | TCP sockets | POSIX semaphores |
| Zero-copy | Yes | Yes | No | Yes (pinned) |
| Co-location | Required | Required | Not required | Not required |
| Security | **RCE via pickle!** | Safe | Safe | Safe |
| Framework | vLLM-Ascend | vLLM | verl | DeepSpeed |

---

## 8. vLLM-Ascend vs Mainline vLLM Key Differences

| Aspect | vLLM (CUDA) | vLLM-Ascend |
|--------|-------------|-------------|
| Platform | CUDA/CUDNN | CANN/torch_npu |
| Communication | NCCL | HCCL |
| Memory allocator | CUDA malloc | CAMEM |
| Attention | FlashInfer/FA2 | AscendC FA |
| Linear | cuBLAS/cUTLASS | ATB Linear / npu_quant_matmul |
| RoPE | vLLM rotary | AscendC rotary |
| Sleep/wake | CUDA IPC | Ascend sleep/wake + NPUIPC |
| Graph capture | CUDA graph | ACL graph (npugraph_ex) |
| MoE | DeepEP (SM90) | MC2+EPLB+DeepEP-Ascend |
| DSA | FlashInfer SFA | AscendC npu_sparse_attn |
| Quantization | FP8, GPTQ, AWQ | FP8 E4M3FN, MXFP4, W4A4 INT4 |

---

## 9. aiter Library — NOT Ascend!

★★★★★★★★★ CORRECTION: aiter = AMD Instinct Ecosystem Transformer Library for **AMD GPUs** (MI300X/MI350X), NOT for Ascend NPUs!

The bug in SGLang #28685 (12th DSV4 failure) is an **AMD GPU** issue:
- `gemm_a8w8_blockscale_bpreshuffle` CK kernel numerically incorrect on gfx950 (MI350X)
- GLM-5.2-FP8 block-fp8 → GSM8K 0.000
- vLLM-Ascend imports aiter for cross-platform compatibility but mocks to `False` on Ascend (lines 3593-3594)

---

## 10. RTX 4090 Pattern Family Transfer

★★★★★★★★★ 6 patterns that carry over from Ascend to CUDA (RTX 4090):

1. **State Lifecycle Mismatch** (#10684): Hadamard class variable → same pattern as SGLang #28676, #28679. RTX 4090 lesson: ANY GPU-resident constant buffer MUST be invalidated/rebuilt at weight-reload boundary.

2. **Sleep/Wake Buffer Preservation**: Ascend CaMemAllocator tag-based offload mirrors vLLM/SGLang sleep/wake. RTX 4090: sleep_level=1 SAFE, sleep_level=2 RISKY. Both platforms face class-variable/device-constant tensor loss.

3. **MoE NaN from Sign Convention** (#10579): torch.abs destroying negative indices. RTX 4090 lesson: ALWAYS verify operator semantics when porting between hardware.

4. **MX Quant Fusion** (#10730): AddRMSNorm+DynamicMxQuant 2x speedup on Ascend. RTX 4090: MXFP8 MoE quantization (#28676) has same cache invalidation requirement. Triton-based fusion viable on SM89.

5. **NPUIPC Security**: pickle.loads RCE mirrors SGLang #28582/#28588. RTX 4090: NEVER deserialize untrusted data over network endpoints.

6. **DSV4 Systematic Instability** (#10724): 8th confirmed failure on Ascend. RTX 4090 MUST: enforce_eager=True, invalidate ALL GPU-resident caches at weight-reload boundary.

---

## References

- vllm-ascend repo: https://github.com/vllm-project/vllm-ascend
- MindIE-LLM: https://github.com/Ascend/MindIE-LLM
- torch_npu: https://github.com/Ascend/pytorch
- Issue #10684: DSA Hadamard sleep/wake ALL-ZERO
- PR #10579: MoE NaN 1-line fix
- PR #10592: NPUIPC weight transfer
- Issue #10724: DSV4 PD-Mix crash (8th DSV4 failure)
- PR #10730: MX quant fusion
- Pattern family: notebook/fundamentals/state-lifecycle-mismatch-pattern-family-derivation.md
