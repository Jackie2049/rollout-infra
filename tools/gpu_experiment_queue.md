# GPU Experiment Queue — 7-Framework Focus (2026-06-16 Update)
> 等待 GPU 可用时执行的实验列表。按优先级排序。
> 高校服务器 (优先): `sshpass -p 'adspzxw123' ssh -o StrictHostKeyChecking=no zxw@219.233.198.62`
> 矩池云 (备用): `ssh -p 28959 root@hz-t3.matpool.com` (密码见MEMORY)
> 目标GPU: RTX 4090 (SM89, 24GB) / A100 (SM80) / H100 (SM90)

## P10 (UNIQUE): BudgetRefiner SLO Profile Data Collection
★★★★★★★★★ **NO OTHER vLLM CONTRIBUTOR HAS RTX 4090 PROFILE DATA → #1 contribution priority**
- **目的**: 收集 BudgetRefiner profile_table.csv 的 RTX 4090 具体数据
- **脚本**: `tools/profile_vllm_budget.py` + `tools/vllm_budgetrefiner_integration.py`
- **数据点**: (ctx_len, d_num) → chunk_size → budget at SLO=50ms
- **方法**: 运行 Qwen3-1.7B + 不同 decode 负载 → 测量每个配置的 ITL → 填充 lookup table
- **预期**: d_num=0→1024, d_num=100→~768, d_num=255→~512 (需实测确认)
- **关联PR**: BudgetRefiner SLO → vLLM upstream (notebook/projects/budgetrefiner-vllm-pr-draft.md)
- **状态**: READY, 等待GPU

## P9: SM89 Batch Invariance Reproduction
★★★★★★★★★ Inductor RMSNorm fusion → batch-dependent → root cause CONFIRMED
- **目的**: 在 RTX 4090 上重现 batch invariance 问题 + 验证 Fusion Guard fix
- **脚本**: `tools/sm89_batch_invariance_repro.py`
- **配置**: 3 configs — (a) baseline torch.mean, (b) Inductor-compiled RMSNorm, (c) our fusion guard
- **指标**: 不同 batch size 的 RMSNorm output 是否一致 (允许1e-3 tolerance)
- **关联PR**: Inductor SM<90 Fusion Guard → PyTorch upstream (notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md)
- **状态**: READY, 等待GPU

## P8: INT8 KV Cache Throughput Benchmark
★★★★ vLLM INT8 Triton fallback → RTX 4090 上可行但未测量吞吐
- **目的**: 测量 INT8 KV cache 在 RTX 4090 上的吞吐 vs BF16 KV cache
- **脚本**: `tools/sm89_kv_cache_cost_analyzer.py` (已有估算, 需实测)
- **指标**: tok/s throughput, latency, memory usage
- **状态**: READY, 等待GPU

## P7: SGLang Deterministic vs vLLM enforce_eager
★★★★★ SGLang deterministic inference → batch-invariant by design → 需GPU验证!
- **目的**: 比较 SGLang --enable-deterministic-inference 与 vLLM --enforce-eager 的吞吐和一致性
- **脚本**: 需编写 benchmark 脚本
- **指标**: throughput (tok/s), batch-dependent output difference, latency
- **预期**: SGLang deterministic 无 throughput loss, vLLM enforce_eager ~10-20% loss
- **状态**: 需写脚本, 等待GPU

## P6 (UNIQUE): AutoEP MoE Smoke Test
★★★★★★★★★ DeepSpeed AutoEP EP=1 → MoE on single GPU → ONLY framework supporting this!
- **目的**: 验证 Qwen3-MoE AutoEP training 在 RTX 4090 上可启动不OOM
- **脚本**: `tools/deepspeed_config_generator.py --scenario moe-autoep --model qwen3-moe --output configs/moe-autoep_rtx4090.json` → deepspeed --num_gpus=1 train.py
- **配置**: ZeRO-2 + CPU_Adam + overlap_comm=False + gradient_clipping=1.0 + auto_ep_preset=Qwen3-MoE
- **指标**: 是否成功启动, peak GPU memory, 训练 step time
- **状态**: CONFIG READY (configs/moe-autoep_rtx4090.json), 等待GPU

## P5: Muon+LoRA Training Experiment
★★★★★ Muon optimizer experimental → 需要与 AdamW baseline 比较收敛
- **目的**: 比较 Muon+LoRA vs AdamW+LoRA 在 GRPO 训练中的收敛速度和质量
- **脚本**: `tools/deepspeed_config_generator.py --scenario lora-grpo-muon` + `--scenario lora-grpo`
- **配置**: lora-grpo (AdamW baseline) vs lora-grpo-muon (Muon experimental)
- **指标**: reward curve, training loss, peak memory
- **警告**: Muon EXPERIMENTAL! 必须同时跑 AdamW baseline 对比!
- **状态**: CONFIG READY, 等待GPU

## P4: OPD+LoRA Distillation Smoke Test
★★★★ DeepSpeed OPD #8027 → Qwen2.5-0.5B student on RTX 4090 → distillation NEW market
- **目的**: 验证 OPD distillation student-only GPU path可行
- **脚本**: `tools/deepspeed_config_generator.py --scenario opd-distill`
- **配置**: Qwen2.5-0.5B student + CPU-offloaded teacher
- **指标**: student GPU memory, teacher CPU memory, distillation loss
- **状态**: CONFIG READY, 等待GPU (OPD PR #8027 DRAFT, 可能需修改)

## P3: rLLM Tinker GRPO Training
★★★★★ RTX 4090 GRPO ranking: rLLM Tinker #1 → 最简+最快+GRPO+bypass
- **目的**: 验证 rLLM Tinker GRPO training pipeline 在 RTX 4090 上运行
- **脚本**: rLLM cookbook (需检查是否有 RTX 4090 config)
- **配置**: group_size=4, batch_size=8, lora_rank=32, bypass_mode=true
- **状态**: 需要拉取 rLLM 最新代码, 等待GPU

## P2: vLLM v0.23.0 + Watermark Benchmark
★★★★ Watermark #44594 merged → preemptions -82%, ITL p99 -56%, throughput +5.1%
- **目的**: 测量 vLLM v0.23.0 + Watermark 在 RTX 4090 上的推理性能
- **模型**: Qwen3-1.7B, OPT-125M
- **指标**: throughput, ITL p99, preemption count
- **状态**: 需要更新 benchmark 脚本到 v0.23.0 API

## P1: 7-Framework Verification Suite
★★★★ 综合验证: DeepSpeed + Megatron + vLLM + verl + MindIE + rLLM + SGLang
- **脚本**: `tools/rtx4090_cross_framework_safety_matrix.py --mode summary`
- **目的**: 确认每个框架的关键 safety item 在真实 GPU 上是否触发
- **状态**: 安全矩阵已建立, 需GPU实测验证

---

## GPU Setup Checklist (once GPU is available)

1. Check GPU: `nvidia-smi` — confirm SM89 (RTX 4090) or SM80 (A100)
2. Install conda env: `conda create -n ai-infra python=3.11 && conda activate ai-infra`
3. Install deps (使用镜像源):
   ```bash
   pip install -i https://mirrors.aliyun.com/pypi/simple/ torch vllm deepspeed sglang
   conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/ ...
   ```
4. Clone repos: vllm, deepspeed, megatron-core, verl, sglang (已克隆到rollout-infra/)
5. Run P10 first: BudgetRefiner profile_table.csv → UNIQUE data → #1 priority
6. Then P9: batch invariance repro → root cause validation → Fusion Guard PR evidence

---

## Completed Experiments (A16/CUDA 11.7)
- Vector Add (自定义 CUDA): 170 GB/s
- Fused Bias+ReLU: 2.32x 加速
- GEMM Benchmark: 15.1 TFLOPS @2048
- HBM Bandwidth: ~170 GB/s
- CUDA Streams: 双 stream 重叠在 A16 上无收益 (10 SM)
- LayerNorm Fusion: 3.2x 加速
- FP16 GEMM: 5.7x over FP32
- vLLM OPT-125M: 163→3729 tok/s (batch 1→64)
- Triton 不兼容 (CUDA 11.7 + PTX 加载失败)
