# RTX 5090 推理影响分析 — AI infra战略决策 (2026-06-15)

> ★★★ RTX 5090 = Blackwell SM120 + 32GB GDDR7 + FP4 native → 改变消费级推理格局
> ★★★★★ 但: SM89(RTX 4090)仍然近期最重要 → SM120 kernel gap = 新贡献机会窗口
> 关联: vLLM SM89贡献策略 → SM120 FP4/MXFP4 → 下阶段战略

## 1. RTX 5090规格

```
★★★★★ RTX 5090 vs RTX 4090 对比:

| Spec | RTX 5090 | RTX 4090 | Delta |
|------|----------|----------|-------|
| Architecture | Blackwell (GB202) | Ada Lovelace (AD102) | 新一代 |
| Compute Capability | SM120 (12.0) | SM89 (8.9) | ★★★ 4代跳跃! |
| CUDA Cores | 21,760 | 16,384 | +33% |
| Tensor Cores | 680 (5th Gen) | 512 (4th Gen) | +33% |
| VRAM | 32 GB GDDR7 | 24 GB GDDR6X | ★★★ +33% |
| Memory Bandwidth | 1,792 GB/s | 1,008 GB/s | ★★★★ +78% |
| FP8 Tensor | ~836 TFLOPS | ~660 TFLOPS | +27% |
| FP4 Tensor | ★★★★ Native! | None | ★★★★ 新能力! |
| TDP | 600W | 450W | +33% power |
| MSRP | $1,999 | $1,599 | +25% |
| PCIe | 5.0 x16 | 4.0 x16 | 升级 |

★★★★ 关键变化:
  → GDDR7 PAM3 signaling → 带宽+78% → 推理吞吐大提升!
  → 5th-gen tensor core → FP4/FP6/MX native → ★★★★ 消费级首次!
  → SM120 → CUDA Toolkit 12.8+ → vLLM需要新kernel compilation
```

## 2. 32GB改变推理格局

```
★★★★★ 32GB vs 24GB 关键对比:

| Model | Quantization | 4090(24GB) | 5090(32GB) | 变化 |
|-------|-------------|-----------|-----------|------|
| 7B/8B | FP16 | 紧但可行 | ★★★ 舒适+32K context | 质变! |
| Mixtral 8x7B | INT4 | ★★ 勉强fit | ★★★ 舒适fit | 质变! |
| 70B | INT4 | ✗✗✗ 不fit | ✗✗✗ 不fit(40GB) | 不变 |
| DeepSeek 32B | INT4 | ✓ | ✓✓✓ 舒适 | 改善 |

★★★★ 关键洞察:
  → 7B FP16推理 → 从"紧"到"舒适" → 32K+ context → 质变!
  → Mixtral 8x7B → 从"勉强"到"舒适" → 实际可用!
  → 70B INT4 → 仍然不fit → 仍需多GPU → 不变!
  → ★★★★ 带宽+78% → 推理吞吐更重要 → tok/s 1.3-1.5x提升
  → ★★★ GRPO训练: 7B FP16可训练+RM可co-locate → 舒适度大提升!
```

## 3. FP4 Native — 量化革命

```
★★★★★ FP4 = 消费级量化革命 → 最重要新特性!

FP4硬件:
  → Blackwell 5th-gen tensor core → FP4 matrix ops直接执行
  → ~2x throughput vs FP8, ~4x vs FP16
  → 内存: FP4 = FP8的一半, FP16的1/4
  → ★★★ MX(Microscaling) → block-level scaling → 保持动态范围 → near-FP16 accuracy!

量化路径影响:
  → ★★★★ FP4 native → INT4(GPTQ/AWQ)将逐渐被替代!
  → FP4 = 浮点+block scaling → 比INT4更好的动态范围+精度
  → 硬件加速 → 无软件dequant overhead → 更快!
  → ★★★★ MXFP4 → 未来量化首选 → 但需要kernel生态!

vLLM影响:
  → ★★★ FP4/MXFP4 kernel for SM120 → ★★★★★ 高价值贡献路径!
  → 当前: vLLM不支持SM120 → custom kernels需要SM120 compilation
  → ★★★★★ 机会窗口: FP4/MXFP4推理kernel → 无现有实现 → 独特贡献!

量化路径对比:
| Path | 状态 | RTX 5090影响 |
|------|------|------------|
| FP16→FP8(INT8) | 当前标准 | 仍然安全默认 |
| FP16→FP4 via MX | ★★★ 新路径 | 4x内存+4x吞吐+MX精度 |
| FP16→INT4(GPTQ/AWQ) | 当前软件only | FP4替代INT4→更优 |
```

## 4. RTX 5090 D (中国版)

```
★★★★ RTX 5090 D — 出口管制版:

| Spec | RTX 5090 | RTX 5090 D |
|------|----------|-----------|
| CUDA Cores | 21,760 | ★★★ 14,592 (减33%!) |
| VRAM | 32GB GDDR7 | 32GB GDDR7 (不变!) |
| Bandwidth | 1,792 GB/s | 1,792 GB/s (不变!) |
| Compute Capability | SM120 | SM120 |

★★★★ 关键观察:
  → VRAM+带宽保持 → 推理吞吐(memory-bound) → 接近完整5090!
  → 计算33%削减 → 训练(prefill/batch inference) → 明显降低
  → 比RTX 4090 D(减11%)更激进 → 美国收紧出口管制!
  → ★★★ 对中国用户: 推理用5090D可行 → 训练受影响 → 但带宽仍强
```

## 5. SM89 → SM120 贡献策略转变

```
★★★★★★ 最关键的战略问题: SM89仍然重要吗?

★★★★★ 结论: SM89近期仍然最重要 → SM120是下一阶段!

| 因素 | 分析 |
|------|------|
| RTX 4090装机量 | ★★★★★ 巨大 → 最广泛消费级AI GPU → 短期仍主导 |
| RTX 5090采用时间 | ★★ 供应紧缺 → 2025-2026慢速采用 → 4090仍主流数月 |
| SM89 kernel成熟度 | ★★★★ 广泛支持 → 所有major框架工作 → 立即impact |
| SM120 kernel缺口 | ★★★ vLLM/FlashAttention/custom → 需SM120 port → 数月 |
| SM120独特特性 | ★★★ FP4/MX → 需SM120-native kernel → 机会窗口 |
| SM89 forward compat | ★★ SM89 code在SM120兼容模式运行 → 但无SM120优化 |

★★★★★ 战略推荐:
  1. ★★★★★ 近期: SM89贡献 → 立即impact → 巨大装机量 → 继续Tier1-2!
  2. ★★★★ 中期: SM120 FP4/MXFP4 kernel → ★★★★ 高价值贡献 → 无现有实现!
  3. ★★★ 镜像: SM89→SM120迁移 → 不是SM89→SM100 → SM100=数据中心(B200)

★★★★★ 时间线估计:
  → Q1-Q2 2026: SM89主要目标 → SM120支持成熟
  → Q3-Q4 2026: SM120开始parity → RTX 5090供应正常化
  → 2027: SM120成为dominant消费级目标 → 5090装机量增长

★★★★★ 可执行 takeaway:
  → 继续SM89贡献now → 立即impact → 建立声誉
  → 开始规划SM120 FP4/MXFP4 kernel work → 下阶段贡献策略
  → ★★★★ SM120 software gap = 我们的机会窗口!
```

## 6. RTX 5090 vs 4090 GRPO训练

```
★★★★ GRPO训练对比:

| 因素 | RTX 4090(24GB) | RTX 5090(32GB) |
|------|---------------|---------------|
| 最大模型size | 7B quantized, tight | ★★★ 7B FP16舒适; 14B quantized可行 |
| Rollout throughput | baseline | ★★★ 1.5-2x (带宽+计算) |
| 并行rollout groups | VRAM限制 | ★★★ 更多groups同时 |
| RM co-locate | ★★ 常需offload | ★★★ 可co-locate policy+RM! |
| FP4 training | ✗ None | ★★★ 新选项(quantized GRPO) |
| 推理部署 | INT4+INT8KV → 4,791 tok/s | ★★★ INT4+FP8KV → 更高tok/s |

★★★★ RTX 5090 GRPO最优:
  → 7B FP16 (不需要quantize!) + LoRA-32 + bypass_mode → 舒适!
  → ★★★ 不需要INT4推理 → FP16直接 → 更简单路径!
  → ★★★ 但: FP4训练 = 新选项 → quantized GRPO loop → 更省内存!
```

## 7. 关键洞察

1. ★★★★★ **SM89仍然近期最重要** → RTX 4090装机量巨大 → SM89贡献有立即impact
2. ★★★★★ **SM120 FP4 = 下阶段高价值贡献** → 无现有vLLM FP4 kernel → 机会窗口
3. ★★★★ **32GB改变7B推理** → FP16舒适 → 32K+ context → 但70B仍需多GPU
4. ★★★★ **带宽+78%比VRAM+33%更重要** → 推理memory-bound → 带宽是瓶颈 → tok/s提升更大
5. ★★★★ **FP4将替代INT4** → 浮点+MX scaling → 更好精度+硬件加速 → INT4(GPTQ/AWQ)逐渐被替代
6. ★★★ **RTX 5090 D推理接近完整5090** → VRAM+带宽保持 → 推理memory-bound → 计算削减影响训练
7. ★★★★ **GRPO在RTX 5090更舒适** → 7B FP16可训练 → RM可co-locate → 不需quantize
8. ★★★ **SM89→SM120是迁移路径** → 不是SM89→SM100 → SM100=数据中心 → SM120=消费级
9. ★★★★ **SM120软件缺口=贡献机会** → vLLM需要SM120 kernel → FlashAttention → FP4推理 → 全部needed

---

Sources:
- [NVIDIA RTX 5090 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
- [NVIDIA RTX 5090 D China](https://www.nvidia.com/en-cn/geforce/graphics-cards/50-series/rtx-5090-d/)
- [CUDA Toolkit 12.8 - SM120](https://docs.nvidia.com/cuda/)
- [OCP Microscaling Formats](https://www.opencompute.org/)
- [vLLM GitHub - SM120 Issues](https://github.com/vllm-project/vllm)
