# DeepSpeed LoRAOptimizedLinear — Source-Level Analysis

> 2026-06-16 | deepspeedai/DeepSpeed | optimized_linear.py | LoRA+ZeRO-2+offload | RTX 4090最优配置
> ★★★★★ LoRAOptimizedLinear = 分离式forward → base_weight_output + scaling * lora_output → 不合并!
> ★★★★★ offload_ratio=0.5 → 累积offload → 50% base weights到CPU → RTX 4090 24GB内存节省!
> ★★★★★ ZeRO-2 + CPU_Adam + LoRAOptimizedLinear(offload_ratio=0.5) = RTX 4090最优DeepSpeed配置

## 1. ★★★★★ LoRAOptimizedLinear Forward — 分离式计算

```
★★★★★★★ LoRAOptimizedLinear.forward() (line 206):

def forward(self, input_tensor):
    if self.disabled:
        return F.linear(input_tensor, self.weight)
    assert self._initialized

    # Gather the sharded base weight
    if self.zero_shards > 1:
        with torch.no_grad():
            base_weight = self.full_weight()      # ← ZeRO-2 all-gather!
    elif self.quantization_config:
        base_weight = self.weight.dequantized()   # ← FP6/8/12 dequant
    else:
        base_weight = self.weight                 # ← 直接使用

    base_weight_output = F.linear(input_tensor, base_weight)
    lora_output = self.lora_weight_2(self.lora_weight_1(input_tensor))
    return base_weight_output + self.lora_scaling_factor * lora_output

★★★★★★★ 关键设计:
  → ★★★★★ 分离式forward → 不合并LoRA到base weight → base_weight_output + scaling * lora_output
  → → ★★★★★★★★ 原因: base weight frozen → requires_grad=False → 不需要每次forward都merge/unmerge
  → → → ★★★★★★★★★ vs vLLM/SGLang推理 → merge LoRA → 一次性 → 但训练需要分离 → 梯度只流向LoRA weights
  → → → → ★★★★★★★★★★★ lora_scaling_factor = lora_alpha / lora_r → 默认 16/64 = 0.25 → LoRA输出缩小4倍!
  → → → → → ★★★★★★★★★★★★★★★ base weight → ds_optim_param=True → 标记 → offload逻辑识别此标记

★★★★★★★ RTX 4090单GPU场景:
  → zero_shards = 1 → 不sharding → 不all-gather → 直接用base weight
  → → ★★★★★★★★ 单GPU = 无ZeRO-2 base weight分片 → 但offload_ratio仍然生效!
  → → → ★★★★★★★★★ offload_ratio=0.5 → 50% base weights offload到CPU → forward时swap back → 内存省50%!
```

## 2. ★★★★★ LoRAConfig — 完整参数定义

```
★★★★★★★ LoRAConfig dataclass (deepspeed/linear/config.py):

@dataclass
class LoRAConfig:
    lora_r: int = 64                              # ← LoRA rank → 默认64 → RTX 4090推荐32 (更小更省)
    lora_alpha: float = 16.                        # ← scaling factor → 16/64=0.25
    base_weight_sharding: int = 1                  # ← ZeRO-2 sharding degree → 单GPU=1
    offload: bool = False                          # ← 全量offload → 太极端 → 不推荐!
    offload_ratio: float = 0.0                     # ← 部分offload → RTX 4090推荐0.5!
    delay_lora_init: bool = False                  # ← 是否延迟初始化
    target_mods: List[str] = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                               'gate_proj', 'up_proj', 'down_proj']  # ← Llama-3.1默认目标模块

★★★★★★★ RTX 4090最优配置:
  → lora_r=32 → lora_alpha=16 → scaling=0.5 → 更大的LoRA影响
  → offload_ratio=0.5 → 50% base weights → CPU → 内存节省
  → base_weight_sharding=1 → 单GPU不分片
  → target_mods = q/k/v/o/gate/up/down → 7个模块 → Llama/Qwen全覆盖

★★★★★★★ LoRA初始化 (init_lora, line 125):
  → lora_weight_1 (A矩阵) → kaiming_uniform → input_dim → lora_r → requires_grad=True
  → lora_weight_2 (B矩阵) → zeros_ → lora_r → output_dim → requires_grad=True
  → → ★★★★★★★★ 初始LoRA输出 = 0 → A*kaiming + B*zeros = 0 → 模型初始不变 → 标准PEFT初始化!
  → weight.requires_grad = False → base weight frozen → 只训练LoRA → ZeRO-2只partition LoRA gradients!
```

## 3. ★★★★★ offload_ratio — 部分参数CPU Offload机制

```
★★★★★★★ _optimized_linear_offload_setup() (engine.py line 465):

def _optimized_linear_offload_setup(self):
    offload_ratio = None
    for _, module in self.module.named_modules():
        if isinstance(module, LoRAOptimizedLinear):
            offload_ratio = module.lora_config.offload_ratio  # ← 从LoRAConfig获取

    if offload_ratio is None:
        return  # ← 无LoRA → 不offload

    total_params = 0
    for _, p in self.module.named_parameters():
        if hasattr(p, 'ds_optim_param'):          # ← 只计算ds_optim_param=True的参数!
            total_params += p.numel()

    offload_limit = total_params * offload_ratio   # ← 累积offload上限!
    logger.info(f'offloading {offload_ratio*100}% of eligible params')

    total_offloaded = 0
    for _, p in self.module.named_parameters():
        if hasattr(p, 'ds_optim_param'):
            if total_offloaded < offload_limit:     # ← 还没达到上限 → offload!
                p.ds_offload = True
                p.offload()                          # ← 移到CPU!
                total_offloaded += p.numel()
            else:
                p.ds_offload = False                 # ← 达到上限 → 留在GPU!

★★★★★★★ 关键机制:
  → ★★★★★★★★ 累积offload → 按参数数量 → 不是按比例每层 → 先offload前面的参数 → 后面的留在GPU!
  → → ★★★★★★★★★ 这意味着: 前面的层 → 更多offload → 后面的层 → 更多GPU → 不是均匀分布!
  → → → ★★★★★★★★★★★★★★ 如果offload_ratio=0.5 → 前半部分参数 → CPU → 后半部分 → GPU → 内存省一半!
  → → → → ★★★★★★★★★★★★★★★★★ ds_optim_param=True → 只有base weights → LoRA weights不留GPU → 不offload!

★★★★★★★ forward时的swap:
  → full_weight() → 如果ds_offload → p.offload(revert=True) → swap back to GPU → all_gather → p.offload()
  → → ★★★★★★★★ 每次forward → offloaded base weight → swap to GPU → compute → swap back to CPU
  → → → ★★★★★★★★★ 性能代价: CPU→GPU swap → PCIe bandwidth → 比GPU-GPU慢 → 但24GB内存不够 → 必须牺牲速度!
  → → → → ★★★★★★★★★★★★★ RTX 4090: 7GB模型 + 50% offload → GPU 3.5GB base + LoRA ~0.5GB → 总4GB → 20GB留给KV/optimizer!
```

## 4. ★★★★★ RTX 4090最优DeepSpeed配置详解

```
★★★★★★★ RTX 4090最优: ZeRO-2 + CPU_Adam + LoRAOptimizedLinear(offload_ratio=0.5) + coalesce_grad_reduction

| Component | Config | Why |
|-----------|--------|-----|
| ZeRO-2 | stage=2 | partition optimizer states + gradients → ZeRO-3 single GPU无用 |
| CPU_Adam | offload_optimizer=cpu | optimizer states全部CPU → 省GPU内存 |
| LoRAOptimizedLinear | offload_ratio=0.5 | 50% base weights → CPU → 留更多GPU空间 |
| coalesce_grad_reduction | enabled | 合并梯度reduce → 减少通信次数 |
| lora_r=32 | LoRA rank=32 | 比默认64更小 → 省LoRA参数 → RTX 4090够用 |
| lora_alpha=16 | scaling=16 | scaling=16/32=0.5 → 适中 |
| target_mods | q/k/v/o/gate/up/down | 7个 → Llama/Qwen全覆盖 |

★★★★★★★ 内存预算估算 (Qwen2.5-7B):
  → 模型参数: 7B × 2bytes(BF16) = 14GB (total)
  → Base weights on GPU: 14GB × 0.5 = 7GB (offload_ratio=0.5)
  → LoRA weights: 7 × (4096×32 + 32×4096) × 2 ≈ 0.6GB
  → Gradients: LoRA only → ~0.6GB
  → Optimizer states: CPU → 0GB GPU
  → KV cache: ~4-6GB
  → Activation: ~2-3GB
  → Total GPU: ~7 + 0.6 + 0.6 + 5 + 3 ≈ 16.2GB ← fits in 24GB!

★★★★★★★ vs 其他配置:
  → ZeRO-3 single GPU → overhead only → 不partition → 无意义 → 不推荐!
  → ZeRO-2 + GPU_Adam → optimizer states在GPU → 多占7GB → 24GB不够!
  → ZeRO-2 + CPU_Adam → optimizer CPU → 省7GB → 但base weights全GPU → 14GB → 太多!
  → → ★★★★★★★★ 必须加LoRA+offload → base weights offload 50% → 7GB → 省一半 → 才能fit 24GB!
```

## 参考
- DeepSpeed deepspeed/linear/optimized_linear.py: OptimizedLinear + LoRAOptimizedLinear (222 lines)
- DeepSpeed deepspeed/linear/config.py: LoRAConfig + QuantizationConfig dataclasses
- DeepSpeed deepspeed/runtime/engine.py: _optimized_linear_offload_setup() (offload_ratio累积逻辑)
- Related notes: deepspeed-training-pipeline-reading.md, deepspeed-latest-developments-2026-06-reading.md, deepspeed-zero3-data-flow.md
