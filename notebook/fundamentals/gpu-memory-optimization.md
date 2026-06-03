# GPU 显存优化实战

> 从原理到实践 — 理解每一字节显存去了哪里，如何最大化利用率

## 1. 显存分配全景

### 1.1 训练时显存占用公式

```
总显存 = 模型参数 + 优化器状态 + 梯度 + 激活值 + 临时缓冲区

详细分解 (以 FP16 训练 + AdamW 为例):

模型参数 (θ):
  FP16 参数:    2 × |θ|  bytes
  FP32 主权重:  4 × |θ|  bytes (mixed precision)
  总计:         6 × |θ|  bytes

优化器状态 (AdamW):
  FP32 momentum: 4 × |θ|  bytes
  FP32 variance: 4 × |θ|  bytes
  总计:          8 × |θ|  bytes

梯度:
  FP16 梯度:    2 × |θ|  bytes

激活值:
  依赖 batch_size × seq_len × hidden_size × num_layers
  估算: ~8 × batch × seq × hidden × layers × depth_factor

临时缓冲区:
  通信缓冲区, CUDA kernel workspace 等
  ~数百 MB 到 1-2 GB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计 (不含激活): 16 × |θ| bytes
  7B 模型:  16 × 7B = 112 GB (!) — 显然不对
  实际: BF16 训练时参数用 BF16, 优化器用 FP32

简化公式 (BF16 + AdamW):
  参数 (BF16):     2 × |θ|
  梯度 (BF16):     2 × |θ|
  优化器 (FP32):   4 × 2 × |θ| = 8 × |θ|  (m + v)
  FP32 主权重:     4 × |θ|
  总计:            16 × |θ|

  7B:   16 × 7 = 112 GB
  13B:  16 × 13 = 208 GB
  70B:  16 × 70 = 1120 GB

ZeRO 优化:
  ZeRO-1: 优化器状态分片 → 16/N + 8 × (1-1/N) × |θ|
  ZeRO-2: + 梯度分片 → 更省
  ZeRO-3: + 参数分片 → 最省
```

### 1.2 推理时显存占用

```
推理显存 = 模型权重 + KV Cache + 激活缓冲

模型权重:
  FP16:   2 × |θ|
  INT8:   1 × |θ|
  INT4:   0.5 × |θ|

KV Cache (per request):
  KV = 2 × num_layers × batch × seq_len × num_kv_heads × head_dim × dtype_size

  例: LLaMA-7B, FP16, seq=4096:
    2 × 32 × 4096 × 32 × 128 × 2 = 2 GB per request!

  FP8 KV Cache 可减半
  GQA (grouped query attention) 减少 num_kv_heads
  MQA (multi-query attention) 最极端

PagedAttention:
  按页分配 KV Cache → 减少碎片
  典型页大小: 16 tokens
```

## 2. PyTorch CUDA 内存管理

### 2.1 CUDA Caching Allocator

```
PyTorch 的 CUDA 内存分配器:

1. torch.cuda.memory_allocated()  — 当前由 tensor 占用的显存
2. torch.cuda.memory_reserved()   — 从 CUDA 申请并被缓存池持有的显存
3. torch.cuda.max_memory_allocated() — 峰值 allocated

内存池工作原理:
  分配: 先从缓存池找 → 找不到则 cudaMalloc
  释放: 不真正 cudaFree → 放回缓存池 (保留给后续分配)

好处: 减少 cudaMalloc/cudaFree 的开销 (这两者很慢)
坏处: 看起来 "泄漏" 了 — nvidia-smi 显示已用但实际可复用

清理:
  torch.cuda.empty_cache()  — 释放缓存池中未使用的显存
  ⚠ 只释放未被 tensor 持有的缓存, 不会释放正在使用的
```

### 2.2 显存监控代码

```python
import torch

def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_alloc = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{tag}] Allocated: {allocated:.2f}GB, "
              f"Reserved: {reserved:.2f}GB, "
              f"Peak: {max_alloc:.2f}GB")

# 使用方式:
print_gpu_memory("Before model")
model = get_model()
print_gpu_memory("After model")
optimizer = torch.optim.AdamW(model.parameters())
print_gpu_memory("After optimizer")

for batch in dataloader:
    print_gpu_memory("Before forward")
    loss = model(batch)
    print_gpu_memory("After forward")
    loss.backward()
    print_gpu_memory("After backward")
    optimizer.step()
    optimizer.zero_grad()
    print_gpu_memory("After step")
```

### 2.3 显存 Snapshot

```python
# PyTorch 2.1+ 支持显存 snapshot (可视化内存分配历史)
torch.cuda.memory._record_memory_history(
    enabled=True,
    context="train_step",
    stacks="python",
)

# ... 运行训练代码 ...

# 导出 snapshot
snapshot = torch.cuda.memory._snapshot()
import pickle
with open("snapshot.pickle", "wb") as f:
    pickle.dump(snapshot, f)

# 上传到 https://pytorch.org/memory_viz 查看
```

## 3. 常见 OOM 场景与解决方案

### 3.1 场景分类

```
场景 1: 模型太大放不下
  症状: 加载模型就 OOM
  方案: ZeRO-3 / 跨节点 TP / CPU offload

场景 2: 训练时激活值爆显存
  症状: forward 或 backward 时 OOM
  方案: gradient checkpointing / 减小 batch / 减小 seq_len

场景 3: 长序列推理 KV Cache 爆显存
  症状: 推理时 OOM
  方案: PagedAttention / 量化 KV / GQA

场景 4: 碎片化导致的 OOM
  症状: 理论上有空间但分配失败
  方案: torch.cuda.empty_cache() / 调整 max_split_size_mb
```

### 3.2 Gradient Checkpointing 显存节省

```
无 checkpointing:
  激活值 = O(n), n = num_layers
  每层的激活都保存用于 backward

有 checkpointing:
  激活值 = O(√n)
  只保存 √n 个 checkpoint, 中间层 forward 时重算

代码:
  model.gradient_checkpointing_enable()

  # 或更细粒度
  from torch.utils.checkpoint import checkpoint
  def custom_forward(x):
      return checkpoint(block_fn, x, use_reentrant=False)
```

### 3.3 ZeRO 各 Stage 对比

```
┌─────────┬──────────────┬──────────────┬──────────────┐
│         │ ZeRO-1       │ ZeRO-2       │ ZeRO-3       │
├─────────┼──────────────┼──────────────┼──────────────┤
│ 分片    │ 优化器状态   │ + 梯度       │ + 参数       │
│ 通信量  │ 与 DDP 相同  │ 略增         │ 显著增加     │
│ 显存省  │ 4x           │ 8x           │ N× (N=GPU数) │
│ 速度    │ 最快         │ 略慢         │ 最慢         │
│ 适用    │ 常规训练     │ 中等模型     │ 大模型       │
└─────────┴──────────────┴──────────────┴──────────────┘

最佳实践:
  - 先尝试 ZeRO-2 (性价比最高)
  - 模型放不下再用 ZeRO-3
  - ZeRO-3 配合 CPU offload 可训练超大模型 (但很慢)
```

## 4. 显存优化策略清单

### 4.1 不影响精度的优化

```
1. BF16 混合精度
   - 比 FP16 省一半空间
   - 不需要 loss scaling
   - A100/H100 原生支持

2. Gradient Checkpointing
   - 用计算换显存
   - 典型: 激活显存降到 √n

3. torch.cuda.empty_cache()
   - 在关键点手动释放缓存
   - 注意: 调用本身有开销

4. PYTORCH_CUDA_ALLOC_CONF 配置
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   - 减少碎片化
   - 大块分配用 128MB 为单位

5. 启用 FlashAttention
   - 激活值从 O(n²) → O(n)
   - 推理和训练都受益
```

### 4.2 有精度损失的优化

```
1. FP8 训练 (H100)
   - 参数和梯度用 FP8
   - 需要 GPU 硬件支持
   - 精度损失通常可接受

2. INT8/INT4 量化
   - 推理时压缩模型
   - GPTQ, AWQ, SmoothQuant
   - 典型损失: <1% accuracy

3. KV Cache 量化
   - FP8 KV Cache (vLLM 支持)
   - 推理显存减半

4. 降低 batch_size / seq_len
   - 直接减小激活值
   - 但可能影响训练效果
```

## 5. 实用工具

### 5.1 显存估算器

```python
def estimate_training_memory(
    param_count_b: float,  # 参数量 (billion)
    gpu_count: int,
    zero_stage: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    precision: str = "bf16",  # fp16, bf16, fp32
):
    """估算训练所需显存 (GB)"""
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}[precision]

    # 模型参数
    params_mem = param_count_b * 1e9 * bytes_per_param  # BF16 params
    params_mem_fp32 = param_count_b * 1e9 * 4  # FP32 master weights

    # 优化器状态
    optimizer_mem = param_count_b * 1e9 * 8  # AdamW: m + v (FP32)

    # 梯度
    grad_mem = param_count_b * 1e9 * bytes_per_param

    # 激活值 (粗略估算)
    # 每层激活: batch × seq × hidden × 4 (forward 各种中间结果)
    activation_per_layer = batch_size * seq_len * hidden_size * 4 * bytes_per_param
    activation_mem = activation_per_layer * num_layers

    # ZeRO 分片
    if zero_stage >= 1:
        optimizer_mem /= gpu_count
    if zero_stage >= 2:
        grad_mem /= gpu_count
    if zero_stage >= 3:
        params_mem /= gpu_count
        params_mem_fp32 /= gpu_count

    total = (params_mem + params_mem_fp32 + optimizer_mem + grad_mem + activation_mem)
    total_gb = total / 1e9

    return {
        "params_gb": params_mem / 1e9,
        "optimizer_gb": optimizer_mem / 1e9,
        "grad_gb": grad_mem / 1e9,
        "activation_gb": activation_mem / 1e9,
        "total_gb": total_gb,
        "per_gpu_gb": total_gb / gpu_count if zero_stage >= 3 else total_gb,
    }

# 示例: LLaMA-7B, 4x A100, ZeRO-2
result = estimate_training_memory(
    param_count_b=7, gpu_count=4, zero_stage=2,
    batch_size=1, seq_len=2048, hidden_size=4096, num_layers=32,
)
for k, v in result.items():
    print(f"  {k}: {v:.1f} GB" if isinstance(v, float) else f"  {k}: {v}")
```

### 5.2 nvidia-smi 高级用法

```bash
# 实时监控
watch -n 0.5 nvidia-smi

# 显存详情
nvidia-smi -q -d MEMORY

# 进程级显存
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 持续记录到文件
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 5 > gpu_log.csv
# 每 5 秒记录一次

# 算力利用率
nvidia-smi dmon -s u -d 1
```

## 6. 关键要点

1. **先估算再训练** — 用估算工具算出显存需求，避免浪费 GPU 时间
2. **BF16 > FP16** — 范围更大不需要 loss scaling，A100+ 首选
3. **ZeRO-2 是默认选择** — 性价比最高，大多数场景够用
4. **FlashAttention 必开** — 训练推理双重收益
5. **梯度检查点是安全垫** — 快 OOM 时第一个想到的工具
6. **监控峰值而非当前值** — `max_memory_allocated()` 才是关键指标
7. **碎片化是隐形杀手** — 配置 `max_split_size_mb` 可缓解

## 参考

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [PyTorch Memory Snapshot Tutorial](https://pytorch.org/tutorials/advanced/memory_snapshot_tutorial.html)
- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
