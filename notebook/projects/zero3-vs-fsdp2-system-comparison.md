# ZeRO-3 vs FSDP2: 分布式训练分片策略系统对比

> 2026-06-15 | 基于7框架源码级研究 + 实战配置 + 通信量计算
> 关键结论: FSDP2是未来方向(ZeRO-3注定被淘汰), 但ZeRO-3的CPU/NVMe offload仍是单GPU小内存的最后出路

## 1. 核心差异: 分片粒度

| 维度 | ZeRO-3 | FSDP2 |
|------|--------|--------|
| **分片单位** | sub-partition(粗粒度,多个参数打包) | per-parameter DTensor(细粒度) |
| **分片算法** | flatten所有参数→连续buffer→narrow分片 | 每个参数独立DTensor shard_dim |
| **Padding浪费** | 有! NCCL对齐(4字节)+分区对齐→小参数浪费 | 无! DTensor按参数实际大小分片 |
| **分片状态管理** | AVAILABLE/NOT_AVAILABLE/INFLIGHT 三状态枚举 | Python属性管理(grad→shard, param→gather) |
| **初始化方式** | `with deepspeed.zero.Init():` metaclass注入 | `torch.distributed.tensor.parallel.DTensor` module wrap |
| **参数属性** | ds_tensor+ds_status+ds_numel+... | DTensor .placements + .spec |

### Padding浪费量化

```
ZeRO-3: 7B模型, 8GPU
  每个参数分区 = ceil(param_numel / 8)
  小参数(如bias=4096): ceil(4096/8)=512 → 512*8=4096 → 实际存储512*2=1KB
  但扁平化打包→所有小参数一起打包→浪费减少但仍存在

FSDP2: 7B模型, 8GPU
  每个参数独立: bias直接DTensor shard→无padding
  q_proj.weight(4096×4096): shard到8个512×4096 → 精确1/N
  无padding → 内存利用100%
```

**关键**: FSDP2无padding → 对于小模型(bias多)节省更大; ZeRO-3扁平化缓解了大部分padding但仍有浪费

## 2. 通信模式对比

| 维度 | ZeRO-3 | FSDP2 |
|------|--------|--------|
| **Forward通信** | AllGather(每层参数) | AllGather(每层参数) |
| **Backward通信** | ReduceScatter(grads) + AllGather(params) | ReduceScatter(grads) only |
| **总通信量/步** | 3Ψ | 2Ψ |
| **ZeRO-3第3次通信** | backward AllGather: 因为参数在forward后释放→backward需重新gather | 不需要: FSDP2保留full param直到backward完成 |
| **通信节省** | — | 33% less vs ZeRO-3 |
| **Overlap机制** | PartitionedParameterCoordinator trace prefetch | forward-order prefetch(简单) |
| **Overlap复杂度** | 3阶段trace(RECORD→COMPLETE→INVALID)+prefetch_bucket_sz+backpressure | 无trace,仅forward顺序预取 |

### 为什么ZeRO-3需要第3次AllGather?

```
ZeRO-3 forward流程:
  1. fetch_sub_module(layer_i) → AllGather layer_i params
  2. compute forward with gathered params
  3. release_sub_module(layer_i) → params回NOT_AVAILABLE状态!
  → forward后所有参数都回到分区状态!

ZeRO-3 backward流程:
  1. 需要参数计算梯度 → 必须再次AllGather所有参数
  2. 这是第3次通信!

FSDP2流程:
  1. forward: AllGather → compute → 保留full param
  2. backward: 直接使用保留的full param → 无需重新gather
  3. backward完成后: ReduceScatter(grads) + reshard(params)

→ FSDP2省了backward AllGather! 但代价是peak时同时持有full model + shards
```

## 3. 内存峰值对比

| 维度 | ZeRO-3 | FSDP2 |
|------|--------|--------|
| **模型参数** | Ψ/N (分片) | Ψ/N (DTensor分片) |
| **梯度** | Ψ/N (ReduceScatter) | Ψ/N (ReduceScatter) |
| **优化器** | 12Ψ/N (Adam shard) | 12Ψ/N (Adam shard) |
| **AllGather temp** | Ψ (当前层full) | Ψ (当前层full) |
| **保留完整参数?** | ✗ (forward后释放) | ✓ (backward用) |
| **Peak公式** | Ψ/N + 12Ψ/N + Ψ(AG temp) | Ψ/N + 12Ψ/N + Ψ(保留) ≈ ZeRO-3 |
| **实际Peak差异** | ≈(1+12/N+1)Ψ ≈ 13Ψ/N+Ψ | ≈(1+1+12/N)Ψ ≈ 2Ψ+12Ψ/N |

```
N=8, 7B模型:
  ZeRO-3: (1+12/8+1)*14GB = 2+1.75+14 = 17.75 + 1.4(act) ≈ 19.15GB per GPU
  FSDP2: (1+1+12/8)*14GB = 2+1.75+14 = 17.75 + 1.4(act) ≈ 19.15GB per GPU
  → Peak几乎相同! FSDP2保留full param的代价被ZeRO-3 AG temp抵消

N=1, 7B模型:
  ZeRO-3: 不适用(单GPU分片无意义)
  FSDP2: 不适用(单GPU分片无意义)
  → 单GPU都需要LoRA+CPU offload方案
```

**关键**: 内存峰值几乎相同! FSDP2的优势不在内存而在通信+compile兼容性

## 4. torch.compile兼容性 — 决定性差异

| 维度 | ZeRO-3 | FSDP2 |
|------|--------|--------|
| **compile兼容** | ✗ 不兼容 | ✓ 完全兼容 |
| **原因** | AllGather是动态操作→每次参数gather触发graph break | DTensor分片是静态→compile看到固定shape |
| **Graph break后果** | 每层都break→大量小graph→fusion失败 | 无break→整个model 1个graph→max fusion |
| **Fusion效果** | 接近eager模式(小kernel launch overhead) | 接近手写Triton kernel(1次launch多ops) |
| **编译开销** | 每次recompile(动态shape触发) | 1次编译+缓存(静态shape) |
| **Recompilation风暴** | √ 每个新batch_size触发 | ✗ Symbolic Shapes消除 |

### 为什么ZeRO-3 compile失败?

```python
# ZeRO-3 backward中的典型代码:
def backward_hook(param):
    # 动态AllGather → 触发graph break!
    gathered_param = all_gather(param.ds_tensor)  # shape取决于batch_size!
    # → compile无法预知shape → 每次重新编译

# FSDP2中的等价代码:
def forward(module, input):
    # DTensor → 静态shape!
    # module.weight 是 DTensor, placements=[Shard(0)]
    # → AllGather后shape固定(由DTensor spec决定)
    # → compile看到静态shape → 1次编译覆盖所有batch_size
```

### 编译加速效果估算

```
7B模型训练:
  Eager: 每层~8个kernel launch → 8×32层=256 launches/step
  ZeRO-3+compile: 每层graph break → ~256 launches(接近eager!)
  FSDP2+compile: 整模型1个graph → ~1-2 launches/step!

  Launch overhead: ~5μs/launch × 256 = 1.28ms/step
  vs FSDP2+compile: ~5μs × 2 = 0.01ms/step → 省1.27ms/step

  更重要的是fusion:
  FSDP2+compile: RMSNorm+SiLU+Residual→1个Triton kernel
  → 省多次memory round-trip → 2-3x计算加速!
```

**这是FSDP2的killer feature**: compile兼容→fusion→2-3x训练速度提升!

## 5. Prefetch策略对比

| 维度 | ZeRO-3 | FSDP2 |
|------|--------|--------|
| **预取策略** | Trace-based(3阶段) | Forward-order(简单) |
| **Trace机制** | RECORD→COMPLETE→INVALID | 无trace |
| **预取精度** | 精确! 记录实际forward顺序→按顺序prefetch | 简单! 假设forward顺序固定→按注册顺序 |
| **自适应** | ✓ 动态调整prefetch_bucket_sz | ✗ 固定策略 |
| **Backpressure** | ✓ max_ongoing=2 events | ✗ 无显式backpressure |
| **内存释放** | ✓ max_reuse_distance→智能释放 | ✗ 固定释放时机 |

```
ZeRO-3 PartitionedParameterCoordinator:
  第1个epoch: RECORD → 记录所有module的forward访问顺序
  第2+epoch: COMPLETE → 按trace顺序prefetch
    prefetch_queue: [module_i+1, module_i+2, ...] → compute module_i时prefetch i+1
    backpressure: max_ongoing=2 → 最多2个AllGather同时进行
  模型改变: INVALID → trace失效 → 回到RECORD模式

  → 精确但复杂! 634行代码管理trace+prefetch+release

FSDP2:
  简单策略: forward(module_i)时, prefetch module_i+1的参数
  → 假设forward顺序固定(大多数Transformer如此)
  → 简单但足够! forward顺序通常确实固定
  → 代码简单(在FSDP2 module hook中几行)
```

**关键**: ZeRO-3的trace更精确但复杂度远高→对标准Transformer模型,FSDP2的简单策略已经足够

## 6. Offload能力对比

| 维度 | ZeRO-3 | FSDP2 |
|------|--------|--------|
| **CPU offload优化器** | ✓ (DeepSpeedCPUAdam) | ✓ (torch.distributed, 2.6+) |
| **CPU offload参数** | ✓ (参数可以在CPU上分区) | ✗ (参数始终在GPU上) |
| **NVMe offload** | ✓ (AsyncPartitionedParameterSwapper) | ✗ |
| **混合offload** | ✓ (部分GPU+部分CPU比例可调) | ✗ |
| **GDS(GPU Direct Storage)** | ✓ (NVMe→GPU直读,1拷贝) | ✗ |
| **Offload粒度** | 子组级别(可调比例) | 模块级别 |
| **单GPU可行性** | ✓ ZeRO-3+NVMe→任意大模型 | ✗ 需2+GPU或LoRA |

```
ZeRO-3 NVMe offload:
  AsyncPartitionedParameterSwapper:
    Swap-out: ds_tensor→NVMe→释放内存→只剩invalid_buffer(1个half!)
    Swap-in: NVMe→CPU pinned→ds_tensor恢复→AllGather可用
    GDS: NVMe→GPU直读(1拷贝) vs 标准AIO: NVMe→CPU→GPU(2拷贝)
    → GPU≈2Ψ_layer+activations → 只持当前层 → 任意大模型!

  RTX 4090限制: 消费级PC无NVMe → verl GRPO更实用

FSDP2 CPU offload:
  2.6+: torch.distributed.FSDP.cpu_offload=True
  → 优化器状态offload到CPU
  → 但参数仍需GPU → 单GPU24GB限制!
  → RTX 4090: 需LoRA才能fit
```

**关键**: ZeRO-3的NVMe offload是理论上的"任意大模型"方案,但RTX 4090实际无NVMe→LoRA+CPU更实用

## 7. 检查点格式对比

| 维度 | ZeRO-3 | FSDP2 |
|------|--------|--------|
| **Checkpoint格式** | Universal checkpoint(跨ZeRO版本) | DTensor state_dict |
| **保存内容** | 每个rank只存1/N分区 → 需合并恢复完整 | 每个rank存DTensor shard → 自动合并 |
| **恢复流程** | 需zero_to_fp32.py转换 | 直接load → DTensor自动unshard |
| **跨框架兼容** | DeepSpeed专用格式 | PyTorch标准格式 |
| **Ease of use** | 复杂(需转换脚本) | 简单(原生PyTorch) |

## 8. 与7框架集成对比

| 框架 | ZeRO-3使用 | FSDP2使用 |
|------|-----------|-----------|
| **DeepSpeed** | 核心实现 | 不使用(自有ZeRO) |
| **Megatron-LM** | 不使用(自有TP/PP) | 不使用(自有TP/PP) |
| **vLLM** | 不使用(推理) | 不使用(推理) |
| **verl** | 可选backend(ZeRO-3) | 推荐backend(FSDP2) |
| **rLLM** | 不使用(Tinker in-process) | VerlBackend可选 |
| **MindIE** | 不使用(昇腾推理) | 不使用 |
| **PyTorch** | 不使用(自有FSDP) | 核心实现(2.6+) |

## 9. RTX 4090 实战决策树

```
RTX 4090 (24GB, PCIe, 7B模型):

单GPU:
  ZeRO-3 → ✗ (分片无意义+PCIe AG灾难)
  FSDP2 → ✗ (分片无意义+24GB不够)
  → 最优: LoRA r16+CPU_Adam+GRPO (17GB fit!)
  → 框架: verl or rLLM TinkerBackend

多GPU(4-8×RTX 4090, PCIe):
  ZeRO-3 → ✗ (3Ψ×PCIe 42GB/step 1.75s → 灾难)
  FSDP2 → ✗ (2Ψ×PCIe 28GB/step 1.17s → 灾难)
  DDP+LoRA → ✓ (0.11GB/step 4.56ms → 可行)
  → 最优: DDP+LoRA+CPU_Adam (每GPU独立训练→AllReduce LoRA grads)
  → 框架: DeepSpeed ZeRO-2+CPU_Adam or verl GRPO+LoRA

多GPU(8×A100/H100, NVLink):
  ZeRO-3 → ✓ (3Ψ×NVLink 42GB/step 0.21s → overlap可进一步优化)
  FSDP2 → ✓✓ (2Ψ×NVLink 28GB/step 0.14s + compile 2-3x → 最佳)
  → 最优: FSDP2+compile → 2-3x训练加速
  → 框架: PyTorch native or verl GRPO+FSDP2

推理:
  ZeRO-3 → 不适用
  FSDP2 → 不适用
  → 最优: vLLM INT4+INT8KV+FlashInfer
```

## 10. 为什么FSDP2是未来方向

```
1. torch.compile兼容 → killer feature
   ZeRO-3永远不可能兼容compile(dynamic AllGather是根本设计问题)
   FSDP2天生兼容 → Inductor fusion最大化 → 2-3x训练速度

2. PyTorch原生 → 无需DeepSpeed依赖
   FSDP2是PyTorch核心功能(2.6+稳定)
   ZeRO-3是DeepSpeed第三方库 → 维护成本+版本兼容问题

3. 33%通信节省 → 2Ψ vs 3Ψ
   对NVLink不重要但对PCIe/IB很重要

4. 无padding浪费 → DTensor精确分片
   小模型节省更大; ZeRO-3扁平化缓解但仍存在

5. 简单prefetch → forward顺序足够
   ZeRO-3 trace 634行代码 → 过度工程

6. PyTorch生态整合 → DTensor统一
   DTensor同时支持FSDP2+TP+EP → 统一抽象
   ZeRO-3的ds_tensor是DeepSpeed专用

但ZeRO-3仍有价值:
1. NVMe offload → 单GPU理论任意大模型(FSDP2无法)
2. CPU offload优化器 → 混合比例可调(FSDP2简单binary)
3. 3阶段trace → 精确prefetch(FSDP2简单策略)
4. Universal checkpoint → 跨版本兼容
5. 成熟生态 → 大量生产验证
```

## 11. 总结: 何时用哪个

| 场景 | 推荐 | 原因 |
|------|------|------|
| NVLink集群(8+GPU) | FSDP2+compile | compile 2-3x加速 + 2Ψ通信 |
| NVLink集群(8+GPU) GRPO | verl GRPO+FSDP2 | GRPO省50%内存+FSDP2加速 |
| PCIe集群(4-8×RTX 4090) | DDP+LoRA+CPU_Adam | PCIe不适合任何sharding |
| 单GPU RTX 4090 | LoRA+CPU_Adam+GRPO | 无sharding需求 |
| 单GPU+NVMe SSD | ZeRO-3+NVMe offload | 理论任意大模型 |
| MoE大模型+NVLink | Megatron TP+EP | EP需要All-to-All |
| 研究实验 | FSDP2+compile | compile+DTensor+TP兼容 |

---

Sources:
- notebook/projects/deepspeed-zero3-data-flow.md (ZeRO-3 Init metaclass + ds_*属性)
- notebook/projects/deepspeed-prefetch-coordinator-reading.md (PartitionedParameterCoordinator)
- notebook/projects/deepspeed-nvme-swap-reading.md (AsyncPartitionedParameterSwapper)
- notebook/projects/deepspeed-distributed-optimizer-source-reading.md (分区算法+步骤流程)
- notebook/fundamentals/pytorch-compiler-roadmap-26-28.md (compile+Inductor+FSDP2)
- notebook/fundamentals/pytorch-inductor-lowering-source-reading.md (SDPA=Fallback+fusion)
- tools/comm_cost_calculator.py (通信量量化)
- tools/zero_memory_peak_simulator.py (内存峰值)
