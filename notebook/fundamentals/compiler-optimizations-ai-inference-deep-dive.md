# Compiler Optimizations for AI Inference Deep Dive

> 2026-06-10 | 编译优化=AI推理的隐形加速器! 从torch.compile到CUDA Graph, 从kernel fusion到memory planning, 编译=性能飞跃的关键路径!
> 关联: torch-compile-benchmark-rtx4090.md, gpu-profiling-workshop-rtx4090.md, cutlass-gemm-rtx4090.md

## 0. 核心定律: 编译 = 消除Python开销 + kernel融合

AI推理性能瓶颈不只是GPU → Python overhead占主导(B=1时) → 编译消除!

关键发现(RTX 4090实测):
- torch.compile forward B=1 → **4.09x**加速! → Python overhead消除!
- torch.compile B≥4 → **0.80-0.97x**反而慢! → Triton GEMM慢于cuBLAS!
- CUDA Graph → 仅**1.01-1.02x** → 不是加速而是稳定jitter!
- → → → 编译=小batch有效+大batch无效 → RTX 4090最优=FlashInfer不是compile!

## 1. torch.compile — PyTorch编译框架

```
torch.compile架构 (PyTorch 2.x):

Dynamo → FX Graph → Backend → Optimized Code

Step 1: TorchDynamo (前端):
  → 捕获Python执行 → JIT → 生成FX Graph → 符号追踪!
  → → → 识别可编译区域 → break graph → 可编译部分+不编译部分!
  → → → → → dynamic shapes → graph break → 重新捕获 → 多图!
  → → → → → → → 支持大部分PyTorch → 但有限制(数据依赖控制流→break!)

Step 2: FX Graph (中间表示):
  → symbolic trace → node-based IR → 算子图 → 数据流!
  → → → 每个node=一个操作 → input→output → 依赖关系!
  → → → → → 可优化: 算子融合+死代码消除+常量折叠+内存优化!

Step 3: Backend (后端编译):
  → Inductor (默认) → Triton kernel生成 → GPU代码!
  → → → → → Triton: Python-like → tl.dot() → GPU kernel → 但慢于cuBLAS!
  → → → → → → → max-autotune: 搜索最佳config → 但RTX 4090选了慢kernel!
  → → → → → → → → → default mode → 用cuBLAS fallback → 更快!

  → AOTAutograd → 自动backward → 但推理不需要 → forward-only!
  → → → → → → → 推理=forward only → 不需要backward → 编译更快!

torch.compile模式:
  → default: 适度优化 → cuBLAS fallback → B≥4时最好(1.07x)!
  → → → max-autotune: 激进优化 → Triton全部 → RTX 4090最差(0.92x)!
  → → → → → reduce-overhead: CUDA Graph模式 → 消除launch overhead!
  → → → → → → → dynamic: 支持dynamic shapes → 但更慢(0.88-0.98x)!

RTX 4090实测关键发现:
  → B=1: 4.09x → Python overhead消除(占比大) → 编译有效!
  → → → B=4: 0.80-0.97x → Triton GEMM慢 → cuBLAS更快 → 编译无效!
  → → → → → B=16+: 1.01-1.05x → GPU compute主导 → Python overhead小 → 编译几乎无收益!
  → → → → → → → 训练: B=4 → 1.96x → backward Python overhead更大 → 编译更有效!
  → → → → → → → → → B≥16: 1.01-1.05x → 训练也无收益 → GPU compute主导!

  → → → → → → → → → → → 结论: 推理用FlashInfer不是compile → 训练B≤4用compile!
```

## 2. CUDA Graph — 消除kernel launch overhead

```
CUDA Graph原理:
  → 捕获kernel序列 → 整体执行 → 单次launch → 消除per-kernel launch overhead!
  → → → 正常: N个kernel → N次launch → N×8us=launch overhead → 累积!
  → → → → → Graph: N个kernel → 1次launch → 1×8us → 消除!

CUDA Graph限制:
  → 固定input shape → 不支持dynamic → 需预分配固定大小tensor!
  → → → vLLM V1: 固定input_ids → CUDA Graph兼容 → 捕获decode step!
  → → → → → 但: 不是所有kernel都支持 → 有些操作需dynamic → break graph!

CUDA Graph实测(RTX 4090):
  → OPT-125M: 2.43x → 大加速 → kernel小 → launch overhead占比大!
  → → → 7B模型: 1.05x → 小加速 → kernel大 → launch overhead占比小!
  → → → → → 原因: 大kernel=MLP 1ms+ → launch 8us→0.8% → 消除仅省0.8%!
  → → → → → → → → → 小kernel=RMSNorm 70us → launch 8us→11% → 消除省11%!

  → CUDA Graph价值=稳定jitter而非加速:
    → → → 正常: TTFT/ITL波动 → jitter → 用户感知!
    → → → → → Graph: 固定执行时间 → 无jitter → 用户体验稳定!
    → → → → → → → 生产=稳定SLI → P99 ITL更稳定 → SLO更容易达标!

  → RTX 4090 CUDA Graph参数:
    → → → graph_batch_sizes → 预捕获多个batch size → 分级!
    → → → → → vLLM: [1, 2, 4, 8, 16, 32] → 6个graph → 6×内存!
    → → → → → → → 7B INT4: 每graph 0.1MB/tok → 6 graph → 0.6MB → 小!
    → → → → → → → → → 但BF16 → 每graph 20MB → 6×120MB → 可接受!

  → RTX 4090: launch overhead≈8us(vs A16 34us) → 小GPU上CUDA Graph收益更小!
```

## 3. Kernel Fusion — 融合多个小kernel

```
Kernel Fusion = 将多个小kernel合并为一个大kernel → 消除中间HBM读写!

为什么fusion有效:
  → 小kernel → 写HBM → 读HBM → 下一个小kernel → HBM往返!
  → → → 融合 → 一次计算 → 中间结果留SRAM/L2 → 不写HBM → 省!
  → → → → → HBM带宽=890GB/s → 读写慢 → 融合=省HBM访问!

经典fusion模式:

1. SiLU Fusion (LLaMA SwiGLU):
   → gate_proj → SiLU → * up_proj → 3个小kernel → 2次HBM写!
   → → → 融合 → 1个kernel → 0次中间写 → CUTLASS SiLU epilogue!
   → → → → → 节省: launch overhead(~8us) + 中间HBM写(2×B×4096×2bytes)!
   → → → → → → → B=32: 独立28.8% overhead → 融合5-6% → 4.8x overhead↓!

2. RMSNorm + Residual Add:
   → RMSNorm → + residual → 2个小kernel → 1次中间写!
   → → → 融合 → 1个kernel → 0中间写 → CUDA RMSNorm kernel!
   → → → → → 节省: 9x over PyTorch → 主要来自butterfly shuffle reduction!

3. Fused QKV Projection:
   → Q_proj + K_proj + V_proj → 3个独立GEMM → 3×launch overhead!
   → → → 融合 → 1个GEMM → QKV拼接 → 38%加速(68us→42us) → launch省!
   → → → → → 但: torch.bmm比sequential慢0.38-0.77x! → strided memory → 不推荐!

4. Attention Fusion (FlashInfer):
   → Q×K → softmax → ×V → 3步 → 中间HBM写O(N²)!
   → → → 融合 → 1个kernel → online softmax → SRAM → 不写O(N²)!
   → → → → → FlashInfer: 推理attention → 15.72x(B=32) → 最大fusion收益!

5. Quantize+GEMM+Dequant (TE):
   → INT4 weight → dequant → GEMM → requant → 3步 → Python 20x慢!
   → → → 融合 → 1个C++ kernel → 无中间 → 量化才有效!
   → → → → → TE: FP8 fused kernel → 1.48-1.59x训练加速 → fusion=量化前提!

Fusion决策树:
  → 小kernel(B=1, <1ms) → fusion有效 → launch overhead占比高!
  → → → 大kernel(>1ms) → fusion收益小 → compute主导 → launch占比低!
  → → → → → 中间HBM读写大的 → fusion有效 → 省带宽!
  → → → → → → → GEMM → cuBLAS最快 → 不fusion → Triton GEMM更慢!
  → → → → → → → → → Attention → FlashInfer → 必须fusion → 生产答案!

RTX 4090最优fusion组合:
  → FlashInfer(attention fusion) + AWQ(Marlin fused dequant) + CUDA RMSNorm
  → → → + CUTLASS SiLU epilogue + TE FP8 → 全fusion → 1.06-3.20x整体!
```

## 4. Memory Planning — 编译期内存优化

```
编译期内存优化:

1. 死代码消除(DCE):
   → 未使用的计算 → 不执行 → 省内存+省计算 → 简单!
   → → → torch.compile → 自动DCE → 但LLM推理几乎无死代码!

2. 常量折叠:
   → 常量计算 → 编译时完成 → 不运行时计算 → 省时间!
   → → → 例如: position embedding → 编译时预计算 → 省prefill!

3. 内存复用:
   → 中间tensor → 同大小 → 复用内存 → 省分配 → vLLM PagedAttention类似!
   → → → torch.compile → 内存planning → reuse → 省peak memory!
   → → → → → LLM推理: hidden_states → 每层复用 → 省内存 → 但需要in-place!

4. In-place Operation:
   → tensor →原地修改 → 不创建新tensor → 省内存+省拷贝!
   → → → 但: 不是所有操作都安全in-place → 需要别名分析!
   → → → → → vLLM: KV cache → in-place写入 → PagedAttention → 安全!

5. View vs Copy:
   → tensor.view → 0拷贝 → 共享底层存储 → 极快!
   → → → tensor.reshape → 可能copy → 不共享 → 慢!
   → → → → → vLLM: block → tensor.view → 3.9x faster → pool slice = view!
   → → → → → → → 编译: reshape→view → 省copy → 省内存 → 简单优化!
```

## 5. vLLM Compilation Pipeline — vLLM的编译策略

```
vLLM编译策略:

1. CUDA Graph (V1默认):
   → 预捕获decode step → 固定batch size → 整体执行!
   → → → 6个batch size → [1, 2, 4, 8, 16, 32] → 分级graph!
   → → → → → 捕获时 → warmup → 捕获 → 存储 → 使用 → 固定!
   → → → → → → → 不支持: dynamic batch → chunked prefill → 仅decode!

2. torch.compile (V1可选):
   → enforce_eager=False → compile → 但vLLM主要用CUDA Graph!
   → → → compile → forward-only → 不需要backward → 更简单!
   → → → → → 但: Triton kernel慢 → B≥4反而0.80x → vLLM不用compile!

3. FlashInfer (V1默认attention):
   → 自定义CUDA kernel → 融合attention → 不经过compile → 直接高效!
   → → → decode: state_t → online softmax → paged KV → 15.72x!
   → → → → → prefill: batch attention → 不用compile → cuBLAS+custom!

4. Custom Triton Kernels (V1 internal):
   → Split-KV + Unified → decode attention → 自定义 → Triton!
   → → → TurboQuant → 3/4-bit KV → Triton kernel → 62.5% KV省!
   → → → → → 但: Triton kernel launch overhead → 生产需要CUDA Graph!

vLLM V1不使用torch.compile的原因:
  → Triton GEMM慢于cuBLAS → 0.80x → 性能损失!
  → → → CUDA Graph已提供稳定jitter → compile不需要!
  → → → → → FlashInfer attention已融合 → 不需要compile融合!
  → → → → → → → vLLM = FlashInfer+cuBLAS+CUDA Graph → 不需要compile!

RTX 4090 vLLM最优:
  → FlashInfer(attention) + cuBLAS(GEMM) + CUDA Graph(decode) + AWQ(Marlin)
  → → → → → 不需要torch.compile → 编译框架不如专用kernel!
```

## 6. JIT vs AOT Compilation — 编译时机选择

```
JIT(Just-In-Time)编译:
  → 运行时编译 → 第一次执行慢 → 之后快 → 灵活!
  → → → torch.compile = JIT → 每次新shape重新编译 → warmup!
  → → → → → Triton = JIT → 生成kernel → 编译→缓存 → 后续快!
  → → → → → → → warmup时间: 7B模型 ≈ 30s → 不可忽略!

AOT(Ahead-Of-Time)编译:
  → 预编译 → 所有kernel提前 → 无warmup → 固定!
  → → → CUTLASS = AOT → cmake编译 → .so → 直接加载 → 无warmup!
  → → → → → CUDA C++ = AOT → nvcc → 编译 → 二进制 → 无warmup!
  → → → → → → → FlashInfer = AOT → 编译+缓存 → 首次加载 → 之后快!

RTX 4090编译时机:
  → AOT推荐: CUTLASS/FlashInfer/CUDA RMSNorm → 预编译 → 无warmup!
  → → → JIT不推荐: torch.compile → warmup 30s → Triton慢于cuBLAS!
  → → → → → 生产=AOT → 开发=JIT → RTX 4090生产=AOT!

CUDA Graph编译时机:
  → 预捕获 → AOT-like → warmup后捕获 → 之后无overhead!
  → → → vLLM: 启动时捕获 → 几秒 → 之后decode用graph → 无overhead!
```

## 7. Core Laws — 编译优化核心定律

1. **Python-Overhead Law**: B=1 → Python overhead占71%(RMSNorm) → compile→4.09x → 消除overhead!
   → → → B≥4 → overhead<10% → compile无效 → GPU compute主导!

2. **Fusion-Launch-Law**: fusion=消除launch overhead+中间HBM → 小kernel有效 → 大kernel无效!
   → → → RMSNorm 9x → SiLU 4.8x → FlashInfer 15.72x → 都是小kernel融合!

3. **Triton-vs-cuBLAS Law**: Triton GEMM慢于cuBLAS → B≥4 torch.compile反而0.80x!
   → → → Triton适合reduction(RMSNorm/softmax) → 不适合GEMM → cuBLAS胜!

4. **CUDA-Graph-Jitter Law**: CUDA Graph=稳定jitter而非加速 → 7B仅1.05x → 但P99稳定!
   → → → 小kernel模型收益大(OPT-125M 2.43x) → 大模型收益小(7B 1.05x)!

5. **AOT-vs-JIT Law**: 生产=AOT(CUTLASS/FlashInfer) → 开发=JIT(torch.compile) → warmup不可忽略!
   → → → JIT warmup≈30s → 生产不可接受 → AOT=零warmup → 推荐!

6. **Quantization-Fusion Law**: 量化=fused kernel前提 → Python dequant=20x慢 → TE/AWQ/Marlin=C++融合 → 量化才有效!
   → → → 融合=消除dequant overhead → INT4才从0.05x→6.7x → 融合是前提!

7. **Compile-Decision Law**: 推理=FlashInfer+cuBLAS+CUDA Graph → 不需要torch.compile → 专用kernel更优!
   → → → 训练=B≤4用compile(default mode) → B≥16无收益 → GPU compute主导!

## 关键参考

- torch.compile: PyTorch 2.x → Dynamo+Inductor → Triton kernel → JIT
- CUDA Graph: 固定执行 → 消除launch overhead → 稳定jitter → vLLM V1用
- CUTLASS: AOT编译 → BF16 GEMM → SiLU epilogue → RTX 4090实测
- FlashInfer: AOT CUDA kernel → paged attention → 15.72x → 生产答案
- TE FP8: C++ fused kernel → quantize+GEMM+dequant → 1.48-1.59x训练
- torch.compile benchmark RTX 4090: results/torch_compile_benchmark.json
- CUDA Graph benchmark: 7B=1.05x → 不是加速 → 稳定jitter