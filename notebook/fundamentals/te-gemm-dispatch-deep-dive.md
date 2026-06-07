# TransformerEngine GEMM Dispatch Deep Dive — RTX 4090 (SM89)

> 2026-06-08 | 源码级追踪: Python → C++ → cuBLASLt
> 硬件: RTX 4090 SM89, CUDA 12.8, cuBLASLt ≥12.8
> 连接: TE FP8 Deep Read → cuBLASLt FP8 GEMM → CUTLASS 3.x Architecture

## 1. 完整 Dispatch Chain (3层)

```
Python层 (gemm.py)           →  general_gemm()
C++ Pybind层 (gemm.cpp)      →  gemm() → nvte_cublas_gemm_v2()
C++ cuBLASLt层 (cublaslt_gemm.cu) →  cublas_gemm() → cublasLtMatmul()
```

**RTX 4090路径**: `general_gemm()` → `tex.generic_gemm()` → `gemm()` → `nvte_cublas_gemm_v2()` → `cublas_gemm()` → `cublasLtMatmul()`

## 2. Python层: `general_gemm()` (gemm.py:105-273)

### 2.1 核心流程

```python
def general_gemm(A, B, out_dtype, quantization_params, layout, ...):
    # 1. Workspace分配
    workspace = get_cublas_workspace(A.device.index, ub=None, grouped_gemm=False)
    # RTX 4090: 4MB, Hopper: 32MB+1024B

    # 2. Custom tensor dispatch
    if is_custom(A) or is_custom(B):
        return custom_gemm(...)  # MXFP8/NVFP4/BlockScaling

    # 3. 构造args
    args = (A, transa, B, transb, out, quantization_params, out_dtype,
            bias, bias_dtype, gelu, gelu_in, grad, workspace,
            workspace.shape[0], accumulate, use_split_accumulator)

    # 4. 调用C++ extension
    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
```

### 2.2 Workspace大小 (gemm.py:34-39)

```python
def get_cublas_workspace_size_bytes():
    if torch.cuda.get_device_properties().major >= 9:  # Hopper/Blackwell
        return 32 * 1024 * 1024 + 1024  # 32MB + 1024B (NVFP4 needs 32MB)
    return 4_194_304  # 4MB — RTX 4090 path!
```

**RTX 4090 workspace = 4MB**: 因为SM89 major=8 < 9, 所以只有4MB。
Hopper需要32MB因为NVFP4/MXFP8 GEMM需要更大的workspace。

### 2.3 Layout对GEMM的影响

| Layout | 含义 | TE Linear Forward | TE Linear Backward |
|--------|------|-------------------|--------------------|
| TN | A^T × B | input(T) × weight(N) | grad_output(T) × weight(N) |
| NN | A × B | input(N) × weight(N) — 不常见 | grad_output(N) × weight(N) |
| NT | A × B^T | input(N) × weight(T) — wgrad | grad_output^T × input |

**RTX 4090限制**: FP8 GEMM只支持TN layout! → TE需要canonicalize所有FP8 GEMM到TN。

## 3. C++ Pybind层: `gemm()` (gemm.cpp:144-414)

### 3.1 关键决策: Fused vs Unfused Quantization

```cpp
// gemm.cpp:222-229
bool unfused_quantization_needed = !quantizer.is_none();
if (low_precision) {
    // 只有DelayedScaling + per-tensor scaling → fused GEMM
    bool is_per_tensor_scaling_input = IsFloat8Tensor(A.ptr()) || IsFloat8Tensor(B.ptr());
    if (IsFloat8Quantizers(quantizer.ptr()) && is_per_tensor_scaling_input)
        unfused_quantization_needed = false;  // FUSED!
}
```

| Recipe | Quantization | RTX 4090路径 |
|--------|-------------|-------------|
| DelayedScaling + Float8Tensor | **Fused** | cuBLASLt内置scale_inv → 直接FP8 GEMM→BF16输出 |
| CurrentScaling + Float8Tensor | **Fused** | 同DelayedScaling(也是per-tensor scaling) |
| MXFP8 BlockScaling | **Unfused** | GEMM→BF16→再量化 |
| NVFP4 | **Unfused** | GEMM→FP32→再量化 |
| BF16→FP8 output | **Unfused** | GEMM→BF16→再量化到FP8 |

**RTX 4090上**: DelayedScaling和CurrentScaling都是fused → cuBLASLt直接处理dequant → 无Python overhead!

### 3.2 Scale Swizzling (gemm.cpp:305-310)

```cpp
auto [A_row_scales, A_col_scales] = swizzle_scales_for_gemm(A_tensor, transa, !transa);
auto [B_row_scales, B_col_scales] = swizzle_scales_for_gemm(B_tensor, !transb, transb);
```

**为什么需要swizzle?** cuBLASLt对scaling factor有特定的内存布局要求:
- TN layout → A需要row-wise scales, B需要row-wise scales
- 但实际数据可能是column-wise scales → 需swizzle重排

### 3.3 FP8 Block Scaling → MXFP8 Emulation (Blackwell only)

```cpp
// gemm.cpp:314-323 (only on SM100+)
if (fp8_block_scaling && sm_arch() >= 100) {
    // Convert FP8 block scaling → MXFP8 format
    convert_block_scaling_to_mxfp8_tensor(A_tensor, transa);
    convert_block_scaling_to_mxfp8_tensor(B_tensor, !transb);
    transa = true;  transb = false;  // Force TN layout
}
```

**RTX 4090**: SM89 < SM100 → 不走这条路径 → BlockScaling不可用!

### 3.4 MatmulConfigWrapper (gemm.cpp:288-298)

```cpp
MatmulConfigWrapper config;
if (grad) {
    config.set_dbias_tensor(bias_tensor.data());    // Backward: compute bias gradient
    config.set_with_dgelu_epilogue(gelu);            // Backward: compute GELU gradient
} else {
    config.set_bias_tensor(bias_tensor.data());      // Forward: add bias in epilogue
    config.set_with_gelu_epilogue(gelu);             // Forward: compute GELU in epilogue
}
config.set_epilogue_aux_tensor(te_pre_gelu_out.data());
config.set_use_split_accumulator(use_split_accumulator);
config.set_sm_count(num_math_sms);
```

**Epilogue fusion**: cuBLASLt支持在GEMM末尾融合:
- `CUBLASLT_EPILOGUE_BIAS`: D = AB + bias
- `CUBLASLT_EPILOGUE_GELU_AUX_BIAS`: D = GELU(AB + bias), pre_gelu_out = AB + bias
- `CUBLASLT_EPILOGUE_BGRADB`: D = AB, dbias = sum(dy)
- `CUBLASLT_EPILOGUE_DGELU_BGRAD`: D = GELU'(AB+bias)*dy, dbias = sum(dy)

**RTX 4090**: 所有这些epilogue fusion都支持(SM89的cuBLASLt功能完整)!

### 3.5 SM Margin (gemm.cpp:284-285)

```cpp
int num_math_sms = sm_count - getenv<int>("NVTE_EXT_MARGIN_SM", sm_count);
```

**NVTE_EXT_MARGIN_SM**: 留一些SM给其他任务(如DP overlap) → GEMM只使用部分SM → 减少对其他kernel的干扰。

## 4. C++ cuBLASLt层: `cublas_gemm()` (cublaslt_gemm.cu:315-799)

### 4.1 CanonicalizeGemmInput — RTX 4090的TN强制转换

这是最关键的函数! 它将任意layout的FP8 GEMM canonicalize为cuBLASLt能接受的格式:

```cpp
GemmParam CanonicalizeGemmInput(const Tensor &A, cublasOperation_t transA,
                                 const Tensor &B, cublasOperation_t transB, int m, int n, int k) {
    // 检查: nvte_is_non_tn_fp8_gemm_supported()
    // RTX 4090 (SM89): 返回0 → 不支持非TN的FP8 GEMM!
    // Blackwell (SM100): 返回1 → 支持任意layout的FP8 GEMM!

    // A配置 — tensor scaling (DelayedScaling/CurrentScaling):
    if (is_tensor_scaling(A.scaling_mode)) {
        ret.A = A.data.dptr;
        ret.transA = transA;
        ret.A_scale_inv = A.scale_inv.dptr;

        if (!is_nvte_non_tn_fp8_gemm_supported && !is_A_transposed) {
            // RTX 4090路径: A不是transposed → 需要TN → 用columnwise data
            if (A.has_columnwise_data() && is_fp8_dtype(A.columnwise_data.dtype)) {
                ret.A = A.columnwise_data.dptr;        // 用转置数据
                ret.transA = CUBLAS_OP_T;               // 强制transposed!
                ret.A_scale_inv = A.columnwise_scale_inv.dptr;
                ret.lda = k;                            // column-wise: lda=k
            }
        }
    }

    // B配置 — 同理:
    if (!is_nvte_non_tn_fp8_gemm_supported && is_B_transposed) {
        // RTX 4090: B是transposed → 需要TN → 用columnwise data
        ret.B = B.columnwise_data.dptr;
        ret.transB = CUBLAS_OP_N;                       // 强制non-transposed!
        ret.B_scale_inv = B.columnwise_scale_inv.dptr;
        ret.ldb = k;
    }
}
```

**关键发现**: RTX 4090的FP8 GEMM必须是TN layout → TE利用columnwise data(预计算转置)来强制所有FP8 GEMM变成TN!

这就是TE存储rowwise+columnwise双量化数据的真正原因:
- 不是为了避免backward重新量化(虽然也有这个好处)
- **主要是为了RTX 4090/Hopper的FP8 TN layout强制转换!**

### 4.2 FP8 Scaling Attributes设置 (cublaslt_gemm.cu:476-499)

```cpp
if (use_fp8) {
    // 1. FAST_ACCUM: 控制FP8累加精度
    const int8_t fastAccuMode = use_split_accumulator ? 0 : 1;  // 0=split, 1=fast
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode);

    // 2. Per-tensor scaling (DelayedScaling/CurrentScaling):
    if (is_tensor_scaling(A.scaling_mode) && is_tensor_scaling(B.scaling_mode)) {
        void *A_scale_inverse = param.A_scale_inv;  // FP32, 1/scale
        void *B_scale_inverse = param.B_scale_inv;  // FP32, 1/scale
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &A_scale_inverse);
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &B_scale_inverse);

        // cuBLAS ≥12.8: 设置scale mode
        scaling_mode_a = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;  // 单个FP32 scale
        scaling_mode_b = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    }
}
```

**FP8 GEMM的数学**:
```
D = alpha × (A_fp8 × scale_inv_A) × (B_fp8 × scale_inv_B) + beta × C
```

cuBLASLt内置处理:
- A_fp8 × scale_inv_A → dequantize A to FP32 in accumulator
- B_fp8 × scale_inv_B → dequantize B to FP32 in accumulator
- 结果乘积已经在高精度 → 输出直接为BF16/FP32

**这就是fused kernel的核心!**: 不需要Python-level dequant → cuBLASLt内部完成!

### 4.3 FP8 Alignment Requirements (cublaslt_gemm.cu:159-163)

```cpp
if (is_fp8_dtype(ret.Atype)) {
    NVTE_CHECK(ret.lda % 16 == 0,
               "Leading dimension requirement on A for FP8 GEMM. Caller must pad.");
}
if (is_fp8_dtype(ret.Btype)) {
    NVTE_CHECK(ret.ldb % 16 == 0,
               "Leading dimension requirement on B for FP8 GEMM. Caller must pad.");
}
```

**为什么lda/ldb需要%16==0?** HMMA tensor core的FP8 GEMM要求:
- FP8数据1 byte → 16 bytes对齐 = 16个FP8元素
- 这确保tensor core可以高效加载FP8数据块

### 4.4 cuBLASLt GEMM Launch (cublaslt_gemm.cu:766-783)

```cpp
// 获取最优算法
cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc,
                                preference, 1, &heuristicResult, &returnedResults);

// 执行GEMM
cublasLtMatmul(handle, operationDesc,
               alpha,                  // scalar: 1.0 (or NVFP4 device pointer)
               param.A, Adesc,         // A matrix + layout
               param.B, Bdesc,         // B matrix + layout
               beta,                   // scalar: 0.0 (not accumulate) or 1.0 (accumulate)
               C, Cdesc,               // C matrix (for accumulation)
               D, Ddesc,               // D output matrix
               &heuristicResult.algo,  // chosen algorithm
               aligned_workspace_ptr,  // workspace
               workspaceSize, stream); // workspace size + CUDA stream
```

**算法选择**: `cublasLtMatmulAlgoGetHeuristic` 自动选择最优GEMM算法:
- RTX 4090: HMMA (FP8 tensor core) → 16×8×16 或 16×8×32 tile sizes
- Hopper: WGMMA → 64×64×128 或更大 tiles
- Blackwell: WGMMA + TMA → 更高效的异步pipeline

### 4.5 FP8 Output Scale Update (cublaslt_gemm.cu:786-791)

```cpp
if (is_fp8_dtype(outputD->data.dtype) && outputD->scale_inv.dptr) {
    update_tensor_scale_inv(outputD, stream);
}
```

**FP8 output path**: 当GEMM输出也要量化为FP8时(例如FP8 output quantizer):
- cuBLASLt在GEMM末尾自动量化输出为FP8
- 需要设置 `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER` 和 `CUBLASLT_MATMUL_DESC_AMAX_D_POINTER`
- `update_tensor_scale_inv` 在GEMM完成后更新scale_inv

## 5. RTX 4090 vs Hopper vs Blackwell Dispatch对比

### 5.1 FP8 GEMM Layout

| GPU | FP8 Layout支持 | TE Canonicalize | Columnwise Data使用 |
|-----|--------------|----------------|--------------------|
| RTX 4090 (SM89) | **仅TN** | 强制所有FP8→TN | 必须用columnwise |
| Hopper (SM90) | **仅TN** | 强制所有FP8→TN | 必须用columnwise |
| Blackwell (SM100) | **任意** | 不强制 | rowwise即可 |

### 5.2 FP8 Scaling Mode

| Recipe | cuBLASLt Scale Mode | cuBLAS Version | RTX 4090 |
|--------|--------------------|---------------|---------|
| DelayedScaling (per-tensor) | `SCALAR_32F` | ≥12.1 | **✅** |
| CurrentScaling (per-tensor) | `SCALAR_32F` | ≥12.1 | **✅** |
| MXFP8 BlockScaling | `VEC32_UE8M0` | ≥12.8 | ❌ (需SM100) |
| FP8 BlockScaling | `VEC128_32F` / `BLK128x128_32F` | ≥12.9 | ❌ (需12.9+SM100) |
| NVFP4 | `VEC16_UE4M3` | ≥12.8 | ❌ (需SM100) |

### 5.3 Workspace Size

| GPU | Workspace | 原因 |
|-----|----------|------|
| RTX 4090 | **4MB** | FP8 TN GEMM足够 |
| Hopper | **32MB+1KB** | NVFP4需要32MB+1KB alignment |
| Blackwell | **32MB+1KB** | 同Hopper |

### 5.4 Epilogue Fusion

| Fusion | RTX 4090 | Hopper | Blackwell |
|--------|---------|--------|-----------|
| BIAS | ✅ | ✅ | ✅ |
| GELU+BIAS | ✅ | ✅ | ✅ |
| dBias | ✅ | ✅ | ✅ |
| dGELU+dBias | ✅ | ✅ | ✅ |
| dbias fusion (量化kernel) | ❌ (需SM100) | ❌ | ✅ |

## 6. 为什么RTX 4090 FP8 GEMM加速1.48-1.59x?

### 6.1 从源码看加速机制

```
1. Python: quantize(BF16→FP8) via tex.quantize() C++ kernel
   → CastVectorizedUnaryKernelLauncher (SIMT vectorized)
   → 生成 A_fp8 + scale_inv_A

2. C++: cublas_gemm()
   → CanonicalizeGemmInput: columnwise data强制TN layout
   → cuBLASLt: A_fp8 × scale_inv_A * B_fp8 × scale_inv_B → BF16 output
   → HMMA tensor core: FP8 E4M3 input → FP32 accumulator → BF16 output
   → Epilogue: BIAS/GELU fusion (可选)

3. 关键: cuBLASLt内部处理了dequant → 无Python overhead!
   → scale_inv在GEMM descriptor里 → cuBLASLt自动乘scale_inv
   → 输出直接BF16 → 无需Python dequantize kernel
```

### 6.2 HMMA FP8 Tensor Core on RTX 4090

RTX 4090 (Ada Lovelace, SM89)的HMMA指令:
- `HMMA.1688.FP8`: 16×8×8 tile, FP8 input, FP32 accumulator
- 每个warp每cycle可做: 2×HMMA = 16×16 FP8 MACs
- FP8 GEMM吞吐量: 2x BF16 GEMM (因为FP8 8bit vs BF16 16bit)
- 实测: 1.48-1.59x → 不完全2x → 量化overhead + kernel launch + workspace

### 6.3 为什么B=1反而慢?

从源码可以看出:
1. `tex.quantize()` kernel launch → 固定overhead (~0.5-1ms)
2. `get_cublas_workspace()` → 4MB allocation (LRU cache, 但首次有cost)
3. `CanonicalizeGemmInput()` → 可能需要columnwise data → 2x存储
4. cuBLASLt heuristic search → 每次GEMM可能重新选算法(有cache但首次慢)
5. GEMM太小(M=512, N=2560) → FP8 HMMA的tile效率低 → overhead占主导

## 7. 与CUTLASS 3.x的连接

### 7.1 TE vs CUTLASS GEMM Backend选择

```
TE的cuBLASLt GEMM:
  - 优点: 自动选最优算法, 无需手写kernel, 支持所有epilogue fusion
  - 缺点: 黑盒, 无法精细控制tile大小/pipeline策略
  - RTX 4090: HMMA SIMT kernel (cuBLASLt自动选择)

CUTLASS 3.x GEMM:
  - 优点: 可控制每一步(tiling, pipeline, epilogue), 可定制fusion
  - 缺点: 需手写kernel, 编译时间长, SM架构dispatch复杂
  - RTX 4090: 必须用SM80 path (cp.async + HMMA, 不支持TMA/WGMMA)
```

**TE的选择**: 用cuBLASLt作为主backend → 稳定+自动优化
**CUTLASS的使用**: 只在需要custom fusion(如量化+GEMM+dbias)时用 → Blackwell专用

### 7.2 TE的CUTLASS使用场景

TE只在以下场景使用CUTLASS(而非cuBLASLt):
1. **Atomic GEMM**: Comm+GEMM overlap的split GEMM → 需自定义chunk分配
2. **Grouped GEMM**: MoE的多expert GEMM → cutlass_grouped_gemm.cuh
3. **FP8 DPA (Dot Product Attention)**: FP8 attention → CUTLASS attention kernel
4. **Blackwell quantize kernel**: 1D/2D TMA加速量化 → 超出cuBLASLt能力

**RTX 4090**: 所有这些custom CUTLASS kernel要么不可用(SM100+)要么不如cuBLASLt!

## 8. 生产决策更新

| 场景 | 最佳配置 | 源码证据 |
|------|---------|---------|
| Training B≥4 | FP8 DelayedScaling TN | cuBLASLt fused (SCALAR_32F scale) + HMMA |
| Training B=1 | BF16 | FP8 quantize overhead > HMMA speedup |
| FP8 Forward only | FP8 DS fused | TN canonicalize + scale_inv in GEMM desc |
| FP8 Backward | FP8 DS columnwise | columnwise data for dgrad/wgrad TN layout |
| GELU+Bias fusion | cuBLASLt epilogue | CUBLASLT_EPILOGUE_GELU_AUX_BIAS |
| Communication quantize | FP8 quantize | scale传输而非数据→带宽减半 |
| MoE Grouped GEMM | CUTLASS grouped | nvte_multi_tensor_gemm on multi-stream |

---

**源码**: `transformer_engine/pytorch/cpp_extensions/gemm.py`, `transformer_engine/pytorch/csrc/extensions/gemm.cpp`, `transformer_engine/common/gemm/cublaslt_gemm.cu`
**关键函数**:
- `general_gemm()` → Python接口 (gemm.py:105)
- `gemm()` → C++ Pybind dispatch (gemm.cpp:144)
- `CanonicalizeGemmInput()` → FP8 TN强制转换 (cublaslt_gemm.cu:102)
- `cublas_gemm()` → cuBLASLt GEMM执行 (cublaslt_gemm.cu:315)
- `cublasLtMatmul()` → NVIDIA底层GEMM (cuBLASLt API)

**相关笔记**: `transformer-engine-fp8-rtx4090.md`, `cutlass-gemm-rtx4090.md`, `fp8-training-benchmark-rtx4090.md`, `tensor-core-architecture.md`