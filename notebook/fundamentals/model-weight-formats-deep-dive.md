# Model Weight Formats Deep Dive — Safetensors Specification + mmap/Zero-Copy + GGUF + ONNX + PyTorch pickle + vLLM Weight Loading Pipeline

> 2026-06-12 | LLM权重存储格式全链路分析: safetensors(二进制规格+8B header+N JSON+data buffer+mmap/zero-copy/lazy load)+GGUF(llama.cpp+内置量化+CPU推理)+ONNX(跨框架+图+权重)+PyTorch pickle(训练checkpoint+安全风险)+vLLM weight loading pipeline(discover→detect→parse→map→allocate→transfer)+训练→分发→部署管线+RTX 4090
> 源码: github.com/huggingface/safetensors (Rust core + Python/Rust/Go bindings)
> 关联: checkpoint.md, vllm-weight-loading-reading.md, quantization.md, fp8-quantization.md

## 0. 核心定律: 格式决定安全+速度+内存 — 训练→分发→部署3阶段不同格式

```
LLM权重生命周期3阶段+对应格式:

阶段1: 训练 → PyTorch .pt/.bin (pickle)
  → 需要optimizer state+scheduler+RNG state → 必须pickle → 训练专用
  → 安全风险: pickle可执行任意代码 → 只在可信环境用!
  → 大文件: 7B BF16=14GB params+56GB optimizer=70GB → 比safetensors大5x!

阶段2: 分发 → Safetensors (.safetensors)
  → 只存权重 → 不存optimizer → 安全+快速+跨框架 → 分发标准!
  → mmap+zero-copy → 加载10GB模型≈ms → 比pickle快10x+!
  → 多进程共享 → 内核管理page cache → 同文件物理内存共享!
  → → HuggingFace Hub默认格式 → 2025标准!

阶段3: 部署 → GGUF/ONNX/TensorRT-LLM
  → GGUF → CPU/边缘推理 → llama.cpp → 内置量化 → 消费级硬件
  → ONNX → 跨框架+硬件优化 → ONNX Runtime → 多后端
  → TensorRT-LLM → NVIDIA GPU最优 → FP8+kernel融合 → 生产GPU推理
  → → 格式选择取决于目标硬件!

关键设计定律:
  → safetensors: 零拷贝+mmap+lazy → 加载最快 → 内存最省 → 多进程共享
  → GGUF: 单文件+内置量化+mmap → CPU推理最优 → llama.cpp生态
  → ONNX: 图+权重+跨框架 → 硬件加速 → 多后端推理
  → pickle: 完整状态+可执行代码 → 训练唯一选择 → 但危险!
  → → → 格式不可替代! 训练用pickle → 分发用safetensors → 部署用GGUF/ONNX
```

## 1. Safetensors 二进制格式规范

```
官方规范 (github.com/huggingface/safetensors README):

文件结构:
  [8字节] → N = header长度 (uint64 little-endian)
  [N字节] → JSON UTF-8 header → tensor metadata + __metadata__
  [剩余]  → byte-buffer → 连续tensor数据 → 无空洞!

Header JSON结构:
  {
    "__metadata__": {"format": "pt"},  // 自由字符串→字符串映射
    "model.layers.0.weight": {
      "dtype": "BF16",                  // 数据类型
      "shape": [4096, 4096],            // tensor形状
      "data_offsets": [0, 33554432]     // [START, END] → 字节偏移(相对byte-buffer开头)
    },
    "model.layers.1.weight": {
      "dtype": "BF16",
      "shape": [4096, 4096],
      "data_offsets": [33554432, 67108864]
    }
  }

dtype支持:
  → BOOL (1bit) → U8 (uint8) → I8 (int8) → I16 (int16) → I32 (int32) → I64 (int64)
  → F16 (float16) → BF16 (bfloat16) → F32 (float32) → F64 (float64)
  → → 不支持: F8(E4M3/E5M2) → 不在规范! → FP8权重需其他格式(GGUF/custom)
  → → → vLLM FP8推理: FP8权重存BF16/F16 safetensors → 运行时quantize → 或自定义格式!

约束:
  → Header必须以{开头(0x7B) → 识别safetensors文件
  → Header可以尾部空格填充(0x20) → 8字节对齐!
  → data_offsets: [BEGIN, END] → END是one-past-end → tensor大小=END-BEGIN
  → byte-buffer必须完全索引 → 无空洞 → 防止polyglot文件(安全!)
  → 不允许重复key → 唯一tensor名
  → 允许空tensor(shape含0) → 不存数据 → header有size
  → 允许0-rank tensor(shape=[]) → scalar值
  → NaN/±Inf允许 → 不检查值 → 只检查结构!

7B模型safetensors文件示例:
  → Llama-2-7B: ~13.5GB safetensors → 多文件(model-00001-of-00002.safetensors等)
  → → 每文件≈6.75GB → header≈几KB → 数据≈6.75GB → header占比<0.001%
  → → 加载时间: mmap≈0.1ms(只映射) → 实际访问时才page fault → 惰性!
  → → → vs pickle .bin: 13.5GB → 反序列化≈30s → mmap≈0.1ms → 300,000x差距!
```

## 2. Safetensors mmap/Zero-Copy/Lazy Loading架构

```
### 2.1 mmap加载流程

safe_open("model.safetensors", framework="pt", device="cpu"):
  1. 读取8字节 → N = header长度
  2. 读取N字节 → JSON header → 解析tensor metadata
  3. mmap整个文件 → mmap(fd, filesize) → 返回虚拟内存映射
  4. → 不读取数据! → 只建立映射 → 页表设置 → 0.1ms!
  5. 返回SafeTensorsFile对象 → 包含metadata+mmap指针

f.get_tensor("model.layers.0.weight"):
  1. 从metadata查找 → dtype=BF16, shape=[4096,4096], data_offsets=[0,33554432]
  2. 计算绝对偏移 → header_end + data_offsets[0] = 8+N+0
  3. 从mmap区域创建tensor → torch.from_buffer(mmap_ptr+offset, dtype, shape)
  4. → 不复制! → 直接引用mmap内存 → zero-copy!
  5. → → 页故障时才从磁盘读取 → 惰性 → 按需加载!

### 2.2 多进程共享物理内存

  Process A → mmap("model.safetensors") → 虚拟地址VA_A → 物理页PA
  Process B → mmap("model.safetensors") → 虚拟地址VA_B → 同一物理页PA!
  → → 内核page cache管理 → 同文件→同物理页 → 2进程共享14GB→只占14GB物理内存!
  → → → vs pickle: 每进程独立反序列化 → 2进程14GB→占28GB → 多进程内存翻倍!

  vLLM多进程推理:
    → 8 GPU workers → 8进程 → 同safetensors文件 → mmap
    → → 14GB模型 → 8进程共享 → 只占14GB物理内存 → 省7×14=98GB!
    → → → vs pickle: 8×14=112GB → mmap省98GB → 8进程推理内存减半!

### 2.3 Lazy Loading选择性加载

  只加载需要的tensor → 不加载全部:
    → f.get_tensor("model.layers.0.weight") → 只映射这一页 → 不读其余!
    → → 模型有200个tensor → 只需要1个 → 只读1个 → 省199个的内存+时间!
    → → → 量化: 加载BF16→quantize→FP8 → 只需要target层 → 惰性quantize!

  vLLM lazy loading架构(2025新):
    → 分离weight iteration和weight materialization
    → → 1. 遍历metadata → 知道所有tensor名和shape → 不加载数据!
    → → 2. 分配GPU内存 → 根据shape和dtype → 不需要CPU数据!
    → → 3. 惰性传输 → mmap→CPU tensor→GPU tensor → 按需 → 省峰值内存!

### 2.4 Rust核心实现架构

  safetensors_rust/ → Rust参考实现 → 最权威!
    → Safetensors::read_metadata() → 只读header → 不读数据 → 惰性!
    → Safetensors::lazy_mmap() → mmap-backed view → 惰性访问
    → Safetensors::deserialize() → 全量加载 → 仍zero-copy(从mmap)
    → Safetensors::serialize() → 写header+data → 顺序写 → 快!

  Python绑定:
    → PyO3桥接 → Rust→Python → 高性能
    → 纯Python fallback → 无Rust时用 → 较慢但仍安全
    → → safetensors/torch.py → PyTorch集成 → load_file/save_file
    → → safetensors/numpy.py → NumPy集成
    → → safetensors/tensorflow.py → TF集成
    → → safetensors/flax.py → JAX/Flax集成

  C++ header-only实现:
    → safetensors_cpp/ → 单头文件 → #include → 无依赖!
    → → vLLM/TensorRT-LLM可能用C++版本 → 高性能 → 无Python overhead

  多语言绑定:
    → Python → Rust → Go → JavaScript/WASM → 全覆盖!
    → → WASM → 浏览器加载 → 模型在Web端 → 未来方向!
```

## 3. GGUF — llama.cpp CPU/边缘推理格式

```
GGUF = GGML Universal Format → llama.cpp专用 → 消费级硬件推理!

文件结构:
  → 单文件 → 所有权重+元数据+tokenizer+量化参数 → 一站式!
  → → header → metadata(KV pairs) → tensor info → tensor data → 连续!
  → → → 不需要多个文件 → 单文件分发 → 下载简单!

量化内置:
  → Q2_K → 2bit量化 → 最小 → 精度损失大 → 极端压缩
  → Q3_K/Q4_K/Q5_K → 3/4/5bit → 平衡 → Q4_K_M最常用
  → Q6_K/Q8_0 → 6/8bit → 高精度 → 体积大 → 接近FP16
  → IQ量化(IQ2_XS/IQ3_XXS) → 2024新 → 极端压缩 → 进一步减小
  → → → 内置量化! → 不需要外部quantize → 直接下载即用!

GGUF关键特性:
  → mmap支持 → 类似safetensors → 惰性加载 → CPU内存映射
  → tokenizer内置 → 不需要额外tokenizer.json → 一文件一切!
  → 元数据丰富 → general.architecture→模型类型→注意力类型→...
  → → 单文件推理 → llama.cpp直接加载 → 无需PyTorch/safetensors依赖!

GGUF vs safetensors:
  → GGUF: 推理专用 → CPU优化 → 内置量化 → 单文件 → llama.cpp生态
  → safetensors: 分发/训练→推理转换 → GPU优化 → 无量化 → 多文件 → HF生态
  → → 不可互换! → safetensors→GGUF需要convert.py → 量化→格式转换!

RTX 4090 GGUF:
  → llama.cpp → CUDA backend → GGUF on GPU → 7B Q4_K_M ≈4GB → 24GB够6个模型!
  → → 但! vLLM不原生支持GGUF → vLLM用safetensors → 不同生态!
  → → → GGUF用于llama.cpp/ollama → safetensors用于vLLM/SGLang → 不同工具!
```

## 4. ONNX — 跨框架模型交换格式

```
ONNX = Open Neural Network Exchange → Microsoft+Meta+AWS → 跨框架标准!

文件结构:
  → 单文件 → 模型图(nodes+edges)+权重+元数据 → protobuf编码
  → → graph = 计算图 → 节点=算子 → 边=tensor流动 → 完整模型定义!
  → → → safetensors只存权重 → ONNX存图+权重 → 更完整但也更复杂!

ONNX Runtime推理后端:
  → CPU → OpenVINO/Intel → x86优化 → AVX-512/VNNI
  → GPU → CUDA/TensorRT → NVIDIA优化 → kernel融合
  → NPU → CoreML/Hexagon → 移动/边缘 → 低功耗
  → → → 多后端! → 同一ONNX模型 → 不同硬件 → 不同后端 → 自动选择!

ONNX量化:
  → 动态量化 → INT8权重 → 运行时quantize → 简单但精度损失
  → 静态量化 → INT8权重+INT8计算 → 校准数据 → 更好精度
  → QDQ(QuantizeDequantize) → 量化节点嵌入图 → 精确控制 → 2025新!
  → → → Olive工具 → PyTorch→ONNX→量化→优化 → 自动化管线!

ONNX vs safetensors:
  → ONNX: 图+权重 → 跨框架 → 多后端 → 推理优化 → 但训练不支持!
  → safetensors: 权重only → 跨框架 → mmap → 分发标准 → 但无图!
  → → → 互补! → ONNX用于硬件优化推理 → safetensors用于模型分发!
```

## 5. PyTorch pickle (.pt/.bin) — 训练checkpoint格式

```
PyTorch pickle = torch.save/torch.load → Python pickle序列化 → 训练专用!

存储内容:
  → state_dict → 模型权重 → 必须存
  → optimizer.state_dict → AdamW m+v → 必须存(恢复训练!)
  → scheduler.state_dict → LR调度状态 → 建议存
  → rng_state → 随机数状态 → 建议存(恢复训练!)
  → epoch/step → 进度标记 → 建议存
  → → 完整训练状态 → 可以精确恢复 → 训练必须!

安全风险:
  → pickle反序列化 → 执行任意Python代码 → 恶意模型文件 = 远程代码执行!
  → → __reduce__ → 自定义反序列化 → 可以import os→os.system("rm -rf /") → 灾难!
  → → → 下载不明来源.pt文件 → 可能被攻击 → 必须信任来源!
  → → → → HuggingFace: 只信任safetensors → 自动将.bin转为safetensors → 安全!

加载性能:
  → 反序列化 → 逐对象重建 → Python解释器开销 → 极慢!
  → → 7B .bin → 反序列化≈30s → vs safetensors mmap≈0.1ms → 300,000x差距!
  → → → 原因: pickle重建每个Python对象 → dict→str→Tensor→逐个→慢!
  → → → → safetensors: mmap→直接内存映射→不重建→快!

内存开销:
  → 反序列化创建新对象 → 每进程独立 → 不能共享物理内存!
  → → 8进程×14GB=112GB → vs safetensors mmap: 14GB(共享) → 8x差距!
  → → → 分布式训练: 每进程独立加载 → 内存翻倍 → 不可接受!
  → → → → vLLM: 用safetensors+mmap → 8进程共享14GB → 省98GB!

PyTorch pickle仍然必要:
  → 训练恢复 → optimizer+scheduler+rng → pickle唯一能存完整状态
  → → safetensors只存权重 → 不存optimizer → 不能恢复训练!
  → → → 训练: 用pickle(.pt) → 分发: 转safetensors → 两种格式并存!
```

## 6. vLLM Weight Loading Pipeline

```
文件: vllm/v1/worker/weight_loader.py + vllm/model_executor/model_loader/

### 6.1 加载管线架构

vLLM weight loading pipeline (discover→detect→parse→map→allocate→transfer):

Step 1: Discovery → 扫描模型目录 → 找所有权重文件
  → glob(model_dir, "*.safetensors") → 找safetensors文件
  → glob(model_dir, "*.bin") → 找pickle文件 → 优先safetensors!

Step 2: Format Detection → 识别文件格式
  → .safetensors → SafetensorsModelLoader → mmap路径
  → .bin → DefaultModelLoader → pickle路径(慢!)
  → .gguf → GGUFModelLoader → llama.cpp路径
  → → 优先safetensors → 最快+mmap → 生产默认!

Step 3: Metadata Parsing → 解析tensor元数据
  → safetensors → 读8字节header长度 → 读JSON header → tensor列表
  → → 不读数据! → 只读metadata → 惰性! → 知道所有tensor名和shape
  → → → vs pickle: 反序列化整个文件 → 慢!

Step 4: Weight Mapping → 映射tensor名到模型参数
  → source_name → target_name → 可能需要rename/concat/split
  → → 例: "model.layers.0.self_attn.q_proj.weight"
  → → → → 映射到 model.self_attn.q_proj.weight → 直接对应!
  → → → → 例: QKV合并 → q/k/v各自weight → concat为一个QKV weight → 合并!
  → → → → → WeightMapping → 映射规则 → 处理rename+concat+split!

Step 5: Memory Allocation → 分配GPU tensor
  → 根据shape+dtype → torch.empty(shape, dtype, device="cuda")
  → → 不填充数据 → 只分配内存 → 快!
  → → → 7B BF16 → 14GB GPU内存 → 分配≈0.5s → 主要是CUDA malloc

Step 6: Weight Transfer → CPU→GPU数据传输
  → mmap→CPU tensor → GPU tensor → cudaMemcpy
  → → 惰性! → 逐tensor传输 → 不一次性全部!
  → → → PCIe: 14GB→≈1.2s(H→D) → 比pickle反序列化快25x!

### 6.2 SafetensorsModelLoader — mmap优化路径

class SafetensorsModelLoader:
  → load_model() → safe_open(path, framework="pt") → mmap!
  → → 遍历f.keys() → 逐tensor → f.get_tensor(name) → mmap→CPU
  → → → CPU→GPU → cudaMemcpy → 逐层加载!
  → → → → 优势: 不反序列化 → 不创建Python对象 → 纯内存映射!
  → → → → → 7B加载时间: mmap≈0.1ms(映射) + H→D≈1.2s ≈1.2s → vs pickle≈31.2s!

  多文件处理:
    → 大模型分多个safetensors文件 → 逐文件mmap
    → → model-00001-of-00008.safetensors → 8个文件 → 8次mmap
    → → → 但mmap几乎0开销 → 8次mmap≈0.8ms → negligible!
    → → → → 顺序遍历文件 → 逐文件逐tensor → 线性加载 → 简单高效!

### 6.3 量化权重加载

  FP8权重加载(2025新):
    → safetensors存BF16权重 → vLLM运行时quantize → FP8
    → → load BF16 → quantize_to_fp8 → 存到FP8 GPU tensor
    → → → 或: 自定义FP8 safetensors → 直接存FP8 → 但safetensors规范不支持F8 dtype!
    → → → → vLLM方案: BF16 safetensors → 运行时FP8 quantize → 兼容规范!
    → → → → → 但开销! → quantize 14GB→1.75GB → ≈1s → 可接受!

  INT4权重加载:
    → AWQ/GPTQ safetensors → 自定义dtype → 可能用U8/I8存储pack后的INT4
    → → 解pack → INT4→BF16 → 或直接INT4 kernel → 无需解pack!
    → → → vLLM: INT4 weight → fused kernel → 直接INT4 GEMM → 不解pack!

### 6.4 FSDP checkpoint → safetensors转换

  FSDP训练 → 分片checkpoint → 每rank只存1/N参数
  → → 8个分片文件 → shard_0.pt/shard_1.pt/.../shard_7.pt → pickle!
  → → → 合并 → consolidate → 全参数 → 转safetensors → 分发!

  合并流程:
    → 每rank加载自己分片 → all_gather重建完整参数 → 单进程合并!
    → → 7B ZeRO-3: 8分片×~2GB=16GB → 合并→14GB safetensors → 去冗余!
    → → → 合并时间 ≈30s(pickle反序列化) → 一次性 → 可接受
    → → → → 合并后永久用safetensors → 不需要再pickle!
```

## 7. 格式选择决策树

```
决策树: 根据场景选择格式

训练场景:
  → 正常训练 → PyTorch .pt → pickle → optimizer+scheduler+rng → 完整恢复
  → 分布式训练 → FSDP分片 .pt → 每rank分片 → 合并后转safetensors
  → → → 只在训练时用pickle → 分发时转safetensors!

分发场景:
  → HuggingFace Hub → safetensors → 默认! → 安全+快速+mmap
  → → .bin → HuggingFace自动转safetensors → 用户只下载safetensors!
  → → → 不分发.pt → 安全风险 → 只在可信环境共享!

推理场景:
  → GPU推理(vLLM/SGLang) → safetensors → mmap → 最快加载!
  → → 量化推理 → safetensors(BF16) → 运行时quantize → 或量化safetensors
  → CPU推理(llama.cpp) → GGUF → 内置量化 → mmap → CPU最优!
  → 跨硬件推理 → ONNX → 多后端 → 硬件加速!
  → NVIDIA GPU最优 → TensorRT-LLM engine → FP8+kernel融合 → 生产级!

RTX 4090决策:
  → 训练 → .pt → 单GPU → optimizer offload → 唯一选择!
  → 推理 → safetensors → vLLM → mmap → 14GB→1.2s → 最快!
  → → INT4推理 → safetensors(AWQ/GPTQ) → vLLM → 量化kernel → 最快!
  → CPU推理 → GGUF(Q4_K_M) → llama.cpp → 4GB→RTX 4090 CUDA → 可行但不如vLLM!
  → → → RTX 4090: safetensors+vLLM → GPU推理最优!
```

## 8. RTX 4090 Weight Format Implications

```
1. safetensors mmap对RTX 4090:
  → 14GB模型 → mmap≈0.1ms → H→D≈1.2s → 总≈1.2s → 极快!
  → → vs pickle: 反序列化≈30s → H→D≈1.2s → 总≈31.2s → 慢26x!
  → → → 生产必须safetensors → 加载快26x → 首次响应快!

2. 多进程共享对RTX 4090:
  → vLLM TP=1 → 单进程 → 不需要共享 → mmap仍然快(避免反序列化)
  → → 但! TP>1 → 多进程 → mmap共享14GB → 只占14GB → 省内存!
  → → → RTX 4090: TP=1 → 不需要多进程共享 → 但mmap零拷贝仍然最优!

3. safetensors不支持FP8 dtype:
  → 规范只有BF16/F16/F32 → FP8(E4M3/E5M2)不在规范!
  → → vLLM FP8推理: BF16 safetensors → 运行时FP8 quantize → ≈1s
  → → → 或: 自定义格式 → FP8 tensor → 不兼容HF Hub!
  → → → → RTX 4090: BF16→FP8 quantize开销≈1s → 可接受 → 但如果能直接FP8更好!

4. GGUF对RTX 4090:
  → llama.cpp CUDA backend → GGUF→GPU → Q4_K_M→4GB→24GB够6个模型!
  → → 但! vLLM不支持GGUF → 不同工具链 → 需要safetensors→GGUF转换
  → → → llama.cpp适合本地实验 → vLLM适合生产服务 → 不同场景!

5. 模型下载+safetensors对RTX 4090:
  → 中国大陆网络 → HuggingFace慢 → 需镜像(hf-mirror.com) → 下载14GB≈10min
  → → safetensors文件大小 ≈模型参数×2(BF16) → 7B=14GB → 可接受
  → → → INT4 AWQ safetensors → ≈4GB → 下载更快 → 量化推理更快!
  → → → → Qwen2.5-7B-Instruct AWQ → ≈4GB → vLLM INT4推理 → 最优配置!

6. 生产最优格式链:
  → 训练: .pt(pickle) → ZeRO-2+offload → 单GPU → optimizer在CPU
  → → 合并: consolidate → 全参数 → 转safetensors → 一次转换
  → → 推理: safetensors → vLLM → INT4+INT8KV+GQA-8 → 最快推理!
  → → → 或: safetensors→AWQ量化→INT4 safetensors → vLLM INT4推理 → 更小更快!
  → → → → 全链路: pickle训练→safetensors分发→vLLM推理 → 格式转换一次!
```

## 参考文献

```
1. Safetensors源码和规范:
   - github.com/huggingface/safetensors — Rust core + 多语言bindings
   - huggingface.co/docs/safetensors/en/spec — 二进制格式规范
   - huggingface.co/blog/safetensors-the-new-way-to-save-tensors — 为什么safetensors

2. GGUF格式:
   - github.com/ggerganov/gguf — GGUF规范
   - github.com/ggerganov/llama.cpp — llama.cpp实现

3. ONNX:
   - onnx.ai — ONNX规范
   - github.com/microsoft/onnxruntime — ONNX Runtime

4. vLLM Weight Loading:
   - vllm/v1/worker/weight_loader.py — vLLM V1 weight loader
   - vllm/model_executor/model_loader/ — ModelLoader hierarchy

5. 我们的笔记:
   - checkpoint.md → checkpoint基础
   - vllm-weight-loading-reading.md → vLLM weight loading详解
   - quantization.md → 量化基础
   - fp8-quantization.md → FP8量化深度