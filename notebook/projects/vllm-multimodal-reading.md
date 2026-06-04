# vLLM V1 Multi-Modal Processing 源码阅读

> Multi-Modal (MM) 处理: 图像/音频/视频输入, Encoder Runner, Hash-based Caching

## 1. MM 数据流概览

```
API 请求 (image + prompt)
    │
    ├── 1. Input Processing (BaseMultiModalProcessor)
    │     ├── 解析 MM 数据 (图像/音频/视频)
    │     ├── Prompt 替换/插入 placeholder tokens
    │     ├── Tokenization
    │     └── 创建 MultiModalFeatureSpec
    │
    ├── 2. EngineCoreRequest (mm_features 字段)
    │     └── list[MultiModalFeatureSpec] 传到 EngineCore
    │
    ├── 3. Scheduler (scheduled_encoder_inputs)
    │     └── 跟踪哪些 MM encoder 需要执行
    │
    ├── 4. EncoderRunner (worker/gpu/mm/encoder_runner.py)
    │     ├── prepare_mm_inputs() — 准备 MM 数据
    │     ├── execute_mm_encoder() — 运行 vision encoder
    │     │   └── model.embed_multimodal()
    │     └── gather_mm_embeddings() — 收集 encoder 输出
    │
    ├── 5. EncoderCache (hash-based)
    │     ├── mm_features: req_id → MM features
    │     └── encoder_outputs: mm_hash → tensors
    │
    └── 6. Model Forward
          ├── get_mm_embeddings() — 获取当前 step 需要的 embeddings
          └── inputs_embeds 代替 input_ids (MM token 位置)
```

## 2. 核心类型

### 2.1 MultiModalFeatureSpec

```python
class MultiModalFeatureSpec:
    data: ...           # 处理后的 MM 数据 (可为 None 如果已缓存)
    modality: str       # 类型: "image" / "audio" / "video"
    identifier: str     # Hash (用于缓存)
    mm_position: ...    # 在 prompt 中的位置
```

### 2.2 MM 数据类型

| 类型 | 类 | 输入格式 |
|------|-----|---------|
| Image | `HfImageItem` | PIL Image, numpy array, torch.Tensor |
| Video | `HfVideoItem` | list of images/arrays/tensors |
| Audio | `HfAudioItem` | list of floats/numpy/tensors |

## 3. MultiModalRegistry (`vllm/multimodal/registry.py`)

MM 处理的中央调度系统:

```python
class MultiModalRegistry:
    def register_processor(...)   # 注册 MM processor
    def supports_multimodal_inputs(...)  # 检查模型是否支持 MM
    def create_processor(...)     # 创建 processor 实例
    def get_dummy_mm_inputs(...)  # 创建 dummy 数据 (profiling)
```

### 3.1 Processor Factory 模式

每个模型注册三个工厂:

```python
@MULTIMODAL_REGISTRY.register_processor(
    processor=llava_processor_factory,        # MM 数据处理
    info=llava_processing_info_factory,       # 处理信息
    dummy_inputs=llava_dummy_inputs_factory,  # Dummy 输入 (profiling)
)
class LlavaForConditionalGeneration(...):
    pass
```

## 4. EncoderRunner (`vllm/v1/worker/gpu/mm/encoder_runner.py`)

V1 中 MM encoder 的执行入口:

```python
class EncoderRunner:
    def prepare_mm_inputs(self, ...):
        """准备 MM 数据用于 encoding"""

    def execute_mm_encoder(self, ...):
        """运行 MM encoder (如 CLIP/SigLIP)"""
        # 调用 model.embed_multimodal()

    def gather_mm_embeddings(self, ...):
        """收集当前 step 需要的 embeddings"""

    def get_inputs_embeds(self, ...):
        """合并 MM embeddings 和 token embeddings"""
```

### 4.1 Pipeline Parallelism 处理

MM encoder 只在 **第一个 PP rank** 上执行:
```python
# model_runner.py execute_model():
if supports_mm and is_first_pp_rank:
    mm_embeddings = model_state.get_mm_embeddings()
    # 传递 inputs_embeds 而非 input_ids
```

## 5. EncoderCache (`vllm/v1/worker/gpu/mm/encoder_cache.py`)

Hash-based 的 MM encoder 输出缓存:

```python
class EncoderCache:
    mm_features: dict[str, ...]      # req_id → MM features
    encoder_outputs: dict[str, ...]  # mm_hash → encoder tensors
```

缓存生命周期:
1. 请求到达 → MM features 注册到 cache
2. Encoder 执行 → 输出按 hash 缓存
3. 请求完成 → 释放对应的 cache entries
4. `free_encoder_mm_hashes`: 调度器追踪需要释放的 hashes

## 6. MM 与调度器交互

### 6.1 SchedulerOutput

```python
@dataclass
class SchedulerOutput:
    scheduled_encoder_inputs: dict[str, list[int]]  # 需要执行的 encoder
    free_encoder_mm_hashes: list[str]               # 需要释放的缓存
```

### 6.2 Encoder 执行时机

```
Step 1: Prefill (新请求, 有 MM 数据)
    ├── Scheduler 决定: scheduled_encoder_inputs = {req_id: [img_0, img_1]}
    ├── EncoderRunner 执行 vision encoder
    ├── 缓存 encoder 输出
    └── Model forward: 用 encoder 输出替代 placeholder tokens

Step 2: Decode (无新 MM 数据)
    ├── 无 scheduled_encoder_inputs
    └── 正常 decode
```

## 7. Vision Encoder 实现

### 7.1 CLIP (`vllm/model_executor/models/clip.py`)

```python
class CLIPVisionModel:
    """CLIP 视觉编码器"""
    def forward(self, pixel_values, ...):
        # Conv + Transformer layers → vision embeddings
```

### 7.2 SigLIP (`vllm/model_executor/models/siglip.py`)

```python
class SiglipVisionModel:
    """SigLIP 视觉编码器"""
```

### 7.3 模型集成 (以 LLaVA 为例)

```python
class LlavaForConditionalGeneration:
    vision_tower: CLIPVisionModel     # Vision encoder
    multi_modal_projector: Linear     # 投影层
    language_model: ...               # LLM

    def embed_multimodal(self, pixel_values):
        # 1. Vision encoder
        image_features = self.vision_tower(pixel_values)
        # 2. Project to LLM hidden dim
        return self.multi_modal_projector(image_features)
```

## 8. MM 处理管线详解

### 8.1 BaseMultiModalProcessor (`vllm/multimodal/processing/processor.py`)

```python
class BaseMultiModalProcessor:
    def _call_process_input(self, ...):
        """处理原始 MM 输入"""
        # 1. 替换 prompt 中的 placeholder
        # 2. Tokenization
        # 3. 创建 placeholder 位置信息
        # 4. 返回 MultiModalFeatureSpec
```

### 8.2 Placeholder Token 处理

```
原始 Prompt: "What's in this image? <image>"
处理后:      [what, 's, in, this, image, ?, <img_0>, <img_0>, ..., <img_0>]
                                                 ↑ placeholder tokens
                                                 (数量 = image feature length)
```

## 9. MM 接收缓存

`EngineCore.mm_receiver_cache`:

```python
# 引擎端缓存: 避免重复处理相同的 MM 数据
mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(vllm_config)
```

MultiModalCacheStats 追踪 MM 缓存命中率:
```python
vllm:mm_cache_queries  # MM 缓存查询次数
vllm:mm_cache_hits     # MM 缓存命中次数
```

## 10. 关键洞察

1. **Lazy Processing**: MM encoder 只在需要时执行，不是所有 step
2. **Hash-based Caching**: 相同图像不会重新 encode (跨请求复用)
3. **PP 只执行一次**: MM encoder 只在第一个 PP rank 执行
4. **Selective Gathering**: 每个 step 只收集当前需要的 embeddings
5. **inputs_embeds 替代 input_ids**: MM token 位置用 encoder 输出替代
6. **模块化设计**: 每种 modality 有独立的 processor 实现
7. **Dummy Inputs**: 用于 profiling 时预分配显存
8. **三种 MM 类型**: Image/Video/Audio 统一通过 MultiModalFeatureSpec
9. **与 Prefix Caching 正交**: MM 缓存和 KV 缓存是独立的
10. **调度器参与**: 调度器追踪 encoder 执行和缓存释放

## 参考资料

- `vllm/multimodal/registry.py` — MM Registry (中央调度)
- `vllm/multimodal/processing/processor.py` — Base Processor
- `vllm/multimodal/inputs.py` — MultiModalFeatureSpec
- `vllm/v1/worker/gpu/mm/encoder_runner.py` — Encoder Runner
- `vllm/v1/worker/gpu/mm/encoder_cache.py` — Encoder Cache
- `vllm/v1/worker/gpu/model_runner.py` — V1 MM 集成
- `vllm/model_executor/models/clip.py` — CLIP Vision Model
- `vllm/model_executor/models/siglip.py` — SigLIP Vision Model
- 相关: [V1 Architecture Map](vllm-v1-architecture-map.md)
