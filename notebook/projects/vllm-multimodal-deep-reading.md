# vLLM V1 Multimodal Inference 架构深度阅读

> 基于源码 `vllm-latest/` (2026-06), 深入分析 multimodal 推理全流程
> 前置: [vllm-multimodal-reading.md](vllm-multimodal-reading.md) 基础概览

---

## 1. Multimodal Input Pipeline: 从原始输入到模型 Embeddings

### 1.1 端到端数据流

```
 用户请求
 "Describe this image <image_url>"
    |
    v
[1] MediaConnector (vllm/multimodal/media/connector.py)
    |-- 下载/加载原始媒体 (URL/file/base64)
    |-- ImageMediaIO / AudioMediaIO / VideoMediaIO
    |-- 线程池并发加载 (global_thread_pool)
    |
    v
[2] BaseMultiModalProcessor.apply() (processor.py:1663)
    |-- _cached_apply_hf_processor()
    |     |-- 计算 mm_hashes (blake3/sha256/sha512)
    |     |-- 检查 processor cache (LRU/SHM)
    |     |-- 仅处理 cache miss 的 items
    |     |-- 调用 HuggingFace processor (CLIPImageProcessor 等)
    |-- _maybe_apply_prompt_updates()
    |     |-- PromptReplacement: <image> -> <img> x N 个 placeholder
    |     |-- PromptInsertion: 在指定位置插入 placeholder tokens
    |     |-- PlaceholderRange 记录每个 MM item 在 prompt 中的位置
    |
    v
[3] MultiModalInput
    |-- prompt_token_ids: [1, 3148, ..., 32000, ..., 32000, ...]
    |                              ^^^^^^^^  N 个 placeholder  ^^^^^^^^
    |-- mm_kwargs: {"pixel_values": tensor, "image_grid_thw": tensor}
    |-- mm_hashes: ["blake3_abc123..."]
    |-- mm_placeholders: {"image": [PlaceholderRange(offset=5, length=576)]}
    |
    v
[4] MultiModalFeatureSpec (per item)
    |-- data: MultiModalKwargsItem (处理后的 tensor 数据)
    |-- modality: "image" / "audio" / "video" / "prompt_embeds"
    |-- identifier: hash string (含 LoRA 前缀)
    |-- mm_position: PlaceholderRange(offset, length, is_embed)
    |-- mm_hash: processor cache hash (不含 LoRA 前缀)
    |
    v
[5] Scheduler (scheduled_encoder_inputs)
    |
    v
[6] GPUModelRunner._execute_mm_encoder() (gpu_model_runner.py:2870)
    |-- model.embed_multimodal(pixel_values=..., ...)
    |-- 输出缓存到 encoder_cache[mm_hash] = tensor
    |
    v
[7] GPUModelRunner._gather_mm_embeddings() (gpu_model_runner.py:3081)
    |-- 从 encoder_cache 取出 embeddings
    |-- is_mm_embed mask 标记哪些位置是 MM tokens
    |
    v
[8] model.embed_input_ids(input_ids, mm_embeds, is_multimodal)
    |-- 先获取 text embeddings
    |-- 再用 mm_embeds 覆盖 is_multimodal=True 的位置
    |-- 输出: inputs_embeds (统一表示)
```

### 1.2 Input Processing 三阶段

`BaseMultiModalProcessor.apply()` (processor.py:1663-1707) 的三个核心步骤:

```
阶段 1: _cached_apply_hf_processor (processor.py:1441)
    输入: ProcessorInputs (prompt + mm_data + mm_uuid)
    处理:
      a) 计算 mm_hashes (每个 MM item 的 hash)
      b) 检查 processor cache (P0/LRU 或 SHM)
      c) 仅对 cache miss 的 items 调用 HF processor
      d) 将 processed data 拆分为 MultiModalKwargsItems
    输出: prompt_ids, mm_info (kwargs + hashes + prompt_updates)

阶段 2: _maybe_apply_prompt_updates (processor.py:1688)
    输入: prompt_ids, mm_kwargs, mm_prompt_updates
    处理:
      a) 遍历每个 MM item 的 PromptReplacement/PromptInsertion
      b) 将 <image> 等占位符替换为 N 个 placeholder token
      c) N = vision encoder 输出的 feature 数量
    输出: 更新后的 prompt_ids, mm_placeholders

阶段 3: 返回 MultiModalInput
    prompt_token_ids + mm_kwargs + mm_hashes + mm_placeholders
```

### 1.3 Prompt Replacement 机制

`PromptReplacement` (processor.py:422-496) 和 `PromptInsertion` (processor.py:353-419):

```
LLaVA 示例:
  原始 prompt: "USER: <image>\nWhat's in this image?\nASSISTANT:"
  Token IDs:   [1, 3148, 1001, 29901, 29871, 32000, 29871, 13, ...]
                                             ^^^^^  1 个 <image> token

  PromptReplacement:
    target: <image> (token_id=32000)
    replacement: <image> x 576 (CLIP 输出 24x24=576 tokens)

  替换后:
    [1, 3148, 1001, 29901, 29871, 32000, ..., 32000, 29871, 13, ...]
                                  ^^^^^^^^^^  576 个 placeholder
    PlaceholderRange(offset=5, length=576)

PromptInsertion 示例 (如 Pixtral):
    target: PromptIndexTargets.start() (prompt 开头)
    insertion: <image> x feature_size
    mode: INSERT
```

`PromptUpdateDetails` (processor.py:206) 支持 `is_embed` mask:
- 对于 `<image_bos> <image>xN <image_eos>` 格式
- 只有 `<image>` 位置被标记为需要 embed
- `<image_bos>` 和 `<image_eos>` 仍用 text embedding

---

## 2. EncoderRunner: Vision Encoder 的独立执行

### 2.1 V1 架构中的 Encoder 执行

V1 中没有独立的 `EncoderRunner` 类。Encoder 执行直接嵌入在
`GPUModelRunner._execute_mm_encoder()` (gpu_model_runner.py:2870-3079) 中。

```
_execute_mm_encoder(scheduler_output)
    |
    v
_batch_mm_inputs_from_scheduler()  (gpu_model_runner.py:2835)
    |-- 遍历 scheduled_encoder_inputs: {req_id: [input_ids]}
    |-- 从 req_state.mm_features[input_id] 取出 data
    |-- 收集 mm_hashes, mm_kwargs, mm_lora_refs
    |
    v
prompt_embeds 透传 (gpu_model_runner.py:2885-2905)
    |-- modality == "prompt_embeds" 时跳过 encoder
    |-- 直接注入 encoder_cache: encoder_cache[mm_hash] = tensor
    |
    v
按 modality 分组批处理 (gpu_model_runner.py:2994)
    |-- group_and_batch_mm_kwargs()
    |-- 同一 modality 的 items 尽量 batch 在一起
    |
    v
执行 encoder:
    if encoder_cudagraph_manager 支持该 modality:
        cudagraph_output = encoder_cudagraph_manager.execute(mm_kwargs)
    else:
        batch_outputs = model.embed_multimodal(**mm_kwargs)
    |
    v
缓存结果 (gpu_model_runner.py:3073-3077)
    for mm_hash, output in zip(mm_hashes, encoder_outputs):
        encoder_cache[mm_hash] = output
        maybe_save_ec_to_connector(encoder_cache, mm_hash)  # P/D 分离
```

### 2.2 model.embed_multimodal() 实现 (LLaVA 示例)

`llava.py:661-666`:

```python
def embed_multimodal(self, **kwargs):
    image_input = self._parse_and_validate_image_input(**kwargs)
    return self._process_image_input(image_input)

# _process_image_input (llava.py:643):
#   1. vision_tower(pixel_values) -> image_features  (CLIP/SigLIP)
#   2. multi_modal_projector(image_features) -> image_embeds  (Linear)
#   输出: (num_items, feature_size, hidden_size) 或 list[Tensor]
```

LLaVA 组件:
```
                    pixel_values
                        |
                        v
            +-----------------------+
            |   vision_tower        |  CLIPVisionModel / SiglipVisionModel
            |   (Conv + Transformer)|
            +-----------------------+
                        |
                image_features
                (576 tokens, 1024 dim)
                        |
                        v
            +-----------------------+
            | multi_modal_projector |  LlavaMultiModalProjector (Linear)
            +-----------------------+
                        |
                image_embeds
                (576 tokens, 4096 dim = LLM hidden_size)
```

### 2.3 Encoder CUDA Graph

`EncoderCudaGraphManager` (encoder_cudagraph.py:53) 对 vision encoder
使用 CUDA Graph 加速:

```
初始化:
    token_budgets = [256, 512, 1024, 2048, ...]  # 2 的幂次
    max_batch_size = min(max_budget // min_budget, min_budget)

捕获:
    for budget in token_budgets:
        _capture_budget_graph(budget)
        |-- model.prepare_encoder_cudagraph_capture_inputs(budget, ...)
        |-- model.encoder_cudagraph_forward(inputs)
        |-- 保存 graph + input_buffers + output_buffer

执行:
    total_tokens = sum(item.output_tokens)
    budget = _find_smallest_fitting_budget(total_tokens)
    |-- prepare_encoder_cudagraph_replay_buffers(mm_kwargs)
    |-- 拷贝数据到 input_buffers
    |-- graph.replay()
    |-- 读取 output_buffer
```

`EncoderCudaGraphConfig` (encoder_cudagraph_defs.py:31):
```python
@dataclass
class EncoderCudaGraphConfig:
    modalities: list[str]           # ["image"] 或 ["image", "video"]
    buffer_keys: list[str]          # 输入 buffer 名称
    out_hidden_size: int            # encoder 输出维度
    padding_logics: dict[str, ...]  # 自定义 padding 逻辑
    max_frames_per_video: int = 1   # 视频帧数上限
```

---

## 3. EncoderCache: Hash-based 缓存机制

### 3.1 三层缓存架构

vLLM V1 的 multimodal 缓存分三层:

```
Layer 1: Processor Cache (P0 = API Server 进程)
    位置: vllm/multimodal/cache.py
    缓存: HF processor 输出 (tensor 数据 + prompt_updates)
    类型: MultiModalProcessorOnlyCache / MultiModalSenderCache / ShmObjectStoreSenderCache
    目的: 避免重复调用 HF processor (图像预处理、resize、normalize)
    大小: 由 --mm-processor-cache-gb 控制

Layer 2: Encoder Cache Manager (Scheduler/Core 进程)
    位置: vllm/v1/core/encoder_cache_manager.py
    缓存: 调度信息 (mm_hash -> set[req_id])
    功能:
      - check_and_update_cache(): 检查 hash 是否已缓存
      - can_allocate(): 检查是否有足够空间
      - allocate(): 分配空间给新 encoder 输出
      - free(): 请求完成时释放引用
      - 驱逐策略: FIFO (freeable OrderedDict, popitem(last=False))
    大小: encoder_cache_size (以 embedding 数量计)

Layer 3: GPU Encoder Cache (Worker 进程)
    位置: gpu_model_runner.py:530-531
    缓存: dict[str, torch.Tensor]  # mm_hash -> encoder_output tensor
    功能:
      - _execute_mm_encoder() 写入
      - _gather_mm_embeddings() 读取
      - free_encoder_mm_hashes 时 pop 释放
    存储: GPU 显存
```

### 3.2 EncoderCacheManager 详细流程

`EncoderCacheManager` (encoder_cache_manager.py:17-266):

```
状态:
    cached: dict[str, set[str]]         # mm_hash -> {req_id_1, req_id_2, ...}
    freeable: OrderedDict[str, int]     # mm_hash -> num_encoder_embeds (无引用)
    freed: list[str]                    # 本次 step 被驱逐的 mm_hash
    num_free_slots: int                 # 可用空间
    num_freeable_slots: int             # 可回收空间

生命周期:
    请求到达 + 调度:
        check_and_update_cache(req, input_id)
            |-- mm_hash = req.mm_features[input_id].identifier
            |-- if mm_hash not in cached: return False (未缓存)
            |-- if cached[mm_hash] 为空: 从 freeable 移除, 恢复 freeable_slots
            |-- cached[mm_hash].add(req_id)  (添加引用)
            |-- return True (已缓存, 跳过 encoder)

        can_allocate(req, input_id, compute_budget, num_to_schedule)
            |-- num_embeds = req.get_num_encoder_embeds(input_id)
            |-- if num_embeds > compute_budget: return False
            |-- if num_embeds <= num_free_slots: return True
            |-- if num_embeds > num_freeable_slots: return False
            |-- 否则: 驱逐 freeable 中最旧的 entries 直到有空间
            |--    freed.append(mm_hash)  (通知 worker 释放 GPU 缓存)
            |-- return True

        allocate(req, input_id)
            |-- cached[mm_hash].add(req_id)
            |-- num_free_slots -= num_encoder_embeds
            |-- num_freeable_slots -= num_encoder_embeds

    请求完成:
        free(req)
            |-- for input_id in cached_input_ids:
            |--     free_encoder_input(req, input_id)
            |--         cached[mm_hash].discard(req_id)
            |--         if cached[mm_hash] 为空:
            |--             freeable[mm_hash] = num_embeds
            |--             num_freeable_slots += num_embeds

    Step 结束:
        get_freed_mm_hashes()
            |-- 返回 freed 列表并清空
            |-- SchedulerOutput.free_encoder_mm_hashes 传给 Worker
            |-- Worker: for hash in freed: encoder_cache.pop(hash)
```

### 3.3 Hash 计算

`MultiModalHasher` (hasher.py:50):

```
算法选择: blake3 (默认) / sha256 / sha512 (FIPS 兼容)

hash_kwargs(**kwargs):
    hasher = blake3()
    for key, value in sorted(kwargs.items()):
        for bytes_ in iter_item_to_bytes(key, value):
            hasher.update(bytes_)
    return hasher.hexdigest()

序列化规则:
    PIL Image  -> EXIF UUID 或 {mode, data as numpy}
    Tensor     -> numpy (bfloat16 特殊处理: view as uint8)
    numpy      -> {dtype, shape, raw bytes}
    str/int/float -> encoded bytes
    其他       -> pickle (fallback)
```

跨请求缓存: 相同的图片 (相同 bytes) 在不同请求中共享 encoder 输出。
`mm_hash` 在 P0 进程计算, `identifier` 在 P1 进程添加 LoRA 前缀。

---

## 4. Input Embedding Composition: Text 与 Image 的交织

### 4.1 embed_input_ids 流程

`SupportsMultiModal.embed_input_ids()` (interfaces.py:374-409):

```
输入: input_ids [1, 3148, 32000, ..., 32000, 13, 5618, ...]
      mm_embeds [Tensor(576, 4096)]  (vision encoder 输出)
      is_multimodal [F, F, T, ..., T, F, F, ...]  (bool mask)

步骤:
    (1) _embed_text_input_ids(input_ids, embed_fn, is_multimodal)
        |-- 如果 _has_oov_mm_tokens:
        |     in_vocab_ids = input_ids.masked_fill(is_multimodal, 0)
        |     text_embeds = embed_fn(in_vocab_ids)  # 跳过 OOV token
        |-- 否则:
              text_embeds = embed_fn(input_ids)  # 正常 embedding

    (2) _merge_multimodal_embeddings(text_embeds, mm_embeds, is_multimodal)
        |-- mm_embeds_flat = flatten(mm_embeds)
        |-- inputs_embeds[is_multimodal] = mm_embeds_flat
        |-- (in-place 操作)

输出: inputs_embeds  (text + vision 混合 embeddings)
```

### 4.2 _merge_multimodal_embeddings

`utils.py:456`:

```python
def _merge_multimodal_embeddings(
    inputs_embeds: Tensor,      # (total_tokens, hidden_size)
    multimodal_embeddings: NestedTensors,  # list of tensors
    is_multimodal: Tensor,      # (total_tokens,) bool
) -> Tensor:
    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
    return inputs_embeds  # in-place
```

关键: `is_multimodal` mask 是一个 boolean tensor, 标记哪些位置
需要用 vision embedding 替换 text embedding。这个 mask 在
`_gather_mm_embeddings()` 中构建。

### 4.3 _gather_mm_embeddings 详解

`gpu_model_runner.py:3081-3177`:

```
对每个请求:
    for req_id in input_batch.req_ids:
        mm_features = req_state.mm_features
        lo, hi = get_mm_features_in_window(mm_features, start, end)

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length

            # 计算 chunked prefill 中的 embedding 范围
            start_idx = max(num_computed_tokens - start_pos, 0)
            end_idx = min(num_computed_tokens + num_scheduled - start_pos, num_tokens)

            # 支持 is_embed mask (部分位置不需要 embedding)
            if is_embed is not None:
                curr_start, curr_end = pos_info.get_embeds_indices_in_range(...)
                mm_embeds_item = encoder_output[curr_start:curr_end]
            else:
                mm_embeds_item = encoder_output[start_idx:end_idx]

            # 更新 is_mm_embed mask
            is_mm_embed[req_start + start_idx : req_start + end_idx] = True

        # Multimodal pruning (如果启用)
        if is_multimodal_pruning_enabled and uses_mrope:
            mm_embeds_req = model.recompute_mrope_positions(...)
```

### 4.4 _preprocess 中的分支处理

`gpu_model_runner.py:3407-3520`:

```
_preprocess() 分三条路径:

路径 A: Multimodal 模型 (supports_mm_inputs && is_first_rank)
    (1) _execute_mm_encoder() -> encoder 输出缓存
    (2) _gather_mm_embeddings() -> mm_embeds + is_mm_embed
    (3) model.embed_input_ids(input_ids, mm_embeds, is_multimodal)
        -> inputs_embeds_scheduled (text + vision 混合)
    (4) copy to self.inputs_embeds.gpu
    (5) input_ids = None, inputs_embeds = GPU tensor
    (6) model_kwargs 包含 MM 相关参数

路径 B: prompt_embeds (enable_prompt_embeds && is_first_rank)
    类似路径 A 但更简单, 只处理预计算的 embeddings

路径 C: 纯文本模型
    input_ids = self.input_ids.gpu  (token IDs)
    inputs_embeds = None
    直接用 token IDs + CUDA Graph (更快)
```

---

## 5. Pipeline Parallelism 处理

### 5.1 PP 只在第一个 Rank 的原因

`gpu_model_runner.py:3428`:

```python
if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
    # Run the multimodal encoder if any.
    self._execute_mm_encoder(scheduler_output)
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
    inputs_embeds_scheduled = self.model.embed_input_ids(
        self.input_ids.gpu[:num_scheduled_tokens],
        multimodal_embeddings=mm_embeds,
        is_multimodal=is_mm_embed,
    )
```

原因分析:

```
PP Stage 0 (is_first_rank = True):
    |-- 执行 vision encoder
    |-- 合并 text + vision embeddings -> inputs_embeds
    |-- LLM forward (前几层) -> intermediate_tensors
    |-- 通过 P2P 通信传给 Stage 1

PP Stage 1+ (is_first_rank = False):
    |-- 接收 intermediate_tensors (sync_and_gather_intermediate_tensors)
    |-- LLM forward (后几层)
    |-- 不需要 vision encoder, 因为 Stage 0 已经把
    |   vision embeddings 编码进了 intermediate_tensors
```

为什么不需要重复执行:
1. Vision encoder 的输出在 Stage 0 被合并到 inputs_embeds
2. inputs_embeds 经过 embedding layer + 几层 Transformer 后变成 intermediate_tensors
3. 后续 Stage 收到的 intermediate_tensors 已经包含了 vision 信息
4. 这与 text-only PP 完全一致, 无需特殊处理

### 5.2 Encoder-Decoder 模型的差异

`gpu_model_runner.py:3506-3513`:

```python
if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
    encoder_outputs = self._execute_mm_encoder(scheduler_output)
    model_kwargs.update({"encoder_outputs": encoder_outputs})
```

Encoder-Decoder (如 Whisper) 与 MM (如 LLaVA) 的区别:
- MM 模型: encoder 输出被 scatter 到 input embeddings 中
- Enc-Dec 模型: encoder 输出作为 cross-attention 的 K/V 传给 decoder

---

## 6. Performance Implications

### 6.1 Vision Encoder 开销

Vision encoder 是 compute-bound 操作:

```
典型 LLaVA (CLIP-ViT-L/14):
    输入: pixel_values (1, 3, 336, 336)
    Vision Tower: CLIPViTTransformer
        24 层 Transformer, hidden_size=1024, num_heads=16
        输出: (1, 576, 1024)  # 24x24 patches
    Projector: Linear(1024, 4096)
        输出: (1, 576, 4096)  # 576 个 vision tokens

    计算量: ~4.4 GFLOPS (ViT-L on 336x336)
    约等于 8B LLM prefill ~500 tokens 的计算量

Video Encoder (如 Qwen2-VL):
    输入: (num_frames, 3, H, W)  # 可能有 30+ 帧
    计算量线性增长: 30 帧 = 30x 图像计算量
    需要 sequential video encoding 限制峰值显存 (gpu_model_runner.py:3013)
```

### 6.2 性能优化: Encoder CUDA Graph

`EncoderCudaGraphManager` 通过消除 kernel launch 开销加速 encoder:

```
正常执行: Python overhead + kernel launch (~10us per op)
CUDA Graph: 一次 replay (~数 us), 所有 op 一起执行

Budget-based graph 选择:
    token_budgets = [256, 512, 1024, 2048, ...]
    每个 budget 预捕获一个 graph
    运行时选最小能容纳的 graph

限制:
    只支持 SupportsEncoderCudaGraph 协议的模型
    动态分辨率的视频可能不适合
```

### 6.3 调度器中的 Encoder Budget

`encoder_cache_manager.py:269-316`:

```python
def compute_mm_encoder_budget(scheduler_config, mm_max_toks_per_item):
    encoder_compute_budget = max(
        scheduler_config.max_num_encoder_input_tokens,
        max_tokens_per_mm_item,  # 保证至少能处理单个 item
    )
    encoder_cache_size = max(
        scheduler_config.encoder_cache_size,
        max_tokens_per_mm_item,
    )
    return encoder_compute_budget, encoder_cache_size
```

```
调度限制:
    encoder_compute_budget: 每 step 可执行的 encoder embedding 数
    encoder_cache_size: GPU 缓存可容纳的 embedding 总数

影响:
    如果 encoder budget 耗尽, 请求的 num_new_tokens 被截断
    只调度到 unschedulable encoder input 之前的 tokens
    (scheduler.py:1236-1248)
```

### 6.4 Multimodal Pruning (EVS)

`evs.py` (Efficient Video Streaming):

```python
def compute_retained_tokens_count(tokens_per_frame, num_frames, q):
    total = tokens_per_frame * num_frames
    retained = int(total * (1 - q))  # q = pruning rate
    return max(tokens_per_frame, retained)  # 至少保留 1 帧
```

通过 attention score 丢弃不重要的 vision tokens, 减少 LLM 需要处理的序列长度。
启用条件: `supports_multimodal_pruning` + `mm_config.is_multimodal_pruning_enabled()`。

---

## 7. 最新特性

### 7.1 Multi-Image 支持

多个图像通过 `MultiModalKwargsItems` 自然支持:

```python
# inputs.py:882-916
class MultiModalKwargsItems(UserDict[str, Sequence[MultiModalKwargsItem]]):
    # 结构: {"image": [item_0, item_1, ...], "audio": [item_0, ...]}
    # 每个 item 有独立的 placeholder range
    # Prompt: "Compare <image_0> and <image_1>"
    #   -> PlaceholderRange(offset=5, length=576)   # image_0
    #   -> PlaceholderRange(offset=582, length=576)  # image_1
```

`PlaceholderRange.is_embed` 支持细粒度控制:
```python
# 对于某些模型, placeholder 中只有部分位置是 vision embedding
# is_embed = [False, True, ..., True, False]  # bos/eos 不需要 embed
# get_embeds_indices_in_range() 计算正确的 embedding 范围
```

### 7.2 Video 支持

Video 作为独立的 modality:

```python
# scheduler.py:3013-3041: Sequential video encoding
if (is_multimodal_pruning_enabled or requires_sequential_video_encoding) \
   and modality == "video" and num_items > 1:
    for video_idx in range(num_items):
        micro_batch_outputs = model.embed_multimodal(**video_mm_inputs)
        batch_outputs_lst.extend(micro_batch_outputs)
```

限制: 视频 encoding 逐个处理, 避免显存爆炸。
`max_frames_per_batch` 控制每批最大帧数。

### 7.3 Audio 支持

Audio 通过 `HfAudioItem` (inputs.py:52) 接受输入:

```python
HfAudioItem = Union[list[float], np.ndarray, torch.Tensor]
# 也可以传入 (audio, sampling_rate) 元组做重采样
```

模型如 `UltravoxModel`, `WhisperForConditionalGeneration` 支持 audio。
Audio 和 video 可同时存在 (Qwen2-Audio 的 `use_audio_in_video` 模式)。

### 7.4 Prompt Embeds 透传

`prompt_embeds` modality (gpu_model_runner.py:2885-2905):

```python
pe_indices = [i for i, (mod, _) in enumerate(mm_kwargs) if mod == "prompt_embeds"]
if pe_indices:
    for i in pe_indices:
        pe_tensor = mm_kwargs[i][1]["embedding"].data
        encoder_cache[mm_hashes[i]] = pe_tensor.to(self.device)
    # 跳过 encoder 执行, 直接缓存
```

允许用户直接传入 pre-computed embeddings, 跳过 vision encoder。

### 7.5 EC Connector (P/D 分离中的 Encoder 缓存)

`ec_connector_model_runner_mixin.py` (25-65):

```python
class ECConnectorModelRunnerMixin:
    @staticmethod
    def maybe_save_ec_to_connector(encoder_cache, mm_hash):
        if has_ec_transfer():
            connector = get_ec_transfer()
            connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)
```

在 Prefill/Decode 分离架构中:
- Prefill 实例计算 encoder 输出并缓存
- 通过 EC Connector (如 NIXL) 将 encoder 缓存传给 Decode 实例
- Decode 实例从远端加载 encoder 输出

### 7.6 Processor Cache (三层缓存)

`cache.py:175-198` 的双进程缓存模型:

```
P0 (API Server):
    is_cached(mm_hash) -> bool  # 不更新驱逐顺序
    get_and_update_item(mm_item, mm_hash) -> processed_item

P1 (Engine Core):
    get_and_update_item(mm_item, mm_hash) -> processed_item

缓存类型:
    None:          禁用
    processor_only: 仅 P0 缓存 (不需要 IPC)
    lru:           P0 + P1 各自 LRU 缓存
    shm:           共享内存 (SingleWriterShmObjectStorage)
```

### 7.7 MM LoRA 支持

`gpu_model_runner.py:2922-2989`:

```
MM 模型支持 LoRA 的 tower 和 connector 两层:
    Tower LoRA:     应用在 vision encoder 上
    Connector LoRA: 应用在 projector 上

映射构建:
    encoder_token_counts = [model.get_num_mm_encoder_tokens(pos_info) ...]
    token_lora_mapping = [lora_id] * num_tokens  # per-token 映射
    tower_mapping = LoRAMapping(token_lora_mapping, prompt_lora_mapping, type=TOWER)

    post_op_counts = [model.get_num_mm_connector_tokens(n) ...]
    connector_mapping = LoRAMapping(connector_token_mapping, ..., type=CONNECTOR)
```

### 7.8 Raw Input Only 模型

某些模型 (如转录模型) 直接处理原始输入, 不走 embedding 路径:

```python
# gpu_model_runner.py:452
self.is_multimodal_raw_input_only_model = model_config.is_multimodal_raw_input_only_model

# gpu_model_runner.py:1623
if not self.is_multimodal_raw_input_only_model:
    return {}
# 直接传递 mm_kwargs 给 model forward, 不做 embedding 替换
```

---

## 8. 架构总图

```
+-------------------+     +---------------------------+
|   API Server (P0) |     |   Engine Core (P1)        |
|                   |     |                           |
| MediaConnector    |     | Scheduler                 |
|   |               |     |   |-- _try_schedule_encoder_inputs()
|   v               |     |   |-- encoder_cache_manager.check_and_update()
| MMProcessor       |     |   |-- can_allocate() / allocate()
|   |-- HF Proc     |     |   |                       |
|   |-- PromptRepl  |     |   v                       |
|   |-- mm_hashes   |     | SchedulerOutput           |
|   |               |     |   |-- scheduled_encoder_inputs
| ProcessorCache    |     |   |-- free_encoder_mm_hashes
|   (LRU/SHM)       |     |                           |
+-------------------+     +---|---------|---------------+
                              |         |
                              v         v
+-----------------------------------------------------+
|  GPU Worker (per PP rank)                            |
|                                                      |
|  GPUModelRunner                                      |
|    |-- encoder_cache: dict[str, Tensor]  (GPU)       |
|    |-- encoder_cudagraph_manager  (optional)         |
|                                                      |
|  _preprocess():                                      |
|    if is_first_rank && supports_mm:                  |
|      [1] _execute_mm_encoder(scheduler_output)       |
|          |-- _batch_mm_inputs_from_scheduler()       |
|          |-- group_and_batch_mm_kwargs()              |
|          |-- model.embed_multimodal(**batch_kwargs)  |
|          |-- encoder_cache[mm_hash] = output         |
|      [2] _gather_mm_embeddings(scheduler_output)     |
|          |-- 从 encoder_cache 取出 embeddings         |
|          |-- 构建 is_mm_embed mask                    |
|      [3] model.embed_input_ids(ids, mm, is_mm)      |
|          |-- text_embeds = embed(input_ids)           |
|          |-- text_embeds[is_mm] = mm_embeds           |
|    else:                                             |
|      input_ids = self.input_ids.gpu  (纯文本)        |
|                                                      |
|  model.forward(input_ids/inputs_embeds, ...)         |
|                                                      |
+-----------------------------------------------------+
```

---

## 9. 关键文件索引

| 文件 | 行数 | 说明 |
|------|------|------|
| `vllm/v1/worker/gpu_model_runner.py:2870` | `_execute_mm_encoder()` | Encoder 执行主入口 |
| `vllm/v1/worker/gpu_model_runner.py:3081` | `_gather_mm_embeddings()` | Embedding 收集与 scatter |
| `vllm/v1/worker/gpu_model_runner.py:3407` | `_preprocess()` | 三条路径分支 |
| `vllm/v1/worker/gpu_model_runner.py:530` | `encoder_cache` | GPU 端 encoder 输出缓存 |
| `vllm/v1/worker/encoder_cudagraph.py` | `EncoderCudaGraphManager` | Encoder CUDA Graph |
| `vllm/v1/worker/encoder_cudagraph_defs.py` | `EncoderCudaGraphConfig` | CUDA Graph 配置 |
| `vllm/v1/worker/ec_connector_model_runner_mixin.py` | EC Connector | P/D 分离 encoder 缓存 |
| `vllm/v1/core/encoder_cache_manager.py` | `EncoderCacheManager` | 调度器端缓存管理 |
| `vllm/v1/core/sched/scheduler.py:1126` | `_try_schedule_encoder_inputs()` | 调度器 encoder 调度 |
| `vllm/multimodal/registry.py` | `MultiModalRegistry` | MM Registry (117+ 模型) |
| `vllm/multimodal/inputs.py` | `PlaceholderRange` | 位置信息核心类型 |
| `vllm/multimodal/hasher.py` | `MultiModalHasher` | blake3 hash 计算 |
| `vllm/multimodal/cache.py` | 三层 processor cache | Processor/Receiver/SHM |
| `vllm/multimodal/processing/processor.py:1663` | `apply()` | MM 处理主流程 |
| `vllm/multimodal/evs.py` | EVS pruning | 视频 token 裁剪 |
| `vllm/multimodal/encoder_budget.py` | `MultiModalBudget` | Encoder 预算计算 |
| `vllm/multimodal/media/connector.py` | `MediaConnector` | 媒体加载 |
| `vllm/model_executor/models/interfaces.py:95` | `SupportsMultiModal` | MM 模型协议 |
| `vllm/model_executor/models/interfaces.py:374` | `embed_input_ids()` | Embedding 合并 |
| `vllm/model_executor/models/llava.py:661` | LLaVA `embed_multimodal()` | 典型 MM 模型示例 |
| `vllm/model_executor/models/utils.py:456` | `_merge_multimodal_embeddings()` | In-place scatter |

---

## 10. 关键设计洞察

1. **双进程缓存协同**: P0 (processor cache) 和 P1 (encoder cache) 通过
   相同的 hash 和驱逐顺序保持一致, 避免跨进程通信

2. **Chunked Prefill 兼容**: `_gather_mm_embeddings` 的 `start_idx/end_idx`
   计算支持 prefill 分块, 只取当前 chunk 需要的 embedding 部分

3. **is_embed mask 的灵活性**: 不是所有 placeholder 位置都需要 vision embedding,
   `<bos>` / `<eos>` 等 token 仍使用 text embedding

4. **Encoder CUDA Graph 的 budget 模式**: 与 LLM CUDA Graph 不同,
   encoder graph 按 token budget 分类, 运行时选最小合适的 graph

5. **Modality 分组批处理**: `group_and_batch_mm_kwargs()` 将同一 modality
   的 items batch 在一起, 但不同 modality 分开处理以保持顺序

6. **Prompt Embeds 透传**: `prompt_embeds` modality 完全跳过 encoder,
   直接注入缓存, 支持外部预计算的 embeddings

7. **调度器感知 Encoder**: 调度器在 `_try_schedule_encoder_inputs()` 中
   考虑 encoder budget 和 cache 容量, 不够时截断 token 调度

8. **Sequential Video Encoding**: 视频 encoding 逐个处理,
   避免 scheduler 放入过多视频样本导致 OOM
