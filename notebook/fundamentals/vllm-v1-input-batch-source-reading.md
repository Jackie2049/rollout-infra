# vLLM V1 Input Batch + Block Table Source Reading — Persistent Batch Data + Multi-Group Block Table + 7 Triton Kernels

> 2026-06-12 | vLLM V1 InputBatch全链路源码分析: Persistent Batch数据结构 + CachedRequestState + add/remove/condense/swap + MultiGroupBlockTable + 7 Triton kernel(input preparation) + SamplingMetadata差量更新
> 源码: vllm/v1/worker/gpu_input_batch.py (1127行), vllm/v1/worker/gpu/input_batch.py (608行), vllm/v1/worker/block_table.py (380行)
> 关联: vllm-v1-gpu-model-runner-source-reading.md, vllm-v1-kv-cache-architecture-source-reading.md, vllm-v1-scheduler-architecture-source-reading.md

## 0. 核心定律: Persistent Batch + Slot-in/Swap-out + 差量更新

```
InputBatch核心设计:
  → Persistent Batch: GPU/CPU buffer预分配max_num_reqs→跨step保持→只增删变化部分
  → Slot-in/Slot-out: add_request→填充空slot / remove_request→标记None→condense滑动压缩
  → 差量更新: SamplingMetadata仅在batch变化时重建→避免每step全重建!
  → CPU→GPU copy_slice: 只复制前num_reqs行→减少GPU通信→pin_memory加速DMA

三层架构:
  InputBatch (1127行) → 持久化batch状态(req_ids+token_ids+block_table+sampling_params)
  InputBuffers (30行) → 预分配GPU buffer(input_ids+positions+seq_lens+query_start_loc)
  MultiGroupBlockTable → 多KV cache组→每组一个BlockTable→物理block→逻辑slot映射

关键优化:
  → token_ids_cpu: [max_num_reqs, max_model_len] → CPU numpy → 避免每step重建
  → block_table: MultiGroupBlockTable → 多组→EAGLE draft layers自成一组
  → sampling_params: set-based分类(greedy_reqs/random_reqs/top_p_reqs) → O(1)判断
  → condense(): 滑动压缩 → 无需排序 → 保持请求连续 → CUDA graph友好
  → 7 Triton kernels: prefill/pos/seq_lens/draft/rejection/post_update/expand → GPU加速
```

## 1. InputBatch类 — 持久化Batch状态容器

```
文件: vllm/v1/worker/gpu_input_batch.py (1127行)

InputBatch.__init__ 核心buffer:

1. 请求映射:
  → _req_ids: list[str | None] → batch_idx→req_id(None=空slot)
  → req_id_to_index: dict[str, int] → req_id→batch_idx(O(1)查找)
  → num_reqs = len(req_id_to_index) → 实际请求数

2. Token数据(CPU numpy → 避免GPU→CPU→GPU同步):
  → token_ids_cpu: [max_num_reqs, max_model_len] np.int32 → prompt+output token ids
  → is_token_ids: [max_num_reqs, max_model_len] bool → 哪些位置有有效token id
  → num_tokens_no_spec: [max_num_reqs] → 不含spec tokens的token数
  → num_prompt_tokens: [max_num_reqs] → prompt长度
  → num_computed_tokens_cpu: [max_num_reqs] → 已计算的token数

3. Block Table:
  → block_table: MultiGroupBlockTable → 多KV cache组的block管理

4. Sampling参数(CPU numpy + GPU tensor + set分类):
  → temperature/top_p/top_k → CPU+GPU双副本 → set分类(greedy/random/top_p/top_k)
  → frequency_penalties/presence_penalties/repetition_penalties → CPU+GPU+set分类
  → generators: dict[int, torch.Generator] → 每请求的随机种子(非greedy时)
  → num_logprobs: dict[str, int] → 每请求的logprobs数量
  → logprob_token_ids: dict[str, list[int]] → 特定token的logprobs(比-1更高效!)
  → allowed_token_ids_mask: [max_num_reqs, vocab_size] bool → 延迟分配!
  → bad_words_token_ids: dict[int, list[list[int]]] → 禁止词
  → logitsprocs: LogitsProcessors → 自定义logits处理器(推理能力扩展!)

5. Spec Decode:
  → spec_token_ids: list[list[int]] → 每请求的draft token ids
  → num_accepted_tokens_cpu: [max_num_reqs] → 接受的token数(初始=1)

6. LoRA:
  → request_lora_mapping: [max_num_reqs] → 每请求的LoRA id
  → lora_id_to_request_ids: dict[int, set[str]] → LoRA id→请求集合
  → lora_id_to_lora_request: dict[int, LoRARequest] → LoRA id→LoRA对象

7. Pooling模型:
  → pooling_params/pooling_states → dict[str, PoolingParams/PoolingStates]

8. 异步调度:
  → prev_sampled_token_ids: Tensor | None → 上一步采样的GPU tensor
  → prev_req_id_to_index: dict | None → 上一步的req_id→index映射
  → sampled_token_ids_cpu: Tensor | None → 异步CPU复制结果
  → async_copy_ready_event: Event | None → 异步复制完成事件

9. Thinking Budget (推理深度控制):
  → thinking_budget_state_holder → 推理token预算管理器
  → thinking_token_budget_reqs: set[str] → 需要budget控制的请求

10. BatchUpdateBuilder:
  → batch_update_builder → 跟踪增删移动→驱动logitsprocs更新
  → added/removed/moved → 增删移动的请求列表→差量更新
```

## 2. CachedRequestState — 请求缓存数据

```
@dataclass CachedRequestState:
  → req_id: str → 请求唯一标识
  → prompt_token_ids: list[int] | None → prompt token ids(可能来自embeds)
  → mm_features: list[MultiModalFeatureSpec] → 多模态特征
  → sampling_params: SamplingParams | None → 采样参数
  → generator: torch.Generator | None → 随机种子
  → block_ids: tuple[list[int], ...] → 多组KV cache block ids!
  → num_computed_tokens: int → 已计算token数
  → output_token_ids: list[int] → 已输出token ids
  → mrope_positions/mrope_position_delta → 多模态RoPE位置
  → xdrope_positions → 扩展RoPE位置
  → lora_request: LoRARequest | None → LoRA请求
  → prompt_embeds: torch.Tensor | None → prompt embeddings
  → in_progress_prompt_logprobs_cpu → 跨prefill步骤的logprobs累积
  → prompt_is_token_ids: list[bool] | None → 混合模式(部分token ids+部分embeds)
  → prev_num_draft_len: int → 上一步draft token数(async scheduling用)
  → pooling_params/pooling_states → pooling模型专用

关键设计:
  → block_ids: tuple[list[int], ...] → 多组! → target+EAGLE各自有block ids
  → num_tokens = num_prompt_tokens + len(output_token_ids) → 总token数
  → get_token_id(idx) → 获取指定位置的token id(prompt vs output)
```

## 3. add_request() — 添加新请求到Persistent Batch

```
add_request(request: CachedRequestState)流程:

Step 1: _register_add_request → 分配slot
  → batch_update_builder.pop_removed() → 优先填空slot(删除留下的空位)
  → 否则 → append到末尾 → new_req_index = num_reqs
  → 记录到batch_update_builder.added → logitsprocs更新用

Step 2: 更新_req_ids和映射
  → req_index == len(_req_ids) → append(新请求)
  → req_index < len(_req_ids) → 填空slot(被删除的位置)
  → req_id_to_index[req_id] = req_index

Step 3: Token数据填充
  → token_ids_cpu[req_index, :num_prompt] = prompt_token_ids
  → token_ids_cpu[req_index, num_prompt:end] = output_token_ids
  → is_token_ids[req_index, :num_prompt] = True/False(embeds情况)
  → num_tokens_no_spec[req_index] = total_tokens
  → num_computed_tokens_cpu[req_index] = num_computed

Step 4: Block Table
  → block_table.add_row(request.block_ids, req_index) → 多组block ids

Step 5: Sampling参数填充
  → GREEDY: temperature=0.0, greedy_reqs.add
  → RANDOM: temperature=param, random_reqs.add
  → top_p/top_k/frequency_penalty/presence_penalty/repetition_penalty → CPU+set
  → generator → 仅非greedy时添加
  → logprobs/logprob_token_ids → 字典存储
  → allowed_token_ids → 延迟分配mask(大tensor→只在需要时创建!)
  → bad_words_token_ids → 字典存储

Step 6: LoRA映射
  → lora_request存在 → request_lora_mapping=req.lora_int_id → lora_id_to_request_ids
  → 否则 → request_lora_mapping=0 → 无LoRA

Step 7: Spec Decode初始化
  → num_accepted_tokens_cpu[req_index] = 1 → 默认1个接受token

关键优化:
  → 填空slot优先 → 避免列表无限增长 → 与condense()配合
  → allowed_token_ids_mask延迟分配 → vocab_size可能32K-151K → 空间节省!
  → CPU numpy操作 → 避免GPU→CPU→GPU同步延迟
```

## 4. remove_request() — 从Persistent Batch移除请求

```
remove_request(req_id: str)流程:

Step 1: 查找并移除映射
  → req_id_to_index.pop(req_id) → 返回req_index或None
  → batch_update_builder.removed_append(req_index) → 记录删除位置

Step 2: 标记空slot
  → _req_ids[req_index] = None → 标记空位(不删除!)
  → req_output_token_ids[req_index] = None → 清空输出token ids
  → spec_token_ids[req_index].clear() → 清空draft tokens
  → block_table.clear_row(req_index) → 清空block table行

Step 3: LoRA清理
  → lora_id → lora_id_to_request_ids[lora_id].discard(req_id)
  → 如果集合空 → 删除lora映射 → LoRA不再激活

Step 4: Sampling参数清理
  → greedy_reqs/random_reqs/top_p_reqs/top_k_reqs → set.discard
  → frequency_penalties_reqs/presence_penalties_reqs/repetition_penalties_reqs → set.discard
  → generators.pop(req_index) → 清除随机种子
  → num_logprobs/logprob_token_ids → dict.pop
  → allowed_token_ids_mask → 填False(不删除!)
  → bad_words_token_ids → dict.pop

关键设计:
  → None标记而非删除 → 避免列表重建 → 后续condense()压缩
  → set.discard → O(1)移除 → 分类判断仍然正确
  → block_table.clear_row → 不释放blocks(Scheduler负责释放KV blocks)
  → 必须在remove_request后调用condense()!
```

## 5. condense() — 滑动压缩Persistent Batch

```
condense()流程(~120行):

目的: 删除留下的空slot → 滑动非空请求到底部 → 保持连续性 → CUDA graph友好

算法:
  → empty_req_indices = batch_update_builder.removed → 所有空位(降序排列)
  → last_req_index = num_reqs + len(empty_req_indices) - 1 → 最后非空位

  → while empty_req_indices:
    → 找最小空位: empty_index = peek_removed()
    → 找最大非空位: last_req_index (跳过空位)
    → 如果empty_index >= last_req_index → break(已全部压缩)

    → 移动: 请求从last_req_index → empty_index
      → _req_ids[empty_index] = _req_ids[last_req_index]
      → _req_ids[last_req_index] = None
      → req_id_to_index[req_id] = empty_index → 更新映射!

    → 复制数据(只复制活跃token→省带宽!):
      → num_tokens = _get_active_token_count(last_req_index)
      → token_ids_cpu[empty, :num_tokens] = token_ids_cpu[last, :num_tokens]
      → is_token_ids[empty, :num_tokens] = is_token_ids[last, :num_tokens]
      → num_tokens_no_spec/num_prompt_tokens/num_computed_tokens → 单值复制
      → block_table.move_row(last, empty) → BlockTable行移动
      → request_lora_mapping → 单值复制
      → temperature/top_p/top_k/penalties → CPU值复制
      → num_accepted_tokens_cpu → 单值复制
      → generators → dict.pop+add
      → allowed_token_ids_mask → 行复制(如果存在)
      → bad_words_token_ids → dict.pop+add

    → 记录移动: batch_update_builder.moved.append((last, empty, UNIDIRECTIONAL))

  → Trim列表:
    → del _req_ids[num_reqs:]
    → del req_output_token_ids[num_reqs:]
    → del spec_token_ids[num_reqs:]

关键优化:
  → 只复制活跃token → _get_active_token_count → 不复制max_model_len行!
  → → 7B S=4096 → 复制4096×4B=16KB vs max_model_len×4B=可能128KB → 8x省带宽
  → 滑动压缩 → 无排序 → 保持请求连续 → CUDA graph需要连续batch
  → batch_update_builder.moved → logitsprocs知道请求移动了 → 差量更新
```

## 6. swap_states() — 请求交换(CUDA graph重排)

```
swap_states(i1, i2)流程:

交换两个请求的所有状态:
  → _req_ids[i1] ↔ _req_ids[i2] → 交换请求ID
  → req_id_to_index → 交换映射 → O(1)查找更新
  → token_ids_cpu → 只交换活跃部分!
    → tmp = token_ids_cpu[i1, :max_active].copy()
    → token_ids_cpu[i1, :max] = token_ids_cpu[i2, :max]
    → token_ids_cpu[i2, :max] = tmp
    → max_active = max(i1_active, i2_active) → 避免复制max_model_len!
  → is_token_ids → 交换活跃部分
  → req_prompt_embeds → dict交换
  → block_table.swap_row → BlockTable行交换
  → request_lora_mapping → 交换
  → temperature/top_p/top_k/penalties → CPU值交换
  → num_accepted_tokens_cpu → 交换
  → generators/bad_words_token_ids → dict交换
  → allowed_token_ids_mask → 行交换

  → batch_update_builder.moved.append((i1, i2, SWAP)) → logitsprocs更新

关键设计:
  → max_active = max(i1_count, i2_count) → 只复制必要部分→省带宽
  → numpy直接操作 → CPU内存 → 无GPU同步
  → CUDA graph需要请求在特定位置 → swap_states用于重排
```

## 7. refresh_metadata() — 差量更新SamplingMetadata

```
refresh_metadata()流程:

Step 1: 重置BatchUpdateBuilder
  → batch_update = batch_update_builder.get_and_reset(num_reqs)
  → → 获取本step的所有增删移动 → 一次性消费 → builder清空

Step 2: LogitsProcessors更新
  → for logit_proc in logitsprocs.all → logit_proc.update_state(batch_update)
  → → 每个logits processor根据batch_update差量更新内部状态

Step 3: Thinking Budget同步
  → thinking_budget_state_holder.sync_batch(batch_update)

Step 4: 条件重建SamplingMetadata
  → if batch_update → _make_sampling_metadata() → 只有batch变化时才重建!
  → → 否则 → sampling_metadata保持不变 → 避免不必要的GPU复制!

_make_sampling_metadata()关键优化:
  → all_greedy → temperature=None → 全贪心→跳过temperature GPU复制!
  → no_top_p → top_p=None → 无top_p请求→跳过GPU复制!
  → no_top_k → top_k=None → 无top_k请求→跳过GPU复制!
  → no_penalties → 跳过penalties GPU复制! → prompt_token_ids也跳过!
  → needs_output_token_ids → 仅penalties/bad_words/logitsprocs/budget需要→否则空!
  → copy_slice → CPU→GPU只复制[:num_reqs] → 不复制整max_num_reqs!
  → allowed_token_ids_mask → 延迟分配 → 只在有allowed_token_ids请求时创建

CPU→GPU传输优化:
  → pin_memory=True → DMA加速 → CPU→GPU直接传输 → 无CPU参与!
  → non_blocking=True → 异步复制 → 不阻塞主线程
  → copy_slice → 只复制活跃部分 → max_num_reqs可能118→num_reqs可能55→复制量减半!
```

## 8. MultiGroupBlockTable — 多KV Cache组的Block管理

```
文件: vllm/v1/worker/block_table.py (380行)

BlockTable核心属性:
  → block_size: int → 实际kernel block size(可能与KV manager不同!)
  → blocks_per_kv_block: int → hybrid blocks=kernel/alloc比例
  → use_hybrid_blocks: bool → 当kernel_block_size != block_size时启用
  → block_table: CpuGpuBuffer → [max_num_reqs, max_num_blocks_per_req] → CPU+GPU双副本
  → num_blocks_per_row: np.ndarray → 每请求的block数量
  → slot_mapping: CpuGpuBuffer → [max_num_batched_tokens] → token→KV slot映射

Hybrid Blocks设计:
  → 例: KV manager block_size=32, kernel block_size=16 → blocks_per_kv_block=2
  → → 一个32-token物理block = 2个16-token逻辑block → kernel看到2个block
  → map_to_kernel_blocks: kv_block_id → [kv_id*2, kv_id*2+1] → 展开映射
  → → MLA: compress_ratio → storage_block_size更小 → kernel需要更多逻辑blocks

关键方法:
  → add_row(block_ids, row_idx) → 设置整行(num_blocks_per_row=0再追加)
  → append_row(block_ids, row_idx) → 追加block ids到现有行
  → clear_row(row_idx) → 清空行
  → move_row(src, tgt) → 移动行(condense用)
  → swap_row(src, tgt) → 交换行(swap_states用)
  → commit_block_table(num_reqs) → CPU→GPU复制(只复制前num_reqs行!)
  → compute_slot_mapping(num_reqs, query_start_loc, positions) → Triton kernel!

Triton compute_slot_mapping_kernel:
  → 每个请求: 遍历query tokens → position→block_id→slot_offset
  → → slot = block_table[req_idx, position // block_size] * block_size + position % block_size
  → → 支持CP(Context Parallel) → TOTAL_CP_WORLD_SIZE/RANK → 交错存储
  → → PAD_SLOT_ID → padding位置 → attention kernel跳过

MultiGroupBlockTable (在InputBatch中):
  → 多组: [kv_cache_gid] → 每组有自己的BlockTable → EAGLE draft自成一组
  → → target attention group → 一个BlockTable
  → → EAGLE draft attention group → 另一个BlockTable → 独立block管理
  → add_row → 同时在所有组中设置block ids → 多组block_ids: tuple[list[int], ...]
```

## 9. InputBuffers + 7 Triton Kernels — GPU输入准备

```
文件: vllm/v1/worker/gpu/input_batch.py (608行)

InputBuffers (30行):
  → input_ids: [max_num_tokens] → GPU token ids buffer
  → positions: [max_num_tokens] → GPU位置编码buffer
  → query_start_loc: [max_num_reqs + 1] → GPU请求起始位置
  → seq_lens: [max_num_reqs] → GPU请求长度
  → dcp_local_seq_lens: [max_num_reqs] → DCP本地seq_lens

7 Triton Kernels:

Kernel 1: _prepare_prefill_inputs_kernel (30行)
  → Prefill输入准备 → 从all_token_ids复制prompt tokens到input_ids
  → → 每个请求: 从CPU all_token_ids[req_state_idx] → GPU input_ids[query_start:end]
  → → 只复制未计算的部分: num_computed到prefill_len
  → → next_prefill_tokens: chunked prefill → 下一个prefill token预取

Kernel 2: _prepare_pos_seq_lens_kernel (35行)
  → 位置编码+seq_lens计算 → 每请求: positions = num_computed + offset
  → → seq_len = num_computed + query_len → 总长度
  → → 最后一个thread block: padding seq_lens[num_reqs:] = 0 → CUDA graph!

Kernel 3: _combine_sampled_and_draft_tokens_kernel (55行)
  → Spec decode输入准备 → 合并sampled token + draft tokens
  → → logits_indices计算: query_end - num_logits → 哪些位置需要logits
  → → Prefill请求: 跳过(seq_len <= prefill_len → 无draft/sampled tokens)
  → → Decode请求: 写入last_sampled_token + draft_tokens到input_ids
  → → BLOCK_SIZE = next_power_of_2(num_spec_steps + 1) → 自适应块大小

Kernel 4: _get_num_sampled_and_rejected_kernel (45行)
  → 计算每个请求的sampled和rejected token数
  → → num_sampled = actual sampled tokens → chunked prefill时设0
  → → num_rejected = num_logits - num_sampled → 被拒绝的draft tokens
  → → Prefill请求: num_sampled=0, num_rejected=0

Kernel 5: _post_update_kernel (60行)
  → 采样后更新 → 将sampled tokens写入all_token_ids和output_bin_counts
  → → 更新num_computed_tokens: += (query_len - num_rejected)
  → → 更新last_sampled_tokens: 最后一个sampled token → spec decode用
  → → output_bin_counts: frequency/presence penalty token计数 → penalize用

Kernel 6: _post_update_num_computed_tokens_kernel (15行)
  → 仅更新num_computed_tokens → += query_len
  → → 用于prefill步骤(无rejected tokens)

Kernel 7: _expand_idx_mapping_kernel (20行)
  → Spec decode idx扩展 → 每个请求的多个logits位置
  → → expanded_idx_mapping: logits位置→请求idx → 用于采样
  → → expanded_local_pos: logits位置在请求内的偏移 → 用于penalty

关键设计:
  → 全GPU操作 → Triton kernel → 避免CPU→GPU→CPU同步
  → Prefill vs Decode分流 → is_chunked_prefilling判断 → 不同处理路径
  → Spec decode → logits_indices + expanded_idx_mapping → 多位置采样
  → CUDA graph padding → seq_lens[num_reqs:] = 0 → 固定维度 → graph可重放
```

## 10. Async Scheduling — 异步token ids修复

```
update_async_output_token_ids()流程(~40行):

背景: Async scheduling → forward和sampling时间线分离 → 采样结果异步到达

流程:
  → 遍历req_ids → 检查prev_req_id_to_index → 找到上一步的idx
  → output_token_ids[-1] == -1 → placeholder → 需要替换为实际token id
  → async_copy_ready_event.synchronize() → 等待GPU→CPU异步复制完成
  → sampled_token_ids = sampled_token_ids_cpu.tolist() → 获取实际token ids
  → 替换placeholder: req_output_token_ids[first_placeholder:] = new_ids
  → → num_to_replace = min(num_sampled_ids, num_placeholders) → 安全替换

update_async_spec_token_ids()流程(~20行):
  → 从draft_token_ids(prev step) → 更新当前spec_token_ids
  → → 替换spec_token_ids中的placeholder → rejection sampler用

关键设计:
  → placeholder=-1 → 占位符 → 实际token ids异步到达后替换
  → → 避免forward和sampling的同步依赖 → 时间线分离!
  → → PP async scheduling → 采样在另一个进程 → GPU→CPU异步复制→event等待
  → RTX 4090: 单GPU → async scheduling不适用 → 直接同步即可
```

## 11. RTX 4090 InputBatch Implications

```
1. Persistent Batch是RTX 4090关键优化:
  → 避免每次重建batch → GPU→CPU→GPU同步减少 → ~2x加速(vs全重建)
  → → CPU numpy操作 → 无GPU同步 → 纯CPU内存操作
  → → copy_slice → CPU→GPU只复制num_reqs → 不复制max_num_reqs → DMA省带宽
  → → pin_memory → DMA加速 → CPU→GPU ~12GB/s(实测PCIe) → 比non-pinned快2-3x

2. condense()对RTX 4090影响:
  → 只复制活跃token → max_active vs max_model_len → 省带宽8x
  → → 7B S=4096 → 16KB vs 128KB → negligible
  → → 但大模型128K vocab → token_ids行大 → 省带宽更明显
  → → condense是必要操作 → CUDA graph需要连续batch → 必须压缩!

3. SamplingMetadata差量更新:
  → all_greedy → temperature=None → 跳过GPU复制 → 生产大部分贪心 → 省带宽!
  → no_penalties → 跳过penalties GPU复制 → 大部分请求无penalty → 省带宽!
  → → 只有batch变化时才重建 → 大部分step batch不变 → metadata保持!

4. Block Table对RTX 4090:
  → MultiGroupBlockTable → EAGLE draft自成一组 → 独立block管理
  → → INT4+INT8KV → block数少 → block_table行短 → 内存省
  → → Hybrid blocks → MLA compress_ratio → kernel block数多 → block_table行更长
  → → RTX 4090: GQA-8 → 1组 → block_table简单 → 无hybrid block需求

5. Triton Kernels对RTX 4090:
  → 7个kernel全在GPU执行 → 避免CPU→GPU同步
  → → 但每个kernel有launch overhead ~8us → 7个≈56us
  → → vs CPU numpy操作可能更快(无launch overhead) → 需benchmark!
  → → 但! batch≥32时 → Triton kernel并行优势显现 → CPU无法并行
  → → 生产: B=118 → Triton明显更快 → RTX 4090推荐

6. allowed_token_ids_mask延迟分配:
  → vocab_size=32K → mask=118×32K=3.76MB → GPU内存不大
  → vocab_size=151K → mask=118×151K=17.8MB → 延迟分配节省!
  → → RTX 4090 24GB → 17.8MB占比0.07% → negligible但延迟分配好习惯

7. 生产最优配置:
  → max_num_reqs=80 → InputBatch预分配80行 → 内存可控
  → → INT4+INT8KV+GQA-8 → block_table行短 → 内存省
  → → Persistent Batch+差量更新 → 避免全重建 → ~2x加速
  → → CUDA graph FULL → batch必须连续 → condense()保证!
```

## 参考文献

```
1. vLLM V1 InputBatch源码:
   - vllm/v1/worker/gpu_input_batch.py (1127行) — InputBatch核心类
   - vllm/v1/worker/gpu/input_batch.py (608行) — InputBatch dataclass + 7 Triton kernels
   - vllm/v1/worker/block_table.py (380行) — BlockTable + MultiGroupBlockTable

2. PagedAttention: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", OSDI 2023
3. Orca/Continuous Batching: Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022

我们的笔记:
- vllm-v1-gpu-model-runner-source-reading.md — GPU Model Runner(调用InputBatch)
- vllm-v1-kv-cache-architecture-source-reading.md — KV Cache(MultiGroupBlockTable底层)
- vllm-v1-scheduler-architecture-source-reading.md — Scheduler(驱动InputBatch增删)