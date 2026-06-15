# DeepSpeed ZeRO-3 Universal Checkpoint + 推理部署 源码级深度阅读

> 2026-06-15 | 源码: deepspeed/utils/zero_to_fp32.py + deepspeed/checkpoint/ds_to_universal.py + deepspeed/runtime/bf16_optimizer.py + deepspeed/linear/optimized_linear.py
> 核心: zero_to_fp32→GatheredTensor延迟合并→per-param交错从flat_groups切片→ZeRO-3 FP32 FLAT_GROUPS vs ZeRO-2 SINGLE_PARTITION; ds_to_universal→3阶段(extract→7种TP merge→save); BF16无loss scaling+HP-LP链接+fp32_groups_flat_partition; LoRA=LoRAOptimizedLinear原生(非PEFT)+fuse/unfuse; 推理部署: ZeRO→FP32→HF→vLLM INT4→4,791tok/s; Universal→跨GPU数量加载→AutoEP专家合并

## 1. zero_to_fp32.py — ZeRO Shard → FP32合并核心

```
zero_to_fp32.py (761 lines):

★ ★ 入口: get_fp32_state_dict_from_zero_checkpoint() (533)
  → 读取latest文件获取tag → 分发ZeRO-2/ZeRO-3路径

★ ★ ZeRO-3合并: _zero3_merge_trainable_params() (437)

核心: GatheredTensor类 (391-434) — 延迟合并机制!
  → 不立即合并 → 只存metadata → .contiguous()时才合并 → 省内存!

GatheredTensor算法:
  1. 从所有rank的optim_states.pt读FP32_FLAT_GROUPS → 每rank一个FP32 flat shard列表
  2. 从model_states.pt读PARAM_SHAPES → 每参数原始(unpartitioned)形状
  3. flat_groups_offset = cumsum(flat_tensor.numel()) → 追踪flat vector偏移
  4. partitioned_numel = ceil(unpartitioned_numel / world_size) → 每rank分区大小
  5. GatheredTensor(fp32_flat_groups, flat_groups_offset, offset, partitioned_numel, shape)
  6. .contiguous() → 实际合并:
     → 每rank: 找偏移量跨哪些flat_groups → 切片正确片段
     → torch.cat(pad_flat_param_chunks, dim=0) → 拼接所有rank片段
     → [:shape.numel()].view(shape) →裁剪填充→reshape → 完整FP32参数!

★ ★ lazy_mode=True (537): 返回GatheredTensor伪tensor → 逐参数释放内存 → 大模型关键!
  → to_torch_tensor() (513): tensor_id追踪 → 正确恢复共享参数(tied weights)
  → ★ vs 直接合并: 所有参数同时materialize → CPU OOM → lazy_mode逐参数 → 安全!

★ ★ ZeRO-2合并: _zero2_merge_trainable_params() (252) — 更简单!
  1. 每参数组: 从所有rank收集shards → merged_partitions
  2. torch.cat(merged_partitions, 0) → 拼接完整vector
  3. .narrow(0, offset, unpartitioned_numel).view(shape) → 逐参数切片
  4. NCCL对齐: align_to = 2 * world_size → 偏移必须2*world_size对齐

★ ZeRO-3 vs ZeRO-2 checkpoint键名差异:
  → ZeRO-2: SINGLE_PARTITION_OF_FP32_GROUPS + PARAM_SLICE_MAPPINGS
  → ZeRO-3: FP32_FLAT_GROUPS + PARAM_SHAPES(in model_states.pt)
  → ★ 关键: ZeRO-3参数形状在model_states.pt中 → 不在optim_states.pt!

★ ★ safetensors输出: convert_zero_checkpoint_to_fp32_state_dict() (598)
  → --safe_serialization → model.safetensors代替pytorch_model.bin
  → --max_shard_size 5GB → 分shard输出 → 大模型安全!
  → split_torch_state_dict_into_shards() (643) → 自动分shard
  → ★ safetensors = vLLM/SGLang推荐格式 → 必须用!
```

## 2. Universal Checkpoint Format — 3阶段转换

```
ds_to_universal.py — 3阶段转换:

★ ★ Phase 1: _extract_zero_shard_files() (363)
  → 从每(PP,TP,DP)rank的optim shard中提取每参数片段
  → 写临时文件: {temp_dir}/{param_name}/{tp_index}/{state_name}.{dp_index}
  → 状态名: fp32 / exp_avg / exp_avg_sq / step
  → 从flat buffer切片: state_flat_tensor.narrow(0, offset, numel).clone()

★ ★ Phase 2: _merge_tp_slice_files() (378) — 7种TP合并策略!

  merge_tp_slices() (232):
    1. TP replicated: 取第一个切片 → 验证所有相等 → replicated_parameters
    2. 平均: sum(slices) / len(slices) → parameters_to_average
    3. 2子参数cat_dim=0: chunk→2子切片→沿dim=0 cat每个子切片→重新cat
    4. ★ 通用子参数: SubparamShape → partition_dim + sub_dim_sizes → 沿分区维度合并
    5. 行并行: torch.cat(slices, dim=1) → parameters_with_row_parallelism
    6. 默认列并行: torch.cat(slices, dim=0) → most common!
    7. ★ vocabulary: 去除pad到original_vocab_size → vocabulary_parameters

  每输出文件保存dict: {PARAM, CAT_DIM, PARAM_N_SUB_PARAMS, VOCAB_TENSOR, SUB_PARAM_SHAPE}

★ ★ Phase 3: _save_optimizer_state() (410)
  → 保存全局optimizer state(排除per-shard数据)
  → 不包含: base_optimizer_state / param_slice_mappings / single_partition_of_fp32_groups

★ Universal目录结构:
  output_folder/
    zero/
      {param_name}/
        fp32.pt          # dict: {PARAM, CAT_DIM, ...}
        exp_avg.pt       # dict: {PARAM, ...}
        exp_avg_sq.pt    # dict: {PARAM, ...}
        step.pt          # step value (optional)
    optimizer_state.pt   # global optimizer state
    mp_rank_00_model_states.pt  # copied (with UC info injected)
    latest_universal              # points to step folder

  Version: UNIVERSAL_CHECKPOINT_VERSION_VALUE = 0.3

★ ★ AutoEP集成: consolidate_autoep_expert_files() (autoep_universal.py:111)
  → 每专家checkpoint → stack成[E_total, H, D] → EP_IS_EXPERT_PARAM=True + EP_NUM_EXPERTS
  → MoE模型 → AutoEP universal checkpoint → 跨EP配置加载!
```

## 3. BF16Optimizer Checkpoint — 无Loss Scaling

```
bf16_optimizer.py (477):

★ ★ state_dict:
  CLIP_GRAD / BASE_OPTIMIZER_STATE / SINGLE_PARTITION_OF_FP32_GROUPS
  GROUP_PADDINGS / PARTITION_COUNT / DS_VERSION / PARAM_SLICE_MAPPINGS

★ ★ BF16特殊处理:
  1. 无loss scaling: custom_loss_scaler=False / external_loss_scale=None (64-65)
     → BF16不需要loss scaling(FP16需要) → 更安全!
  2. 双精度: bf16_groups_flat(LP) + fp32_groups_flat_partition(HP)
     → LP=BF16参数 → HP=FP32分区 → optimizer在HP上操作
  3. ★ HP-LP链接: link_hp_params() (248)
     → _hp_mapping: 每个LP参数的lp_fragment_address(start, numel)
     → HP分区→LP参数映射 → update_lp_params() → FP32→BF16→AllGather恢复完整BF16
  4. grad_acc_dtype: FP32推荐 → BF16可选 → BF16时step后清除梯度(326-328)
  5. 分区方案: BF16参数flatten → align到2*world_size(NCCL 4字节对齐) → 等量分区
  6. 文件前缀: bf16_zero_pp_rank_ → vs ZeRO-3 zero_pp_rank_

★ Universal加载: _load_universal_checkpoint() (539)
  → load_hp_checkpoint_state_from_checkpoint_dir("bf16_groups", checkpoint_folder)
  → load_hp_checkpoint_state() → per-param加载 → TP/EP切片
```

## 4. LoRA Checkpoint — 原生实现(非PEFT)

```
★ ★ DeepSpeed LoRA = LoRAOptimizedLinear → 原生 → 不是PEFT!

LoRAOptimizedLinear (optimized_linear.py:76):
  → LoRAConfig (config.py:13): lora_r=64, lora_alpha=16, base_weight_sharding, offload, target_mods
  → lora_weight_1 (A矩阵, input→lora_r) + lora_weight_2 (B矩阵, lora_r→output) (143-149)
  → A: Kaiming均匀初始化 / B: 零初始化 → 与PEFT一致
  → ★ 基础权重: requires_grad=False + ds_optim_param=True → ZeRO不分片 → 当frozen_param处理!

★ ★ base_weight_sharding: zero_shards>1时 → 基础权重在rank间分区
  → _load_from_state_dict() (161): incoming_param.flatten().narrow(0, rank*shape_local, shape_local)

★ ★ Hybrid Engine LoRA (hybrid_engine.py):
  fuse_lora() (63): param.data += lora_scaling * torch.matmul(lora_left_weight.t(), lora_right_weight.t())
  unfuse_lora() (72): param.data -= lora_scaling * torch.matmul(...)
  → 推理=merge → 训练=unmerge → 每step切换 → ZeRO混合引擎!

★ ★ Checkpoint: LoRA参数 = 可训练参数 → 在ZeRO checkpoint中正常保存
  → 无特殊LoRA checkpoint格式 → 是model state_dict中的参数
  → zero_to_fp32.py提取 → 包含LoRA权重 → 需要手动分离或merge

★ ★ vs PEFT:
  → DeepSpeed LoRA: 原生 → LoRAOptimizedLinear → 初始化时直接替换线性层
  → PEFT: 包装器 → PeftModel → 增加额外层 → 与DeepSpeed不兼容
  → ★ 转换: zero_to_fp32 → 包含LoRA → PEFT format → 需手动分离 → 或merge后只保存base

★ ★ 量化支持: LoRAOptimizedLinear支持QuantizedParameter (110)
  → 基础权重量化存储 → LoRA权重BF16 → full_weight()反量化 → AllGather → 推理merge
```

## 5. 推理部署实战 — ZeRO→vLLM/SGLang/TRT-LLM

```
★ ★ 完整部署pipeline: ZeRO shard → FP32合并 → HF格式 → 推理框架

Step 1: ZeRO-3 → FP32合并
  # CLI (推荐大模型)
  python zero_to_fp32.py ./ckpt/global_step500/ ./extracted/ --safe_serialization --max_shard_size 5GB

  # 或: Python API (集成到脚本)
  from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
  state_dict = get_fp32_state_dict_from_zero_checkpoint("./ckpt/", tag="global_step500")

Step 2: FP32 → HF格式
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("base-model", state_dict=state_dict, torch_dtype=torch.bfloat16)
  model.save_pretrained("./hf_deploy/", safe_serialization=True)
  tokenizer = AutoTokenizer.from_pretrained("base-model")
  tokenizer.save_pretrained("./hf_deploy/")

  → 输出: model.safetensors + config.json + tokenizer.json → ★ HF标准格式!

Step 3: HF → vLLM serving
  # 单GPU
  vllm serve ./hf_deploy/

  # INT4量化(推荐!)
  vllm serve ./hf_deploy/ --quantization gptq

  # 多GPU TP
  vllm serve ./hf_deploy/ --tensor-parallel-size 4

★ ★ SGLang serving:
  # 方法A: 转HF后加载(推荐)
  python -m sglang.launch_server --model-path ./hf_deploy/ --load-format hf

  # 方法B: 直接加载ZeRO checkpoint(实验性)
  python -m sglang.launch_server --model-path ./ckpt/ --load-format ds --tokenizer-path base-model

★ ★ TRT-LLM路径(4步):
  1. ZeRO → FP32 → HF (同Step 1-2)
  2. HF → TRT-LLM weights: convert_llama_hf.py --model_dir ./hf_deploy/ --output_dir ./trt_weights/ --tp_size 4
  3. TRT-LLM engine build: trtllm-build --model_dir ./trt_weights/ --engine_dir ./trt_engine/
  4. TensorRT serving → 最高吞吐!

★ ★ RTX 4090最优路径:
  → ZeRO-2+LoRA训练 → universal ckpt → FP32 → HF → INT4量化 → vLLM → 4,791 tok/s
  → 或: EAGLE spec decode → 9,088 tok/s → ★ 最优!
  → ZeRO-3训练 → RTX 4090 PCIe不可行(3Ψ通信灾难) → 只能ZeRO-2+LoRA
```

## 6. 7个常见转换错误与解决方案

```
★ ★ 实战最常见的7个错误:

1. ★ Missing rank shard files
   → ValueError: Expected N of '*_optim_states.pt' but found M files
   → 原因: 训练crash或GPU故障 → checkpoint不完整
   → 解决: 确保所有zero_pp_rank_*文件完整 → PARTITION_COUNT验证

2. ★ BF16提取失败(旧版DeepSpeed)
   → KeyError: 'fp16_params' → 老脚本假设FP16
   → 解决: DeepSpeed >= 0.9.0 → BF16_ZERO_FILE_PREFIX = 'bf16_' + ZERO_FILE_PREFIX

3. ★ CPU OOM提取(最常见!)
   → 进程killed → 所有shard同时加载 → CPU内存不够
   → 解决: lazy_mode=True → 逐参数materialize → 或 --safe_serialization + --max_shard_size 5GB

4. ★ Shape mismatch加载HF
   → RuntimeError: size mismatch for X → 形状不一致
   → 原因: param_persistence_threshold变化 / padding未strip / ds_config.json不一致
   → 解决: 相同模型架构 + 验证threshold + 确保strip padding

5. ★ Missing config.json/tokenizer
   → vLLM/SGLang拒绝加载 → ZeRO checkpoint无这些文件!
   → 解决: save_pretrained()生成config.json + save tokenizer

6. ★ model.save_pretrained()直接在ZeRO模型上
   → NotImplementedError → 分区参数不能直接保存
   → 解决: 先zero_to_fp32提取 → 再load到新模型 → 再save_pretrained

7. ★ NVMe checkpoint不兼容
   → ZeRO-3常规checkpoint ≠ NVMe checkpoint → 格式不同
   → 解决: ds_to_universal.py转换 → 或zero_to_fp32.py提取 → 再HF格式
```

## 7. Universal Checkpoint加载 — 跨GPU数量

```
★ ★ Universal format核心价值: 跨DP/TP/EP配置加载!

load_hp_checkpoint_state() (universal_checkpoint.py:99):
  → 从{param_name}/{state_key}.pt逐参数加载
  → 获取完整未分区参数: ckpt_dict[PARAM]
  → TP切片: full_hp_param.chunk(tp_world_size, dim=chunk_dim)[tp_rank] (213)
  → EP切片: full_hp_param[ep_start:ep_end] (137)
  → AutoTP分区: _resolve_autotp_partition() (34) → ds_autotp_universal_checkpoint_meta属性

ZeRO-3 universal加载(stage3.py:3167):
  → load_hp_checkpoint_state_from_checkpoint_dir_stage3()
  → 从zero/{param_name}/{key}.pt逐参数加载
  → flat vector: .view(-1) → partitioned_numel = ceil(numel / world_size) → 添加填充
  → 按rank切片: .narrow(0, rank * partitioned_numel, partitioned_numel)

★ ★ 关键: Universal format → 训练8GPU → 推理1GPU → 不同world_size → 可以!
  → legacy format → 训练8GPU → 必须8GPU加载 → 不能改变world_size!
  → Universal = 训练→推理的关键桥梁 → 改GPU数量 → 不需要retrain!
```

## 8. 关键设计洞察

```
1. GatheredTensor延迟合并 → 大模型提取关键!
   → 70B模型 → 140GB CPU RAM → 同时materialize → OOM!
   → lazy_mode → 逐参数.contiguous() → 释放前一个 → O(1)额外内存 → 安全!
   → ★ 这是zero_to_fp32对大模型的核心优化 → 延迟→省内存!

2. ZeRO-3 vs ZeRO-2合并算法 → ZeRO-2更简单但ZeRO-3更灵活!
   → ZeRO-2: torch.cat所有shards → 1个flat vector → narrow切片 → 简单
   → ZeRO-3: GatheredTensor → 逐参数交错 → 每rank flat_group跨参数边界 → 复杂但精确
   → ★ ZeRO-3需要GatheredTensor因为参数在flat_groups中交错 → 不能简单cat!

3. Universal checkpoint → 跨GPU数量 → 训练→推理桥梁!
   → 旧格式: 固定GPU数量 → 不能1GPU推理8GPU训练的checkpoint → 灾难!
   → Universal: 逐参数 → 可改变DP/TP/EP配置 → 1GPU推理8GPU训练 → 极好!
   → ★ 这是ZeRO checkpoint的最大限制解决 → Universal format = 必用!

4. BF16Optimizer HP-LP链接 → 双精度但无loss scaling!
   → HP=FP32分区(optimizer用) → LP=BF16参数(model用) → 链接映射
   → update_lp_params() → FP32→BF16→AllGather → 恢复完整BF16 → 推理用
   → ★ BF16比FP16安全 → 无loss scaling → 无overflow/underflow → 训练推荐!

5. LoRA原生 vs PEFT → DeepSpeed自己实现 → 不兼容PEFT!
   → DeepSpeed LoRA: LoRAOptimizedLinear → 初始化时直接替换 → ZeRO兼容
   → PEFT: PeftModel wrapper → 与ZeRO-3参数分片不兼容 → conflict!
   → ★ DeepSpeed训练 → LoRA原生 → zero_to_fp32提取 → 需手动分离adapter

6. 推理部署路径: ZeRO→FP32→HF→量化→serving → 4步!
   → Step1: zero_to_fp32.py → FP32合并 → lazy_mode省内存
   → Step2: HF model.save_pretrained → safetensors → 标准格式
   → Step3: 量化(INT4/FP8) → 推理加速 → 4-8x吞吐提升
   → Step4: vLLM/SGLang serving → continuous batching → 生产级!
   → ★ 每步都是必需的 → 不能跳过 → ZeRO格式不能直接推理!

7. 7种TP合并策略 → Universal format处理所有并行模式!
   → replicated / averaged / 2-sub-param / general sub-param / row-parallel / column-parallel / vocabulary
   → ★ 不同的参数可能需要不同的合并策略 → TP→HF需要知道每个参数的并行方式!
   → vs FSDP2: DTensor自带Shard信息 → 不需要7种策略 → 更简单!

8. CPU OOM是大模型提取最大瓶颈 → lazy_mode是解决方案!
   → 70B模型: FP32 state_dict = 280GB → CPU RAM不够 → OOM!
   → lazy_mode=True → GatheredTensor → 逐参数 → 释放 → peak=1个参数大小 → 安全
   → ★ 这是生产环境常见问题 → 必须用lazy_mode或分shard输出!
```

---

Sources:
- deepspeed/utils/zero_to_fp32.py (761 lines — GatheredTensor + ZeRO-2/3 merge + safetensors)
- deepspeed/checkpoint/ds_to_universal.py (Universal format converter — 3-phase + 7 TP merge strategies)
- deepspeed/checkpoint/universal_checkpoint.py (Universal format loader — load_hp_checkpoint_state)
- deepspeed/checkpoint/constants.py (All checkpoint key constants)
- deepspeed/checkpoint/autoep_universal.py (AutoEP expert consolidation)
- deepspeed/runtime/bf16_optimizer.py (BF16Optimizer — HP-LP link + no loss scaling)
- deepspeed/runtime/zero/stage3.py (ZeRO-3 state_dict + load)
- deepspeed/runtime/zero/stage_1_and_2.py (ZeRO-2 state_dict + load)
- deepspeed/linear/optimized_linear.py (LoRAOptimizedLinear — native LoRA)
- deepspeed/module_inject/containers/features/hybrid_engine.py (LoRA fuse/unfuse)
- Background agents research (DeepSpeed universal checkpoint + deployment)
