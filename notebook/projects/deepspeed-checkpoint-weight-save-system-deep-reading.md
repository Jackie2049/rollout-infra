# DeepSpeed Checkpoint/Weight Save System — Deep Reading

> 2026-06-19 | Complete checkpoint lifecycle: save, load, extract, convert, deploy
> ★★★★★★★★ Key: ZeRO-2 vs ZeRO-3 format differences, weight extraction for rollout, CPU_Adam state format, LoRA checkpoint integration
> RTX 4090 GRPO: MUST use ZeRO-2 + CPU_Adam, lazy_mode=True for weight extraction, universal format for cross-GPU loading

---

## 1. Checkpoint Save Call Tree

```
save_checkpoint(save_dir, tag, client_state) [engine.py:4032]
  |-- optimizer.checkpoint_event_prologue() [ZeRO-3: partition_all_parameters]
  |-- checkpoint_engine.makedirs(save_dir)
  |-- dist.barrier()
  |-- tag = f"global_step{self.global_steps}"
  |-- _create_checkpoint_file(save_dir, tag)
  |-- _save_checkpoint(save_dir, tag, client_state) [engine.py:4382]
  |     |-- state = dict(
  |     |     module=module_state_dict(),
  |     |     buffer_names=...,
  |     |     optimizer=None (for ZeRO),
  |     |     param_shapes=_get_zero_param_shapes() (for ZeRO),
  |     |     frozen_param_shapes/frozen_param_fragments,
  |     |     shared_params, lr_scheduler, global_steps, dp_world_size,
  |     |     ds_config, ds_version
  |     |   )
  |     |-- checkpoint_engine.save(state, save_path)
  |-- if save_zero_checkpoint:
  |     |-- _save_zero_checkpoint(save_path, tag) [engine.py:4569]
  |     |     |-- zero_sd = dict(
  |     |     |     optimizer_state_dict=optimizer.state_dict(),
  |     |     |     ds_config, ds_version
  |     |     |   )
  |     |     |-- checkpoint_engine.save(zero_sd, zero_checkpoint_name)
  |     |     |-- _copy_recovery_script() → zero_to_fp32.py copied to checkpoint dir
  |-- optimizer.checkpoint_event_epilogue() [ZeRO-3: invalidate+gather]
  |-- checkpoint_engine.commit()
  |-- write 'latest' file on rank 0
```

---

## 2. ZeRO-2 vs ZeRO-3 Checkpoint Structure

### ZeRO-2: Two files per rank
```
checkpoint_dir/global_stepXXX/
  mp_rank_00_model_states.pt              # Per MP rank (model weights)
  zero_pp_rank_X_mp_rank_00_optim_states.pt  # Per DP rank (optimizer state)
  (BF16: bf16_zero_pp_rank_X_mp_rank_00_optim_states.pt)
```

model_states.pt: module state_dict + buffer_names + param_shapes + frozen params + shared params + lr_scheduler + global_steps + dp/mp world sizes

optim_states.pt (ZeRO-2, stage_1_and_2.py line 2453):
```python
{
  "loss_scaler", "dynamic_loss_scale", "overflow", "clip_grad",
  "base_optimizer_state": PyTorch optimizer state_dict,
  "single_partition_of_fp32_groups": [Tensor],  # Per param_group: this rank's FP32 partition
  "zero_stage": ZeroStageEnum,  # 1 or 2
  "group_paddings": [int],  # NCCL alignment padding
  "partition_count": [int],  # DP world size at save time
  "param_slice_mappings": [dict],  # {name: fragment_address(start, numel)}
}
```

### ZeRO-3: Two files per rank (different internal format)
```
checkpoint_dir/global_stepXXX/
  zero_pp_rank_X_mp_rank_00_model_states.pt  # Per DP rank (PARTITIONED weights!)
  zero_pp_rank_X_mp_rank_00_optim_states.pt  # Per DP rank
```

★★★★★★★★ KEY DIFFERENCE: ZeRO-3 model_states has PARTITIONED weights (ds_tensor fragments), NOT full weights!

optim_states.pt (ZeRO-3, stage3.py line 2961):
```python
{
  "zero_stage": ZeroStageEnum.weights,  # Always 3
  "loss_scaler", "dynamic_loss_scale", "overflow",
  "partition_count": int,
  "optimizer_state_dict": PyTorch optimizer state_dict,
  "fp32_flat_groups": [Tensor],  # Per sub_group: full flat FP32 partition
}
```

★★★★★★★★ ZeRO-2 uses SINGLE_PARTITION_OF_FP32_GROUPS (per param_group)
★★★★★★★★ ZeRO-3 uses FP32_FLAT_GROUPS (per sub_group) — different key names!

---

## 3. Weight Extraction for Rollout Reload

### Method A: zero_to_fp32.py CLI (auto-copied to checkpoint dir)
```bash
python zero_to_fp32.py ./ckpt/ ./output/ --safe_serialization --max_shard_size 5GB
```

### Method B: get_fp32_state_dict_from_zero_checkpoint() API (zero_to_fp32.py line 533)
```python
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, tag="global_step500")
# ★★★★★★★★ Use lazy_mode=True for large models to avoid CPU OOM
```

### Method C: _zero3_consolidated_16bit_state_dict() (ZeRO-3 only, engine.py line 4625)
- Gathers parameters layer-by-layer via GatheredParameters context
- Moves to CPU one layer at a time (memory-efficient)
- Requires stage3_gather_16bit_weights_on_model_save=True

### ZeRO-2 Extraction Algorithm (_zero2_merge_trainable_params)
1. Collect SINGLE_PARTITION_OF_FP32_GROUPS from all DP ranks
2. torch.cat(merged_partitions) per param_group → full FP32 vector
3. narrow(0, offset, numel).view(shape) per param
4. NCCL alignment: offset must align to 2 * world_size

### ZeRO-3 Extraction Algorithm (_zero3_merge_trainable_params)
1. Uses GatheredTensor lazy merge class
2. Reads FP32_FLAT_GROUPS + PARAM_SHAPES
3. lazy_mode=True: .contiguous() triggers actual merge, materialize per param
4. [:numel].view(shape) strips padding

---

## 4. CPU_Adam Optimizer State Format

CPU_Adam state per parameter:
```python
state[p] = {
    'step':       int,
    'exp_avg':    Tensor (FP32, CPU),  # Momentum
    'exp_avg_sq': Tensor (FP32, CPU),  # Variance
}
```

★★★★★★★★ All states on CPU (device=torch.device('cpu'))
★★★★★★★★ With pin_memory=True: pinned for faster GPU-CPU transfer
★★★★★★★★ In ZeRO-2 context: optimizer has ONE key per param_group (flat partition), not per original param

---

## 5. LoRA/PEFT Checkpoint Integration

### DeepSpeed Native LoRA (LoRAOptimizedLinear)
- LoRA parameters (lora_weight_1, lora_weight_2) are regular trainable params
- Base weights: requires_grad=False + ds_optim_param=True → treated as frozen
- In checkpoint: LoRA weights in module state_dict alongside base weights
- fuse_lora()/unfuse_lora() for inference/training mode switching

### ★★★★★★★★ PEFT INCOMPATIBLE with ZeRO-3
- PEFT PeftModel wrapper NOT compatible with ZeRO-3 parameter partitioning
- DeepSpeed native LoRA is ONLY ZeRO-compatible LoRA approach

---

## 6. RTX 4090 GRPO Checkpoint Best Practices

1. ★★★★★★★★ ALWAYS ZeRO-2 + CPU_Adam on RTX 4090 (ZeRO-3 pure overhead dp=1)
2. Save at end of each GRPO epoch or fixed step intervals
3. ★★★★★★★★ Use lazy_mode=True for weight extraction (avoid CPU OOM)
4. Universal checkpoint for cross-GPU deployment: ds_to_universal converter
5. LoRA: fuse before extraction OR manually separate adapter weights
6. gradient_clipping=1.0 ALWAYS (default 0 → dangerous for GRPO)
7. overlap_comm=False ALWAYS on single GPU (#8061 NaN)
8. zero_to_fp32.py auto-copied to every checkpoint dir
9. All ranks MUST call save_checkpoint (not just rank 0)
10. Tag format: global_step{N}, must be consistent across ranks

---

## 7. Common Checkpoint Bugs

1. ★★★★★★★★ #8072/#8073: ZeRO-3+PEFT LoRA dtype regression (INCOMPLETE fix!)
2. #8075: fd leak (missing close() call), long-running GRPO exhausts fds
3. #8068: gradient_clipping default 0 (MUST set 1.0)
4. CPU OOM during weight extraction → lazy_mode=True solution
5. Missing rank shard files → PARTITION_COUNT validation
6. ZeRO-3 cannot save then immediately load → must reinitialize engine
7. DP world size mismatch on load → only universal format supports cross-DP
8. ZeRO-3 elastic_checkpoint NotImplementedError (not supported!)
9. BF16 extraction failure in old versions → DeepSpeed >= 0.9.0 needed
10. Shape mismatch loading HF → validate threshold + strip padding

---

## References

- DeepSpeed engine.py: L4032 (save), L3677 (load), L4382 (_save), L4569 (_save_zero)
- ZeRO-2: stage_1_and_2.py L2453 (state_dict), L2607 (load_state_dict)
- ZeRO-3: stage3.py L2961 (state_dict), L3062 (load_state_dict)
- BF16: bf16_optimizer.py L477 (state_dict)
- zero_to_fp32.py: L533 (get_fp32_state_dict), L252 (ZeRO-2 merge), L437 (ZeRO-3 merge)
- CPU_Adam: cpu_adam.py (state format, C++ kernel)
