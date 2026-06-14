# verl vs rLLM Transform Pipeline 对比 (源码级)

> 2026-06-15 | 核心: verl直接DataProto ← 单步rollout输出; rLLM添加prefix-merge层 ← 多轮Episode/TrajectoryGroup
> 源码: rllm/trainer/verl/transform.py (625行) vs verl/trainer/ppo/ray_trainer.py (1771行) + verl/protocol.py (1346行)

## 1. 数据类型对比

### verl 原生数据流
```
DataLoader → batch (input_ids, attention_mask) + uid
  → ActorRolloutRefWorker.generate_sequences() → DataProto
    batch: {input_ids, responses, logprobs, ...}
    non_tensor_batch: {uid, reward_fn, ...}
  → compute_reward → DataProto + rewards
  → compute_advantage → DataProto + advantages
  → ActorRolloutRefWorker.update_policy() → 训练
```

**关键特征**: verl是**单步**数据流 — 一个prompt → 一个response → 一行数据

### rLLM 数据流
```
Task → @rllm.rollout → Episode {task, trajectories}
  → TrajectoryGroup {trajectories, group_id}
    → transform_episodes_to_dataproto()
      → _process_episode → _process_trajectory
        → prefix-merge: [A0, obs1, A1, obs2, A2...] → mask=[1,0,1,0,1...]
      → AccumulatedData → _batch_tensors_and_build_data_proto → DataProto
    → update_dataproto_with_advantages → DataProto + advantages
    → VerlBackend.train() → verl训练
```

**关键特征**: rLLM是**多步**数据流 — 一个task → N轮对话 → prefix-merge → 一行数据(通常)

## 2. 核心差异: Prefix-Merge

### 2.1 verl: 单步 → 单行

```python
# verl ray_trainer.py
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
# → 每个prompt × n个response → n行
# 每行: [prompt_ids | response_ids]
# response全是action tokens, 全部参与loss计算
```

**特点**:
- 每个step = 一行
- response = 纯action tokens
- loss = 所有response token的log_probs × advantages
- GRPO: 同一uid的n个response → 组归一化advantage

### 2.2 rLLM: 多步prefix-merge → 单行(通常)

```python
# rllm transform.py: _process_trajectory()
# 步骤1: 创建segment
seg = _new_segment(valid_steps[0])
# prompt = step.model_output.prompt_ids
# response = step.model_output.completion_ids (action tokens)
# mask = [1] * len(action)  ← 全部是action

# 步骤2: 检查prefix-extension
for step in valid_steps[1:]:
    if prompt_ids[:len(seg["full_seq"])] == seg["full_seq"]:
        # Cumulative — 合并!
        delta_obs = prompt_ids[len(seg["full_seq"]):]  # 工具输出/观测
        action = step.model_output.completion_ids

        seg["response"].extend(delta_obs)    # 观测tokens
        seg["response"].extend(action)        # action tokens
        seg["mask"].extend([0] * len(delta_obs))  # 观测=0
        seg["mask"].extend([1] * len(action))     # action=1
    else:
        # Prefix break → emit当前segment, start新segment
        _emit(seg)
        seg = _new_segment(step)

# 结果: 一行 = [prompt | A0, obs1, A1, obs2, A2, ...]
# mask  = [1, 0, 1, 0, 1, ...]  ← 只有action tokens参与loss
```

**关键对比**:

| 维度 | verl原生 | rLLM prefix-merge |
|------|----------|-------------------|
| 数据粒度 | 1步=1行 | N步→1行(通常) |
| response内容 | 纯action tokens | action + interleaved obs tokens |
| loss参与 | 全部response tokens | 只有mask=1的action tokens |
| 多轮支持 | 无(每个step独立) | 天然支持(prefix-merge保留上下文) |
| tokenization drift | 不考虑 | TokenAccumulator消除 |
| observation处理 | 不存在(单步) | mask=0 → 不参与loss但参与模型forward |

## 3. Advantage传播机制对比

### 3.1 verl GRPO advantage

```python
# verl core_algos.py
uid_list = batch.non_tensor_batch['uid'].tolist()
for uid in unique_uids:
    group_indices = [i for i, u in enumerate(uid_list) if u == uid]
    group_rewards = rewards[group_indices]  # scalar per row
    advantages[group_indices] = (group_rewards - mean_r) / (std_r + eps)
# → advantage是scalar, broadcast到所有response tokens
# → 所有token同等advantage, 无mask区分
```

### 3.2 rLLM GRPO advantage

```python
# rllm transform.py: update_dataproto_with_advantages()
adv_by_traj_uid = {}
for trajectory in item.trajectories:
    adv = next((s.advantage for s in trajectory.steps if s.advantage is not None), 0.0)
    adv_by_traj_uid[trajectory.uid] = adv

advantages = [0.0 if is_pad[i] else adv_by_traj_uid.get(str(step_ids[i]), 0.0)]
advantage_tensor = _build_per_step_advantages(batch.batch["response_mask"], advantages)
# → advantages * response_mask
# → action tokens获得advantage, observation tokens获得0
# → 关键差异: mask=0的地方advantage=0 → loss=0 → 不影响梯度
```

**对比**:

| 方面 | verl | rLLM |
|------|------|------|
| advantage粒度 | per uid group (scalar) | per trajectory uid (scalar) |
| broadcast范围 | 全response tokens | response_mask=1的action tokens |
| observation tokens | 不存在 | advantage=0 → 不影响loss |
| 等价性 | 单步时完全等价 | 多步时更精确(obs不污染梯度) |

## 4. DataProto字段对比

### 4.1 verl原生DataProto (ray_trainer.py)

```python
batch.batch = {
    "input_ids": [bs, seq_len],        # prompt + response
    "attention_mask": [bs, seq_len],    # 0/1
    "position_ids": [bs, seq_len],      # cumulative
    "responses": [bs, resp_len],        # right-padded
    "old_log_probs": [bs, resp_len],    # rollout log probs
    "ref_log_probs": [bs, resp_len],    # reference log probs
    "advantages": [bs, resp_len],       # advantage values
    "rewards": [bs, resp_len],          # token-level rewards
}
batch.non_tensor_batch = {
    "uid": [bs],                        # unique prompt identifier
    "reward_fn": callable,              # reward function
}
```

### 4.2 rLLM增强DataProto (transform.py)

```python
batch.batch = {
    "input_ids": [bs, max_prompt + max_response],  # left-pad prompt + right-pad response
    "attention_mask": [bs, max_prompt + max_response],
    "position_ids": [bs, max_prompt + max_response],  # 或 [bs, 4, seq_len] (多模态)
    "prompts": [bs, max_prompt_len],           # left-padded
    "responses": [bs, max_response_len],       # right-padded
    "response_mask": [bs, max_response_len],   # ★ 关键: 1=action, 0=observation
    "traj_rewards": [bs, max_response_len],    # trajectory-level reward at last token
    "step_rewards": [bs, max_response_len],    # step-level reward at last token
    "advantages": [bs, max_response_len],      # advantage × response_mask
    "returns": [bs, max_response_len],         # = advantages
    "rollout_log_probs": [bs, max_response_len], # ★ importance sampling用
    "routed_experts": [bs, seq, layers, topk],  # ★ R3 router replay
}
batch.non_tensor_batch = {
    "episode_ids": [bs],              # ★ episode-level grouping
    "trajectory_ids": [bs],           # ★ trajectory-level grouping
    "step_ids": [bs],                 # = trajectory.uid for advantage lookup
    "batch_ids": [bs],                # unique batch identifier
    "step_nums": [bs],                # steps per row (after merge)
    "is_correct": [bs],               # episode-level correctness
    "termination_reasons": [bs],      # why episode ended
    "metrics": [bs],                  # episode-level metrics
    "is_valid": [bs],                 # valid row flag
    "is_last_step": [bs],             # last step of trajectory?
    "is_pad_step": [bs],              # padding row flag
    "group_roles": [bs],              # ★ per-row role name (for per-role loss)
    "multi_modal_inputs": [bs],       # ★ Qwen2VL/Qwen3VL inputs
}
batch.meta_info = {
    "repeat_counts": [...],           # per-episode repeat counts
    "merge_metrics": {...},           # merge compression stats
}
```

**关键新增字段**:
- `response_mask`: 多轮mask → 只有action tokens参与loss
- `episode_ids/trajectory_ids/step_ids`: 3层grouping → 精细advantage分配
- `group_roles`: per-role loss routing → 不同agent角色可以不同loss权重
- `rollout_log_probs`: importance sampling + bypass mode
- `routed_experts`: R3 router replay → MoE expert routing复用
- `traj_rewards + step_rewards`: 双层reward → trajectory级 + step级

## 5. 前缀共享机制对比

### 5.1 verl prefix sharing

```python
# verl ray_trainer.py
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
# → 相同prompt × n → vLLM prefix caching自动复用KV

# verl prefix_grouper_utils.py
# → 训练时: prefix_grouper按prefix分组 → 合享forward pass
# → 但这是训练端优化, 不影响数据格式
```

**层次**: rollout端(vLLM prefix caching) + 训练端(PrefixGrouper)

### 5.2 rLLM prefix handling

```python
# rllm TokenAccumulator: 消除multi-turn tokenization drift
# prev_prompt_ids + prev_completion_ids → bridge_to_next_turn → 正确拼接

# rllm transform.py: prefix-merge
# 多步trajectory → prefix-extension检测 → 合并为单行
# → prompt = 第一步的prompt (最小)
# → response = [A0, obs1, A1, obs2, A2, ...] (全部后续内容)
# → 相当于把N步的KV cache合并到一行 → 更高效forward
```

**层次**: rollout端(TokenAccumulator drift-free) + transform端(prefix-merge压缩) + 训练端(response_mask精确loss)

## 6. 等价性分析

### 6.1 单步场景: 完全等价

当agent只执行一步(无多轮对话):
- verl: prompt → response → 一行 → advantage
- rLLM: Episode(1 trajectory, 1 step) → prefix-merge → 一行 → advantage
- 结果: **完全等价** — 相同的input_ids, responses, advantages

### 6.2 多步场景: rLLM更精确

当agent执行N步对话(如ReAct/tool-call):
- verl: 需要手动构造N行(每步一行) → loss按N行累加 → step数多的trajectory权重更大
- rLLM: prefix-merge → 一行 → loss按一行计算 → `seq-mean-token-mean` → 每trajectory等权重

**关键洞察**: rLLM的prefix-merge + `loss_agg_mode=seq-mean-token-mean` 确保:
- 6步trajectory = 1行 → 与2步trajectory的1行 → loss权重相同
- verl原生: 6步trajectory = 6行 → loss权重6倍于2步trajectory → 不公平

### 6.3 GRPO场景: 组归一化差异

- verl: uid = prompt identifier → 同一prompt的n个response → 组归一化
- rLLM: trajectory.uid → 同一trajectory的prefix-merged行 → 一个trajectory一行 → 不需要组内对比
- 当rLLM用GRPO: TrajectoryGroup内多个trajectory → 组归一化 → 与verl等价

## 7. 架构层次关系

```
verl (底层训练引擎):
  ├── DataProto 数据交换协议
  ├── RayPPOTrainer 训练循环
  ├── ActorRolloutRefWorker 训练+推理混合
  └── core_algos.py 优势估计+策略损失

rLLM (上层编排框架):
  ├── Episode/Trajectory/TrajectoryGroup 数据模型
  ├── @rllm.rollout/@rllm.evaluator 定义agent
  ├── TokenAccumulator 消除drift
  ├── transform.py: Episode → DataProto (prefix-merge)
  ├── VerlBackend: 调用verl训练
  └── 其他backend: tinker/fireworks

→ rLLM是verl的上层编排, 类似RLlib对Ray
→ rLLM transform.py是连接两层的关键桥梁
```

## 8. RTX 4090实战意义

### 8.1 单GPU GRPO训练

```
verl方案:
  7B + GRPO + BF16 + LoRA + CPU Adam → 20.04GB fits RTX 4090
  rollout.n=8 → vLLM异步rollout

rLLM方案:
  同上 + 但需额外:
    - TokenAccumulator内存(~0, 纯CPU)
    - prefix-merge CPU计算(~0, 纯Python)
    - response_mask额外tensor → ~0.5MB (可忽略)
  → 内存开销几乎相同
```

### 8.2 多轮Agent训练

```
verl原生: 不支持多轮 → 需要手动构造
rLLM: 天然支持 → prefix-merge + response_mask
→ RTX 4090: 多轮agent GRPO训练, rLLM是唯一可行方案(不需手动构造)
```

## 9. 总结

| 方面 | verl原生 | rLLM (over verl) |
|------|----------|-------------------|
| 数据粒度 | 单步 | 多步→单行(prefix-merge) |
| 多轮支持 | ❌ | ✅ (TokenAccumulator + mask) |
| observation处理 | 不存在 | mask=0 → 不影响梯度 |
| advantage分配 | 全token broadcast | action token only (×mask) |
| loss公平性 | step数不等权 | trajectory等权 |
| 数据字段 | 6-8 tensor fields | 10+ tensor + 13 non_tensor |
| 多模态 | 基本支持 | Qwen2VL/3VL position_ids |
| MoE routing | 不支持 | R3 router replay |
| 实用场景 | 单步GRPO/PPO | Agentic RL (tool-call, ReAct) |

**核心洞察**: rLLM transform.py不只是"翻译"数据格式, 它是**语义转换**:
- Episode/TrajectoryGroup → DataProto + response_mask + multi-level ids
- prefix-merge压缩 → 减少forward pass次数 → 节省GPU内存和计算
- response_mask → 精确loss → 只训练action tokens → 更高效梯度
- 这就是为什么rLLM说"same code eval+train" — agent代码天然产生多步轨迹, rLLM自动处理

## 10. 下一步

- [ ] 在GPU可用时实测rLLM GRPO + prefix-merge → 验证mask正确性
- [ ] 研究 verl PrefixGrouper vs rLLM prefix-merge 的训练效率对比
- [ ] 研究 rLLM tinker backend transform → 与verl backend transform差异
- [ ] 研究 R3 router replay 在MoE模型中的实际效果
