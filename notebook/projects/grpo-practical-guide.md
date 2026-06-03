# GRPO 实战指南 (verl)

> 使用 verl 框架从零跑通 GRPO 训练的完整实践指南

## 1. GRPO 回顾

GRPO (Group Relative Policy Optimization) 是 PPO 的简化版：
- **不需要 Critic/Value 模型** (用组均值替代)
- **Outcome-level reward** (每个 response 一个标量奖励)
- **优势 = (score - group_mean) / group_std**
- 模型数量: PPO 4个 (Actor+Critic+Ref+RM) → GRPO 2-3个 (Actor+Ref(optional)+RM(optional))

## 2. 环境准备

### 2.1 安装 verl

```bash
# 从源码安装
pip install -e .[vllm]

# 或使用 Docker
docker pull verlai/verl:base-verl0.5-cu126-cudnn9-torch2.7.0-fa2.7.4
```

### 2.2 硬件要求

| 模型大小 | 最小 GPU | 推荐 GPU | 备注 |
|---------|---------|---------|------|
| 0.5B | 1× GPU (24GB) | 1× A100 | 快速验证 |
| 7B | 1× GPU (80GB) | 4-8× A100 | 配合 offload |
| 8B | 4× GPU | 8× A100 | 生产推荐 |
| 70B | 8× H100 | 16× H100 | 大规模训练 |

关键: `param_offload=True` + `optimizer_offload=True` 可将显存需求减半。

## 3. 数据准备

### 3.1 数据格式: Parquet

```python
# 每行包含以下列:
{
    "data_source": "openai/gsm8k",           # 奖励路由标识
    "prompt": [{"role": "user", "content": "问题文本"}],
    "ability": "math",                        # 能力标签
    "reward_model": {"style": "rule", "ground_truth": "42"},
    "extra_info": {"split": "train", "index": 0}
}
```

### 3.2 内置数据预处理

```bash
# GSM8k 数学题
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# MATH 数据集
python3 examples/data_preprocess/math_dataset.py --local_save_dir ~/data/math
```

### 3.3 关键配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data.train_files` | 训练 parquet 路径 | - |
| `data.val_files` | 验证 parquet 路径 | - |
| `data.train_batch_size` | 每 step 的 **prompt 数** | 32 |
| `data.max_prompt_length` | prompt 最大长度 | 256 |
| `data.max_response_length` | response 最大长度 | 256 |
| `data.truncation` | 超长 prompt 处理: error/left/right | error |

## 4. 启动训练

### 4.1 最小命令 (单 GPU, 7B)

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=$HOME/data/math/train.parquet \
  data.val_files=$HOME/data/math/test.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=15
```

### 4.2 生产配置 (8 GPU, 8B)

```bash
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
```

关键默认值:
- `train_batch_size=1024`, `ppo_mini_batch_size=256`
- `max_prompt_length=1024`, `max_response_length=2048`
- `actor_lr=1e-6`, `kl_loss_coef=0.001`
- `rollout_n=5` (每 prompt 5 个 response)
- `rollout_tp=2` (vLLM TP)

## 5. GRPO 核心旋钮

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm.adv_estimator` | gae | **必须设为 grpo** |
| `actor_rollout_ref.rollout.n` | - | 每 prompt 采样数 (≥2) |
| `algorithm.norm_adv_by_std_in_grpo` | True | 是否用 std 归一化优势 |
| `actor_rollout_ref.actor.use_kl_loss` | False | KL 损失 (需要 Ref Policy) |
| `actor_rollout_ref.actor.clip_ratio` | 0.2 | PPO clip 范围 |
| `actor_rollout_ref.actor.entropy_coeff` | 0 | 熵正则系数 |

## 6. 奖励函数

### 6.1 内置奖励

verl 为常见数据集提供内置评分:
- GSM8k: 数学计算正确性
- MATH: 数学证明正确性
- DAPO: 动态 advantage
- Code: 代码执行

### 6.2 自定义奖励

```python
# my_reward.py
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """返回 float 或 dict"""
    # data_source: 数据集标识 (e.g., "my_dataset")
    # solution_str: 模型生成的 response
    # ground_truth: 标准答案
    if solution_str.strip() == ground_truth.strip():
        return 1.0
    return 0.0
```

```bash
reward.custom_reward_function.path=/path/to/my_reward.py
reward.custom_reward_function.name=compute_score
```

返回 dict 支持多维奖励:
```python
return {"score": 0.8, "format": 1.0, "correctness": 0.0}
```

## 7. 训练循环

```
每个 epoch:
  1. DataLoader 采样 train_batch_size 个 prompt
  2. 分配 uid (每个 prompt 唯一标识)
  3. 重复 rollout.n 次 → 总轨迹数 = batch_size × n
  4. vLLM/SGLang rollout 生成 response
  5. 计算 ref log prob (如果 use_kl_loss)
  6. 奖励函数评分 (compute_score 调度)
  7. GRPO 优势估计 (按 uid 分组)
  8. Actor 更新 (PPO clip loss)
  9. 验证 (定期)
```

## 8. Dr. GRPO 变体

来自 "Understanding R1-Zero-Like Training" (arxiv 2503.20783):

```bash
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
actor_rollout_ref.actor.use_kl_loss=False
algorithm.norm_adv_by_std_in_grpo=False
```

## 9. 资源优化技巧

| 技巧 | 效果 | 配置 |
|------|------|------|
| 梯度检查点 | 显存 -40% | `enable_gradient_checkpointing=True` |
| 参数 offload | 显存 -50% | `param_offload=True` |
| 优化器 offload | 显存 -30% | `optimizer_offload=True` |
| 混合精度 | 显存 -35% | BF16 默认 |
| Prefix Grouper | 计算量 -40% | `use_prefix_grouper=True` |
| 减少 rollout.n | 计算量 -n× | `rollout.n=2` (最小) |

## 10. 验证和调试

```bash
# 查看训练日志
trainer.logger='[console]'

# 或使用 WandB
trainer.logger=['console','wandb']
wandb.project=grpo-experiment

# 常见问题:
# 1. OOM → 增大 offload, 减小 batch_size/max_length
# 2. 奖励全零 → 检查 compute_score 和 ground_truth 格式
# 3. NaN loss → 降低 lr, 检查 reward 分布
```

## 参考资料

- verl 源码: `verl/` (shallow clone)
- GRPO 示例: `verl/examples/grpo_trainer/`
- 配置文件: `verl/trainer/config/ppo_trainer.yaml`
- 相关笔记: [RLHF/GRPO 基础](../fundamentals/rlhf-training-infra.md), [verl 源码](verl-source-reading.md)
