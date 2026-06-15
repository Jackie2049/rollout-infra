# rLLM TinkerBackend GRPO 实战指南 (RTX 4090 单GPU)

> 从零跑通 GRPO 训练的完整实践指南 — 不是理论, 是可执行的命令和配置

**结论速查**: RTX 4090 单GPU RL训练最优路径 = rLLM TinkerBackend + GRPO + LoRA-32 + bypass_mode + rule-based reward

---

## 目录

1. [为什么选 TinkerBackend](#1-为什么选-tinkerbackend)
2. [环境安装](#2-环境安装)
3. [数据准备](#3-数据准备)
4. [训练启动 — CLI 方式](#4-训练启动--cli-方式)
5. [训练启动 — Python API 方式](#5-训练启动--python-api-方式)
6. [完整配置参考](#6-完整配置参考)
7. [Reward 函数详解](#7-reward-函数详解)
8. [bypass_mode 与优势计算](#8-bypass_mode-与优势计算)
9. [LoRA 配置详解](#9-lora-配置详解)
10. [RTX 4090 显存验证](#10-rtx-4090-显存验证)
11. [训练监控](#11-训练监控)
12. [Checkpoint 管理](#12-checkpoint-管理)
13. [pass@k 评估](#13-passk-评估)
14. [LoRA 合并 + INT4 量化部署](#14-lora-合并--int4-量化部署)
15. [完整训练流程 (Step-by-Step)](#15-完整训练流程-step-by-step)
16. [常见问题与排错](#16-常见问题与排错)

---

## 1. 为什么选 TinkerBackend

★★★ RTX 4090 单GPU最优路径 = TinkerBackend, 理由:

| 对比项 | TinkerBackend | VerlBackend |
|--------|---------------|-------------|
| GPU需求 | 单GPU, in-process | 多GPU, Ray分布式 |
| LoRA | 自动初始化, 零手动代码 | 需手动配置+Ray actor |
| 权重同步 | 零拷贝 (in-process) | ZMQ/NCCL/RDMA |
| Ref model | LoRA disable_adapter → 同GPU | 需额外GPU或offload |
| bypass_mode | 默认true → 省forward pass | 默认false |
| 框架开销 | 最小 (asyncio) | Ray + vLLM subprocess |
| RTX 4090 | ★★★ 最优选择 | PCIe scaling灾难 |

核心优势: TinkerBackend在单GPU上in-process运行, 没有Ray/vLLM subprocess开销, LoRA自动初始化+零拷贝权重同步+bypass_mode=三个关键优化叠加.

---

## 2. 环境安装

### 2.1 本地安装 (Mac/开发机)

```bash
# 克隆仓库
git clone https://github.com/rllm-org/rLLM.git
cd rLLM

# Python 3.11+ 必需 (tinker依赖)
uv venv --python 3.11
source .venv/bin/activate

# 安装 rllm + tinker backend
uv pip install -e ".[tinker]"

# 安装 math cookbook (含reward函数和agent flow)
uv pip install --no-deps -e cookbooks/math

# 安装 reward 依赖 (sympy等)
uv pip install -e ".[rewards]"
```

### 2.2 GPU 服务器安装 (RTX 4090)

```bash
# SSH到GPU服务器
ssh zxw@219.223.198.62

# 创建conda环境
source ~/anaconda3/bin/activate llm
conda create -n rllm-tinker python=3.11 -y
conda activate rllm-tinker

# 安装PyTorch (RTX 4090 SM89)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 安装rllm
git clone https://github.com/rllm-org/rLLM.git
cd RLLM
pip install -e ".[tinker]"
pip install -e ".[rewards]"
pip install --no-deps -e cookbooks/math
```

### 2.3 验证安装

```bash
# 检查CLI可用
rllm --version

# 检查agent列表
rllm agent list    # 应显示 "math"

# 检查dataset可用
rllm dataset list --all
```

---

## 3. 数据准备

### 3.1 内置数据集 (推荐快速上手)

★★ rLLM内置50+数据集, 一键拉取:

```bash
# 数学训练集
rllm dataset pull hendrycks_math    # MATH训练集 (7.5K问题)
rllm dataset pull gsm8k             # GSM8K (8.5K小学数学)
rllm dataset pull deepscaler_math   # ~40K AIME/AMC/Omni-MATH
rllm dataset pull math500           # 500题MATH测试集

# 评估集
rllm dataset pull aime2024          # AIME 2024 (30题, pass@k评估最佳)
rllm dataset pull countdown         # Countdown game (小数据集, 快速验证)

# 代码
rllm dataset pull livecodebench     # 编程评估

# 查看数据集内容
rllm dataset inspect gsm8k -n 3     # 显示3个样本
```

数据存储位置: `~/.rllm/datasets/` (自动管理, Parquet格式)

### 3.2 自定义数据集

```python
from rllm.data.dataset import DatasetRegistry

# 从JSON/JSONL加载自定义数据
custom_data = [
    {"question": "What is 2+2?", "ground_truth": "4"},
    {"question": "Solve x^2=16", "ground_truth": ["4", "-4"]},
]

# 注册到rLLM (自动保存为Parquet)
dataset = DatasetRegistry.register_dataset(
    name="my_math",
    data=custom_data,
    split="train",
    source="local",
    description="My custom math problems",
)
```

### 3.3 数据格式要求

rLLM数据集每行需包含以下字段(至少一个):
- `question` 或 `problem` 或 `prompt` — 问题描述
- `ground_truth` — 正确答案 (reward函数需要)
- `data_source` — 数据来源标识 (可选)

GRPO数据流: `batch → interleave_tasks(group_size=N) → N次重复每个问题 → N个rollout → TrajectoryGroup → advantage计算`

---

## 4. 训练启动 — CLI 方式

★★★ 最简单方式, 一行命令启动:

```bash
# 数学 GRPO 训练 (最基础版)
rllm train hendrycks_math \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --group-size 8 \
    --batch-size 16 \
    --lora-rank 32 \
    --lr 2e-5 \
    --epochs 1 \
    --val-freq 10
```

### RTX 4090 优化配置

★★★ RTX 4090专用配置 — group_size=4+batch_size=8 (单GPU显存极限):

```bash
rllm train gsm8k \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --group-size 4 \
    --batch-size 8 \
    --lora-rank 32 \
    --lr 2e-5 \
    --max-steps 100 \
    --val-freq 10 \
    --save-freq 20 \
    --output /data/rllm-checkpoints/gsm8k-1.5b-grpo
```

7B模型配置 (RTX 4090显存更紧张):

```bash
rllm train hendrycks_math \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --group-size 4 \
    --batch-size 4 \
    --lora-rank 16 \
    --lr 1e-5 \
    --max-steps 50 \
    --output /data/rllm-checkpoints/math-7b-grpo
```

### CLI 参数详解

| 参数 | 默认值 | RTX 4090推荐 | 说明 |
|------|--------|-------------|------|
| `--model` | Qwen/Qwen3-8B | Qwen2.5-Math-1.5B/7B | 基座模型路径 |
| `--group-size` | 8 | 4-8 | GRPO每问题rollout数 |
| `--batch-size` | 32 | 4-8 | 训练batch大小 |
| `--lora-rank` | 32 | 16-32 | LoRA秩 |
| `--lr` | 2e-5 | 1e-5~2e-5 | 学习率 |
| `--epochs` | 1 | 1 | 训练轮数 |
| `--max-steps` | None | 50-100 | 最大步数(覆盖epochs) |
| `--val-freq` | 5 | 10 | 验证频率 |
| `--save-freq` | 20 | 20 | checkpoint保存频率 |
| `--output` | None | 显式设置 | checkpoint目录 |

### 关键: --config 自定义YAML覆盖

```bash
# 用自定义YAML覆盖任何参数
rllm train gsm8k \
    --config my_config.yaml \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --group-size 4
```

`my_config.yaml` 示例:

```yaml
rllm:
  algorithm:
    adv_estimator: grpo
    norm_adv_by_std_in_grpo: false    # Dr.GRPO: 不除std
    rollout_correction:
      bypass_mode: true               # ★★★ 省forward pass
    kl_beta: 0.001                    # KL惩罚系数
  trainer:
    logger: ['console', 'wandb']
    project_name: rllm-4090-grpo
training:
  fuse_forward_backward_and_optim_step: true  # ★ 融合fwd+bwd+optim → 减少GPU通信
```

---

## 5. 训练启动 — Python API 方式

适合需要自定义workflow或reward的场景.

### 5.1 最简训练脚本 (Math GRPO)

```python
"""train_math_grpo.py — MATH GRPO训练"""
import hydra
from omegaconf import DictConfig
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("hendrycks_math", "train")
    val_dataset = DatasetRegistry.load_dataset("math500", "test")

    trainer = AgentTrainer(
        backend="tinker",
        agent_flow=math_flow,           # from cookbooks/math
        evaluator=math_evaluator,        # from cookbooks/math
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    try:
        trainer.train()
    finally:
        trainer.shutdown()

if __name__ == "__main__":
    main()
```

运行:

```bash
python train_math_grpo.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen2.5-Math-1.5B-Instruct \
    model.lora_rank=32 \
    training.group_size=4 \
    data.train_batch_size=8 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.rollout_correction.bypass_mode=true
```

### 5.2 自定义 Reward 函数训练

```python
"""train_custom_reward.py — 自定义reward训练"""
import hydra
from omegaconf import DictConfig
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.trainer import AgentTrainer
from rllm.workflows.simple_workflow import SimpleWorkflow

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    val_dataset = DatasetRegistry.load_dataset("countdown", "test")

    trainer = AgentTrainer(
        backend="tinker",
        workflow_class=SimpleWorkflow,
        workflow_args={"reward_function": countdown_reward_fn},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    try:
        trainer.train()
    finally:
        trainer.shutdown()

if __name__ == "__main__":
    main()
```

### 5.3 Verifiers 环境训练

```bash
# 安装verifiers
pip install verifiers

# 使用verifiers环境训练
bash examples/verifiers_env/train_verifiers_tinker.sh
```

关键参数:

```bash
python3 -m examples.verifiers_env.train \
    --config-name=tinker_rl_trainer \
    +backend=tinker \
    model.name="Qwen/Qwen3-4B-Instruct-2507" \
    model.lora_rank=32 \
    +verifiers.env_id="primeintellect/alphabet-sort" \
    algorithm.adv_estimator=grpo \
    training.group_size=16 \
    training.learning_rate=2e-5 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    data.train_batch_size=64 \
    trainer.total_epochs=10
```

---

## 6. 完整配置参考

★★★ 两个YAML模板控制所有配置:

### 6.1 base.yaml (后端无关, 核心算法配置)

关键字段:

```yaml
rllm:
  algorithm:
    adv_estimator: grpo              # ★ GRPO优势估计
    norm_adv_by_std_in_grpo: true    # ★ 除std (false=Dr.GRPO)
    kl_beta: 0.0                     # KL惩罚系数
    eps_clip: 0.2                    # PPO clip epsilon
    rollout_correction:
      bypass_mode: null              # ★ null → tinker默认true
      tis_mode: null                 # TIS重要性采样
      tis_cap: 2.0                   # TIS权重上限
  rollout:
    n: 8                             # 每问题rollout数
    n_val: 1                         # 验证rollout数
    train:
      temperature: 1.0               # ★ 必须1.0 (Tinker要求)
      top_p: 1.0                     # ★ 必须1.0 (Tinker要求)
    val:
      temperature: 1.0
      top_p: 1.0
  trainer:
    total_epochs: 10
    logger: ['console']
    project_name: 'rllm-training'
    test_freq: 5
    save_freq: 20
  data:
    train_batch_size: 64
    max_prompt_length: 2048
    max_response_length: 30720
```

### 6.2 tinker.yaml (Tinker后端专用)

关键字段:

```yaml
tinker_base_url: null                # null = 本地Tinker服务
fuse_forward_backward_and_optim_step: false  # ★ true = 融合加速

model:
  name: "Qwen/Qwen3-8B"
  lora_rank: 32                      # ★ LoRA秩
  train_unembed: true                # 训练输出嵌入层LoRA
  train_attn: true                   # 训练注意力层LoRA
  train_mlp: true                    # 训练MLP层LoRA

training:
  group_size: ???                    # ★ 必须CLI指定
  learning_rate: 2e-5
  lr_schedule: "constant"
  max_length: 32768
  num_minibatches: 1                 # ★ 只支持1
  default_local_dir: 'checkpoints/${project}/${experiment}'
  resume_mode: auto                  # auto/disable/resume_path

rllm:
  backend: tinker
  algorithm:
    rollout_correction:
      bypass_mode: true              # ★★★ Tinker默认=true
```

### 6.3 RTX 4090 专用配置模板

★★★ 创建文件 `rtx4090_grpo.yaml`:

```yaml
# RTX 4090 单GPU GRPO训练配置
# 适用模型: Qwen2.5-Math-1.5B / Qwen3-4B / Qwen2.5-Math-7B (LoRA-16)

model:
  name: "Qwen/Qwen2.5-Math-1.5B-Instruct"
  lora_rank: 32
  train_unembed: true
  train_attn: true
  train_mlp: true

training:
  group_size: 4                      # ★ 4 rollouts (RTX 4090显存极限)
  learning_rate: 2e-5
  lr_schedule: "constant"
  max_length: 4096                   # ★ 缩短seq_len → 省显存
  num_minibatches: 1

data:
  train_batch_size: 8                # ★ 小batch → 单GPU可行
  val_batch_size: 32
  max_prompt_length: 2048
  max_response_length: 2048          # ★ 缩短 → 省显存

rllm:
  algorithm:
    adv_estimator: grpo
    norm_adv_by_std_in_grpo: true
    rollout_correction:
      bypass_mode: true              # ★★★ 省forward pass
    kl_beta: 0.001                   # 轻度KL惩罚
  rollout:
    n: 4
    n_val: 1
    train:
      temperature: 1.0               # ★ Tinker要求1.0
      top_p: 1.0
  trainer:
    total_epochs: 1
    total_batches: 100               # ★ 用total_batches控制步数
    logger: ['console', 'wandb']
    project_name: 'rtx4090-grpo'
    experiment_name: 'math-1.5b'
    test_freq: 10
    save_freq: 20

rollout_engine:
  reasoning_effort: "low"            # ★ 低reasoning → 短输出 → 省显存
  disable_thinking: false
```

---

## 7. Reward 函数详解

### 7.1 内置 Reward 函数

★★★ rLLM提供5种内置reward函数:

| 函数 | 位置 | 用途 | 评分方式 |
|------|------|------|----------|
| `math_reward_fn` | `rllm/rewards/reward_fn.py` | 数学问题 | sympy+mathd等价判定 |
| `code_reward_fn` | `rllm/rewards/reward_fn.py` | 编程问题 | 执行测试 |
| `countdown_reward_fn` | `rllm/rewards/countdown_reward.py` | Countdown游戏 | 算式验证+结果验证 |
| `search_reward_fn` | `rllm/rewards/reward_fn.py` | 搜索问答 | F1+recall |
| `f1_reward_fn` | `rllm/rewards/reward_fn.py` | 文本匹配 | token F1 |

### 7.2 math_reward_fn — 最常用

★★★ 数学reward是GRPO最常用的, 完全基于规则, 无GPU需求:

工作流:
1. 提取 `\boxed{ANSWER}` 格式答案
2. 没有 `\boxed{}` → format_error_reward=0 (可配置为部分分)
3. 用 `grade_answer_mathd()` (数值等价) + `grade_answer_sympy()` (符号等价) 验证
4. 正确 → reward=1.0, 错误 → reward=0.0

```python
# RewardConfig 默认值
correct_reward: float = 1.0       # 正确奖励
incorrect_reward: float = 0.0     # 错误奖励
format_error_reward: float = 0.0  # 格式错误奖励
unk_error_reward: float = 0.0     # 未知错误奖励
toolcall_bonus: float = 0.5       # 使用工具的额外奖励
apply_format_reward: bool = False # 是否给格式正确但答案错误的半分
```

关键: `0.5` 匹配 `\frac{1}{2}`, `\sqrt{4}` 匹配 `2` — sympy覆盖大部分数学等价.

### 7.3 自定义 Reward 函数

```python
from rllm.rewards.reward_types import RewardOutput

def my_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    自定义reward函数协议.
    task_info: 包含question/ground_truth的数据字典
    action: 模型的输出字符串
    返回: RewardOutput(reward=float, is_correct=bool, metadata=dict)
    """
    ground_truth = task_info.get("ground_truth", "")
    # 你的评分逻辑
    is_correct = (action.strip() == ground_truth.strip())
    reward = 1.0 if is_correct else 0.0
    return RewardOutput(reward=reward, is_correct=is_correct)

# 使用:
trainer = AgentTrainer(
    backend="tinker",
    workflow_class=SimpleWorkflow,
    workflow_args={"reward_function": my_reward_fn},
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)
```

★★ RTX 4090只能用rule-based reward — 没有GPU空间给RM模型. math_reward_fn/code_reward_fn完全CPU计算, 对RTX 4090是唯一选择.

---

## 8. bypass_mode 与优势计算

### 8.1 bypass_mode — ★★★ RTX 4090最重要的优化

bypass_mode=true 意味: **用rollout时的logprobs作为pi_old, 不需要额外的forward pass计算pi_old**

PPO标准流程: pi_old = policy.forward(trajectory) → 需要一个forward pass
bypass_mode: pi_old = rollout_logprobs → 零额外forward pass

★★★ 这节省了:
- 50% forward pass计算 (GRPO只需要1次fwd+bwd, 不需要2次fwd)
- 显存 (不需要存储两份logprobs)
- 时间 (每个step省1个forward pass)

Tinker后端默认 bypass_mode=true (在tinker.yaml中):
```yaml
rllm:
  algorithm:
    rollout_correction:
      bypass_mode: true    # ★★★ 默认true → 自动省forward
```

无需手动设置, 已是默认行为. 但如果显式设置为false, 会增加约50%的计算开销.

### 8.2 GRPO 优势计算

GRPO公式: `advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)`

rLLM中GRPO默认映射到Tinker的 `ppo` loss函数:
```python
# tinker_policy_trainer.py
ADV_TO_LOSS_FN_AUTO_MAP = {
    rLLMAdvantageEstimator.GRPO: "ppo",    # ★★★ GRPO→PPO loss
    rLLMAdvantageEstimator.REINFORCE: "importance_sampling",
    rLLMAdvantageEstimator.RLOO: "importance_sampling",
}
```

### 8.3 Dr.GRPO — 不除std

★★ Dr.GRPO论文发现: 小group中std可能接近0 → 除std导致梯度消失

设置:
```bash
# Dr.GRPO: 不除std
rllm.algorithm.norm_adv_by_std_in_grpo=false
```

或YAML:
```yaml
rllm:
  algorithm:
    norm_adv_by_std_in_grpo: false    # Dr.GRPO
```

### 8.4 TIS 重要性采样 (跨步staleness纠正)

仅用于async训练 (RTX 4090单GPU不适用). 如需了解:

```yaml
rllm:
  algorithm:
    rollout_correction:
      tis_mode: "token"      # token-level IS
      tis_cap: 2.0           # IS权重上限clamp
```

---

## 9. LoRA 配置详解

### 9.1 LoRA 自动初始化

★★★ TinkerBackend自动创建LoRA训练client, 无需手动代码:

```python
# TinkerPolicyTrainer.initialize_async() 自动执行:
self.training_client = await self.service_client.create_lora_training_client_async(
    base_model=self.config.model.name,     # "Qwen/Qwen2.5-Math-7B-Instruct"
    rank=self.config.model.lora_rank,      # 32
    train_unembed=train_unembed,           # true → 输出嵌入层也训练
    train_attn=train_attn,                 # true → 注意力层
    train_mlp=train_mlp,                   # true → MLP层
)
```

★★ 关键: LoRA秩越大 → 训练参数越多 → 显存需求越大
- rank=16: ~64MB LoRA参数 (7B模型)
- rank=32: ~128MB LoRA参数 (7B模型)
- rank=64: ~256MB LoRA参数 (7B模型)

### 9.2 LoRA层选择

| 配置 | 说明 | RTX 4090推荐 |
|------|------|-------------|
| `train_attn=true` | 训练Q/K/V/O投影LoRA | ★ true |
| `train_mlp=true` | 训练gate/up/down投影LoRA | ★ true |
| `train_unembed=true` | 训练输出嵌入层LoRA | ★ true (merge后推理同效) |

★★ `train_unembed=false` 仅在Fireworks部署时需要 (兼容性限制). 本地部署保持true.

### 9.3 Ref Model — LoRA巧妙绕过

★★★ GRPO需要ref model计算KL: pi_ref vs pi_old

TinkerBackend解决方案:
- `ref_in_actor=true` → 同一个training client
- 推理时: `no_lora_adapter=True` → `disable_adapter()` → base model logprobs
- ★★★ 不需要额外GPU! base model就是ref model (LoRA只是adapter, 关掉就是原模型)

```python
# 权重同步: rollout采样client
sampler_future = await self.training_client.save_weights_for_sampler_async(name="000000")
# → GPU-only merge → 新SamplingClient → 零拷贝替换rollout engine的采样client
```

---

## 10. RTX 4090 显存验证

### 10.1 显存估算

★★★ 7B模型 LoRA-32 GRPO (group_size=4):

| 组成 | 显存 | 备注 |
|------|------|------|
| 模型权重 (BF16) | ~14GB | 7B × 2bytes |
| LoRA参数 | ~128MB | rank=32 × attn+mlp+unembed |
| KV cache (rollout) | ~2-4GB | 4×rollout × seq_len |
| Optimizer state | ~256MB | Adam (LoRA only) |
| Activations | ~1-2GB | fwd+bwd临时 |
| 总计 | ~17-18GB | ★ RTX 4090 24GB可行! |

4B模型 LoRA-32 group_size=8:

| 组成 | 显存 | 备注 |
|------|------|------|
| 模型权重 (BF16) | ~8GB | 4B × 2bytes |
| LoRA参数 | ~64MB | rank=32 |
| KV cache (rollout) | ~3-6GB | 8×rollout |
| Optimizer state | ~128MB | Adam |
| Activations | ~1GB | |
| 总计 | ~12-14GB | ★★★ 更安全 |

### 10.2 安全配置

★★ RTX 4090 安全配置建议:

| 模型 | group_size | batch_size | lora_rank | max_length | 备注 |
|------|-----------|-----------|-----------|------------|------|
| 1.5B | 8 | 16 | 32 | 8192 | ★★★ 最安全 |
| 4B | 4-8 | 8-16 | 32 | 4096 | ★ 安全 |
| 7B | 4 | 4-8 | 16 | 4096 | ★ 边界 |
| 7B | 8 | 32 | 32 | 32768 | ❌ OOM风险 |

### 10.3 OOM急救

```bash
# 减group_size
training.group_size=2                # 最小可行值

# 减batch_size
data.train_batch_size=4

# 减LoRA rank
model.lora_rank=8

# 缩短max_length
training.max_length=2048

# 缩短response
data.max_response_length=1024

# 降低reasoning_effort
rollout_engine.reasoning_effort="low"
```

---

## 11. 训练监控

### 11.1 Console 输出

默认 `logger=['console']` — 每步输出:

```
Step 10/100 | reward_mean=0.35 | reward_std=0.21 | advantage_mean=0.12 | loss=0.08
```

### 11.2 wandb 监控

```bash
# 启用wandb
rllm train gsm8k \
    --config rtx4090_grpo.yaml \
    +rllm.trainer.logger="['console', 'wandb']" \
    +rllm.trainer.project_name=rtx4090-grpo \
    +rllm.trainer.experiment_name=gsm8k-1.5b

# 或Python API
trainer = AgentTrainer(
    ...
    config=config,  # config中设置logger=['console','wandb']
)
```

### 11.3 rLLM UI 监控

★★★ rLLM提供实时Web UI监控:

```bash
# 登录
rllm login

# 训练时启用UI
rllm train gsm8k --ui

# 或在配置中
rllm.trainer.logger="['console', 'ui']"
```

UI地址: https://ui.rllm-project.com

### 11.4 关键指标

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `reward_mean` | 平均reward | ↑ 上升 |
| `reward_std` | reward标准差 | — 稳定 |
| `advantage_mean` | 平均优势 | ↑ 接近0是坏的 |
| `train/loss` | 训练loss | ↓ 下降 |
| `train/kl` | KL散度 | — 不能太大 |
| `val/reward_mean` | 验证reward | ↑ 上升 |
| `batch/steps_per_traj` | 每轨迹步数 | — =1是健康的 |
| `batch/action_token_ratio` | action token比例 | — =1是单步 |

---

## 12. Checkpoint 管理

### 12.1 自动Checkpoint

TinkerBackend每 `save_freq` 步保存checkpoint:

```bash
# 目录结构
checkpoints/rtx4090-grpo/gsm8k-1.5b/
├── global_step_000020/
│   ├── checkpoint.json              # state_path + sampler_path + dataloader_state
│   └── (Tinker内部: weights + sampler_weights)
├── global_step_000040/
│   └── checkpoint.json
├── latest_checkpointed_iteration.txt  # 内容: "40"
```

### 12.2 Resume训练

★★ 三种resume模式:

```bash
# auto模式: 从最新checkpoint自动恢复
training.resume_mode=auto              # 默认

# disable模式: 从头开始训练
training.resume_mode=disable

# resume_path模式: 指定checkpoint路径
training.resume_mode=resume_path
training.resume_from_path=checkpoints/.../global_step_000040
```

### 12.3 权重同步机制

★★★ TinkerBackend的权重同步是最高效的:

```python
# 每步执行:
# 1. forward_backward → 计算梯度
# 2. optim_step → 更新LoRA权重 (GPU-only)
# 3. save_weights_for_sampler → GPU-only merge → 新SamplingClient
# 4. rollout_engine.set_sampling_client → 零拷贝替换
```

关键: 全程GPU操作, 无CPU round-trip, 无跨进程通信.

---

## 13. pass@k 评估

### 13.1 eval 基础用法

```bash
# 基础评估 (pass@1)
rllm eval math500 \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --base-url http://localhost:8000/v1

# 评估自定义checkpoint
rllm eval math500 \
    --agent math \
    --evaluator math \
    --model "tinker://uuid/weights/000060" \
    --base-url http://localhost:8000/v1
```

### 13.2 ★★★ pass@k 评估

★★★ `--attempts N` 执行N次独立rollout, 计算pass@k for k=1..N:

```bash
# pass@8 评估 (GRPO最佳对齐)
rllm eval aime2024 \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --base-url http://localhost:8000/v1 \
    --attempts 8

# pass@16 评估 (更高k)
rllm eval aime2024 \
    --attempts 16 \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --base-url http://localhost:8000/v1
```

输出示例:
```
Results:
  Accuracy:  35.0% (21/60)
  Errors:    0
  pass@1:    35.0%
  pass@2:    52.3%
  pass@4:    71.5%
  pass@8:    84.2%
```

★★★ pass@k公式 (Chen et al. 2021 unbiased estimator):
`pass@k = 1 - C(n-c, k) / C(n, k)` — n=attempts, c=正确数

### 13.3 本地模型评估

需要先启动vLLM/SGLang推理服务器:

```bash
# 启动vLLM (RTX 4090)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --port 8000

# 然后评估
rllm eval math500 \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --agent math \
    --evaluator math \
    --attempts 8
```

### 13.4 快速测试评估

```bash
# 限制20题快速测试
rllm eval math500 \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --base-url http://localhost:8000/v1 \
    --max-examples 20
```

---

## 14. LoRA 合并 + INT4 量化部署

### 14.1 LoRA 合并

★★ TinkerBackend训练产出的是LoRA adapter, 需要合并到base model才能独立部署:

```python
"""merge_lora.py — 合并LoRA到base model"""
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 方法1: 使用PEFT自动合并
model = AutoPeftModelForCausalLM.from_pretrained(
    "checkpoints/.../global_step_000100",  # LoRA checkpoint路径
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model/")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct").save_pretrained("merged_model/")

# 方法2: 手动合并 (如果PEFT格式不兼容)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
peft_model = PeftModel.from_pretrained(base_model, "checkpoints/.../global_step_000100")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model/")
```

★★★ 注意: Tinker的checkpoint格式是 `checkpoint.json + sampler_weights`, 不是标准PEFT格式. 合并可能需要:

```python
# Tinker内部合并 (推荐)
# training_client.save_weights_for_sampler 已经做了GPU merge
# SamplingClient使用的是合并后的权重
# 但导出到HF格式需要额外步骤

# 查看checkpoint内容
import json
with open("checkpoints/.../global_step_000100/checkpoint.json") as f:
    ckpt = json.load(f)
    print(ckpt["state_path"])      # Tinker内部权重路径
    print(ckpt["sampler_path"])    # Tinker采样器路径
```

★★★ 实际导出流程 (2026-06当前状态):

Tinker的checkpoint存储在Tinker服务内部, 需要通过Tinker API导出为HF格式:

```python
# 通过training_client导出 (如果Tinker支持)
await self.training_client.save_state_async(name="final")
# 这保存了完整的训练状态 (base + LoRA)

# 导出为独立HF格式 — 可能需要:
# 1. 从Tinker的state_path加载权重
# 2. 用PEFT merge_and_unload合并
# 3. 保存为HF格式
```

★★ 截至当前, Tinker的checkpoint导出到标准HF格式的具体流程可能需要查阅tinker-cookbook文档. 最简单方案:

1. 训练完成后用 `rllm eval` 直接评估 (Tinker ServingClient推理)
2. 或部署到Tinker云服务继续推理
3. 如需导出到vLLM → 需要PEFT merge → 具体步骤见下

### 14.2 ★★★ INT4 量化部署 (vLLM)

★★★ RTX 4090推理最优路径: INT4 + INT8KV + GQA-8 → 4,791 tok/s

量化流程:

```bash
# Step 1: 合并LoRA到base model (见14.1)
# 得到 merged_model/ 目录 (BF16完整模型)

# Step 2: INT4量化 (AutoGPTQ 或 vLLM内置)
# 方法A: AutoGPTQ量化
python -m auto_gptq \
    --model_name merged_model/ \
    --bits 4 \
    --group_size 128 \
    --output_dir merged_model-int4/

# 方法B: vLLM直接加载 (如果模型已有INT4版本)
# vLLM v0.23+ 支持 INT4 Triton fallback → RTX 4090更多模型

# Step 3: 启动vLLM INT4推理
python -m vllm.entrypoints.openai.api_server \
    --model merged_model-int4/ \
    --quantization gptq_int4 \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype int8 \
    --max-model-len 4096 \
    --port 8000
```

★★★ RTX 4090 INT4推理预估:

| 配置 | 吞吐 | 备注 |
|------|------|------|
| BF16 7B | ~550 tok/s | 基线 |
| INT4 7B + INT8KV | ~4,791 tok/s | ★★★ 最优 |
| INT4 7B + EAGLE | ~9,088 tok/s | ★★★★ speculative |
| INT4 1.5B + INT8KV | ~8,000+ tok/s | 小模型更快 |

---

## 15. 完整训练流程 (Step-by-Step)

★★★ 从零到部署的完整流程:

### Step 1: 环境准备

```bash
# GPU服务器
conda create -n rllm-tinker python=3.11 -y
conda activate rllm-tinker
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

git clone https://github.com/rllm-org/rLLM.git
cd RLLM
pip install -e ".[tinker]"
pip install -e ".[rewards]"
pip install --no-deps -e cookbooks/math

# 验证
rllm --version
rllm agent list
```

### Step 2: 数据准备

```bash
# 拉取数据集
rllm dataset pull gsm8k
rllm dataset pull math500
rllm dataset inspect gsm8k -n 3   # 检查数据
```

### Step 3: 训练

```bash
# ★★★ 一行命令启动GRPO训练
rllm train gsm8k \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --group-size 4 \
    --batch-size 8 \
    --lora-rank 32 \
    --lr 2e-5 \
    --max-steps 100 \
    --val-freq 10 \
    --save-freq 20 \
    --output /data/rllm-checkpoints/gsm8k-1.5b
```

### Step 4: 监控

```bash
# Console日志自动输出
# wandb: 在配置中添加 logger=['console','wandb']
# UI: rllm train gsm8k --ui
```

### Step 5: 评估

```bash
# ★★★ pass@k评估 (与GRPO group_size对齐)
rllm eval math500 \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --base-url http://localhost:8000/v1 \
    --attempts 4     # 与group_size对齐
```

### Step 6: Checkpoint管理

```bash
# Resume训练
rllm train gsm8k \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --output /data/rllm-checkpoints/gsm8k-1.5b \
    # + training.resume_mode=auto (自动从最新checkpoint恢复)
```

### Step 7: LoRA合并 + 部署

```python
# 合并LoRA到base model
# (见14.1详细流程)
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained("checkpoint_path")
merged = model.merge_and_unload()
merged.save_pretrained("merged_model/")
```

```bash
# INT4量化 + vLLM部署
# (见14.2详细流程)
python -m vllm.entrypoints.openai.api_server \
    --model merged_model-int4/ \
    --quantization gptq_int4 \
    --kv-cache-dtype int8 \
    --port 8000
```

---

## 16. 常见问题与排错

### Q: Python版本错误

Tinker需要Python >= 3.11:
```bash
conda create -n rllm-tinker python=3.11
```

### Q: Sampling参数警告

★★ Tinker要求 temperature=1.0 和 top_p=1.0:
```bash
# 错误: temperature=0.6 → 会导致logprob不一致
# 正确: rllm.rollout.train.temperature=1.0
```
详见: https://github.com/thinking-machines-lab/tinker-cookbook/pull/86

### Q: OOM (显存不足)

★★★ 按优先级降低参数:
1. 减 `group_size` → 4 → 2 (最有效)
2. 减 `batch_size` → 8 → 4
3. 减 `lora_rank` → 16 → 8
4. 减 `max_length` → 4096 → 2048
5. 减 `max_response_length` → 2048 → 1024
6. 设 `rollout_engine.reasoning_effort=low`

### Q: num_minibatches只能设1

TinkerBackend目前只支持 `num_minibatches=1`, 设其他值会触发警告.

### Q: Checkpoint找不到

确保checkpoint目录存在:
```bash
mkdir -p /data/rllm-checkpoints/gsm8k-1.5b
```

### Q: 数据集加载失败

```bash
# 检查数据集是否存在
rllm dataset list --all
rllm dataset inspect gsm8k

# 重新拉取
rllm dataset pull gsm8k
```

### Q: LoRA合并导出失败

★★ Tinker的checkpoint格式不是标准PEFT格式. 当前可能需要:
1. 通过Tinker API导出为HF格式
2. 或使用 `tinker-cookbook` 的导出工具
3. 详见: https://github.com/thinking-machines-lab/tinker-cookbook

### Q: router_replay不支持

TinkerBackend不支持 `router_replay` (MoE路由回放). 设为 `disabled` 即可.

---

## 附录A: 训练数据流全景

```
rllm train gsm8k --agent math --evaluator math --model Qwen/...
│
├── CLI (rllm/cli/train.py)
│   ├── build_train_config() → OmegaConf合并base.yaml+tinker.yaml+CLI覆盖
│   ├── resolve catalog → agent_flow + evaluator
│   ├── pull dataset → DatasetRegistry
│   └── AgentTrainer(backend="tinker", ...) → trainer.train()
│
├── UnifiedTrainer.fit_async()
│   ├── backend.on_train_start → TinkerPolicyTrainer.initialize_async()
│   │   └── create_lora_training_client_async(base_model, rank=32, ...)
│   │   └── save_weights_for_sampler → SamplingClient
│   │
│   ├── for each batch:
│   │   ├── Stage 1: generate_episodes
│   │   │   └── interleave_tasks(batch, group_size=4) → 4×每个问题
│   │   │   └── agent_workflow_engine.execute_tasks → Episodes
│   │   │
│   │   ├── Stage 2: transform_episodes_to_trajectory_groups
│   │   │   └── 每个group = 1个问题 × 4个trajectory
│   │   │   └── reward = math_reward_fn(task, action) → scalar
│   │   │
│   │   ├── Stage 3: rejection_sampling (optional)
│   │   │
│   │   ├── Stage 4: transform_to_backend_batch → Tinker Datum
│   │   │   └── trajectory_to_datums → right-shifted input + left-shifted target
│   │   │   └── mask=[0,0,...,0,1,1,...,1] → 只有action tokens参与loss
│   │   │
│   │   ├── Stage 5: process_backend_batch → forward_backward
│   │   │   └── ★★★ bypass_mode=true: pi_old=rollout_logprobs → 省forward
│   │   │   └── ★★★ GRPO→ppo loss: clip(pi/pi_old, 0.2) × advantage × mask
│   │   │
│   │   ├── Stage 6: compute_advantages → GRPO formula
│   │   │   └── advantage = (reward - group_mean) / group_std
│   │   │
│   │   ├── Stage 7: update_policy → optim_step
│   │   │   └── Adam(LoRA params only, lr=2e-5)
│   │   │
│   │   └── Stage 8: on_batch_end → save_checkpoint + new SamplingClient
│   │       └── ★★★ GPU-only merge → 零拷贝权重同步
│   │
│   └── backend.on_train_end → 保存最终checkpoint
```

## 附录B: Tinker Loss 函数映射

| 优势估计器 | 默认Tinker loss | 可override |
|-----------|----------------|-----------|
| GRPO | `ppo` | importance_sampling / cispo / dro / cross_entropy |
| REINFORCE | `importance_sampling` | ppo / cispo / dro |
| REINFORCE++BL | `importance_sampling` | ppo / cispo / dro |
| RLOO | `importance_sampling` | ppo / cispo / dro |
| OTHER | `importance_sampling` | 任何 |

Override方式:
```bash
rllm.algorithm.loss_fn=importance_sampling   # 覆盖默认
```

## 附录C: 配置层级优先级

```
1. CLI显式覆盖 (--group-size=4)     → 最高优先
2. --config YAML覆盖                 → 第二优先
3. rllm.* YAML (base.yaml)           → 第三优先
4. Tinker-native YAML (tinker.yaml)  → 第四优先
5. Hydra defaults                     → 最低优先
```

★★★ 关键: `rllm.*` > `Tinker-native` > `defaults`

## 附录D: RTX 4090 vs 其他GPU对比

| GPU | 可行配置 | 吞吐预估 |
|-----|---------|---------|
| RTX 4090 (24GB) | 1.5B-7B LoRA, group=4-8 | ★★★ 单GPU最优 |
| A100 (80GB) | 7B-13B LoRA, group=8-16 | 更大模型 |
| H100 (80GB) | 7B-70B LoRA, group=16+ | ★★★★ NVLink+SM90 |
| RTX 3090 (24GB) | 1.5B-4B LoRA, group=4 | SM86, 略慢 |

★★★ RTX 4090核心限制: PCIe无法多GPU → 只能单GPU → 只能LoRA → 只能rule-based reward → 但TinkerBackend恰好是最优适配!

---

## 附录E: 关键源码位置

| 文件 | 位置 | 内容 |
|------|------|------|
| TinkerBackend | `rllm/trainer/tinker/tinker_backend.py` | 后端实现 |
| TinkerPolicyTrainer | `rllm/trainer/tinker/tinker_policy_trainer.py` | LoRA+训练+checkpoint |
| Tinker transform | `rllm/trainer/tinker/transform.py` | Trajectory→Datum转换 |
| base.yaml | `rllm/trainer/config/rllm/base.yaml` | 核心配置 |
| tinker.yaml | `rllm/trainer/config/rllm/backend/tinker.yaml` | Tinker配置 |
| math_reward_fn | `rllm/rewards/reward_fn.py` → `rllm/rewards/math_reward.py` | 数学reward |
| countdown_reward | `rllm/rewards/countdown_reward.py` | Countdown reward |
| DatasetRegistry | `rllm/data/dataset.py` | 数据管理 |
| AgentTrainer | `rllm/trainer/unified_trainer.py` → `AgentTrainer` class at line 946 | 训练入口 |
| UnifiedTrainer | `rllm/trainer/unified_trainer.py` | 8阶段训练pipeline |
| CLI train | `rllm/cli/train.py` | `rllm train` 命令 |
| CLI eval | `rllm/cli/eval.py` | `rllm eval` 命令 |
| AlgorithmConfig | `rllm/trainer/algorithms/config.py` | 算法配置 |
| EvalResult | `rllm/eval/results.py` | pass@k计算 |
| SimpleWorkflow | `rllm/workflows/simple_workflow.py` | 单步workflow |

---

> 最后更新: 2026-06-15 | 基于rLLM v0.3.0-pre (commit main) | RTX 4090 8×RTX4090 SM89
