# verl Reward Integration 源码级深度阅读

> 2026-06-15 | 源码: verl/trainer/ppo/reward.py + ray_trainer.py + workers/reward_manager/ + experimental/reward_loop/ + core_algos.py + utils/reward_score/
> 核心: 两层架构(legacy RewardManager + experimental RewardLoop) → 3 reward路径(rule/DisRM/GenRM) → outcome reward=最后token非零 → GRPO group-relative → colocate RM sleep/wake → RTX 4090=rule-based唯一可行

## 1. 系统架构概览: 两层奖励系统 ★★

verl的奖励系统经历了重大架构演进,目前存在**两套并行系统**:

```
┌──────────────────────────────────────────────────────────┐
│                  Ray PPO Trainer (ray_trainer.py)         │
│                                                          │
│  1. 生成rollout → gen_batch_output                        │
│  2. 计算reward → _compute_reward_colocate OR extract_reward│
│  3. KL-in-reward → apply_kl_penalty (可选)                │
│  4. 计算advantage → compute_advantage (GRPO/GRPO_VEC/...) │
│  5. 更新actor → _update_actor                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Legacy Layer (verl/workers/reward_manager/)              │
│  ├── AbstractRewardManager (__call__ interface)           │
│  ├── NaiveRewardManager   (loop per-item)                │
│  ├── BatchRewardManager   (batch scoring)                │
│  ├── DAPORewardManager    (overlong penalty)              │
│  ├── PrimeRewardManager   (async ProcessPoolExecutor)    │
│  └ Registry: @register decorator                         │
│                                                          │
│  ★ Experimental Layer (verl/experimental/reward_loop/)    │
│  ├── RewardLoopManager → distributed worker orchestration │
│  ├── RewardLoopWorker   → async compute_score (3 paths)  │
│  ├── RewardManagerBase  → async run_single interface     │
│  ├── RewardModelManager → RM server lifecycle(sleep/wake) │
│  ├── NaiveRewardManager  (async, event loop executor)    │
│  ├── RemoteRewardManager (Ray actor, separate process)   │
│  ├── RateLimitedRewardManager (3-layer rate limiting)    │
│  ├── GDPORewardManager   (per-dimension reward)          │
│  └ Registry: @register decorator (separate from legacy)  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**★ 关键区别**:
- Legacy: `__call__(data) → reward_tensor` (同步, 直接返回tensor)
- Experimental: `async run_single(data) → {"reward_score", "reward_extra_info"}` (异步, 返回dict)
- Experimental是**默认路径**: ray_trainer.py 通过 `load_reward_manager()` 加载experimental RewardManager
- 两套registry互不干扰: legacy用`REWARD_MANAGER_REGISTRY`, experimental用`REWARD_MANAGER`
- ★ RateLimitedRewardManager **同时注册到两套registry** (`@register_manager("rate_limited")` + `@register_manager_legacy("rate_limited")`)

## 2. 奖励计算3条路径 ★★★

### Path 1: Rule-Based Reward (custom_reward_function) ★★★

**配置**: `reward.custom_reward_function.path` + `reward.custom_reward_function.name`

**加载流程** (reward.py:50-86):
```python
def get_custom_reward_fn(config):
    module_path = config.reward.custom_reward_function.path  # e.g., "my_reward.py"
    fn_name = config.reward.custom_reward_function.name      # e.g., "compute_score"

    # ★ 动态导入: load_extern_object → importlib动态加载外部文件
    raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)

    # ★ 合入reward_kwargs: partial(_call_with_kwargs, raw_fn, reward_kwargs)
    # → 调用时自动把reward_kwargs合并进函数参数
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)  # sync → wrapped
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)  # async → wrapped async
```

**函数签名** (文档: docs/preparation/reward_function.rst):
```python
# Sync版本
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **reward_kwargs):
    # data_source: 数据集来源标识 → 决定用什么评分逻辑
    # solution_str: 模型生成的文本 (decoded from token IDs)
    # ground_truth: 正确答案 (从parquet文件的reward_model.ground_truth字段)
    # extra_info: 额外信息 (num_turns, rollout_reward_scores等)
    # reward_kwargs: 从config合并的额外参数

    return 1.0  # 返回float → reward = score
    # 或返回dict → {"score": 1.0, "format_reward": 0.5, ...} → GDPO需要!

# Async版本 ★ (推荐用于需要API调用的场景)
async def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                        reward_router_address=None, reward_model_tokenizer=None):
    # ★ reward_router_address: RM HTTP路由地址 → GenRM时使用
    # ★ reward_model_tokenizer: RM的tokenizer → GenRM时使用

    async with aiohttp.ClientSession() as session:
        result = await session.post(reward_router_address + "/v1/chat/completions", ...)
    return {"score": parsed_score}
```

**★★★ 自定义reward函数=RTX 4090唯一可行路径**: 纯CPU执行 → 无GPU开销 → 与rollout/actor共存

### Path 2: Discriminative Reward Model (DisRM)

**配置**: `reward.reward_model.enable=True` + `reward.reward_model.model_path` + 无custom_reward_function

**流程** (reward_loop.py:231-270):
```python
async def compute_score_disrm(self, data):
    # Step 1: 预处理 → tokenizer.apply_chat_template → 构造RM输入
    disrm_prompt = await self._preprocess_reward_inputs(data)
    # → 把prompt+response组成chat格式 → RM判断质量

    # Step 2: HTTP请求 → RM router → RM server推理
    engine_name = config.reward.reward_model.rollout.name
    if engine_name == "vllm":
        payloads = {"model": model_name, "input": disrm_prompt, "use_activation": False}
        output = await self._post_request(payloads, "classify")
        rm_score = output["data"][-1]["probs"][-1]  # ★ 取最后token的prob → 分类概率
    elif engine_name == "sglang":
        payloads = {"model": model_name, "input": disrm_prompt}
        output = await self._post_request(payloads, "v1/embeddings")
        rm_score = output["data"][-1]["embedding"][-1]  # ★ 取embedding最后一维

    return {"reward_score": rm_score}  # 单scalar
```

**DisRM本质**: classifier → 给prompt+response打分 → 输出概率/embedding → 取最后一个维度作为score
**★ 需要额外GPU**: RM模型需要独立GPU运行(vLLM/SGLang server) → RTX 4090不可行

### Path 3: Generative Reward Model (GenRM)

**配置**: `reward.reward_model.enable=True` + `reward.custom_reward_function.path` (必须!)

**为什么GenRM必须指定custom_reward_function** (reward_loop.py:146-155):
```python
async def compute_score(self, data):
    if config.reward.custom_reward_function.path is not None:
        # ★ 直接用user-customized reward function
        return await self.reward_manager.run_single(data)
    else:
        if config.reward.reward_model.enable:
            # ★ we assume the rm is disrm
            # ★ genrm must set custom_reward_function → 否则走DisRM路径!
            return await self.compute_score_disrm(data[-1:])
        else:
            return await self.reward_manager.run_single(data)  # default rule-based
```

**GenRM流程**: custom_reward_function → 构造GenRM prompt → HTTP请求到router → 生成文本 → parse score
**★ GenRM更灵活**: 可以做LLM-as-judge → 但需要自定义prompt模板和parser → 用户完全掌控

### 3条路径决策树 ★★

```
custom_reward_function.path != None?
  └─ Yes → Path 1: reward_manager.run_single(data) → user自定义逻辑
  │        (可能内部调用RM router → GenRM场景)
  └─ No → reward.reward_model.enable?
           └─ Yes → Path 2: compute_score_disrm(data[-1:]) → DisRM classifier
           │        (只取最后一个response序列 → multi-turn时只评最终)
           └─ No → Path 3: reward_manager.run_single(data) → default_compute_score
                    (rule-based → 按data_source分派 → gsm8k/math/code等)
```

## 3. Outcome Reward: 只有最后token非零 ★★★

**★★★ 这是verl GRPO最核心的设计**: 所有RewardManager都把reward放在最后一个valid token位置!

### Legacy RewardManager (naive.py:54-100):

```python
reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
# ★ 所有位置初始化为0!

for i in range(len(data)):
    score = self.compute_score(data_source=..., solution_str=..., ground_truth=..., ...)
    # ★★★ 只在最后一个valid token赋值!
    reward_tensor[i, valid_response_length - 1] = reward  # 不是所有token, 只有最后一个!
```

### Experimental RewardManager (base.py:62-82):

```python
@classmethod
def assemble_rm_scores(cls, data, scores):
    prompt_length = data.batch["prompts"].size(1)
    valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
    rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    # ★★★ 同样只在最后位置赋值!
    rm_scores[torch.arange(rm_scores.size(0)), valid_response_length - 1] = scores
    return rm_scores
```

### 所有RewardManager一致的outcome pattern:

| Manager | 代码位置 | 赋值行 |
|---------|---------|--------|
| NaiveRewardManager (legacy) | naive.py:100 | `reward_tensor[i, valid_response_length - 1] = reward` |
| BatchRewardManager | batch.py:110 | `reward_tensor[i, length - 1] = reward` |
| DAPORewardManager | dapo.py:132 | `reward_tensor[i, valid_response_length - 1] = reward` |
| PrimeRewardManager | prime.py:177 | `reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]` |
| assemble_rm_scores (base) | base.py:79-80 | `rm_scores[..., valid_response_length - 1] = scores` |
| RateLimitedRewardManager | limited.py:527 | `reward_tensor[i, valid_response_length - 1] = reward` |

**★★★ 统一结论**: reward_tensor形状=[bs, response_length], 只有`reward_tensor[i, last_valid-1]`非零!

### 如何流入GRPO advantage ★★★

```python
# core_algos.py:304 → GRPO advantage第一步
scores = token_level_rewards.sum(dim=-1)
# ★ 因为只有最后token非零 → sum(dim=-1) = 最后token的值 = scalar reward!

# 然后group-relative:
id2score[index[i]].append(scores[i])  # 按prompt分组
a_i = (r_i - μ_g) / (σ_g + ε)        # group内标准化
advantages = scalars.unsqueeze(-1) * response_mask  # broadcast到所有token
```

**Outcome reward与GRPO的完美对齐**:
- Outcome reward = 单scalar → sum(dim=-1) = scalar → GRPO直接用
- GRPO不需要per-token reward → 只需要group内的相对排名
- ★★ 如果用process reward (per-token nonzero), GRPO仍然会sum → 语义变了!
- ★★ 但verl的GRPO advantage是outcome-only → 只适合outcome reward!

## 4. GRPO Advantage: 奖励塑造与归一化 ★★★

### 4.1 GRPO (loop-based) — core_algos.py:268-331

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    scores = token_level_rewards.sum(dim=-1)  # outcome → scalar per response

    # ★ 按index(prompt ID)分组 → rollout_n=8时, 同prompt8个response形成1个group
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # ★★★ Group normalization: μ_g和σ_g
    for idx in id2score:
        if len(id2score[idx]) == 1:  # ★ 单样本group → μ=0, σ=1
            id2mean[idx] = 0.0
            id2std[idx] = 1.0
        elif len(id2score[idx]) > 1:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)

    # ★★★ 核心公式:
    # norm_adv_by_std_in_grpo=True (原始GRPO):
    #   a_i = (r_i - μ_g) / (σ_g + ε)  → 标准化 → 约零均值, 方差约1
    # norm_adv_by_std_in_grpo=False (★ Dr.GRPO, arxiv 2503.20783):
    #   a_i = r_i - μ_g  → ★ 不除std → 防止小groupσ=0时梯度消失!

    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]

    # ★ broadcast: 每个token获得相同advantage → 乘response_mask
    scores = scores.unsqueeze(-1) * response_mask
    return scores, scores  # advantage = returns (outcome-only, no value function)
```

**★★★ 关键设计决策**:
- 单样本group(rollout_n=1): μ=0, σ=1 → advantage=score本身 → **无group normalization** → 等效REINFORCE
- Dr.GRPO(False): 不除std → 当group内σ很小(所有response分数接近)时 → 除σ→advantage爆炸 → Dr.GRPO避免此问题
- advantage=returns: outcome-only → 不需要value function → GRPO跳过critic

### 4.2 GRPO_VECTORIZED — core_algos.py:335-358

```python
@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(...):
    # ★ 同公式但vectorized → 不需要Python loop → 快10-100x!
    scores = token_level_rewards.sum(dim=-1)
    g = as_torch_index(index, device=scores.device)
    mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)

    if norm_adv_by_std_in_grpo:
        scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
    else:
        scalars = scores - mean_g[g]

    advantages = scalars.unsqueeze(-1) * response_mask
    return advantages, advantages
```

**★★ 推荐使用GRPO_VECTORIZED**: 大batch时快10-100x, 数学结果完全一致!

### 4.3 GDPO: Per-Dimension Reward Normalization ★★

```python
@register_adv_est(AdvantageEstimator.GDPO)
def compute_gdpo_outcome_advantage(...):
    # ★★★ GDPO核心思想: 不是先sum再normalize, 而是per-dimension normalize再weighted sum!

    # Step 1: 每个reward dimension独立做GRPO normalization
    for i in range(num_scores):
        normalized_score, _ = compute_grpo_outcome_advantage(
            token_level_rewards=score_list[i],  # 第i个reward维度
            ...
        )
        new_advantage += weights[i] * normalized_score  # weighted aggregation

    # Step 2: batch-level whiten → 全局标准化
    advantages = verl_F.masked_whiten(new_advantage, response_mask) * response_mask
```

**★★★ GDPO vs GRPO**: GRPO = sum → normalize → GDPO = normalize → weighted_sum → whiten
- GDPO防止dominant reward信号淹没弱信号 → 每个维度独立标准化
- 需要custom_reward_function返回dict: `{"score": 1.0, "format_reward": 0.5, "accuracy_reward": 0.8}`
- config.algorithm.gdpo_reward_keys = ["format_reward", "accuracy_reward"]
- config.algorithm.gdpo_reward_weights = [0.3, 0.7] → 可选权重
- ★ RTX 4090可行: rule-based返回dict → 纯CPU → 不需要RM → GDPO+rule=RTX4090最优?

### 4.4 GRPO_PASSK: 只有最佳response非零 ★

```python
@register_adv_est(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(...):
    # ★★★ 只有最好的response获得非零advantage!
    # a_best = (r_max - r_second_max) / (σ + ε)
    # 其他response → advantage = 0 → 不参与训练!
    for idx in id2scores:
        topk, topk_idx = torch.topk(rewards, 2)
        r_max, r_second_max = topk[0], topk[1]
```

**适用**: pass@k evaluation → 只训练最优解 → 与rLLM pass@k eval对齐

### 4.5 KL-in-Reward vs KL-in-Loss ★★

```
GRPO默认: use_kl_in_reward=False + use_kl_loss=True → KL只在loss端

KL-in-reward (apply_kl_penalty, ray_trainer.py:76-115):
  token_level_rewards = token_level_scores - beta * kld
  → reward端减KL → 直接影响advantage → 更强的约束

KL-in-loss (默认):
  total_loss = pg_loss + kl_loss_coef * KL(pi || pi_ref) - entropy_coef * H
  → loss端加KL → 不影响reward → 更温和的约束
  → kl_loss_coef=0.001 (default) → 很小的KL约束
```

**★★ RTX 4090推荐**: KL-in-loss → 不需要额外ref forward → bypass_mode时零forward开销!

## 5. RewardManager 详细分析 ★★

### 5.1 NaiveRewardManager (legacy) — naive.py:27-122

```python
@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    def __call__(self, data, return_dict=False):
        # ★ 先检查rm_scores → 如果已存在直接返回 → 不重复计算!
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):  # ★ per-item loop → 慢但灵活
            data_item = data[i]
            # decode: tokenizer.decode(valid_response_ids)
            # 提取: ground_truth, data_source, extra_info, reward_scores
            score = self.compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
            # dict → score["score"] + reward_extra_info
            # float → score = result
            reward_tensor[i, valid_response_length - 1] = reward  # ★ outcome!
```

### 5.2 BatchRewardManager — batch.py:26-128

```python
@register("batch")
class BatchRewardManager(AbstractRewardManager):
    def verify(self, data):
        # ★ 批量decode → 批量scoring → 不是per-item!
        responses_str = [tokenizer.decode(response_ids[i][:valid_len]) for i in range(len(data))]
        scores = self.compute_score(
            data_sources=data_sources,     # ★ 列表!
            solution_strs=responses_str,    # ★ 列表! (不是solution_str)
            ground_truths=ground_truths,    # ★ 列表! (不是ground_truth)
            extra_infos=extras,             # ★ 列表!
            **self.reward_kwargs,
        )
        # ★ compute_score接收batch输入 → 可以做batch级处理 → 更高效
```

**★★★ Batch vs Naive关键差异**:
- Naive: `compute_score(data_source, solution_str, ground_truth, extra_info)` → 单item
- Batch: `compute_score(data_sources, solution_strs, ground_truths, extra_infos)` → 批量
- Batch的compute_score可以内部做batch优化 → 适合外部API调用场景

### 5.3 DAPORewardManager — dapo.py:26-154

```python
@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    # ★★★ overlong penalty: 超长response惩罚!
    def __init__(self, ..., overlong_buffer_cfg=None, max_resp_len=None):
        # overlong_buffer_cfg.enable → 开启惩罚
        # overlong_buffer_cfg.len → buffer长度
        # overlong_buffer_cfg.penalty_factor → 惩罚系数

    def __call__(self, data, ...):
        reward = score
        if self.overlong_buffer_cfg.enable:
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_reward = min(-exceed_len / overlong_buffer_len * penalty_factor, 0)
            # ★★★ reward += overlong_reward → 只减不增!
            reward += overlong_reward
```

**★★ DAPO overlong penalty**: 防止模型生成过长response → 惩罚线性增长 → reward=score+overlong_penalty

### 5.4 PrimeRewardManager — prime.py:103-189

```python
@register("prime")
class PrimeRewardManager(AbstractRewardManager):
    def verify(self, data):
        # ★★★ 异步并行: ProcessPoolExecutor + asyncio.gather!
        scores = run_reward_scoring(
            self.compute_score,
            completions=sequences_str,
            references=ground_truth,
            tasks=data_sources,
            extra_info=extra_info,
            num_processes=64,  # ★ 64个进程并行!
        )
        # timeout=300s → 超时=0分 → 进程自动kill

    def __call__(self, data, ...):
        scores = self.verify(data)
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
```

**★★★ PRIME特点**: ProcessPoolExecutor(64进程) + async + timeout → CPU密集型reward的最优选择!

### 5.5 Experimental NaiveRewardManager — reward_loop/naive.py:24-99

```python
@register("naive")  # experimental registry
class NaiveRewardManager(RewardManagerBase):  # ★ 不同的基类!
    async def run_single(self, data):  # ★ async interface!
        data = data[-1:]  # ★★★ multi-sequence时只取最后一个sequence!
        # → multi-turn agent loop → 只有最终response参与reward

        # ★★ 异步解码: run_in_executor → 不阻塞event loop
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # ★★ 异步评分: sync → run_in_executor; async → await directly
        if self.is_async_reward_score:
            result = await self.compute_score(...)
        else:
            result = await self.loop.run_in_executor(None, lambda: self.compute_score(...))

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
```

### 5.6 RemoteRewardManager — reward_loop/remote.py:40-131

```python
@register("remote")
class RemoteRewardManager(RewardManagerBase):
    def __init__(self, ...):
        # ★★★ Ray actor → 独立进程 → 避免thread pool问题!
        # https://github.com/verl-project/verl/issues/3407
        # Math-Verify等CPU密集型任务 → thread pool有fork安全问题
        self.reward_worker = [
            RewardComputeWorker.options(
                NodeAffinitySchedulingStrategy(node_id=..., soft=True)
            ).remote(self.compute_score)
            for _ in range(num_reward_workers)
        ]
        self.reward_worker_pool = itertools.cycle(self.reward_worker)  # ★ round-robin

    async def run_single(self, data):
        reward_worker = self.choose_reward_worker()
        result = await reward_worker.compute_score.remote(...)  # ★ Ray remote call
```

**★★★ Remote vs Naive**: Remote用Ray actor → 独立进程 → 解决thread pool的fork安全问题 → Math-Verify推荐

### 5.7 RateLimitedRewardManager — limited.py:172-540 ★★

```python
@register_manager("rate_limited")
@register_manager_legacy("rate_limited")
class RateLimitedRewardManager(RewardManagerBase):
    # ★★★ 三层限速: concurrency + RPM + TPM
    # ★ 全局class-level → 所有worker共享 → 跨节点统一限速!

    # Layer 1: Concurrency → asyncio.Semaphore(max_concurrent)
    # Layer 2: RPM → AsyncTokenBucket(max_rpm / 60 tokens/sec)
    # Layer 3: TPM → AsyncTokenBucket(max_tpm / 60 tokens/sec)

    async def run_single(self, data):
        # 1. acquire RPM token (1.0 per request)
        if self._rpm_limiter is not None:
            await self._rpm_limiter.acquire(1.0)

        # 2. acquire TPM tokens (estimated_tokens_per_request per request)
        if self._tpm_limiter is not None:
            await self._tpm_limiter.acquire(estimated_tokens)

        # 3. acquire concurrency semaphore
        async with self._semaphore:
            result = await asyncio.wait_for(
                self._compute_reward(...), timeout=self.timeout  # ★ 300s timeout
            )

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
```

**★★★ 适用场景**: LLM-as-judge → 外部API → 速率限制 → 3层保护 → 防止API限速/超限

### 5.8 全Registry清单 ★★

| 名称 | 层 | 基类 | 接口 | 特点 |
|------|---|------|------|------|
| naive (legacy) | workers/ | AbstractRewardManager | `__call__` | per-item loop, sync |
| batch (legacy) | workers/ | AbstractRewardManager | `__call__` | batch scoring, dict return |
| dapo (legacy) | workers/ | AbstractRewardManager | `__call__` | overlong penalty |
| prime (legacy) | workers/ | AbstractRewardManager | `__call__` | ProcessPoolExecutor(64) |
| naive (exp) | experimental/ | RewardManagerBase | `async run_single` | async, event loop executor |
| remote (exp) | experimental/ | RewardManagerBase | `async run_single` | Ray actor, separate process |
| rate_limited (both) | both registries | RewardManagerBase | `async run_single` | 3-layer rate limiting |
| gdpo (exp) | experimental/ | RewardManagerBase | `async run_single` | per-dimension reward dict |
| dapo (exp) | experimental/ | RewardManagerBase | `async run_single` | same as legacy dapo |

## 6. Colocated RM: Sleep/Wake模式 ★★★

### 6.1 架构

```
┌─────────────────────────────────────────────────────────────┐
│ Colocate Mode (default): RM共享actor/rollout的GPU            │
│                                                             │
│  训练step:                                                  │
│  1. Rollout model醒来 → 生成rollout                          │
│  2. Rollout model睡觉 → 释放GPU                              │
│  3. ★ RM model醒来 → 加载到同一GPU → 推理                    │
│  4. ★ RM model睡觉 → 释放GPU                                 │
│  5. Actor model醒来 → 继续训练                                │
│                                                             │
│  Standalone Mode: RM有独立resource_pool                      │
│  → RM始终醒着 → 与rollout并行 → streaming reward             │
│  → enable_resource_pool=True + nnodes + n_gpus_per_node      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 RewardModelManager — reward_model.py:27-124

```python
class RewardModelManager:
    def __init__(self, config, resource_pool=None):
        # ★ RM replica = vLLM/SGLang/TRT-LLM server
        # resource_pool → colocate模式 → 共享actor GPU
        # None → standalone模式 → 独立GPU pool

        rollout_replica_class = get_rollout_replica_class(rollout_config.name)
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=i, config=rollout_config,
                model_config=HFModelConfig(path=config.model_path),
                is_reward_model=True,  # ★ 标记为RM → 不同初始化逻辑
            )
            for i in range(num_replicas)
        ]

        # ★ Colocate: split_resource_pool → 每replica一部分GPU
        if self.resource_pool:
            split_resource_pools = split_resource_pool(resource_pool, split_size=rollout_world_size)
            self._run_all([server.init_colocated(pool) for server, pool in zip(...)])

        # ★ 初始化路由 → load balancing
        self._initialize_router()  # naive_router → HTTP proxy

    def wake_up(self):
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        self._run_all([replica.sleep() for replica in self.rollout_replicas])
```

### 6.3 compute_rm_score 流程 — reward_loop.py:323-352 ★★★

```python
class RewardLoopManager:
    def compute_rm_score(self, data):
        # ★★★ Step 1: Wake up RM (colocate模式)
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        # ★ Step 2: 分chunk到worker → parallel计算
        chunks = data.chunk(len(self.reward_loop_workers))
        outputs = ray.get([worker.compute_score_batch.remote(chunk) ...])

        # ★ Step 3: assemble rm_scores → outcome tensor (只有最后token非零!)
        scores = [item["reward_score"] for item in outputs_flat]
        rm_scores = self.reward_manager_cls.assemble_rm_scores(data, scores)

        # ★ Step 4: Sleep RM (释放GPU → actor可以继续训练)
        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()

        return DataProto(
            batch=TensorDict({"rm_scores": rm_scores}, batch_size=len(data)),
            non_tensor_batch={key: np.array([...]) for key in reward_extra_keys},
            meta_info={"reward_extra_keys": reward_extra_keys}
        )
```

### 6.4 训练loop中的reward集成 — ray_trainer.py:1518-1603 ★★★

```python
# ★ 训练step的reward部分:
with marked_timer("reward", timing_raw):
    # ★ Step 1: Colocate RM → wake → compute → sleep
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        batch_reward = self._compute_reward_colocate(batch)  # → wake RM → compute → sleep RM
        batch = batch.union(batch_reward)  # → rm_scores加入batch

    # ★ Step 2: extract reward → 统一接口
    reward_tensor, reward_extra_infos_dict = extract_reward(batch)
    # → extract_reward: batch.batch["rm_scores"] → reward_tensor
    # → 如果没有rm_scores → 用reward_manager计算 → 同样是outcome tensor

# ★ Step 3: reward → token_level_scores
batch.batch["token_level_scores"] = reward_tensor

# ★ Step 4: KL-in-reward (可选)
if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward)
    # → token_level_rewards = token_level_scores - beta * kld
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    # ★★★ 直接赋值! reward = scores → 无KL修改

# ★ Step 5: compute advantage
batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, ...)
# → GRPO: scores = token_level_rewards.sum(dim=-1) → group-relative → broadcast
```

### 6.5 Streaming Reward ★★

```python
# ray_trainer.py:940-956
# ★★★ streaming条件: rule-based OR RM with extra resource pool
enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

# ★ rule-based (no RM): reward_loop_workers传给agent loop → 每个sample立即reward
# ★ RM standalone (enable_resource_pool=True): RM有独立GPU → 可以并行
# ★ RM colocate (default): 不能stream → 必须等所有rollout完成 → 然后wake RM
```

**★★★ Colocate vs Standalone对比**:

| 特性 | Colocate | Standalone |
|------|----------|------------|
| GPU | 共享actor GPU | 独立GPU pool |
| 执行 | 先rollout→wake RM→compute→sleep RM | RM始终醒着→并行 |
| Streaming | ✗ 必须等全部rollout | ✓ 每个sample立即reward |
| GPU内存 | 需要RM能fit进actor sleep后的空间 | 需要额外GPU |
| 配置 | 默认 | enable_resource_pool=True |
| RM延迟 | 高(serialize) | 低(parallel) |

## 7. Built-in Reward Score函数 ★★

### verl/utils/reward_score/__init__.py: default_compute_score

```python
def default_compute_score(data_source, solution_str, ground_truth, extra_info=None,
                          sandbox_fusion_url=None, concurrent_semaphore=None, memory_limit_mb=None, **kwargs):
    # ★ 按data_source分派到具体评分函数:
    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)  # 字符串匹配

    elif data_source in ["lighteval/MATH", ...]:
        from . import math_reward
        res = math_reward.compute_score(solution_str, ground_truth)  # LaTeX对比

    elif data_source in ["math_dapo", "math", ...]:
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)

    elif data_source in ["codecontests", "apps", ...]:
        if sandbox_fusion_url:  # ★ sandbox执行 → 需要URL!
            from . import sandbox_fusion
            res = sandbox_fusion.compute_score(sandbox_fusion_url, ..., solution_str, ground_truth)
        else:
            from . import prime_code
            res = prime_code.compute_score(...)

    elif data_source in ["searchR1_nq", ...]:
        from . import search_r1_like_qa_em
        res = search_r1_like_qa_em.compute_score(...)

    else:
        raise NotImplementedError(f"Reward function not implemented for {data_source}")
```

**★★ 已支持的数据集**: GSM8k, MATH, MATH-500, math_dapo, AIME, numina系列, codecontests/apps/codeforces/taco, geo3k, searchR1系列

**★★ Sandbox Fusion**: 代码任务 → 需要执行代码 → sandbox_fusion_url → 云端代码执行 → concurrent_semaphore控制并发

## 8. 配置体系 ★★

### reward.yaml (verl/trainer/config/reward/reward.yaml):

```yaml
num_workers: 8  # ★ RewardLoopWorker数量 → 控制并行度

custom_reward_function:
  path: null       # ★ 外部文件路径 → null=用default_compute_score
  name: compute_score  # ★ 函数名 → 默认compute_score
  reward_kwargs: {}    # ★ 合入的额外参数

reward_manager:
  source: register  # register(内置) or importlib(外部)
  name: naive       # ★ RewardManager类型 → naive/dapo/batch/prime/remote/rate_limited/gdpo
  module:
    path: null      # importlib时需要
    name: custom_reward_manager

reward_model:
  enable: False     # ★★★ 是否启用RM → False=rule-based only!
  enable_resource_pool: False  # True=standalone RM
  n_gpus_per_node: 8
  nnodes: 0
  model_path: null  # RM模型路径
  rollout:
    name: ???       # vllm/sglang/trtllm
    tensor_model_parallel_size: 2  # ★ RM默认TP=2 → 2GPU minimum!
    gpu_memory_utilization: 0.5
    free_cache_engine: true  # ★ sleep时释放KV cache

sandbox_fusion:
  url: null            # sandbox URL
  max_concurrent: 64   # 最大并发sandbox请求
  memory_limit_mb: 1024
```

### ★★★ GRPO典型配置 (RTX 4090):

```yaml
reward:
  num_workers: 4              # RTX 4090: 4 worker足够
  custom_reward_function:
    path: my_reward.py        # ★ 自定义reward函数
    name: compute_score
  reward_manager:
    name: naive               # 简单per-item → CPU执行
  reward_model:
    enable: False             # ★★★ 关闭RM → rule-based only!
```

## 9. 完整数据流: Rollout → Reward → Advantage ★★★

```
训练step的数据流 (ray_trainer.py):

1. Rollout
   gen_batch → async_rollout_manager.generate_sequences(batch)
   → gen_batch_output: {input_ids, responses, attention_mask, log_probs, ...}

2. ★ Reward Computation
   if use_rm:
     → _compute_reward_colocate(batch)
       → reward_loop_manager.compute_rm_score(batch)
         → RM.wake_up()
         → RewardLoopWorker.compute_score_batch.remote(chunk)
           → if custom_reward_function: reward_manager.run_single(data)
           → elif DisRM: compute_score_disrm(data)
           → else: reward_manager.run_single(data) → default_compute_score
         → assemble_rm_scores: zeros→最后token赋值→outcome tensor
         → RM.sleep()
       → batch.union(batch_reward) → batch.batch["rm_scores"] = outcome tensor
   else:
     → extract_reward(batch)
       → reward_tensor = batch.batch["rm_scores"] (if exists)
       → or reward_manager(batch) → outcome tensor (只有最后token非零)

3. ★ Reward → Scores
   token_level_scores = reward_tensor  # (bs, response_length), 只有最后token非零

4. ★ KL-in-Reward (可选)
   if use_kl_in_reward:
     token_level_rewards = token_level_scores - beta * KL(old_log_probs, ref_log_prob)
   else:
     token_level_rewards = token_level_scores  # ★★★ GRPO默认: reward=scores

5. ★ Advantage Computation
   compute_advantage(batch, adv_estimator="grpo_vectorized")
   → scores = token_level_rewards.sum(dim=-1)  # ★ sum=scalar (因为只有最后token非零!)
   → group_mean_std → μ_g, σ_g per prompt
   → scalars = (scores - μ_g) / (σ_g + ε) or (scores - μ_g)  # Dr.GRPO
   → advantages = scalars.unsqueeze(-1) * response_mask  # broadcast到所有token
   → batch.batch["advantages"] = advantages

6. Actor Update
   → _update_actor(batch) → PPO/GRPO loss → backward → optimizer step
```

## 10. ★★★ RTX 4090专项分析

### 10.1 Rule-Based Reward = 唯一可行路径 ★★★

| Reward路径 | GPU需求 | RTX 4090可行性 | 原因 |
|-----------|---------|---------------|------|
| Rule-based (custom_reward_function) | 0 GPU | ✓✓✓ 完美可行 | 纯CPU → tokenizer.decode + score函数 |
| DisRM (7B classifier) | 2+ GPU | ✗ 不可行 | TP=2 → 需2GPU → 单GPU24GB → 即使colocate也fit不进 |
| DisRM (INT4 7B) | 1 GPU | ✗ 不推荐 | ~3.5GB权重 → 但vLLM/SGLang server overhead → KV cache → 总>5GB → marginal |
| GenRM (7B) | 1+ GPU | ✗ 不可行 | 同DisRM → 需要LLM server |
| Standalone RM | 2+ GPU total | ✗ 不可行 | 需要额外GPU → RTX 4090只有1个 |

**★★★ 结论**: RTX 4090只能用rule-based reward → 无GPU空间给RM → custom_reward_function是唯一选择

### 10.2 Outcome Reward与GRPO的完美对齐 ★★★

```
Outcome reward: reward_tensor[i, last_valid-1] = reward → 其他位置=0
→ token_level_rewards.sum(dim=-1) = scalar_reward
→ GRPO: a_i = (r_i - μ_g) / (σ_g + ε) → group-relative标准化
→ ★★★ 天然对齐! GRPO本就是outcome-only算法!

Process reward (per-token nonzero):
→ token_level_rewards.sum(dim=-1) ≠ scalar → 是累计奖励
→ GRPO仍然sum → 语义变了 → 变成"累计奖励的group-relative"
→ ★★★ verl不支持process reward → 所有Manager都是outcome!
→ 如需process reward → 需要custom compute_score返回per-token scores → 但目前没有Manager支持per-token!
```

**★★★ GRPO+outcome=RTX 4090最优**: 不需要process reward → 不需要RM → rule-based outcome → 纯CPU → 单GPU训练+推理

### 10.3 Sleep/Wake RM: 理论可能但实际不行 ★★

```
Colocate RM sleep/wake on RTX 4090:
  Step 1: actor model ~14GB (7B BF16) + LoRA ~0.5GB + KV cache ~2GB → ~17GB active
  Step 2: actor sleep → 释放KV+部分权重 → ~14GB remain
  Step 3: RM wake → 7B BF16 ~14GB → ★ 只剩~10GB → RM fit但KV cache不够!
  Step 4: RM compute → 低吞吐 → blocking → training loop卡住
  Step 5: RM sleep → actor wake → 继续训练

  ★★★ 实际问题:
  1. RM推理吞吐极低(vLLM 7B on 10GB → ~50tok/s → 阻塞training)
  2. sleep/wake开销: 模型加载+卸载 → 每step ~10-30s → 严重拖慢训练
  3. 内存碎片: actor sleep → RM wake → actor wake → 内存管理复杂
  4. INT4 RM: ~3.5GB → 但vLLM INT4不支持所有RM → 不通用

  ★★★ 结论: sleep/wake RM on RTX 4090 = 不现实 → rule-based是最优解
```

### 10.4 GDPO + Rule-Based: RTX 4090新可能性 ★★

```
GDPO需要: custom_reward_function返回dict → {"score": 1.0, "format": 0.5, "accuracy": 0.8}
→ 纯CPU → 不需要RM → RTX 4090可行!

示例: math reward函数
  def compute_score(data_source, solution_str, ground_truth, extra_info):
      format_score = check_format(solution_str)      # 0 or 0.1
      accuracy_score = check_answer(solution_str, ground_truth)  # 0 or 1
      return {"score": accuracy_score, "format_reward": format_score, "accuracy_reward": accuracy_score}

→ GDPO: 每个维度独立GRPO normalization → weighted sum → whiten
→ ★★★ 防止accuracy(0/1)淹没format(0/0.1) → 多维度独立标准化!
→ RTX 4090最优: GDPO + rule-based + GRPO_VECTORIZED + LoRA + bypass_mode
```

### 10.5 推荐配置 ★★★

```yaml
# ★★★ RTX 4090最优GRPO配置 (reward部分)
reward:
  num_workers: 4  # 4 CPU workers
  custom_reward_function:
    path: my_math_reward.py  # 自定义 → 纯CPU → 返回dict(支持GDPO)
    name: compute_score
  reward_manager:
    name: naive  # 简单per-item → CPU → 无GPU开销
  reward_model:
    enable: False  # ★★★ 关闭RM!

algorithm:
  adv_estimator: grpo_vectorized  # ★ 推荐(快10-100x)
  norm_adv_by_std_in_grpo: False  # ★ Dr.GRPO → 防止梯度消失
  use_kl_in_reward: False  # ★ KL只在loss端 → bypass_mode友好
  use_kl_loss: True  # ★ loss端KL → kl_coef=0.001
```

## 11. 关键发现总结 ★★★

1. **★★★ 两层Reward系统**: legacy(AbstractRewardManager.__call__) + experimental(RewardManagerBase.async run_single) → experimental是默认路径 → 通过load_reward_manager()加载

2. **★★★ Outcome reward = 最后token非零**: 所有Manager一致 → reward_tensor[i, last_valid-1] = score → sum(dim=-1) = scalar → GRPO天然对齐

3. **★★★ 3条reward路径**: custom_reward_function → DisRM(HTTP classify/embeddings) → GenRM(custom+HTTP) → 决策树清晰

4. **★★★ Colocate RM sleep/wake**: wake_up→compute→sleep → 共享GPU → 但RTX 4090不可行 → 内存太紧

5. **★★★ Streaming reward**: rule-based only → reward_loop_workers传给agent loop → 每个sample立即reward → RTX 4090可受益

6. **★★★ GDPO**: per-dimension normalization → 防止dominant信号淹没 → rule-based返回dict → RTX 4090可行!

7. **★★★ Dr.GRPO**: norm_adv_by_std_in_grpo=False → 不除σ_g → 防止小group梯度消失 → RTX 4090推荐

8. **★★★ RTX 4090唯一可行路径**: rule-based + outcome reward + GRPO_VECTORIZED + Dr.GRPO → 纯CPU reward → 单GPU训练+推理

## 源码文件索引

| 文件 | 关键内容 |
|------|---------|
| verl/trainer/ppo/reward.py | get_custom_reward_fn, load_reward_manager, extract_reward |
| verl/trainer/ppo/ray_trainer.py | _compute_reward_colocate, apply_kl_penalty, training loop reward integration |
| verl/trainer/ppo/core_algos.py | GRPO/GRPO_VEC/GDPO/GRPO_PASSK advantage, kl_penalty |
| verl/workers/reward_manager/abstract.py | AbstractRewardManager ABC, _extract_reward_from_rm_scores |
| verl/workers/reward_manager/naive.py | NaiveRewardManager (legacy) |
| verl/workers/reward_manager/batch.py | BatchRewardManager (batch interface) |
| verl/workers/reward_manager/dapo.py | DAPORewardManager (overlong penalty) |
| verl/workers/reward_manager/prime.py | PrimeRewardManager (ProcessPoolExecutor) |
| verl/workers/reward_manager/registry.py | Legacy REWARD_MANAGER_REGISTRY |
| verl/workers/config/reward.py | RewardConfig, RewardModelConfig, RewardManagerConfig |
| verl/experimental/reward_loop/reward_loop.py | RewardLoopManager, RewardLoopWorker, compute_rm_score |
| verl/experimental/reward_loop/reward_model.py | RewardModelManager (sleep/wake, router) |
| verl/experimental/reward_loop/reward_manager/base.py | RewardManagerBase ABC, assemble_rm_scores |
| verl/experimental/reward_loop/reward_manager/naive.py | NaiveRewardManager (async) |
| verl/experimental/reward_loop/reward_manager/remote.py | RemoteRewardManager (Ray actor) |
| verl/experimental/reward_loop/reward_manager/limited.py | RateLimitedRewardManager (3-layer rate limiting) |
| verl/experimental/reward_loop/reward_manager/gdpo.py | GDPORewardManager |
| verl/utils/reward_score/__init__.py | default_compute_score (gsm8k/MATH/math_dapo/code/geo3k/searchR1) |
| verl/trainer/config/reward/reward.yaml | Default reward config |
