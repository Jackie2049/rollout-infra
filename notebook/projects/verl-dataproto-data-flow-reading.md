# verl DataProto 数据流架构源码阅读

> 2026-06-15 | 源码: verl/utils/dp_rpc.py + verl/utils/tensor_dict.py + verl/trainer/ppo/ray_trainer.py
> 核心: DataProto是verl分布式RL训练的唯一数据抽象 → split-merge-reconstruct模式 → 跨WorkerGroup零拷贝数据流

## 1. DataProto类层次

```
DataProtoBase
  ├── meta_info: dict (非tensor元数据)
  ├── batch: BatchTensorDict (tensor数据容器)
  └── pad_batch: BatchTensorDict (padding后的数据,用于对齐batch_size)

DataProto (extends DataProtoBase)
  ├── split(split_size): 按batch维度分割 → list[DataProto]
  ├── merge(data_proto_list): 按batch维度合并 → DataProto
  ├── reconstructor: callable → 知道如何合并split后的结果
  ├── select(batch_ids): 选择特定batch索引
  ├── slice(length): 截取前N个batch
  ├── union(other): 合并两个DataProto的meta
  └── reorder_by_indicies(indices): 重排batch顺序
```

## 2. BatchTensorDict设计

```python
# verl/utils/tensor_dict.py
class BatchTensorDict:
    """所有tensor共享batch_dim的字典容器"""
    # 核心属性:
    # - 所有tensor必须有相同的batch_size(第0维)
    # - split/merge沿batch_dim操作
    # - 支持torch.compile(静态shape)

    def split(self, split_size):
        # 每个tensor沿dim=0切分 → 每个chunk是新的BatchTensorDict
        chunks = {}
        for key, tensor in self.items():
            chunks[key] = tensor.split(split_size, dim=0)
        return [BatchTensorDict(dict(zip(self.keys(), v))) for v in zip(*chunks.values())]

    def merge(self, tensor_dict_list):
        # 每个key沿dim=0 cat → 重建完整BatchTensorDict
        merged = {}
        for key in self.keys():
            merged[key] = torch.cat([td[key] for td in tensor_dict_list], dim=0)
        return BatchTensorDict(merged)
```

**关键**: BatchTensorDict强制所有tensor共享batch维度 → split/merge只需沿dim=0 → 极简!

## 3. DataProto核心API

```python
# verl/utils/dp_rpc.py
class DataProto(DataProtoBase):

    def split(self, split_size):
        """按batch维度分割DataProto → list[DataProto]

        用途: 将一个batch分成多个chunk → 发给多个Worker
        流程: batch.split(split_size) → 每个chunk成为新DataProto
        meta_info: 深拷贝到每个chunk(非tensor数据不可split → 全量复制)
        """
        data_proto_list = []
        batch_list = self.batch.split(split_size)
        for batch in batch_list:
            data_proto = DataProto(batch=batch, meta_info=deepcopy(self.meta_info))
            data_proto_list.append(data_proto)
        return data_proto_list

    def merge(self, data_proto_list):
        """合并多个DataProto → 1个DataProto

        用途: 从多个Worker收集结果 → 合成完整batch
        流程: BatchTensorDict.merge → meta_info取第一个(假设一致)
        """
        batch = BatchTensorDict.merge([dp.batch for dp in data_proto_list])
        meta_info = data_proto_list[0].meta_info  # 取第一个的meta
        return DataProto(batch=batch, meta_info=meta_info)

    def reconstructor(self):
        """返回merge函数 → 用于Ray RPC回调

        用途: split后发送给远程Worker → Worker返回结果 → reconstructor知道如何合并
        流程: return lambda data_proto_list: DataProto.merge(data_proto_list)
        关键: reconstructor随split数据一起发送 → coordinator不需要知道合并逻辑!
        """
        return lambda data_proto_list: self.merge(data_proto_list)
```

## 4. Split-Merge-Reconstruct模式详解

```
完整数据流 (PPO训练loop):

1. ActorRolloutRefWorker.generate_sequences:
   → 输出: DataProto(batch={tokens, log_probs, ...}, meta_info={uid, ...})

2. Coordinator拿到DataProto → split(split_size=N_workers):
   → [DataProto_chunk_0, DataProto_chunk_1, ..., DataProto_chunk_N]
   → 每个chunk包含1/N的batch + 完整meta_info(深拷贝)

3. 每个chunk发送到对应Worker(通过Ray RPC):
   → CriticWorker.infer(chunk_i) → chunk_i新增values tensor
   → RewardWorker.compute(chunk_i) → chunk_i新增rewards tensor

4. Coordinator收集所有Worker返回的chunk → reconstructor合并:
   → DataProto.merge([chunk_0_result, chunk_1_result, ...])
   → 重建完整DataProto, 包含所有Worker的输出

5. Coordinator发送完整DataProto到下一步Worker:
   → ActorWorker.update_policy(full_DataProto)

关键: split→distribute→process→merge→next step!
```

### 为什么meta_info不split?

```
meta_info是非tensor数据(uid/token_ids/config flags等):
  → 不可沿batch维度切分!
  → 解决: 深拷贝到每个chunk → 每个Worker都有完整meta_info
  → 代价: meta_info内存=N×原始(每个Worker都持有完整副本)
  → 优化: meta_info通常很小(几KB) → 深拷贝成本可忽略

  但如果meta_info很大(如长文本)?
  → 需要手动优化: 只传必要的meta字段 → 减少深拷贝开销
```

## 5. DataProto在PPO训练loop中的流动

```
ray_trainer.py 中的数据流:

Step 1: ActorRolloutRefWorker
  Input:  gen_batch = DataProto(batch={input_ids, attention_mask})
  Output: gen_batch_output = DataProto(batch={
    input_ids,           # prompt + response tokens
    responses,           # response tokens only
    log_probs,           # old log probabilities
    response_mask,       # mask for response tokens
  }, meta_info={
    uid,                 # unique identifier for each prompt
    temperature,         # sampling temperature
    rollout_n,           # number of completions per prompt (for GRPO)
  })

Step 2: Reward Computation
  Input:  gen_batch_output (split by coordinator)
  Output: gen_batch_output + batch.token_level_scores / token_level_rewards

Step 3: [PPO ONLY] CriticWorker.infer
  Input:  gen_batch_output (split by coordinator)
  Output: gen_batch_output + batch.values [bsz, resp_len]

Step 4: Advantage Computation
  PPO:  GAE(rewards, values, gamma, lam) → advantages [bsz, resp_len]
  GRPO: group_mean_std(rewards, uid) → advantages [bsz] (broadcast via mask)

Step 5: ActorWorker.update_policy
  Input:  DataProto(batch={input_ids, log_probs, advantages, response_mask})
  Output: train_stats (loss, reward, entropy, etc.)
```

## 6. Reconstructor机制深度分析

```
Reconstructor的核心价值: 解耦数据格式和处理逻辑

问题: 不同Worker需要不同数据格式
  Actor: 需要input_ids + log_probs + advantages
  Critic: 需要input_ids + attention_mask (只看上下文,不看response logits)
  Reward: 需要response tokens (只计算reward,不需要log_probs)

解决: DataProto作为统一容器 → Worker自己从DataProto提取所需字段
  → 不需要为每个Worker定制不同的数据格式!
  → 每个Worker只需: DataProto.batch[需要的key]

Reconstructor: 知道如何合并split后的结果
  → Coordinator不需要知道每个Worker内部的数据格式!
  → 只需: split → 发给Worker → 收回 → reconstructor.merge → 下一步

→ 整个数据流变成: DataProto → split → [Worker_i] → merge → DataProto → split → [Worker_j] → merge → ...
→ 每个步骤只关心: input DataProto → output DataProto (新增字段)
→ Worker间数据格式完全解耦!
```

## 7. 与rLLM数据流对比

| 维度 | verl DataProto | rLLM Tinker |
|------|---------------|-------------|
| **数据容器** | DataProto(batch+meta_info) | Tinker.Datum(prompt+response+reward+mask) |
| **跨Worker传输** | split→Ray RPC→merge | 无(in-process→直接Python调用) |
| **数据格式** | 统一BatchTensorDict | 自定义Datum结构 |
| **Reconstructor** | 需要(Ray远程合并) | 不需要(本地直接传递) |
| **meta_info深拷贝** | 是(每个Worker完整副本) | 否(本地共享) |
| **序列化开销** | Ray serialization(ZMQ/pickle) | 无(in-process→Python引用) |
| **适用规模** | 多GPU分布式 | 单GPU |

```
verl: DataProto → Ray → 远程Worker → serialization overhead
  → 多GPU必要: 通信开销 unavoidable
  → 灵活: 不同Worker可运行不同模型/不同GPU

rLLM Tinker: Datum → in-process → 直接Python调用
  → 单GPU最优: 无通信开销
  → 简单: 但无法跨GPU/跨节点
```

## 8. DataProto vs rLLM prefix-merge数据增强

```
verl单步RL:
  DataProto: input_ids=[prompt+response], log_probs, advantages
  → 1个response per prompt (或N个response for GRPO)

rLLM多步RL(prefix-merge):
  Datum: input_ids=[A0,obs1,A1,obs2,A2...], mask=[1,0,1,0,1...]
  → 多个observation+action交织 → 只action tokens(A)参与loss
  → mask使observation tokens的advantage=0 → loss只惩罚action选择
  → prefix-merge: 所有[A0,obs1,A1,obs2,A2]合并为1个序列 → 1次forward搞定!

  verl如果要支持多步: 需要修改DataProto → 新增response_mask/episode_ids
  → rLLM已在DataProto字段扩展: response_mask, episode_ids, trajectory_ids, group_roles, routed_experts
```

## 9. 实战: DataProto内存估算

```
7B模型, GRPO训练, rollout_n=4, batch_size=4, seq_len=2048:

DataProto.batch字段:
  input_ids:        [4×4, 2048] × 2 bytes (int64→实际int32=2B) = 4×4×2048×2 = 65.5KB
  log_probs:        [4×4, 2048] × 2 bytes (BF16) = 65.5KB
  response_mask:    [4×4, 2048] × 1 byte (bool) = 32.8KB
  token_level_rewards: [4×4, 2048] × 2 bytes = 65.5KB
  advantages:       [4×4, 2048] × 2 bytes = 65.5KB

DataProto.meta_info字段:
  uid:              4×4 uuid strings ≈ 1KB
  rollout_n:        int ≈ 16 bytes
  temperature:      float ≈ 16 bytes

Total DataProto ≈ 230KB per batch → 极小!
→ DataProto本身不是内存瓶颈 → 模型参数才是!

对比: 模型参数=14GB → DataProto=230KB → 60000×差距!
→ DataProto overhead完全可忽略!
```

## 10. 关键设计洞察

```
1. 统一数据抽象 → 一行代码切换模型
   DataProto不关心模型类型 → 任何模型只需输出DataProto格式
   → Qwen/Llama/ChatGLM → 统一接口 → verl自动处理

2. split-merge-reconstruct → Ray分布式核心
   split: 1个batch→N个chunk → N个Worker并行处理
   merge: N个结果→1个batch → coordinator合并
   reconstructor: 随数据发送 → Worker不需要知道合并逻辑

3. meta_info深拷贝 → 简化但代价小
   非tensor数据全量复制 → 每个Worker持有完整meta → 几KB→可忽略

4. BatchTensorDict → batch维度约束
   所有tensor共享batch_dim → split/merge只需dim=0 → 极简

5. DataProto扩展 → rLLM兼容
   rLLM在DataProto字段上扩展(response_mask/episode_ids等)
   → 理论上rLLM可以无缝使用verl的数据格式
   → 但rLLM Tinker选择Datum → 更简单、in-process无split/merge需求

6. 限制: meta_info不支持split → 大meta_data场景需优化
   → 如长文本prompt → 可能需要分chunk传递meta
```

---

Sources:
- [verl GitHub - DataProto implementation](https://github.com/volcengine/verl/blob/main/verl/utils/dp_rpc.py)
- [verl Documentation - DataProto Architecture](https://verl.readthedocs.io/en/latest/architecture/dataproto.html)
- [verl arXiv paper (2025)](https://arxiv.org/abs/2502.12025)
- notebook/projects/verl-ppo-vs-grpo-training-loop-comparison.md
- notebook/projects/rllm-gateway-backend-trainer-source-reading.md
