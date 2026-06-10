# Distributed Systems Deep Dive

> 2026-06-08 | 分布式系统=AI Infra基础设施! 从CAP定理→一致性→共识→分布式训练→分布式推理, 5层理论体系
> 关联: fsdp2-scaling-benchmark-rtx4090.md, nccl-multi-gpu-benchmark-rtx4090.md, moe-serving-deep-dive.md

## 0. 核心定律: 分布式系统 = 在不可靠节点上构建可靠服务

```
分布式系统根本挑战:
  → 网络不可靠: 消息可能延迟/丢失/重复 → 不保证送达!
  → 节点不可靠: 机器可能crash/reboot/永久故障 → 不保证运行!
  → 时间不可靠: 不同节点时钟不同步 → 不保证顺序!
  → → → 3个不可靠 → 分布式系统必须容错 → 这是设计出发点!

与AI Infra联系:
  → 8×RTX 4090训练 → 任意GPU可能crash → ZeRO/FSDP必须容错!
  → → NCCL AllReduce → 如果1个GPU故障 → 整个AllReduce失败 → 训练中断!
  → → → → 需要elastic training → 动态调整GPU数量 → Ray/PyTorch Elastic!
  → → → → → vLLM distributed → EngineCoreProc → 多GPU推理 → 容错+重连!

RTX 4090分布式现实:
  → PCIe 8×4090 → AllReduce 2.76 GB/s → 比NVLink慢3.3x!
  → → 无P2P(消费级GPU) → 不能直接GPU→GPU → 必须经CPU/PCIe!
  → → → FSDP 8GPU=0.46x → 比1GPU慢2x! → PCIe scaling灾难性!
  → → → → RTX 4090分布式训练/推理 = 必须NVLink → 否则不如单GPU!
```

## 1. CAP定理 — 分布式系统不可能三角

```
CAP定理:
  → Consistency: 所有节点看到相同数据 → 一致性!
  → Availability: 每个请求都得到响应 → 可用性!
  → Partition tolerance: 网络分区时系统仍运行 → 分区容忍!
  → → → 三者只能同时满足2个! → 不可能三角!

  CP系统:牺牲Availability → 网络分区时拒绝请求 → 保证一致!
    → → ZooKeeper/Raft → 一致优先 → 分区时不响应 → 等3节点中2节点同意!
    → → → AI Infra: checkpoint写入 → 所有节点必须一致 → CP!

  AP系统:牺牲Consistency → 网络分区时继续响应 → 但数据可能不一致!
    → → DNS/Cassandra → 可用优先 → 分区时各节点独立响应 → 最终一致!
    → → → AI Infra: 推理请求 → 不需所有GPU一致 → AP OK!

  → AI Infra的实际选择:
    → 训练: CP! → 所有GPU必须看到相同模型参数 → 一致性是训练基础!
    → → → FSDP AllReduce → 所有GPU同步 → CP → 如果1GPU失联 → 训练停止!
    → 推理: AP! → 不同GPU可以服务不同请求 → 不需要严格一致!
    → → → vLLM distributed → 每GPU独立推理 → AP → GPU故障→请求转到其他GPU!
    → → → → → 训练=CP / 推理=AP → 这就是为什么训练比推理更难容错!

RTX 4090 CAP影响:
  → FSDP训练: CP → AllReduce需要所有GPU → 1GPU故障=训练中断!
  → → → → 需要elastic training → 动态调整GPU数量 → 但RTX 4090 PCIe scaling已经差!
  → vLLM推理: AP → 每GPU独立 → 故障→切换 → 但需额外GPU做backup!
```

## 2. 共识算法 — 让不可靠节点达成一致

```
共识问题: N个节点 → 有些可能故障 → 如何让存活节点对某个值达成一致?

Paxos (Lamport, 1998):
  → 理论基础 → 但极难理解 → "The Part-Time Parliament" → 用古希腊议会比喻!
  → → 核心: Proposer提出 → Acceptor投票 → Learner学习 → 多轮投票 → 多数同意=共识!
  → → → 需要quorum(多数): 2f+1节点容忍f个故障 → 3节点容忍1故障 → 5节点容忍2故障!
  → → → → Paxos保证: 所有存活节点最终同意同一个值 → safety!
  → → → → → 但Paxos不保证进展 → 可能无限轮投票 → liveness需要额外机制!

Raft (Ongaro & Ousterhout, 2014):
  → "可理解的Paxos" → 设计目标=易懂 → 比Paxos容易实现正确!
  → → 核心机制:
    → → 1. Leader Election: 1个Leader+多个Follower → Leader管理所有决策!
    → → → → 心跳机制 → Leader定期发送heartbeat → Follower收到=正常!
    → → → → → Leader故障 → Follower超时 → 变成Candidate → 请求投票 → 多数同意=新Leader!
    → → 2. Log Replication: Leader接收client请求 → 写入本地log → 复制到Follower → 多数写入=committed!
    → → → → → 所有log最终一致 → Leader保证顺序 → Follower按顺序应用!
    → → 3. Safety: 已committed的log永不丢失 → 即使Leader变更 → 新Leader必须包含所有committed log!

  → 与AI Infra联系:
    → → Ray GCS(Global Control Service) → Raft → 所有调度决策通过GCS → 一致!
    → → → → Ray GCS crash → 所有worker失联 → 但Raft保证快速选举新GCS → 继续运行!
    → → → → → verl用Ray → Ray内部用Raft → 分布式训练的共识基础!
    → → → → → → 我们8×RTX 4090训练 → Ray → Raft → GPU故障→Ray重新调度!

BFT (Byzantine Fault Tolerance):
  → 更强容错 → 不仅容忍crash → 还容忍恶意节点(发送错误数据)!
  → → 需要3f+1节点容忍f个恶意 → 4节点容忍1恶意 → 比crash-fault更贵!
  → → → AI Infra通常不需要BFT → GPU crash ≠ GPU恶意 → crash-fault够了!
  → → → → 但!恶意模型更新(如 poisoned gradient) → 可能需要BFT验证 → 防攻击!
  → → → → → 通常用gradient verification(检查梯度是否异常) → 不用BFT → 更省资源!

RTX 4090共识成本:
  → Raft: 心跳1ms → 几乎零overhead → 但选举需要~100ms(Leader故障→新Leader)!
  → → → → 100ms选举 → 训练可能停100ms → 可接受!
  → → NCCL AllReduce: 不是共识 → 是集体操作 → 所有GPU同步 → 不是投票!
  → → → → → AllReduce = 保证所有GPU得到相同结果 → 语义=consensus → 但机制=different!
```

## 3. 分布式训练 — 通信与一致性

```
分布式训练通信模式:
  → AllReduce: 所有GPU同步梯度 → ∑∇ → 平均 → 每GPU得到相同梯度!
  → → → Ring AllReduce: N-1步 → 每步传2/N数据 → 总通信=2×(N-1)/N × data_size!
  → → → → NCCL实现 → RTX 4090实测 AllReduce 2.76 GB/s(100MB) → 比理论慢!

  → ReduceScatter + AllGather: FSDP用 → 先分片Reduce → 再收集 → 等效AllReduce但内存省!
  → → → → ZeRO-3: ReduceScatter(梯度分片) → AllGather(参数收集) → 两步 → 等效!

  → P2P通信: Pipeline Parallelism用 → GPU间直接传输 → 点对点!
  → → → → Ring Exchange最快(NCCL) → 1次send+recv → 同时进行!
  → → → → → RTX 4090 P2P disabled → 必须经PCIe+CPU → 慢!

  → All-to-All: MoE EP用 → 每个expert把token发给对应GPU → 全对全!
  → → → → NVLink: <1ms → PCIe: >10ms → RTX 4090 PCIe不可行!

与ZeRO/FSDP联系(之前benchmark实测):
  → FSDP1 2GPU=1.12x →勉强OK → 通信占~50%
  → FSDP1 4GPU=0.69x →比1GPU慢! → 通信占>70%!
  → FSDP1 8GPU=0.46x →灾难! → 通信占>80%!
  → → → PCIe bandwidth是瓶颈 → 不是算法问题 → 硬件限制!
  → → → → NVLink预估8GPU可达3x → 但RTX 4090只有PCIe → 6.6x差距!

Elastic Training:
  → 动态调整GPU数量 → GPU故障→缩减 → GPU恢复→扩展!
  → → → Ray/PyTorch Elastic → 自动管理 → 但需要checkpoint一致性!
  → → → → → Checkpoint共识 → 所有GPU必须保存同一版本 → 否则恢复不一致!
  → → → → → → 用Raft选coordinator → coordinator决定checkpoint版本 → 一致!
```

## 4. 分布式推理 — 容错与负载均衡

```
分布式推理挑战:
  → 不像训练 → 推理不需要所有GPU同步 → 更容易容错!
  → → → 但! TP(Tensor Parallelism)需要所有GPU → 1GPU故障=整个推理失败!
  → → → → TP容错 → 需要GPU backup → 或降级到更少GPU → 但模型权重需重新切分!

  → vLLM分布式推理:
    → EngineCoreProc → 双进程+三线程 → ZMQ IPC → 主进程→GPU进程→辅助线程!
    → → → DPEngineCoreProc(MoE) → 32步All-Reduce → Wave协调 → 两阶段暂停!
    → → → → → 如果1GPU故障 → DPEngineCoreProc暂停 → Ray重新调度 → 切换到其他GPU!

  → 负载均衡:
    → 推理请求分布 → 不能所有请求打到同一GPU → 需均衡!
    → → → Round-robin: 最简单 → 但不考虑GPU负载差异!
    → → → → Least-loaded: 选负载最低GPU → 更公平 → 但需监控!
    → → → → → vLLM scheduler → unified token budget → 所有GPU共享budget → 自然均衡!

RTX 4090分布式推理:
  → 单GPU推理 → 7B INT4+INT8KV → 4,791 tok/s → 最practical!
  → → → 不需要分布式 → 单GPU够 → 但需要足够memory!
  → → → → 7B INT4 → 3.5GB → 24GB → 大量空间给KV → B=118 → 高并发!
  → → → → → → RTX 4090最优 = 单GPU推理 → 不需分布式 → 最省成本!
  → → → → → → → 如果需要更大模型(70B) → TP=4GPU → 但PCIe scaling差 → 需NVLink!

PD分离(Prefill-Decode Separation):
  → Prefill GPU(compute-bound) + Decode GPU(memory-bound) → 互补!
  → → → PCIe KV transfer仅3%TTFT → PD可行(修正之前"PCIe PD不可行"!)
  → → → → 但需要NVLink才能充分利用 → RTX 4090 PCIe PD有limitation!
```

## 5. Fault Tolerance — 故障恢复策略

```
故障类型:
  → Crash fault: GPU突然故障 → 不再响应 → 最常见!
  → → → 恢复: 检查点恢复 → 从最近checkpoint继续 → 损失进度!
  → → → → → verl checkpoint → 定期保存 → 故障→加载 → 继续 → 损失=故障前进度!

  → Network partition: 网络分区 → 部分GPU不可达 → 但仍在运行!
  → → → 恢复: 选新coordinator → 分区内的GPU继续 → 分区外的等待 → 合并!
  → → → → → CP系统: 等分区恢复 → AP系统: 各分区继续 → 最终合并!

  → Slow node: GPU慢 → 比其他GPU3x延迟 → 拖累整体!
  → → → 恢复: 降级或踢出 → 慢GPU→等待→超时→排除 → AllReduce不含它!
  → → → → → FSDP straggler → NCCL timeout → 默认30min → 太长! → 需设更短!

Checkpoint一致性:
  → 所有GPU必须保存同一step的checkpoint → 否则恢复不一致!
  → → → 用barrier同步 → 所有GPU到同一step → 然后一起保存 → 一致!
  → → → → → 但barrier开销 → 每step都barrier→太慢 → 只在save checkpoint时barrier!

  → 异步checkpoint → GPU独立保存 → 不等barrier → 但可能不一致!
  → → → → → 需要: 最新一致版本 ≤ 最新保存版本 → 恢复到最新一致版本!
  → → → → → → verl异步checkpoint → 已实现 → 更快但需要版本管理!

RTX 4090容错:
  → 8GPU → 1故障 → 7GPU继续 → 但FSDP 7GPU可能更慢! → 不划算!
  → → → → → 最好: 重启故障GPU → 重新加入 → 但需要checkpoint恢复 → 时间成本!
  → → → → → → 实际: GPU故障→人工修复→重启训练→从checkpoint继续 → 最简单!
```

## 6. Core Laws — 分布式系统核心定律

```
1. CAP Law: Consistency+Availability+Partition → 只能同时满足2个!
   → → 训练=CP → 推理=AP → 不同场景不同选择!
   → → → AI Infra必须理解CAP → 才能做出正确架构决策!

2. Consensus Law: N节点共识需要 ≥ ⌈(N+1)/2⌉ 存活节点 → majority quorum!
   → → 3节点 → 需2存活 → 5节点 → 需3存活 → 容忍⌊(N-1)/2⌋故障!
   → → → BFT需 ≥ ⌈(2N+1)/3⌉ → 更多节点 → 但AI Infra通常不需BFT!

3. Communication-Bound Law: 分布式训练加速 ∝ 1/(1 + compute_time/comm_time)
   → → compute_fast → comm占比高 → scaling差 → RTX 4090案例!
   → → → compute_slow → comm占比低 → scaling好 → A100案例!
   → → → → → GPU越强 → scaling越难 → 因为compute快 → comm成瓶颈!

4. Fault Tolerance Law: 容错成本 ∝ redundancy × latency
   → → 冗余GPU=成本 → 但减少故障影响 → trade-off!
   → → → 快速恢复=低latency → 但需要更多preparation → trade-off!
   → → → → → RTX 4090最优=不冗余(成本限制) → 快速checkpoint恢复(简单策略)

5. PD Separation Law: PD分离收益 ∝ compute_memory_complementarity
   → → Prefill(compute-bound) + Decode(memory-bound) → 互补 → 收益大!
   → → → 但需要NVLink(PCIe限制) → RTX 4090 = 单GPU最优(不需PD!)
   → → → → → H100 NVLink → PD分离 → 最大收益 → 未来方向!
```

## 关键论文与参考

```
- CAP定理 (Brewer, 2000): 一致性/可用性/分区容忍 → 不可能三角!
- Paxos (Lamport, 1998): 理论共识基础 → 极难理解但最robust
- Raft (Ongaro & Ousterhout, 2014): 可理解的共识 → 2014 USENIX best paper!
- MIT 6.824: 分布式系统课程 → Raft+Paxos+分布式事务 → 最佳学习资源
- Kleppmann "Designing Data-Intensive Applications": 最全面的分布式系统书!
- NCCL (NVIDIA): 集体通信 → AllReduce → 分布式训练基础
- Ray GCS: Raft-based调度 → 分布式训练的共识层
- PyTorch Elastic: 动态训练 → GPU故障→自动调整 → elastic training
- verl checkpoint: 异步checkpoint → 版本管理 → 容错恢复
- Flexible Paxos (Howard et al., 2017): quorum不需全重叠 → 更灵活共识

Sources:
- [Raft Paper](https://raft.github.io/raft.pdf)
- [MIT 6.824](https://pdos.csail.mit.edu/6.824/)
- [Designing Data-Intensive Applications](https://dataintensive.net/)
- [Raft Visualization](https://thesecretlivesofdata.com/)