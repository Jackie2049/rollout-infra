# 分布式文件系统

> Lustre / GPFS / NFS — 大规模训练的存储层

## 1. 为什么需要分布式文件系统

```
训练集群存储需求:
  - 数据集: 数 TB 到数百 TB
  - Checkpoint: 每次 100GB-数 TB
  - 模型权重: 数 GB 到数百 GB
  - 日志: 数 GB

单机存储不够 → 需要跨节点共享的文件系统
```

## 2. 常见分布式文件系统

| 文件系统 | 特点 | 适用场景 |
|---------|------|---------|
| Lustre | HPC 标配，高聚合带宽 | 大规模训练集群 |
| GPFS/Spectrum Scale | IBM，商业级，强一致性 | 企业级集群 |
| NFS | 简单，通用 | 小规模，开发环境 |
| Ceph | 开源，对象+块+文件 | 云原生环境 |
| BeeGFS | 高性能，易部署 | 中小规模 GPU 集群 |

## 3. Lustre 架构

```
┌─────────────────────────────────────────────┐
│  Client (训练节点)                           │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
│  │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│          │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘          │
│     └───────┴───────┴───────┘              │
│             │ Lustre Client                  │
└─────────────┼───────────────────────────────┘
              │
    ┌─────────┴─────────┐
    │  Lustre Network    │  (InfiniBand/RoCE)
    └────┬──────────┬────┘
         │          │
   ┌─────┴───┐  ┌──┴──────┐
   │  MDT    │  │  OSTs   │
   │(Metadata│  │(Object   │
   │ Server) │  │ Storage  │
   │         │  │ Targets) │
   │ 文件名   │  │ 文件数据  │
   │ 权限     │  │ 条带化    │
   │ 布局     │  │          │
   └─────────┘  └──────────┘
```

### 3.1 条带化 (Striping)

```
一个大文件被分片存储在多个 OST 上:
  文件: [stripe1][stripe2][stripe3][stripe4]
         ↓       ↓       ↓       ↓
        OST-0   OST-1   OST-2   OST-3

优势:
  - 聚合带宽 = 单 OST 带宽 × OST 数量
  - 并行读取/写入
  - 负载均衡

设置条带:
  lfs setstripe -c 4 /path/to/file  # 4 个 OST 条带
  lfs setstripe -S 4M /path/to/dir  # 4MB 条带大小
```

## 4. 训练场景的存储优化

### 4.1 Checkpoint 写入

```
优化策略:
  1. 每个节点写入本地 SSD → 完成后异步同步到 Lustre
  2. 只由 rank 0 写入 (避免并发写冲突)
  3. 使用分布式 checkpoint (每 rank 写自己的分片)

避免:
  ✗ 所有 rank 同时写同一文件
  ✗ 频繁 sync/fsync
  ✗ 小文件写入 (元数据瓶颈)
```

### 4.2 数据读取

```
优化策略:
  1. 预处理为少量大文件 (减少元数据操作)
  2. 使用 WebDataset tar 格式
  3. 首次读取后缓存到本地 SSD (burst buffer)
  4. 避免随机读取 (顺序读取性能好 10x)

Lustre 性能参考:
  顺序读: 10-100 GB/s (聚合)
  随机读: 1-10 GB/s (受元数据限制)
  顺序写: 5-50 GB/s
  元数据操作: ~10,000 ops/s (vs 本地 SSD ~100,000 ops/s)
```

## 5. 学习要点

1. **Lustre 是 HPC 标配** — 了解 MDT/OST/条带化概念
2. **大文件 >> 小文件** — 预处理数据减少元数据压力
3. **Checkpoint 写入要异步** — 避免阻塞训练
4. **本地缓存 (burst buffer)** — 减少分布式 FS 读取次数

## 参考

- [Lustre Documentation](https://doc.lustre.org/lustre_manual.xhtml)
- [BeeGFS Documentation](https://www.beegfs.io/documentation/)
