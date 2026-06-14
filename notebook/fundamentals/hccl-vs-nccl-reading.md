# HCCL vs NCCL 对比研究笔记

> HCCL = Huawei Collective Communication Library (华为昇腾集合通信库)
> NCCL = NVIDIA Collective Communication Library
> 研究方法: Web search + 文档研究 (源码在Gitee, 需CANN环境)

## 1. 架构对比

### HCCL架构层

| 层 | 功能 | NCCL对应 |
|----|------|----------|
| HcclComm (通信域) | 管理参与集合操作的设备组 | ncclComm |
| 集合操作层 | AllReduce/Broadcast等标准原语 | 同 |
| 传输层 | HCCS(内节点)+RoCE(跨节点) | NVLink+IB/RoCE/TCP |

### API命名对照

| NCCL | HCCL | 说明 |
|------|------|------|
| `ncclGetUniqueId` | `HcclGetRootInfo` | 生成集群唯一ID |
| `ncclCommInitRank` | `HcclCommInitRank` | 按rank初始化通信域 |
| `ncclAllReduce` | `HcclAllReduce` | AllReduce |
| `ncclBroadcast` | `HcclBroadcast` | Broadcast |
| `ncclAllGather` | `HcclAllGather` | AllGather |
| `ncclReduceScatter` | `HcclReduceScatter` | ReduceScatter |
| `ncclSend/Recv` | `HcclSend/HcclRecv` | P2P通信 |
| `ncclGroupStart/End` | `HcclGroupStart/End` | 融合批处理 |

**初始化流程**: Root rank→HcclGetRootInfo()→广播ID→所有rank→HcclCommInitRank()→开始集合操作

---

## 2. 传输层对比

| 特性 | HCCS (昇腾内节点) | NVLink/NVSwitch (NVIDIA) |
|------|-------------------|-------------------------|
| 带宽 | ~392 GB/s 双向聚合 | 300-900 GB/s (NVLink4/NVSwitch) |
| 拓扑 | **全mesh** 8-NPU (7端口/芯片) | Ring/Mesh via NVSwitch |
| PCIe | **不涉及!** 直接芯片互连 | NVLink bypass PCIe |
| 协议 | 华为私有 | NVIDIA私有 |

| 特性 | RoCE v2 (昇腾跨节点) | InfiniBand/RoCE (NVIDIA) |
|------|----------------------|--------------------------|
| 带宽 | 100 GbE/link | 200-400 Gb/s (IB) |
| 协议 | 标准 RDMA over Ethernet | IB/RoCE/TCP |
| 拓扑 | Ring跨服务器 | Ring/Tree跨节点 |

**关键差异**: Atlas 800训练服务器用HCCS全mesh连接8个910芯片 → **消除PCIe瓶颈** → 与NVIDIA NVLink/NVSwitch不同但目的相同

---

## 3. 算法选择对比

| 算法 | HCCL | NCCL |
|------|------|------|
| Ring | 大数据(>256KB), 带宽最优 | 同 |
| Mesh/Tree | **HCCL=Mesh**(全mesh all-to-all, 利用HCCS拓扑) | **NCCL=Tree**(二叉树) |
| Hierarchical | Mesh(内)+Ring(跨) | Ring(内)+Ring(跨) |
| CollNet | ❌ 无 | ✅ NVLink-connected |

**HCCL Mesh vs NCCL Tree**: HCCL默认内节点用Mesh(因910全mesh拓扑) → 更适合小消息; NCCL默认内节点用Ring → 对大消息带宽更优

**HCCL两级Hierarchical**:
1. Level 1: 内节点ReduceScatter (HCCS Ring/Mesh)
2. Level 2: 跨节点AllReduce (RoCE Ring) → 减少跨节点流量 O(N→G)
3. Level 3: 内节点AllGather (HCCS Ring/Mesh)

---

## 4. 支持的数据类型

| 类型 | HCCL | NCCL |
|------|------|------|
| float16 | ✅ | ✅ |
| float32 | ✅ | ✅ |
| float64 | ❌ | ✅ |
| **bfloat16** | ❌ | ✅ (训练关键!) |
| int8 | ✅ | ✅ |
| int32 | ✅ | ✅ |

**HCCL缺失bfloat16**: 这是重大限制! BF16是现代训练的标准精度 → HCCL不支持 → Ascend训练只能用FP16 → 精度问题更大

---

## 5. 集合操作支持

| 操作 | HCCL | NCCL | EP/PP适用 |
|------|------|------|----------|
| AllReduce | ✅ | ✅ | DP梯度同步 |
| AllGather | ✅ | ✅ | SP/ZeRO-3 param gather |
| ReduceScatter | ✅ | ✅ | ZeRO-2梯度/SP |
| All-to-All | ✅ | ✅ | MoE Expert Parallel |
| Send/Recv | ✅ | ✅ | Pipeline Parallel |
| Group Fusion | ✅ | ✅ | 批量小操作融合 |

---

## 6. 性能对比 (Ascend 910B vs NVIDIA)

### AllReduce 8-NPU 单节点

| 平台 | 数据大小 | 有效带宽 |
|------|----------|----------|
| Ascend 910B (HCCS Ring) | 64 MB | ~28 GB/s |
| Ascend 910B (HCCS Mesh) | 64 MB | ~12 GB/s |
| H100 (NVLink) | 64 MB | ~60-80 GB/s |
| A100 (NVLink) | 64 MB | ~30-40 GB/s |
| RTX 4090 (PCIe) | 64 MB | ~6 GB/s |

### 跨节点Hierarchical

| 配置 | Ascend 910B | H100 (IB) |
|------|-------------|-----------|
| 16 NPU (2 servers) | ~20 GB/s | ~40-50 GB/s |
| 32 NPU (4 servers) | ~1.3x vs flat Ring | ~2x vs flat Ring |

### Ascend 910C预估

- 8-NPU AllReduce: ~35+ GB/s (vs 910B ~28)
- HBM3带宽提升: 1.2→1.6 TB/s
- FP16: ~320 TFLOPS
- INT8: ~640 TOPS

---

## 7. 拓扑检测与配置

| 维度 | HCCL | NCCL |
|------|------|------|
| 自动检测 | ✅ 但不如NCCL健壮 | ✅ 非常健壮 |
| 手动配置 | **Rank table JSON** (必需!) | 环境变量即可 |
| 调试 | `HCCL_LOG_LEVEL` | `NCCL_DEBUG=INFO/WARN/TRACE` |

**HCCL关键环境变量**:
```
HCCL_ALGO=Ring|Mesh|Hierarchical
HCCL_INTRA_ALGO=Ring|Mesh
HCCL_INTER_ALGO=Ring
HCCL_CONNECT_TIMEOUT=1800  # 大集群需增大!
HCCL_WHITELIST_DISABLE     # 绕过安全检查(常见需求)
HCCL_SOCKET_IFNAME         # 网络接口选择
```

---

## 8. 与框架集成

| 框架 | HCCL集成方式 | 说明 |
|------|-------------|------|
| **MindSpore** | 原生集成 | `backend="npu"` → HCCL底层 |
| **PyTorch (torch_npu)** | NCCL→HCCL兼容层 | `torch_npu/csrc/distributed/` → symbol映射 |
| **vLLM-Ascend** | HCCL注册为自定义backend | `backend="hccl"` → all_reduce/all_gather/broadcast |
| **MindIE** | CANN层内 | 5层架构底层 → ATB用HCCL做TP通信 |

**torch_npu兼容层**: `backend="nccl"` → `backend="npu"` + `torch.cuda` → `torch_npu.npu` → C++ shim自动映射NCCL→HCCL API

---

## 9. PP和EP支持

| 并行类型 | HCCL操作 | 说明 |
|----------|---------|------|
| DP | AllReduce | 梯度同步 |
| TP | AllReduce | 权重切分同步 |
| PP | HcclSend/HcclRecv | 阶段间P2P |
| EP | HcclAlltoAll | MoE expert路由 |

**EP关键挑战**: AlltoAll overhead → HCCL优化AlltoAll是MoE训练效率关键 → expert load imbalance加剧问题

**Ascend内存限制**: 910A=32GB, 910B/910C=64GB → 少于H100(80GB) → 影响EP配置

---

## 10. HCCL已知限制 vs NCCL

| 限制 | 影响 |
|------|------|
| **无BF16支持** | 训练只能FP16 → 精度风险更大 |
| 初始化慢 | 大集群rank init比NCCL慢 |
| 拓扑检测弱 | 需手动rank table JSON → 配置复杂 |
| 文档中英混合 | 部分文档中文 → 非中文用户不便 |
| 社区小 | 少于NCCL → 第三方工具少 |
| PyTorch摩擦 | torch_npu集成有摩擦点(hangs/gradient sync issues) |
| 跨节点扩展 | RoCE带宽低于IB → 多节点瓶颈更大 |
| 调试工具少 | hccl_test覆盖窄于nccl_test |

---

## 11. RTX 4090影响

```
RTX 4090 + HCCL:
  - RTX 4090 = NVIDIA GPU → HCCL不可用(Ascend专用)!
  - MindIE = Ascend专用 → RTX 4090不可用
  - 替代: vLLM(NCCL) / DeepSpeed(NCCL) / PyTorch(NCCL)
  → 结论: RTX 4090完全不需要考虑HCCL → NCCL是唯一backend
```

---

## 12. 开源状态

| 项目 | 地址 | License |
|------|------|---------|
| HCCL源码 | https://gitee.com/ascend/HCCL | Apache 2.0 |
| HCCL测试 | https://gitee.com/ascend/hccl_test | - |
| MindSpore | https://github.com/mindspore-ai | Apache 2.0 |
| Ascend PyTorch | https://github.com/Ascend/pytorch | BSD-3 |

---

## 与7框架的关系

| 框架 | 通信库 | HCCL关系 |
|------|--------|----------|
| DeepSpeed | NCCL | ❌ 不用HCCL |
| Megatron-LM | NCCL | ❌ 不用HCCL |
| vLLM | NCCL / HCCL(Ascend版) | vLLM-Ascend用HCCL替代NCCL |
| verl | NCCL(FSDP/Megatron) | ❌ 不用HCCL |
| rLLM | NCCL(via verl) | ❌ 不用HCCL |
| **MindIE** | **HCCL** | **✅ 核心通信库!** |
| PyTorch | NCCL/HCCL/Gloo | torch_npu提供HCCL兼容层 |

---

## 参考资料

- [HCCL API Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/80RC2/apiref/hcclapi/)
- [HCCL Architecture DevRef](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/80RC2/devref/hccldevref/)
- [HCCL Source](https://gitee.com/ascend/HCCL)
- [MindSpore Distributed Training](https://www.mindspore.cn/docs/en/master/advanced/train/distributed_training_ascend.html)
- [Ascend PyTorch](https://github.com/Ascend/pytorch)
