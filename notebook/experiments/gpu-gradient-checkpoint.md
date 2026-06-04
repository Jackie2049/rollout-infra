# Gradient Checkpointing GPU 实验 — A16 实测

> 2026-06-04 | A16 15GB (10 SMs, CUDA 11.8)

## 核心概念

训练时保存所有中间激活 → 内存爆炸。
Gradient Checkpointing: 只保存部分，反向时重新计算。

## 1. 内存节省

| Config | Params | No-CKPT | With-CKPT | Saving |
|--------|--------|---------|-----------|--------|
| Small (4L, H=512) | 17.7M | 236 MB | 131 MB | 45% |
| Medium (8L, H=768) | 64.4M | 547 MB | 330 MB | 40% |
| Large (12L, H=1024) | 161.4M | 1045 MB | 726 MB | 30% |

**结论**: 小模型节省更多（激活占比更大）。

## 2. 计算开销

| Strategy | Overhead |
|----------|----------|
| Every layer (12/12) | +28% |
| Every 2nd (6/12) | +14% |
| Every 3rd (4/12) | +10% |
| First+Last (2/12) | +5% |

**结论**: Every 2nd-3rd layer 是最佳权衡。

## 3. 最大 Batch Size

16L, H=1024, S=256:
- No checkpoint: max batch = 32 (11.5 GB)
- With checkpoint: max batch = 64 (4.6 GB)

**2x batch size → 2x throughput → 总训练时间可能更短！**

## 4. 内存-时间权衡曲线 (12L, B=8, S=512)

| Strategy | Memory | Time | vs No-CKPT |
|----------|--------|------|------------|
| None | 3090 MB | 1295 ms | baseline |
| Every 6 | 2636 MB | 1345 ms | -15% mem, +4% time |
| Every 3 | 2183 MB | 1406 ms | -29% mem, +9% time |
| Every 2 | 1729 MB | 1469 ms | -44% mem, +13% time |
| Every 1 | 748 MB | 1668 ms | -76% mem, +29% time |

## 5. 最佳实践

1. **Megatron-LM**: Selective checkpoint (只 checkpoint attention block)
2. **PyTorch FSDP**: `use_activation_checkpointing=True`
3. **策略选择**: 先不加，OOM 后加 every 2nd，还不够加 every 1
4. **与 ZeRO 组合**: Checkpoint 省 activation + ZeRO 省 optimizer = 极致内存节省

## 相关笔记

- [Training Memory](gpu-training-latency-model.md) — 内存组成分析
- [Distributed Training](../../projects/megatron-lm-reading.md) — ZeRO 优化
