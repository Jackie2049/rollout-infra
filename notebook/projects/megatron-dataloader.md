# Megatron-LM DataLoader 分析

> 目录: `megatron/core/datasets/` (16 文件)
> 分析日期: 2026-06-04

## 架构总览

```
BlendedMegatronDatasetBuilder
  ├── BlendedDataset (多数据集混合)
  │   └── GPTDataset (单个数据集)
  │       └── IndexedDataset (.idx + .bin)
  ├── DataLoader (PyTorch)
  │   ├── num_workers, pin_memory, prefetch_factor
  │   └── collate_fn: packing + padding
  └── Dataset Config: 序列打包 + 文档边界
```

## IndexedDataset — MMap 二进制格式

### 文件格式
```
.bin: 连续 token 二进制数据 (int16/int32)
.idx: 索引文件 (header + per-doc offsets)
      Header: "MMIDIDX\x00\x00" (8 bytes)
      Body:   [doc_offset_0, doc_offset_1, ...] × dtype_size
```

### 特性
- **零拷贝**: `numpy.memmap()` 直接映射磁盘文件到内存
- **随机访问**: 索引支持 O(1) 跳到任意文档
- **支持 S3**: 自动缓存索引文件到本地

```python
class IndexedDataset:
    def get(self, idx: int) -> np.ndarray:
        offset = self.index[idx]
        length = self.index[idx + 1] - offset
        return self.bin_buffer[offset : offset + length]
```

## GPTDataset — LLM 训练数据集

### 序列打包策略

```
Raw documents:
  [doc A: 3000 tok][doc B: 500 tok][doc C: 2000 tok][doc D: 1000 tok]...

Packed sequences (seq_length=2048):
  [A:0-2048][A:2048-3000 | B:0-500 | C:0-1048][C:1048-2000 | D:0-1000 | A:0-8]...

Key: 文档边界 token 用于 attention mask (防止 cross-doc attention)
```

### 关键特性

1. **Document boundary detection**: 每个文档末尾插入 EOD token
2. **Sequence packing**: 高效拼接短文档到同一序列
3. **Cross-contamination prevention**: attention mask 阻止不同文档间的 attention
4. **Fixed-length sequences**: 固定 `seq_length`, 超长文档截断

## BlendedDataset — 多数据集混合

```python
class BlendedDataset:
    """
    按权重从多个子数据集中采样:

    Subset A (code):  60%
    Subset B (wiki):  30%
    Subset C (books): 10%

    支持:
    - 均匀混合 (oversampling 小数据集)
    - 加权混合 (based on weights)
    - 数据集尺寸管理 (epoch-based cycling)
    """
```

### 混合策略

- **Blend**: 每个 batch 按权重从各子集采样
- **Cyclic**: 按顺序循环遍历（小数据集 oversample）
- **Stratified**: 保证每个 batch 各子集比例

## DataLoader 配置

```python
DataLoader(
    dataset,
    batch_size=micro_batch_size,
    num_workers=4,          # 预取进程数
    pin_memory=True,        # 零成本 GPU 传输优化
    prefetch_factor=2,      # 每个 worker 预取 batch 数
    drop_last=True,         # 丢弃不完整 batch
)

# pin_memory: 将数据拷贝到 pinned memory → GPU 异步拷贝更快
# LLM 训练中 DataLoader 开销 <3% (数据准备被计算完全覆盖)
```

## 与 DP/TP/PP 的交互

```
DP:  每个 rank 看到不同数据 (distributed_sampler)
TP:  每个 rank 看到相同数据 (no sharding)
PP:  每个 stage 看到相同 micro-batch (但不同 chunk)
CP:  sequence 沿 context dim 切分

数据分发:
  DataParallel: batch_size_per_rank = global_batch / dp_size
  ContextParallel: sequence 切分 → 每个 rank 看到部分 sequence
```

## BlendedMegatronDatasetBuilder

```python
class BlendedMegatronDatasetBuilder:
    """
    统一构建器:
    1. 解析 datasets 配置 (weights, paths)
    2. 创建子数据集 (GPTDataset, T5Dataset, etc.)
    3. 创建 BlendedDataset
    4. 创建 DataLoader
    5. 管理数据集生命周期 (splitting, caching, shuffling)
    """
```

## 性能考量

| 因素 | 影响 | 建议 |
|------|------|------|
| `num_workers` | CPU 利用率 | 4-8 (不要超过 CPU cores) |
| `pin_memory` | GPU 传输 | **Always True** (零成本) |
| `prefetch_factor` | GPU 等待 | 2-3 (覆盖计算时间) |
| `memmap` | 内存使用 | 零拷贝，适合大数据集 |
| Document boundaries | Attention mask | 影响模型质量 |

## 代码位置

| 文件 | 内容 |
|------|------|
| `indexed_dataset.py` | MMap 二进制格式 (MMIDIDX) |
| `gpt_dataset.py` | GPT 数据集 + 序列打包 |
| `blended_dataset.py` | 多数据集混合 |
| `blended_megatron_dataset_builder.py` | 统一构建器 |
| `megatron_dataset.py` | 基类 |
| `helpers.py` | 辅助函数 |
