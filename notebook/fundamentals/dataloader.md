# DataLoader 优化技术

> 训练中的数据加载瓶颈 — 为什么 GPU 在等数据

## 1. 问题：数据加载成为瓶颈

```
理想训练循环:
  GPU 计算 → GPU 计算 → GPU 计算 → ...

实际训练循环:
  GPU 计算 → 等数据 → GPU 计算 → 等数据 → ...
             ↑ I/O 瓶颈！

数据加载时间占比:
  小模型 (1.3B): ~10-30% 时间在等数据
  大模型 (70B):  ~5-10% (计算时间更长，掩盖了 I/O)
  多模态 (图像+文本): ~20-40% (图像解码慢)
```

## 2. Prefetch 预取

### 2.1 基本原理

```
无 Prefetch:
  时间: |--计算step1--|---加载数据2---|--计算step2--|---加载数据3---|
  GPU 在加载期间空闲！

有 Prefetch:
  时间: |--计算step1--|--计算step2--|--计算step3--|
        | 加载数据2 | 加载数据3 | 加载数据4 | ← 后台线程
  GPU 几乎不等待！
```

### 2.2 PyTorch DataLoader Prefetch

```python
# PyTorch DataLoader 内置 prefetch
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,       # 多进程并行加载
    prefetch_factor=2,   # 每个 worker 预取 2 个 batch
    pin_memory=True,     # 使用 page-locked 内存，加速 CPU→GPU 传输
)

# 效果:
#   num_workers=0: 单线程加载，GPU 等待
#   num_workers=4: 4 个进程并行预取
#   prefetch_factor=2: 总共缓存 4×2=8 个 batch
```

### 2.3 CUDA Prefetch

```python
# 将数据预取到 GPU
class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            # 异步拷贝到 GPU
            self.next_batch = {
                k: v.to(device="cuda", non_blocking=True)
                for k, v in self.next_batch.items()
            }

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            self.preload()  # 立即开始预取下一个
        return batch
```

## 3. Memory Mapping (mmap)

### 3.1 原理

```
普通文件读取:
  磁盘 → 内核缓冲区 → 用户空间 → Python 对象 → Tensor
  多次拷贝，慢

mmap 读取:
  磁盘 → GPU/CPU 直接访问映射区域
  零拷贝（或极少拷贝），快
```

### 3.2 NumPy mmap

```python
# 大文件 mmap 加载
data = np.memmap("large_dataset.npy", dtype=np.float32, mode="r", shape=(1_000_000, 1024))

# 只加载需要的部分 (按需 page fault)
batch = data[0:32]  # 只读取 32 行，不加载整个文件
```

### 3.3 HuggingFace Datasets 的 mmap

```python
from datasets import load_dataset

# HuggingFace datasets 默认使用 memory-mapped Apache Arrow 格式
dataset = load_dataset("allenai/c4", "en", streaming=False)
# 文件在磁盘上，访问时按需读取 → 内存占用极小
# 可以处理比内存大得多的数据集
```

## 4. Streaming Dataset

### 4.1 为什么需要 Streaming

```
问题: 某些数据集 > 1 TB (如 C4, RedPajama)
不可能全部下载到本地磁盘

解决: Streaming 模式
  - 边下载边训练
  - 不需要完整数据集在本地
  - 数据顺序随机化需要特殊处理
```

### 4.2 HuggingFace Streaming

```python
dataset = load_dataset("allenai/c4", "en", streaming=True)
# 不下载，直接从 HuggingFace Hub 流式读取

for batch in DataLoader(dataset, batch_size=32):
    train_step(batch)
```

### 4.3 Megatron-LM 的数据加载

```
Megatron-LM 使用预处理的数据格式:
  1. 预处理: 原始文本 → tokenize → 二进制格式 (.bin/.idx)
  2. 索引文件 (.idx): 记录每个 document 的 offset 和长度
  3. 数据文件 (.bin): 连续存储 token IDs
  4. 加载时: 通过索引文件定位，mmap 读取

优势:
  - 极快的随机访问 (O(1) seek)
  - 低内存占用 (mmap)
  - 支持高效 shuffle (只 shuffle 索引)
```

## 5. 分布式文件系统上的 DataLoader

### 5.1 瓶颈

```
Lustre/GPFS 上的数据加载:
  - 高聚合带宽 (100+ GB/s)
  - 高延迟 (~ms 级 vs 本地 SSD ~μs 级)
  - 元数据操作 (ls, stat) 很慢
  - 小文件性能差

典型问题:
  - 1000 个 worker 同时读取 → 元数据服务器过载
  - 小文件 (每个样本一个文件) → 效率极低
  - 随机读取 → 缓存失效
```

### 5.2 最佳实践

```
1. 预处理为少量大文件 (如 WebDataset tar 格式)
   好处: 减少元数据操作，顺序读取

2. 缓存到本地 SSD
   首次从分布式 FS 读取 → 缓存到本地
   后续从本地读取 → 快

3. 使用 blended datasets
   多个小数据集混合为一个大流

4. 避免全局 shuffle
   用 local shuffle + epoch shuffle 替代
```

## 6. WebDataset 格式

```
传统: 每个样本一个文件
  data/000001.json  → 100 万个小文件 → 元数据灾难

WebDataset: 样本打包到 tar 文件
  shard-000000.tar  (含 10000 个样本)
  shard-000001.tar
  ...

每个 tar 内:
  000001.jpg  000001.json  000001.txt
  000002.jpg  000002.json  000002.txt
  ...

优势:
  - 顺序读取 (高带宽利用)
  - 少量大文件 (低元数据开销)
  - 天然支持 streaming
```

## 7. 学习要点

1. **Prefetch 是最基本的优化** — 多 worker + pin_memory + prefetch_factor
2. **mmap 实现零拷贝加载** — 处理大数据集的关键
3. **Streaming 用于超大数据集** — 边下载边训练
4. **Megatron 的二进制格式** — 预处理 + mmap + 索引，最高效
5. **分布式 FS 上要避免小文件** — 用 tar/WebDataset 打包
6. **I/O 和计算可以完全重叠** — 正确配置后数据加载不再是瓶颈

## 参考

- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html)
- [WebDataset](https://github.com/webdataset/webdataset)
- [Megatron-LM Data Preprocessing](https://github.com/NVIDIA/Megatron-LM#data-preprocessing)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
