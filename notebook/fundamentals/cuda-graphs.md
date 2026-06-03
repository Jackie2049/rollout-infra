# CUDA Graphs

> 减少 CPU launch 开销 — GPU 推理加速的关键技术

## 1. 问题：Kernel Launch 开销

```
普通 CUDA 执行:
  CPU 发起 kernel → GPU 执行 → CPU 发起下一个 kernel → ...

每次 kernel launch 的 CPU 端开销: ~5-20 μs

推理场景 (如 GPT-1.3B):
  每个 token 生成需要 ~100+ 个 kernel
  CPU launch 开销: 100 × 10μs = 1ms
  如果 GPU 计算只需 2ms → 50% 时间花在 launch 上!

小 batch 推理尤其严重:
  batch=1: 计算 2ms, launch 1ms → 33% 开销
  batch=32: 计算 20ms, launch 1ms → 5% 开销
  batch=256: 计算 80ms, launch 1ms → 1% 开销

CUDA Graphs 解决方案:
  预先录制所有 kernel → 一次性提交 → 消除逐个 launch 的开销
```

## 2. CUDA Graphs 原理

### 2.1 基本概念

```
传统执行模式:
  CPU → GPU: launch kernel A
  CPU → GPU: launch kernel B
  CPU → GPU: launch kernel C
  每次 launch 都需要 CPU→GPU 通信

CUDA Graph 执行模式:
  1. 录制阶段 (Instantiate):
     CPU 录制所有 kernel 依赖关系 → 生成执行图 (Graph)
     只执行一次

  2. 重放阶段 (Launch):
     CPU → GPU: launch 整个 Graph
     GPU 自动按图执行所有 kernel
     只需一次 CPU→GPU 通信

  关键: 输入数据的地址不变, 只改内容
```

### 2.2 PyTorch CUDA Graph API

```python
import torch

# 静态形状 + 静态地址
static_input = torch.randn(1, 512, device='cuda')
static_output = torch.randn(1, 512, device='cuda')

# 录制 CUDA Graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# 重放 (每次只改输入数据的内容, 地址不变)
for batch in dataloader:
    static_input.copy_(batch)  # 复制数据到静态地址
    g.replay()                  # 重放 Graph
    result = static_output      # 读取输出
```

### 2.3 vLLM 中的 CUDA Graphs

```
vLLM 大量使用 CUDA Graphs 加速推理:

1. Decode 阶段:
   - 每 token 生成只需要少量计算
   - kernel launch 开销占比大
   - 使用 CUDA Graph 消除 launch 开销

2. Batch Size 分桶:
   - CUDA Graph 要求固定输入形状
   - vLLM 预录制多个 batch size 的 Graph
   - 例如: batch = [1, 2, 4, 8, 16, 32, 64, ...]
   - 实际 batch=5 时使用 batch=8 的 Graph (浪费 3 个 slot)

3. 关键代码路径:
   - 捕获: model.capture_model()
   - 重放: model.forward() → 检查是否有匹配的 Graph
```

## 3. 使用场景

### 3.1 适合 CUDA Graph 的场景

```
✅ 推理 (特别是小 batch):
   - 每步计算量小, launch 开销占比大
   - 模型结构固定, kernel 序列不变

✅ 训练中的固定模式:
   - embedding lookup
   - 固定 shape 的 MLP 层

✅ 重复执行相同操作:
   - 每 step 执行相同的 kernel 序列
```

### 3.2 不适合 CUDA Graph 的场景

```
❌ 动态形状:
   - 变长序列 (除非 pad 到固定长度)
   - 动态 batch size (除非分桶)

❌ 条件分支:
   - Graph 中不能有 if/else
   - 所有 kernel 都必须执行

❌ CPU→GPU 同步:
   - Graph 执行期间不能回 CPU
   - 所有数据流在 GPU 内完成

❌ 大 batch 训练:
   - 计算时间长, launch 开销占比小
   - 收益不明显
```

## 4. 内存影响

```
CUDA Graph 的显存开销:

1. Graph 本身的元数据: 通常 < 10 MB
2. 固定输入/输出缓冲区:
   - 每个 Graph 需要专属的输入输出 tensor
   - 如果录制多个 batch size, 每个都要分配

3. 临时缓冲区:
   - Graph 执行过程中的中间结果
   - 不随 step 释放 (因为是预录制的)

vLLM 中的内存管理:
   --enforce-eager  # 禁用 CUDA Graph (省显存)
   默认: 启用 CUDA Graph

   小 GPU (如 A16 15GB) 可能需要 --enforce-eager
   因为录制多个 batch size 的 Graph 会占用大量显存
```

## 5. 性能收益

```
典型收益 (推理):

LLaMA-7B, batch=1:
  无 CUDA Graph: ~25 ms/token
  有 CUDA Graph: ~18 ms/token
  加速: ~28%

LLaMA-7B, batch=16:
  无 CUDA Graph: ~80 ms/batch
  有 CUDA Graph: ~72 ms/batch
  加速: ~10%

LLaMA-7B, batch=128:
  无 CUDA Graph: ~400 ms/batch
  有 CUDA Graph: ~395 ms/batch
  加速: ~1%

结论: batch 越小收益越大
```

## 6. 调试技巧

```python
# 检查 CUDA Graph 是否生效
# vLLM 启动时日志中会有:
#   "Capturing CUDA graphs for decode..."

# 禁用 CUDA Graph
# vLLM:
python -m vllm.entrypoints.openai.api_server \
  --model ... --enforce-eager

# PyTorch: 不使用 torch.cuda.graph 即可

# 常见问题:
# 1. Graph 录制失败:
#    原因: 模型中有动态控制流
#    解决: 确保模型 forward 没有条件分支

# 2. OOM during capture:
#    原因: 多个 batch size 的 Graph 占用太多显存
#    解决: 减少 capture 的 batch size 数量

# 3. 结果不正确:
#    原因: 输入地址改变了 (没有用 copy_)
#    解决: 确保用 copy_() 而不是重新赋值
```

## 参考

- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [vLLM CUDA Graph Implementation](https://github.com/vllm-project/vllm)
