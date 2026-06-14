# PyTorch custom_op 源码阅读: 新扩展API

> 2026-06-15 | 源码分析 + RFC + 文档
> 参考: pytorch/pytorch torch/_custom_ops/README.md
> RFC Issue: #118517 "Custom Op extensibility API RFC"
> 作者: Horace He (PyTorch team)

## 1. 为什么 custom_op 重要

旧方式的问题:
- `torch.autograd.Function`: 导致 torch.compile graph break → 性能损失
- `torch.ops` (C++ TORCH_LIBRARY): 需写C++ kernel注册 → 开发成本高
- 两者都无法与 torch.export 兼容 → 部署困难

**custom_op 解决方案**: 纯Python定义 → 全dispatcher集成 → compile/export无缝

```
torch.autograd.Function → graph break ❌
torch.ops (C++) → 需C++ ❌
torch._custom_ops → 纯Python + compile + export + autograd ✅ ← 最佳方案!
```

## 2. 核心API

### 2.1 @custom_op 装饰器

```python
from torch._custom_ops import custom_op

@custom_op("my_namespace::my_op")
def my_op(x: Tensor, scale: float) -> Tensor:
    """docstring is required!"""
    ...

# 注册不同dispatch key的实现:
@my_op.register_impl("cuda")
def my_op_cuda(x, scale):
    return ...  # 自定义CUDA实现

@my_op.register_impl("cpu")
def my_op_cpu(x, scale):
    return ...  # CPU实现

@my_op.register_impl("autograd")
def my_op_autograd(x, scale):
    # 自定义backward公式
    ...

@my_op.register_fake  # meta impl for shape inference
def my_op_meta(x, scale):
    return torch.empty_like(x)
```

### 2.2 关键特性

| 特性 | 说明 |
|------|------|
| 纯Python定义 | 无需C++ TORCH_LIBRARY宏 |
| Dispatcher原生集成 | 获得真实dispatch key → 所有dispatch规则生效 |
| torch.compile无缝 | Dynamo/AOTAutograd可见 → 不graph break |
| torch.export兼容 | 注册meta/fake impl → 可导出部署 |
| 自定义backward | 注册"autograd" impl → 自定义梯度公式 |
| Namespace hygiene | 用户命名空间(my_lib::op) → 不冲突 |

## 3. 与旧方式对比

| 维度 | torch._custom_ops (新) | torch.ops (旧C++) | torch.autograd.Function |
|------|------------------------|-------------------|------------------------|
| Python定义? | ✅ | ❌(需C++ kernel) | ✅ |
| torch.compile? | ✅(无缝) | ✅(需注册) | ❌(graph break!) |
| Dispatcher集成? | ✅(完整) | ✅(完整) | ❌ |
| torch.export? | ✅(with meta) | ✅ | ❌(不支持) |
| Custom backward? | ✅(autograd impl) | ✅ | ✅ |
| vmap/FuncTorch? | ✅(自动) | ✅ | ❌ |
| 开发成本? | 低(纯Python) | 高(C++编译) | 低(但功能受限) |

→ **custom_op是2025+的PyTorch扩展首选方案**

## 4. Dispatcher集成详解

```
PyTorch Dispatcher:
    op → dispatch key → kernel function

传统: torch.ops.aten.add → [CPU, CUDA, Meta, Autograd, ...]
custom: my_namespace::my_op → [CPU(用户注册), CUDA(用户注册), Meta(用户注册), Autograd(用户注册), ...]

所有dispatch规则自动生效:
  - vmap/FuncTorch → 批量规则
  - DynamicShapes → 形状推导
  - AOTAutograd → 编译时autograd
  - FakeTensor → shape inference (torch.compile依赖)
  - 未来dispatch key → 自动兼容
```

## 5. 代码流程

```
用户定义 @custom_op("ns::op")
    │
    ├── torch._C._dispatch_register_kernel(ns, "cuda", kernel)
    │     → PyTorch dispatcher → 真实dispatch entry
    │
    ├── torch.compile (Dynamo)
    │     → 识别custom_op → 不graph break → AOTAutograd处理
    │     → 需要 register_fake (meta impl) → FakeTensor推导shape
    │
    ├── torch.export
    │     → trace custom_op → 保留在exported graph中
    │     → 需要 register_fake → 导出时推导shape
    │     → ExecuTorch/AOTInductor → 运行时查找实现
    │
    └── Autograd
          → register_impl("autograd") → 自定义backward公式
          → 或依赖fallback → PyTorch自动推导(如果有实现)
```

## 6. 与7框架的关系

| 框架 | custom_op应用 |
|------|--------------|
| DeepSpeed | 可能用custom_op注册ZeRO特定op |
| Megatron-LM | 已用C++ ops → 未来可迁移到custom_op |
| vLLM | vLLM V1用torch.compile → custom_op可注册自定义kernel |
| verl | verl用torch.compile → custom_op可注册RL特定op |
| rLLM | 间接(通过verl/vLLM) |
| MindIE | ATB ops → 类似custom_op但昇腾专用 |
| PyTorch | custom_op是PyTorch自身的新扩展API |

### vLLM + torch.compile + custom_op

vLLM V1 默认 `torch.compile(level=3)` → 如果需要自定义op:
```python
# 注册vLLM自定义op (未来方向)
@custom_op("vllm::paged_attention")
def paged_attention(...) -> Tensor:
    ...

@paged_attention.register_impl("cuda")
def paged_attention_cuda(...):
    return torch.ops.vllm.paged_attention(...)  # 调用现有C++ kernel

@paged_attention.register_fake
def paged_attention_meta(...):
    return torch.empty(...)  # shape inference
```

→ vLLM的C++ ops可以通过custom_op包装 → torch.compile完全可见 → 消除graph break

## 7. 实际使用示例

```python
# 最简示例: 注册一个带自定义backward的op
from torch._custom_ops import custom_op
import torch

@custom_op("mylib::scaled_add")
def scaled_add(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    """Adds y * scale to x."""
    ...

@scaled_add.register_impl("cuda")
def scaled_add_cuda(x, y, scale):
    return x + y * scale  # 或调用自定义CUDA kernel

@scaled_add.register_impl("cpu")
def scaled_add_cpu(x, y, scale):
    return x + y * scale

@scaled_add.register_fake
def scaled_add_meta(x, y, scale):
    return torch.empty_like(x)

# 现在可以:
model = torch.compile(model)  # 无graph break!
result = scaled_add(a, b, 0.5)  # 正常调用
exported = torch.export.export(model, ...)  # 可导出!
```

## 8. Timeline与未来

| 时间 | 状态 |
|------|------|
| 2023-2024 | torch._custom_ops 实验性 |
| PyTorch 2.3-2.4 (2024) | @custom_op API稳定化+文档 |
| PyTorch 2.5 (Late 2024) | torch.export兼容+structured backward |
| 2025-2026 | 从_private→public namespace; 更强export+backward支持 |

## 9. 下一步

- [x] 了解custom_op核心API和对比 ✅
- [ ] 阅读pytorch/pytorch torch/_custom_ops/源码
- [ ] 实际注册一个custom_op并测试torch.compile兼容性
- [ ] 分析vLLM现有C++ ops可否通过custom_op包装
- [ ] 对比custom_op vs Triton kernel vs C++ extension

---

Sources:
- [PyTorch Custom Ops Docs](https://pytorch.org/docs/main/custom_ops.html)
- [GitHub README](https://github.com/pytorch/pytorch/blob/main/torch/_custom_ops/README.md)
- [RFC Issue #118517](https://github.com/pytorch/pytorch/issues/118517)
