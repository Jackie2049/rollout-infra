# 分布式训练排错指南

> 实战中最常见的问题和解决方案 — 把踩坑经验系统化

## 1. NCCL 通信问题

### 1.1 NCCL Timeout

```
错误信息:
  NCCL error: unhandled system error, NCCL version 2.x.x
  Watchdog caught collective operation timeout

诊断:
  1. 检查网络连通性
     ssh node1 "ping -c 3 node2"
     ssh node1 "ibstat"  # 检查 IB 卡状态

  2. 检查 NCCL 日志
     NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL python train.py

  3. 检查端口占用
     ssh node1 "netstat -tlnp | grep 29500"

常见原因与修复:
  | 原因                  | 修复                                    |
  |----------------------|----------------------------------------|
  | 节点间网络不通         | 检查 /etc/hosts, 防火墙, IB 链路         |
  | IB 驱动异常            | 重载 ib_uverbs 模块                     |
  | 多网卡路由错误          | 设置 NCCL_SOCKET_IFNAME=eth0            |
  | NCCL 初始化慢          | 增大 NCCL_COMM_BLOCKING=1              |
  | GPU 挂起               | nvidia-smi -q -d PSTATE 看是否异常      |

关键环境变量:
  export NCCL_DEBUG=INFO
  export NCCL_SOCKET_IFNAME=eth0     # 指定网卡
  export NCCL_IB_DISABLE=0           # 启用 IB (0=启用)
  export NCCL_IB_HCA=mlx5_0,mlx5_1  # 指定 IB 卡
  export NCCL_TIMEOUT=1800           # 增大超时 (秒)
  export NCCL_COMM_BLOCKING=1        # 阻塞初始化 (避免死锁)
```

### 1.2 NCCL 版本不匹配

```
错误信息:
  NCCL version mismatch between nodes

修复:
  # 检查所有节点 NCCL 版本
  python -c "import torch; print(torch.cuda.nccl.version())"

  # 统一安装
  pip install nvidia-nccl-cu12==2.x.x -i https://mirrors.aliyun.com/pypi/simple/
```

## 2. 显存问题 (OOM)

### 2.1 CUDA Out of Memory

```
错误信息:
  CUDA out of memory. Tried to allocate X.XX GiB

诊断步骤:
  1. 打印显存峰值
     torch.cuda.max_memory_allocated() / 1024**3  # GB

  2. 使用 tools/memory_estimator.py 估算理论需求

  3. 分析显存 breakdown:
     python -c "
     import torch
     t = torch.cuda.memory_allocated() / 1e9
     r = torch.cuda.memory_reserved() / 1e9
     print(f'Allocated: {t:.1f}GB, Reserved: {r:.1f}GB')
     "

解决方案 (按优先级):
  1. 减小 batch_size / micro_batch_size
  2. 启用 gradient checkpointing
     model.gradient_checkpointing_enable()
  3. 使用 ZeRO 优化器
     DeepSpeedStrategy(zero_stage=2)
  4. 减少 seq_length
  5. 使用 BF16 而不是 FP32
  6. 使用 TP/PP 分片模型

显存公式速查:
  DDP:       16Ψ bytes (Ψ = 参数量)
  ZeRO-1:    4Ψ + 12Ψ/DP
  ZeRO-2:    2Ψ + 14Ψ/DP
  ZeRO-3:    16Ψ/DP (+ activation)
  梯度检查点: activation × sqrt(L)/L
```

### 2.2 Inference OOM (KV Cache)

```
错误信息:
  torch.OutOfMemoryError: CUDA out of memory (vLLM)

vLLM 显存管理:
  --gpu-memory-utilization 0.9   # GPU 显存利用率 (默认 0.9)
  --max-model-len 4096           # 最大序列长度
  --enforce-eager                # 禁用 CUDA graph (省显存)

KV Cache 调优:
  --block-size 16                # KV Cache block 大小
  --swap-space 4                 # CPU swap 空间 (GB)

计算 KV Cache 需求:
  KV Cache per token = 2 × num_layers × num_kv_heads × head_dim × dtype_size
  例: Llama-70B, BF16:
    = 2 × 80 × 8 × 128 × 2 bytes
    = 327,680 bytes ≈ 0.31 MB/token
  1000 tokens → 310 MB
  80000 tokens → 24.8 GB (单个请求!)
```

## 3. 训练数值问题

### 3.1 Loss 爆炸 (NaN/Inf)

```
症状:
  loss = nan  或  loss 突然变为 1e+10

诊断:
  1. 检查 loss curve
  2. 检查梯度范数: torch.nn.utils.clip_grad_norm_
  3. 检查学习率是否过大

常见原因与修复:
  | 原因                  | 修复                                    |
  |----------------------|----------------------------------------|
  | FP16 溢出             | 使用 BF16, 或增大 loss_scale            |
  | 学习率过大            | 降低 LR, warmup 更长                    |
  | 数据问题              | 检查输入是否有 NaN/Inf                   |
  | 梯度爆炸              | gradient clipping (max_norm=1.0)        |
  | 除零                  | 添加 eps=1e-6                           |

混合精度调试:
  torch.autograd.set_detect_anomaly(True)  # 检测 NaN 源头

  # FP16 的 Loss Scaling
  scaler = torch.amp.GradScaler(
      init_scale=2**16,        # 初始 scale
      growth_factor=2.0,       # 成功后增大
      backoff_factor=0.5,      # 失败后缩小
      growth_interval=1000,    # 多少步增大一次
  )
```

### 3.2 Loss 不下降

```
诊断步骤:
  1. 检查学习率:
     - 太大: loss 震荡
     - 太小: loss 不下降
     - 最优: loss 稳定下降

  2. 检查数据:
     - 数据 shuffle 是否正确
     - label 是否正确
     - batch size 是否合适

  3. 检查梯度:
     # 打印梯度范数
     for name, p in model.named_parameters():
         if p.grad is not None:
             print(f"{name}: grad_norm={p.grad.norm():.4f}")

  4. 检查 DDP 同步:
     # 确保所有 rank 梯度一致
     torch.distributed.all_reduce(grad, op=ReduceOp.SUM)

常见问题:
  - 忘记 optimizer.zero_grad()
  - 梯度累积设置错误 (accumulation_steps 不匹配)
  - 数据并行但 batch_size 没有对应增大
  - 混合精度 loss_scale 太小 → 梯度下溢
```

## 4. 性能问题

### 4.1 GPU 利用率低

```
症状:
  nvidia-smi 显示 GPU-Util < 50%

诊断:
  1. 看 GPU-Util 和 Memory-Util 的关系
     GPU-Util 低 + Memory-Util 高 → 计算瓶颈
     GPU-Util 低 + Memory-Util 低 → 数据加载或 CPU 瓶颈

  2. Profile 数据加载
     dataloader_iter = iter(dataloader)
     import time
     t0 = time.time()
     batch = next(dataloader_iter)
     print(f"Data load time: {time.time()-t0:.3f}s")

  3. 使用 Nsight Systems
     nsys profile -o trace python train.py

解决方案:
  | 症状                  | 原因                    | 修复                     |
  |----------------------|------------------------|--------------------------|
  | GPU 空闲等待数据       | DataLoader 慢          | num_workers, prefetch    |
  | GPU 短暂活跃长空闲     | 小 batch_size           | 增大 batch 或 gradient accumulation |
  | GPU 活跃但利用率低     | kernel launch 开销      | torch.compile, 融合操作   |
  | 通信时间长            | NCCL 效率低             | 见 4.2                    |
```

### 4.2 通信瓶颈

```
诊断:
  1. 计算通信占比
     通信时间 / 总时间 > 20% → 有优化空间

  2. 检查是否用了 NVLink
     nvidia-smi nvlink --status

  3. NCCL 调优
     export NCCL_MIN_NRINGS=4            # 增加通道数
     export NCCL_BUFFSIZE=4194304        # 增大 buffer
     export CUDA_DEVICE_MAX_CONNECTIONS=1  # Megatron 推荐

通信优化策略:
  1. 计算-通信重叠 (NCCL async)
  2. 增大 batch_size (摊薄通信开销)
  3. 梯度累积 (减少通信频率)
  4. ZeRO-2 而不是 ZeRO-3 (减少 all-gather)
  5. TP 而不是 DP (NVLink 比 IB 快)
```

## 5. Checkpoint 问题

### 5.1 Checkpoint 保存失败

```
错误信息:
  RuntimeError: Unable to open file
  OSError: [Errno 28] No space left on device

诊断:
  df -h /path/to/checkpoint   # 检查磁盘空间
  ls -la /path/to/checkpoint  # 检查权限

解决方案:
  1. 异步保存 (不阻塞训练)
     # PyTorch
     torch.distributed.barrier()
     if rank == 0:
         import threading
         t = threading.Thread(target=save_checkpoint, args=(model, path))
         t.start()

  2. 分片保存 (每个 rank 保存自己的分片)
     # DeepSpeed / FSDP
     model.save_checkpoint(path)

  3. 压缩保存
     torch.save(state_dict, path, _use_new_zipfile_serialization=True)

  4. safetensors 格式 (更安全、更快)
     from safetensors.torch import save_file
     save_file(state_dict, "model.safetensors")
```

### 5.2 Checkpoint 加载不匹配

```
错误信息:
  Missing key(s) in state_dict
  Unexpected key(s) in state_dict

修复:
  1. strict=False 加载
     model.load_state_dict(state_dict, strict=False)

  2. 部分加载 (fine-tuning)
     pretrained = torch.load("pretrained.pt")
     model_dict = model.state_dict()
     pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
     model_dict.update(pretrained)
     model.load_state_dict(model_dict)

  3. TP/PP 分片加载
     # 每个 rank 只加载自己的分片
     shard_path = f"checkpoint/rank_{rank}.pt"
     state = torch.load(shard_path, map_location="cpu")
```

## 6. 数据加载问题

### 6.1 DataLoader 慢

```
症状:
  GPU 利用率低, 数据加载是瓶颈

优化:
  1. 增加 num_workers
     DataLoader(dataset, num_workers=8, prefetch_factor=2)

  2. pin_memory
     DataLoader(dataset, pin_memory=True)

  3. 持久化 workers (避免重启)
     DataLoader(dataset, persistent_workers=True)

  4. 内存映射 (大数据集)
     # WebDataset / mmap
     dataset = WebDataset("data/{00000..00999}.tar")

  5. CUDA prefetcher
     class CUDAPrefetcher:
         def __init__(self, loader):
             self.loader = iter(loader)
             self.stream = torch.cuda.Stream()
             self.preload()

         def preload(self):
             with torch.cuda.stream(self.stream):
                 self.next_batch = next(self.loader)
                 self.next_batch = [x.cuda(non_blocking=True)
                                    for x in self.next_batch]
```

## 7. 快速诊断 Checklist

```
遇到问题时按顺序检查:

□ 1. 显存: nvidia-smi 看 GPU-Util 和 Memory
□ 2. 磁盘: df -h 看空间
□ 3. 网络: ibstat, ping 节点
□ 4. 日志: NCCL_DEBUG=INFO 看通信日志
□ 5. 梯度: 检查 grad_norm 是否 NaN
□ 6. Loss: 画 loss curve 看趋势
□ 7. 数据: 检查输入输出是否正常
□ 8. 版本: torch, nccl, cuda 版本匹配
□ 9. 随机种子: 是否设置了 torch.manual_seed
□ 10. 环境变量: CUDA_DEVICE_MAX_CONNECTIONS 等

救命命令:
  # 强制清理 GPU
  nvidia-smi --gpu-reset -i 0

  # 查看进程
  nvidia-smi pmon -c 5

  # 查看僵尸进程
  fuser -v /dev/nvidia*

  # 重置 NCCL
  export NCCL_SHM_DISABLE=1
```

## 参考

- [PyTorch DDP Troubleshooting](https://pytorch.org/docs/stable/notes/ddp.html)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [DeepSpeed Troubleshooting](https://www.deepspeed.ai/troubleshooting/)
- [vLLM FAQ](https://docs.vllm.ai/en/latest/getting_started/faq.html)
