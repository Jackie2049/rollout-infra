# Kubernetes GPU 调度基础

> 在 K8s 上管理 GPU 资源 — 大规模训练的编排层

## 1. 为什么需要 K8s GPU 调度

```
场景: 公司有 100 台 GPU 服务器，多个团队共享
问题:
  - 谁在用哪些 GPU？
  - 任务排队等待
  - GPU 利用率低 (平均 ~30-40%)
  - 故障恢复复杂

解决: K8s + GPU 调度器
  - 统一资源池
  - 自动调度和排队
  - 故障自动恢复
  - 多租户隔离
```

## 2. 基础概念

### 2.1 GPU Device Plugin

```yaml
# NVIDIA Device Plugin — 让 K8s 识别 GPU
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin
spec:
  template:
    spec:
      containers:
      - name: nvidia-device-plugin
        image: nvcr.io/nvidia/k8s-device-plugin:v0.14.0
        # 向 K8s 报告每个节点的 GPU 数量和型号
```

### 2.2 请求 GPU 资源

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training
spec:
  containers:
  - name: training
    image: pytorch/pytorch:2.2.0-cuda12.1
    resources:
      limits:
        nvidia.com/gpu: 4  # 请求 4 个 GPU
    command: ["torchrun", "--nproc_per_node=4", "train.py"]
```

### 2.3 GPU 类型选择

```yaml
# 指定 GPU 型号
nodeSelector:
  gpu-type: "a100-80g"

# 或使用 node affinity
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: nvidia.com/gpu.product
          operator: In
          values: ["A100-SXM4-80GB"]
```

## 3. 高级调度特性

### 3.1 MIG (Multi-Instance GPU)

```yaml
# A100/H100 支持将 1 个 GPU 切分为多个实例
# 例: 1×A100-80G → 7×MIG 实例 (各 10GB)
resources:
  limits:
    nvidia.com/mig-1g.10gb: 1  # 使用 1/7 个 GPU
```

### 3.2 GPU 共享

```
时间分片 (Time-Slicing):
  1 个 GPU 被多个 Pod 共享（时间片轮转）
  适合推理任务（利用率不高时）

配置:
  nvidia.com/gpu.replicas: 10  # 1 个 GPU 模拟为 10 个
```

### 3.3 Gang Scheduling

```
分布式训练需要所有 Pod 同时启动:
  如果只调度了 3/4 个 Pod → 训练无法开始 → GPU 浪费

Gang Scheduling:
  要么所有 Pod 同时启动，要么全部不启动
  使用 Volcano 或 Kube-batch 调度器
```

## 4. 训练框架集成

### 4.1 Kubeflow

```
Kubeflow Training Operator:
  - 原生支持 PyTorch/Megatron 分布式训练
  - 自动配置 MASTER_ADDR, RANK, WORLD_SIZE
  - 支持 Pod 故障自动重启

PyTorchJob 示例:
  apiVersion: kubeflow.org/v1
  kind: PyTorchJob
  spec:
    pytorchReplicaSpecs:
      Master:
        replicas: 1
        template:
          spec:
            containers:
            - name: pytorch
              resources:
                limits:
                  nvidia.com/gpu: 8
      Worker:
        replicas: 7
        ...
```

### 4.2 多容器 Pod 模式

```
Sidecar 模式:
  主容器: 训练进程
  Sidecar: 日志收集、监控、数据同步

Init Container:
  数据下载/预处理
  → 完成后启动训练容器
```

## 5. 常见挑战

| 挑战 | 解决方案 |
|------|---------|
| GPU 利用率低 | MIG + 时间分片 + 自动缩放 |
| 分布式训练调度 | Gang Scheduling (Volcano) |
| 存储访问慢 | CSI driver + 分布式缓存 |
| 网络配置复杂 | CNI plugin + RDMA 支持 |
| 监控缺失 | DCGM exporter + Prometheus + Grafana |

## 6. 学习要点

1. **Device Plugin 是基础** — 让 K8s 识别和管理 GPU
2. **Gang Scheduling 对分布式训练必需** — 所有 Pod 同时启动
3. **MIG 可以提高 GPU 利用率** — 将大 GPU 切给小任务
4. **Kubeflow 简化了 ML 工作流** — 原生支持 PyTorch 分布式训练

## 参考

- [NVIDIA GPU Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
- [Kubeflow Training Operator](https://github.com/kubeflow/training-operator)
- [Volcano Scheduler](https://volcano.sh/en/docs/)
- [NVIDIA GPU Sharing](https://docs.nvidia.com/datacenter/cloud-native/gpu-sharing.html)
