# 知识管理 → Claude Code Skills 转化计划

> 状态: Plan, 日期: 2026-06-04

## 什么是 Claude Code Skill

一个 Skill 是: YAML frontmatter 描述 + Markdown 指令体，存放于 `.claude/skills/<name>/SKILL.md`。当用户提及匹配的关键词时自动加载。

## 候选 Skills (按优先级)

### 1. `gpu-experiment` — GPU 实验运行器 (★ 最高优先级)

**触发词**: GPU experiment, A16 benchmark, run gpu test, 跑GPU实验

**功能**:
- 从 46 个脚本库中选择运行对应实验
- 自动推送脚本到 GPU 服务器并运行
- 拉取结果并创建笔记

**资产**:
- 46 个 GPU 脚本 (`tools/gpu_*.py`)
- 19 个实验笔记 (`notebook/experiments/gpu-*.md`)
- GPU 连接凭证 (MEMORY.md)

**转化步骤**:
1. 创建 Skill 目录和 SKILL.md
2. 列出所有可用实验及其快速描述
3. 提供标准化的运行→拉取→笔记工作流
4. 集成 GPU 连接和 conda 激活

### 2. `vllm-v1-nav` — vLLM V1 源码导航

**触发词**: vllm v1 source, vllm architecture, vllm scheduler

**功能**: 快速定位 vLLM V1 的任意子系统代码位置

**资产**:
- 11 篇 vLLM V1 分析笔记 (每个子系统一篇)
- 完整的文件路径和行号索引

**转化步骤**:
1. 从各笔记中提取文件→行号→功能映射
2. 建立关键词→文件映射表
3. 集成到 Skill 文件中

### 3. `megatron-nav` — Megatron-LM 源码导航

**触发词**: megatron source, megatron pp, megatron tp

**资产**: 6 篇 Megatron 分析笔记

### 4. `inference-perf` — LLM 推理性能估算

**触发词**: inference latency, throughput estimate, roofline

**功能**: 给定模型配置，估算推理延迟和吞吐

**资产**:
- Roofline 模型知识
- GPU benchmark 数据
- 各模型实测数据

### 5. `train-perf` — 训练性能模型

**触发词**: training memory, training throughput, ZeRO

**功能**: 估算训练所需 GPU 数量和内存

**资产**: 训练内存计算器、优化器分析

---

## 转化工作流

```
现有资产 (笔记 + 脚本)
  ↓ 提取核心信息
Skill 内容 (SKILL.md)
  ├── YAML frontmatter (name, description, triggers)
  ├── 功能描述
  ├── 使用方法
  └── 参考资料链接 (→ notebook/ 原文)
  ↓ 测试
可用 Skill (用户可调用)
```

## 实施时间线

| Phase | Skills | 状态 |
|-------|--------|:---:|
| Phase 1 | `gpu-experiment` + `vllm-v1-nav` | 本周 |
| Phase 2 | `megatron-nav` + `inference-perf` | 下周 |
| Phase 3 | `train-perf` + others | 后续 |

## 当前知识资产统计

| 类别 | 数量 |
|------|:---:|
| GPU 实验脚本 | 46 |
| vLLM V1 源码笔记 | 11 |
| Megatron-LM 笔记 | 6 |
| 实验笔记 | 19 |
| 项目分析笔记 | 20+ |
| 总学习项 | 205+ |
