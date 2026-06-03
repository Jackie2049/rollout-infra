# rollout-infra — AI Infra 工程师成长之路

> 从零开始，系统学习 AI 基础设施，成为一名强悍的 AI Infra 工程师

## 仓库运作规则

### 目录结构

```
rollout-infra/
├── notebook/       学习笔记和源码分析（Markdown 格式）
├── tools/          可运行的脚本和工具（Python）
└── diary/          每日工作日志（每天一个 YYYY-MM-DD.md 文件）
```

### 日志记录规范

- **文件**: `diary/YYYY-MM-DD.md`，每天一个文件
- **颗粒度**: 每完成一件事（学一个知识点、做一个实验、写一个工具）就记录一条
- **格式**: `HH:MM — 标题` + 2-3 句话，包含做了什么、为什么做、得出了什么结论或思考
- **顺序**: 按时间正序排列（最早的在最前面）
- **要求**: 不要太简略（不能只列名字），也不要太冗长；要能看到进展、结论和思考

### 笔记记录规范

- **基础知识**: `notebook/fundamentals/` — 每篇包含目标、核心概念、实践记录、参考资料
- **项目源码**: `notebook/projects/` — 源码阅读和分析笔记
- **实验记录**: `notebook/experiments/` — GPU 实验数据和结论
- **规划文档**: `notebook/roadmap.md`, `notebook/knowledge-map.md`

### 工具沉淀规范

- 每个工具独立可运行（有 `if __name__ == "__main__"` 入口）
- 包含 docstring 说明用法
- 验证状态标注在 notebook/README.md 中

### Git 管理规范

- 每完成一个有意义的任务就 commit + push
- commit message 简要说明变更内容
- 所有变更通过 git 管理，不手动管理文件

### 环境信息

- **本地 conda**: `ai-infra` (Python 3.11, macOS)
- **GPU 服务器**: matpool.com A16 15GB, `myconda` 环境 (Python 3.9, PyTorch 2.0+cu117)
- **GPU 连接**: `source /root/miniconda3/bin/activate myconda`（必须激活，不能用系统 Python）
- **镜像源**: pip 用清华/阿里云镜像（中国大陆网络）

## 统计

| 类别 | 数量 |
|------|------|
| 基础知识笔记 | 22 篇 |
| 项目源码阅读 | 9 篇 |
| 实验记录 | 1 篇 |
| 实用工具脚本 | 12 个 |
| **总计** | **45 项** |

## GitHub

https://github.com/Jackie2049/rollout-infra
