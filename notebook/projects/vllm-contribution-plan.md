# vLLM 开源贡献计划

> 基于 good first issue 调研的贡献路线图

## 贡献指南要点

### 提交规范
- **DCO**: 每个提交需要 `Signed-off-by:` (`git commit -s`)
- **PR 前缀**: `[Bugfix]`, `[CI/Build]`, `[Doc]`, `[Model]`, `[Frontend]`, `[Kernel]`, `[Core]`, `[Misc]`
- **AI 辅助**: 必须在 PR 中声明使用了 AI 工具，不允许纯 agent PR
- **Review**: 至少一个 maintainer 批准，5 个工作日无回复可在 Slack #vllm-dev 催促

### 开发环境

```bash
# Clone (中国镜像)
git clone https://gh-proxy.com/https://github.com/vllm-project/vllm.git

# Python 3.12 推荐
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

# Python-only 开发 (最快)
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# Linting
uv pip install pre-commit>=4.5.1
pre-commit install

# 测试
uv pip install -r requirements/common.txt -r requirements/dev.txt
pytest tests/
```

## Good First Issues 排名

### Tier 1: 最容易入门

| Issue | 描述 | 类型 |
|-------|------|------|
| [#41230](https://github.com/vllm-project/vllm/issues/41230) | 文档: NIXL KV connector metrics | Doc | ✅ [PR #44438](https://github.com/vllm-project/vllm/pull/44438) |
| [#31414](https://github.com/vllm-project/vllm/issues/31414) | 统一 flashinfer 工具模块 | Refactor |
| [#38734](https://github.com/vllm-project/vllm/issues/38734) | SarvamMLA Transformers v5 兼容 | Model fix |
| [#38425](https://github.com/vllm-project/vllm/issues/38425) | InternVL2 Transformers v5 兼容 | Model fix |
| [#38389](https://github.com/vllm-project/vllm/issues/38389) | Isaac Transformers v5 兼容 | Model fix |
| [#38736](https://github.com/vllm-project/vllm/issues/38736) | Tarsier2 Transformers v5 兼容 | Model fix |

### Tier 2: 稍有挑战

| Issue | 描述 | 类型 |
|-------|------|------|
| [#39479](https://github.com/vllm-project/vllm/issues/39479) | Config hashing refactor 后续任务 | Refactor |
| [#32268](https://github.com/vllm-project/vllm/issues/32268) | 重构 Int8ScaledMM 使用 QuantKey | Refactor |
| [#31249](https://github.com/vllm-project/vllm/issues/31249) | 改进环境变量声明和处理器 | RFC/Design |

### Tier 3: 中等难度

| Issue | 描述 | 类型 |
|-------|------|------|
| [#37753](https://github.com/vllm-project/vllm/issues/37753) | 统一 MoE Oracle 类结构 | Refactor |
| [#32588](https://github.com/vllm-project/vllm/issues/32588) | Whisper 音频 >30s 时间戳 bug | Bugfix |

## 推荐路径

```
第一步: #41230 (文档) → 学习 PR 流程、CI、review
第二步: #31414 (重构) → 熟悉代码结构
第三步: #38734 或类似 (Model fix) → 理解模型注册机制
第四步: 选择一个 Tier 2/3 issue → 深入贡献
```

## 注意事项

- GPU 到位后再开始实际贡献（需要运行测试）
- 可以先阅读代码、理解架构（已有 `notebook/projects/vllm-architecture.md`）
- 每个 PR 保持小而聚焦
- 先在 issue 中评论表达贡献意愿
