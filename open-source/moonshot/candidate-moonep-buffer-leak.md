# Candidate: MoonEP Buffer 错误路径资源泄漏 (#17)

> Issue: [MoonshotAI/MoonEP#17](https://github.com/MoonshotAI/MoonEP/issues/17)
> 状态：`verified-static`（代码追踪确认泄漏点；**mc_fd 修复方案明确，mc_handle 需进一步查 release 机制**）
> 首次发现：2026-08-04
> **竞争风险：低** — 0 评论、0 assignee、reporter `yurekami` 未开 PR（与 #7/#8/#9 的 morluto 自认领不同）

## 1. Bug

`create_nvl_dist_multicast_tensor` → `_create_nvl_multicast_view`（`moonep/buffer.py`）在错误路径上泄漏两类资源：
1. **`mc_fd`（root rank 的 multicast POSIX fd）**：`_exchange_ipc_fds` 若抛 → `os.close(mc_fd)` 跳过 → fd 泄漏
2. **`mc_handle`（所有 rank 的 multicast 对象）**：bind 前 fail（add_device / barrier / bind_map 抛）→ multicast 对象无 finally 释放

## 2. 根因（代码追踪确认）

`_create_nvl_multicast_view`（约 buffer.py:198-241）：

```python
if is_root:
    mc_handle, mc_fd = nvl_multicast_create(size_bytes, world_size)
else:
    mc_handle, mc_fd = 0, None

fds = _exchange_ipc_fds(mc_fd, [0], local_rank, world_size, group)  # ← 可抛
if is_root:
    os.close(mc_fd)        # ← 上一行抛则跳过 → mc_fd 泄漏
root_fd = fds[0]
try:
    if not is_root:
        mc_handle = nvl_multicast_import(root_fd)
finally:
    os.close(root_fd)      # ← root_fd 有保护 ✓

nvl_multicast_add_device(mc_handle)       # ← 可抛
dist.barrier(group=group)                  # ← 可抛
mc_view = nvl_multicast_bind_map(mc_handle, owned_handle, size_bytes, world_size)  # ← 可抛
dist.barrier(group=group)                  # ← 可抛
return mc_view
# ↑ 任一抛则 mc_handle 泄漏（无 finally 释放）
```

**文件内先例**（正确做法）：同文件 `create_nvl_dist_tensor`（:148-167）和 `_map_nvl_dist_tensor`（:116-143）都用 `try/finally` 保护 fd 和 handle 释放。

## 3. ⚠️ Phase ④ 发现：报告者分析有 gap

报告者 #17 说修复"复用已有 binding（`nvl_release_mem_handle` at bindings.cu:57）"。**核实后发现：**
- bindings.cu 暴露的 multicast 相关函数：`nvl_multicast_create/import/add_device/bind_map`
- **没有 `nvl_multicast_release`**——只有 `nvl_release_mem_handle`（VMM mem handle 用，非 multicast 对象）

→ `mc_fd` 修复确实 trivial（try/finally + `os.close`，照先例）。
→ 但 `mc_handle` 的正确释放**可能需要新加 C++ binding**（对应 CUDA driver `cuMulticastRelease`），或确认 multicast 对象靠 mc_view tensor 的 gc/驱动自动回收（需查 CUDA driver 语义）。

报告者"one-line fix + 已有 binding"对 mc_fd 成立，对 mc_handle **不成立**。

## 4. 修复方案

**mc_fd（trivial，明确）**：
```python
# _create_nvl_multicast_view，root rank 分支
if is_root:
    mc_handle, mc_fd = nvl_multicast_create(size_bytes, world_size)
    try:
        fds = _exchange_ipc_fds(mc_fd, [0], local_rank, world_size, group)
    finally:
        os.close(mc_fd)
else:
    ...
```

**mc_handle（需进一步调查）**：
- 选项 A：加 C++ binding `nvl_multicast_release`（调 `cuMulticastRelease`），在 bind 前 fail 的 finally 里调
- 选项 B：确认 mc_view tensor 通过 keepalive 持有 mc_handle，bind 后靠 gc 释放；bind 前 fail 才需显式释放
- 需查 `nvl_multicast_bind_map` 实现确认 mc_view 是否持有 mc_handle 引用

## 5. 待办

- [x] 查 `nvl_multicast_bind_map` 源码（`csrc/nvl_shared_buffer.cuh:252-254`）→ 确认 mc_view 的 deleter 释放 mc_handle；**bind 前 fail 才泄漏**（deleter 还没接管）
- [ ] 若修 mc_handle：需新 C++ binding `nvl_multicast_release`（对 `cuMulticastRelease`）+ Python 暴露，在 bind 前 fail 的 finally 里调
- [ ] 起草 patch：mc_fd 部分可直接写（trivial）；mc_handle 部分待决定（见评估）
- [ ] **无法在 4090 测试**（MoonEP 需 NVSwitch SHARP，H100 NVL 级硬件）—— 测试需在合适硬件或 CI 上由 maintainer 验证
- [ ] （可选）在 issue #17 下 comment 指出 mc_handle release binding 缺失这点（**对外，需你同意**）—— 这反而是个有价值的观察，可能比直接提 PR 更受 maintainer 欢迎

## 6. 最终评估

- I=3（错误路径资源泄漏，非正常路径 bug；severity 低，reporter 自己说）
- C=2（mc_fd trivial）→ 若只修 mc_fd；C=4（mc_fd + 新 C++ binding）→ 完整修
- A=4（无认领；maintainer review 慢 + 报告者分析有 gap 可能引发讨论）
- F=3（CUDA driver 资源管理命中专长，但无硬件无法自测）

**三种路径**：

| 路径 | 内容 | 性价比 | 风险 |
|---|---|---|---|
| A. 只修 mc_fd | try/finally + os.close（照文件内先例） | (3×4×3)/2 = **18** | 低，但只修一半 |
| B. mc_fd + 新 mc_handle release binding | + C++ 改动 cuMulticastRelease | (3×4×3)/4 = **9** | 中，无硬件无法自测 |
| C. issue comment 指出 binding gap | 不提 PR，先互动 | 战略性高 | 零 PR 风险 |

**推荐 C → A**：MoonEP maintainer review 慢（7 PR 堆着）、bug issue 0 回应。对这种仓库，**先 comment 互动建立信任**比直接甩 PR 有效。#17 的 mc_handle gap 是好的 comment 切入点——展示深度阅读，问对问题。互动后若 maintainer 认可方向，再提路径 A 的 PR。
