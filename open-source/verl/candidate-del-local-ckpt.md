# Candidate: del_local_ckpt_after_load 不删本地 checkpoint

> Issue: [volcengine/verl#7213](https://github.com/volcengine/verl/issues/7213)
> 状态：`verified-static`（代码追踪确认 bug + 根因；**尚未实跑复现**）
> 首次发现：2026-08-03

## 0. ⚠️ 竞争风险（必读）

- Issue 作者 **`yuyz-cyber`**（2026-07-31 开 issue，无 assignee / 0 评论 / 无关联 PR）
- 作者在 issue 末尾明确写：**"I can submit a focused fix with CPU distributed regression coverage."**
- 即：作者**已经认领意图**，但 2 天没动静。
- **策略含义**：不能闷头闷头做完直接提 PR（会撞车 + 失礼）。需要先在 issue 下 comment 协调（问作者是否还打算做，或我们来做）。**这一步是对外动作，需你同意。**

## 1. Bug 现象

`trainer.del_local_ckpt_after_load=True` + `resume_mode: auto` 时，resume 后 `global_step_<N>/actor`（及 critic）目录**不被删除**，磁盘被历史 resume checkpoint 填满。

## 2. 根因（代码追踪 + issue 正文双重确认）

调用链（`verl/trainer/ppo/ray_trainer.py:1092`）：

```python
actor_path = os.path.join(global_step_folder, "actor")   # ← 纯本地路径，无 hdfs://
self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=cfg.del_local_ckpt_after_load)
```

FSDP 侧（`verl/utils/checkpoint/fsdp_checkpoint_manager.py`）：

```python
remote_model_path = os.path.join(local_path, f"model_world_size_{...}_rank_{...}.pt")
local_model_path  = copy_to_local(remote_model_path)      # 本地 no-op，仍无 hdfs://
if self.rank == 0 and del_local_after_load:
    os.remove(local_model_path) if is_non_local(local_model_path) else None   # ← 恒 False
```

`is_non_local`（`verl/utils/fs.py:34`）= `path.startswith("hdfs://")`。

**根因**：`is_non_local` 检查的是 `copy_to_local` 返回的**本地路径**（恒无 `hdfs://` 前缀）→ 恒 `False` → `os.remove()` 永不执行。

Megatron 侧（`megatron_checkpoint_manager.py:763,977`）额外问题：`local_path` 是**角色目录**（actor/），即使条件成立，`os.remove(目录)` 也会因是目录而失败。

## 3. 受影响位置

| 文件 | 行 | 问题 |
|---|---|---|
| `fsdp_checkpoint_manager.py` | 222-226 | `is_non_local(local_model_path)` 恒 False |
| `megatron_checkpoint_manager.py` | 763-765 | 同上 + `os.remove(目录)` 不可用 |
| `megatron_checkpoint_manager.py` | 977-979 | 同上 |

## 4. 期望行为（issue 正文，比我的初稿更精细）

- 所有 rank load 完（**barrier 之后**）→ rank 0 删除 actor 或 critic **角色目录**（`shutil.rmtree`，不是 `os.remove` 单文件）
- **保留** `global_step_<N>` 目录和 `data.pt`（driver 仍需要 dataloader/rng 状态）
- 附带 CPU 分布式回归测试

## 5. Patch 设计（待与 maintainer/作者对齐）

两个方案：

**方案 A：checkpoint manager 内部删**（各 manager 删自己的角色目录）
- 把删除块移到 `torch.distributed.barrier()` **之后**
- 改 `os.remove(文件)` 为 `shutil.rmtree(local_path)`（`local_path` 即 actor/ 角色目录）
- 修正条件：删掉误导性的 `is_non_local` guard（或改成"仅当路径是本次 resume 的本地 ckpt 才删"）

**方案 B：trainer 层集中删**（`ray_trainer.py` 在 load 返回后由 driver/rank0 `shutil.rmtree(actor_path)`）
- 同步依赖 manager 内部已有的 barrier
- 更集中，但跨 manager 的"删不删"逻辑散到 trainer

倾向 **方案 A**（同步点在 manager 内部 barrier 后，最准确）。具体实现细节需在 PR 前与作者/maintainer 对齐——尤其 `del_local_ckpt_after_load` 对**纯本地**路径也生效这一语义变化要写进文档。

## 6. 待办

- [ ] **（对外，需你同意）** 在 #7213 下 comment：询问作者 `yuyz-cyber` 是否还打算提交，如不方便我们来接手（附根因分析 + 修复方案）
- [ ] 实跑复现：4090 单卡 FSDP 训练 → 存 ckpt → `del_local_ckpt_after_load=True` resume → 看 `actor/` 是否残留
- [ ] 应用 patch 后复现 → 确认目录被删、`data.pt` 保留
- [ ] 回归：`del_local_ckpt_after_load=False`（默认）行为不变
- [ ] 补 CPU 分布式回归测试

## 7. 评估（已下调）

- 原评：I=3 C=2 A=5 F=4 → 30
- **修正**：作者已认领意图 → **A 从 5 降到 2** → 性价比 30 → **12**
- 结论：仍是干净 P0，但**需先协调归属**，否则会撞车。
