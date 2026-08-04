# Candidate: dynamic-cp ceil 切分致高 DP rank 空 (#6786)

> Issue: [volcengine/verl#6786](https://github.com/volcengine/verl/issues/6786)
> 标签：bug
> 状态：`verified-works`（纯逻辑复现 + even 切分 patch 4 用例验证 + merge 安全确认）
> 首次发现：2026-08-04
> **竞争风险：低** — 0 评论、0 assignee、reporter `liujia-cc` 未认领修复

## 1. Bug

`verl/utils/megatron_utils.py` 的 dynamic-cp 切分用 `math.ceil` 分配 seq 到 local_dp_rank，当 `len(seq_len_effective)` 不被 `local_dp_size` 整除时，**高 local_dp_rank 收到空 indices** → 下游 `index_select_tensor_dict` / nested tensor 崩。

reporter 例：`len=9, local_dp_size=4` → ranks 收到 `[3,3,3,0]`，rank 3 空。

## 2. 根因（代码追踪确认）

`megatron_utils.py` 切分块（约 :1787-1792）：

```python
num_seq_per_local_cp = math.ceil(len(seq_len_effective) / local_dp_size)
start_idx = local_dp_rank * num_seq_per_local_cp
end_idx = min(start_idx + num_seq_per_local_cp, len(seq_len_effective))
selected_indices = indices[start_idx:end_idx]
```

`ceil(9/4)=3` → rank 3: `start=9, end=min(12,9)=9` → `indices[9:9]=[]`。经典 ceil 切分末尾空。

**边界**：reporter 说 `len < local_dp_size` 也触发——但 line 1768 `if len < dp_size: return batch` 已 early return，而 `local_dp_size = dp_size // local_cp_size ≤ dp_size`，所以进入切分时必有 `len ≥ dp_size ≥ local_dp_size`，第二点实际不触发（reporter 泛指）。

## 3. Patch（已实现 + 4 用例验证）

even 切分（floor / floor+1 均匀分配），消除空 rank：

```python
num_seq = len(seq_len_effective)
base = num_seq // local_dp_size
remainder = num_seq % local_dp_size
if local_dp_rank < remainder:
    start_idx = local_dp_rank * (base + 1)
    end_idx = start_idx + base + 1
else:
    start_idx = local_dp_rank * base + remainder
    end_idx = start_idx + base
selected_indices = indices[start_idx:end_idx]
```

## 4. 验证结果（最小复现脚本 `/tmp/repro_6786.py`）

| 用例 | buggy (ceil) | fix (even) |
|---|---|---|
| len=9, D=4（reporter 例） | `{[0,1,2],[3,4,5],[6,7,8],[]}` ❌ 空 | `{[0,1,2],[3,4],[5,6],[7,8]}` ✅ |
| len=3, D=4（len<D，物理限制） | 两者都有空（该路径被 early return 拦） | — |
| len=8, D=4（整除回归） | `{[0,1],[2,3],[4,5],[6,7]}` | **完全一致** ✅ |
| len=16, D=4（整除回归） | 均匀 | **完全一致** ✅ |

**覆盖检查**（7 组 L/D）：当 `len ≥ local_dp_size`，even 切分**全覆盖 + 无空 rank**；ceil 在 `len % D != 0` 时有空。

## 5. merge 安全性确认

`dynamic_cp_merge_output`（:1812-1824）用 `all_gather_object` + `unbind` 拼接，**不依赖均匀切分**，只要每 rank 非空即可。even 切分保持连续性 + 每 rank 非空 → merge 正确还原。patch 不引入回归。

## 6. 待办

- [ ] 补单测：在 `tests/` 加 dynamic-cp 切分用例（需 stub NestedTensor/分布式，或抽切分逻辑为可测纯函数）
- [ ] pre-commit + 查重（`gh pr list --search "6786 in:body"`）
- [ ] **（对外，需你同意）** 在 fork 开 PR
- [x] patch 在本地 `verl-pr/verl_main/verl/utils/megatron_utils.py`，未碰远端

## 7. 评估

- I=3（特定配置下崩溃，需 `len % local_dp_size != 0`；非默认路径） 
- C=2（纯切分逻辑替换 + 单测；merge 已确认安全）
- A=4（无认领；reporter repro 薄但给了行号；修复方向标准）
- F=4（Megatron 路径，我们 Megatron 经验契合，纯逻辑本地可验证）
- **性价比 = (3×4×4)/2 = 24** —— 强候选，patch 已就绪
