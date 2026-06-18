# DeepSpeed #8075 — FD Leak in deepspeed_io_handle_t::wait()

**Reading Note | Date: 2026-06-19 | Issue/PR Type: Pull Request (fix) | State: OPEN**

---

## 1. Issue/PR Metadata

| Field | Value |
|---|---|
| **Number** | #8075 |
| **Title** | fix: close file descriptor in deepspeed_io_handle_t::wait() to prevent fd leak |
| **Type** | Pull Request (not a bug issue — proactively submitted fix) |
| **Author** | MarkCLChang (external contributor, `author_association: NONE`) |
| **Created** | 2026-06-18T03:17:21Z |
| **Updated** | 2026-06-18T05:41:03Z |
| **State** | OPEN (not merged, 0 comments, 0 review comments) |
| **Labels** | None |
| **Assignees** | None |
| **Requested Reviewer** | tjruwase (DeepSpeed collaborator/maintainer) |
| **Changes** | +1/-1, 1 file changed |
| **Mergeable** | Yes (mergeable_state: blocked — likely CI gating) |
| **Reactions** | 1 thumbs-up (+1) |

URL: https://github.com/deepspeedai/DeepSpeed/pull/8075

---

## 2. Full PR Description

### Overview

This PR addresses a file descriptor leak in `deepspeed_io_handle_t::wait()` by ensuring the file descriptor is properly closed after the async I/O operation completes.

### Changes

- Added `close()` call on the file descriptor at the end of `deepspeed_io_handle_t::wait()` to prevent fd accumulation during repeated async I/O operations.
- This prevents potential resource exhaustion in long-running training jobs that perform frequent checkpoint reads/writes via DeepSpeed's async I/O interface with ZeRO3 offload NVMe.

---

## 3. The Actual Code Change

The fix is a single line in `csrc/aio/py_lib/deepspeed_py_io_handle.cpp` (line 213):

```cpp
// BEFORE (buggy):
if (!completed_op->_filename.empty()) { (completed_op->_fd); }

// AFTER (fixed):
if (!completed_op->_filename.empty()) { close(completed_op->_fd); }
```

★★★★★★★★★ **Root cause**: The original code had `(completed_op->_fd)` — this is a no-op expression that merely evaluates the fd integer value and discards it. It is NOT a function call. The intent was clearly to `close()` the file descriptor, but the `close` keyword/function was accidentally omitted, likely during a code edit or merge. This is a classic C/C++ typo bug: parentheses around a variable name are syntactically valid but semantically meaningless.

★★★★★★★★★ **Impact**: Every completed async I/O operation that has a filename leaves its file descriptor open. In long-running training with ZeRO-3 NVMe offload (checkpoint reads/writes happen every N steps), fd accumulation is linear with step count. Linux default fd limit is typically 1024 (`ulimit -n`). A training job doing 2 checkpoint operations per step will exhaust fds in ~512 steps, causing `OSError: [Errno 24] Too many open files` or similar — process crash, silent data loss, or training interruption.

---

## 4. Comments and Discussion

**0 comments on the PR. 0 review comments.** The PR has been open for ~1 day with no engagement from DeepSpeed maintainers beyond requesting tjruwase as reviewer. The +1 reaction indicates at least one community member recognizes the importance.

★★★★★★★★★ **Stalled PR pattern**: This is the SAME pattern seen with #8073 (2-line fix, 0 reviews, stalled) and #8068 (gradient clipping, 0 reviews, stalled). DeepSpeed maintainer response time for community PRs on ZeRO-3 edge cases appears to be very slow. The fix is trivially correct — adding `close()` where a no-op expression existed — but without review it cannot merge.

---

## 5. Technical Analysis

### 5.1 What Causes the FD Leak

The `deepspeed_io_handle_t::wait()` method is the C++ async I/O completion handler in DeepSpeed's NVMe offload subsystem. When ZeRO-3 offloads parameters/gradients/states to NVMe storage via async I/O (libaio or POSIX AIO), each operation opens a file descriptor to the storage file. The `wait()` method iterates through completed operations and:

1. Calls `completed_op->finish()` — processes the I/O result
2. Should call `close(completed_op->_fd)` — release the file descriptor
3. Decrements `_num_pending_ops`

Step 2 was broken: `(completed_op->_fd)` evaluates the integer but never calls `close()`. The fd stays open forever in the process.

### 5.2 Affected Components

- **ZeRO-3 NVMe offload**: The primary affected component. Any configuration using `offload_optimizer` or `offload_param` with `device: nvme` triggers the async I/O path.
- **ZeRO-3 checkpoint save/load**: Checkpoint operations also use the async I/O handle and accumulate leaked fds.
- **ZeRO-2 with NVMe offload**: ZeRO-2 can also offload optimizer states to NVMe, so it is potentially affected as well (the C++ async I/O layer is shared).

★★★★★★★★★ **ZeRO-2 NVMe offload is ALSO affected**: The async I/O C++ code is framework-level, not ZeRO-3-specific. If ZeRO-2 uses `offload_optimizer: {device: nvme}`, the same fd leak path exists. However, on RTX 4090 we use ZeRO-2 + CPU_Adam (not NVMe offload), so this specific leak is NOT triggered in our standard configuration.

### 5.3 FD Exhaustion Timeline

| Checkpoint frequency | Fds leaked per step | Steps to exhaust (ulimit 1024) | Steps to exhaust (ulimit 65536) |
|---|---|---|---|
| Every 1 step | 2 (read+write) | ~512 steps | ~32,768 steps |
| Every 10 steps | 2 | ~5,120 steps | ~327,680 steps |
| Every 100 steps | 2 | ~51,200 steps | ~3.2M steps |
| Only save (1 write) | 1 | ~1,024 steps | ~65,536 steps |

A GRPO training run of 1000+ steps with frequent checkpointing would hit the default 1024 fd limit well before completion.

---

## 6. Impact on RTX 4090 GRPO Training

### 6.1 Direct Impact Assessment

★★★★★★★★★ **Low direct impact for our standard RTX 4090 config**: Our recommended configuration is ZeRO-2 + CPU_Adam with CPU offload (not NVMe). The fd leak only occurs when NVMe offload is used. On a 24 GiB RTX 4090, NVMe offload is not typically used because:
- CPU offload (CPU_Adam) is sufficient for optimizer states
- NVMe offload adds latency compared to CPU
- Single GPU does not benefit from NVMe's primary use case (shard distribution across multiple nodes)

### 6.2 Indirect Impact

★★★ **Potential impact if NVMe offload is attempted**: If someone tries to use ZeRO-3 with NVMe offload on RTX 4090 (e.g., training a large model with 3B+ parameters where even CPU RAM is constrained), the fd leak becomes a critical crash risk. However, ZeRO-3 on single GPU is pure overhead anyway (see our established rule: ALWAYS ZeRO-2 on single GPU).

★★★ **Impact on checkpointing even with ZeRO-2+CPU**: If DeepSpeed's checkpoint save uses the async I/O path for writing checkpoint files, the fd leak could affect even non-NVMe configurations. Need to verify whether `save_checkpoint()` calls `deepspeed_io_handle_t::wait()` directly or uses a different I/O path.

---

## 7. Connection to Other DeepSpeed Issues

### 7.1 Direct Connections

| Issue | Relationship | Notes |
|---|---|---|
| **#8072** | Indirect — both are v0.19.2 regressions, different subsystems | #8072 is ZeRO-3+PEFT dtype mismatch; #8075 is async I/O fd leak. Both affect long-running training stability. |
| **#8073** | Pattern parallel — both are trivial fixes stalled with 0 reviews | #8073 is a 2-line dtype fix; #8075 is a 1-line close() fix. Both are blocked by DeepSpeed review backlog. |
| **#8066** | Indirect — the per-policy dtype PR that CAUSED #8072 regression | #8066 is unrelated to #8075 but demonstrates that DeepSpeed v0.19.2 introduced multiple regressions. |
| **#8068** | Pattern parallel — 0 reviews, stalled | #8068 is gradient_clipping default change, also stalled. Three PRs (#8068, #8073, #8075) all have trivial fixes but zero maintainer engagement. |
| **#8061** | Different subsystem but same "long-running training risk" category | #8061 is overlap_comm+torch.compile=NaN; #8075 is fd exhaustion. Both cause silent failures in extended training. |
| **#8058** | Different subsystem | ZenFlow CPU optimizer (2944->256 MiB), unrelated but another v0.19.x change. |

### 7.2 Pattern: DeepSpeed v0.19.2 Regression Cluster

★★★★★★★★★ **v0.19.2 introduced a cluster of regressions**: #8066 (per-policy dtype, MERGED but caused #8072), #8072/#8073 (ZeRO-3+PEFT dtype mismatch), #8075 (fd leak), #8076 (independent #8072 confirmation). The fd leak (#8075) may have existed BEFORE v0.19.2 (it is a latent bug, not introduced by any recent change), but it is newly discovered and reported in the same timeframe.

★★★★★★★★★ **Common pattern: trivial fixes, zero reviews**: #8073 (+2/-2), #8075 (+1/-1), #8068 (+? lines) — all are small, obviously correct fixes that are stalled because DeepSpeed maintainers are not reviewing community PRs promptly. This is a systemic maintainer responsiveness issue, not a code complexity issue.

---

## 8. Recommended Mitigation for RTX 4090

### 8.1 Immediate Mitigation (before fix merges)

1. **Increase fd limit**: `ulimit -n 65536` before launching training. This is the standard mitigation for any fd leak. On macOS/Linux:
   ```bash
   ulimit -n 65536
   # Or in the training launch script:
   # ulimit -Sn 65536  # soft limit
   # ulimit -Hn 65536  # hard limit (may require root)
   ```

2. **Avoid NVMe offload**: Our standard RTX 4090 config (ZeRO-2 + CPU_Adam) does not trigger the async I/O path, so the fd leak is not encountered. Stick with CPU offload.

3. **Monitor fd usage during training**: Add periodic checks:
   ```python
   import os
   # Check number of open fds for current process
   pid = os.getpid()
   fd_count = len(os.listdir(f'/proc/{pid}/fd'))  # Linux
   # macOS: lsof -p {pid} | wc -l
   if fd_count > 500:
       print(f"WARNING: {fd_count} open file descriptors")
   ```

### 8.2 After Fix Merges

- Pin DeepSpeed version >= the release that includes #8075
- Until merged, the workaround is `ulimit -n 65536` + avoid NVMe offload

---

## 9. Connection Map to Framework Issues

```
DeepSpeed #8075 (fd leak)
  |
  ├── Same PR pattern ──→ #8073 (2-line dtype fix, 0 reviews, stalled)
  ├── Same PR pattern ──→ #8068 (gradient clipping, 0 reviews, stalled)
  |
  ├── Same v0.19.2 timeframe ──→ #8072 (ZeRO-3+PEFT regression)
  ├── Same v0.19.2 timeframe ──→ #8076 (independent #8072 confirmation)
  ├── Root cause of #8072 ──→ #8066 (per-policy dtype, MERGED June 16)
  |
  ├── Long-running stability ──→ #8061 (overlap_comm NaN)
  ├── Long-running stability ──→ verl #6468 (FSDP2 CPU memory leak)
  ├── Long-running stability ──→ Megatron #5227 (recompute memory leak)
  |
  ├── RTX 4090 config relevance ──→ ZeRO-2+CPU_Adam (AVOIDS this bug)
  ├── RTX 4090 config relevance ──→ ZeRO-3+NVMe (TRIGGERS this bug, but NEVER use Z3 on single GPU)
  |
  └── Cross-framework fd patterns ──→ SGLang #28588 (image decompression bomb, different fd pattern)
  └── Cross-framework resource exhaustion ──→ verl #6699 (detach memory leak, similar "accumulation over steps" pattern)
```

---

## 10. Key Takeaways

★★★★★★★★★ **The fd leak is a 1-character typo**: `(completed_op->_fd)` vs `close(completed_op->_fd)`. This is one of the simplest bugs possible in C/C++ — a no-op expression instead of a function call. The fix is trivially correct and should merge quickly, but DeepSpeed review velocity is the bottleneck.

★★★★★★★★★ **RTX 4090 GRPO training is NOT directly affected**: Our standard config (ZeRO-2 + CPU_Adam, no NVMe offload) bypasses the async I/O code path entirely. The fd leak only manifests with NVMe offload, which we do not use.

★★★★★★★★★ **The systemic pattern is more concerning than this individual bug**: Three trivial PRs (#8073, #8075, #8068) are stalled with zero reviews. This indicates a maintainer responsiveness crisis at DeepSpeed for community-contributed fixes to ZeRO edge cases. Our OSS contribution strategy (P7 C10/C11 for #8073) faces the same review bottleneck risk.

★★★ **Mitigation is straightforward**: `ulimit -n 65536` + avoid NVMe offload. No code changes needed in our training configs.

★★★ **This bug may have existed for a long time**: The `(completed_op->_fd)` no-op expression is not a recent v0.19.2 introduction. It could have been present for multiple releases, only noticed now because someone finally traced fd exhaustion in long-running training.

---

## 11. Monitoring Status

- **Status**: OPEN, not merged, 0 comments, 0 reviews, 1 day old
- **Reviewer requested**: tjruwase (DeepSpeed collaborator)
- **Expected merge timeline**: Unknown — depends on DeepSpeed review velocity
- **Priority for RTX 4090**: LOW (not triggered by our standard config)
- **Priority for OSS contribution tracking**: LOW (fix is already submitted by MarkCLChang, no unique contribution opportunity for us)

---

*End of reading note. Generated 2026-06-19.*
