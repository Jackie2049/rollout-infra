# DeepSpeed #8061: The Bug That Was Closed but Never Fixed

## Timeline

| Date | Event |
|------|-------|
| June 6, 2026 | #8061 filed: overlap_comm+torch.compile=NaN |
| June 13 | hwchen2017 and cx2009 (maintainers) engaged, asked for repro |
| June 15 | Reporter cjxjxjx posted evidence matrix: multi-stream data race |
| June 23 | PR #8080 proposed (+96/-1): wait on ALL IPG-bucket producer streams |
| July 14 04:59 UTC | **Issue CLOSED** by hwchen2017, but **PR #8080 was NEVER MERGED** |
| July 14 | Our analysis in diary: "CLOSED (overlap_comm NaN fixed July 14!)" — **this was wrong** |

## Correction

DeepSpeed #8061 was **closed administratively** but **no fix was ever merged**. The root cause (multi-stream data race between contiguous gradient bucket copies and the ZeRO reduction stream) remains unfixed in the codebase.

This is significant because:
1. Two maintainers were actively engaged (hwchen2017, cx2009)
2. PR #8080 had a clear fix (+96/-1)
3. The issue was closed without explanation, without merging the fix, and without linking to any resolution

This appears to be an administrative close rather than a technical resolution.

## Impact

For RTX 4090 GRPO training:
- `overlap_comm=True` STILL causes NaN with `torch.compile` on single GPU
- `overlap_comm=False` is STILL mandatory for RTX 4090
- No fix available until PR #8080 or equivalent is merged

## Monitoring

- PR #8080 is NOT listed as merged in the repo
- No new PRs have been opened to fix this issue
- Check if #8080 gets re-opened or if a new fix PR appears

## Deeper Analysis: Why Was #8061 Hard to Fix?

The fix requires:
1. Tracking ALL copy_streams per IPG bucket (not just the last one)
2. Waiting on ALL producer streams in `average_tensor()`
3. This adds synchronization latency that may partially defeat the purpose of overlap

The maintainer may have decided the fix cost (performance regression from additional synchronization) outweighed the benefit (fixing an edge case triggered by torch.compile). However, this leaves users with a silent corruption bug — the worst possible pattern.
