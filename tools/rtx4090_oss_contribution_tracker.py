#!/usr/bin/env python3
"""RTX 4090 OSS Contribution Status Tracker

Track all planned and drafted OSS contributions for RTX 4090 across
vLLM, PyTorch, verl, and rLLM projects. Includes priority ranking,
current status, and next steps.

Usage:
  python3 tools/rtx4090_oss_contribution_tracker.py --mode status
  python3 tools/rtx4090_oss_contribution_tracker.py --mode priority
  python3 tools/rtx4090_oss_contribution_tracker.py --mode next-steps
  python3 tools/rtx4090_oss_contribution_tracker.py --mode all
"""

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Tier(Enum):
    TIER1_COMMENT = "Tier 1: Comment on existing issue/PR"
    TIER2_PR = "Tier 2: Submit new PR"
    TIER3_COLLAB = "Tier 3: Collaborative contribution"


class Status(Enum):
    DRAFT_READY = "draft_ready"
    GPU_VALIDATION_NEEDED = "gpu_validation_needed"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class Contribution:
    id: str
    title: str
    target: str  # e.g., "pytorch/pytorch", "vllm-project/vllm"
    tier: Tier
    priority: int  # 1-10, higher = more important
    status: Status
    issue_ref: Optional[str] = None
    draft_ref: Optional[str] = None  # notebook path
    description: str = ""
    next_steps: list = field(default_factory=list)
    gpu_needed: bool = False
    unique_contribution: bool = False  # Is this something only WE can provide?


CONTRIBUTIONS = [
    # ===== TOP PRIORITY =====
    Contribution(
        id="BR1",
        title="BudgetRefiner SLO → vLLM upstream PR",
        target="vllm-project/vllm",
        tier=Tier.TIER2_PR,
        priority=10,
        status=Status.DRAFT_READY,
        draft_ref="notebook/projects/budgetrefiner-vllm-pr-draft.md",
        description="Port BudgetRefiner SLO from vLLM-Ascend to standard vLLM V1. "
                    "95%+ GPU-generic. Dynamic token budget + SLO-aware scheduling + decode-first. "
                    "RTX 4090 profile_table.csv = unique contribution no other contributor has!",
        next_steps=[
            "Collect RTX 4090 profile_table.csv (GPU needed)",
            "Write RFC issue on vllm-project/vllm",
            "Get community feedback",
            "Implement BudgetRefiner + decode-first + SLO config",
            "Run vLLM CI tests",
            "Submit PR",
        ],
        gpu_needed=True,
        unique_contribution=True,
    ),
    Contribution(
        id="IF1",
        title="Inductor SM<90 Fusion Guard → PyTorch upstream PR",
        target="pytorch/pytorch",
        tier=Tier.TIER2_PR,
        priority=9,
        status=Status.DRAFT_READY,
        draft_ref="notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md",
        description="5-line modification to choices.py can_fuse_vertical. "
                    "Prevents vertical reduction fusion on SM<90 → keeps torch.mean "
                    "batch-invariant override effective. 5 precedents verified.",
        next_steps=[
            "Run vLLM on RTX 4090 with guard → verify batch invariance",
            "Benchmark: guard ON vs OFF on RTX 4090",
            "Draft PyTorch issue → reference vLLM #39096",
            "Run PyTorch Inductor CI tests",
            "Submit PR",
        ],
        gpu_needed=True,
        unique_contribution=False,
    ),
    # ===== TIER 1 COMMENTS =====
    Contribution(
        id="C1",
        title="Comment on vLLM #44879/#45038 — 3 FP8 KV path distinction",
        target="vllm-project/vllm",
        tier=Tier.TIER1_COMMENT,
        priority=8,
        status=Status.DRAFT_READY,
        issue_ref="vllm-project/vllm #44879, #45038",
        description="Explain the 3 FP8 KV path distinction on SM89: "
                    "Triton FP8 ALLOWED, FlashInfer FP8 BLOCKED, compressed-tensors FP8 CRASH. "
                    "Suggest fail-fast guard in compressed-tensors override.",
        next_steps=["Post comment on #44879 and #45038"],
        gpu_needed=False,
        unique_contribution=False,
    ),
    Contribution(
        id="C2",
        title="Comment on vLLM #39096 — batch invariance root cause",
        target="vllm-project/vllm",
        tier=Tier.TIER1_COMMENT,
        priority=7,
        status=Status.DRAFT_READY,
        issue_ref="vllm-project/vllm #39096",
        description="Explain root cause: Inductor fuses RMSNorm → tl.sum() inline → "
                    "XBLOCK varies → batch-dependent results. Link to PyTorch #185814.",
        next_steps=["Post comment on #39096"],
        gpu_needed=False,
        unique_contribution=False,
    ),
    Contribution(
        id="C3",
        title="Comment on vLLM #44594 — BudgetRefiner complementary to Watermark",
        target="vllm-project/vllm",
        tier=Tier.TIER1_COMMENT,
        priority=7,
        status=Status.DRAFT_READY,
        issue_ref="vllm-project/vllm #44594",
        description="Watermark handles KV cache pressure (reactive). BudgetRefiner "
                    "handles compute time pressure (proactive). Both needed for production SLO.",
        next_steps=["Post comment on #44594"],
        gpu_needed=False,
        unique_contribution=False,
    ),
    Contribution(
        id="C4",
        title="Comment on vLLM #44701 — prefix hash collision",
        target="vllm-project/vllm",
        tier=Tier.TIER1_COMMENT,
        priority=6,
        status=Status.DRAFT_READY,
        issue_ref="vllm-project/vllm #44701",
        description="Analysis of prefix hash collision and impact on SM89.",
        next_steps=["Post comment on #44701"],
        gpu_needed=False,
        unique_contribution=False,
    ),
    Contribution(
        id="C5",
        title="Comment on verl #6401 — RTX 4090 GRPO configuration",
        target="volcengine/verl",
        tier=Tier.TIER1_COMMENT,
        priority=5,
        status=Status.DRAFT_READY,
        issue_ref="volcengine/verl #6401",
        description="RTX 4090 GRPO configuration guide: bypass_mode=True, "
                    "detach_metrics=True, CPPO+bypass recommended.",
        next_steps=["Post comment on #6401"],
        gpu_needed=False,
        unique_contribution=False,
    ),
    Contribution(
        id="C6",
        title="rLLM cookbook — RTX 4090 Tinker recipe",
        target="rllm-org/rllm",
        tier=Tier.TIER1_COMMENT,
        priority=5,
        status=Status.DRAFT_READY,
        description="Add train_tinker_rtx4090.sh recipe to rLLM cookbooks/math/.",
        next_steps=["Submit PR to rllm-org/rllm with cookbook addition"],
        gpu_needed=False,
        unique_contribution=False,
    ),
    Contribution(
        id="C7",
        title="Comment on rLLM #605 — GRPO grouping bug fix proposal",
        target="rllm-org/rllm",
        tier=Tier.TIER1_COMMENT,
        priority=9,
        status=Status.DRAFT_READY,
        issue_ref="rllm-org/rllm #605",
        description="Source-level analysis of GRPO grouping bug: trajectory.uid vs task_ids → "
                    "group size 1 → advantage = raw reward → GRPO BROKEN. 1-line fix for enable=False "
                    "(task_ids), few lines for per_step ({task_id}_step{k}). "
                    "CRITICAL RTX 4090 blocker — rLLM GRPO unusable until fixed.",
        draft_ref="notebook/tier1-comments/rllm-605-grpo-grouping-bug-comment-draft.md",
        next_steps=["Post comment on #605 with verified fix proposal"],
        gpu_needed=False,
        unique_contribution=True,
    ),
    Contribution(
        id="C8",
        title="Comment on Megatron #5394 — Cross-framework Muon clipping pattern",
        target="NVIDIA/Megatron-LM",
        tier=Tier.TIER1_COMMENT,
        priority=8,
        status=Status.DRAFT_READY,
        issue_ref="NVIDIA/Megatron-LM #5394",
        description="Cross-framework analysis: same bug found in DeepSpeed #8068/#7776 + "
                    "Megatron #5394. Universal insight: scale-invariant optimizers MUST NOT be "
                    "globally clipped. PR #5395 skip_grad_norm_clip attribute aligned with "
                    "DeepSpeed pattern. Complementary: Emerging-Optimizers #230 NS eps robustness.",
        draft_ref="notebook/tier1-comments/megatron-5394-muon-clipping-comment-draft.md",
        next_steps=["Post comment on #5394 with cross-framework evidence"],
        gpu_needed=False,
        unique_contribution=True,
    ),
    Contribution(
        id="C9",
        title="verl #6699 unfixed backend detach fix — Automodel/Megatron/TorchTitan",
        target="verl-project/verl",
        tier=Tier.TIER2_PR,
        priority=7,
        status=Status.DRAFT_READY,
        issue_ref="verl-project/verl #6699",
        description="Apply same detach() fix from #6699 to 3 unfixed engine backends: "
                    "AutomodelEngine (lines 708-712), MegatronEngine (lines 1013-1017), "
                    "TorchTitanEngine (lines 730-734). Same 11-line pattern: "
                    "model_output = {key: value.detach() if torch.is_tensor(value) and "
                    "value.grad_fn is not None else value for key, value in model_output.items()}. "
                    "CRITICAL: these backends will OOM with LoRA + long sequences on RTX 4090.",
        next_steps=["Submit PR to verl-project/verl applying detach fix to 3 backends"],
        gpu_needed=False,
        unique_contribution=True,
    ),
    Contribution(
        id="C10",
        title="Comment on vLLM #45683 — Deterministic MoE combine for GRPO stability",
        target="vllm-project/vllm",
        tier=Tier.TIER1_COMMENT,
        priority=7,
        status=Status.DRAFT_READY,
        issue_ref="vllm-project/vllm #45683",
        description="CRITICAL for GRPO MoE: cross-rank summation order in MoE combine "
                    "was NOT deterministic → breaks reward stability. This PR fixes "
                    "reduce_scatterv → fixed-root reduce + scatter. Alignment with SGLang "
                    "#27869 MoE top-k combine. Complementary with VLLM_BATCH_INVARIANT.",
        draft_ref="notebook/tier1-comments/vllm-45683-moe-deterministic-combine-comment-draft.md",
        next_steps=["Post comment on #45683 with GRPO MoE stability analysis"],
        gpu_needed=False,
        unique_contribution=True,
    ),
    # ===== TIER 2 PRs (future) =====
    Contribution(
        id="AE1",
        title="DeepSpeed AutoEP RTX 4090 MoE benchmark → DeepSpeed cookbook",
        target="microsoft/DeepSpeed",
        tier=Tier.TIER2_PR,
        priority=6,
        status=Status.GPU_VALIDATION_NEEDED,
        issue_ref="microsoft/DeepSpeed #7938",
        description="RTX 4090 MoE training benchmark with AutoEP EP=1 singleton + LoRA + CPU_Adam. "
                    "First-ever RTX 4090 MoE training validation. "
                    "Qwen3-MoE (A0.6B+B4B) fits 24GB with LoRA rank=32. "
                    "DeepSpeed cookbook addition with complete config.",
        next_steps=[
            "Run Qwen3-MoE AutoEP on RTX 4090 (GPU needed)",
            "Benchmark step time and memory",
            "Write DeepSpeed cookbook entry",
            "Submit PR to microsoft/DeepSpeed",
        ],
        gpu_needed=True,
        unique_contribution=True,
    ),
    Contribution(
        id="QK1",
        title="QuantKey refactor → systematic SM89 FP8 guard",
        target="vllm-project/vllm",
        tier=Tier.TIER2_PR,
        priority=4,
        status=Status.GPU_VALIDATION_NEEDED,
        issue_ref="vllm-project/vllm #32268",
        description="Refactor boolean→QuantKey for quantization config. "
                    "Foundation for systematic SM89 FP8 guard (Phase 2: requires_sm).",
        next_steps=["Implement QuantKey refactor", "Add requires_sm field", "Submit PR"],
        gpu_needed=True,
        unique_contribution=False,
    ),
    Contribution(
        id="FP4",
        title="SM120 FP4/MXFP4 kernel → vLLM contribution",
        target="vllm-project/vllm",
        tier=Tier.TIER2_PR,
        priority=3,
        status=Status.GPU_VALIDATION_NEEDED,
        description="NEXT-PHASE: RTX 5090 FP4/MXFP4 kernel for vLLM. "
                    "PyTorch v2.12 MXFP4 AOTI shim = kernel infrastructure. "
                    "vLLM currently has NO FP4 support → contribution window.",
        next_steps=[
            "Research MXFP4 kernel requirements",
            "Wait for RTX 5090 availability",
            "Implement FP4 quantization path",
            "Submit PR",
        ],
        gpu_needed=True,
        unique_contribution=True,
    ),
]


def run_status(args):
    """Display current status of all contributions."""
    print("=" * 80)
    print("RTX 4090 OSS Contribution Status Tracker")
    print("=" * 80)
    print()

    # Group by status
    by_status = {}
    for c in CONTRIBUTIONS:
        by_status.setdefault(c.status, []).append(c)

    for status, items in sorted(by_status.items(), key=lambda x: -max(c.priority for c in x[1])):
        print(f"## {status.value} ({len(items)} items)")
        for c in sorted(items, key=lambda x: -x.priority):
            unique_mark = " ★ UNIQUE" if c.unique_contribution else ""
            gpu_mark = " [GPU needed]" if c.gpu_needed else ""
            print(f"  [{c.id}] P{c.priority} {c.title}{unique_mark}{gpu_mark}")
            if c.issue_ref:
                print(f"    Issue: {c.issue_ref}")
            if c.draft_ref:
                print(f"    Draft: {c.draft_ref}")
            print(f"    Target: {c.target}")
            print()

    print(f"Total contributions: {len(CONTRIBUTIONS)}")
    print(f"Draft ready: {len([c for c in CONTRIBUTIONS if c.status == Status.DRAFT_READY])}")
    print(f"GPU validation needed: {len([c for c in CONTRIBUTIONS if c.status == Status.GPU_VALIDATION_NEEDED])}")
    print(f"Unique contributions: {len([c for c in CONTRIBUTIONS if c.unique_contribution])}")


def run_priority(args):
    """Display contributions sorted by priority."""
    print("=" * 80)
    print("RTX 4090 OSS Contribution Priority Ranking")
    print("=" * 80)
    print()

    for c in sorted(CONTRIBUTIONS, key=lambda x: -x.priority):
        unique_mark = " ★★★★★★★★★★★★★★★ UNIQUE CONTRIBUTION" if c.unique_contribution else ""
        gpu_mark = " [GPU NEEDED]" if c.gpu_needed else ""
        tier_mark = f"[{c.tier.value}]"

        print(f"P{c.priority:2d} {c.id} {tier_mark} {c.title}{unique_mark}{gpu_mark}")
        print(f"     {c.description[:100]}...")
        print()


def run_next_steps(args):
    """Display next steps for draft-ready contributions."""
    print("=" * 80)
    print("RTX 4090 OSS Contribution — Next Steps")
    print("=" * 80)
    print()

    draft_ready = [c for c in CONTRIBUTIONS if c.status == Status.DRAFT_READY]
    for c in sorted(draft_ready, key=lambda x: -x.priority):
        print(f"[{c.id}] {c.title}")
        print(f"  Status: {c.status.value}")
        if c.gpu_needed:
            print(f"  ⚠️  GPU validation needed — both servers offline")
        print(f"  Next steps:")
        for i, step in enumerate(c.next_steps, 1):
            print(f"    {i}. {step}")
        print()


def run_all(args):
    """Display all information."""
    run_status(args)
    print()
    run_priority(args)
    print()
    run_next_steps(args)


def main():
    parser = argparse.ArgumentParser(
        description="RTX 4090 OSS Contribution Status Tracker")
    parser.add_argument("--mode",
                        choices=["status", "priority", "next-steps", "all"],
                        default="status",
                        help="Display mode")
    args = parser.parse_args()

    modes = {
        "status": run_status,
        "priority": run_priority,
        "next-steps": run_next_steps,
        "all": run_all,
    }
    modes[args.mode](args)


if __name__ == "__main__":
    main()
