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
