#!/usr/bin/env python3
"""
Seven Framework Issue Tracker
==============================
Live tracking of critical issues across 7 AI infra frameworks.
Checks GitHub API for current status of all monitored issues/PRs.

Frameworks: DeepSpeed, Megatron-LM, vLLM, verl, MindIE/vLLM-Ascend, rLLM, SGLang, PyTorch

Modes:
  track   - Check all monitored issues for current status
  recent  - Show recently updated issues (last 7 days)
  summary - Dashboard summary with critical counts
  fresh   - Discover NEW issues from last 7 days not yet in our tracker
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# Monitored Issues Database
# ============================================================

MONITORED = {
    "DeepSpeed": {
        "repo": "deepspeedai/DeepSpeed",
        "issues": {
            "8061": {"type": "issue", "priority": "CRITICAL", "desc": "overlap_comm CUDA stream race NaN", "our_draft": "P7 C16"},
            "8068": {"type": "pull", "priority": "HIGH", "desc": "gradient_clipping default 0→1.0", "our_draft": "P7 C15"},
            "8072": {"type": "issue", "priority": "HIGH", "desc": "ZeRO-3+PEFT LoRA regression", "our_draft": "P7 C10"},
            "8073": {"type": "pull", "priority": "HIGH", "desc": "ZeRO-3+PEFT fix 2-line", "our_draft": "P7 C10"},
            "8075": {"type": "pull", "priority": "MEDIUM", "desc": "fd leak close() fix +1/-1", "our_draft": "P7 C14"},
            "8076": {"type": "issue", "priority": "HIGH", "desc": "Independent #8072 confirmation"},
            "8058": {"type": "pull", "priority": "HIGH", "desc": "ZenFlow CPU optimizer"},
        },
    },
    "Megatron-LM": {
        "repo": "NVIDIA/Megatron-LM",
        "issues": {
            "5395": {"type": "pull", "priority": "HIGH", "desc": "skip_grad_norm_clip +15/-1"},
            "5387": {"type": "pull", "priority": "HIGH", "desc": "MFSDPv2 DBuffer APPROVED"},
            "5317": {"type": "issue", "priority": "CRITICAL", "desc": "DSv4-Hybrid rotary NaN (11th DSV4 failure)"},
            "5396": {"type": "pull", "priority": "MEDIUM", "desc": "GDN L2-norm fold"},
            "5398": {"type": "pull", "priority": "MEDIUM", "desc": "NVIDIA reviewer concerns"},
            "5400": {"type": "pull", "priority": "MEDIUM", "desc": "6th Muon blocker"},
            "5401": {"type": "pull", "priority": "MEDIUM", "desc": "unknown"},
            "5179": {"type": "issue", "priority": "HIGH", "desc": "Muon PyPI stub 4th blocker"},
        },
    },
    "vLLM": {
        "repo": "vllm-project/vllm",
        "issues": {
            "46118": {"type": "issue", "priority": "CRITICAL", "desc": "MTP+grammar FSM conflict 58% failure"},
            "46085": {"type": "pull", "priority": "CRITICAL", "desc": "aot_eager piecewise compilation"},
            "46105": {"type": "issue", "priority": "HIGH", "desc": "DFlash tracker 130+ issues"},
            "46007": {"type": "pull", "priority": "MEDIUM", "desc": "Orthrus spec decode 17.76%"},
            "45683": {"type": "issue", "priority": "HIGH", "desc": "Deterministic MoE combine"},
            "45819": {"type": "pull", "priority": "HIGH", "desc": "GDN batch invariance"},
            "46088": {"type": "issue", "priority": "HIGH", "desc": "MTP kv-dtype garbage"},
            "45656": {"type": "pull", "priority": "CRITICAL", "desc": "MoE is_sym guard MERGED"},
        },
    },
    "verl": {
        "repo": "verl-project/verl",
        "issues": {
            "6794": {"type": "pull", "priority": "HIGH", "desc": "delta weight sync ~100x reduction"},
            "6782": {"type": "issue", "priority": "HIGH", "desc": "LoRA rank=64 breaks EOS"},
            "6468": {"type": "issue", "priority": "CRITICAL", "desc": "FSDP2 CPU memory leak 0.6-6.3 GiB/step"},
            "6699": {"type": "pull", "priority": "HIGH", "desc": "detach memory fix 4x reduction"},
            "6731": {"type": "pull", "priority": "MEDIUM", "desc": "CPPO bypass_mode"},
            "6512": {"type": "pull", "priority": "CRITICAL", "desc": "per-unit LoRA summon 10x memory MERGED"},
            "6572": {"type": "pull", "priority": "MEDIUM", "desc": "deterministic inference"},
        },
    },
    "vLLM-Ascend": {
        "repo": "vllm-project/vllm-ascend",
        "issues": {
            "10684": {"type": "issue", "priority": "CRITICAL", "desc": "DSA Hadamard sleep/wake ALL-ZERO"},
            "10579": {"type": "pull", "priority": "HIGH", "desc": "MoE NaN 1-line fix 0 reviews", "our_draft": "P6 C13"},
            "10592": {"type": "pull", "priority": "MEDIUM", "desc": "NPUIPC weight transfer +787"},
            "10724": {"type": "issue", "priority": "HIGH", "desc": "DSV4 PD-Mix crash"},
            "10730": {"type": "pull", "priority": "MEDIUM", "desc": "MX quant fusion"},
            "10645": {"type": "pull", "priority": "MEDIUM", "desc": "DSV4 chat fix"},
            "10077": {"type": "pull", "priority": "MEDIUM", "desc": "Layerwise KV Pooling MERGED"},
        },
    },
    "rLLM": {
        "repo": "rllm-org/rllm",
        "issues": {
            "605": {"type": "issue", "priority": "CRITICAL", "desc": "GRPO grouping bug ZERO comments", "our_draft": "P9 C7"},
            "663": {"type": "pull", "priority": "CRITICAL", "desc": "Step.output fix ALL rewards=0 MERGED"},
            "664": {"type": "pull", "priority": "MEDIUM", "desc": "Fireworks SWE-RL"},
        },
    },
    "SGLang": {
        "repo": "sgl-project/sglang",
        "issues": {
            "28676": {"type": "pull", "priority": "CRITICAL", "desc": "MXFP8 MoE cache CLOBBERED", "our_draft": "P7 C11"},
            "28695": {"type": "pull", "priority": "HIGH", "desc": "ReplaySSM +13.1% throughput"},
            "28679": {"type": "issue", "priority": "CRITICAL", "desc": "GDN intermittent degeneracy"},
            "28608": {"type": "pull", "priority": "HIGH", "desc": "RolloutKV prefix pinning +768/-5"},
            "28685": {"type": "issue", "priority": "CRITICAL", "desc": "GLM-5.2 FP8 block-fp8 wrong MI350X (12th DSV4)"},
            "28703": {"type": "pull", "priority": "HIGH", "desc": "DSA LoRA targets GLM-5.1"},
            "27097": {"type": "pull", "priority": "MEDIUM", "desc": "multi-LoRA determinism"},
            "28583": {"type": "pull", "priority": "MEDIUM", "desc": "revert head_dim regression MERGED"},
        },
    },
    "PyTorch": {
        "repo": "pytorch/pytorch",
        "issues": {
            "187653": {"type": "pull", "priority": "HIGH", "desc": "NanDetectMode forward NaN detection"},
            "187620": {"type": "pull", "priority": "HIGH", "desc": "PartialOffloadPolicy DRAFT"},
            "184119": {"type": "pull", "priority": "HIGH", "desc": "SM89 fp8 guard validates P9"},
            "187636": {"type": "pull", "priority": "MEDIUM", "desc": "autotune_at_compile_time progressing"},
        },
    },
}


def gh_api(endpoint, repo=None):
    """Call gh api and return parsed JSON."""
    if repo:
        url = f"repos/{repo}/{endpoint}"
    else:
        url = endpoint
    try:
        result = subprocess.run(
            ["gh", "api", url, "--jq", ".state, .updated_at, .comments, .title"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 4:
            return {
                "state": lines[0],
                "updated_at": lines[1],
                "comments": int(lines[2]) if lines[2].isdigit() else 0,
                "title": lines[3],
            }
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def get_issue_status(repo, number, is_pr=False):
    """Get current status of an issue or PR."""
    endpoint = f"pulls/{number}" if is_pr else f"issues/{number}"
    return gh_api(endpoint, repo=repo)


def check_all_monitored():
    """Check status of all monitored issues."""
    results = {}
    total_checked = 0
    total_failed = 0

    for framework, config in MONITORED.items():
        repo = config["repo"]
        framework_results = {}

        for number, info in config["issues"].items():
            is_pr = info["type"] == "pull"
            status = get_issue_status(repo, number, is_pr)

            if status:
                framework_results[number] = {
                    **info,
                    "current_state": status["state"],
                    "updated_at": status["updated_at"],
                    "current_comments": status["comments"],
                    "current_title": status.get("title", info["desc"]),
                }
                total_checked += 1
            else:
                framework_results[number] = {
                    **info,
                    "current_state": "API_ERROR",
                    "updated_at": "N/A",
                    "current_comments": 0,
                    "current_title": info["desc"],
                }
                total_failed += 1

        results[framework] = framework_results

    return results, total_checked, total_failed


def format_track(results, total_checked, total_failed):
    """Format full tracking output."""
    output = []
    output.append("=" * 80)
    output.append("SEVEN FRAMEWORK ISSUE TRACKER — Live Status Check")
    output.append(f"Checked: {total_checked}/{total_checked + total_failed} issues")
    output.append("=" * 80)

    merged_count = 0
    critical_open = 0
    stalled = 0

    for framework, issues in results.items():
        output.append(f"\n{'─' * 40}")
        output.append(f"  {framework} ({MONITORED[framework]['repo']})")
        output.append(f"{'─' * 40}")

        for number, info in sorted(issues.items(), key=lambda x: (
            0 if x[1].get("priority") == "CRITICAL" else 1 if x[1].get("priority") == "HIGH" else 2
        )):
            state = info["current_state"]
            priority = info.get("priority", "MEDIUM")
            our_draft = info.get("our_draft", "")

            # State indicators
            if state == "closed":
                state_str = "MERGED/CLOSED ✓"
                merged_count += 1
            elif state == "open":
                state_str = "OPEN ●"
                if priority == "CRITICAL":
                    critical_open += 1
                if info.get("current_comments", 0) == 0 and info.get("type") == "pull":
                    stalled += 1
            else:
                state_str = f"{state} ⚠"

            draft_str = f" → {our_draft}" if our_draft else ""
            comments_str = f" ({info.get('current_comments', 0)} comments)" if state == "open" else ""

            priority_indicator = "★★★" if priority == "CRITICAL" else "★★" if priority == "HIGH" else "★"

            output.append(
                f"  {priority_indicator} #{number}: {info.get('current_title', info['desc'])}"
                f"\n    State: {state_str}{comments_str} | Updated: {info.get('updated_at', 'N/A')[:10]}"
                f"{draft_str}"
            )

    output.append(f"\n{'=' * 80}")
    output.append(f"SUMMARY: {merged_count} merged/closed, {critical_open} CRITICAL open, {stalled} stalled PRs (0 comments)")
    output.append(f"         Total monitored: {total_checked + total_failed} across 8 frameworks")
    output.append("=" * 80)

    return "\n".join(output)


def format_summary(results):
    """Compact dashboard summary."""
    output = []
    output.append("7-FRAMEWORK CRITICAL DASHBOARD")
    output.append("─" * 60)

    totals = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "MERGED": 0, "STALLED": 0}

    for framework, issues in results.items():
        critical = sum(1 for i in issues.values() if i.get("priority") == "CRITICAL" and i.get("current_state") == "open")
        high = sum(1 for i in issues.values() if i.get("priority") == "HIGH" and i.get("current_state") == "open")
        merged = sum(1 for i in issues.values() if i.get("current_state") == "closed")
        stalled = sum(1 for i in issues.values()
                       if i.get("current_state") == "open" and i.get("type") == "pull"
                       and i.get("current_comments", 0) == 0)

        totals["CRITICAL"] += critical
        totals["HIGH"] += high
        totals["MERGED"] += merged
        totals["STALLED"] += stalled

        output.append(f"  {framework:15s} │ CRITICAL: {critical} │ HIGH: {high} │ MERGED: {merged} │ STALLED: {stalled}")

    output.append("─" * 60)
    output.append(f"  TOTAL           │ CRITICAL: {totals['CRITICAL']} │ HIGH: {totals['HIGH']} │ MERGED: {totals['MERGED']} │ STALLED: {totals['STALLED']}")
    output.append("─" * 60)

    # Our drafts
    our_drafts = []
    for framework, issues in results.items():
        for number, info in issues.items():
            if info.get("our_draft"):
                our_drafts.append(f"  {info['our_draft']}: {framework} #{number}")

    if our_drafts:
        output.append("\nOUR COMMENT DRAFTS (need user authorization!):")
        for d in our_drafts:
            output.append(d)

    return "\n".join(output)


def find_recent_issues(repo, days=7):
    """Find issues updated in the last N days."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    try:
        # Get raw JSON and parse ourselves for better title extraction
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/issues?since={since}&per_page=20&state=open"],
            capture_output=True, timeout=30
        )
        if result.returncode != 0:
            return []
        issues_raw = json.loads(result.stdout)
        issues = []
        for item in issues_raw:
            # Skip PRs that appear in issues endpoint (they show up as type "pull_request")
            if "pull_request" in item:
                continue
            issues.append({
                "number": item["number"],
                "title": item["title"],
                "updated_at": item["updated_at"],
            })
        return issues
    except Exception:
        return []


def format_recent(days=7):
    """Show recently updated issues across all frameworks."""
    output = []
    output.append(f"RECENT ISSUES (last {days} days) — 7 Frameworks")
    output.append("=" * 80)

    for framework, config in MONITORED.items():
        repo = config["repo"]
        recent = find_recent_issues(repo, days)

        if recent:
            output.append(f"\n  {framework} ({repo}):")
            for issue in recent[:10]:
                num = issue["number"]
                # Check if we're already tracking this
                tracked = num in config["issues"]
                tag = " ★ TRACKED" if tracked else " ★★★ NEW!" if not tracked else ""
                output.append(f"    #{num}: {issue['title'][:60]}{tag}")
        else:
            output.append(f"\n  {framework} ({repo}): No recent issues found")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Seven Framework Issue Tracker")
    parser.add_argument("mode", choices=["track", "recent", "summary", "fresh"],
                        help="track=check all, recent=last 7 days, summary=dashboard, fresh=discover new")
    parser.add_argument("--days", type=int, default=7, help="Days to look back (recent/fresh modes)")
    parser.add_argument("--framework", type=str, default=None,
                        help="Filter to specific framework (DeepSpeed/Megatron/vLLM/verl/vLLM-Ascend/rLLM/SGLang/PyTorch)")
    args = parser.parse_args()

    if args.mode == "track":
        results, checked, failed = check_all_monitored()
        print(format_track(results, checked, failed))

    elif args.mode == "summary":
        results, _, _ = check_all_monitored()
        print(format_summary(results))

    elif args.mode == "recent":
        print(format_recent(args.days))

    elif args.mode == "fresh":
        output = []
        output.append("DISCOVERING NEW ISSUES NOT IN OUR TRACKER")
        output.append("=" * 80)

        new_found = 0
        for framework, config in MONITORED.items():
            repo = config["repo"]
            recent = find_recent_issues(repo, args.days)

            new_issues = [i for i in recent if str(i["number"]) not in config["issues"]]

            if new_issues:
                output.append(f"\n  {framework} ({repo}) — {len(new_issues)} NEW issues:")
                for issue in new_issues[:5]:
                    output.append(f"    #{issue['number']}: {issue['title'][:70]}")
                new_found += len(new_issues)
            else:
                output.append(f"\n  {framework} ({repo}) — No new issues")

        output.append(f"\n{'=' * 80}")
        output.append(f"TOTAL NEW ISSUES DISCOVERED: {new_found}")
        output.append("=" * 80)

        print("\n".join(output))


if __name__ == "__main__":
    main()
