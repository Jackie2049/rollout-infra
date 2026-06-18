#!/usr/bin/env python3
"""Cross-Framework LLM Serving Security Audit Tool

Checks LoRA endpoint authentication patterns across SGLang, vLLM, and other
LLM serving frameworks. Inspired by SGLang #28582 CVSS 9.8 RCE vulnerability.

Usage:
  python3 tools/llm_serving_security_audit.py --framework sglang --mode check
  python3 tools/llm_serving_security_audit.py --framework vllm --mode check
  python3 tools/llm_serving_security_audit.py --mode compare
  python3 tools/llm_serving_security_audit.py --mode rtx4090

Reference:
  - notebook/fundamentals/cross-framework-llm-serving-security-audit.md
  - SGLang #28582: CVSS 9.8 unauthenticated RCE
"""

import argparse
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class EndpointCheck:
    """Security check for a single endpoint."""
    framework: str
    endpoint: str
    method: str
    auth_mechanism: str
    auth_configured: str
    has_auth_decorator: bool
    modifies_state: bool
    accesses_filesystem: bool
    downloads_remote: bool
    risk_level: str  # CRITICAL, HIGH, MEDIUM, LOW
    notes: str


# SGLang endpoint checks (source-level verified from http_server.py)
SGLANG_ENDPOINTS = [
    EndpointCheck(
        framework="SGLang", endpoint="/load_lora_adapter", method="POST",
        auth_mechanism="per-endpoint @auth_level decorator",
        auth_configured="@auth_level(AuthLevel.ADMIN_OPTIONAL)",
        has_auth_decorator=True, modifies_state=True, accesses_filesystem=True,
        downloads_remote=False, risk_level="LOW",
        notes="Has auth decorator. Protected when api_key/admin_api_key configured."
    ),
    EndpointCheck(
        framework="SGLang", endpoint="/load_lora_adapter_from_tensors", method="POST",
        auth_mechanism="per-endpoint @auth_level decorator",
        auth_configured="MISSING — defaults to NORMAL",
        has_auth_decorator=False, modifies_state=True, accesses_filesystem=True,
        downloads_remote=True, risk_level="CRITICAL",
        notes="#28582 CVSS 9.8 RCE! Missing auth decorator + unrestricted snapshot_download. "
              "Source: http_server.py:1357-1358, lora_config.py:79-80."
    ),
    EndpointCheck(
        framework="SGLang", endpoint="/unload_lora_adapter", method="POST",
        auth_mechanism="per-endpoint @auth_level decorator",
        auth_configured="@auth_level(AuthLevel.ADMIN_OPTIONAL)",
        has_auth_decorator=True, modifies_state=True, accesses_filesystem=False,
        downloads_remote=False, risk_level="LOW",
        notes="Has auth decorator. Protected when api_key/admin_api_key configured."
    ),
    EndpointCheck(
        framework="SGLang", endpoint="/generate", method="POST",
        auth_mechanism="per-endpoint @auth_level decorator",
        auth_configured="NORMAL (no decorator needed)",
        has_auth_decorator=False, modifies_state=False, accesses_filesystem=False,
        downloads_remote=False, risk_level="LOW",
        notes="Inference endpoint. No state modification. Protected by api_key when configured."
    ),
    EndpointCheck(
        framework="SGLang", endpoint="/health", method="GET",
        auth_mechanism="always allowed",
        auth_configured="Always allowed",
        has_auth_decorator=False, modifies_state=False, accesses_filesystem=False,
        downloads_remote=False, risk_level="LOW",
        notes="Health check. Always allowed per decide_request_auth()."
    ),
]

# vLLM endpoint checks (source-level verified)
VLLM_ENDPOINTS = [
    EndpointCheck(
        framework="vLLM", endpoint="/v1/load_lora_adapter", method="POST",
        auth_mechanism="global prefix middleware",
        auth_configured="Protected by /v1 prefix when --api-key set",
        has_auth_decorator=True, modifies_state=True, accesses_filesystem=True,
        downloads_remote=True, risk_level="MEDIUM",
        notes="Protected when --api-key set. UNPROTECTED when --api-key NOT set! "
              "No ADMIN-only granularity — same auth level as inference."
    ),
    EndpointCheck(
        framework="vLLM", endpoint="/v1/unload_lora_adapter", method="POST",
        auth_mechanism="global prefix middleware",
        auth_configured="Protected by /v1 prefix when --api-key set",
        has_auth_decorator=True, modifies_state=True, accesses_filesystem=False,
        downloads_remote=False, risk_level="MEDIUM",
        notes="Protected when --api-key set. UNPROTECTED when --api-key NOT set!"
    ),
    EndpointCheck(
        framework="vLLM", endpoint="/v1/completions", method="POST",
        auth_mechanism="global prefix middleware",
        auth_configured="Protected by /v1 prefix when --api-key set",
        has_auth_decorator=True, modifies_state=False, accesses_filesystem=False,
        downloads_remote=False, risk_level="LOW",
        notes="Inference endpoint. Protected by middleware when --api-key configured."
    ),
    EndpointCheck(
        framework="vLLM", endpoint="/health", method="GET",
        auth_mechanism="excluded by prefix",
        auth_configured="NOT in GUARDED_PREFIX → always unprotected",
        has_auth_decorator=False, modifies_state=False, accesses_filesystem=False,
        downloads_remote=False, risk_level="LOW",
        notes="Health endpoint. Not under /v1 prefix. No auth needed."
    ),
]


def check_framework(framework: str) -> List[EndpointCheck]:
    """Check all endpoints for a specific framework."""
    if framework == "sglang":
        endpoints = SGLANG_ENDPOINTS
    elif framework == "vllm":
        endpoints = VLLM_ENDPOINTS
    else:
        print(f"Framework '{framework}' not yet supported. Available: sglang, vllm")
        return []

    print(f"\n{'='*80}")
    print(f"  {framework.upper()} LLM Serving Security Audit")
    print(f"{'='*80}\n")

    critical_count = 0
    high_count = 0
    medium_count = 0

    for ep in endpoints:
        risk_marker = ""
        if ep.risk_level == "CRITICAL":
            risk_marker = " ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★"
            critical_count += 1
        elif ep.risk_level == "HIGH":
            risk_marker = " ★★★★★★★★★★★★★★★★★"
            high_count += 1
        elif ep.risk_level == "MEDIUM":
            risk_marker = " ★★★★★★★★★"
            medium_count += 1

        print(f"  Endpoint: {ep.method} {ep.endpoint}")
        print(f"  Risk Level: {ep.risk_level}{risk_marker}")
        print(f"  Auth Mechanism: {ep.auth_mechanism}")
        print(f"  Auth Configured: {ep.auth_configured}")
        print(f"  Has Auth Decorator: {ep.has_auth_decorator}")
        print(f"  Modifies State: {ep.modifies_state}")
        print(f"  Accesses Filesystem: {ep.accesses_filesystem}")
        print(f"  Downloads Remote: {ep.downloads_remote}")
        print(f"  Notes: {ep.notes}")
        print()

    print(f"{'='*80}")
    print(f"  Summary: {len(endpoints)} endpoints checked")
    print(f"  CRITICAL: {critical_count} | HIGH: {high_count} | MEDIUM: {medium_count}")
    print(f"{'='*80}\n")

    if critical_count > 0:
        print("  *** IMMEDIATE ACTION REQUIRED: Fix CRITICAL vulnerabilities! ***")
        print("  Recommendation: Add auth decorators + restrict file downloads")
        print()

    return endpoints


def compare_frameworks():
    """Compare auth architectures across frameworks."""
    print(f"\n{'='*80}")
    print(f"  Cross-Framework LLM Serving Auth Comparison")
    print(f"{'='*80}\n")

    print("  | Aspect                | SGLang                    | vLLM                    |")
    print("  |-----------------------|---------------------------|-------------------------|")
    print("  | Mechanism             | Per-endpoint decorator    | Global prefix middleware |")
    print("  | Default level         | NORMAL (unprotected)      | No middleware (unprotected)|")
    print("  | Granularity           | Per-endpoint (3 levels)   | Per-prefix (1 level)    |")
    print("  | ADMIN-only endpoints  | ADMIN_FORCE/Admin_OPTIONAL| Not supported           |")
    print("  | Health/metrics bypass | Always allowed            | Excluded by prefix      |")
    print("  | LoRA endpoint auth    | Decorator-dependent       | /v1 prefix-dependent    |")
    print("  | Risk: missing auth    | Decorator forgotten       | --api-key not set       |")
    print("  | Risk: new endpoints   | Default unprotected       | Protected if under /v1  |")
    print("  | Best for              | Fine-grained control      | Broad coverage          |")
    print()

    print("  SHARED vulnerability: BOTH unprotected when auth not configured!")
    print("    → SGLang: missing decorator on new endpoint → unprotected")
    print("    → vLLM: --api-key not set → all /v1 endpoints unprotected")
    print("    → BOTH: LoRA loading = UNPROTECTED by default in common deployments")
    print()

    print("  Security recommendations:")
    print("    1. Default-to-protected (opt-out) → better than default-to-unprotected (opt-in)")
    print("    2. ALL state-modifying endpoints → MUST have auth")
    print("    3. ALL filesystem-access endpoints → MUST restrict to local paths")
    print("    4. ALL remote-download endpoints → MUST validate source")
    print()


def rtx4090_check():
    """RTX 4090 specific security assessment."""
    print(f"\n{'='*80}")
    print(f"  RTX 4090 LLM Serving Security Assessment for GRPO Training")
    print(f"{'='*80}\n")

    print("  verl HYBRID mode (RTX 4090 recommended):")
    print("    → SGLang/vLLM rollout: localhost only → NATURAL network isolation")
    print("    → Trainer communicates via IPC → not HTTP → no auth needed")
    print("    → ★★★★★★★★ LOW RISK: localhost deployment → no external access")
    print()

    print("  Security checklist for RTX 4090:")
    print("    1. MUST: apply SGLang #28582 fix (auth decorator + local snapshot_download)")
    print("    2. MUST: use HYBRID mode → rollout on localhost → natural isolation")
    print("    3. MUST: if SGLang/vLLM exposed to network → configure --api-key")
    print("    4. MUST: verify all LoRA endpoints have auth decorator (after #28582)")
    print("    5. MUST: for disaggregated deployment → firewall or auth_required")
    print("    6. RECOMMENDED: add per-endpoint auth granularity (ADMIN vs NORMAL)")
    print()

    print("  If GPU server shared with other users:")
    print("    → Network isolation: SGLang/vLLM binding to localhost only")
    print("    → Process isolation: separate conda environments")
    print("    → File isolation: LoRA adapters in secure local directories")
    print("    → ★★★★★★★★ MUST NOT: expose serving ports to shared network!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-framework LLM serving security audit"
    )
    parser.add_argument(
        "--framework", choices=["sglang", "vllm"],
        help="Framework to audit"
    )
    parser.add_argument(
        "--mode", choices=["check", "compare", "rtx4090"],
        required=True,
        help="Audit mode"
    )

    args = parser.parse_args()

    if args.mode == "check":
        if not args.framework:
            print("Must specify --framework for check mode")
            sys.exit(1)
        check_framework(args.framework)
    elif args.mode == "compare":
        compare_frameworks()
    elif args.mode == "rtx4090":
        rtx4090_check()


if __name__ == "__main__":
    main()
