# Cross-Framework LLM Serving Security Audit — LoRA Endpoint Auth

> 2026-06-18 | Source-level comparison of SGLang vs vLLM auth mechanisms
> ★★★★★★★★ Triggered by SGLang #28582 CVSS 9.8 RCE (missing auth decorator)
> ★★★★★★★★ vLLM has SAME vulnerability pattern — but different auth architecture
> ★★★★★★★★ Both frameworks: LoRA endpoints unprotected when --api-key NOT set!

---

## 1. SGLang Auth Architecture: Per-Endpoint Decorator

```
★★★★★★★★★ SGLang uses per-endpoint @auth_level decorator (sglang/srt/utils/auth.py):

AuthLevel enum:
  → NORMAL: legacy behavior (api_key protects all when configured)
  → ADMIN_OPTIONAL: can be accessed without key if no keys configured,
    or with api_key/admin_api_key depending on server config
  → ADMIN_FORCE: requires admin_api_key; if unset → 403 even with api_key

★★★★★★★★★ Middleware flow (add_api_key_middleware):
  → _ApiKeyASGIMiddleware intercepts ALL HTTP requests
  → Resolves auth level from route endpoint metadata (func._auth_level)
  → Calls decide_request_auth() → pure decision function → easy to test
  → If no auth_level decorator → defaults to NORMAL

★★★★★★★★★ VULNERABILITY (#28582):
  → /load_lora_adapter_from_tensors → NO @auth_level decorator
  → Defaults to NORMAL → if api_key NOT configured → COMPLETELY UNPROTECTED
  → Sibling endpoints (/load_lora_adapter, /unload_lora_adapter) → HAVE decorator
  → ★★★★★★★★ Pattern: new endpoint added → forgot decorator → NORMAL default → RCE!

★★★★★★★★★ SGLang auth PROBLEM:
  → Per-endpoint decorator = opt-in → EASY to miss on new endpoints
  → Default = NORMAL (unprotected) → missing decorator → silent vulnerability
  → Need: audit ALL endpoints → ensure state-modifying ones have decorator
  → ★★★★★★★★ Better: default to ADMIN_OPTIONAL → opt-out for safe endpoints
```

---

## 2. vLLM Auth Architecture: Global Middleware by Prefix

```
★★★★★★★★★ vLLM uses global middleware approach (vllm/entrypoints/serve/utils/server_utils.py):

AuthenticationMiddleware:
  → Pure ASGI middleware → checks ALL requests
  → GUARDED_PREFIX = ("/v1", "/v2", "/inference")
  → If path starts with GUARDED_PREFIX → verify bearer token
  → If path NOT in GUARDED_PREFIX → skip auth (health, metrics, etc.)
  → If --api-key NOT set → middleware NOT added → ALL endpoints unprotected!

★★★★★★★★★ vLLM auth flow:
  1. CLI args: --api-key or VLLM_API_KEY env var
  2. If any token configured → add AuthenticationMiddleware
  3. Middleware checks: Bearer token hash matches any configured token
  4. ALL /v1 routes → protected (including /v1/load_lora_adapter)
  5. Non-/v1 routes → unprotected (health, metrics, etc.)

★★★★★★★★★ vLLM LoRA endpoint protection:
  → /v1/load_lora_adapter → under /v1 prefix → PROTECTED when --api-key set
  → /v1/unload_lora_adapter → under /v1 prefix → PROTECTED when --api-key set
  → ★★★★★★★★ BUT: if --api-key NOT configured → BOTH UNPROTECTED!
  → Same vulnerability as SGLang #28582 → just different trigger condition

★★★★★★★★★ vLLM auth PROBLEM:
  → Global middleware = broader coverage → harder to miss individual endpoints
  → BUT: no per-endpoint granularity → ALL /v1 routes same auth level
  → Cannot differentiate: ADMIN endpoints vs NORMAL endpoints
  → ★★★★★★★★ No ADMIN_FORCE concept → all /v1 routes = same token
  → Cannot restrict LoRA loading to admin-only while allowing inference to regular users
```

---

## 3. Comparison: SGLang vs vLLM Auth

```
★★★★★★★★★ Auth architecture comparison:

| Aspect                | SGLang                    | vLLM                    |
|-----------------------|---------------------------|-------------------------|
| Mechanism             | Per-endpoint decorator    | Global prefix middleware |
| Default level         | NORMAL (unprotected)      | No middleware (unprotected) |
| Granularity           | Per-endpoint (3 levels)   | Per-prefix (1 level)    |
| ADMIN-only endpoints  | ADMIN_FORCE/Admin_OPTIONAL| Not supported           |
| Health/metrics bypass | Always allowed            | Excluded by prefix      |
| LoRA endpoint auth    | Decorator-dependent       | /v1 prefix-dependent    |
| Risk: missing auth    | Decorator forgotten       | --api-key not set       |
| Risk: new endpoints   | Default unprotected       | Protected if under /v1  |
| Best for              | Fine-grained control      | Broad coverage          |

★★★★★★★★★ SHARED vulnerability: BOTH unprotected when auth not configured!
  → SGLang: missing decorator on new endpoint → unprotected
  → vLLM: --api-key not set → all /v1 endpoints unprotected
  → ★★★★★★★★ BOTH: LoRA loading = UNPROTECTED by default in common deployments

★★★★★★★★★ RTX 4090 GRPO security assessment:
  → SGLang rollout: runs locally → network isolation → low risk
  → vLLM rollout: runs locally → network isolation → low risk
  → BUT: if deployed with network access → MUST configure auth!
  → ★★★★★★★★ verl HYBRID mode: SGLang/vLLM runs alongside trainer → localhost only → low risk
  → ★★★★★★★★ MUST: firewall or auth_required when exposed to network
```

---

## 4. Security Recommendations for RTX 4090

```
★★★★★★★★★ RTX 4090 LoRA serving security checklist:

1. SGLang deployments:
   → MUST: set --api-key or --admin-api-key
   → MUST: verify ALL LoRA endpoints have @auth_level decorator (after #28582 fix)
   → MUST: apply #28582 fix (add decorator + restrict snapshot_download)
   → MUST: network isolation if deployed beyond localhost

2. vLLM deployments:
   → MUST: set --api-key for ANY network-accessible deployment
   → RECOMMENDED: add per-endpoint auth granularity (ADMIN vs NORMAL)
   → MUST: network isolation if deployed beyond localhost
   → ★★★★★★★★ vLLM doesn't have ADMIN-only LoRA control → all /v1 same access level

3. verl HYBRID mode:
   → SGLang/vLLM rollout runs on localhost → NATURAL network isolation
   → Trainer communicates via IPC → not HTTP → no auth needed
   → ★★★★★★★★ RTX 4090 GRPO: HYBRID mode = SAFEST → localhost only → auth optional
   → BUT: if disaggregated deployment → MUST configure auth on rollout server

★★★★★★★★★ Universal security principle (from #28582):
  → ALL endpoints that modify model state → MUST have auth
  → ALL endpoints that access file system → MUST restrict to local paths
  → ALL endpoints that download from remote → MUST validate source
  → ★★★★★★★★ Default should be PROTECTED → opt-out for safe endpoints
```

---

## Key Findings Summary

★★★★★★★★★ SGLang #28582: per-endpoint decorator missed → NORMAL default → CVSS 9.8 RCE
★★★★★★★★★ vLLM: SAME vulnerability when --api-key NOT set → global middleware not added → ALL /v1 unprotected
★★★★★★★★★ vLLM advantage: new endpoints under /v1 → automatically protected (when auth configured)
★★★★★★★★★ SGLang advantage: per-endpoint granularity → ADMIN_FORCE vs ADMIN_OPTIONAL vs NORMAL
★★★★★★★★★ BOTH: LoRA endpoints unprotected by default → MUST configure auth for network deployments
★★★★★★★★★ RTX 4090: HYBRID mode = localhost only → natural isolation → low risk → but still MUST apply #28582
★★★★★★★★★ Recommendation: default-to-protected (opt-out) → better than default-to-unprotected (opt-in)

---

## References

- SGLang #28582: https://github.com/sgl-project/sglang/pull/28582
- SGLang auth.py: sglang/python/sglang/srt/utils/auth.py (209 lines, full source read)
- vLLM server_utils.py: vllm/vllm/entrypoints/serve/utils/server_utils.py (AuthenticationMiddleware)
- vLLM LoRA serving: vllm/vllm/entrypoints/openai/models/serving.py
- SGLang #28582 source-level reading: notebook/projects/sglang-28582-rce-security-vulnerability-reading.md
- verl HYBRID weight sync: notebook/projects/verl-fsdp2-source-deep-reading.md (6-step flow)
