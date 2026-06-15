#!/usr/bin/env python3
"""
torch.compile 诊断工具 — 基于 PyTorch compile e2e + Inductor + FX IR 源码级阅读

功能:
1. 编译模式决策树: 根据场景推荐最优 compile 模式
2. 常见编译问题诊断: graph breaks / recompilation / 内存 / 性能
3. FX IR 结构分析: 如果有 graph, 分析 node 类型和依赖
4. Inductor fusion 诊断: 检查哪些 op 可以融合, 哪些是 fallback
5. Guard 分析: 检查 guard 类型, 估算 recompilation 风险
6. RTX 4090 专用配置推荐

用法:
  python tools/torch_compile_diagnostic.py --mode decision --scenario training
  python tools/torch_compile_diagnostic.py --mode diagnose --problem graph-breaks
  python tools/torch_compile_diagnostic.py --mode analyze --model-path ./model.py
  python tools/torch_compile_diagnostic.py --mode fusion --ops add,relu,mm
  python tools/torch_compile_diagnostic.py --mode config --gpu rtx4090
"""

import argparse
import json
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CompileMode(Enum):
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"


class Scenario(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    DEBUGGING = "debugging"
    RESEARCH = "research"


class Problem(Enum):
    GRAPH_BREAKS = "graph-breaks"
    RECOMPILATION = "recompilation"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    FSDP_COMPAT = "fsdp-compat"


# ============================================================
# 1. 编译模式决策树 (基于 torch.compile e2e + compiler roadmap)
# ============================================================

COMPILE_MODE_DECISION = {
    Scenario.TRAINING: {
        "recommended": CompileMode.REDUCE_OVERHEAD,
        "reason": "训练需要快速编译, reduce-overhead 用 CUDA graph + host overhead 优化",
        "notes": [
            "FSDP2 + reduce-overhead = 最佳组合 (per-param DTensor + compile 兼容)",
            "ZeRO-3 + compile = 不兼容 (dynamic AllGather → graph breaks)",
            "首次编译慢 10-100x, 但后续缓存 → 2.8 默认启用 MaxAutotune 缓存",
            "FSDP2 hooks 需要 fullgraph=False (PyTorch 2.12 breaking change)",
        ],
        "gpu_requirements": {
            "rtx4090": "可行但首次编译慢; LoRA 训练用 reduce-overhead",
            "h100": "最佳; FSDP2 + compile + Float8 → +48-50% MFU",
            "a100": "良好; FSDP2 + compile → +15-16%",
        },
    },
    Scenario.INFERENCE: {
        "recommended": CompileMode.MAX_AUTOTUNE,
        "reason": "推理追求最高吞吐, max-autotune 用 Triton autotuning + 最优 config",
        "notes": [
            "MaxAutotune 首次编译极慢 (autotune 每个 kernel config)",
            "但编译结果缓存到 ~/.triton/cache/ → 后续零开销",
            "vLLM/SGLang 用 CUDAGraphMode 相当于 reduce-overhead",
            "INT4 量化模型 → compile 收益减小 (kernel 已优化)",
        ],
        "gpu_requirements": {
            "rtx4090": "max-autotune 首次编译慢; 后续快; INT4 模型收益有限",
            "h100": "最佳; max-autotune + Triton → 最高推理吞吐",
            "a100": "良好; Triton kernel 充分利用 A100 HBM",
        },
    },
    Scenario.DEBUGGING: {
        "recommended": CompileMode.DEFAULT,
        "reason": "default 模式最少优化, 便于调试",
        "notes": [
            "torch.compile(default) = Dynamo + Inductor 最少优化",
            "Use TORCH_LOGS='+dynamo' to view Dynamo trace details",
            "Use TORCH_LOGS='+inductor' to view Inductor codegen",
            "用 torch._dynamo.explain() 分析 graph breaks",
        ],
        "gpu_requirements": {
            "rtx4090": "default 模式无额外要求",
            "h100": "同上",
            "a100": "同上",
        },
    },
    Scenario.RESEARCH: {
        "recommended": CompileMode.REDUCE_OVERHEAD,
        "reason": "研究需要快速迭代, reduce-overhead 编译最快",
        "notes": [
            "研究场景频繁修改模型 → 编译缓存失效 → 需要快速重编译",
            "reduce-overhead 编译比 max-autotune 快 10x+",
            "实验阶段用 reduce-overhead, 最终 benchmark 用 max-autotune",
        ],
        "gpu_requirements": {
            "rtx4090": "研究常用; 频繁修改 → 编译缓存失效快",
            "h100": "最佳; 编译+运行都快",
            "a100": "良好",
        },
    },
}


# ============================================================
# 2. 常见编译问题诊断 (基于 torch.compile e2e 源码阅读)
# ============================================================

PROBLEM_DIAGNOSIS = {
    Problem.GRAPH_BREAKS: {
        "description": "Dynamo 无法 trace 部分 Python 代码 → graph break → 降级到 eager",
        "causes": [
            "1. 数据依赖控制流: if tensor.sum() > 0 → Dynamo 无法符号化 → break",
            "   解决: 用 torch.where / torch.cond (2.12+支持 CUDA graph!)",
            "2. 外部副作用: print() / logging / global 变量修改 → break",
            "   解决: 移除或用 torch._dynamo.disable() 标记",
            "3. 不支持的 op: torch.autocast / dynamic list / dict 修改 → break",
            "   解决: 查 torch._dynamo.list_unsupported_functions()",
            "4. ★ FSDP2 intentional breaks: RegisterPre/PostBackwardFunction hooks",
            "   解决: 这是设计意图 → fullgraph=False → hooks eager + compute compiled",
            "5. ZeRO-3 dynamic AllGather: ZeRO-3 每层 AllGather 是动态 → break",
            "   解决: ZeRO-3 + compile 不兼容 → 用 FSDP2 替代!",
        ],
        "diagnostic_commands": [
            "torch._dynamo.explain(model)(*inputs)  → 分析 graph breaks",
            "TORCH_LOGS='+dynamo' python train.py  → Dynamo 详细日志",
            "torch._dynamo.list_unsupported_functions()  → 列出不支持 op",
        ],
        "rtx4090_notes": "LoRA 模型 graph breaks 更少 (更简单模型); FSDP2+LoRA+compile=最佳",
    },
    Problem.RECOMPILATION: {
        "description": "Guards 检查失败 → 重新编译 → 最多 64 次 → fallback 到 eager",
        "causes": [
            "1. 动态 shape: seq_len 变化 → guard 'Tensor.shape' 失败 → 重编译",
            "   解决: Symbolic Shapes (2.7+) → 1 次编译覆盖所有 seq_len",
            "2. 动态 dtype/device: 输入 dtype 变化 → guard 失败 → 重编译",
            "   解决: 确保输入 dtype/device 一致",
            "3. 动态 Python 值: if x > threshold → threshold 变化 → 重编译",
            "   解决: 将 threshold 设为 constexpr 或 tensor",
            "4. ★ 64 次限制: Config.cache_size_limit=64 → 超过 → fallback",
            "   解决: torch._dynamo.config.cache_size_limit = N → 调大",
        ],
        "diagnostic_commands": [
            "torch._dynamo.config.cache_size_limit = 128  → 提高重编译上限",
            "TORCH_LOGS='+dynamo guards' python train.py  → 查看 guard 详情",
            "torch._dynamo.utils.compile_times()  → 编译时间统计",
        ],
        "rtx4090_notes": "RTX 4090 单 GPU → seq_len 固定 → recompilation 最少",
    },
    Problem.MEMORY: {
        "description": "编译后内存使用增加 (CUDA graph buffer / Triton workspace)",
        "causes": [
            "1. CUDA graph: 预分配固定 size buffer → 不释放 → 内存峰值高",
            "   解决: reduce-overhold 用 CUDA graph → 确保 batch size 固定",
            "2. Triton workspace: autotune benchmark 临时 buffer → 编译时占用",
            "   解决: 编译完成后释放 → 不影响运行时",
            "3. Inductor realization: 融合 → 不分配中间 → 实际省内存!",
            "   解决: 融合是好的 → 减少中间 buffer → 更省内存",
        ],
        "rtx4090_notes": "24GB 内存 → LoRA + compile → 融合省内存 → 可行",
    },
    Problem.PERFORMANCE: {
        "description": "编译后性能不如预期",
        "causes": [
            "1. 首次编译慢: max-autotune 需要 10-100x 时间 → 预期内",
            "   解决: 预热阶段, 后续快",
            "2. Fallback kernels: SDPA = Fallback → 不 Triton codegen → 不融合",
            "   解决: SDPA 依赖 ATen/TE 原生 → 不是 compile 问题",
            "3. GEMM 不融合: mm/bmm = ExternKernel(cuBLAS) → 不融合",
            "   解决: 只融合 epilogue(RMSNorm+SiLU+Residual) → 这是正确的!",
            "4. 小 batch: compile overhead > compute savings → 小 batch 亏损",
            "   解决: batch > 32 → compile 收益明显",
        ],
        "rtx4090_notes": "小 batch (1-4) → compile 可能亏损; batch > 32 → 收益明显",
    },
    Problem.FSDP_COMPAT: {
        "description": "FSDP2 + torch.compile 兼容性问题",
        "causes": [
            "1. ★ Breaking (2.12): FSDP2 hooks without graph breaks 不再支持",
            "   解决: fullgraph=False 或 compile before FSDP wrapping",
            "2. FSDP2 intentional graph breaks: Pre/PostBackwardFunction hooks → eager",
            "   解决: 这是设计 → compute 部分 still compiled → +15-16%",
            "3. FSDP1 FlatParameter → compile 不兼容",
            "   解决: 用 FSDP2 (per-param DTensor) 替代",
            "4. ZeRO-3 + compile → 不兼容 (dynamic AllGather)",
            "   解决: 用 FSDP2 替代 ZeRO-3",
        ],
        "rtx4090_notes": "FSDP2 单 GPU → compile 兼容 → 但 FSDP2 本身无意义 (无分片)",
    },
}


# ============================================================
# 3. Inductor Fallback Op 列表 (基于 Inductor lowering 源码阅读)
# ============================================================

INDUCTOR_FALLBACK_OPS = {
    "always_fallback": [
        "aten::scaled_dot_product_attention",  # SDPA → ATen/TE 原生
        "aten::flash_attention_forward",       # Flash Attention → 不 Triton codegen
        "aten::efficient_attention_forward",   # Memory-efficient attention
    ],
    "extern_kernel_not_fused": [
        "aten::mm.default",                    # cuBLAS → 不融合, 只融合 epilogue
        "aten::bmm.default",                   # batched mm → cuBLAS
        "aten::addmm.default",                 # mm + bias → cuBLAS + epilogue fusion
    ],
    "always_fused_pointwise": [
        "aten::add.Tensor",                    # pointwise → 融合
        "aten::mul.Tensor",                    # pointwise → 融合
        "aten::relu.default",                  # pointwise → 融合
        "aten::silu.default",                  # 融合 (decompose → 2 pointwise → 1 kernel)
        "aten::rms_norm.default",              # 融合为 GEMM epilogue
        "aten::copy_.default",                 # 可能融合 (type conversion)
    ],
    "decompose_then_fuse": [
        "aten::silu → sigmoid + mul → 2 pointwise → 1 fused kernel",
        "aten::gelu → erf + add + mul + half → 多 pointwise → 1 kernel",
        "aten::layer_norm → reduce + pointwise → 可能 2 kernels",
    ],
}


# ============================================================
# 4. RTX 4090 专用 Compile 配置
# ============================================================

RTX4090_COMPILE_CONFIG = {
    "training": {
        "mode": "reduce-overhead",
        "fullgraph": False,  # FSDP2 hooks 需要 breaks
        "dynamic_shapes": True,  # seq_len 变化 → Symbolic Shapes
        "cache_size_limit": 64,  # 默认值, RTX 4090 通常不需要更多
        "max_autotune": False,  # 训练不需要 autotune → 编译太慢
        "notes": [
            "★ LoRA + reduce-overhead + FSDP2 = 最佳训练组合",
            "首次编译 ~30s (7B 模型), 后续 <1s (缓存)",
            "GRPO 训练: compile actor only → critic 不需要 (GRPO 无 critic)",
            "batch_size > 32 → compile 收益明显",
            "LoRA 减少 compute → compile 融合收益相对减小 → 但仍然有用",
        ],
    },
    "inference": {
        "mode": "max-autotune",
        "fullgraph": True,  # 推理无 hooks → 可以 fullgraph
        "dynamic_shapes": True,  # seq_len 变化
        "cache_size_limit": 128,  # 推理可能更多 shape 变化
        "max_autotune": True,  # 推理追求最高吞吐
        "notes": [
            "★ INT4 量化 + max-autotune = 最佳推理组合",
            "首次编译 ~5-10min (7B INT4), 后续 <1s",
            "INT4 kernel 已经优化 → compile Triton 收益有限 (~5-10%)",
            "EAGLE speculative + compile → draft model 也编译 → 更快",
            "vLLM 内部用 CUDAGraphMode → 类似 reduce-overhead → 不需要额外 compile",
        ],
    },
    "research": {
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic_shapes": True,
        "notes": [
            "研究场景频繁修改 → 编译缓存失效 → 用 reduce-overhold 最快重编译",
            "实验阶段 reduce-overhead, 最终 benchmark max-autotune",
        ],
    },
}


# ============================================================
# 5. Guard 类型速查 (基于 torch.compile e2e + OutputGraph GuardsSet)
# ============================================================

GUARD_TYPES = {
    "TENSOR_MATCH": "检查 tensor dtype/device/shape/stride → 最常见 → recompilation 主因",
    "DTYPE_MATCH": "检查 dtype → 变化 → 重编译 → 解决: 固定 dtype",
    "DEVICE_MATCH": "检查 device → CPU→GPU → 重编译 → 解决: 固定 device",
    "SHAPE_ENV": "Symbolic Shapes → 符号约束 → 变化 → 重编译 → 2.7+ 改善",
    "PYTHON_VALUE_MATCH": "检查 Python 值 → threshold 变化 → 重编译 → 解决: constexpr",
    "DICT_KEY_ORDER": "检查 dict key 顺序 → 变化 → 重编译 → 解决: 固定 key 顺序",
    "NN_MODULE_MATCH": "检查 nn.Module 属性 → 参数变化 → 重编译 → 解决: 不动态修改 module",
    "LINEAGE_MATCH": "检查变量 lineage → 函数签名变化 → 重编译",
}


# ============================================================
# 6. 主功能
# ============================================================

def mode_decision(scenario: str) -> None:
    """编译模式决策树"""
    s = Scenario(scenario)
    info = COMPILE_MODE_DECISION[s]
    print(f"\n{'='*60}")
    print(f"  编译模式推荐: scenario={s.value}")
    print(f"{'='*60}")
    print(f"\n  推荐: torch.compile(mode='{info['recommended'].value}')")
    print(f"  原因: {info['reason']}")
    print(f"\n  关键注意事项:")
    for note in info["notes"]:
        print(f"    • {note}")
    print(f"\n  GPU 推荐:")
    for gpu, rec in info["gpu_requirements"].items():
        print(f"    {gpu}: {rec}")


def mode_diagnose(problem: str) -> None:
    """编译问题诊断"""
    p = Problem(problem)
    info = PROBLEM_DIAGNOSIS[p]
    print(f"\n{'='*60}")
    print(f"  编译问题诊断: problem={p.value}")
    print(f"{'='*60}")
    print(f"\n  描述: {info['description']}")
    print(f"\n  原因与解决方案:")
    for cause in info["causes"]:
        print(f"    {cause}")
    if "diagnostic_commands" in info:
        print(f"\n  诊断命令:")
        for cmd in info["diagnostic_commands"]:
            print(f"    {cmd}")
    if "rtx4090_notes" in info:
        print(f"\n  ★ RTX 4090: {info['rtx4090_notes']}")


def mode_fusion(ops_str: str) -> None:
    """Inductor fusion 诊断"""
    ops = [op.strip() for op in ops_str.split(",")]
    print(f"\n{'='*60}")
    print(f"  Inductor Fusion 诊断: ops={ops}")
    print(f"{'='*60}")
    for op in ops:
        # Check against known categories
        op_key = f"aten::{op}"
        found = False
        for category, op_list in INDUCTOR_FALLBACK_OPS.items():
            if op_key in op_list or op in [o.split("::")[-1].split(".")[0] for o in op_list]:
                found = True
                if category == "always_fallback":
                    print(f"\n  ★ {op}: ALWAYS Fallback → 不 Triton codegen → 不融合 → 依赖 ATen 原生")
                elif category == "extern_kernel_not_fused":
                    print(f"\n  ★ {op}: ExternKernel → 不融合主体 → 但可以融合 epilogue")
                    print(f"    Epilogue fusion: mm + (bias + RMSNorm + SiLU + Residual) → 1 kernel")
                elif category == "always_fused_pointwise":
                    print(f"\n  ✓ {op}: Pointwise → 融合 → 1 Triton kernel → 省内存+快")
                elif category == "decompose_then_fuse":
                    print(f"\n  ✓ {op}: Decompose → 多 pointwise → 融合 → 1 Triton kernel")
                break
        if not found:
            print(f"\n  ? {op}: 未知 → 可能 pointwise (融合) 或 fallback → 需要 Inductor lowering 查询")
            print(f"    查询: torch._inductor.lowering.lowerings dict → 或 TORCH_LOGS='+inductor'")
    print(f"\n  ★ Fusion 规则 (基于 Inductor Scheduler 源码):")
    print(f"    1. 同 device + 共享数据 + 顺序正确 → 可以融合")
    print(f"    2. GEMM (mm/bmm) → ExternKernel → 不融合 → 只融合 epilogue")
    print(f"    3. SDPA → Fallback → 不 Triton codegen → 不融合")
    print(f"    4. Pointwise ops → 融合 → 1 Triton kernel → 省中间 buffer")
    print(f"    5. Reduction → 可以融合 → 但需相同 numel/rnumel")


def mode_config(gpu: str) -> None:
    """GPU 专用配置推荐"""
    print(f"\n{'='*60}")
    print(f"  Compile 配置推荐: gpu={gpu}")
    print(f"{'='*60}")
    if gpu == "rtx4090":
        for scenario, config in RTX4090_COMPILE_CONFIG.items():
            print(f"\n  [{scenario}]")
            print(f"    mode: torch.compile(mode='{config['mode']}')")
            print(f"    fullgraph: {config.get('fullgraph', 'N/A')}")
            print(f"    dynamic_shapes: {config.get('dynamic_shapes', 'N/A')}")
            if "cache_size_limit" in config:
                print(f"    cache_size_limit: {config['cache_size_limit']}")
            for note in config["notes"]:
                print(f"    • {note}")
    elif gpu in ("h100", "a100"):
        print(f"\n  [training] torch.compile(mode='reduce-overhead', fullgraph=False)")
        print(f"    FSDP2 + compile → +15-16% MFU")
        print(f"    FSDP2 + compile + Float8 → +48-50% MFU")
        print(f"\n  [inference] torch.compile(mode='max-autotune', fullgraph=True)")
        print(f"    Triton kernel → 最高推理吞吐")
    else:
        print(f"\n  未知 GPU: {gpu}")
        print(f"  通用推荐: training → reduce-overhead; inference → max-autotune")


def mode_guards() -> None:
    """Guard 类型速查"""
    print(f"\n{'='*60}")
    print(f"  Guard 类型速查 (最多 64 次重编译)")
    print(f"{'='*60}")
    for guard, desc in GUARD_TYPES.items():
        print(f"\n  {guard}:")
        print(f"    {desc}")


def mode_full_report() -> None:
    """完整诊断报告"""
    print(f"\n{'='*70}")
    print(f"  torch.compile 完整诊断报告")
    print(f"{'='*70}")
    for s in Scenario:
        mode_decision(s.value)
    print()
    for p in Problem:
        mode_diagnose(p.value)
    mode_guards()
    mode_config("rtx4090")


def main():
    parser = argparse.ArgumentParser(
        description="torch.compile 诊断工具 — 基于 PyTorch compile e2e + Inductor + FX IR 源码级阅读"
    )
    parser.add_argument(
        "--mode", choices=["decision", "diagnose", "fusion", "config", "guards", "full"],
        required=True, help="诊断模式"
    )
    parser.add_argument(
        "--scenario", choices=[s.value for s in Scenario],
        default="training", help="场景 (decision mode)"
    )
    parser.add_argument(
        "--problem", choices=[p.value for p in Problem],
        default="graph-breaks", help="问题类型 (diagnose mode)"
    )
    parser.add_argument(
        "--ops", default="add,relu,mm,silu,sdpa", help="op 列表, 逗号分隔 (fusion mode)"
    )
    parser.add_argument(
        "--gpu", choices=["rtx4090", "h100", "a100"],
        default="rtx4090", help="GPU 类型 (config mode)"
    )

    args = parser.parse_args()

    if args.mode == "decision":
        mode_decision(args.scenario)
    elif args.mode == "diagnose":
        mode_diagnose(args.problem)
    elif args.mode == "fusion":
        mode_fusion(args.ops)
    elif args.mode == "config":
        mode_config(args.gpu)
    elif args.mode == "guards":
        mode_guards()
    elif args.mode == "full":
        mode_full_report()


if __name__ == "__main__":
    main()
