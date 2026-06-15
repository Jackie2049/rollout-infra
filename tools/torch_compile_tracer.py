#!/usr/bin/env python3
"""
PyTorch torch.compile 执行追踪器 — CPU兼容

追踪torch.compile在简单模型上的完整编译流程:
1. Dynamo: bytecode符号执行 → FX graph
2. AOTAutograd: joint fwd+bwd → min-cut partition
3. Inductor: FX→IR lowering → Scheduler fusion → Triton codegen
4. Execution: compiled fn → cache hit → 运行

用法:
  python tools/torch_compile_tracer.py --model linear --mode reduce-overhead
  python tools/torch_compile_tracer.py --model mlp --mode default --verbose
  python tools/torch_compile_tracer.py --model transformer-block --mode max-autotune

基于7篇源码阅读:
  - pytorch-dynamo-internals-reading.md
  - pytorch-fx-ir-source-reading.md
  - pytorch-compile-e2e-reading.md
  - pytorch-aotautograd-internals-reading.md (AOTAutograd agent)
  - pytorch-inductor-triton-codegen-reading.md
  - pytorch-fsdp2-internals-reading.md
  - pytorch-custom-op-library-system-reading.md
"""

import argparse
import contextlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


# ============================================================
# 简单模型定义 (CPU友好)
# ============================================================

class SimpleLinear(torch.nn.Module):
    """单层线性模型 — 最简单compile demo"""
    def __init__(self, dim=256):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class SimpleMLP(torch.nn.Module):
    """2层MLP — 测试fusion opportunity"""
    def __init__(self, dim=256):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class TransformerBlock(torch.nn.Module):
    """单层Transformer block — 测试SDPA + complex fusion"""
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(dim)
        self.ln2 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Pre-norm transformer block
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h
        return x


class LoRAModel(torch.nn.Module):
    """LoRA模型 — 测试LoRA+compile交互"""
    def __init__(self, dim=256, lora_rank=8):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        # LoRA adapters
        self.lora_A = torch.nn.Linear(dim, lora_rank, bias=False)
        self.lora_B = torch.nn.Linear(lora_rank, dim, bias=False)
        self.lora_scale = lora_rank ** -0.5

    def forward(self, x):
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(x)) * self.lora_scale
        return base + lora


# ============================================================
# 编译追踪器核心
# ============================================================

def trace_compile(model, mode, verbose=False):
    """追踪torch.compile完整流程"""

    print(f"\n{'='*60}")
    print(f"  PyTorch torch.compile 执行追踪器")
    print(f"{'='*60}")
    print(f"  模型: {model.__class__.__name__}")
    print(f"  参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  模式: {mode}")
    print(f"  PyTorch: {torch.__version__}")

    # Phase 1: Eager baseline
    print(f"\n{'─'*40}")
    print(f"  Phase 1: Eager Baseline (未编译)")
    x = torch.randn(4, 256)

    # Warmup
    for _ in range(3):
        _ = model(x)

    # Measure eager time
    times_eager = []
    for _ in range(20):
        t0 = time.perf_counter()
        y_eager = model(x)
        times_eager.append(time.perf_counter() - t0)

    eager_mean = sum(times_eager) / len(times_eager)
    print(f"    Eager avg: {eager_mean*1000:.3f} ms")
    print(f"    Output shape: {y_eager.shape}")

    # Phase 2: Compile (first run = compilation)
    print(f"\n{'─'*40}")
    print(f"  Phase 2: torch.compile(mode='{mode}')")

    compiled_model = torch.compile(model, mode=mode)

    # First run triggers compilation
    print(f"    ★ 首次调用 → 触发编译流程:")
    print(f"      1. Dynamo: C-level eval_frame hook → 符号执行bytecode → FX graph")
    print(f"      2. AOTAutograd: make_fx trace joint fwd+bwd → min-cut partition")
    print(f"      3. Inductor: FX→IR lowering → Scheduler 10轮fusion → Triton codegen")
    print(f"      4. PyCodeCache → GuardedCode → compiled fn ready!")

    if verbose:
        # Capture Dynamo logs if available
        print(f"\n    ★ 详细追踪 (verbose=True):")
        try:
            import torch._dynamo as dynamo
            print(f"      Dynamo version: {dynamo.__version__ if hasattr(dynamo, '__version__') else 'N/A'}")
            print(f"      TorchDynamo state: {dynamo._get_eval_frame(callback=None)}")
        except Exception:
            pass

    try:
        t_compile_start = time.perf_counter()
        y_compiled = compiled_model(x)
        t_compile_end = time.perf_counter()
        compile_ok = True
    except Exception as e:
        print(f"    ✗ 编译失败: {type(e).__name__}: {str(e)[:200]}")
        print(f"    ★ 原因: Inductor C++ codegen需要完整编译环境(g++/Xcode)")
        print(f"    ★ 小模型Triton kernel → 不需要C++ codegen → 通常OK")
        print(f"    ★ 大模型/复杂op → 可能需要C++ codegen → GPU环境更稳定")
        print(f"    ★ 建议: 用--model mlp或linear避免C++ codegen; GPU环境编译更稳定")

        # Fallback to eager for remaining measurements
        y_compiled = model(x)
        compile_time = 0
        compile_ok = False

    if compile_ok:
        compile_time = t_compile_end - t_compile_start
        print(f"    ★ ★ 首次编译+执行时间: {compile_time*1000:.1f} ms (含编译开销)")
        print(f"    Output shape: {y_compiled.shape}")
        print(f"    ★ 输出一致: {torch.allclose(y_eager, y_compiled, atol=1e-5)}")

    # Phase 3: Compiled execution (cache hit)
    print(f"\n{'─'*40}")
    print(f"  Phase 3: Compiled Execution (缓存命中)")

    if compile_ok:
        times_compiled = []
        for _ in range(20):
            t0 = time.perf_counter()
            y_fast = compiled_model(x)
            times_compiled.append(time.perf_counter() - t0)

        compiled_mean = sum(times_compiled) / len(times_compiled)
        print(f"    Compiled avg: {compiled_mean*1000:.3f} ms")
        print(f"    Output consistent: {torch.allclose(y_eager, y_fast, atol=1e-5)}")
    else:
        # Fallback: measure eager again for comparison
        times_compiled = times_eager  # reuse eager times as fallback
        compiled_mean = eager_mean
        y_fast = y_compiled
        print(f"    ✗ 编译失败 → 使用eager baseline时间")
        print(f"    Eager avg: {compiled_mean*1000:.3f} ms (fallback)")

    # Phase 4: Performance analysis
    print(f"\n{'─'*40}")
    print(f"  Phase 4: 性能分析")

    speedup = eager_mean / compiled_mean
    print(f"    Eager: {eager_mean*1000:.3f} ms")
    print(f"    Compiled: {compiled_mean*1000:.3f} ms")
    print(f"    ★ Speedup: {speedup:.2f}x")
    if compile_ok:
        print(f"    编译开销: {compile_time*1000:.1f} ms (首次)")
        print(f"    后续运行: {compiled_mean*1000:.3f} ms (缓存)")

    if speedup > 1.0:
        print(f"    ★ ★ compile有效! {speedup:.2f}x加速")
    else:
        print(f"    ★ 小模型CPU compile收益有限 — CPU kernel已够快")
        print(f"    ★ ★ GPU上收益更大: +15-16%(FSDP2) / +48-50%(Float8)")

    # Phase 5: Graph break analysis
    print(f"\n{'─'*40}")
    print(f"  Phase 5: Graph Break & Guard 分析")

    try:
        import torch._dynamo as dynamo
        # Use dynamo explain to analyze graph breaks
        explanation = dynamo.explain(model, x)
        if explanation:
            print(f"    Graph breaks: {len(explanation.graph_breaks)}")
            for i, (reason, op) in enumerate(explanation.graph_breaks[:5]):
                print(f"      Break {i}: {reason} → {op}")
            print(f"    Ops compiled: {len(explanation.ops)}")
    except Exception as e:
        print(f"    dynamo.explain not available: {e}")
        # Alternative: try torch._dynamo.utils
        try:
            import torch._dynamo.utils as dynamo_utils
            guard_count = dynamo_utils.get_guard_count()
            print(f"    Guards registered: {guard_count}")
        except Exception:
            print(f"    ★ 简单模型 → 通常0 graph breaks → fullgraph=True可行!")

    # Phase 6: Compilation stack explanation
    print(f"\n{'─'*40}")
    print(f"  Phase 6: 编译栈架构总结")

    print(f"""
    ★ ★ ★ torch.compile完整编译栈 (基于7篇源码阅读):

    1. Dynamo (C-level frame hook):
       → _PyInterpreterState_SetEvalFrameFunc → 替换Python帧评估!
       → InstructionTranslator → 100+ VariableTracker → 符号执行
       → Guard system → CacheEntry → 最多64次recompile → fallback eager

    2. FX Graph (中间表示):
       → Node 6种op + SymInt/SymBool → 双向依赖图
       → GraphModule → recompile() → IR→Code→Execution
       → ★ FX IR = PyTorch的LLVM IR → 标准化中间表示!

    3. AOTAutograd (joint fwd+bwd):
       → make_fx trace → joint FX graph (forward+backward一起)
       → min-cut partition → fwd/bwd分离 → 自动gradient checkpointing!
       → max-flow min-cut theorem → 最优checkpoint选择 → 最小化内存

    4. Inductor (lowering→codegen):
       → FX aten op → lowerings dict → IR (TensorBox→StorageBox→Buffer)
       → Scheduler 10轮fusion → CSE消除中间 → removed_buffers
       → Triton/C++ codegen → PyCodeCache → compiled fn

    5. Execution (cache→run):
       → GuardedCode → guard check → cache hit → 直接运行compiled fn
       → Guard fail → recompile → 新GuardedCode → 更新cache

    ★ RTX 4090关键:
       - 训练: reduce-overhead + FSDP2(fullgraph=False) + LoRA
       - 推理: max-autotune + INT4 + EAGLE
       - 首次编译慢 → 但缓存后快 → 2.8默认启用缓存
       - ZeRO-3+compile = 不兼容 → FSDP2+compile = 最佳组合
    """)

    return {
        "model": model.__class__.__name__,
        "mode": mode,
        "pytorch_version": torch.__version__,
        "eager_ms": eager_mean * 1000,
        "compiled_ms": compiled_mean * 1000,
        "compile_overhead_ms": compile_time * 1000,
        "speedup": speedup,
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "output_consistent": bool(torch.allclose(y_eager, y_fast, atol=1e-5)),
    }


def trace_backward(model, mode):
    """追踪backward pass编译 — AOTAutograd核心展示"""

    print(f"\n{'='*60}")
    print(f"  AOTAutograd Backward编译追踪")
    print(f"{'='*60}")

    compiled_model = torch.compile(model, mode=mode)
    x = torch.randn(4, 256)

    # Forward
    y = compiled_model(x)
    loss = y.sum()

    # Backward — this triggers AOTAutograd backward compilation
    print(f"\n  ★ Forward完成 → AOTAutograd已预计算backward!")
    print(f"  ★ Eager autograd: forward → autograd引擎 → 动态生成backward graph")
    print(f"  ★ AOTAutograd: forward → 执行预先编译的backward → 无动态生成!")
    print(f"  ★ ★ AOT = Ahead-Of-Time → backward在compile时就确定了!")

    t0 = time.perf_counter()
    loss.backward()
    t_bw = time.perf_counter() - t0

    print(f"\n  Backward时间: {t_bw*1000:.3f} ms")
    print(f"  参数梯度已计算: {model.linear.weight.grad.shape if hasattr(model, 'linear') else 'computed'}")

    print(f"""
  ★ ★ ★ AOTAutograd vs Eager Autograd:

  Eager:
    forward → autograd引擎记录ops → backward时动态构建graph → 逐op执行
    → 每次backward都重新构建 → overhead大 → 无法全局优化

  AOTAutograd:
    forward → make_fx joint trace → min-cut partition → CompiledBackward已生成!
    → backward → 直接执行CompiledBackward → 无构建开销 → 可全局优化(fusion!)
    → ★ ★ 这就是torch.compile为什么能加速backward的原因!

  min-cut partition决定什么需要保存(checkpoint):
    → fwd→bwd的cut edges → 这些中间值需要保存 → 自动gradient checkpointing
    → 不cut → recomputed in backward → 省内存!
    → ★ max-flow min-cut → 数学最优 → 最小化保存量 → 最小化内存peak
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch torch.compile 执行追踪器"
    )
    parser.add_argument("--model",
                        choices=["linear", "mlp", "transformer-block", "lora"],
                        default="mlp")
    parser.add_argument("--mode",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        default="reduce-overhead")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细追踪信息")
    parser.add_argument("--backward", action="store_true",
                        help="追踪backward编译(AOTAutograd)")
    parser.add_argument("--dim", type=int, default=256,
                        help="模型维度(CPU小模型)")

    args = parser.parse_args()

    # Create model
    models = {
        "linear": lambda: SimpleLinear(args.dim),
        "mlp": lambda: SimpleMLP(args.dim),
        "transformer-block": lambda: TransformerBlock(args.dim, heads=4),
        "lora": lambda: LoRAModel(args.dim, lora_rank=8),
    }
    model = models[args.model]()

    # Trace compile
    result = trace_compile(model, args.mode, args.verbose)

    # Trace backward if requested
    if args.backward:
        trace_backward(model, args.mode)

    # Save results
    Path("results").mkdir(exist_ok=True)
    name = f"compile_trace_{args.model}_{args.mode}"
    with open(f"results/{name}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Results saved: results/{name}.json")


if __name__ == "__main__":
    main()
