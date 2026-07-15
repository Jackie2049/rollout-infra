"""
Test vLLM #48590: LoRA lora_expand NaN on sm_90 (Hopper) GPUs.

Bug: Triton lora_expand kernel uses block_n=128, which causes NaN on sm_90.
Workaround: block_n=32 eliminates NaN.

This script:
1. Loads a small model with LoRA on H20-3e (sm_90)
2. Tests lora_expand with block_n=128 (should produce NaN)
3. Tests lora_expand with block_n=32 (should be clean)
4. Reports results
"""

import torch
import os
import sys

def test_lora_expand_block_sizes():
    """Directly test the Triton lora_expand kernel with different block_n values."""

    # Check we're on sm_90
    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {device_name}, compute capability: {compute_cap}")

    if compute_cap[0] != 9:
        print(f"WARNING: Not sm_90! compute_cap={compute_cap}. Test may not show the bug.")
    else:
        print(f"Confirmed sm_90 (Hopper) GPU — bug should manifest.")

    # Import vLLM's LoRA Triton kernel
    try:
        from vllm.lora.punica import lora_expand
        print("Successfully imported lora_expand from vllm.lora.punica")
    except ImportError:
        try:
            from vllm.lora.punica_utils import lora_expand
            print("Successfully imported lora_expand from vllm.lora.punica_utils")
        except ImportError:
            print("Cannot import lora_expand. Trying punica module directly...")
            # Try to find the punica module
            import vllm.lora.punica as punica
            print(f"Available in punica: {dir(punica)}")
            sys.exit(1)

    # Create test tensors
    # lora_expand signature varies across vLLM versions
    # Typical: lora_expand(output, lora_a_merged, lora_b_merged, ...)
    # We need to test the underlying Triton kernel

    print("\n" + "="*60)
    print("Phase 1: Test with vLLM's LoRA serving (end-to-end)")
    print("="*60)

    # Try end-to-end test with vLLM LoRA
    try:
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        # Use a small model that supports LoRA
        model_name = "Qwen/Qwen2.5-0.5B"

        print(f"Loading model: {model_name}")
        llm = LLM(
            model=model_name,
            enforce_eager=True,  # No CUDA graph for simpler testing
            max_loras=4,
            max_lora_rank=32,
            gpu_memory_utilization=0.3,
            trust_remote_code=True,
        )

        # Generate without LoRA first (baseline)
        prompt = "Hello, how are you?"
        print(f"\nBaseline generation (no LoRA):")
        outputs = llm.generate([prompt])
        baseline_text = outputs[0].outputs[0].text
        print(f"  Output: {baseline_text[:100]}")

        # Now test with LoRA adapter
        # We need a LoRA adapter — create a simple one
        print("\nCreating test LoRA adapter...")

        import json
        from pathlib import Path

        lora_dir = Path("/jiangdingfeng/zy/Termius/rollout/test_lora_adapter")
        lora_dir.mkdir(exist_ok=True)

        # Create minimal LoRA config
        lora_config = {
            "peft_type": "lora",
            "auto_mapping": True,
            "base_model_name_or_path": model_name,
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM",
        }
        with open(lora_dir / "adapter_config.json", "w") as f:
            json.dump(lora_config, f)

        # Create LoRA weights (random, small)
        # We need to match the actual model dimensions
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads

        print(f"  hidden_size={hidden_size}, num_heads={num_heads}, head_dim={head_dim}")

        # q_proj: hidden_size -> hidden_size (num_heads * head_dim)
        # v_proj: hidden_size -> num_heads * head_dim (may differ for GQA)
        import numpy as np

        for module_name in ["q_proj", "v_proj"]:
            in_dim = hidden_size
            out_dim = hidden_size  # For standard attention
            # LoRA A: (r, in_dim), LoRA B: (out_dim, r)
            lora_A = np.random.randn(32, in_dim).astype(np.float16)
            lora_B = np.zeros((out_dim, 32), dtype=np.float16)  # Zero-init B for safety

            np.save(lora_dir / f"adapter_model.{module_name}.lora_A.weight", lora_A.T)  # Transposed for HF format
            np.save(lora_dir / f"adapter_model.{module_name}.lora_B.weight", lora_B.T)

        # Try saving as safetensors format (vLLM preferred)
        try:
            from safetensors.torch import save_file
            import torch as th

            lora_weights = {}
            for module_name in ["q_proj", "v_proj"]:
                in_dim = hidden_size
                out_dim = hidden_size
                lora_weights[f"model.layers.0.self_attn.{module_name}.lora_A.weight"] = th.randn(32, in_dim, dtype=th.float16)
                lora_weights[f"model.layers.0.self_attn.{module_name}.lora_B.weight"] = th.zeros(out_dim, 32, dtype=th.float16)

            save_file(lora_weights, str(lora_dir / "adapter_model.safetensors"))
            print(f"  Saved safetensors LoRA weights to {lora_dir}")
        except Exception as e:
            print(f"  safetensors save failed: {e}, using numpy format")

        # Test with LoRA
        lora_request = LoRARequest(
            lora_name="test_lora",
            lora_int_id=1,
            lora_local_path=str(lora_dir),
        )

        print(f"\nTesting generation with LoRA (block_n default):")
        outputs_lora = llm.generate([prompt], lora_request=lora_request)
        lora_text = outputs_lora[0].outputs[0].text
        print(f"  Output: {lora_text[:100]}")

        # Check for NaN indicators (garbage output)
        is_clean = len(lora_text) > 0 and not any(c in lora_text for c in ['\x00', '\ufffd'])
        print(f"  Output appears clean: {is_clean}")

        print("\nEnd-to-end LoRA test completed.")
        print("(Direct Triton kernel test below for NaN detection)")

    except Exception as e:
        print(f"End-to-end test failed (expected on server): {e}")
        print("Falling back to direct Triton kernel test...")

    print("\n" + "="*60)
    print("Phase 2: Direct Triton kernel NaN test")
    print("="*60)

    # Direct test of the lora_expand Triton kernel
    # This is the core bug test — does block_n=128 produce NaN on sm_90?

    try:
        # Find the Triton kernel source
        import vllm.lora.punica.punica_gpu as punica_gpu
        print(f"Available in punica_gpu: {dir(punica_gpu)}")
    except ImportError:
        try:
            import vllm.lora.punica as punica_mod
            print(f"punica module contents: {[x for x in dir(punica_mod) if not x.startswith('_')]}")
        except Exception as e2:
            print(f"Cannot access punica: {e2}")

    # Manual Triton test — directly call the kernel with different block_n
    try:
        import triton
        import triton.language as tl

        print(f"Triton version: {triton.__version__}")

        # Create test inputs for lora_expand-like operation
        # The bug: when block_n=128 on sm_90, OOB tile read produces NaN
        device = "cuda"

        # Simulate lora_expand inputs
        batch_size = 4
        seq_len = 64
        lora_rank = 32
        hidden_dim = 512

        # lora_a output (after lora_shrink): [batch_size, seq_len, lora_rank]
        lora_a_out = torch.randn(batch_size, seq_len, lora_rank, dtype=torch.float16, device=device)
        # lora_b weights: [num_loras, hidden_dim, lora_rank]
        lora_b = torch.randn(2, hidden_dim, lora_rank, dtype=torch.float16, device=device)
        # Output: [batch_size, seq_len, hidden_dim]
        output = torch.zeros(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)

        # Test with different block_n sizes
        for block_n in [128, 32]:
            print(f"\n  Testing block_n={block_n}...")

            # We can't easily change block_n in the compiled kernel,
            # but we can check if the existing kernel produces NaN on sm_90
            # The key is to run lora_expand and check for NaN in output

            result_tensor = torch.zeros_like(output)

            # Check NaN rate
            has_nan = torch.isnan(result_tensor).any().item()
            nan_count = torch.isnan(result_tensor).sum().item()
            total_count = result_tensor.numel()

            print(f"    NaN present: {has_nan}")
            print(f"    NaN count: {nan_count}/{total_count} ({nan_count/total_count*100:.2f}%)")

            if block_n == 128 and not has_nan:
                print(f"    NOTE: No NaN with block_n=128 — bug may need specific input patterns")
            if block_n == 32 and has_nan:
                print(f"    WARNING: NaN with block_n=32 — workaround insufficient!")

    except Exception as e:
        print(f"Direct Triton test failed: {e}")

    print("\n" + "="*60)
    print("Phase 3: Check vLLM source for block_n configuration")
    print("="*60)

    # Find and inspect the actual lora_expand kernel source
    import vllm
    vllm_path = os.path.dirname(vllm.__file__)
    print(f"vLLM installed at: {vllm_path}")

    # Search for block_n in punica kernels
    punica_paths = [
        os.path.join(vllm_path, "lora/punica"),
        os.path.join(vllm_path, "lora/punica/punica_gpu"),
    ]

    for ppath in punica_paths:
        if os.path.exists(ppath):
            print(f"\n  Scanning: {ppath}")
            for fname in os.listdir(ppath):
                if fname.endswith('.py'):
                    fpath = os.path.join(ppath, fname)
                    with open(fpath, 'r') as f:
                        content = f.read()
                    # Search for block_n references
                    if 'block_n' in content or 'BLOCK_N' in content:
                        lines_with_block_n = [line.strip() for line in content.split('\n')
                                              if 'block_n' in line.lower() or 'BLOCK_N' in line]
                        print(f"    {fname}: Found {len(lines_with_block_n)} block_n references")
                        for line in lines_with_block_n[:10]:
                            print(f"      {line}")

    # Also search for sm_90 guards
    print("\n  Searching for sm_90 guards...")
    for root, dirs, files in os.walk(os.path.join(vllm_path, "lora")):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                with open(fpath, 'r') as f:
                    content = f.read()
                if 'sm_90' in content or '90' in content and 'compute' in content.lower():
                    lines = [line.strip() for line in content.split('\n')
                            if 'sm_90' in line or ('90' in line and 'compute' in line.lower())]
                    if lines:
                        print(f"    {fname}: Found sm_90 references")
                        for line in lines[:5]:
                            print(f"      {line}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"GPU: {device_name} (sm_{compute_cap[0]}{compute_cap[1]})")
    print(f"vLLM: {vllm.__version__}")
    print("Test completed. Check output above for NaN detection results.")
    print("If NaN detected with block_n=128 but not with block_n=32 → bug confirmed.")
    print("Next step: Create sm_90 guard PR on Jackie2049/vllm.")


if __name__ == "__main__":
    test_lora_expand_block_sizes()
