#!/usr/bin/env python3
"""Minimal GRPO Training Validation Test on H20-3e

Runs a short GRPO training loop (3 steps) with TRL UP-GRPO loss_type
to validate our patches work in actual training runtime.

Uses Qwen2.5-0.5B-Instruct for fast turnaround on H20-3e.

Run on partner server:
  cd /jiangdingfeng/zy/Termius/rollout
  source /jiangdingfeng/miniconda3/etc/profile.d/conda.sh
  conda activate env-rollout
  python scripts/grpo_training_runtime_test_h20.py
"""

import json
import os
import sys
import time
import traceback

RESULTS_FILE = "/jiangdingfeng/zy/Termius/rollout/results/grpo_training_runtime_test_h20_results.json"


def make_reward_func():
    """Create a simple reward function for testing."""
    def reward_func(completions, **kwargs):
        """Return a constant positive reward for each completion."""
        import torch
        return [1.0] * len(completions)
    return reward_func


def main():
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "server": "10.26.6.88:31954 (4×H20-3e)",
        "tests": {},
    }

    # ── Test 1: UP-GRPO short training loop ──
    print("=" * 60)
    print("Test 1: UP-GRPO short training loop (loss_type='up')")
    print("=" * 60)

    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset

        # Load model and tokenizer
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"[INFO] Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Load a small dataset
        print("[INFO] Loading dataset: trl-lib/tldr")
        dataset = load_dataset("trl-lib/tldr", split="train[:8]")

        # GRPOConfig with UP-GRPO
        config = GRPOConfig(
            output_dir="/jiangdingfeng/zy/Termius/rollout/results/grpo_up_test",
            loss_type="up",
            num_generations=4,
            per_device_train_batch_size=4,
            max_steps=3,
            max_completion_length=64,
            bf16=True,
            seed=42,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            generation_batch_size=4,
        )
        print(f"[INFO] GRPOConfig created: loss_type={config.loss_type}, num_generations={config.num_generations}")

        # Create reward function
        reward_func = make_reward_func()

        # Create trainer
        trainer = GRPOTrainer(
            model=model,
            args=config,
            reward_funcs=[reward_func],
            processing_class=tokenizer,
            train_dataset=dataset,
        )
        print("[INFO] GRPOTrainer created successfully")

        # Run training
        print("[INFO] Starting training (3 steps)...")
        train_result = trainer.train()
        print(f"[INFO] Training completed!")
        print(f"[INFO] Train loss: {train_result.training_loss:.4f}")
        print(f"[INFO] Train metrics: {train_result.metrics}")

        results["tests"]["1_up_grpo_training"] = {
            "status": "PASS",
            "loss": train_result.training_loss,
            "steps": train_result.metrics.get("total_flos", 0),
            "train_runtime": train_result.metrics.get("train_runtime", 0),
        }

    except Exception as e:
        traceback.print_exc()
        results["tests"]["1_up_grpo_training"] = {"status": "FAIL", "error": str(e)}

    # ── Test 2: Standard GRPO training loop (backward compat) ──
    print("=" * 60)
    print("Test 2: Standard GRPO training loop (loss_type='grpo')")
    print("=" * 60)

    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("trl-lib/tldr", split="train[:8]")

        config = GRPOConfig(
            output_dir="/jiangdingfeng/zy/Termius/rollout/results/grpo_standard_test",
            loss_type="grpo",
            num_generations=4,
            per_device_train_batch_size=4,
            max_steps=3,
            max_completion_length=64,
            bf16=True,
            seed=42,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            generation_batch_size=4,
        )

        reward_func = make_reward_func()

        trainer = GRPOTrainer(
            model=model,
            args=config,
            reward_funcs=[reward_func],
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        print("[INFO] Starting standard GRPO training (3 steps)...")
        train_result = trainer.train()
        print(f"[INFO] Standard GRPO training completed!")
        print(f"[INFO] Train loss: {train_result.training_loss:.4f}")

        results["tests"]["2_grpo_training"] = {
            "status": "PASS",
            "loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
        }

    except Exception as e:
        traceback.print_exc()
        results["tests"]["2_grpo_training"] = {"status": "FAIL", "error": str(e)}

    # ── Summary ──
    passed = sum(1 for t in results["tests"].values() if t["status"] == "PASS")
    failed = sum(1 for t in results["tests"].values() if t["status"] == "FAIL")
    results["summary"] = f"{passed}/{passed+failed} passed, {failed} failed"
    print(f"\n=== {results['summary']} ===")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
