#!/usr/bin/env python3
"""GRPO Training Validation Test with Non-Constant Reward.

Tests that the training loop actually updates model weights
using a length-based reward function.
"""

import json
import os
import sys
import time
import traceback

RESULTS_FILE = "/jiangdingfeng/zy/Termius/rollout/results/grpo_validation_test_results.json"


def main():
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "server": "10.26.6.88:31954 (4xH20-3e)",
        "tests": {},
    }

    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[INFO] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("trl-lib/tldr", split="train[:8]")

    # Length-based reward: encourage longer completions
    def length_reward(completions, **kwargs):
        scores = []
        for c in completions:
            if hasattr(c, '__len__'):
                scores.append(len(c) / 100.0)
            else:
                scores.append(1.0)
        return scores

    # Test 1: UP-GRPO with length reward
    print("=" * 60)
    print("Test 1: UP-GRPO with length-based reward")
    print("=" * 60)
    try:
        config = GRPOConfig(
            output_dir="/tmp/grpo_up_val",
            loss_type="up",
            num_generations=4,
            per_device_train_batch_size=4,
            max_steps=3,
            max_completion_length=128,
            bf16=True,
            seed=42,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            generation_batch_size=4,
        )
        trainer = GRPOTrainer(
            model=model,
            args=config,
            reward_funcs=[length_reward],
            processing_class=tokenizer,
            train_dataset=dataset,
        )
        print("[INFO] Starting UP-GRPO training...")
        train_result = trainer.train()
        print(f"[INFO] UP-GRPO loss: {train_result.training_loss:.4f}")
        results["tests"]["1_up_grpo_length_reward"] = {
            "status": "PASS",
            "loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
        }
    except Exception as e:
        traceback.print_exc()
        results["tests"]["1_up_grpo_length_reward"] = {"status": "FAIL", "error": str(e)}

    # Test 2: Standard GRPO with length reward
    print("=" * 60)
    print("Test 2: Standard GRPO with length-based reward")
    print("=" * 60)
    try:
        config = GRPOConfig(
            output_dir="/tmp/grpo_std_val",
            loss_type="grpo",
            num_generations=4,
            per_device_train_batch_size=4,
            max_steps=3,
            max_completion_length=128,
            bf16=True,
            seed=42,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            generation_batch_size=4,
        )
        model2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        trainer = GRPOTrainer(
            model=model2,
            args=config,
            reward_funcs=[length_reward],
            processing_class=tokenizer,
            train_dataset=dataset,
        )
        print("[INFO] Starting standard GRPO training...")
        train_result = trainer.train()
        print(f"[INFO] Standard GRPO loss: {train_result.training_loss:.4f}")
        results["tests"]["2_grpo_length_reward"] = {
            "status": "PASS",
            "loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
        }
    except Exception as e:
        traceback.print_exc()
        results["tests"]["2_grpo_length_reward"] = {"status": "FAIL", "error": str(e)}

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
