#!/usr/bin/env python3
"""BPE MiniGPT Training with Large Synthetic Corpus
====================================================
Uses generated children's stories (~300 stories, ~150K chars) for training.
Compares BPE vs char tokenizer with enough data for meaningful results.

Usage:
  1. python tools/generate_corpus.py  # generates corpus
  2. python tools/bpe_minigpt_large.py  # trains models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from bpe_minigpt_train import (
    BPETokenizer, CharTokenizer, MiniGPT, prepare_data,
    get_batch, train_model, evaluate_generation,
)
from generate_corpus import generate_corpus
import math
import json
import time
import torch


def run_experiments(device='cuda'):
    print("=" * 70)
    print("BPE MiniGPT — Large Corpus Experiments")
    print(f"Device: {device}")
    print("=" * 70)

    # Generate corpus
    corpus = generate_corpus(n_stories=300)
    print(f"\nCorpus: {len(corpus)} chars, {corpus.count(chr(10)+chr(10))} stories")

    results = {}

    # ----------------------------------------------------------
    # Experiment 1: Tokenizer Comparison
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: BPE vs Char — Large Corpus")
    print("=" * 70)

    # BPE tokenizer (vocab=512)
    bpe = BPETokenizer()
    bpe.train(corpus, vocab_size=512, verbose=False)
    bpe_encoded = bpe.encode(corpus)
    print(f"\nBPE (vocab=512): {len(bpe_encoded)} tokens, "
          f"compression={len(corpus)/len(bpe_encoded):.2f}x")

    # Char tokenizer
    char_tok = CharTokenizer()
    char_tok.train(corpus)
    char_encoded = char_tok.encode(corpus)
    print(f"Char (vocab={char_tok.vocab_size}): {len(char_encoded)} tokens, "
          f"compression=1.00x")

    # Train both
    base_config = {
        'block_size': 256,
        'n_layer': 6,
        'n_head': 6,
        'd_model': 192,
        'dropout': 0.1,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'max_iters': 3000,
        'eval_interval': 300,
        'eval_iters': 50,
        'warmup_iters': 150,
    }

    # BPE training
    print("\n--- BPE Training ---")
    train_bpe, val_bpe = prepare_data(corpus, bpe, base_config['block_size'])
    print(f"Train: {len(train_bpe)}, Val: {len(val_bpe)} tokens")
    cfg_bpe = {**base_config, 'vocab_size': bpe.vocab_size}
    model_bpe = MiniGPT(cfg_bpe).to(device)
    res_bpe, time_bpe, best_bpe = train_model(
        model_bpe, train_bpe, val_bpe, cfg_bpe, device
    )

    # Char training
    print("\n--- Char Training ---")
    train_char, val_char = prepare_data(corpus, char_tok, base_config['block_size'])
    print(f"Train: {len(train_char)}, Val: {len(val_char)} tokens")
    cfg_char = {**base_config, 'vocab_size': char_tok.vocab_size}
    model_char = MiniGPT(cfg_char).to(device)
    res_char, time_char, best_char = train_model(
        model_char, train_char, val_char, cfg_char, device
    )

    # Comparison
    bpe_comp = len(corpus) / len(train_bpe)
    char_comp = 1.0
    bpe_bpc = best_bpe / math.log(2) / bpe_comp
    char_bpc = best_char / math.log(2) / char_comp

    print(f"\n--- Comparison ---")
    print(f"  BPE: val_loss={best_bpe:.4f}, PPL={math.exp(best_bpe):.1f}, "
          f"BPC={bpe_bpc:.4f}, time={time_bpe:.1f}s")
    print(f"  Char: val_loss={best_char:.4f}, PPL={math.exp(best_char):.1f}, "
          f"BPC={char_bpc:.4f}, time={time_char:.1f}s")
    print(f"  BPE BPC advantage: {(char_bpc-bpe_bpc)/char_bpc*100:.1f}%")

    results['comparison'] = {
        'bpe': {'loss': best_bpe, 'bpc': bpe_bpc, 'ppl': math.exp(best_bpe),
                'time': time_bpe, 'tokens': len(train_bpe), 'compression': bpe_comp},
        'char': {'loss': best_char, 'bpc': char_bpc, 'ppl': math.exp(best_char),
                 'time': time_char, 'tokens': len(train_char), 'compression': char_comp},
        'corpus_chars': len(corpus),
    }

    # ----------------------------------------------------------
    # Experiment 2: Model Scaling with BPE
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Model Scaling with BPE (vocab=512)")
    print("=" * 70)

    bpe_best = BPETokenizer()
    bpe_best.train(corpus, vocab_size=512)
    train_d, val_d = prepare_data(corpus, bpe_best, 256)

    scaling_configs = [
        {'name': 'tiny',   'n_layer': 2, 'n_head': 2, 'd_model': 64,  'max_iters': 2000},
        {'name': 'small',  'n_layer': 4, 'n_head': 4, 'd_model': 128, 'max_iters': 2500},
        {'name': 'medium', 'n_layer': 6, 'n_head': 6, 'd_model': 192, 'max_iters': 3000},
        {'name': 'large',  'n_layer': 8, 'n_head': 8, 'd_model': 256, 'max_iters': 3000},
        {'name': 'xlarge', 'n_layer': 10, 'n_head': 8, 'd_model': 320, 'max_iters': 3000},
    ]

    for mc in scaling_configs:
        cfg = {
            'block_size': 256,
            'n_layer': mc['n_layer'],
            'n_head': mc['n_head'],
            'd_model': mc['d_model'],
            'dropout': 0.1,
            'vocab_size': bpe_best.vocab_size,
            'batch_size': 64,
            'learning_rate': 3e-4,
            'max_iters': mc['max_iters'],
            'eval_interval': 250,
            'eval_iters': 50,
            'warmup_iters': mc['max_iters'] // 10,
        }

        print(f"\n--- {mc['name']} (d={mc['d_model']}, L={mc['n_layer']}) ---")
        model = MiniGPT(cfg).to(device)
        res, t, best_vl = train_model(model, train_d, val_d, cfg, device, verbose=False)

        # Generate sample
        prompts = ["Once upon a time,", "The little girl", "In the forest"]
        samples = []
        for prompt in prompts:
            gen = evaluate_generation(model, bpe_best, prompt,
                                     max_tokens=120, temperature=0.8, top_k=40, device=device)
            samples.append(gen)

        n_params = sum(p.numel() for p in model.parameters())
        bpc = best_vl / math.log(2) / (len(corpus) / len(train_d))
        print(f"  Params: {n_params/1e6:.2f}M, val_loss={best_vl:.4f}, "
              f"BPC={bpc:.4f}, PPL={math.exp(best_vl):.1f}, time={t:.1f}s")
        print(f"  Sample: {samples[0][:150]}...")

        results[f'scale_{mc["name"]}'] = {
            'params_M': n_params / 1e6,
            'val_loss': best_vl,
            'bpc': bpc,
            'ppl': math.exp(best_vl),
            'time': t,
            'sample': samples[0][:200],
        }

    # ----------------------------------------------------------
    # Experiment 3: BPE Vocab Size with Large Corpus
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: BPE Vocab Size Effect (Large Corpus)")
    print("=" * 70)

    for vocab_size in [512, 1024, 2048]:
        bpe_v = BPETokenizer()
        bpe_v.train(corpus, vocab_size=vocab_size)
        train_v, val_v = prepare_data(corpus, bpe_v, 256)
        comp = len(corpus) / len(train_v)

        cfg = {
            'block_size': 256, 'n_layer': 6, 'n_head': 6, 'd_model': 192,
            'dropout': 0.1, 'vocab_size': bpe_v.vocab_size,
            'batch_size': 64, 'learning_rate': 3e-4,
            'max_iters': 2500, 'eval_interval': 250, 'eval_iters': 50,
            'warmup_iters': 250,
        }

        model = MiniGPT(cfg).to(device)
        res, t, best_vl = train_model(model, train_v, val_v, cfg, device, verbose=False)
        bpc = best_vl / math.log(2) / comp

        print(f"\n  Vocab={vocab_size}: val_loss={best_vl:.4f}, BPC={bpc:.4f}, "
              f"PPL={math.exp(best_vl):.1f}, comp={comp:.2f}x, time={t:.1f}s")

        results[f'vocab_{vocab_size}'] = {
            'val_loss': best_vl, 'bpc': bpc, 'ppl': math.exp(best_vl),
            'compression': comp, 'time': t, 'n_tokens': len(train_v),
        }

    # ----------------------------------------------------------
    # Experiment 4: Generation Quality Showcase
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Generation Quality (Best Model)")
    print("=" * 70)

    prompts = [
        "Once upon a time,",
        "The little girl walked",
        "In the magical forest,",
        "One sunny morning,",
        "The rabbit and the child",
    ]

    for temp in [0.5, 0.8, 1.2]:
        print(f"\n--- Temperature={temp} ---")
        for prompt in prompts[:3]:
            gen = evaluate_generation(model_bpe, bpe, prompt,
                                     max_tokens=150, temperature=temp, top_k=50, device=device)
            print(f"  [{prompt}] → {gen[:200]}")

    # Save results
    output_file = 'bpe_large_corpus_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")
    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    run_experiments(device=device)
