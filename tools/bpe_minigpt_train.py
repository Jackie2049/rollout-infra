#!/usr/bin/env python3
"""BPE Tokenizer + MiniGPT Training from Scratch
=================================================
End-to-end training with Byte Pair Encoding tokenizer:
1. Byte-level BPE tokenizer (GPT-2 style, from scratch)
2. LLaMA-like architecture (RMSNorm + SwiGLU + Pre-norm + Weight Tying)
3. Training with cosine schedule + warmup
4. Experiments: BPE vs char-level, vocab size, model scaling

Key educational insight: BPE solves the vocabulary trade-off:
- Character-level: small vocab but long sequences (poor context)
- Word-level: huge vocab, lots of <UNK> tokens
- BPE: learns subword units automatically, handles any input

Requires: PyTorch (no tiktoken/sentencepiece needed)
Recommended: RTX 4090 (24GB) for full experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import os
import re
from dataclasses import dataclass, field
from collections import Counter, defaultdict


# ============================================================
# 1. Byte Pair Encoding (BPE) Tokenizer — From Scratch
# ============================================================

class BPETokenizer:
    """Byte-level BPE tokenizer (GPT-2 style).

    Algorithm:
    1. Convert text to bytes (base vocab = 256)
    2. Count frequency of all adjacent byte pairs
    3. Merge the most frequent pair into a new token
    4. Repeat until desired vocab size
    5. At inference: apply merges greedily left-to-right

    Why byte-level?
    - Works on any text (no Unicode issues)
    - Base vocab = 256 (all possible bytes)
    - No <UNK> tokens ever needed
    - GPT-2, GPT-3, GPT-4 all use this approach
    """

    def __init__(self):
        self.merges = []          # List of (pair_a, pair_b) merges in order
        self.vocab = {}           # token_id -> token_bytes
        self.inverse_vocab = {}   # token_bytes -> token_id
        # Initialize base vocab with all 256 bytes
        for i in range(256):
            self.vocab[i] = bytes([i])
            self.inverse_vocab[bytes([i])] = i

    def train(self, text, vocab_size=512, verbose=False):
        """Train BPE on text to learn merges.

        Args:
            text: Training text
            vocab_size: Target vocabulary size (must be > 256)
            verbose: Print merge statistics
        """
        assert vocab_size > 256, "vocab_size must be > 256 (base byte vocab)"

        # Convert text to bytes
        text_bytes = text.encode('utf-8')

        # Split into words (GPT-2 style: split on whitespace boundaries)
        # We use a simple regex approach: split on spaces, keeping spaces attached
        words = re.findall(r'\S+|\s+', text)

        # Convert each word to byte sequences and count word frequencies
        word_freqs = Counter()
        for word in words:
            word_bytes = tuple(word.encode('utf-8'))
            word_freqs[word_bytes] += 1

        # Each word is a list of tokens (initially single bytes)
        # word_splits: word_bytes -> list of token_ids
        word_splits = {}
        for word_bytes in word_freqs:
            word_splits[word_bytes] = list(word_bytes)

        num_merges = vocab_size - 256
        for merge_idx in range(num_merges):
            # Count all adjacent pairs across all words (weighted by word frequency)
            pair_counts = Counter()
            for word_bytes, tokens in word_splits.items():
                freq = word_freqs[word_bytes]
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            new_token_id = 256 + merge_idx

            # Record the merge
            self.merges.append(best_pair)

            # Create new token's byte representation
            new_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_token_id] = new_bytes
            self.inverse_vocab[new_bytes] = new_token_id

            # Apply merge to all words
            for word_bytes in word_splits:
                tokens = word_splits[word_bytes]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        new_tokens.append(new_token_id)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_splits[word_bytes] = new_tokens

            if verbose and (merge_idx < 10 or merge_idx % 50 == 0):
                pair_str = self._decode_bytes(self.vocab[best_pair[0]]) + "+" + \
                           self._decode_bytes(self.vocab[best_pair[1]])
                merged_str = self._decode_bytes(new_bytes)
                print(f"  Merge {merge_idx+1}/{num_merges}: "
                      f"'{pair_str}' -> '{merged_str}' "
                      f"(freq={pair_counts[best_pair]})")

        if verbose:
            print(f"\nBPE Training complete: {len(self.merges)} merges, "
                  f"vocab_size={len(self.vocab)}")

    def encode(self, text):
        """Encode text to token IDs using learned merges."""
        # Split into words (same as training)
        words = re.findall(r'\S+|\s+', text)
        token_ids = []

        for word in words:
            # Start with byte-level tokens
            word_tokens = list(word.encode('utf-8'))

            # Apply merges in order (greedy)
            for pair_a, pair_b in self.merges:
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and \
                       word_tokens[i] == pair_a and word_tokens[i + 1] == pair_b:
                        # Find the merged token id
                        merged_bytes = self.vocab[pair_a] + self.vocab[pair_b]
                        if merged_bytes in self.inverse_vocab:
                            new_tokens.append(self.inverse_vocab[merged_bytes])
                        else:
                            new_tokens.append(word_tokens[i])
                            new_tokens.append(word_tokens[i + 1])
                            i += 1
                            continue
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens

            token_ids.extend(word_tokens)

        return token_ids

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        byte_list = []
        for tid in token_ids:
            if tid in self.vocab:
                byte_list.append(self.vocab[tid])
            else:
                byte_list.append(b'<UNK>')
        return b''.join(byte_list).decode('utf-8', errors='replace')

    def _decode_bytes(self, b):
        """Safely decode bytes for display."""
        return b.decode('utf-8', errors='replace')

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, path):
        """Save tokenizer to JSON."""
        data = {
            'merges': self.merges,
            'vocab': {str(k): list(v) for k, v in self.vocab.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path):
        """Load tokenizer from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.merges = [tuple(m) for m in data['merges']]
        self.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}


class CharTokenizer:
    """Simple character-level tokenizer for comparison."""

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def train(self, text):
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for i, c in enumerate(chars)}

    def encode(self, text):
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)

    @property
    def vocab_size(self):
        return len(self.char_to_id)


# ============================================================
# 2. Training Corpus
# ============================================================

# Extended English corpus (~10K characters) for meaningful BPE training
TRAINING_CORPUS = """Once upon a time, there was a little girl named Lucy. She lived in a small village near the edge of a great forest. Every morning, Lucy would walk to school with her friends. They loved to play games and tell stories about the magical creatures that lived in the woods.

The forest was full of tall trees and colorful flowers. Birds sang beautiful songs from the branches, and butterflies danced in the sunlight. A clear stream ran through the middle of the forest, its waters cool and fresh.

One day, Lucy found a small rabbit near the path. The rabbit was white with pink ears and bright blue eyes. She picked up the rabbit and carried it home carefully. Her mother said they could keep it as a pet, and Lucy named the rabbit Snowball.

Every day after school, Lucy would feed Snowball fresh carrots and lettuce. Snowball grew quickly and became very playful. He liked to hop around the garden and hide behind the rose bushes. Lucy and Snowball became the best of friends.

When spring came, the flowers bloomed again in the forest. Lucy and Snowball explored the forest together. They discovered a hidden clearing where wildflowers grew in every color of the rainbow. A family of deer grazed peacefully nearby.

Summer brought warm days and long evenings. Lucy and her friends would swim in the stream and build forts from fallen branches. Snowball would watch from the bank, twitching his pink nose with curiosity. Sometimes fireflies would appear at dusk, lighting up the forest like tiny stars.

Autumn transformed the forest into a canvas of gold and crimson. Lucy collected colorful leaves and pressed them in her favorite book. The animals prepared for winter, gathering nuts and seeds. Snowball grew a thicker coat of white fur to keep warm.

Winter covered everything in a blanket of white snow. Lucy built snowmen in the garden and Snowball hopped beside her, leaving tiny footprints in the fresh snow. They would sit by the fireplace in the evenings, Lucy reading stories aloud while Snowball dozed peacefully in her lap.

One cold winter night, Lucy heard a strange sound coming from the forest. She wrapped herself in her warmest coat and ventured outside. There, beneath the great oak tree, she found a golden flower glowing softly in the darkness. It was the most beautiful thing she had ever seen.

Lucy carefully dug up the golden flower and planted it in a pot by her window. Every night, the flower would glow with a warm golden light, filling her room with comfort and peace. She showed it to her friends at school, and they were amazed.

The golden flower never wilted. It sat on her windowsill, glowing every night through the coldest winter. The village people came from far and wide to see the magical flower. They told stories about the enchanted forest and the wonderful things that could be found within.

When spring returned, Lucy planted the golden flower in her garden. It grew taller and more beautiful than ever, spreading its warm light across the entire village. The flowers in the garden bloomed brighter, the vegetables grew larger, and everyone in the village was happy.

Lucy learned that the forest was full of wonderful surprises. She continued to explore with Snowball by her side, discovering new paths and hidden places. Every adventure brought something new and exciting, and Lucy cherished each moment.

Years passed, and Lucy grew up to become the village storyteller. She would sit beneath the great oak tree and tell the children tales of her adventures with Snowball, the golden flower, and the enchanted forest. The children listened with wide eyes, dreaming of their own adventures.

And so, in the little village near the great forest, the stories of Lucy and Snowball were told for generations. The golden flower still blooms in the garden, glowing softly each night, a reminder that magic can be found in the most unexpected places.

The village school was a small building with a red roof and white walls. Inside, there were wooden desks arranged in neat rows, and a large blackboard at the front of the room. The teacher, Mrs. Thompson, was kind and patient. She taught the children reading, writing, and arithmetic.

Every Friday, Mrs. Thompson would read a story to the class. The children loved story time the most. They would sit quietly, listening with rapt attention as Mrs. Thompson's voice brought the characters to life. Lucy always sat in the front row, her eyes bright with wonder.

One Friday, Mrs. Thompson announced a special project. Each student would write their own story and read it to the class. Lucy was excited. She decided to write about her adventures in the enchanted forest with Snowball. She worked on her story every evening, carefully choosing each word.

When the day came to present their stories, Lucy stood before the class with her heart beating fast. She began to read, and the words flowed like magic. The other children were enchanted by her tale of the golden flower and the mysterious forest. Mrs. Thompson smiled proudly.

The seasons continued their endless cycle in the village. Spring brought new life, summer brought warmth and play, autumn brought harvest and colors, and winter brought rest and wonder. Through it all, Lucy and Snowball remained the closest of friends, exploring the world together with curiosity and joy.

The forest changed with the years too. New trees grew where old ones had fallen. New paths appeared where none had been before. But the magic remained, hidden in the shadows and the sunlight, waiting to be discovered by those with curious hearts and open minds.

One summer evening, Lucy and Snowball found a new path they had never seen before. It led to a small pond surrounded by weeping willows. Fireflies danced above the water, and the moon reflected perfectly on the still surface. It was a place of perfect peace and beauty.

Lucy sat by the pond and wrote in her journal, describing everything she saw and felt. She wrote about the colors of the sunset, the sound of the frogs singing, the feel of the cool grass beneath her feet. She wanted to remember every detail of this magical place.

Snowball explored the edges of the pond, sniffing at the wildflowers and watching the fish swim lazily in the clear water. He seemed just as enchanted by the place as Lucy was. They stayed until the stars came out, content in each other's company and the beauty of the night.

The next day, Lucy returned to the pond with her sketchbook. She drew the weeping willows, the fireflies, and Snowball by the water's edge. Her drawing was simple but captured the feeling of the place perfectly. She added it to her collection of forest discoveries.

As the years went by, Lucy's collection of stories and drawings grew. She filled notebook after notebook with her adventures. The villagers often asked to see her drawings and hear her stories. Lucy was happy to share, for she believed that the best adventures were those shared with others.

And so the story of Lucy and Snowball continued, growing richer with each passing year. The enchanted forest held countless secrets, and Lucy was determined to discover them all. With Snowball by her side and a heart full of wonder, she knew that the greatest adventures were still ahead."""


# ============================================================
# 3. Model Architecture (LLaMA-like)
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config):
        super().__init__()
        assert config['d_model'] % config['n_head'] == 0
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        self.d_model = config['d_model']

        # QKV projection (fused for efficiency)
        self.qkv = nn.Linear(config['d_model'], 3 * config['d_model'], bias=False)
        self.proj = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config['block_size'], config['block_size']))
            .unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, T, C = x.shape

        # Fused QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class SwiGLU_MLP(nn.Module):
    """SwiGLU MLP (LLaMA-style). Uses 2/3 * 4d as hidden dim."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 / 3 * 4 * config['d_model'])
        self.w_gate = nn.Linear(config['d_model'], hidden_dim, bias=False)
        self.w_up = nn.Linear(config['d_model'], hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, config['d_model'], bias=False)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SwiGLU_MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Pre-norm + residual
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """GPT model with LLaMA-like architecture."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_emb = nn.Embedding(config['block_size'], config['d_model'])
        self.drop = nn.Dropout(config['dropout'])

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['n_layer'])
        ])

        self.ln_f = RMSNorm(config['d_model'])
        # Weight tying: output projection shares weights with token embedding
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.head.weight = self.token_emb.weight

        # Init weights
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"MiniGPT: {n_params/1e6:.2f}M parameters, "
              f"vocab={config['vocab_size']}, d={config['d_model']}, "
              f"layers={config['n_layer']}, heads={config['n_head']}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config['block_size']

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config['vocab_size']),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================
# 4. Data Loading
# ============================================================

def prepare_data(text, tokenizer, block_size, train_split=0.9):
    """Tokenize text and create train/val splits."""
    token_ids = tokenizer.encode(text)
    n = int(len(token_ids) * train_split)
    train_data = token_ids[:n]
    val_data = token_ids[n:]
    return train_data, val_data


def get_batch(data, block_size, batch_size, device):
    """Get a random batch of sequences."""
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        block_size = max(1, len(data) // 2 - 1)
        max_start = len(data) - block_size - 1

    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+1+block_size]) for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# 5. Training Loop
# ============================================================

def train_model(model, train_data, val_data, config, device, verbose=True):
    """Train the model with cosine schedule + warmup."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    block_size = min(config['block_size'], len(train_data) // 2 - 1)
    eval_iters = min(config['eval_iters'], max(5, len(val_data) // (block_size * config['batch_size'])))

    # Cosine schedule with warmup
    warmup_iters = min(config['warmup_iters'], config['max_iters'] // 4)

    def get_lr(it):
        if it < warmup_iters:
            return config['learning_rate'] * it / warmup_iters
        progress = (it - warmup_iters) / max(1, config['max_iters'] - warmup_iters)
        return config['learning_rate'] * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    results = {'train_loss': [], 'val_loss': [], 'lr': [], 'steps': []}
    best_val_loss = float('inf')
    start_time = time.time()

    for step in range(config['max_iters']):
        # Evaluate
        if step % config['eval_interval'] == 0 or step == config['max_iters'] - 1:
            model.eval()
            losses = []
            for _ in range(eval_iters):
                X, Y = get_batch(val_data, block_size, config['batch_size'], device)
                _, loss = model(X, Y)
                losses.append(loss.item())
            val_loss = sum(losses) / len(losses)
            results['val_loss'].append(val_loss)

            # Also get train loss
            X, Y = get_batch(train_data, block_size, config['batch_size'], device)
            _, train_loss = model(X, Y)
            results['train_loss'].append(train_loss.item())
            results['steps'].append(step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if verbose:
                elapsed = time.time() - start_time
                tok_per_sec = (step + 1) * config['batch_size'] * block_size / max(elapsed, 1e-6)
                print(f"Step {step:4d}/{config['max_iters']}: "
                      f"train_loss={train_loss.item():.4f}, "
                      f"val_loss={val_loss:.4f} (best={best_val_loss:.4f}), "
                      f"lr={get_lr(step):.2e}, "
                      f"tok/s={tok_per_sec:.0f}")

            model.train()

        # Training step
        X, Y = get_batch(train_data, block_size, config['batch_size'], device)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    elapsed = time.time() - start_time
    return results, elapsed, best_val_loss


# ============================================================
# 6. Generation Quality Evaluation
# ============================================================

def evaluate_generation(model, tokenizer, prompt, max_tokens=100,
                       temperature=0.8, top_k=50, device='cpu'):
    """Generate text and evaluate quality."""
    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    output = model.generate(idx, max_tokens, temperature=temperature, top_k=top_k)
    generated = tokenizer.decode(output[0].tolist())
    return generated


# ============================================================
# 7. Experiments
# ============================================================

def run_all_experiments(device='cuda'):
    """Run all experiments comparing BPE vs character-level tokenization."""

    print("=" * 70)
    print("BPE Tokenizer + MiniGPT Training Experiments")
    print(f"Device: {device}")
    print("=" * 70)

    results = {}

    # ----------------------------------------------------------
    # Experiment 1: BPE Tokenizer Analysis
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: BPE Tokenizer — Vocabulary Analysis")
    print("=" * 70)

    for vocab_size in [512, 1024, 2048]:
        print(f"\n--- BPE vocab_size={vocab_size} ---")
        bpe = BPETokenizer()
        bpe.train(TRAINING_CORPUS, vocab_size=vocab_size, verbose=(vocab_size == 512))

        encoded = bpe.encode(TRAINING_CORPUS)
        compression_ratio = len(TRAINING_CORPUS) / len(encoded)

        print(f"  Original chars: {len(TRAINING_CORPUS)}")
        print(f"  Token IDs: {len(encoded)}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Unique tokens used: {len(set(encoded))}/{bpe.vocab_size}")

        # Roundtrip test
        decoded = bpe.decode(encoded)
        assert decoded == TRAINING_CORPUS, f"Roundtrip failed for vocab_size={vocab_size}!"
        print(f"  Roundtrip: PASS")

        results[f'bpe_v{vocab_size}'] = {
            'vocab_size': bpe.vocab_size,
            'n_tokens': len(encoded),
            'compression': compression_ratio,
            'unique_tokens': len(set(encoded)),
        }

    # Char tokenizer baseline
    char_tok = CharTokenizer()
    char_tok.train(TRAINING_CORPUS)
    char_encoded = char_tok.encode(TRAINING_CORPUS)
    results['char'] = {
        'vocab_size': char_tok.vocab_size,
        'n_tokens': len(char_encoded),
        'compression': 1.0,
    }
    print(f"\n--- Char tokenizer baseline ---")
    print(f"  Vocab size: {char_tok.vocab_size}")
    print(f"  Tokens: {len(char_encoded)} (same as chars)")
    print(f"  Compression: 1.00x")

    # ----------------------------------------------------------
    # Experiment 2: BPE vs Char — Model Training Comparison
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: BPE vs Char Tokenizer — Training Comparison")
    print("=" * 70)

    base_config = {
        'block_size': 128,
        'n_layer': 4,
        'n_head': 4,
        'd_model': 128,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'max_iters': 2000,
        'eval_interval': 200,
        'eval_iters': 50,
        'warmup_iters': 100,
    }

    # Train with BPE (vocab=512)
    print("\n--- Training with BPE (vocab=512) ---")
    bpe512 = BPETokenizer()
    bpe512.train(TRAINING_CORPUS, vocab_size=512)
    train_data_bpe, val_data_bpe = prepare_data(TRAINING_CORPUS, bpe512, base_config['block_size'])
    print(f"  Train tokens: {len(train_data_bpe)}, Val tokens: {len(val_data_bpe)}")

    config_bpe = {**base_config, 'vocab_size': bpe512.vocab_size}
    model_bpe = MiniGPT(config_bpe).to(device)
    res_bpe, time_bpe, best_val_bpe = train_model(
        model_bpe, train_data_bpe, val_data_bpe, config_bpe, device
    )

    # Train with character-level
    print("\n--- Training with Char tokenizer ---")
    train_data_char, val_data_char = prepare_data(TRAINING_CORPUS, char_tok, base_config['block_size'])
    print(f"  Train tokens: {len(train_data_char)}, Val tokens: {len(val_data_char)}")

    config_char = {**base_config, 'vocab_size': char_tok.vocab_size}
    model_char = MiniGPT(config_char).to(device)
    res_char, time_char, best_val_char = train_model(
        model_char, train_data_char, val_data_char, config_char, device
    )

    print(f"\n--- Comparison ---")
    # Raw loss is NOT comparable across different vocab sizes!
    # Use bits-per-character (BPC) = loss / log(2) * (1 / compression_ratio)
    bpe_bpc = best_val_bpe / math.log(2) * (len(train_data_bpe) / len(TRAINING_CORPUS.split('\n', 1)[0]))
    bpe_bpc_total = best_val_bpe * len(train_data_bpe) / (len(TRAINING_CORPUS) * math.log(2)) * math.log(2)
    # Simpler: BPC = cross_entropy / ln(2) * (tokens/chars) = loss * tokens / (chars * ln(2))
    # But actually: BPC = sum of losses / total_characters
    # Per-token loss is already cross_entropy = -log(p), so BPC = loss * (n_tokens/n_chars) / ln(2)
    # Wait, this is simpler: bits/char = nats_per_token * tokens/chars / ln(2) * ln(2) = nats/char
    # BPE BPC = val_loss * (n_bpe_tokens / n_chars) — because each BPE token covers ~compression_ratio chars
    bpe_comp = len(TRAINING_CORPUS) / len(train_data_bpe)  # chars per token
    char_comp = 1.0  # 1 char per token
    bpe_bpc_val = best_val_bpe / math.log(2) / bpe_comp  # bits per character
    char_bpc_val = best_val_char / math.log(2) / char_comp
    bpe_ppl = math.exp(best_val_bpe)
    char_ppl = math.exp(best_val_char)

    print(f"  BPE (vocab=512): val_loss={best_val_bpe:.4f}, PPL={bpe_ppl:.2f}, "
          f"BPC={bpe_bpc_val:.4f}, comp={bpe_comp:.2f}x, time={time_bpe:.1f}s")
    print(f"  Char (vocab={char_tok.vocab_size}): val_loss={best_val_char:.4f}, PPL={char_ppl:.2f}, "
          f"BPC={char_bpc_val:.4f}, comp={char_comp:.2f}x, time={time_char:.1f}s")
    print(f"  Note: BPC (bits-per-character) enables fair comparison across tokenizers")

    results['exp2_comparison'] = {
        'bpe_val_loss': best_val_bpe,
        'bpe_time': time_bpe,
        'bpe_tokens': len(train_data_bpe),
        'char_val_loss': best_val_char,
        'char_time': time_char,
        'char_tokens': len(train_data_char),
    }

    # ----------------------------------------------------------
    # Experiment 3: BPE Vocab Size Effect
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: BPE Vocab Size Effect on Training")
    print("=" * 70)

    small_config = {
        'block_size': 128,
        'n_layer': 4,
        'n_head': 4,
        'd_model': 128,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'max_iters': 1500,
        'eval_interval': 150,
        'eval_iters': 30,
        'warmup_iters': 75,
    }

    for vocab_size in [512, 1024, 2048]:
        print(f"\n--- BPE vocab={vocab_size} ---")
        bpe = BPETokenizer()
        bpe.train(TRAINING_CORPUS, vocab_size=vocab_size)
        train_d, val_d = prepare_data(TRAINING_CORPUS, bpe, small_config['block_size'])

        cfg = {**small_config, 'vocab_size': bpe.vocab_size}
        model = MiniGPT(cfg).to(device)
        res, t, best_vl = train_model(model, train_d, val_d, cfg, device, verbose=False)
        print(f"  val_loss={best_vl:.4f}, time={t:.1f}s, n_tokens={len(train_d)}, "
              f"compression={len(TRAINING_CORPUS)/len(train_d):.2f}x")

        results[f'exp3_vocab{vocab_size}'] = {
            'val_loss': best_vl, 'time': t,
            'n_tokens': len(train_d),
            'compression': len(TRAINING_CORPUS) / len(train_d),
        }

    # ----------------------------------------------------------
    # Experiment 4: Model Scaling with BPE
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Model Scaling with BPE Tokenizer")
    print("=" * 70)

    bpe_best = BPETokenizer()
    bpe_best.train(TRAINING_CORPUS, vocab_size=512)
    train_d, val_d = prepare_data(TRAINING_CORPUS, bpe_best, 128)

    model_configs = [
        {'name': 'tiny',   'n_layer': 2, 'n_head': 2, 'd_model': 64},
        {'name': 'small',  'n_layer': 4, 'n_head': 4, 'd_model': 128},
        {'name': 'medium', 'n_layer': 6, 'n_head': 6, 'd_model': 192},
        {'name': 'large',  'n_layer': 8, 'n_head': 8, 'd_model': 256},
    ]

    for mc in model_configs:
        cfg = {
            'block_size': 128,
            'n_layer': mc['n_layer'],
            'n_head': mc['n_head'],
            'd_model': mc['d_model'],
            'dropout': 0.1,
            'vocab_size': bpe_best.vocab_size,
            'batch_size': 32,
            'learning_rate': 3e-4,
            'max_iters': 1500,
            'eval_interval': 150,
            'eval_iters': 30,
            'warmup_iters': 75,
        }

        print(f"\n--- {mc['name']} (d={mc['d_model']}, L={mc['n_layer']}, H={mc['n_head']}) ---")
        model = MiniGPT(cfg).to(device)
        res, t, best_vl = train_model(model, train_d, val_d, cfg, device, verbose=False)

        # Generate sample
        sample = evaluate_generation(model, bpe_best, "Once upon a time,",
                                     max_tokens=80, temperature=0.8, top_k=40, device=device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params/1e6:.2f}M, val_loss={best_vl:.4f}, time={t:.1f}s")
        print(f"  Sample: {sample[:120]}...")

        results[f'exp4_{mc["name"]}'] = {
            'params_M': n_params / 1e6,
            'val_loss': best_vl,
            'time': t,
            'sample': sample[:200],
        }

    # ----------------------------------------------------------
    # Experiment 5: Generation Quality Comparison
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Generation Quality — BPE vs Char")
    print("=" * 70)

    prompts = ["Once upon a time,", "The forest was", "Lucy and Snowball"]

    print("\n--- BPE Model Generation ---")
    for prompt in prompts:
        gen_bpe = evaluate_generation(model_bpe, bpe512, prompt,
                                      max_tokens=100, temperature=0.8, device=device)
        print(f"  Prompt: '{prompt}'")
        print(f"  Generated: {gen_bpe[:150]}")
        print()

    print("--- Char Model Generation ---")
    for prompt in prompts:
        gen_char = evaluate_generation(model_char, char_tok, prompt,
                                       max_tokens=100, temperature=0.8, device=device)
        print(f"  Prompt: '{prompt}'")
        print(f"  Generated: {gen_char[:150]}")
        print()

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: BPE Tokenizer Key Findings")
    print("=" * 70)

    print("""
1. BPE Compression:
   - Byte-level base vocab (256) + learned merges
   - Typical compression: 2-4x (English text)
   - More merges = more compression but larger vocab

2. BPE vs Char-level Training:
   - BPE: fewer tokens = shorter sequences = better context utilization
   - BPE: larger vocab = more parameters in embedding/output layers
   - Sweet spot: vocab=512-1024 for small corpora

3. Why BPE Wins for LLMs:
   - Handles any input (no <UNK>)
   - Learns meaningful subwords ("ing", "tion", "ed")
   - Balances sequence length vs vocabulary size
   - GPT-2/3/4 use byte-level BPE with ~50K merges

4. Real-world Scale:
   - GPT-2: 50,257 tokens, ~4 chars/token
   - GPT-4 (cl100k): 100,256 tokens, ~4 chars/token
   - LLaMA (SentencePiece): 32,000 tokens
   - Compression ratio stays ~4x for English
    """)

    # Save results
    with open('bpe_minigpt_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to bpe_minigpt_results.json")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = run_all_experiments(device=device)
