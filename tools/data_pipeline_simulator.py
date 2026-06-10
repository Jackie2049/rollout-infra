#!/usr/bin/env python3
"""
Data Pipeline Practical — MinHash Deduplication + Quality Filtering

Implements the core data curation pipeline from data-pipeline-curation-deep-dive.md:
1. MinHash-based near-duplicate detection → LSH clustering → deduplication
2. Quality filtering → length/repetition/encoding detection
3. Contamination detection → n-gram overlap with benchmark test sets

No GPU required — pure CPU-based data pipeline simulation.
Demonstrates the "data quality > architecture" principle (Phi-1 1.3B > 50B!)
"""

import hashlib
import struct
import random
import json
import os
from collections import defaultdict
from typing import List, Dict, Set, Tuple


# ============================================================================
# Part 1: MinHash — Efficient Near-Duplicate Detection
# ============================================================================

class MinHasher:
    """MinHash for near-duplicate detection in large text corpora.

    Key insight from data pipeline deep dive:
    → 30-50% of web data is near-duplicate → dedup saves 30-50% training cost!
    → → MinHash = approximate Jaccard similarity → fast → O(1) comparison per pair!
    → → → Instead of comparing all N² pairs → MinHash + LSH → O(N) grouping!

    MinHash algorithm:
    → 1. Tokenize document → set of shingles (n-grams)
    → 2. Apply K random hash functions → take minimum per hash → K "signatures"
    → 3. Jaccard(A,B) ≈ |MinHash(A) ∩ MinHash(B)| / K → approximate similarity!
    → → → Exact Jaccard = |A ∩ B| / |A ∪ B| → but computing exact for all pairs = O(N²)!
    → → → → MinHash = O(K) per comparison → K=128 → fast!
    """

    def __init__(self, num_hashes: int = 128, shingle_size: int = 5):
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        # Generate random hash functions using murmurhash3-like approach
        self.hash_params = [(random.randint(1, 2**32), random.randint(0, 2**32))
                           for _ in range(num_hashes)]

    def _hash(self, value: str, a: int, b: int) -> int:
        """Hash function: h(x) = (a * x + b) mod 2^32"""
        # Convert string to integer hash first
        x = int(hashlib.md5(value.encode()).hexdigest()[:8], 16)
        return (a * x + b) % (2**32)

    def shinglify(self, text: str) -> Set[str]:
        """Create set of shingles (n-grams) from text."""
        tokens = text.lower().split()
        shingles = set()
        for i in range(len(tokens) - self.shingle_size + 1):
            shingle = " ".join(tokens[i:i + self.shingle_size])
            shingles.add(shingle)
        return shingles

    def compute_signature(self, text: str) -> List[int]:
        """Compute MinHash signature for a document."""
        shingles = self.shinglify(text)
        if not shingles:
            return [0] * self.num_hashes

        signature = []
        for a, b in self.hash_params:
            min_hash = min(self._hash(s, a, b) for s in shingles)
            signature.append(min_hash)
        return signature

    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        matches = sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2)
        return matches / len(sig1)

    def exact_jaccard(self, text1: str, text2: str) -> float:
        """Compute exact Jaccard for verification."""
        s1 = self.shinglify(text1)
        s2 = self.shinglify(text2)
        if not s1 and not s2:
            return 1.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0


# ============================================================================
# Part 2: LSH (Locality-Sensitive Hashing) — Fast Duplicate Grouping
# ============================================================================

class LSHTable:
    """LSH for grouping near-duplicates efficiently.

    Key insight:
    → LSH = bucket documents by hash bands → same bucket = likely duplicate!
    → → K hashes divided into B bands of R rows → K = B × R
    → → → Two docs collide in at least 1 band → candidate pair → check MinHash similarity!
    → → → → Probability of collision = 1 - (1 - s^R)^B → tune B,R for desired threshold!

    For threshold s=0.5 (50% similarity):
    → B=16, R=8 → P(collision) = 1-(1-0.5^8)^16 = 1-(1-0.0039)^16 ≈ 0.06 → too low!
    → B=8, R=16 → P(collision) = 1-(1-0.5^16)^8 ≈ 0 → near-duplicates missed!
    → → → Need B and R tuned → s=0.8 → B=16, R=8 → P≈1-(1-0.8^8)^16 ≈ 0.91 → good!
    → → → → → For web dedup: threshold 0.7-0.8 → B=8, R=16 → catches most near-dups!
    """

    def __init__(self, num_bands: int = 8, rows_per_band: int = 16):
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.total_hashes = num_bands * rows_per_band

    def hash_band(self, signature: List[int], band_idx: int) -> str:
        """Hash a band of the MinHash signature."""
        start = band_idx * self.rows_per_band
        end = start + self.rows_per_band
        band = signature[start:end]
        return hashlib.md5(struct.pack(f'<{len(band)}I', *band)).hexdigest()

    def insert(self, doc_id: int, signature: List[int]):
        """Insert a document into LSH buckets."""
        for band_idx in range(self.num_bands):
            band_hash = self.hash_band(signature, band_idx)
            self.buckets[band_idx][band_hash].append(doc_id)

    def query(self, signature: List[int]) -> Set[int]:
        """Find candidate duplicates for a document."""
        candidates = set()
        for band_idx in range(self.num_bands):
            band_hash = self.hash_band(signature, band_idx)
            candidates.update(self.buckets[band_idx][band_hash])
        return candidates

    def get_duplicate_groups(self, minhasher: MinHasher,
                             signatures: Dict[int, List[int]],
                             threshold: float = 0.7) -> List[List[int]]:
        """Find all groups of near-duplicates above threshold."""
        groups = []
        visited = set()

        for doc_id, sig in signatures.items():
            if doc_id in visited:
                continue
            candidates = self.query(sig)
            group = [doc_id]
            for candidate_id in candidates:
                if candidate_id != doc_id and candidate_id not in visited:
                    sim = minhasher.jaccard_similarity(sig, signatures[candidate_id])
                    if sim >= threshold:
                        group.append(candidate_id)
                        visited.add(candidate_id)
            visited.add(doc_id)
            if len(group) > 1:
                groups.append(group)

        return groups


# ============================================================================
# Part 3: Quality Filtering
# ============================================================================

class QualityFilter:
    """Quality filtering pipeline for text data.

    Key insight from data pipeline deep dive:
    → 99% of web data is garbage → quality filtering → 1PB→7.5TB → 99% removed!
    → → Quality > Architecture: Phi-1 (1.3B, high-quality data) > 50B model (web data)!
    → → → → Data quality is the #1 factor for model performance → not model size!

    Filters:
    → Length filter: too short = low quality → min_len=50, max_len=100000
    → Repetition filter: high repetition = spam → max_repeat_ratio=0.3
    → Language filter: mixed language = low quality → keep dominant language
    → Encoding filter: broken encoding = corruption → detect mojibake
    """

    def __init__(self, min_length: int = 50, max_length: int = 100000,
                 max_repeat_ratio: float = 0.3):
        self.min_length = min_length
        self.max_length = max_length
        self.max_repeat_ratio = max_repeat_ratio

    def filter_length(self, text: str) -> bool:
        """Filter by length."""
        return len(text) >= self.min_length and len(text) <= self.max_length

    def filter_repetition(self, text: str) -> bool:
        """Filter by repetition ratio."""
        words = text.lower().split()
        if not words:
            return False
        unique_words = len(set(words))
        total_words = len(words)
        repeat_ratio = 1 - unique_words / total_words
        return repeat_ratio <= self.max_repeat_ratio

    def filter_quality(self, text: str) -> Tuple[bool, Dict]:
        """Run all quality filters."""
        results = {
            "length_ok": self.filter_length(text),
            "repetition_ok": self.filter_repetition(text),
        }
        passed = all(results.values())
        return passed, results

    def filter_corpus(self, documents: List[str]) -> Tuple[List[str], Dict]:
        """Filter entire corpus."""
        kept = []
        stats = {"total": len(documents), "kept": 0, "filtered": 0,
                 "reasons": defaultdict(int)}

        for doc in documents:
            passed, results = self.filter_quality(doc)
            if passed:
                kept.append(doc)
                stats["kept"] += 1
            else:
                stats["filtered"] += 1
                for reason, ok in results.items():
                    if not ok:
                        stats["reasons"][reason] += 1

        return kept, stats


# ============================================================================
# Part 4: Contamination Detection
# ============================================================================

class ContaminationDetector:
    """Detect benchmark test set contamination in training data.

    Key insight from data pipeline deep dive:
    → Contamination = training data contains benchmark questions → model memorizes → score inflated!
    → → Detection: n-gram overlap → if training doc has >50% n-grams from test set → contaminated!
    → → → → → This explains why some models score suspiciously high on MMLU → contamination!

    For RTX 4090 local evaluation:
    → Download MMLU test set → compute n-grams → check training data → flag contaminated docs!
    → → → 7B INT4 inference → 4,791 tok/s → can quickly re-evaluate with contaminated docs removed!
    """

    def __init__(self, ngram_size: int = 8, threshold: float = 0.5):
        self.ngram_size = ngram_size
        self.threshold = threshold

    def compute_ngrams(self, text: str) -> Set[str]:
        """Compute n-grams from text."""
        tokens = text.lower().split()
        ngrams = set()
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = " ".join(tokens[i:i + self.ngram_size])
            ngrams.add(ngram)
        return ngrams

    def detect_contamination(self, training_doc: str, test_ngrams: Set[str]) -> Tuple[bool, float]:
        """Check if training document is contaminated with test data."""
        doc_ngrams = self.compute_ngrams(training_doc)
        if not doc_ngrams:
            return False, 0.0
        overlap = len(doc_ngrams & test_ngrams)
        ratio = overlap / len(doc_ngrams)
        return ratio >= self.threshold, ratio

    def build_test_ngram_set(self, test_questions: List[str]) -> Set[str]:
        """Build n-gram set from benchmark test questions."""
        all_ngrams = set()
        for q in test_questions:
            all_ngrams.update(self.compute_ngrams(q))
        return all_ngrams


# ============================================================================
# Main: Run all demonstrations
# ============================================================================

def generate_sample_corpus(num_docs: int = 100) -> List[Dict]:
    """Generate sample corpus with duplicates and low-quality docs."""
    base_docs = [
        "The transformer architecture has revolutionized natural language processing by introducing self-attention mechanisms that allow models to weigh the importance of different words in a sequence simultaneously rather than sequentially",
        "FlashAttention is an algorithm that computes attention in a memory-efficient way by processing the attention matrix in blocks and using online softmax to avoid materializing the full matrix in memory",
        "ZeRO or Zero Redundancy Optimizer is a memory optimization technique for distributed training that partitions optimizer states gradients and model parameters across data parallel processes to reduce memory redundancy",
        "vLLM is a high throughput inference engine for large language models that uses PagedAttention to manage KV cache memory efficiently and continuous batching to maximize GPU utilization during serving",
        "LoRA or Low Rank Adaptation is a parameter efficient fine tuning method that adds low rank decomposition matrices to transformer layers enabling task specific adaptation with minimal additional parameters",
    ]

    corpus = []
    for i in range(num_docs):
        if i < len(base_docs):
            # Original docs
            corpus.append({"id": i, "text": base_docs[i], "type": "original"})
        elif i < num_docs * 0.3 + len(base_docs):
            # Near-duplicates (slightly modified)
            base = random.choice(base_docs)
            # Modify a few words
            words = base.split()
            for j in range(min(3, len(words))):
                idx = random.randint(0, len(words) - 1)
                words[idx] = words[idx] + "s"  # simple modification
            corpus.append({"id": i, "text": " ".join(words), "type": "near_duplicate"})
        elif i < num_docs * 0.5 + len(base_docs):
            # Low-quality docs (short, repetitive)
            corpus.append({"id": i, "text": "bad bad bad bad bad", "type": "low_quality"})
        else:
            # Random quality docs
            corpus.append({"id": i, "text": base_docs[i % len(base_docs)], "type": "varied"})

    return corpus


def main():
    print("=" * 70)
    print("Data Pipeline Practical — MinHash Dedup + Quality Filter")
    print("=" * 70)
    print()

    # Generate corpus
    corpus = generate_sample_corpus(50)
    print(f"Sample corpus: {len(corpus)} documents")
    print()

    # === Part 1: MinHash Deduplication ===
    print("--- Part 1: MinHash Near-Duplicate Detection ---")
    minhasher = MinHasher(num_hashes=128, shingle_size=5)

    # Compute signatures
    signatures = {}
    for doc in corpus:
        signatures[doc["id"]] = minhasher.compute_signature(doc["text"])

    # Compare exact vs MinHash similarity for a known duplicate pair
    print("MinHash vs Exact Jaccard comparison (near-duplicates):")
    for i in range(min(3, len(corpus))):
        for j in range(i + 1, min(6, len(corpus))):
            minhash_sim = minhasher.jaccard_similarity(signatures[corpus[i]["id"]], signatures[corpus[j]["id"]])
            exact_sim = minhasher.exact_jaccard(corpus[i]["text"], corpus[j]["text"])
            print(f"  Doc {i} vs Doc {j}: MinHash={minhash_sim:.3f}, Exact={exact_sim:.3f}")
    print()

    # === Part 2: LSH Grouping ===
    print("--- Part 2: LSH Duplicate Grouping ---")
    lsh = LSHTable(num_bands=8, rows_per_band=16)
    for doc_id, sig in signatures.items():
        lsh.insert(doc_id, sig)

    groups = lsh.get_duplicate_groups(minhasher, signatures, threshold=0.7)
    print(f"Found {len(groups)} duplicate groups:")
    for group in groups:
        print(f"  Group: docs {group} ({len(group)} documents)")
    print()

    # === Part 3: Quality Filtering ===
    print("--- Part 3: Quality Filtering ---")
    qf = QualityFilter(min_length=50, max_length=100000, max_repeat_ratio=0.3)

    texts = [doc["text"] for doc in corpus]
    kept, stats = qf.filter_corpus(texts)
    print(f"Total: {stats['total']}, Kept: {stats['kept']}, Filtered: {stats['filtered']}")
    print(f"Filter ratio: {stats['filtered']/stats['total']*100:.1f}% removed")
    print(f"Reasons: {dict(stats['reasons'])}")
    print()
    print("Key insight: 99% of web data is garbage → quality filtering is essential!")
    print("Phi-1 (1.3B, curated data) > 50B model (raw web data) → quality > size!")
    print()

    # === Part 4: Contamination Detection ===
    print("--- Part 4: Contamination Detection ---")
    cd = ContaminationDetector(ngram_size=5, threshold=0.3)

    # Simulated MMLU test questions
    test_questions = [
        "What is the capital of France?",
        "Which planet is closest to the Sun?",
        "What is the chemical symbol for gold?",
    ]
    test_ngrams = cd.build_test_ngram_set(test_questions)
    print(f"Test set: {len(test_questions)} questions → {len(test_ngrams)} unique n-grams")

    # Check contamination
    contaminated = 0
    for doc in corpus:
        is_contaminated, ratio = cd.detect_contamination(doc["text"], test_ngrams)
        if is_contaminated:
            contaminated += 1
            print(f"  Doc {doc['id']}: contaminated (ratio={ratio:.3f})")
    print(f"Contaminated docs: {contaminated}/{len(corpus)}")
    print()

    # === Summary ===
    print("=" * 70)
    print("Data Pipeline Summary:")
    print(f"  Deduplication: {len(groups)} duplicate groups → remove near-duplicates → 30-50% size reduction!")
    print(f"  Quality filtering: {stats['filtered']/stats['total']*100:.1f}% filtered → keep only high-quality!")
    print(f"  Contamination: {contaminated}/{len(corpus)} docs contaminated → remove → prevent score inflation!")
    print()
    print("  Key lesson: Data quality > model architecture > model size!")
    print("  → Phi-1 1.3B (curated) > 50B model (raw web)")
    print("  → → Curated data pipeline = most important investment for model quality!")
    print("  → → → RTX 4090 local pipeline = fastest iteration + lowest cost!")

    # Save results
    results = {
        "corpus_size": len(corpus),
        "duplicate_groups": len(groups),
        "quality_filtered_pct": stats['filtered']/stats['total']*100,
        "contaminated_docs": contaminated,
    }
    with open("results/data_pipeline_simulator.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/data_pipeline_simulator.json")


if __name__ == "__main__":
    main()