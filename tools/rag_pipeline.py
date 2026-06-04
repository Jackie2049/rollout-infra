#!/usr/bin/env python3
"""Simple RAG (Retrieval Augmented Generation) Pipeline
=======================================================
Demonstrates RAG from first principles:
1. Document chunking (fixed-size, sentence, recursive)
2. Embedding generation (using a simple encoder)
3. Vector similarity search (cosine similarity)
4. Context injection into generation (prompt engineering)

No external model downloads needed — uses synthetic documents
and a learned embedding model.

Educational purpose: understand RAG systems from first principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 1. Document Processing
# ============================================================

@dataclass
class Chunk:
    """A text chunk with metadata."""
    chunk_id: int
    text: str
    tokens: list[int]
    source_doc: int
    start_char: int
    end_char: int


def create_synthetic_documents(vocab_size=256, num_docs=50, doc_length=200):
    """Create synthetic documents for RAG testing."""
    documents = []
    topics = [
        "transformer", "attention", "training", "inference",
        "optimization", "quantization", "distillation", "embedding",
        "generation", "sampling", "batching", "caching",
        "parallel", "distributed", "gradient", "normalization",
    ]

    for doc_id in range(num_docs):
        topic = topics[doc_id % len(topics)]
        # Create document with thematic tokens
        base = hash(topic) % (vocab_size // 4) + 10
        tokens = []
        for i in range(doc_length):
            # Mix of topic-specific and random tokens
            if torch.randint(0, 3, (1,)).item() == 0:
                tokens.append(torch.randint(10, vocab_size // 2, (1,)).item())
            else:
                # Topic-specific with variation
                tokens.append((base + i * 3 + torch.randint(0, 5, (1,)).item()) % (vocab_size // 2))
        documents.append({
            "doc_id": doc_id,
            "topic": topic,
            "tokens": tokens,
            "text": f"[{topic} doc {doc_id}]",  # Label for readability
        })
    return documents


def chunk_fixed_size(tokens, chunk_size=64, overlap=8):
    """Fixed-size chunking with optional overlap."""
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(tokens) - chunk_size + 1, step):
        chunks.append(tokens[i:i + chunk_size])
    if len(tokens) % step != 0 and len(tokens) > chunk_size:
        chunks.append(tokens[-chunk_size:])
    return chunks


def chunk_recursive(tokens, min_size=32, max_size=128, vocab_size=256):
    """Recursive chunking: split at special boundaries, then size."""
    # Find natural boundaries (large token jumps)
    boundaries = [0]
    for i in range(1, len(tokens)):
        if abs(tokens[i] - tokens[i-1]) > vocab_size // 4:
            boundaries.append(i)
    boundaries.append(len(tokens))

    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment = tokens[start:end]
        if len(segment) <= max_size:
            if len(segment) >= min_size:
                chunks.append(segment)
        else:
            # Split further
            for j in range(0, len(segment), max_size):
                sub = segment[j:j + max_size]
                if len(sub) >= min_size:
                    chunks.append(sub)
    return chunks


# ============================================================
# 2. Embedding Model
# ============================================================

class SimpleEmbeddingModel(nn.Module):
    """Simple embedding model: token IDs → fixed-size vector.

    In production, this would be a pre-trained model like:
    - text-embedding-ada-002 (OpenAI)
    - BGE-base-en (BAAI)
    - E5-base (Microsoft)
    - GTE-base (Alibaba)

    Here we use a simple learned embedding + mean pooling.
    """
    def __init__(self, vocab_size=256, d_model=64, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(self, token_ids):
        """Encode token sequence into embedding vector.

        Args:
            token_ids: [B, T] token IDs

        Returns:
            embedding: [B, d_model] normalized embedding vector
        """
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        # Mean pooling
        x = x.mean(dim=1)
        # Project
        x = self.proj(x)
        # L2 normalize (for cosine similarity)
        x = F.normalize(x, p=2, dim=-1)
        return x


# ============================================================
# 3. Vector Store
# ============================================================

class SimpleVectorStore:
    """In-memory vector store with cosine similarity search.

    Production systems use:
    - FAISS (Facebook) — GPU-accelerated ANN
    - Milvus — distributed vector DB
    - Pinecone — managed vector DB
    - ChromaDB — lightweight, Python-native
    - Qdrant — Rust-based, high performance
    """
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.embeddings = []  # List of [d_model] tensors
        self.metadata = []    # List of dicts

    def add(self, embedding, metadata):
        """Add a chunk embedding with metadata."""
        self.embeddings.append(embedding)
        self.metadata.append(metadata)

    def search(self, query_embedding, top_k=5):
        """Search for most similar chunks.

        Args:
            query_embedding: [d_model] normalized query embedding
            top_k: number of results

        Returns:
            List of (score, metadata) tuples, sorted by score descending
        """
        if not self.embeddings:
            return []

        # Stack all embeddings: [N, d_model]
        all_emb = torch.stack(self.embeddings)

        # Cosine similarity (both are L2-normalized)
        scores = (all_emb @ query_embedding).squeeze(-1)  # [N]

        # Top-k
        top_scores, top_indices = torch.topk(scores, min(top_k, len(self.embeddings)))

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((score, self.metadata[idx]))
        return results

    def __len__(self):
        return len(self.embeddings)


# ============================================================
# 4. Simple Generator (for context-augmented generation)
# ============================================================

class SimpleGenerator(nn.Module):
    """Simple autoregressive generator.

    In production, this would be a pre-trained LLM like:
    - GPT-4, Claude, LLaMA, Qwen, Mistral
    """
    def __init__(self, vocab_size=256, d_model=128, n_head=4, n_layer=2, max_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head,
                dim_feedforward=4 * d_model,
                batch_first=True, dropout=0.1
            ),
            num_layers=n_layer
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=32, temperature=0.8, top_k=20):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================
# 5. RAG Pipeline
# ============================================================

class RAGPipeline:
    """Complete RAG pipeline: Index → Retrieve → Generate."""

    def __init__(self, vocab_size=256, embed_dim=64, device='cpu'):
        self.vocab_size = vocab_size
        self.device = device
        self.embed_model = SimpleEmbeddingModel(vocab_size, embed_dim).to(device)
        self.vector_store = SimpleVectorStore(embed_dim)
        self.generator = SimpleGenerator(vocab_size).to(device)

    def index_documents(self, documents, chunking='fixed', chunk_size=64, overlap=8):
        """Index documents into vector store.

        Pipeline:
        1. Chunk documents
        2. Generate embeddings for each chunk
        3. Store in vector store
        """
        self.embed_model.eval()
        total_chunks = 0

        for doc in documents:
            tokens = doc['tokens']

            # Chunking
            if chunking == 'fixed':
                chunks = chunk_fixed_size(tokens, chunk_size, overlap)
            elif chunking == 'recursive':
                chunks = chunk_recursive(tokens, min_size=32, max_size=chunk_size*2, vocab_size=self.vocab_size)
            else:
                chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

            for chunk_idx, chunk_tokens in enumerate(chunks):
                # Pad to chunk_size if needed
                if len(chunk_tokens) < chunk_size:
                    chunk_tokens = chunk_tokens + [0] * (chunk_size - len(chunk_tokens))

                token_tensor = torch.tensor([chunk_tokens], device=self.device)
                embedding = self.embed_model(token_tensor).squeeze(0)  # [d_model]

                self.vector_store.add(
                    embedding,
                    metadata={
                        'doc_id': doc['doc_id'],
                        'topic': doc['topic'],
                        'chunk_idx': chunk_idx,
                        'tokens': chunk_tokens[:32],  # Store first 32 for reference
                    }
                )
                total_chunks += 1

        return total_chunks

    def retrieve(self, query_tokens, top_k=5):
        """Retrieve relevant chunks for a query.

        Pipeline:
        1. Embed query
        2. Search vector store
        3. Return top-k results
        """
        self.embed_model.eval()
        if len(query_tokens) < 64:
            query_tokens = query_tokens + [0] * (64 - len(query_tokens))
        query_tensor = torch.tensor([query_tokens[:64]], device=self.device)
        query_embedding = self.embed_model(query_tensor).squeeze(0)

        results = self.vector_store.search(query_embedding, top_k)
        return results

    def generate_with_context(self, query_tokens, context_chunks, max_new_tokens=32):
        """Generate response with retrieved context.

        Pipeline:
        1. Combine context chunks with query
        2. Feed to generator
        3. Return generated tokens
        """
        # Combine context tokens (from top chunks) with query
        context_tokens = []
        for score, meta in context_chunks[:3]:  # Use top 3 chunks
            context_tokens.extend(meta['tokens'][:32])

        # Format: [context_tokens] + [sep] + [query_tokens]
        sep_token = self.vocab_size - 1
        combined = context_tokens + [sep_token] + query_tokens

        # Truncate to max_len
        if len(combined) > self.generator.max_len:
            combined = combined[-self.generator.max_len:]

        input_tensor = torch.tensor([combined], device=self.device)
        output = self.generator.generate(input_tensor, max_new_tokens=max_new_tokens)
        return output[0].tolist()


# ============================================================
# Experiments
# ============================================================

def experiment_chunking_strategies(pipeline, documents):
    """Compare different chunking strategies."""
    print("\n  --- Experiment: Chunking Strategies ---")

    strategies = [
        ('fixed-64-0', 'fixed', 64, 0),
        ('fixed-64-16', 'fixed', 64, 16),
        ('fixed-128-0', 'fixed', 128, 0),
        ('recursive', 'recursive', 64, 0),
    ]

    results = {}
    for name, strategy, size, overlap in strategies:
        pipeline.vector_store = SimpleVectorStore(64)
        n_chunks = pipeline.index_documents(documents, strategy, size, overlap)
        results[name] = n_chunks
        print(f"    {name}: {n_chunks} chunks")

    return results


def experiment_retrieval_quality(pipeline, documents):
    """Test retrieval quality: can we find the right document?"""
    print("\n  --- Experiment: Retrieval Quality ---")

    pipeline.vector_store = SimpleVectorStore(64)
    pipeline.index_documents(documents, 'fixed', 64, 8)

    # Create queries based on document topics
    topics = list(set(doc['topic'] for doc in documents))
    correct = 0
    total = 0

    for topic in topics[:20]:
        # Query: tokens similar to the topic
        base = hash(topic) % 64 + 10
        query_tokens = [(base + i * 3) % 128 for i in range(32)]

        results = pipeline.retrieve(query_tokens, top_k=5)
        if results:
            top_topic = results[0][1]['topic']
            is_correct = top_topic == topic
            correct += is_correct
            total += 1
            if total <= 5:
                print(f"    Query '{topic}': top result = '{top_topic}' "
                      f"({'correct' if is_correct else 'wrong'}, score={results[0][0]:.4f})")

    accuracy = correct / max(total, 1)
    print(f"\n    Retrieval accuracy: {correct}/{total} ({accuracy:.1%})")
    return accuracy


def experiment_top_k_sensitivity(pipeline, documents):
    """Test how top_k affects retrieval quality."""
    print("\n  --- Experiment: Top-K Sensitivity ---")

    pipeline.vector_store = SimpleVectorStore(64)
    pipeline.index_documents(documents, 'fixed', 64, 8)

    topics = list(set(doc['topic'] for doc in documents))

    for top_k in [1, 3, 5, 10, 20]:
        hits = 0
        total = 0
        for topic in topics[:20]:
            base = hash(topic) % 64 + 10
            query_tokens = [(base + i * 3) % 128 for i in range(32)]
            results = pipeline.retrieve(query_tokens, top_k=top_k)
            # Check if correct topic is in top-k
            retrieved_topics = [meta['topic'] for _, meta in results]
            hits += topic in retrieved_topics
            total += 1
        recall = hits / max(total, 1)
        print(f"    top_k={top_k:>3}: recall={recall:.1%} ({hits}/{total} queries hit)")

    return


def experiment_embedding_dim(pipeline, documents, device):
    """Test how embedding dimension affects retrieval quality."""
    print("\n  --- Experiment: Embedding Dimension ---")

    topics = list(set(doc['topic'] for doc in documents))

    for dim in [16, 32, 64, 128]:
        embed_model = SimpleEmbeddingModel(256, dim).to(device).eval()
        vector_store = SimpleVectorStore(dim)

        # Index
        for doc in documents:
            tokens = doc['tokens'][:64]
            if len(tokens) < 64:
                tokens = tokens + [0] * (64 - len(tokens))
            with torch.no_grad():
                emb = embed_model(torch.tensor([tokens], device=device)).squeeze(0)
            vector_store.add(emb, {'doc_id': doc['doc_id'], 'topic': doc['topic']})

        # Query
        correct = 0
        total = 0
        for topic in topics[:20]:
            base = hash(topic) % 64 + 10
            query_tokens = [(base + i * 3) % 128 for i in range(32)]
            if len(query_tokens) < 64:
                query_tokens = query_tokens + [0] * (64 - len(query_tokens))
            with torch.no_grad():
                q_emb = embed_model(torch.tensor([query_tokens], device=device)).squeeze(0)
            results = vector_store.search(q_emb, top_k=1)
            if results and results[0][1]['topic'] == topic:
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        print(f"    dim={dim:>4}: accuracy={accuracy:.1%}")

    return


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("RAG (Retrieval Augmented Generation) Pipeline")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print(f"\n  Device: CPU")

    results = {}
    vocab_size = 256

    # Create documents
    print("\n  Creating synthetic documents...")
    documents = create_synthetic_documents(vocab_size, num_docs=50, doc_length=200)
    print(f"  Created {len(documents)} documents, topics: {list(set(d['topic'] for d in documents))}")

    # Initialize pipeline
    pipeline = RAGPipeline(vocab_size, embed_dim=64, device=device)

    # Exp 1: Chunking strategies
    chunk_results = experiment_chunking_strategies(pipeline, documents)
    results["chunking"] = chunk_results

    # Exp 2: Retrieval quality
    retrieval_acc = experiment_retrieval_quality(pipeline, documents)
    results["retrieval_accuracy"] = round(retrieval_acc, 3)

    # Exp 3: Top-k sensitivity
    experiment_top_k_sensitivity(pipeline, documents)

    # Exp 4: Embedding dimension
    experiment_embedding_dim(pipeline, documents, device)

    # Summary
    print("\n" + "=" * 60)
    print("RAG Pipeline Architecture Summary")
    print("=" * 60)
    print("""
    Production RAG Pipeline:
    ┌─────────────────────────────────────────────┐
    │ 1. INDEX: Documents → Chunks → Embeddings → Vector DB    │
    │    - Chunking: fixed/recursive/semantic                    │
    │    - Embedding: BGE/E5/GTE (768-1536 dim)                 │
    │    - Vector DB: FAISS/Milvus/Pinecone/ChromaDB            │
    │                                                            │
    │ 2. RETRIEVE: Query → Embed → Search → Top-K chunks        │
    │    - Similarity: cosine/dot-product/ANN                    │
    │    - Re-ranking: cross-encoder rerank (optional)           │
    │    - Top-K: typically 3-10 chunks                          │
    │                                                            │
    │ 3. GENERATE: [Context + Query] → LLM → Response            │
    │    - Context injection: prompt engineering                 │
    │    - Generator: GPT-4/Claude/LLaMA/Qwen                   │
    │    - Citation: map generated tokens → source chunks        │
    └─────────────────────────────────────────────┘

    Key Design Decisions:
    - Chunk size: 256-512 tokens (smaller = more precise, larger = more context)
    - Overlap: 10-20% (prevents losing info at boundaries)
    - Embedding dim: 768-1536 (trade-off: precision vs storage)
    - Top-K: 3-5 (more context helps, but can confuse the model)

    Advanced Techniques (2024-2026):
    - Semantic chunking (split by meaning, not size)
    - Hybrid search (vector + keyword/BM25)
    - Re-ranking with cross-encoders
    - Query decomposition (complex query → sub-queries)
    - Adaptive retrieval (decide when RAG is needed)
    - Graph RAG (knowledge graph + vector search)
    - Multi-modal RAG (images, tables, code)
    """)

    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak GPU memory: {mem:.1f} MB")
        results["gpu_memory_mb"] = round(mem, 1)

    with open("rag_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to rag_results.json")


if __name__ == "__main__":
    main()
