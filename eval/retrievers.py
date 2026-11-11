"""Retrieval configurations evaluated by the ablation.

They mirror frontend/streamlit_app.py:
  * BM25:   rank_bm25.BM25Okapi over `text.lower().split()` tokens.
  * FAISS:  sentence-transformers/all-MiniLM-L6-v2 embeddings (not normalised, as
            LangChain's HuggingFaceEmbeddings default) in a flat L2 index, which is
            what langchain_community.vectorstores.FAISS.from_documents builds.
  * Rerank: cross-encoder/ms-marco-MiniLM-L-6-v2.

Hybrid uses Reciprocal Rank Fusion (RRF, k=60) over the top-`pool` hits of each
retriever. The app's current hybrid (semantic hits followed by BM25 hits, deduped)
is included as `app_concat` for reference.
"""

from __future__ import annotations

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


class Retrievers:
    def __init__(self, passages: list[dict], device: str = "cpu", pool: int = 20) -> None:
        # Import order matters: on macOS, loading faiss before torch makes both
        # link conflicting OpenMP runtimes and the process segfaults on encode.
        from sentence_transformers import CrossEncoder, SentenceTransformer  # isort: skip
        import faiss  # isort: skip
        from rank_bm25 import BM25Okapi

        self.ids = [p["id"] for p in passages]
        self.texts = [p["text"] for p in passages]
        self.pool = pool
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        vectors = self.embedder.encode(self.texts, batch_size=64, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors.astype(np.float32))
        self.reranker = CrossEncoder(RERANK_MODEL, device=device)

    # --- single retrievers -------------------------------------------------
    def bm25_only(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(query.lower().split())
        top = np.argsort(scores)[::-1][:k]
        return [self.ids[i] for i in top]

    def faiss_only(self, query: str, k: int) -> list[str]:
        vec = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        _, idx = self.index.search(vec, k)
        return [self.ids[i] for i in idx[0]]

    # --- combinations ------------------------------------------------------
    def hybrid(self, query: str, k: int) -> list[str]:
        scores: dict[str, float] = {}
        for ranked in (self.faiss_only(query, self.pool), self.bm25_only(query, self.pool)):
            for rank, pid in enumerate(ranked, start=1):
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(scores, key=scores.get, reverse=True)[:k]

    def hybrid_rerank(self, query: str, k: int) -> list[str]:
        candidates = self.hybrid(query, self.pool)
        text_of = dict(zip(self.ids, self.texts, strict=True))
        scores = self.reranker.predict([(query, text_of[c]) for c in candidates])
        order = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in order]

    def app_concat(self, query: str, k: int) -> list[str]:
        """The Streamlit app's current merge: semantic first, then BM25, deduped."""
        merged: list[str] = []
        for pid in self.faiss_only(query, k) + self.bm25_only(query, k):
            if pid not in merged:
                merged.append(pid)
        return merged[:k]

    def configs(self) -> dict:
        return {
            "BM25 only": self.bm25_only,
            "FAISS only": self.faiss_only,
            "Hybrid (RRF)": self.hybrid,
            "Hybrid + cross-encoder": self.hybrid_rerank,
            "App hybrid (concat, ref.)": self.app_concat,
        }
