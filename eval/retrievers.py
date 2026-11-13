"""Retrieval configurations evaluated by the ablation.

Ranking, fusion and reranking use frontend/ragnova_retrieval.py, the module the
Streamlit app calls, so the benchmarked logic is the app's logic:
  * BM25:   rank_bm25.BM25Okapi over `tokenize()` (lowercase, whitespace split).
  * FAISS:  sentence-transformers/all-MiniLM-L6-v2 embeddings (not normalised, as
            LangChain's HuggingFaceEmbeddings default) in a flat L2 index, which is
            what langchain_community.vectorstores.FAISS.from_documents builds.
  * Hybrid: Reciprocal Rank Fusion (k=60) of the top-`pool` hits of each retriever.
  * Rerank: cross-encoder/ms-marco-MiniLM-L-6-v2 over the top-`pool` fused candidates.

"App pipeline" runs the app's own code path end to end: LangChain FAISS +
HuggingFaceEmbeddings built exactly as frontend/streamlit_app.py builds them, then
ragnova_retrieval.retrieve() and the same reranker. It should match
"Hybrid + cross-encoder". "Concat merge (previous app)" is the merge the app used
before RRF (semantic hits, then BM25 hits, deduplicated), kept for reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from ragnova_retrieval import (  # noqa: E402
    CANDIDATE_POOL,
    hybrid_candidates,
    rerank,
    retrieve,
    tokenize,
    top_indices,
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retrievers:
    def __init__(self, passages: list[dict], device: str = "cpu", pool: int = CANDIDATE_POOL,
                 app_pipeline: bool = True) -> None:
        # Import order matters: on macOS, loading faiss before torch makes both
        # link conflicting OpenMP runtimes and the process segfaults on encode.
        from sentence_transformers import CrossEncoder, SentenceTransformer  # isort: skip
        import faiss  # isort: skip
        from rank_bm25 import BM25Okapi

        self.ids = [p["id"] for p in passages]
        self.texts = [p["text"] for p in passages]
        self.text_of = dict(zip(self.ids, self.texts, strict=True))
        self.pool = pool
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        vectors = self.embedder.encode(self.texts, batch_size=64, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors.astype(np.float32))
        self.reranker = CrossEncoder(RERANK_MODEL, device=device)
        self.app_store = self._build_app_store(passages) if app_pipeline else None

    @staticmethod
    def _build_app_store(passages: list[dict]):
        """Vector store built the way frontend/streamlit_app.py builds it."""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        docs = [Document(page_content=p["text"], metadata={"chunk_idx": i})
                for i, p in enumerate(passages)]
        return FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name=EMBED_MODEL))

    def _rerank(self, query: str, candidates: list[str], k: int) -> list[str]:
        scores = self.reranker.predict([(query, self.text_of[c]) for c in candidates]).tolist()
        return [c for c, _ in rerank(candidates, scores, k)]

    # --- single retrievers -------------------------------------------------
    def bm25_only(self, query: str, k: int) -> list[str]:
        return [self.ids[i] for i in top_indices(self.bm25.get_scores(tokenize(query)), k)]

    def faiss_only(self, query: str, k: int) -> list[str]:
        vec = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        _, idx = self.index.search(vec, k)
        return [self.ids[i] for i in idx[0]]

    # --- combinations ------------------------------------------------------
    def hybrid(self, query: str, k: int) -> list[str]:
        fused = hybrid_candidates(self.faiss_only(query, self.pool),
                                  self.bm25_only(query, self.pool), self.pool)
        return fused[:k]

    def hybrid_rerank(self, query: str, k: int) -> list[str]:
        return self._rerank(query, self.hybrid(query, self.pool), k)

    def app_pipeline(self, query: str, k: int) -> list[str]:
        candidates = [self.ids[i] for i in retrieve(query, self.app_store, self.bm25, self.pool)]
        return self._rerank(query, candidates, k)

    def concat_merge(self, query: str, k: int) -> list[str]:
        """The app's previous merge: semantic first, then BM25, deduped."""
        merged: list[str] = []
        for pid in self.faiss_only(query, k) + self.bm25_only(query, k):
            if pid not in merged:
                merged.append(pid)
        return merged[:k]

    def configs(self) -> dict:
        configs = {
            "BM25 only": self.bm25_only,
            "FAISS only": self.faiss_only,
            "Hybrid (RRF)": self.hybrid,
            "Hybrid + cross-encoder": self.hybrid_rerank,
            "Concat merge (previous app)": self.concat_merge,
        }
        if self.app_store is not None:
            configs["App pipeline (frontend code)"] = self.app_pipeline
        return configs
