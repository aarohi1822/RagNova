"""Retrieval pipeline shared by the Streamlit app and the offline evaluation.

frontend/streamlit_app.py and eval/retrievers.py both call these functions, so the
configuration measured offline ("Hybrid + cross-encoder" in results/metrics.json)
is the code the app runs. The module is pure Python: callers pass in the vector
store, BM25 index and reranker scores, so it can be unit-tested without models.

Pipeline: FAISS top-`pool` + BM25 top-`pool` -> Reciprocal Rank Fusion -> top-`pool`
candidates -> cross-encoder rerank -> top-k passages, which are used for the
answer, the citations, the grounding check and the LLM judge.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import Any

RRF_K = 60  # standard RRF constant
CANDIDATE_POOL = 20  # hits taken from each retriever, fused and reranked
ANSWER_TOP_K = 3  # passages given to the LLM and shown as citations

# LangChain's default "stuff" QA prompt (also used by eval/judge_export.py).
ANSWER_PROMPT = (
    "Use the following pieces of context to answer the question at the end. If you don't "
    "know the answer, just say that you don't know, don't try to make up an answer.\n\n"
    "{context}\n\nQuestion: {question}\nHelpful Answer:"
)


def tokenize(text: str) -> list[str]:
    """BM25 tokenisation used everywhere: lowercase, split on whitespace."""
    return text.lower().split()


def top_indices(scores: Sequence[float], n: int) -> list[int]:
    """Indices of the n highest scores, highest first; ties keep their original order."""
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:n]


def rrf_fuse(rankings: Iterable[Sequence[Hashable]], k: int = RRF_K) -> list[Hashable]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)).

    Ties keep the order in which items were first seen (the first ranking wins).
    """
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda key: -scores[key])


def hybrid_candidates(
    semantic: Sequence[Hashable], lexical: Sequence[Hashable], pool: int = CANDIDATE_POOL
) -> list[Hashable]:
    """Fuse the top-`pool` semantic and lexical hits with RRF; keep the top `pool`."""
    return rrf_fuse([list(semantic)[:pool], list(lexical)[:pool]])[:pool]


def retrieve(query: str, vectorstore: Any, bm25: Any | None,
             pool: int = CANDIDATE_POOL) -> list[int]:
    """Candidate chunk indices for `query`.

    `vectorstore.similarity_search(query, k)` must return documents whose
    `metadata["chunk_idx"]` is the chunk's position; `bm25.get_scores(tokens)` must
    score every chunk in the same order. With `bm25=None` only semantic hits are used.
    """
    semantic = [doc.metadata["chunk_idx"] for doc in vectorstore.similarity_search(query, k=pool)]
    if bm25 is None:
        return semantic[:pool]
    lexical = top_indices(bm25.get_scores(tokenize(query)), pool)
    return hybrid_candidates(semantic, lexical, pool)


def rerank(candidates: Sequence[Any], scores: Sequence[float],
           top_k: int = ANSWER_TOP_K) -> list[tuple[Any, float]]:
    """Order candidates by reranker score (highest first) and keep the top `top_k`."""
    return [(candidates[i], float(scores[i])) for i in top_indices(scores, top_k)]


def build_answer_prompt(question: str, passages: Sequence[str], history: str = "") -> str:
    """Prompt for answer generation from exactly the passages that will be cited."""
    prompt = ANSWER_PROMPT.format(context="\n\n".join(passages), question=question)
    if history:
        prompt = f"Conversation so far:\n{history}\n\n{prompt}"
    return prompt
