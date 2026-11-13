"""Shared retrieval logic used by both the app and the evaluation (pure Python)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from ragnova_retrieval import (  # noqa: E402
    build_answer_prompt,
    hybrid_candidates,
    rerank,
    retrieve,
    rrf_fuse,
    tokenize,
    top_indices,
)


def test_rrf_fuse_scores_and_order() -> None:
    # "b" is ranked 2nd and 1st -> 1/62 + 1/61 beats "a" (1/61 only) and "c" (1/62 only).
    assert rrf_fuse([["a", "b"], ["b", "c"]], k=60) == ["b", "a", "c"]


def test_rrf_ties_keep_first_seen_order() -> None:
    assert rrf_fuse([["x"], ["y"]]) == ["x", "y"]


def test_hybrid_candidates_truncates_inputs_and_output() -> None:
    semantic = list(range(30))
    lexical = list(range(29, -1, -1))
    fused = hybrid_candidates(semantic, lexical, pool=5)
    assert len(fused) == 5
    assert set(fused) <= set(semantic[:5]) | set(lexical[:5])


def test_top_indices_highest_first_stable_ties() -> None:
    assert top_indices([0.1, 0.9, 0.5, 0.9], 3) == [1, 3, 2]


def test_rerank_orders_by_score() -> None:
    assert rerank(["a", "b", "c"], [0.1, 3.0, 2.0], top_k=2) == [("b", 3.0), ("c", 2.0)]


class FakeStore:
    def __init__(self, order):
        self.order = order

    def similarity_search(self, query, k):
        return [SimpleNamespace(metadata={"chunk_idx": i}) for i in self.order[:k]]


class FakeBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        return self.scores


def test_retrieve_fuses_semantic_and_bm25() -> None:
    store = FakeStore([0, 1, 2])
    bm25 = FakeBM25([0.0, 0.0, 0.0, 9.0])  # chunk 3 only found by BM25
    fused = retrieve("q", store, bm25, pool=2)
    assert fused == [0, 3]  # tie at rank 1 -> semantic hit first, BM25 hit second


def test_retrieve_semantic_only_without_bm25() -> None:
    assert retrieve("q", FakeStore([4, 2, 7]), None, pool=2) == [4, 2]


def test_tokenize() -> None:
    assert tokenize("GDPR Article 30") == ["gdpr", "article", "30"]


def test_answer_prompt_matches_prompt_used_for_saved_judge_answers() -> None:
    """eval/judge/judge_outputs.jsonl was generated with this exact prompt text."""
    expected = (
        "Use the following pieces of context to answer the question at the end. If you don't "
        "know the answer, just say that you don't know, don't try to make up an answer.\n\n"
        "P1\n\nP2\n\nQuestion: Q?\nHelpful Answer:"
    )
    assert build_answer_prompt("Q?", ["P1", "P2"]) == expected


def test_answer_prompt_prepends_history() -> None:
    prompt = build_answer_prompt("Q?", ["P"], history="User: hi\nAssistant: hello")
    assert prompt.startswith("Conversation so far:\nUser: hi\nAssistant: hello\n\n")
    assert prompt.endswith("Question: Q?\nHelpful Answer:")


@pytest.mark.parametrize("pool", [1, 3, 20])
def test_candidate_pool_bounds(pool: int) -> None:
    assert len(hybrid_candidates(range(50), range(50), pool)) <= pool
