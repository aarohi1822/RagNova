"""Benchmark integrity and metric correctness (no ML dependencies needed)."""

import json
import math
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from eval_metrics import (  # noqa: E402
    bootstrap_ci,
    cohen_kappa,
    hit_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def test_benchmark_has_60_queries_with_valid_gold_ids() -> None:
    bench = _jsonl(EVAL_DIR / "benchmark.jsonl")
    passage_ids = {p["id"] for p in _jsonl(EVAL_DIR / "corpus" / "passages.jsonl")}
    assert len(bench) == 60
    assert len({q["qid"] for q in bench}) == 60
    assert len({q["query"] for q in bench}) == 60
    for q in bench:
        assert q["gold_ids"], q["qid"]
        assert set(q["gold_ids"]) <= passage_ids, q["qid"]
        assert q["type"] in {"lexical", "paraphrase"}


def test_benchmark_matches_spec() -> None:
    """benchmark.jsonl must be regenerated whenever benchmark_spec.py changes."""
    from benchmark_spec import QUERIES
    from make_benchmark import load_passages, resolve

    bench = _jsonl(EVAL_DIR / "benchmark.jsonl")
    passages = load_passages()
    assert [q["query"] for q in bench] == [s["query"] for s in QUERIES]
    assert [q["gold_ids"] for q in bench] == [resolve(s, passages) for s in QUERIES]


def test_ranking_metrics() -> None:
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert recall_at_k(ranked, ["b", "z"], 5) == 0.5
    assert hit_at_k(ranked, ["f"], 5) == 0.0
    assert reciprocal_rank(ranked, ["c", "e"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ["z"]) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(ranked, [], 5)


def test_percentile_matches_numpy_linear() -> None:
    values = [1, 2, 3, 4, 100]
    assert percentile(values, 50) == 3
    assert percentile(values, 95) == pytest.approx(80.8)


def test_cohen_kappa() -> None:
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    # textbook example: p_o=0.7, p_e=0.5 -> kappa=0.4
    human = [True] * 5 + [False] * 5
    judge = [True] * 4 + [False] + [True] * 2 + [False] * 3
    assert cohen_kappa(human, judge) == pytest.approx(0.4)
    assert math.isnan(cohen_kappa([True, True], [True, True]))


def test_bootstrap_ci_brackets_mean() -> None:
    lo, hi = bootstrap_ci([0.0, 1.0] * 30)
    assert lo < 0.5 < hi


def test_readme_results_table_is_generated_from_metrics_json() -> None:
    """The README table must be exactly what eval/report.py renders from results/metrics.json."""
    from report import END, METRICS_PATH, README_PATH, START, render

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    readme = README_PATH.read_text(encoding="utf-8")
    block = START + readme.split(START, 1)[1].split(END, 1)[0] + END
    assert block == render(metrics)
