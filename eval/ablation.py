"""Retrieval ablation over eval/benchmark.jsonl.

Reports Recall@5, Hit@5, MRR@10 (with 95% bootstrap CIs) and per-query
retrieval latency p50/p95 for each configuration in retrievers.Retrievers.

Latency is wall-clock time for one retrieval call (query encoding + search
[+ rerank]) with indexes already built. Each query is timed `--repeats` times
after a warm-up and the per-query median is used, so p50/p95 are over queries.

  python eval/ablation.py [--device cpu|mps|cuda] [--repeats 3]
Writes eval/results/ablation.{json,md} and eval/results/ablation_runs.jsonl.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import date
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from eval_metrics import (  # noqa: E402
    bootstrap_ci,
    hit_at_k,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from retrievers import EMBED_MODEL, RERANK_MODEL, Retrievers  # noqa: E402
from seeds import DEFAULT_SEED, set_seeds  # noqa: E402

RESULTS_DIR = EVAL_DIR / "results"
K_RECALL, K_MRR = 5, 10


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def run(device: str, repeats: int, pool: int, seed: int = DEFAULT_SEED) -> dict:
    set_seeds(seed)
    passages = load_jsonl(EVAL_DIR / "corpus" / "passages.jsonl")
    bench = load_jsonl(EVAL_DIR / "benchmark.jsonl")
    print(f"building indexes over {len(passages)} passages on {device} ...", flush=True)
    retr = Retrievers(passages, device=device, pool=pool)

    results, runs = {}, []
    for name, fn in retr.configs().items():
        for q in bench[:5]:  # warm-up (model kernels, caches)
            fn(q["query"], K_MRR)
        recalls, hits, rrs, lat_ms = [], [], [], []
        by_type: dict[str, list[float]] = {}
        for q in bench:
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                ranked = fn(q["query"], K_MRR)
                times.append((time.perf_counter() - t0) * 1000)
            lat_ms.append(statistics.median(times))
            r = recall_at_k(ranked, q["gold_ids"], K_RECALL)
            recalls.append(r)
            hits.append(hit_at_k(ranked, q["gold_ids"], K_RECALL))
            rrs.append(reciprocal_rank(ranked, q["gold_ids"], K_MRR))
            by_type.setdefault(q["type"], []).append(r)
            runs.append({"config": name, "qid": q["qid"], "ranked": ranked,
                         "recall@5": r, "rr@10": rrs[-1], "latency_ms": round(lat_ms[-1], 2)})
        results[name] = {
            "recall@5": mean(recalls), "recall@5_ci": bootstrap_ci(recalls, seed=seed),
            "hit@5": mean(hits),
            "mrr@10": mean(rrs), "mrr@10_ci": bootstrap_ci(rrs, seed=seed),
            "latency_p50_ms": percentile(lat_ms, 50), "latency_p95_ms": percentile(lat_ms, 95),
            "recall@5_by_type": {t: mean(v) for t, v in sorted(by_type.items())},
        }
        print(f"  {name:<28} R@5={results[name]['recall@5']:.3f} "
              f"MRR={results[name]['mrr@10']:.3f} p50={results[name]['latency_p50_ms']:.1f}ms",
              flush=True)

    meta = {
        "date": date.today().isoformat(),
        "queries": len(bench),
        "query_types": {t: sum(q["type"] == t for q in bench) for t in ("lexical", "paraphrase")},
        "passages": len(passages),
        "embedding_model": EMBED_MODEL, "reranker_model": RERANK_MODEL,
        "candidate_pool": pool, "device": device, "repeats": repeats, "seed": seed,
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
    }
    return {"meta": meta, "results": results, "runs": runs}


def to_markdown(out: dict) -> str:
    m, res = out["meta"], out["results"]
    lines = [
        "# Retrieval ablation",
        "",
        f"{m['queries']} queries ({m['query_types']['lexical']} lexical, "
        f"{m['query_types']['paraphrase']} paraphrase) over {m['passages']} passages. "
        f"Run {m['date']} on {m['machine']}, device={m['device']}.",
        "",
        "| Configuration | Recall@5 [95% CI] | Hit@5 | MRR@10 [95% CI] | p50 ms | p95 ms "
        "| R@5 lexical | R@5 paraphrase |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res.items():
        lo, hi = r["recall@5_ci"]
        mlo, mhi = r["mrr@10_ci"]
        bt = r["recall@5_by_type"]
        lines.append(
            f"| {name} | {r['recall@5']:.3f} [{lo:.2f}–{hi:.2f}] | {r['hit@5']:.3f} "
            f"| {r['mrr@10']:.3f} [{mlo:.2f}–{mhi:.2f}] | {r['latency_p50_ms']:.1f} "
            f"| {r['latency_p95_ms']:.1f} | {bt.get('lexical', 0):.3f} "
            f"| {bt.get('paraphrase', 0):.3f} |"
        )
    lines += [
        "",
        f"- Embeddings: `{m['embedding_model']}` (FAISS flat L2). "
        f"Reranker: `{m['reranker_model']}`.",
        f"- Hybrid = Reciprocal Rank Fusion (k=60) of the top-{m['candidate_pool']} BM25 and FAISS "
        f"hits; the cross-encoder reranks those top-{m['candidate_pool']} fused candidates.",
        "- *App hybrid (concat, ref.)* is the merge `frontend/streamlit_app.py` uses today "
        "(semantic hits, then BM25 hits, deduped).",
        "- Recall@5 = share of a query's gold passages in the top 5, averaged over queries. "
        "MRR@10 = mean reciprocal rank of the first gold passage within the top 10.",
        f"- Latency = one retrieval call with indexes built (median of {m['repeats']} runs per "
        "query); excludes LLM generation. "
        "CIs are percentile bootstrap over queries (2,000 resamples).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pool", type=int, default=20, help="candidates fused/reranked")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    out = run(args.device, args.repeats, args.pool, args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "ablation.json").write_text(
        json.dumps({"meta": out["meta"], "results": out["results"]}, indent=2) + "\n")
    with (RESULTS_DIR / "ablation_runs.jsonl").open("w") as fh:
        for r in out["runs"]:
            fh.write(json.dumps(r) + "\n")
    md = to_markdown(out)
    (RESULTS_DIR / "ablation.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
