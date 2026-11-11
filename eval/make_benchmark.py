"""Resolve eval/benchmark_spec.py into eval/benchmark.jsonl with gold passage ids.

Usage:
  python eval/make_benchmark.py            # write benchmark.jsonl
  python eval/make_benchmark.py --review   # also print every gold passage for manual review
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from benchmark_spec import QUERIES  # noqa: E402

PASSAGES_PATH = EVAL_DIR / "corpus" / "passages.jsonl"
BENCHMARK_PATH = EVAL_DIR / "benchmark.jsonl"


def load_passages() -> list[dict]:
    with PASSAGES_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def resolve(spec: dict, passages: list[dict]) -> list[str]:
    in_doc = [re.compile(p, re.IGNORECASE) for p in spec["evidence"]]
    anywhere = [re.compile(p, re.IGNORECASE) for p in spec.get("cross_doc_evidence", [])]
    gold = [
        p["id"] for p in passages
        if (p["doc"] == spec["doc"] and any(rx.search(p["text"]) for rx in in_doc))
        or any(rx.search(p["text"]) for rx in anywhere)
    ]
    gold = [g for g in gold if g not in spec.get("drop", [])] + list(spec.get("add", []))
    return list(dict.fromkeys(gold))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()

    passages = load_passages()
    by_id = {p["id"]: p for p in passages}
    records, problems = [], []
    for i, spec in enumerate(QUERIES, start=1):
        qid = f"q{i:03d}"
        gold = resolve(spec, passages)
        if not gold:
            problems.append(f"{qid}: no gold passage matched {spec['evidence']}")
        missing = [g for g in gold if g not in by_id]
        if missing:
            problems.append(f"{qid}: unknown passage ids {missing}")
        records.append({
            "qid": qid,
            "query": spec["query"],
            "type": spec["type"],
            "doc": spec["doc"],
            "gold_ids": gold,
            "answer": spec["answer"],
        })
        if args.review:
            print(f"\n[{qid}] ({spec['type']}) {spec['query']}\n  A: {spec['answer']}")
            for g in gold:
                print(f"  - {g}: {by_id[g]['text'][:220].replace(chr(10), ' ')}")

    if problems:
        sys.exit("benchmark problems:\n" + "\n".join(problems))

    with BENCHMARK_PATH.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_gold = sum(len(r["gold_ids"]) for r in records)
    types = {t: sum(r["type"] == t for r in records) for t in ("lexical", "paraphrase")}
    print(f"\nwrote {len(records)} queries, {n_gold} gold passages, types={types} "
          f"-> {BENCHMARK_PATH.name}")


if __name__ == "__main__":
    main()
