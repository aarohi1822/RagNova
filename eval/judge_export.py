"""Export 30 RAG answers for manual faithfulness labeling, plus LLM-judge scores.

For 30 benchmark queries (every second one, so all 16 documents are covered):
  1. retrieve the top-3 passages with hybrid + cross-encoder (as the app does),
  2. generate an answer with the app's QA prompt,
  3. score faithfulness with the app's LLM-judge prompt, in two variants:
       app_truncated - exactly what frontend/streamlit_app.py sends: the first
                       300 characters of each of the 3 chunks;
       full_context  - same prompt with the full chunks the generator saw.

Outputs (in eval/judge/):
  labeling_sheet.csv   - blind sheet for you: question, context, answer, and an
                         empty `faithful` column (yes/no). No judge scores.
  judge_outputs.jsonl  - judge scores, kept separate so labeling stays blind.

  python eval/judge_export.py [--backend auto|groq|local] [--force]
Refuses to overwrite a sheet that already has labels unless --force is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent / "frontend"))

from llm import get_llm  # noqa: E402
from seeds import DEFAULT_SEED, set_seeds  # noqa: E402
from ragnova_retrieval import build_answer_prompt  # noqa: E402

JUDGE_DIR = EVAL_DIR / "judge"
SHEET_PATH = JUDGE_DIR / "labeling_sheet.csv"
JUDGE_PATH = JUDGE_DIR / "judge_outputs.jsonl"
PARTIAL_PATH = JUDGE_DIR / ".partial.jsonl"  # resume checkpoint, deleted on success
N_ITEMS = 30
TOP_K = 3

# Verbatim from evaluate_answer() in frontend/streamlit_app.py.
JUDGE_PROMPT = """You are an evaluation judge. Score the following RAG answer.

Question: {question}
Context chunks: {context}
Answer: {answer}

Return ONLY a JSON object with these two keys:
- "faithfulness": float 0-1 (is the answer supported by the context?)
- "relevance": float 0-1 (does the answer address the question?)

Example: {{"faithfulness": 0.9, "relevance": 0.85}}"""


def parse_judge(text: str) -> float | None:
    """Same parsing as the app, plus a regex fallback; None if unparseable."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return float(json.loads(cleaned).get("faithfulness"))
    except (ValueError, TypeError, AttributeError):
        match = re.search(r'"faithfulness"\s*:\s*([0-9.]+)', cleaned)
        return float(match.group(1)) if match else None


def sheet_has_labels() -> bool:
    if not SHEET_PATH.exists():
        return False
    with SHEET_PATH.open(newline="", encoding="utf-8") as fh:
        return any(row.get("faithful", "").strip() for row in csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="auto", choices=["auto", "groq", "local"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    set_seeds(args.seed)  # decoding is greedy; seeding guards any sampling paths

    if sheet_has_labels() and not args.force:
        sys.exit(f"{SHEET_PATH} already contains labels; refusing to overwrite (use --force).")

    passages = [json.loads(line) for line in (EVAL_DIR / "corpus/passages.jsonl").open()]
    text_of = {p["id"]: p["text"] for p in passages}
    bench = [json.loads(line) for line in (EVAL_DIR / "benchmark.jsonl").open()]
    items = bench[::2][:N_ITEMS]
    JUDGE_DIR.mkdir(exist_ok=True)

    # Stage 1: retrieval for every item, then free the retrieval models so the
    # LLM fits in memory on small machines (8 GB laptops swap badly otherwise).
    import gc

    from retrievers import Retrievers  # heavy import, after the safety check

    retr = Retrievers(passages, device="cpu", app_pipeline=False)
    retrieved = {q["qid"]: retr.hybrid_rerank(q["query"], TOP_K) for q in items}
    del retr
    gc.collect()

    # Stage 2: generation + judging, checkpointed per item so a crash can resume.
    done = {}
    if PARTIAL_PATH.exists():
        done = {r["qid"]: r for r in map(json.loads, PARTIAL_PATH.open(encoding="utf-8"))}
    llm = get_llm(args.backend)
    print(f"generator + judge backend: {llm.name} (resuming {len(done)} items)", flush=True)

    for q in items:
        if q["qid"] in done and done[q["qid"]]["backend"] == llm.name:
            continue
        chunks = [text_of[i] for i in retrieved[q["qid"]]]
        answer = llm.complete(build_answer_prompt(q["query"], chunks))  # same prompt as the app
        scores, raws = {}, {}
        for variant, judge_ctx in {
            "app_truncated": "\n".join(c[:300] for c in chunks),
            "full_context": "\n".join(chunks),
        }.items():
            raw = llm.complete(JUDGE_PROMPT.format(question=q["query"], context=judge_ctx,
                                                   answer=answer), max_tokens=64)
            raws[variant], scores[variant] = raw, parse_judge(raw)
        done[q["qid"]] = {"qid": q["qid"], "answer": answer, "backend": llm.name,
                          "faithfulness": scores, "raw": raws}
        with PARTIAL_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(done[q["qid"]], ensure_ascii=False) + "\n")
        print(f"  {q['qid']} judge={scores}", flush=True)

    sheet_rows, judge_rows = [], []
    for n, q in enumerate(items, start=1):
        item_id, ids, r = f"j{n:02d}", retrieved[q["qid"]], done[q["qid"]]
        sheet_rows.append({
            "item_id": item_id, "qid": q["qid"], "question": q["query"],
            "context": "\n\n---\n\n".join(f"[{i}] {text_of[i]}" for i in ids),
            "answer": r["answer"], "faithful": "", "notes": "",
        })
        judge_rows.append({
            "item_id": item_id, "qid": q["qid"], "retrieved_ids": ids,
            "gold_retrieved": bool(set(ids) & set(q["gold_ids"])),
            "backend": r["backend"], "seed": args.seed,
            "faithfulness": r["faithfulness"], "raw": r["raw"],
        })

    with SHEET_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sheet_rows[0]))
        writer.writeheader()
        writer.writerows(sheet_rows)
    with JUDGE_PATH.open("w", encoding="utf-8") as fh:
        for r in judge_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    PARTIAL_PATH.unlink(missing_ok=True)
    print(f"\nwrote {len(sheet_rows)} items -> {SHEET_PATH.relative_to(EVAL_DIR.parent)} "
          f"(fill the `faithful` column with yes/no), judge scores -> {JUDGE_PATH.name}")


if __name__ == "__main__":
    main()
