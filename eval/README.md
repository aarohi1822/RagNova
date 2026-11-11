# Retrieval & judge evaluation

Offline, reproducible evaluation for RAGNova's retrieval stack and its LLM-as-judge.

```
eval/
  corpus/docs/*.txt        16 Wikipedia articles on privacy/compliance law (CC BY-SA 4.0, pinned revisions)
  corpus/sources.json      title, revision id, URL, license for each article
  corpus/passages.jsonl    1,425 frozen passages (id = "<doc>::<nnnn>"), chunked like the app (400/50)
  benchmark_spec.py        queries + evidence patterns (source of truth)
  benchmark.jsonl          60 queries with gold passage ids (generated; do not hand-edit)
  ablation.py              retrieval ablation -> results/ablation.{md,json}
  judge_export.py          30 answers -> judge/labeling_sheet.csv (+ judge/judge_outputs.jsonl)
  judge_agreement.py       your labels vs the LLM judge -> results/judge_agreement.{md,json}
  report.py                collects everything -> ../results/metrics.json + README results table
  seeds.py                 fixes python/numpy/torch/transformers seeds (default 0, --seed to change)
```

All randomness is seeded (default `--seed 0`): bootstrap resampling, torch/transformers, numpy.
Retrieval and greedy decoding are deterministic, so quality metrics reproduce exactly. Latency is
wall-clock and varies slightly from run to run.

After running the experiments, run `python eval/report.py`. It writes `results/metrics.json` and
regenerates the results table in the top-level README from that file. Don't edit that table by
hand; a test checks that it matches `metrics.json`.

Setup: `python -m venv .venv && .venv/bin/pip install -r requirements-eval.txt`

## 1. Benchmark

`benchmark.jsonl` has one record per line:

```json
{"qid": "q001", "query": "...", "type": "paraphrase", "doc": "ccpa",
 "gold_ids": ["ccpa::0006"], "answer": "reference answer"}
```

- **60 queries** across all 16 documents: 18 *lexical* (they share names, numbers or section ids
  with the passage) and 42 *paraphrase* (reworded, with little term overlap).
- **Gold ids** are every passage in the query's document that matches the query's evidence regex,
  plus manual `add`/`drop` overrides. All 71 gold passages were reviewed by reading them
  (`python eval/make_benchmark.py --review` prints them).
- To change a query, edit `benchmark_spec.py`, then run `python eval/make_benchmark.py`.
  `tests/test_eval_benchmark.py` fails if the two files drift apart.
- Passage ids depend on the chunker. If you re-chunk (`build_corpus.py`), gold ids must be
  re-resolved; `--fetch` re-downloads the current article revisions and changes the text.

## 2. Retrieval ablation

```
python eval/ablation.py            # CPU; --device mps to use Apple GPU
```

Configurations: BM25 only, FAISS only, Hybrid (RRF), Hybrid + cross-encoder rerank, plus the app's
current concat-merge for reference. The script reports Recall@5, Hit@5 and MRR@10 with bootstrap
95% CIs, and p50/p95 retrieval latency. Results go to `results/ablation.md`.

## 3. Judge reliability

```
python eval/judge_export.py        # uses GROQ_API_KEY if set, else local Qwen2.5-1.5B-Instruct
# label judge/labeling_sheet.csv
python eval/judge_agreement.py
```

The labeling sheet is **blind**: judge scores are kept in `judge/judge_outputs.jsonl`. Re-running
the export will not overwrite a sheet that already has labels (unless you pass `--force`).

### Labeling guidelines

Fill the `faithful` column with `yes` or `no` for every row (`notes` is optional).

- **yes**: every factual claim in the answer is supported by the `context` shown in that row.
  An answer that correctly says "I don't know" when the context lacks the answer is also `yes`.
- **no**: the answer states anything that the context does not support, or that contradicts it.
  This includes numbers, dates, names, or details added from outside knowledge, even if they
  happen to be true.
- Judge only against the context. Don't judge against the reference answer or against what you
  know about the law.
- Ignore style, length and whether the answer is complete. Faithfulness is not relevance.

The agreement script reports raw agreement, Cohen's κ with a bootstrap CI, a confusion matrix, a
sweep over the judge's yes/no threshold, and the list of disagreements. It does this for two
judge variants:

- `app_truncated`: exactly what `frontend/streamlit_app.py` does. The judge sees only the first
  300 characters of each chunk.
- `full_context`: the same prompt, but the judge sees the full chunks the generator used.

## Caveats

- The queries and gold labels were written by one annotator (Claude, reviewed by reading the
  passages), not by independent annotators. The lexical/paraphrase split is an authoring
  choice, not a measured property.
- 60 queries give Recall@5 CIs about ±0.1 wide. Overlapping CIs mean a difference is not
  significant.
- Latency covers retrieval only, on one machine, with warm indexes. The app currently rebuilds
  BM25 on every query and adds LLM calls, so end-to-end latency in the app is much higher.
- The default local backend uses the same small model to generate answers and to judge them
  (self-evaluation). For the number that matters in production, run the export with
  `GROQ_API_KEY` set so the judge is the app's `llama-3.1-8b-instant`.
