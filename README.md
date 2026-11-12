<div align="center">

# 🔍 RAGNova

**Question answering over your own documents, with hybrid retrieval, cross-encoder reranking, citations and a measured evaluation.**

[![Python](https://img.shields.io/badge/Python-3.11%20|%203.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![CI](https://github.com/aarohi1822/RagNova/actions/workflows/ci.yml/badge.svg)](https://github.com/aarohi1822/RagNova/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[Live demo](https://ragnova-aarohi.streamlit.app) · [Results](#results) · [How to run](#how-to-run) · [Limitations](#limitations)

</div>

> The demo runs on Streamlit Community Cloud, which puts idle apps to sleep. If you see a
> "wake up" screen, click it and wait about a minute for the app to start.

---

## Problem

General-purpose LLMs answer from what they memorised, not from *your* documents, and they
present unsupported claims as fluently as supported ones. A document Q&A system has to
(1) find the right passages in a private corpus, (2) answer only from them, (3) show where each
answer came from, and (4) let you measure whether the first three are actually happening.
Many RAG demos stop at (1) and (2) and never test (4).

## Approach

```
                              User question
                                    │
                         ┌──────────▼──────────┐
                         │   Query rewriter    │  ← LLM rewrites the question for retrieval
                         └──────────┬──────────┘
               ┌────────────────────┴────────────────────┐
               │ evidence path                           │ answer path
  ┌────────────▼─────────────┐              ┌────────────▼─────────────┐
  │  FAISS + BM25 candidates │              │  FAISS top-3 retrieval   │
  │            │             │              │  (LangChain conversation │
  │  Cross-encoder rerank    │              │   chain, 2-turn memory)  │
  │            │             │              └────────────┬─────────────┘
  │  Top-3 passages          │              ┌────────────▼─────────────┐
  └────────────┬─────────────┘              │  Llama 3.1 8B via Groq   │
               │                            └────────────┬─────────────┘
               │                                         │ answer
               └────────────────────┬────────────────────┘
                         ┌──────────▼──────────┐
                         │   Grounding check   │  ← word overlap: answer vs top-3 passages
                         └──────────┬──────────┘
                         ┌──────────▼──────────┐
                         │    LLM-as-judge     │  ← faithfulness + relevance scores
                         └──────────┬──────────┘
                                    ▼
           Answer + citations (top-3 passages) + confidence + scores
```

| Stage | Implementation |
|---|---|
| Ingestion | PDF, DOCX and TXT via LangChain loaders; 400-character chunks with 50-character overlap |
| Retrieval | FAISS (`all-MiniLM-L6-v2` embeddings) plus BM25 keyword search |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` picks the top-3 passages used for citations, grounding and judging |
| Generation | `llama-3.1-8b-instant` on Groq, via a LangChain conversation chain that does its own FAISS top-3 search, with a two-turn memory window |
| Grounding check | Word-overlap heuristic; replaces evasive or low-overlap answers with a warning |
| Evaluation | Per-answer faithfulness and relevance scored by an LLM judge; SQLite history; CSV export |
| Extras | Query rewriting, multi-document comparison, downloadable answers |

The app is a single Streamlit file, [`frontend/streamlit_app.py`](frontend/streamlit_app.py).
[`backend/`](backend/) holds an earlier FastAPI/Chroma version of the pipeline that the demo
does not use.

## Why each design decision?

**FAISS over a hosted vector DB (e.g. Pinecone)** → Runs locally, costs nothing, works offline.
An in-memory index is enough for the handful of documents a user uploads in one session.

**BM25 alongside semantic search** → Embeddings can miss exact terms such as names, section
numbers and acronyms; BM25 matches them directly. The benchmark reports Recall@5 separately for
keyword-style and paraphrased questions ([`eval/results/ablation.md`](eval/results/ablation.md)).
Getting the benefit requires a real fusion step such as Reciprocal Rank Fusion (see
[Limitations](#limitations)).

**Cross-encoder reranking** → Bi-encoder retrieval is fast but coarse. A cross-encoder reads the
question and each candidate together and scores them much more precisely. In the benchmark it
was the largest single improvement, at the cost of the highest latency.

**Llama 3.1 8B on Groq** → Fast inference and a free tier, so the demo runs without paid APIs.

**Two-turn memory window** → Follow-up questions still work, while prompts stay short, which
keeps each Groq call cheap and fast.

**LLM-as-judge** → Gives every answer a faithfulness and relevance score without labeled data.
Checked against human labels it did not separate faithful from unfaithful answers (see
[Results](#results)), so its scores are a signal to validate, not a guarantee.

**Offline, seeded benchmark** → Fixed corpus, gold passages and random seeds, so every number in
this README can be regenerated with one command and checked.

## Results

The retrieval stack and the LLM judge were evaluated offline on a fixed benchmark: 60 questions
with hand-checked gold passages over 16 public Wikipedia articles on privacy and compliance law.
The table below is generated from [`results/metrics.json`](results/metrics.json) by
[`eval/report.py`](eval/report.py); don't edit it by hand.

<!-- results:start -->
<!-- generated by eval/report.py from results/metrics.json; do not edit by hand -->

Retrieval on a 60-query benchmark (71 gold passages, 1425 passages from 16 public documents). Latency is retrieval only (no LLM), cpu, Darwin arm64, Python 3.12.4, seed 0, measured 2026-11-11.

| Configuration | Recall@5 [95% CI] | MRR@10 [95% CI] | p50 latency | p95 latency |
|---|---|---|---|---|
| BM25 only | 0.592 [0.47–0.71] | 0.495 [0.38–0.61] | 1.3 ms | 1.8 ms |
| FAISS only | 0.767 [0.66–0.87] | 0.613 [0.51–0.71] | 6.8 ms | 7.3 ms |
| Hybrid (RRF) | 0.758 [0.65–0.86] | 0.626 [0.52–0.73] | 9.0 ms | 9.4 ms |
| Hybrid + cross-encoder | 0.833 [0.73–0.92] | 0.732 [0.63–0.83] | 130.7 ms | 145.7 ms |
| App hybrid (concat, ref.) | 0.767 [0.66–0.87] | 0.613 [0.51–0.71] | 8.8 ms | 9.4 ms |

| LLM-judge reliability | n | Agreement | Cohen's κ [95% CI] | Judge yes-rate |
|---|---|---|---|---|
| human vs judge (app_truncated) | 26 | 0.65 | 0.00 [0.00–0.00] | 1.00 |
| human vs judge (full_context) | 26 | 0.65 | 0.00 [0.00–0.00] | 1.00 |
| claude vs judge (app_truncated) | 30 | 0.60 | 0.00 [0.00–0.00] | 1.00 |
| claude vs judge (full_context) | 30 | 0.60 | 0.00 [0.00–0.00] | 1.00 |
| human vs claude (inter-annotator) | 26 | 0.69 | 0.35 [-0.04–0.69] | – |

Judge: `local:Qwen/Qwen2.5-1.5B-Instruct`, threshold 0.5, 30 answers; 4 marked unsure by the human rater and excluded from human comparisons. Claude labels were made independently before the human labels.

Method, caveats and how to reproduce: [eval/README.md](eval/README.md).
<!-- results:end -->

**What the numbers say:**

- **Reranking is the only clear improvement.** The cross-encoder gives the best Recall@5 and MRR,
  at the cost of the highest latency. With 60 queries the confidence intervals overlap, so treat
  the other differences as indicative.
- **The app's current hybrid merge adds nothing.** It lists semantic hits first, so BM25 results
  never reach the top 5 and it scores exactly like FAISS alone. Reciprocal Rank Fusion (the
  "Hybrid (RRF)" row) is the proper way to combine the two.
- **The LLM judge does not detect unfaithful answers.** It rated every answer as faithful, so its
  agreement with human labels is no better than chance (see Cohen's κ above). The in-app faithfulness scores
  should not be read as a quality guarantee.

Method, labeling guidelines and caveats: [`eval/README.md`](eval/README.md).

## How to run

**The app** (Python 3.11 or 3.12):

```bash
git clone https://github.com/aarohi1822/RagNova.git
cd RagNova
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here      # free key at https://console.groq.com
streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501, upload documents in the sidebar, click **Ingest Documents**, then ask
questions. On Streamlit Community Cloud, set `GROQ_API_KEY` under the app's **Settings → Secrets**.

**The evaluation** (separate environment, no API key needed):

```bash
python -m venv .venv-eval && source .venv-eval/bin/activate
pip install -r requirements-eval.txt

python eval/ablation.py              # retrieval comparison  -> eval/results/ablation.md
python eval/judge_export.py          # 30 answers to label   -> eval/judge/labeling_sheet.csv
python eval/judge_agreement.py       # labels vs LLM judge   -> eval/results/judge_agreement.md
python eval/report.py                # -> results/metrics.json and the table above
```

All randomness is seeded (`--seed`, default 0). Retrieval quality numbers reproduce exactly;
latency varies slightly between runs and machines.

**Tests:** `pytest -q` (the same checks run in CI on every push).

## Limitations

- **Answers and citations come from different retrieval paths.** The answer is generated from a
  plain FAISS top-3 search inside the LangChain chain; the hybrid + reranked passages are only
  used for the citations, grounding check and judge. The reranker therefore does not influence the
  answer, and the cited passages may differ from the ones the model read. (The offline judge
  check generated answers from the reranked passages, i.e. the intended design.)
- **Small, single-domain benchmark.** 60 queries over 16 legal and regulatory articles. The
  queries and gold labels were written by one annotator, so results may not transfer to other
  domains or document types.
- **Judge reliability is measured on a proxy judge.** No API key was available offline, so the
  reliability check used a local `Qwen2.5-1.5B-Instruct`, not the app's Groq judge. Only 30
  answers were labeled, and the human–Claude inter-annotator agreement was only fair. Run `eval/judge_export.py` with `GROQ_API_KEY` set to test the production judge.
- **Hybrid retrieval as shipped is FAISS-only in practice** (see Results). Switching the app to RRF
  is a one-function change that has not been made yet.
- **The grounding check is a heuristic.** It counts overlapping words; it does not verify claims.
- **Latency figures cover retrieval only,** with warm indexes on one laptop CPU. End-to-end
  latency in the app also includes query rewriting, generation and judging (three LLM calls), and
  BM25 is rebuilt on every query.
- **In-session index.** Uploaded documents live in memory for the session; there is no persistent
  vector store in the demo.
- **Docker files are out of date.** The `Dockerfile` expects a `pyproject.toml` that is kept as
  `pyproject.local.toml` (renamed for Streamlit Cloud), so `docker-compose up` currently fails.

## Project layout

```
frontend/streamlit_app.py   the app
eval/                       benchmark, ablation, judge-reliability scripts and data
results/metrics.json        all reported numbers
backend/                    earlier FastAPI/Chroma implementation (not used by the demo)
tests/                      unit tests, run in CI
docs/                       architecture and deployment notes
```

## Author

**Aarohi Gaurav Sharma** · B.Tech CSE, COER University, Roorkee · IIT Roorkee Research Intern × 2 ·
[GitHub](https://github.com/aarohi1822)

Released under the [MIT License](LICENSE).
