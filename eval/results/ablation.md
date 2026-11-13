# Retrieval ablation

60 queries (18 lexical, 42 paraphrase) over 1425 passages. Run 2026-11-13 on Darwin arm64, Python 3.12.4, device=cpu.

| Configuration | Recall@5 [95% CI] | Hit@5 | MRR@10 [95% CI] | p50 ms | p95 ms | R@5 lexical | R@5 paraphrase |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.592 [0.47–0.71] | 0.600 | 0.495 [0.38–0.61] | 1.6 | 2.8 | 0.833 | 0.488 |
| FAISS only | 0.767 [0.66–0.87] | 0.783 | 0.613 [0.51–0.71] | 7.9 | 9.0 | 0.889 | 0.714 |
| Hybrid (RRF) | 0.758 [0.65–0.86] | 0.783 | 0.626 [0.52–0.73] | 10.8 | 13.8 | 0.917 | 0.690 |
| Hybrid + cross-encoder | 0.833 [0.73–0.92] | 0.850 | 0.732 [0.63–0.83] | 162.6 | 222.0 | 0.972 | 0.774 |
| Concat merge (previous app) | 0.767 [0.66–0.87] | 0.783 | 0.613 [0.51–0.71] | 10.7 | 11.8 | 0.889 | 0.714 |
| App pipeline (frontend code) | 0.833 [0.73–0.92] | 0.850 | 0.732 [0.63–0.83] | 153.1 | 215.6 | 0.972 | 0.774 |

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (FAISS flat L2). Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Hybrid = Reciprocal Rank Fusion (k=60) of the top-20 BM25 and FAISS hits; the cross-encoder reranks those top-20 fused candidates.
- *App hybrid (concat, ref.)* is the merge `frontend/streamlit_app.py` uses today (semantic hits, then BM25 hits, deduped).
- Recall@5 = share of a query's gold passages in the top 5, averaged over queries. MRR@10 = mean reciprocal rank of the first gold passage within the top 10.
- Latency = one retrieval call with indexes built (median of 3 runs per query); excludes LLM generation. CIs are percentile bootstrap over queries (2,000 resamples).
