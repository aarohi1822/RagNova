<div align="center">

# 🔍 RAGNova
### AI Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=for-the-badge)](https://groq.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467DF?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Production-grade Retrieval-Augmented Generation system with hybrid retrieval, cross-encoder reranking, LLM-as-judge evaluation, and persistent session history.**

[🚀 Live Demo](https://ragnova-aarohi.streamlit.app) · [📖 Docs](#architecture) · [⭐ Star this repo](#)

</div>

---

## ✨ What makes RAGNova different?

Most RAG projects are basic PDF chatbots. RAGNova is built like a real production system:

```
User Query
    │
    ▼
┌─────────────────────┐
│   Query Rewriter    │  ← LLM rewrites question for better retrieval
└────────┬────────────┘
         │
    ┌────▼─────────────────────────────┐
    │         Hybrid Retrieval         │
    │  ┌──────────────┐  ┌──────────┐ │
    │  │ FAISS Semantic│  │ BM25 KW  │ │  ← Two retrieval signals merged
    │  └──────────────┘  └──────────┘ │
    └────────────────┬─────────────────┘
                     │
         ┌───────────▼───────────┐
         │  Cross-Encoder Rerank │  ← ms-marco MiniLM scores each chunk
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Grounding Check     │  ← Hallucination filter
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  LLaMA 3.1 via Groq   │  ← Fast inference, zero cost
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  LLM-as-Judge Eval    │  ← Faithfulness + Relevance scoring
         └───────────┬───────────┘
                     │
                  Answer + Citations + Scores
```

---

## 🏗️ Architecture

```
ragnova/
├── frontend/
│   └── streamlit_app.py        # Full app — retrieval, reranking, eval, UI
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── retrieval/
│   │   │   ├── hybrid.py       # BM25 + FAISS fusion
│   │   │   ├── reranker.py     # Cross-encoder reranking
│   │   │   └── query_rewriter.py
│   │   ├── services/
│   │   │   ├── ingestion.py    # PDF/DOCX/TXT parser
│   │   │   ├── chunking.py     # Recursive text splitting
│   │   │   └── evaluation.py  # Faithfulness + relevance scoring
│   │   └── storage/
│   │       └── chroma_store.py
├── tests/
│   ├── test_chunking.py
│   └── test_evaluation.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── IMPLEMENTATION_GUIDE.md
│   └── DEPLOYMENT.md
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

---

## 🚀 Features

| Feature | Description | Status |
|---|---|---|
| 📄 Multi-format ingestion | PDF, DOCX, TXT with metadata extraction | ✅ |
| 🔀 Hybrid retrieval | FAISS semantic + BM25 keyword fusion | ✅ |
| 🎯 Cross-encoder reranking | `ms-marco-MiniLM-L-6-v2` scores each chunk | ✅ |
| 🔁 Query rewriting | LLM rewrites question before retrieval | ✅ |
| 🛡️ Hallucination guard | Grounding check — no fake answers | ✅ |
| 📊 LLM-as-judge eval | Per-query faithfulness + relevance scoring | ✅ |
| 💬 Session memory | Windowed conversation buffer (k=2) | ✅ |
| 🗂️ Multi-doc comparison | Compare N documents on any question | ✅ |
| 💾 Persistent history | SQLite-backed session storage | ✅ |
| 🔢 Confidence scores | Per-chunk reranker score shown in UI | ✅ |
| ⬇️ Export | Download answers + eval reports as TXT/CSV | ✅ |
| 🐳 Docker ready | Full `docker-compose.yml` included | ✅ |

---

## 📊 Evaluation Dashboard

RAGNova tracks every query automatically:

```
┌──────────────────────────────────────────────────────┐
│              RAG Evaluation Dashboard                │
├──────────────┬──────────────┬──────────┬─────────────┤
│ Avg Faith.   │ Avg Relevance│ Avg Lat. │ Total Qs    │
│    0.40      │    0.70      │  621 ms  │     2       │
├──────────────┴──────────────┴──────────┴─────────────┤
│ Timestamp           │ Question           │ F    │ R  │
│ 2026-05-05T06:51:24 │ history of AI...   │ 0.4  │0.7 │
│ 2026-05-05T06:48:03 │ history of AI...   │ 0.4  │0.7 │
└──────────────────────────────────────────────────────┘
                         ↓
              [ Download as CSV ]
```

---

## ⚡ Tech Stack

```
┌─────────────────────────────────────────────────────┐
│                    RAGNova Stack                    │
├─────────────────┬───────────────────────────────────┤
│ LLM Inference   │ Groq · LLaMA 3.1 8B Instant       │
│ Embeddings      │ HuggingFace · all-MiniLM-L6-v2    │
│ Vector Store    │ FAISS (local, zero cost)           │
│ Keyword Search  │ BM25Okapi (rank-bm25)              │
│ Reranker        │ CrossEncoder ms-marco-MiniLM-L-6   │
│ Frontend        │ Streamlit                          │
│ Backend         │ FastAPI                            │
│ Persistence     │ SQLite                             │
│ Containerized   │ Docker + docker-compose            │
│ CI/CD           │ GitHub Actions                     │
└─────────────────┴───────────────────────────────────┘
```

**Cost to run: ~$0** — No OpenAI, no Pinecone, no paid APIs.

---

## 🛠️ Quick Start

```bash
# 1. Clone
git clone https://github.com/aarohi1822/RagNova.git
cd RagNova

# 2. Install
pip install -e ".[dev]"

# 3. Set your key
cp .env.example .env
# Add GROQ_API_KEY to .env

# 4. Run
streamlit run frontend/streamlit_app.py
```

**Or with Docker:**
```bash
docker-compose up --build
```

---

## 🔑 Environment Variables

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

---

## 📦 Dependencies

```toml
# Core
langchain
langchain-groq
langchain-community
faiss-cpu
sentence-transformers   # CrossEncoder reranker
rank-bm25               # Hybrid retrieval
streamlit

# Loaders
pypdf
docx2txt

# Eval & persistence
pandas
numpy
sqlite3 (stdlib)
```

---

## 🧠 Why each design decision?

**FAISS over Pinecone** → Local, no latency, no cost, works offline. Good for <1M vectors.

**BM25 + Semantic fusion** → Semantic alone misses exact keyword matches. Hybrid catches both.

**Cross-encoder reranking** → Bi-encoder retrieval is fast but imprecise. CrossEncoder re-scores top-N with full query-chunk attention for higher precision.

**Windowed memory (k=2)** → Prevents context overflow in Groq's 8K token limit while keeping conversational continuity.

**LLM-as-judge** → Automated eval without labeled datasets. Scales to any domain.

---

## 📈 Roadmap

- [ ] GraphRAG — entity extraction + chunk linking
- [ ] Streamlit Cloud → HuggingFace Spaces migration
- [ ] Domain specialization (legal / medical / research)
- [ ] RAGAS integration for formal benchmarking
- [ ] REST API via FastAPI backend

---

## 👩‍💻 Author

**Aarohi Gaurav Sharma**
B.Tech CSE · COER University, Roorkee
IIT Roorkee Research Intern × 2

[![GitHub](https://img.shields.io/badge/GitHub-aarohi1822-181717?style=flat&logo=github)](https://github.com/aarohi1822)

---

<div align="center">

**If this helped you, drop a ⭐ — it keeps the project alive.**

</div>
