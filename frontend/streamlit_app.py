import os
import io
import time
import json
import sqlite3
import tempfile
import datetime
import numpy as np
import streamlit as st

from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.schema import Document

# Optional: BM25 for hybrid retrieval
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Optional: Cross-encoder reranker
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(page_title="RAGNova", layout="wide", page_icon="🔍")
st.title("🔍 RAGNova")
st.caption("AI Document Intelligence Platform — Hybrid Retrieval · Reranking · Citations · Evaluation")

# ── API Key Setup ───────────────────────────────────────────
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("Please add GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# ── SQLite Session Persistence ───────────────────────────────
DB_PATH = "/tmp/ragnova_sessions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            citations TEXT,
            latency_ms INTEGER,
            faithfulness_score REAL,
            relevance_score REAL
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(question, answer, citations, latency_ms, faithfulness, relevance):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.datetime.now().isoformat(),
            question,
            answer,
            json.dumps(citations),
            latency_ms,
            faithfulness,
            relevance,
        ),
    )
    conn.commit()
    conn.close()

def load_from_db(limit=50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, question, answer, citations, latency_ms, faithfulness_score, relevance_score "
        "FROM sessions ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows

init_db()

# ── Session State ───────────────────────────────────────────
defaults = {
    "vectorstore": None,
    "all_chunks": [],
    "chat_history": [],
    "eval_log": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=2,
    )

# ── Cached Resources ────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_reranker():
    if RERANKER_AVAILABLE:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return None

@st.cache_resource
def get_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=GROQ_API_KEY,
    )

# ── File Loader ─────────────────────────────────────────────
def load_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            return []
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_name"] = uploaded_file.name
        return docs
    except Exception as e:
        st.warning(f"Failed to load {uploaded_file.name}: {str(e)}")
        return []

# ── Vectorstore + BM25 Builder ──────────────────────────────
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    st.session_state.all_chunks = chunks
    return FAISS.from_documents(chunks, get_embeddings())

# ── Hybrid Retrieval ────────────────────────────────────────
def hybrid_retrieve(query: str, k: int = 4):
    vectorstore = st.session_state.vectorstore
    chunks = st.session_state.all_chunks

    semantic_docs = vectorstore.similarity_search(query, k=k)

    if not BM25_AVAILABLE or not chunks:
        return semantic_docs

    tokenized = [c.page_content.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]
    bm25_docs = [chunks[i] for i in top_idx]

    seen = set()
    merged = []
    for doc in semantic_docs + bm25_docs:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged[:k + 2]

# ── Reranker ────────────────────────────────────────────────
def rerank(query: str, docs, top_k: int = 3):
    reranker = get_reranker()
    if reranker is None or not docs:
        return docs[:top_k], [1.0] * min(top_k, len(docs))

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs).tolist()
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [d for _, d in ranked[:top_k]]
    top_scores = [s for s, _ in ranked[:top_k]]

    max_s, min_s = max(top_scores), min(top_scores)
    if max_s != min_s:
        top_scores = [(s - min_s) / (max_s - min_s) for s in top_scores]
    else:
        top_scores = [1.0] * len(top_scores)

    return top_docs, top_scores

# ── No-Answer Grounding Check ────────────────────────────────
def is_grounded(answer: str, chunks) -> bool:
    evasion_phrases = [
        "i don't know", "i cannot find", "not mentioned",
        "no information", "not provided", "cannot answer", "outside the scope",
    ]
    ans_lower = answer.lower()
    if any(p in ans_lower for p in evasion_phrases):
        return False
    all_text = " ".join(c.page_content.lower() for c in chunks)
    answer_words = set(ans_lower.split())
    chunk_words = set(all_text.split())
    return len(answer_words & chunk_words) > 5

# ── Query Rewriter ───────────────────────────────────────────
def rewrite_query(question: str) -> str:
    llm = get_llm()
    prompt = (
        "Rewrite this question to be more specific and retrieval-friendly for a document QA system. "
        f"Return only the rewritten question, nothing else.\n\nOriginal: {question}"
    )
    try:
        result = llm.invoke(prompt)
        return result.content.strip()
    except Exception:
        return question

# ── LLM-as-Judge Evaluation ──────────────────────────────────
def evaluate_answer(question: str, answer: str, chunks) -> dict:
    llm = get_llm()
    context = "\n".join(c.page_content[:300] for c in chunks[:3])
    prompt = f"""You are an evaluation judge. Score the following RAG answer.

Question: {question}
Context chunks: {context}
Answer: {answer}

Return ONLY a JSON object with these two keys:
- "faithfulness": float 0-1 (is the answer supported by the context?)
- "relevance": float 0-1 (does the answer address the question?)

Example: {{"faithfulness": 0.9, "relevance": 0.85}}"""
    try:
        result = llm.invoke(prompt)
        text = result.content.strip().replace("```json", "").replace("```", "").strip()
        scores = json.loads(text)
        return {
            "faithfulness": round(float(scores.get("faithfulness", 0.0)), 2),
            "relevance": round(float(scores.get("relevance", 0.0)), 2),
        }
    except Exception:
        return {"faithfulness": 0.0, "relevance": 0.0}

# ── Multi-Doc Comparison ─────────────────────────────────────
def compare_docs(question: str, doc_names: list) -> str:
    llm = get_llm()
    chunks = st.session_state.all_chunks
    per_doc = {}
    for name in doc_names:
        relevant = [c for c in chunks if c.metadata.get("source_name") == name][:3]
        per_doc[name] = "\n".join(c.page_content[:300] for c in relevant)
    context_block = "\n\n".join(f"=== {name} ===\n{text}" for name, text in per_doc.items())
    prompt = (
        f"Compare the following documents on this question: '{question}'\n\n"
        f"{context_block}\n\n"
        "Give a structured comparison with clear differences and similarities."
    )
    try:
        result = llm.invoke(prompt)
        return result.content.strip()
    except Exception as e:
        return f"Comparison failed: {str(e)}"

# ── Citation Formatter ───────────────────────────────────────
def format_citations(source_docs, scores=None) -> list:
    seen = set()
    citations = []
    for i, doc in enumerate(source_docs):
        excerpt = doc.page_content.strip()[:200]
        key = excerpt[:80]
        if key in seen:
            continue
        seen.add(key)
        score = scores[i] if scores and i < len(scores) else None
        citations.append({
            "chunk_id": f"chunk-{i+1}",
            "source_name": doc.metadata.get("source_name", doc.metadata.get("source", "unknown")),
            "page": doc.metadata.get("page", "—"),
            "excerpt": excerpt,
            "confidence": round(score, 2) if score is not None else "—",
        })
    return citations

# ── Download Text Builder ────────────────────────────────────
def build_download_text(question, answer, citations, faithfulness, relevance, latency_ms):
    lines = [
        "RAGNova — Answer Export",
        "=" * 40,
        f"Question: {question}",
        f"Latency: {latency_ms}ms | Faithfulness: {faithfulness} | Relevance: {relevance}",
        "",
        "Answer:",
        answer,
        "",
        "Citations:",
    ]
    for c in citations:
        lines.append(f"- [{c['chunk_id']}] {c['source_name']} | page {c['page']} | confidence {c['confidence']}")
        lines.append(f"  {c['excerpt']}")
        lines.append("")
    return "\n".join(lines)

# ── QA Chain ────────────────────────────────────────────────
def get_chain(retriever):
    return ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        output_key="answer",
    )

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Admin Panel")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) ready.")

    if st.button("Ingest Documents", use_container_width=True):
        if not uploaded_files:
            st.error("No files uploaded.")
        else:
            with st.spinner("Indexing..."):
                all_docs = []
                for file in uploaded_files:
                    docs = load_file(file)
                    if docs:
                        all_docs.extend(docs)
                if not all_docs:
                    st.error("No readable content extracted.")
                else:
                    try:
                        st.session_state.vectorstore = build_vectorstore(all_docs)
                        st.session_state.memory.clear()
                        st.session_state.chat_history = []
                        st.session_state.eval_log = []
                        st.success(f"✅ Indexed {len(uploaded_files)} file(s) → {len(st.session_state.all_chunks)} chunks")
                    except Exception as e:
                        st.error(f"Ingestion failed: {str(e)}")

    if st.session_state.vectorstore:
        st.info("✅ Vectorstore ready")
        doc_names = list({c.metadata.get("source_name", "") for c in st.session_state.all_chunks})
        st.caption(f"Docs: {', '.join(doc_names)}")
    else:
        st.warning("No documents ingested yet.")

    st.divider()
    st.subheader("🛠 Features")
    use_query_rewrite = st.toggle("Query Rewriting", value=True)
    use_hybrid = st.toggle("Hybrid Retrieval (BM25 + Semantic)", value=BM25_AVAILABLE)
    use_reranker = st.toggle("Cross-Encoder Reranking", value=RERANKER_AVAILABLE)
    use_eval = st.toggle("LLM-as-Judge Evaluation", value=True)

    st.divider()
    st.caption("Groq · HuggingFace · FAISS · BM25 · CrossEncoder")

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask", "🔀 Compare Docs", "📊 Evaluation", "🕓 History"])

# ── TAB 1: Ask ───────────────────────────────────────────────
with tab1:
    question = st.text_area("Ask a question from your uploaded documents", height=80)

    if st.button("Generate Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        elif st.session_state.vectorstore is None:
            st.error("Please ingest documents first.")
        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    # Step 1: Query rewriting
                    final_query = rewrite_query(question) if use_query_rewrite else question
                    if use_query_rewrite and final_query != question:
                        st.caption(f"🔁 Rewritten query: *{final_query}*")

                    # Step 2: Hybrid retrieval
                    if use_hybrid:
                        candidate_docs = hybrid_retrieve(final_query, k=6)
                    else:
                        candidate_docs = st.session_state.vectorstore.similarity_search(final_query, k=4)

                    # Step 3: Reranking
                    if use_reranker:
                        ranked_docs, conf_scores = rerank(final_query, candidate_docs, top_k=3)
                    else:
                        ranked_docs = candidate_docs[:3]
                        conf_scores = [1.0] * len(ranked_docs)

                    # Step 4: Generation
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    chain = get_chain(retriever)
                    start_time = time.time()
                    result = chain.invoke({"question": final_query})
                    latency_ms = int((time.time() - start_time) * 1000)
                    answer = result["answer"]

                    # Step 5: Grounding check
                    grounded = is_grounded(answer, ranked_docs)
                    if not grounded:
                        answer = "⚠️ I could not find a grounded answer in your documents for this question."

                    # Step 6: Evaluation
                    eval_scores = {"faithfulness": 0.0, "relevance": 0.0}
                    if use_eval and grounded:
                        with st.spinner("Evaluating answer quality..."):
                            eval_scores = evaluate_answer(question, answer, ranked_docs)

                    # Step 7: Citations
                    citations = format_citations(ranked_docs, conf_scores)

                    # Persist
                    save_to_db(
                        question, answer, citations, latency_ms,
                        eval_scores["faithfulness"], eval_scores["relevance"]
                    )

                    st.session_state.chat_history.append({
                        "question": question,
                        "rewritten": final_query,
                        "answer": answer,
                        "citations": citations,
                        "latency_ms": latency_ms,
                        "faithfulness": eval_scores["faithfulness"],
                        "relevance": eval_scores["relevance"],
                    })
                    st.session_state.eval_log.append(eval_scores)

                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")

    # Render chat history
    for entry in reversed(st.session_state.chat_history):
        st.divider()
        st.markdown(f"**Q:** {entry['question']}")
        if entry["rewritten"] != entry["question"]:
            st.caption(f"Rewritten: {entry['rewritten']}")

        st.subheader("Answer")
        st.write(entry["answer"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency (ms)", entry["latency_ms"])
        c2.metric("Citations", len(entry["citations"]))
        c3.metric("Faithfulness", entry["faithfulness"])
        c4.metric("Relevance", entry["relevance"])

        download_text = build_download_text(
            entry["question"], entry["answer"], entry["citations"],
            entry["faithfulness"], entry["relevance"], entry["latency_ms"]
        )
        st.download_button(
            "⬇️ Download Answer",
            data=download_text.encode("utf-8"),
            file_name=f"ragnova_answer_{entry['latency_ms']}.txt",
            mime="text/plain",
            key=f"dl_{entry['latency_ms']}_{entry['question'][:10]}",
        )

        st.subheader("Retrieved Evidence")
        for c in entry["citations"]:
            with st.expander(
                f"{c['source_name']} | {c['chunk_id']} | page {c['page']} | confidence {c['confidence']}"
            ):
                st.write(c["excerpt"])

# ── TAB 2: Compare Docs ──────────────────────────────────────
with tab2:
    st.subheader("🔀 Multi-Document Comparison")

    if not st.session_state.all_chunks:
        st.info("Ingest at least 2 documents to use comparison.")
    else:
        doc_names = list({c.metadata.get("source_name", "") for c in st.session_state.all_chunks})

        if len(doc_names) < 2:
            st.warning("Upload at least 2 different documents for comparison.")
        else:
            selected_docs = st.multiselect("Select documents to compare", doc_names, default=doc_names[:2])
            compare_q = st.text_input("What do you want to compare across these documents?")

            if st.button("Compare", use_container_width=True):
                if not compare_q.strip():
                    st.warning("Enter a comparison question.")
                elif len(selected_docs) < 2:
                    st.warning("Select at least 2 documents.")
                else:
                    with st.spinner("Comparing..."):
                        comparison = compare_docs(compare_q, selected_docs)
                        st.markdown(comparison)
                        st.download_button(
                            "⬇️ Download Comparison",
                            data=comparison.encode("utf-8"),
                            file_name="ragnova_comparison.txt",
                            mime="text/plain",
                        )

# ── TAB 3: Evaluation Dashboard ──────────────────────────────
with tab3:
    st.subheader("📊 RAG Evaluation Dashboard")
    rows = load_from_db(limit=50)

    if not rows:
        st.info("No evaluation data yet. Ask some questions first.")
    else:
        import pandas as pd

        df = pd.DataFrame(rows, columns=[
            "Timestamp", "Question", "Answer", "Citations",
            "Latency (ms)", "Faithfulness", "Relevance"
        ])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Faithfulness", f"{df['Faithfulness'].mean():.2f}")
        c2.metric("Avg Relevance", f"{df['Relevance'].mean():.2f}")
        c3.metric("Avg Latency (ms)", f"{df['Latency (ms)'].mean():.0f}")
        c4.metric("Total Queries", len(df))

        st.divider()
        st.dataframe(
            df[["Timestamp", "Question", "Faithfulness", "Relevance", "Latency (ms)"]],
            use_container_width=True,
        )

        csv = df[["Timestamp", "Question", "Answer", "Faithfulness", "Relevance", "Latency (ms)"]].to_csv(index=False)
        st.download_button(
            "⬇️ Download Evaluation Report (CSV)",
            data=csv.encode("utf-8"),
            file_name="ragnova_eval_report.csv",
            mime="text/csv",
        )

# ── TAB 4: History ───────────────────────────────────────────
with tab4:
    st.subheader("🕓 Persistent Session History")
    rows = load_from_db(limit=20)

    if not rows:
        st.info("No history yet.")
    else:
        for row in rows:
            ts, q, ans, cits, lat, faith, rel = row
            with st.expander(f"[{ts[:16]}] {q[:80]}"):
                st.markdown(f"**Answer:** {ans}")
                st.caption(f"Latency: {lat}ms | Faithfulness: {faith} | Relevance: {rel}")