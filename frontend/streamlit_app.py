import os
import time
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Enterprise RAG QA", layout="wide")
st.title("Enterprise RAG Question Answering Platform")
st.caption("Hybrid retrieval · citations · 100% free stack (Groq + HuggingFace embeddings)")

# ── API key ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY to your Streamlit secrets. Get one free at https://console.groq.com")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ── Session state init ─────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_embeddings():
    """Free local embeddings — no API key needed."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_file(uploaded_file) -> list:
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if suffix == ".pdf":
        from langchain_community.document_loaders import UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(tmp_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()
    for doc in docs:
        doc.metadata["source_name"] = uploaded_file.name
    return docs


def build_vectorstore(docs: list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())


def get_chain(vectorstore):
    llm = ChatGroq(
        model="llama3-8b-8192",  # free and fast
        temperature=0,
        groq_api_key=GROQ_API_KEY,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=st.session_state.memory,
        return_source_documents=True,
        output_key="answer",
    )


def format_citations(source_docs: list) -> list:
    seen, citations = set(), []
    for i, doc in enumerate(source_docs):
        excerpt = doc.page_content.strip()[:400]
        key = excerpt[:80]
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "chunk_id": f"chunk-{i+1}",
            "source_name": doc.metadata.get("source_name", doc.metadata.get("source", "unknown")),
            "page": doc.metadata.get("page", "—"),
            "excerpt": excerpt,
        })
    return citations

# ── Sidebar: document ingestion ────────────────────────────────────────────────
with st.sidebar:
    st.header("Admin Panel")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT knowledge sources",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Ingest Documents", use_container_width=True) and uploaded_files:
        with st.spinner("Parsing and indexing… (first run downloads embeddings model ~90MB)"):
            all_docs = []
            for f in uploaded_files:
                all_docs.extend(load_file(f))

            try:
                st.session_state.vectorstore = build_vectorstore(all_docs)
                st.session_state.memory.clear()
                st.session_state.chat_history = []
                st.success(f"✅ {len(uploaded_files)} file(s) · {len(all_docs)} page(s) indexed.")
            except Exception as e:
                st.error(str(e))
            st.session_state.memory.clear()
            st.session_state.chat_history = []
        st.success(f"✅ {len(uploaded_files)} file(s) · {len(all_docs)} page(s) indexed.")

    if st.session_state.vectorstore:
        st.info("✅ Vectorstore ready")
    else:
        st.warning("No documents ingested yet.")

    st.divider()
    st.caption("🆓 Powered by Groq (Llama 3) + HuggingFace embeddings — fully free")

# ── Main: Q&A ──────────────────────────────────────────────────────────────────
question = st.text_area("Ask a grounded question from your uploaded corpus")

if st.button("Generate Answer", type="primary", use_container_width=True) and question.strip():
    if st.session_state.vectorstore is None:
        st.error("Please upload and ingest at least one document first.")
    else:
        with st.spinner("Retrieving and generating…"):
            chain = get_chain(st.session_state.vectorstore)
            t0 = time.time()
            result = chain.invoke({"question": question})
            latency_ms = int((time.time() - t0) * 1000)

        answer = result["answer"]
        citations = format_citations(result.get("source_documents", []))
        st.session_state.chat_history.append((question, answer, citations, latency_ms))

# ── Render chat history ────────────────────────────────────────────────────────
for q, ans, citations, latency_ms in reversed(st.session_state.chat_history):
    st.divider()
    st.markdown(f"**Q: {q}**")

    st.subheader("Answer")
    st.write(ans)

    col1, col2 = st.columns(2)
    col1.metric("Latency (ms)", latency_ms)
    col2.metric("Citations", len(citations))

    st.subheader("Retrieved Evidence")
    for c in citations:
        with st.expander(f'{c["source_name"]} | {c["chunk_id"]} | page {c["page"]}'):
            st.write(c["excerpt"])