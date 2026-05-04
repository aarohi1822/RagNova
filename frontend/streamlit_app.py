import os
import time
import tempfile
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Enterprise RAG QA", layout="wide")
st.title("Enterprise RAG Question Answering Platform")
st.caption("Hybrid retrieval, reranking, citations, and deployment-ready architecture.")

# ── Configuration ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Add OPENAI_API_KEY to your Streamlit secrets or environment variables.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Backend API configuration (for deployed backend)
# Set API_URL in Streamlit Cloud → Settings → Secrets (e.g., https://api.yourapp.com)
API_URL = os.getenv("API_URL") or st.secrets.get("API_URL", "http://localhost:8000")
BACKEND_AVAILABLE = None  # Will be checked on first use

# Show deployment mode info
st.info(
    f"""
    **Deployment Mode:** {('🌐 Streamlit Cloud' if 'streamlitcloud' in API_URL or 'https://' in API_URL else '💻 Local')}
    
    **API Endpoint:** `{API_URL}`
    """
)


def get_session_with_retries():
    """Create a requests session with automatic retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def check_backend_health():
    """Check if backend API is reachable."""
    try:
        session = get_session_with_retries()
        response = session.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

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
    st.session_state.chat_history = []  # list of (question, answer, citations)

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_file(uploaded_file) -> list:
    """Save upload to a temp file and load with the right LangChain loader."""
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()
    # Attach original filename as metadata
    for doc in docs:
        doc.metadata["source_name"] = uploaded_file.name
    return docs


def build_vectorstore(docs: list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)


def get_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=st.session_state.memory,
        return_source_documents=True,
        output_key="answer",
    )
    return chain


def format_citations(source_docs: list) -> list:
    seen = set()
    citations = []
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
            "score": "—",
        })
    return citations

# ── Sidebar: Admin & Configuration ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Show current API endpoint
    st.caption("Backend API URL")
    st.code(API_URL, language="text")
    
    # Check backend health
    if st.button("🔄 Check Backend Connection", use_container_width=True):
        with st.spinner("Checking backend…"):
            if check_backend_health():
                st.success("✅ Backend is reachable")
                st.session_state.backend_available = True
            else:
                st.error(f"❌ Cannot reach backend at {API_URL}\n\n**Fix options:**\n1. Deploy backend to a public URL\n2. Set API_URL secret in Streamlit Cloud\n3. Run `docker-compose up` for local development")
                st.session_state.backend_available = False
    
    st.divider()
    st.header("📄 Admin Panel")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT knowledge sources",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Ingest Documents", use_container_width=True) and uploaded_files:
        with st.spinner("Parsing and indexing documents…"):
            all_docs = []
            for f in uploaded_files:
                all_docs.extend(load_file(f))
            st.session_state.vectorstore = build_vectorstore(all_docs)
            # Reset memory on new ingestion
            st.session_state.memory.clear()
            st.session_state.chat_history = []
        st.success(f"Ingested {len(uploaded_files)} file(s) — {len(all_docs)} page(s) indexed.")

    if st.session_state.vectorstore:
        st.info("✅ Vectorstore ready")
    else:
        st.warning("No documents ingested yet.")

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

        # Store in history
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
    for citation in citations:
        with st.expander(f'{citation["source_name"]} | {citation["chunk_id"]} | page {citation["page"]}'):
            st.write(citation["excerpt"])
            st.caption(f'Score: {citation["score"]}')