import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise RAG QA", layout="wide")

st.title("Enterprise RAG Question Answering Platform")
st.caption("Hybrid retrieval, reranking, citations, and deployment-ready architecture.")

with st.sidebar:
    st.header("Admin Panel")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT knowledge sources",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Ingest Documents", use_container_width=True) and uploaded_files:
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        response = requests.post(f"{API_URL}/documents/upload", files=files, timeout=120)
        if response.ok:
            st.success(response.json())
        else:
            st.error(response.text)

session_id = st.text_input("Session ID", value="portfolio-demo")
question = st.text_area("Ask a grounded question from your uploaded corpus")

if st.button("Generate Answer", type="primary", use_container_width=True) and question.strip():
    payload = {"question": question, "session_id": session_id}
    response = requests.post(f"{API_URL}/chat/ask", json=payload, timeout=120)
    if response.ok:
        data = response.json()
        st.subheader("Answer")
        st.write(data["answer"])

        col1, col2 = st.columns(2)
        col1.metric("Latency (ms)", data["latency_ms"])
        col2.metric("Citations", len(data["citations"]))

        st.subheader("Retrieved Evidence")
        for citation in data["citations"]:
            with st.expander(f'{citation["source_name"]} | {citation["chunk_id"]}'):
                st.write(citation["excerpt"])
                st.caption(f'Score: {citation["score"]}')

        if data["validation_notes"]:
            st.subheader("Validation Notes")
            for note in data["validation_notes"]:
                st.warning(note)
    else:
        st.error(response.text)

