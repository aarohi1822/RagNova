# Architecture

## System Design

```mermaid
flowchart LR
    A["Admin Upload Panel"] --> B["FastAPI Ingestion API"]
    B --> C["Parser Layer<br/>PDF / DOCX / TXT"]
    C --> D["Chunking Service<br/>Recursive + metadata-aware"]
    D --> E["Embedding Service<br/>Sentence Transformers"]
    D --> F["Keyword Index<br/>BM25"]
    E --> G["Chroma Vector Store"]
    H["User Query"] --> I["Query Rewriter"]
    I --> J["Hybrid Retriever"]
    F --> J
    G --> J
    J --> K["Re-ranking Layer"]
    K --> L["LLM Answer Generator"]
    L --> M["Validation Layer"]
    M --> N["Cited Answer + Metrics + Trace"]
```

## Enterprise Rationale

- Hybrid retrieval improves recall for exact terms, abbreviations, policies, and semantically paraphrased queries.
- Re-ranking raises precision at the final evidence layer and gives better source attribution quality.
- Query rewriting plus session memory support multi-turn enterprise knowledge workflows.
- Validation notes create a lightweight answer-quality guardrail and interview-friendly "hallucination reduction" story.
- The modular service layout supports future upgrades to Pinecone, pgvector, Graph RAG, SSO, audit logs, and observability.

