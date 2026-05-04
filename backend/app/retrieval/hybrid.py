from __future__ import annotations

from collections import defaultdict

from app.core.config import get_settings
from app.models.domain import RetrievedChunk
from app.retrieval.keyword import KeywordRetriever
from app.services.embedder import EmbeddingService
from app.storage.chroma_store import ChromaVectorStore

settings = get_settings()


class HybridRetriever:
    def __init__(self) -> None:
        self.embedder = EmbeddingService()
        self.vector_store = ChromaVectorStore()
        self.keyword = KeywordRetriever()

    def warm_keyword_index(self) -> None:
        raw = self.vector_store.collection.get(include=["documents", "metadatas"])
        ids = raw.get("ids", [])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])
        chunks = [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=str(metadata.get("document_id", "")),
                source_name=str(metadata.get("source_name", "unknown")),
                content=content,
                page_number=metadata.get("page_number"),
                metadata={k: str(v) for k, v in metadata.items()},
            )
            for chunk_id, content, metadata in zip(ids, documents, metadatas, strict=False)
        ]
        self.keyword.rebuild(chunks)

    def search(self, query: str) -> list[RetrievedChunk]:
        vector_query = self.embedder.encode([query])[0]
        semantic_hits = self.vector_store.search(vector_query, limit=settings.top_k)
        if self.keyword.bm25 is None:
            self.warm_keyword_index()
        keyword_hits = self.keyword.search(query, limit=settings.top_k)

        merged: dict[str, float] = defaultdict(float)
        canonical: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(semantic_hits, start=1):
            merged[chunk.chunk_id] += 1 / (rank + 1)
            canonical[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(keyword_hits, start=1):
            merged[chunk.chunk_id] += 1 / (rank + 1)
            canonical.setdefault(chunk.chunk_id, chunk)

        fused = []
        for chunk_id, score in merged.items():
            chunk = canonical[chunk_id]
            chunk.score = score
            fused.append(chunk)

        return sorted(fused, key=lambda item: item.score, reverse=True)[: settings.top_k]

