from __future__ import annotations

import uuid

import chromadb

from app.core.config import get_settings
from app.models.domain import RetrievedChunk

settings = get_settings()


class ChromaVectorStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.vector_db_path)
        self.collection = self.client.get_or_create_collection(name="rag_documents")

    def upsert(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        ids = [chunk.get("chunk_id", str(uuid.uuid4())) for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk["document_id"],
                "source_name": chunk["source_name"],
                "page_number": chunk.get("page_number"),
                **chunk.get("metadata", {}),
            }
            for chunk in chunks
        ]
        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def search(self, embedding: list[float], limit: int = 8) -> list[RetrievedChunk]:
        result = self.collection.query(query_embeddings=[embedding], n_results=limit)
        documents = result.get("documents", [[]])[0]
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: list[RetrievedChunk] = []

        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    source_name=str(metadata.get("source_name", "unknown")),
                    content=content,
                    page_number=metadata.get("page_number"),
                    score=1 / (1 + float(distance)),
                    metadata={k: str(v) for k, v in metadata.items()},
                )
            )
        return chunks

