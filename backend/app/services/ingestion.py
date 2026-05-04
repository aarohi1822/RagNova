from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings
from app.schemas.documents import IngestionResponse
from app.services.chunking import ChunkingService
from app.services.embedder import EmbeddingService
from app.services.parser import DocumentParser
from app.storage.chroma_store import ChromaVectorStore

settings = get_settings()


class IngestionService:
    def __init__(self) -> None:
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.parser = DocumentParser()
        self.chunker = ChunkingService()
        self.embedder = EmbeddingService()
        self.vector_store = ChromaVectorStore()

    async def ingest(self, files: list[UploadFile]) -> IngestionResponse:
        all_chunks: list[dict] = []
        sources: list[str] = []

        for uploaded in files:
            target = self.upload_dir / uploaded.filename
            content = await uploaded.read()
            target.write_bytes(content)

            text, pages = self.parser.parse(target)
            document_id = str(uuid.uuid4())
            sources.append(uploaded.filename)

            if not text.strip():
                continue

            for page in pages:
                chunks = self.chunker.split(page["content"])
                for idx, chunk_text in enumerate(chunks):
                    all_chunks.append(
                        {
                            "chunk_id": f"{document_id}-{page.get('page_number', 'na')}-{idx}",
                            "document_id": document_id,
                            "source_name": uploaded.filename,
                            "page_number": page.get("page_number"),
                            "content": chunk_text,
                            "metadata": {"ingestion_type": target.suffix.lower()},
                        }
                    )

        if not all_chunks:
            raise ValueError("No readable document content found during ingestion.")

        embeddings = self.embedder.encode([chunk["content"] for chunk in all_chunks])
        self.vector_store.upsert(all_chunks, embeddings)
        return IngestionResponse(
            documents_indexed=len(sources),
            chunks_created=len(all_chunks),
            sources=sources,
        )

