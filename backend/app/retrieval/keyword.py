from __future__ import annotations

from rank_bm25 import BM25Okapi

from app.models.domain import RetrievedChunk


class KeywordRetriever:
    def __init__(self) -> None:
        self.corpus: list[list[str]] = []
        self.chunks: list[RetrievedChunk] = []
        self.bm25: BM25Okapi | None = None

    def rebuild(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.corpus = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, limit: int = 8) -> list[RetrievedChunk]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(
            zip(self.chunks, scores, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_name=chunk.source_name,
                content=chunk.content,
                page_number=chunk.page_number,
                score=float(score),
                metadata=chunk.metadata,
            )
            for chunk, score in ranked
        ]

