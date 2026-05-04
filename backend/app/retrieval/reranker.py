from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.models.domain import RetrievedChunk

settings = get_settings()


class Reranker:
    def __init__(self) -> None:
        self.model = CrossEncoder(settings.reranker_model)

    def rerank(self, query: str, chunks: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs)
        rescored = []
        for chunk, score in zip(chunks, scores, strict=False):
            chunk.score = float(score)
            rescored.append(chunk)
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]

