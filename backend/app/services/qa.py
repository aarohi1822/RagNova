from __future__ import annotations

import time

from app.core.config import get_settings
from app.llm.generator import AnswerGenerator
from app.memory.session_memory import SessionMemory
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import Reranker
from app.schemas.chat import Citation, ChatRequest, ChatResponse
from app.services.validator import AnswerValidator

settings = get_settings()
memory = SessionMemory()


class QAService:
    def __init__(self) -> None:
        self.rewriter = QueryRewriter()
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.generator = AnswerGenerator()
        self.validator = AnswerValidator()

    async def answer(self, payload: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        history = memory.get(payload.session_id)
        rewritten = self.rewriter.rewrite(payload.question, history)
        candidates = self.retriever.search(rewritten)
        top_chunks = self.reranker.rerank(rewritten, candidates, limit=settings.rerank_top_k)
        answer = self.generator.generate(rewritten, top_chunks, history)
        notes = self.validator.validate(answer, top_chunks)

        memory.append(payload.session_id, f"user: {payload.question}")
        memory.append(payload.session_id, f"assistant: {answer}")

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        citations = [
            Citation(
                source_name=chunk.source_name,
                chunk_id=chunk.chunk_id,
                excerpt=chunk.content[:240],
                score=round(chunk.score, 4),
            )
            for chunk in top_chunks
        ]
        return ChatResponse(
            answer=answer,
            rewritten_query=rewritten,
            citations=citations,
            latency_ms=latency_ms,
            validation_notes=notes,
        )
