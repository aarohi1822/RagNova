from __future__ import annotations

from openai import OpenAI

from app.core.config import get_settings
from app.models.domain import RetrievedChunk

settings = get_settings()


class AnswerGenerator:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def build_prompt(self, query: str, chunks: list[RetrievedChunk], history: list[str]) -> str:
        context = "\n\n".join(
            [
                f"[{chunk.chunk_id}] source={chunk.source_name} page={chunk.page_number}\n{chunk.content}"
                for chunk in chunks
            ]
        )
        history_block = "\n".join(history[-4:])
        return (
            "You are an enterprise-grade retrieval QA assistant. "
            "Answer only from the supplied context, cite source ids inline, "
            "state uncertainty when evidence is incomplete, and never invent facts.\n\n"
            f"Conversation history:\n{history_block}\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}"
        )

    def generate(self, query: str, chunks: list[RetrievedChunk], history: list[str]) -> str:
        prompt = self.build_prompt(query, chunks, history)
        if not self.client:
            citations = ", ".join(f"{chunk.source_name}:{chunk.chunk_id}" for chunk in chunks[:2])
            return (
                "Demo mode answer: connect an LLM API key to enable full generation. "
                f"Top evidence came from {citations}."
            )

        response = self.client.responses.create(
            model=settings.llm_model,
            input=prompt,
            temperature=0.1,
        )
        return response.output_text

