from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)
    session_id: str = "default"
    filters: dict[str, str] | None = None


class Citation(BaseModel):
    source_name: str
    chunk_id: str
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    rewritten_query: str
    citations: list[Citation]
    latency_ms: float
    validation_notes: list[str]

