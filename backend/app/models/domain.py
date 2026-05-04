from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source_name: str
    content: str
    page_number: int | None = None
    score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

