from pydantic import BaseModel


class IngestionResponse(BaseModel):
    documents_indexed: int
    chunks_created: int
    sources: list[str]

