from app.services.chunking import ChunkingService


def test_chunking_produces_non_empty_splits() -> None:
    service = ChunkingService()
    text = "A" * 2400
    chunks = service.split(text)
    assert chunks
    assert len(chunks) >= 2

