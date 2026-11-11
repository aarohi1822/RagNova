from app.services.chunking import ChunkingService


def test_chunking_produces_non_empty_splits() -> None:
    service = ChunkingService()
    # ~2,400 chars of word-separated text. (A single 2,400-char token cannot be split:
    # the splitter only breaks on paragraphs, newlines, ". " and spaces.)
    text = "word " * 480
    chunks = service.split(text)
    assert chunks
    assert len(chunks) >= 2

