from pathlib import Path

import docx2txt
from pypdf import PdfReader


class DocumentParser:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def parse(self, path: Path) -> tuple[str, list[dict]]:
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        if suffix == ".txt":
            text = path.read_text(encoding="utf-8")
            return text, [{"page_number": None, "content": text}]

        if suffix == ".docx":
            text = docx2txt.process(str(path))
            return text, [{"page_number": None, "content": text}]

        reader = PdfReader(str(path))
        pages: list[dict] = []
        for index, page in enumerate(reader.pages, start=1):
            content = page.extract_text() or ""
            pages.append({"page_number": index, "content": content})
        return "\n".join(page["content"] for page in pages), pages

