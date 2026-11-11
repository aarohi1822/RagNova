"""Build the frozen evaluation corpus.

Two steps:
  --fetch   download plain-text Wikipedia articles (current revision) into
            eval/corpus/docs/ and record their revision ids in sources.json.
            Only needed once; the fetched text is committed so the benchmark
            is reproducible even after the articles are edited upstream.
  (default) chunk eval/corpus/docs/*.txt into eval/corpus/passages.jsonl
            using the same splitter settings as the Streamlit app
            (RecursiveCharacterTextSplitter, chunk_size=400, overlap=50).

Passage ids are "<doc_slug>::<4-digit index>" and are what eval/benchmark.jsonl
references as gold ids. Re-chunking changes ids, so passages.jsonl is committed
and treated as frozen; bump CORPUS_VERSION if you ever regenerate it.

Article text is from Wikipedia, licensed CC BY-SA 4.0 (see corpus/sources.json).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

CORPUS_VERSION = "v1"
EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
DOCS_DIR = CORPUS_DIR / "docs"
SOURCES_PATH = CORPUS_DIR / "sources.json"
PASSAGES_PATH = CORPUS_DIR / "passages.jsonl"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

ARTICLES = {
    "gdpr": "General Data Protection Regulation",
    "ccpa": "California Consumer Privacy Act",
    "hipaa": "Health Insurance Portability and Accountability Act",
    "ferpa": "Family Educational Rights and Privacy Act",
    "coppa": "Children's Online Privacy Protection Act",
    "sox": "Sarbanes–Oxley Act",
    "pci_dss": "Payment Card Industry Data Security Standard",
    "iso27001": "ISO/IEC 27001",
    "nist_csf": "NIST Cybersecurity Framework",
    "eprivacy": "ePrivacy Directive",
    "dsa": "Digital Services Act",
    "ai_act": "Artificial Intelligence Act",
    "dpa2018": "Data Protection Act 2018",
    "pipeda": "Personal Information Protection and Electronic Documents Act",
    "lgpd": "General Personal Data Protection Law",
    "dodd_frank": "Dodd–Frank Wall Street Reform and Consumer Protection Act",
}

# Trailing sections that are lists of links/citations, not prose.
DROP_SECTIONS = {
    "see also", "references", "external links", "further reading",
    "notes", "sources", "bibliography", "citations", "footnotes",
}

USER_AGENT = "RAGNova-eval/1.0 (offline retrieval benchmark builder)"


def fetch() -> None:
    import requests

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    sources = []
    for slug, title in ARTICLES.items():
        for attempt in range(6):
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "format": "json", "formatversion": 2,
                    "prop": "extracts|revisions", "explaintext": 1, "exsectionformat": "wiki",
                    "rvprop": "ids|timestamp", "redirects": 1, "titles": title,
                },
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            if resp.status_code != 429:
                break
            time.sleep(5 * 2**attempt)  # be polite to the API
        resp.raise_for_status()
        time.sleep(2)
        page = resp.json()["query"]["pages"][0]
        text = clean(page["extract"])
        rev = page["revisions"][0]
        (DOCS_DIR / f"{slug}.txt").write_text(text, encoding="utf-8")
        sources.append({
            "slug": slug,
            "title": page["title"],
            "revision_id": rev["revid"],
            "revision_timestamp": rev["timestamp"],
            "url": f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}",
            "license": "CC BY-SA 4.0",
            "chars": len(text),
        })
        print(f"fetched {slug:<12} rev={rev['revid']} chars={len(text)}")
    SOURCES_PATH.write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n")


def clean(text: str) -> str:
    """Drop citation/link sections and collapse empty headings."""
    out: list[str] = []
    skipping = False
    for line in text.splitlines():
        heading = re.fullmatch(r"\s*(=+)\s*(.*?)\s*=+\s*", line)
        if heading:
            level, name = len(heading.group(1)), heading.group(2).strip().lower()
            if level == 2:
                skipping = name in DROP_SECTIONS
            if skipping:
                continue
            out.append(heading.group(2).strip())
            continue
        if not skipping:
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def chunk() -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    sources = {s["slug"]: s for s in json.loads(SOURCES_PATH.read_text())}
    n = 0
    with PASSAGES_PATH.open("w", encoding="utf-8") as fh:
        for slug in ARTICLES:
            text = (DOCS_DIR / f"{slug}.txt").read_text(encoding="utf-8")
            for idx, piece in enumerate(splitter.split_text(text)):
                record = {
                    "id": f"{slug}::{idx:04d}",
                    "doc": slug,
                    "title": sources[slug]["title"],
                    "text": piece,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} passages ({CORPUS_VERSION}) -> {PASSAGES_PATH.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true", help="re-download article text first")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    chunk()
