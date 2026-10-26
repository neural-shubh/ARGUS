from datetime import datetime
from pathlib import Path
import hashlib
import json
import re
import warnings

import fitz
from bs4 import BeautifulSoup

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt"}


def _normalize_date(value) -> str | None:
    """Return an ISO-8601 string (sortable) or None if it can't be parsed."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).strip()).isoformat()
    except ValueError:
        warnings.warn(f"Could not parse publication date: {value!r}")
        return None


def _html_metadata(soup: BeautifulSoup) -> dict:
    """Best-effort source metadata from common <meta>/<link> tags."""
    def meta(*names):
        for name in names:
            tag = (soup.find("meta", attrs={"property": name})
                   or soup.find("meta", attrs={"name": name}))
            if tag and tag.get("content"):
                return tag["content"].strip()
        return None

    published = meta("article:published_time", "date", "pubdate")
    if not published and (tag := soup.find("time", attrs={"datetime": True})):
        published = tag["datetime"]
    if not published and (found := re.search(r'"datePublished"\s*:\s*"([^"]+)"', str(soup))):
        published = found.group(1)  # JSON-LD, used by many news sites

    canonical = soup.find("link", rel="canonical")
    return {
        "source_name": meta("og:site_name", "application-name"),
        "published_at": published,
        "url": (canonical.get("href") if canonical else None) or meta("og:url"),
    }


def _sidecar_metadata(path: Path) -> dict:
    """Optional <name>.meta.json next to the document; overrides HTML tags."""
    sidecar = path.with_name(f"{path.stem}.meta.json")
    if not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        warnings.warn(f"Ignoring invalid metadata file {sidecar.name}: {exc}")
        return {}
    return {k: data.get(k) for k in ("source_name", "published_at", "url")}


def load_document(file_path: str | Path) -> dict:
    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    metadata = {"source_name": None, "published_at": None, "url": None}

    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as pdf:
            text = "\n".join(page.get_text() for page in pdf)

    elif path.suffix.lower() in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        # Read metadata before the tree is stripped.
        metadata.update({k: v for k, v in _html_metadata(soup).items() if v})

        # Use the largest <article> (the main story, not teaser cards), else the whole page,
        # and drop menus, footers and sidebars so boilerplate doesn't become "claims".
        articles = soup.find_all("article")
        root = max(articles, key=lambda a: len(a.get_text()), default=soup)
        for tag in root(["script", "style", "noscript", "nav", "footer", "aside", "form"]):
            tag.decompose()

        text = root.get_text(separator="\n", strip=True)

    else:
        text = path.read_text(encoding="utf-8-sig", errors="replace")

    metadata.update({k: v for k, v in _sidecar_metadata(path).items() if v})

    # Normalize line endings and repeated whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = "\n".join(line for line in text.splitlines() if line)

    if not text.strip():
        raise ValueError(f"No extractable text found in: {path}")

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return {
        "source_path": str(path),
        "filename": path.name,
        "file_type": path.suffix.lower(),
        "text": text,
        "content_hash": content_hash,
        "character_count": len(text),
        "source_name": metadata["source_name"],
        "published_at": _normalize_date(metadata["published_at"]),
        "url": metadata["url"],
    }


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary < 0:
                boundary = text.rfind(" ", start, end)

            if boundary > start + chunk_size // 2:
                end = boundary

        content = text[start:end].strip()

        if content:
            left_trim = len(text[start:end]) - len(text[start:end].lstrip())
            right_trim = len(text[start:end].rstrip())

            chunks.append({
                "chunk_index": len(chunks),
                "start_char": start + left_trim,
                "end_char": start + right_trim,
                "text": content,
            })

        if end >= len(text):
            break

        start += step

    return chunks