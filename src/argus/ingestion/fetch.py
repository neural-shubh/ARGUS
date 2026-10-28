import hashlib
import ipaddress
import json
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

import requests


def fetch_url(url: str, directory: Path) -> Path:
    """Download a page into `directory` with a .meta.json recording its source and URL."""
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    name = f"{re.sub(r'[^a-z0-9]+', '-', domain).strip('-')}-{hashlib.md5(url.encode()).hexdigest()[:8]}"
    page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    page.raise_for_status()
    path = Path(directory) / f"{name}.html"
    path.write_text(page.text, encoding="utf-8")
    path.with_name(f"{name}.meta.json").write_text(json.dumps({"source_name": domain, "url": url}))
    return path


def is_public_url(url: str) -> bool:
    """Basic safety check for a hosted app: http(s) only, and the host must resolve to public addresses."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        return all(ipaddress.ip_address(info[4][0]).is_global for info in socket.getaddrinfo(parts.hostname, None))
    except (OSError, ValueError):
        return False
