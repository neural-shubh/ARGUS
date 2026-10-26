import json, re, sys
from pathlib import Path
from urllib.parse import urlparse

import requests

urls = [u.strip() for u in Path("urls.txt").read_text(encoding="utf-8-sig").splitlines() if u.strip().startswith("http")]
if not urls:
    sys.exit("No URLs found in urls.txt. Put one link per line, each starting with http.")
print(f"Found {len(urls)} URLs")
out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)

for n, url in enumerate(urls, 1):
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    name = f"{n:03d}-{re.sub(r'[^a-z0-9]+', '-', domain).strip('-')}"
    try:
        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        page.raise_for_status()
    except requests.RequestException as exc:
        print(f"FAILED {url}: {exc}")
        continue
    (out / f"{name}.html").write_text(page.text, encoding="utf-8")
    (out / f"{name}.meta.json").write_text(json.dumps({"source_name": domain, "url": url}))
    print(f"saved {name}.html")