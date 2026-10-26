# ARGUS

**Point it at a pile of articles and ask what happened, who said it first, and what the evidence is.**

| | |
|---|---|
 **Live demo** | [https://YOUR-APP-URL.streamlit.app](https://YOUR-APP-URL.streamlit.app) *(replace after deploying)* 


![Ask tab](docs/screenshot-ask.png)

## Example (real output on the demo corpus)

> **Q:** Who first accused Shumer of fraud, and when?
>
> **A:** Shin Megami Boson first accused Matt Shumer of "fraud in the AI research community" on September 8, as reported by venturebeat.com [1] and news.aibase.com [9].
>
> **Evidence:** [1] venturebeat.com (2024-09-11), [4] tomsguide.com (2024-09-12), [9] news.aibase.com, each shown with the exact sentence it came from.

Every answer ships with the numbered source sentences, so it can be checked in seconds.

## What it gives you

| You get | How |
|---|---|
| Claims, each tied to its source sentence | A local LLM extracts them; any claim whose quoted evidence is not in the document is discarded |
| Narratives | Claims embedded and clustered, so differently worded versions of one claim sit together |
| **Who reported it first** | Claims are ordered by publication date; later repeats and reposts are listed as repeats, not as extra confirmation |
| Answers with citations | Retrieval over claims, then a short answer restricted to the retrieved items |
| Your own corpus | Upload files or paste URLs in the app; no code changes |

## Pipeline

`documents` → text extraction (main article only) → chunks → NER + sentence embeddings → SQLite → claim extraction → clustering, duplicate detection, first-reporter tracing → question answering

| Stage | Tool |
|---|---|
| Text and metadata | BeautifulSoup, PyMuPDF (dates from meta tags, JSON-LD, `<time>`) |
| Entities | `dslim/bert-base-NER` |
| Embeddings | `all-MiniLM-L6-v2` |
| Claims and answers | Any Ollama model (tested with `qwen2.5:7b`), JSON-schema constrained |
| Conflicting figures | `nli-deberta-v3-small` cross-encoder (experimental) |
| Storage and UI | SQLite, Streamlit |

## Demo corpus results

| | |
|---|---|
| Articles ingested | 14 (Reflection 70B benchmark dispute) |
| Chunks / claims after cleanup | 110 / 424 |
| Exact duplicates caught | 1 (same article under two URLs) |
| Near-duplicates flagged | 1 (same story on two domains) |

## Run it

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b && export ARGUS_MODEL=qwen2.5:7b
python scripts/ingest.py data/raw && python scripts/extract_claims.py
streamlit run app.py
```

`OLLAMA_HOST` can point at a remote GPU. `pytest` runs the core tests.

## Where it fails

- The 7B model occasionally embellishes a sentence in an answer. The evidence list is there to catch that.
- The disagreement tab only compares claims with figures or dates, and the demo corpus's disagreements are mostly qualitative, so it finds few.
- A later article repeating a claim may be independent reporting; ARGUS reports "repeated", not "unconfirmed".
- Thresholds were tuned on one corpus.

## Layout

```
app.py              Streamlit app
scripts/            ingest, fetch_urls, extract_claims, cluster, contradictions, investigate, clean_claims
src/argus/          ingestion, nlp (ner, embeddings, claims), storage, analysis, extraction
tests/              core logic tests
```
