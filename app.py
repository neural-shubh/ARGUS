import json, os, sqlite3, sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from argus.extraction import extract_pending
from argus.ingestion.fetch import fetch_url
from argus.pipeline import ArgusPipeline
from argus.analysis import contradictions, investigate, load_claims, narratives
from argus.nlp.embeddings import EmbeddingEngine
from argus.storage.database import ArgusDatabase

st.set_page_config(page_title="ARGUS", layout="wide")
st.title("ARGUS: narrative and source investigation")

ROOT = Path(__file__).parent
MAIN_DB, DEMO_DB = ROOT / "data/database/argus.db", ROOT / "data/demo/argus_demo.db"


def has_claims(path):
    """True if the database file exists and holds claims (opened read-only, so nothing is created)."""
    if not path.exists():
        return False
    try:
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            return conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] > 0
    except sqlite3.Error:
        return False


if has_claims(MAIN_DB):
    DEMO = False
elif has_claims(DEMO_DB):
    DEMO = True  # hosted copy: use the bundled demo corpus
else:
    size = DEMO_DB.stat().st_size if DEMO_DB.exists() else "missing"
    st.error(f"No claims database found. Expected data/demo/argus_demo.db (currently: {size}). "
             "Run scripts/extract_claims.py locally, or add the demo database to the repo.")
    st.stop()



@st.cache_resource
def setup():
    claims = load_claims(ArgusDatabase(DEMO_DB if DEMO else MAIN_DB))
    engine = EmbeddingEngine()
    vectors = engine.encode([c["claim"] for c in claims])
    return claims, engine, vectors, narratives(claims, vectors)


def answer_model():
    """Local Ollama model when running locally; Gemini on the hosted demo if a key is set in secrets; else None."""
    if not DEMO:
        return os.getenv("ARGUS_MODEL", "llama3.2:3b")
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
        return "gemini:" + st.secrets.get("GEMINI_MODEL", "gemini-3.5-flash")
    except Exception:  # no secrets configured: evidence-only mode
        return None


claims, engine, vectors, story = setup()
st.caption(f"{len(claims)} claims from {len({c['doc'] for c in claims})} documents")


@st.cache_resource
def get_pipeline():
    return ArgusPipeline(embedding_engine=engine)


ask_tab, narrative_tab, disagree_tab, add_tab = st.tabs(["Ask", "Narratives and sources", "Possible disagreements", "Add documents"])

with ask_tab:
    question = st.text_input("Ask about the corpus", "Did independent testers reproduce the benchmark results?")
    if st.button("Investigate") and question:
        try:
            with st.spinner("Reading the evidence..."):
                answer, evidence = investigate(question, claims, vectors, engine, answer_model())
        except Exception as exc:
            st.error(f"Could not reach the language model: {exc}")
            st.stop()
        st.write(answer or "Demo mode: the language model is off, so this is the evidence ARGUS retrieved for your question:")
        for n, c in enumerate(evidence, 1):
            st.markdown(f'**[{n}]** {c["source"]} ({c["date"][:10] or "undated"}): "{c["evidence"]}"')

with narrative_tab:
    min_size = st.slider("Only narratives with at least this many claims", 1, 10, 2)
    for n in (n for n in story if n["size"] >= min_size):
        with st.expander(f"{n['size']} claims: {n['claims'][0]['claim'][:100]}"):
            for it in n["claims"]:
                repeats = f"; repeated by {', '.join(it['repeated_by'])}" if it["repeated_by"] else ""
                st.markdown(f"**{it['date'] or 'undated'}**: {it['claim']}  \n*first reported by {it['origin']}{repeats}*")

with disagree_tab:
    st.caption("Experimental: compares claims that quote figures or dates. It can miss disagreements and flag false ones.")
    saved = ROOT / "data/demo/disagreements.json"
    if DEMO:
        st.caption("Precomputed on this corpus.")
        results = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else None
        if results is None:
            st.info("No saved results in this copy of the demo.")
        elif not results:
            st.info("No confident disagreements were found on this corpus. Its disputes are mostly qualitative, with no shared figures to compare.")
        for r in results or []:
            st.markdown(f"**{r['score']:.2f}**  \n" + "  \n".join(f"{c['source']} ({c['date'][:10] or 'undated'}): {c['claim']}" for c in (r["older"], r["newer"])))
    elif st.button("Find possible disagreements"):
        from sentence_transformers import CrossEncoder

        found = [r for r in contradictions(claims, vectors, CrossEncoder("cross-encoder/nli-deberta-v3-small")) if r["score"] >= 0.8]
        st.info("No confident disagreements found.") if not found else None
        for r in found:
            st.markdown(f"**{r['score']:.2f}**  \n" + "  \n".join(f"{c['source']} ({c['date'][:10] or 'undated'}): {c['claim']}" for c in (r["older"], r["newer"])))

with add_tab:
    st.caption("Add your own articles. They are read, split into claims by the configured model, and added to the corpus.")
    if DEMO:
        st.markdown("""**Analysing your own documents runs locally**, because it needs a language model and a private database.

```
git clone https://github.com/neural-shubh/ARGUS && cd ARGUS
pip install -r requirements.txt
ollama pull qwen2.5:7b
streamlit run app.py
```

Then use this tab to upload files or paste URLs; see the README for details.""")
        st.stop()
    if "added" in st.session_state:
        st.success(st.session_state.pop("added"))
    files = st.file_uploader("Upload .txt, .html or .pdf files", type=["txt", "html", "htm", "pdf"], accept_multiple_files=True)
    urls = st.text_area("...or paste article URLs, one per line")
    if st.button("Add and analyse") and (files or urls.strip()):
        raw = Path("data/raw")
        raw.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in files:
            paths.append(raw / Path(f.name).name)
            paths[-1].write_bytes(f.getvalue())
        for url in (u.strip() for u in urls.splitlines() if u.strip().startswith("http")):
            try:
                paths.append(fetch_url(url, raw))
            except Exception as exc:
                st.warning(f"Could not fetch {url}: {exc}")
        added = 0
        for path in paths:
            try:
                added += get_pipeline().process_file(path)["status"] == "processed"
            except Exception as exc:
                st.warning(f"Could not read {path.name}: {exc}")
        bar, failures = st.progress(0.0, "Extracting claims..."), []
        extract_pending(ArgusDatabase(), lambda m, i, n: (bar.progress(i / n if n else 1.0), failures.append(m) if "failed" in m else None))
        st.session_state["added"] = f"Added {added} document(s)." + (f" {len(failures)} chunk(s) failed (is the model running?); rerun to retry." if failures else "")
        st.cache_resource.clear()
        st.rerun()
