import os, sqlite3, sys, tempfile, uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from argus.analysis import investigate, load_claims, narratives
try:
    from argus.extraction import add_documents
except ImportError:  # the deployed files are out of date: say which one, instead of a redacted crash
    import argus.extraction as _ex

    st.error(f"Out-of-date file on the server: {_ex.__file__} ({Path(_ex.__file__).stat().st_size} bytes) "
             f"defines {[n for n in dir(_ex) if not n.startswith('_')]}. The current file is about "
             f"2606 bytes and defines extract_pending and add_documents.")
    st.stop()
from argus.ingestion.fetch import fetch_url, is_public_url
from argus.nlp.embeddings import EmbeddingEngine
from argus.pipeline import ArgusPipeline
from argus.storage.database import ArgusDatabase

st.set_page_config(page_title="ARGUS", layout="wide")
st.title("ARGUS: narrative and source investigation")

ROOT = Path(__file__).parent
MAIN_DB, DEMO_DB = ROOT / "data/database/argus.db", ROOT / "data/demo/argus_demo.db"
MAX_DOCS, MAX_CHUNKS, MAX_BYTES = 3, 24, 2_000_000  # limits for the hosted demo


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
    DEMO = True  # hosted copy: bundled demo corpus, plus a private temporary database per visitor
else:
    size = DEMO_DB.stat().st_size if DEMO_DB.exists() else "missing"
    st.error(f"No claims database found. Expected data/demo/argus_demo.db (currently: {size}). "
             "Run scripts/extract_claims.py locally, or add the demo database to the repo.")
    st.stop()


class NoEntities:  # the hosted demo skips the heavy entity model
    def extract_batch(self, texts, batch_size=16):
        return [[] for _ in texts]


@st.cache_resource
def get_engine():
    return EmbeddingEngine()


@st.cache_resource
def load_corpus(db_path):
    claims = load_claims(ArgusDatabase(Path(db_path)))
    vectors = get_engine().encode([c["claim"] for c in claims])
    return claims, vectors, narratives(claims, vectors)


@st.cache_resource
def get_pipeline():
    return ArgusPipeline(embedding_engine=get_engine())


def answer_model():
    """Local Ollama model when running locally; Gemini on the hosted demo if a key is set in secrets; else None."""
    if not DEMO:
        return os.getenv("ARGUS_MODEL", "llama3.2:3b")
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
        return "gemini:" + st.secrets.get("GEMINI_MODEL", "gemini-3.5-flash")
    except Exception:  # no secrets configured: evidence-only mode
        return None


if DEMO and "user_db" not in st.session_state:  # each hosted visitor gets a private temporary database
    st.session_state["user_db"] = str(Path(tempfile.gettempdir()) / f"argus_{uuid.uuid4().hex[:8]}.db")
USER_DB = Path(st.session_state["user_db"]) if DEMO else None

db, own = (DEMO_DB if DEMO else MAIN_DB), False
if DEMO and has_claims(USER_DB):
    if st.session_state.pop("switch_to_own", False):
        st.session_state["corpus"] = "My documents"
    own = st.radio("Corpus", ["Reflection 70B demo", "My documents"], horizontal=True, key="corpus") == "My documents"
    db = USER_DB if own else DEMO_DB

engine = get_engine()
claims, vectors, story = load_corpus(str(db))
st.caption(f"{len(claims)} claims from {len({c['doc'] for c in claims})} documents")

ask_tab, narrative_tab, add_tab = st.tabs(["Ask", "Narratives and sources", "Add documents"])

with ask_tab:
    question = st.text_input("Ask about the corpus", "Did independent testers reproduce the benchmark results?")
    if st.button("Investigate") and question:
        try:
            with st.spinner("Reading the evidence..."):
                answer, evidence = investigate(question, claims, vectors, engine, answer_model())
        except Exception as exc:
            st.error(f"Could not reach the language model: {exc}")
            st.stop()
        st.write(answer or "The language model is off here, so this is the evidence ARGUS retrieved for your question:")
        for n, c in enumerate(evidence, 1):
            st.markdown(f'**[{n}]** {c["source"]} ({c["date"][:10] or "undated"}): "{c["evidence"]}"')

with narrative_tab:
    min_size = st.slider("Only narratives with at least this many claims", 1, 10, 2 if not own else 1)
    for n in (n for n in story if n["size"] >= min_size):
        with st.expander(f"{n['size']} claims: {n['claims'][0]['claim'][:100]}"):
            for it in n["claims"]:
                repeats = f"; repeated by {', '.join(it['repeated_by'])}" if it["repeated_by"] else ""
                st.markdown(f"**{it['date'] or 'undated'}**: {it['claim']}  \n*first reported by {it['origin']}{repeats}*")

with add_tab:
    st.caption("Add your own articles. They are read, split into claims by the configured model, and analysed in the other tabs.")
    model = answer_model()
    if model is None:
        st.info("No language model is configured for this app, so documents cannot be analysed here.")
        st.stop()
    if DEMO:
        st.caption(f"Hosted demo: up to {MAX_DOCS} documents of {MAX_BYTES // 1_000_000} MB each, private to this session. "
                   "Only the first part of each document is analysed.")
    if "added" in st.session_state:
        st.success(st.session_state.pop("added"))
    files = st.file_uploader("Upload .txt, .html or .pdf files", type=["txt", "html", "htm", "pdf"], accept_multiple_files=True)
    urls = st.text_area("...or paste article URLs, one per line")
    if st.button("Add and analyse") and (files or urls.strip()):
        target = USER_DB if DEMO else MAIN_DB
        raw = Path(tempfile.mkdtemp()) if DEMO else Path("data/raw")
        raw.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in files:
            if DEMO and (len(paths) >= MAX_DOCS or f.size > MAX_BYTES):
                st.warning(f"Skipped {f.name} (hosted demo limits).")
                continue
            paths.append(raw / Path(f.name).name)
            paths[-1].write_bytes(f.getvalue())
        for url in (u.strip() for u in urls.splitlines() if u.strip().startswith("http")):
            if DEMO and (len(paths) >= MAX_DOCS or not is_public_url(url)):
                st.warning(f"Skipped {url} (hosted demo limits, or not a public address).")
                continue
            try:
                paths.append(fetch_url(url, raw))
            except Exception as exc:
                st.warning(f"Could not fetch {url}: {exc}")
        if DEMO:
            os.environ["ARGUS_MODEL"] = model
        bar, problems = st.progress(0.0, "Reading and extracting claims..."), []
        pipeline = (ArgusPipeline(database=ArgusDatabase(target), entity_extractor=NoEntities(), embedding_engine=engine)
                    if DEMO else get_pipeline())
        added = add_documents(paths, ArgusDatabase(target), pipeline, lambda m, i, n: (
            bar.progress(i / n if n else 1.0), problems.append(m) if ("failed" in m or "Could not" in m) else None),
            MAX_CHUNKS if DEMO else None)
        load_corpus.clear()
        st.session_state["added"] = f"Added {added} document(s)." + (f" {len(problems)} problem(s): {problems[0][:150]}" if problems else "")
        st.session_state["switch_to_own"] = DEMO
        st.rerun()
