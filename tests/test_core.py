import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from argus.analysis import FIGURE
from argus.ingestion.reader import chunk_text
from argus.nlp.claims import resolve_dates
from argus.storage.database import ArgusDatabase


def test_chunks_cover_the_text_and_respect_the_size():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_text(text, chunk_size=200, overlap=30)
    assert all(len(c["text"]) <= 200 for c in chunks)
    assert chunks[0]["start_char"] == 0 and chunks[-1]["end_char"] == len(text)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_relative_dates_become_absolute():
    assert "(2024-09-08)" in resolve_dates("He replied on Sunday.", "2024-09-11T01:10:57")
    assert "in November 2026" in resolve_dates("It ships next month.", "2026-10-05T09:00:00")
    assert resolve_dates("Shumer will speak on Friday.", "2024-09-11T00:00:00") == "Shumer will speak on Friday."


def test_figures_ignore_model_names_but_keep_real_numbers():
    assert not FIGURE.search("Reflection 70B and Aurora-2 on 2026-10-05")
    assert FIGURE.search("It scored 91 percent") and FIGURE.search("trained on 4,000 GPUs")


def test_earlier_document_becomes_the_original(tmp_path):
    db = ArgusDatabase(tmp_path / "t.db")
    vector = np.array([[1.0, 0.0]], dtype=np.float32)
    chunk = [{"chunk_index": 0, "start_char": 0, "end_char": 1, "text": "x", "entities": []}]

    def save(name, date, **extra):
        doc = {"source_path": name, "filename": name, "file_type": ".txt", "text": "x", "content_hash": name,
               "character_count": 1, "source_name": name, "published_at": date, "url": None}
        return db.save_document(doc, chunk, vector, **extra)

    late = save("late", "2026-10-05T14:00:00")
    early = save("early", "2026-10-05T09:00:00", duplicate_of=late, similarity_to_original=0.99)
    db.promote_if_earlier(early, late, 0.99)
    with db.connect() as conn:
        rows = dict(conn.execute("SELECT filename, duplicate_of FROM documents"))
    assert rows["early"] is None and rows["late"] == early