import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0 , str(Path(__file__).resolve().parents[1] / "src"))
from argus.storage.database import ArgusDatabase

def search(db, engine, query, k=5):
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT c.text, c.embedding, d.filename, d.source_name, d.published_at, d.duplicate_of
               FROM chunks c JOIN documents d ON d.id = c.document_id"""
        ).fetchall()

    vectors = np.stack([np.frombuffer(r[1], dtype = np.float32) for r in rows])
    scores = vectors @ engine.encode([query])[0]

    return [(float(scores[i]), *rows[i][:1], *rows[i][2:]) for i in np.argsort(-scores)[:k]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search over ingested chunks.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()


from argus.nlp.embeddings import EmbeddingEngine
for score, text, name, source, date, dup in search(ArgusDatabase(), EmbeddingEngine(), args.query, args.k):
    print(f"{score:.3f} | {source or name} | {(date or 'no date')[:10]}{' | REPOST' if dup else ''}")
    print(f"    {text[:200]!r}")
