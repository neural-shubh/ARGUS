import argparse, re, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from argus.nlp.embeddings import EmbeddingEngine
from argus.storage.database import ArgusDatabase

parser = argparse.ArgumentParser(description="Remove junk and off-topic claims (dry run unless --apply).")
parser.add_argument("--apply", action="store_true", help="actually delete the flagged claims")
parser.add_argument("--stds", type=float, default=1.5, help="drop claims this many std-devs less on-topic than average")
args = parser.parse_args()

JUNK = re.compile(r"\b(the passage|is mentioned|newsletter|archive|comment section|reader-supported|recent stories|subscribe|readers)\b", re.I)

with ArgusDatabase().connect() as conn:
    rows = conn.execute("SELECT id, claim FROM claims").fetchall()
    vectors = EmbeddingEngine().encode([r[1] for r in rows])
    centre = vectors.mean(0) / np.linalg.norm(vectors.mean(0))
    relevance = vectors @ centre  # how close each claim is to the corpus's main topic
    cutoff = relevance.mean() - args.stds * relevance.std()

    flagged = []
    for (cid, claim), score in zip(rows, relevance):
        reason = ("too short" if len(claim.split()) < 5 else "boilerplate" if JUNK.search(claim)
                  else "off-topic" if score < cutoff else None)
        if reason:
            flagged.append((cid, claim, reason))

    print(f"{len(flagged)} of {len(rows)} claims flagged: {dict(Counter(f[2] for f in flagged))}")
    for _, claim, reason in flagged[:25]:
        print(f"  [{reason}] {claim[:110]}")
    if args.apply:
        conn.executemany("DELETE FROM claims WHERE id = ?", [(f[0],) for f in flagged])
        conn.commit()
        print("Deleted.")
    else:
        print("Dry run. Re-run with --apply to delete.")