import argparse, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.cluster import AgglomerativeClustering
from argus.storage.database import ArgusDatabase

parser = argparse.ArgumentParser(description="Group claims into narratives and trace who reported each first.")
parser.add_argument("--threshold", type=float, default=0.4, help="max cosine distance within a narrative")
parser.add_argument("--repeat", type=float, default=0.25, help="max cosine distance for two claims to count as the same claim")
args = parser.parse_args()

with ArgusDatabase().connect() as conn:
    rows = conn.execute(
        """SELECT c.claim, COALESCE(d.source_name, d.filename), COALESCE(d.published_at, ''), d.duplicate_of IS NOT NULL
           FROM claims c JOIN chunks ch ON ch.id = c.chunk_id JOIN documents d ON d.id = ch.document_id"""
    ).fetchall()
if len(rows) < 2:
    sys.exit(f"Only {len(rows)} claims in the database. Run scripts/extract_claims.py first.")

from argus.nlp.embeddings import EmbeddingEngine
vectors = EmbeddingEngine().encode([r[0] for r in rows])


def numbers(claim):
    """Figures in a claim (ignores model names like Aurora-2 and ISO dates)."""
    return set(re.findall(r"(?<![\w-])\d+(?:[.,]\d+)*(?![\w-])", claim))


def group(indices, threshold, same_claim=False):
    if len(indices) < 2:
        return [indices]
    distance = 1 - vectors[indices] @ vectors[indices].T
    if same_claim:  # claims quoting different figures (or only one quoting any) are never the same claim
        for a, i in enumerate(indices):
            for b, j in enumerate(indices):
                if numbers(rows[i][0]) != numbers(rows[j][0]):
                    distance[a, b] = 1.0
    labels = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold
    ).fit_predict(distance.clip(0))
    groups = {}
    for i, label in zip(indices, labels):
        groups.setdefault(label, []).append(i)
    return list(groups.values())


for narrative in sorted(group(list(range(len(rows))), args.threshold), key=len, reverse=True):
    print(f"\n== Narrative: {len(narrative)} claims")
    for same in sorted(group(narrative, args.repeat, same_claim=True), key=lambda g: min(rows[i][2] or "9999" for i in g)):
        same.sort(key=lambda i: (rows[i][2] or "9999", rows[i][3]))  # earliest first; originals before reposts
        claim, origin, date, _ = rows[same[0]]
        repeats = sorted({rows[i][1] + (" (repost)" if rows[i][3] else "") for i in same[1:] if rows[i][1] != origin})
        same_source = sum(rows[i][1] == origin for i in same[1:])  # merged wordings from the origin itself
        print(f"  [{date[:10] or '?'}] {claim}\n      first reported by {origin}" + (f"; repeated by {', '.join(repeats)}" if repeats else "")
              + (f" (+{same_source} similar claim(s) from {origin})" if same_source else ""))