import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from argus.analysis import contradictions, load_claims
from argus.nlp.embeddings import EmbeddingEngine
from argus.storage.database import ArgusDatabase

parser = argparse.ArgumentParser(description="Find claims from different documents that disagree.")
parser.add_argument("--similarity", type=float, default=0.6, help="min topical similarity for a pair to be compared")
parser.add_argument("--confidence", type=float, default=0.7, help="min NLI probability to report")
parser.add_argument("--all", action="store_true", help="also compare claims without figures or dates (noisy)")
args = parser.parse_args()

claims = load_claims(ArgusDatabase())
if len(claims) < 2:
    sys.exit(f"Only {len(claims)} claims in the database. Run scripts/extract_claims.py first.")
vectors = EmbeddingEngine().encode([c["claim"] for c in claims])

from sentence_transformers import CrossEncoder
results = contradictions(claims, vectors, CrossEncoder("cross-encoder/nli-deberta-v3-small"), args.similarity, figures_only=not args.all)
if not results:
    sys.exit("No comparable claim pairs found. Try a lower --similarity.")

print(f"\nCompared {len(results)} claim pairs from different documents.")
for rank, r in enumerate(results):
    if r["score"] < args.confidence and rank >= 5:
        break  # always show the 5 closest pairs, so a quiet run still tells you something
    label = "CONTRADICTION" if r["score"] >= args.confidence else "closest, below --confidence"
    print(f"\n{label} ({r['score']:.2f}, topic similarity {r['similarity']:.2f})")
    for c in (r["older"], r["newer"]):
        print(f"  [{c['date'][:10] or '?'}] {c['source']}: {c['claim']}")