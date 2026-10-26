import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from argus.analysis import investigate, load_claims
from argus.nlp.claims import MODEL
from argus.nlp.embeddings import EmbeddingEngine
from argus.storage.database import ArgusDatabase

parser = argparse.ArgumentParser(description="Answer a question using only extracted claims, with citations.")
parser.add_argument("question")
parser.add_argument("-k", type=int, default=10, help="how many claims to retrieve")
args = parser.parse_args()

claims = load_claims(ArgusDatabase())
if not claims:
    sys.exit("No claims found. Run scripts/extract_claims.py first.")
engine = EmbeddingEngine()
answer, evidence = investigate(args.question, claims, engine.encode([c["claim"] for c in claims]), engine, MODEL, args.k)

print(answer)
print("\nEvidence:")
for n, c in enumerate(evidence, 1):
    print(f'[{n}] {c["source"]} ({c["date"][:10]}): "{c["evidence"]}"')