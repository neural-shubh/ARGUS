import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentence_transformers import CrossEncoder
from argus.analysis import contradictions, load_claims
from argus.nlp.embeddings import EmbeddingEngine
from argus.storage.database import ArgusDatabase

claims = load_claims(ArgusDatabase())
vectors = EmbeddingEngine().encode([c["claim"] for c in claims])
found = [r for r in contradictions(claims, vectors, CrossEncoder("cross-encoder/nli-deberta-v3-small")) if r["score"] >= 0.8]
pick = lambda c: {k: c[k] for k in ("source", "date", "claim")}
out = [{"score": round(float(r["score"]), 2), "older": pick(r["older"]), "newer": pick(r["newer"])} for r in found[:20]]
Path("data/demo").mkdir(parents=True, exist_ok=True)
Path("data/demo/disagreements.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"Saved {len(out)} possible disagreements to data/demo/disagreements.json")
