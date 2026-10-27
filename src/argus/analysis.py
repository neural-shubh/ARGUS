import re

import numpy as np
from sklearn.cluster import AgglomerativeClustering

UNDATED = "9999"  # undated documents sort last, so they are never "first"
COLUMNS = ("claim", "evidence", "source", "date", "repost", "doc")


def load_claims(db) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT c.claim, c.evidence, COALESCE(d.source_name, d.filename), COALESCE(d.published_at, ''),
                      d.duplicate_of IS NOT NULL, d.id
               FROM claims c JOIN chunks ch ON ch.id = c.chunk_id JOIN documents d ON d.id = ch.document_id"""
        ).fetchall()
    return [dict(zip(COLUMNS, r)) for r in rows]


def _numbers(claim):
    """Figures in a claim (ignores model names like Aurora-2 and ISO dates)."""
    return set(re.findall(r"(?<![\w-])\d+(?:[.,]\d+)*(?![\w-])", claim))


def _group(claims, vectors, indices, threshold, same_claim=False):
    if len(indices) < 2:
        return [indices]
    distance = 1 - vectors[indices] @ vectors[indices].T
    if same_claim:  # claims quoting different figures (or only one quoting any) are never the same claim
        figures = [_numbers(claims[i]["claim"]) for i in indices]
        for a in range(len(indices)):
            for b in range(len(indices)):
                if figures[a] != figures[b]:
                    distance[a, b] = 1.0
    labels = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold
    ).fit_predict(distance.clip(0))
    groups = {}
    for i, label in zip(indices, labels):
        groups.setdefault(label, []).append(i)
    return list(groups.values())


def narratives(claims, vectors, threshold=0.4, repeat=0.25):
    """Narratives, each a list of distinct claims with their first reporter and who repeated them."""
    result = []
    for narrative in sorted(_group(claims, vectors, list(range(len(claims))), threshold), key=len, reverse=True):
        items = []
        for same in _group(claims, vectors, narrative, repeat, same_claim=True):
            same.sort(key=lambda i: (claims[i]["date"] or UNDATED, claims[i]["repost"]))
            first, rest = claims[same[0]], [claims[i] for i in same[1:]]
            items.append({
                "claim": first["claim"], "date": first["date"][:10], "origin": first["source"],
                "repeated_by": sorted({c["source"] + (" (repost)" if c["repost"] else "")
                                       for c in rest if c["source"] != first["source"]}),
                "similar_from_origin": sum(c["source"] == first["source"] for c in rest),
            })
        items.sort(key=lambda it: it["date"] or UNDATED)
        result.append({"size": len(narrative), "claims": items})
    return result


# A real figure: ignores digits glued to names like 70B, 405B or Aurora-2, and ISO dates.
FIGURE = re.compile(r"(?<![\w-])\d+(?:[.,]\d+)*(?![\w-])")


def contradictions(claims, vectors, nli, similarity=0.6, max_pairs=300, figures_only=True):
    """Claim pairs from different documents, scored by how strongly they contradict (best first)."""
    sims = vectors @ vectors.T
    pairs = [(i, j) for i in range(len(claims)) for j in range(i + 1, len(claims))
             if sims[i, j] >= similarity and claims[i]["doc"] != claims[j]["doc"]
             and (not figures_only or (FIGURE.search(claims[i]["claim"]) and FIGURE.search(claims[j]["claim"])))]  # NLI is only reliable on figures/dates
    pairs = sorted(pairs, key=lambda p: -sims[p])[:max_pairs]  # keeps CPU time bounded
    if not pairs:
        return []

    def prob(texts):
        logits = np.asarray(nli.predict(texts))
        p = np.exp(logits - logits.max(1, keepdims=True))
        return (p / p.sum(1, keepdims=True))[:, 0]  # label order: contradiction, entailment, neutral

    forward = [(claims[i]["claim"], claims[j]["claim"]) for i, j in pairs]
    # NLI is directional; only trust a contradiction that holds both ways round.
    scores = np.minimum(prob(forward), prob([(b, a) for a, b in forward]))
    results = []
    for (i, j), score in zip(pairs, scores):
        older, newer = sorted((claims[i], claims[j]), key=lambda c: c["date"] or UNDATED)
        results.append({"score": float(score), "similarity": float(sims[i, j]), "older": older, "newer": newer})
    return sorted(results, key=lambda r: -r["score"])


def investigate(question, claims, vectors, engine, model, k=10):
    """Answer a question from the most relevant claims only. Returns (answer, evidence claims)."""
    pool = [i for i, c in enumerate(claims) if not c["repost"]]  # reposts add nothing new
    scores = vectors[pool] @ engine.encode([question])[0]
    top, seen = [], set()
    for idx in np.argsort(-scores):  # best first, skipping claims resting on a sentence already used
        claim = claims[pool[idx]]
        if claim["evidence"] not in seen:
            seen.add(claim["evidence"])
            top.append(claim)
        if len(top) == k:
            break
    top.sort(key=lambda c: c["date"] or UNDATED)  # oldest first, so the story reads in order
    if model is None:  # demo mode: no language model, return the retrieved evidence only
        return None, top
    import ollama

    items = "\n".join(
        f'[{n}] {c["date"][:10]}, published by {c["source"]}\n    claim: {c["claim"]}\n    original sentence: "{c["evidence"]}"'
        for n, c in enumerate(top, 1)
    )
    prompt = f"""Answer the question using ONLY the numbered items below, citing them like [2].
- Start with a one-line direct answer (Yes / No / Unclear), then explain.
- "Published by" is the outlet, not the speaker. Attribute a statement to a person or organization only if the original sentence says they said it.
- If items disagree, state each side with its source and date.
- If a figure appears in several items, say which source reported it first.
- If the items do not answer the question, say so. Do not add outside knowledge.
- Keep the whole answer to 3-5 sentences of plain prose; do not comment on items one by one.

Items:
{items}

Question: {question}"""
    reply = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
    return reply["message"]["content"], top
