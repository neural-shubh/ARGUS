import re

from argus.nlp.claims import extract_claims

JUNK = re.compile(r"\b(the passage|is mentioned|newsletter|archive|comment section|reader-supported|recent stories|subscribe|readers)\b", re.I)


def extract_pending(db, progress=lambda message, done, total: print(message)):
    """Extract claims for every chunk that has none yet; returns how many chunks were processed."""
    with db.connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY, chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            claim TEXT NOT NULL, attribution TEXT, evidence TEXT NOT NULL)""")
        conn.execute("CREATE TABLE IF NOT EXISTS claimed_chunks (chunk_id INTEGER PRIMARY KEY)")
        todo = conn.execute(
            """SELECT c.id, c.text, COALESCE(d.source_name, d.filename), d.published_at FROM chunks c
               JOIN documents d ON d.id = c.document_id
               WHERE c.id NOT IN (SELECT chunk_id FROM claimed_chunks)"""
        ).fetchall()
        progress(f"{len(todo)} chunks to process", 0, len(todo))

        for done, (chunk_id, text, source, published) in enumerate(todo, 1):
            try:
                claims = extract_claims(text, source, published)
            except Exception as exc: 
                progress(f"chunk {chunk_id} failed: {exc}", done, len(todo))
                continue
            claims = [c for c in claims if len(c.claim.split()) >= 5 and not JUNK.search(c.claim)]
            conn.executemany(
                "INSERT INTO claims (chunk_id, claim, attribution, evidence) VALUES (?, ?, ?, ?)",
                [(chunk_id, c.claim, c.attribution, c.evidence) for c in claims],
            )
            conn.execute("INSERT INTO claimed_chunks VALUES (?)", (chunk_id,))
            conn.commit()
            progress(f"chunk {chunk_id}: {len(claims)} claims", done, len(todo))
    return len(todo)