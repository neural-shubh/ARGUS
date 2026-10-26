import sqlite3
import json
from pathlib import Path

import numpy as np


class ArgusDatabase:
    def __init__(
        self,
        db_path: str | Path = "data/database/argus.db",
    ):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.initialize()

    def connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self):
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    character_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    source_name TEXT,
                    published_at TEXT,
                    url TEXT,
                    doc_embedding BLOB,
                    duplicate_of INTEGER REFERENCES documents(id),
                    similarity_to_original REAL,
                    UNIQUE(source_path, content_hash)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    FOREIGN KEY(document_id)
                        REFERENCES documents(id) ON DELETE CASCADE,
                    UNIQUE(document_id, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY,
                    chunk_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    label TEXT NOT NULL,
                    score REAL NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    FOREIGN KEY(chunk_id)
                        REFERENCES chunks(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document
                    ON chunks(document_id);

                CREATE INDEX IF NOT EXISTS idx_entities_label
                    ON entities(label);

                CREATE INDEX IF NOT EXISTS idx_entities_text
                    ON entities(text);
            """)
            self._migrate(conn)

    # Columns added after the first release; applied to existing databases.
    _NEW_DOCUMENT_COLUMNS = {
        "source_name": "TEXT",
        "published_at": "TEXT",
        "url": "TEXT",
        "doc_embedding": "BLOB",
        "duplicate_of": "INTEGER REFERENCES documents(id)",
        "similarity_to_original": "REAL",
    }

    def _migrate(self, conn):
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(documents)")
        }
        for name, ddl in self._NEW_DOCUMENT_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {ddl}")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_published "
            "ON documents(published_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_duplicate "
            "ON documents(duplicate_of)"
        )

        # Backfill document vectors for rows saved before this column existed.
        missing = conn.execute(
            "SELECT id FROM documents WHERE doc_embedding IS NULL"
        ).fetchall()
        for (doc_id,) in missing:
            rows = conn.execute(
                "SELECT embedding FROM chunks WHERE document_id = ?",
                (doc_id,),
            ).fetchall()
            if not rows:
                continue
            vectors = np.stack(
                [np.frombuffer(r[0], dtype=np.float32) for r in rows]
            )
            conn.execute(
                "UPDATE documents SET doc_embedding = ? WHERE id = ?",
                (self._document_vector(vectors).tobytes(), doc_id),
            )

    @staticmethod
    def _document_vector(chunk_vectors: np.ndarray) -> np.ndarray:
        """Mean of chunk embeddings, re-normalized to unit length."""
        mean = np.asarray(chunk_vectors, dtype=np.float32).mean(axis=0)
        norm = np.linalg.norm(mean)
        return (mean / norm if norm > 0 else mean).astype(np.float32)

    def promote_if_earlier(self, new_id: int, old_id: int, similarity: float):
        """If the new document was published before its match, it becomes the original."""
        with self.connect() as conn:
            new, old = (
                conn.execute("SELECT published_at FROM documents WHERE id = ?", (i,)).fetchone()[0]
                for i in (new_id, old_id)
            )
            if new and old and new < old:
                conn.execute(
                    "UPDATE documents SET duplicate_of = ?, similarity_to_original = ? "
                    "WHERE id = ? OR duplicate_of = ?",
                    (new_id, similarity, old_id, old_id),
                )
                conn.execute(
                    "UPDATE documents SET duplicate_of = NULL, similarity_to_original = NULL WHERE id = ?",
                    (new_id,),
                )

    def find_near_duplicate(
        self, doc_vector: np.ndarray, threshold: float
    ) -> tuple[int, float] | None:
        """Most similar *original* document at or above threshold, if any."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, doc_embedding FROM documents
                WHERE duplicate_of IS NULL AND doc_embedding IS NOT NULL
                """
            ).fetchall()

        if not rows:
            return None

        ids = [r[0] for r in rows]
        matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        similarities = matrix @ np.asarray(doc_vector, dtype=np.float32)
        best = int(np.argmax(similarities))

        if similarities[best] >= threshold:
            return ids[best], float(similarities[best])
        return None

    def document_exists(self, source_path: str, content_hash: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM documents
                WHERE source_path = ? AND content_hash = ?
                """,
                (source_path, content_hash),
            ).fetchone()

        return row is not None

    def save_document(
        self,
        document: dict,
        chunks: list[dict],
        embeddings: np.ndarray,
        duplicate_of: int | None = None,
        similarity_to_original: float | None = None,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")

        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO documents (
                    source_path, filename, file_type,
                    content_hash, text, character_count,
                    source_name, published_at, url,
                    doc_embedding, duplicate_of, similarity_to_original
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document["source_path"],
                    document["filename"],
                    document["file_type"],
                    document["content_hash"],
                    document["text"],
                    document["character_count"],
                    document.get("source_name"),
                    document.get("published_at"),
                    document.get("url"),
                    self._document_vector(embeddings).tobytes(),
                    duplicate_of,
                    similarity_to_original,
                ),
            )
            document_id = cursor.lastrowid

            for chunk, vector in zip(chunks, embeddings):
                vector = np.asarray(vector, dtype=np.float32)

                cursor = conn.execute(
                    """
                    INSERT INTO chunks (
                        document_id, chunk_index, start_char,
                        end_char, text, embedding, embedding_dim
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        chunk["chunk_index"],
                        chunk["start_char"],
                        chunk["end_char"],
                        chunk["text"],
                        vector.tobytes(),
                        vector.size,
                    ),
                )
                chunk_id = cursor.lastrowid

                conn.executemany(
                    """
                    INSERT INTO entities (
                        chunk_id, text, label, score,
                        start_char, end_char
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk_id,
                            entity["text"],
                            entity["label"],
                            entity["score"],
                            entity["start_char"],
                            entity["end_char"],
                        )
                        for entity in chunk.get("entities", [])
                    ],
                )

        return document_id

    def stats(self) -> dict:
        with self.connect() as conn:
            documents = conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]

            chunks = conn.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]

            entities = conn.execute(
                "SELECT COUNT(*) FROM entities"
            ).fetchone()[0]

            near_duplicates = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE duplicate_of IS NOT NULL"
            ).fetchone()[0]

        return {
            "documents": documents,
            "near_duplicates": near_duplicates,
            "chunks": chunks,
            "entities": entities,
            "database": str(self.db_path),
        }

    def get_embedding(self, chunk_id: int) -> np.ndarray:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT embedding, embedding_dim
                FROM chunks WHERE id = ?
                """,
                (chunk_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"Chunk not found: {chunk_id}")

        vector = np.frombuffer(row[0], dtype=np.float32).copy()

        if vector.size != row[1]:
            raise ValueError("Stored embedding dimension mismatch")

        return vector

    def search_entity(self, query: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT e.text, e.label, e.score,
                       d.filename, c.text AS chunk_text
                FROM entities e
                JOIN chunks c ON c.id = e.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE e.text = ? COLLATE NOCASE
                ORDER BY e.score DESC
                LIMIT 50
                """,
                (query,),
            ).fetchall()

        return [
            {
                "entity": row[0],
                "label": row[1],
                "score": row[2],
                "filename": row[3],
                "chunk_text": row[4],
            }
            for row in rows
        ]