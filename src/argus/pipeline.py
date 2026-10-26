from pathlib import Path

import numpy as np

from argus.ingestion.reader import load_document, chunk_text
from argus.storage.database import ArgusDatabase

# Cosine similarity between mean-chunk document vectors at/above which a
# document is flagged as a near-duplicate (syndicated/reposted copy).
# Tune on real data: lower catches more rewrites, higher is stricter.
DEFAULT_DUPLICATE_THRESHOLD = 0.97


class ArgusPipeline:
    def __init__(
        self,
        database: ArgusDatabase | None = None,
        entity_extractor=None,
        embedding_engine=None,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    ):
        self.database = database or ArgusDatabase()
        self.duplicate_threshold = duplicate_threshold

        if entity_extractor is None:
            import torch
            from argus.nlp.ner import EntityExtractor

            entity_extractor = EntityExtractor(
                device=0 if torch.cuda.is_available() else -1
            )
        if embedding_engine is None:
            from argus.nlp.embeddings import EmbeddingEngine

            embedding_engine = EmbeddingEngine()

        self.entity_extractor = entity_extractor
        self.embedding_engine = embedding_engine

    def process_file(self, file_path: str | Path) -> dict:
        document = load_document(file_path)

        if self.database.document_exists(
            document["source_path"],
            document["content_hash"],
        ):
            return {
                "status": "skipped",
                "filename": document["filename"],
                "reason": "This file version is already stored.",
            }

        chunks = chunk_text(document["text"])

        if not chunks:
            raise ValueError("No non-empty chunks were generated")

        texts = [chunk["text"] for chunk in chunks]

        # One batched NER pass instead of one model call per chunk.
        all_entities = self.entity_extractor.extract_batch(texts)

        for chunk, entities in zip(chunks, all_entities):
            for entity in entities:
                entity["start_char"] += chunk["start_char"]
                entity["end_char"] += chunk["start_char"]
            chunk["entities"] = entities

        embeddings = self.embedding_engine.encode(texts)

        # Near-duplicate check against documents already stored.
        doc_vector = self.database._document_vector(embeddings)
        match = self.database.find_near_duplicate(
            doc_vector, self.duplicate_threshold
        )
        duplicate_of, similarity = match if match else (None, None)

        document_id = self.database.save_document(
            document=document,
            chunks=chunks,
            embeddings=embeddings,
            duplicate_of=duplicate_of,
            similarity_to_original=similarity,
        )

        if duplicate_of is not None:
            self.database.promote_if_earlier(document_id, duplicate_of, similarity)

        result = {
            "status": "processed",
            "document_id": document_id,
            "filename": document["filename"],
            "characters": document["character_count"],
            "chunks": len(chunks),
            "entities": sum(len(c["entities"]) for c in chunks),
            "embedding_dimensions": int(embeddings.shape[1]),
            "published_at": document["published_at"],
            "source_name": document["source_name"],
        }
        if duplicate_of is not None:
            result["near_duplicate_of"] = duplicate_of
            result["similarity"] = round(similarity, 4)
        return result