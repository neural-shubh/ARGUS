from transformers import pipeline


class EntityExtractor:
    def __init__(self, model_name: str = "dslim/bert-base-NER", device: int = -1):
        self.model_name = model_name
        self.model = pipeline(
            task="ner",
            model=model_name,
            aggregation_strategy="simple",
            device=device,
        )

    @staticmethod
    def _format(predictions) -> list[dict]:
        return [
            {
                "text": item["word"],
                "label": item["entity_group"],
                "score": float(item["score"]),
                "start_char": int(item["start"]),
                "end_char": int(item["end"]),
            }
            for item in predictions
        ]

    def extract(self, text: str) -> list[dict]:
        if not text.strip():
            return []
        return self._format(self.model(text))

    def extract_batch(self, texts: list[str], batch_size: int = 16) -> list[list[dict]]:
        """Run NER over many texts at once; returns one entity list per text."""
        results: list[list[dict]] = [[] for _ in texts]
        indexed = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not indexed:
            return results

        outputs = self.model([t for _, t in indexed], batch_size=batch_size)
        for (i, _), predictions in zip(indexed, outputs):
            results[i] = self._format(predictions)
        return results