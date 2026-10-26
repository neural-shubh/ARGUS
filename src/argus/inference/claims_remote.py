
import os

from dotenv import load_dotenv
from gradio_client import Client


load_dotenv()


class RemoteClaimExtractor:
    def __init__(self):
        space_id = os.environ["HF_SPACE_ID"]
        token = os.environ["HF_TOKEN"]

        self.client = Client(
            space_id,
            token=token,
        )

    def extract_batch(
        self,
        texts: list[str],
        sources: list[str],
    ) -> list[dict]:
        return self.client.predict(
            texts,
            sources,
            api_name="/extract_batch",
        )
