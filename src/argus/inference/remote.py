import os

from dotenv import load_dotenv
from gradio_client import Client

load_dotenv()


class RemoteInference:
    def __init__(self):
        token = os.getenv("HF_TOKEN")
        space_id = os.getenv("HF_SPACE_ID")

        if not token:
            raise RuntimeError("HF_TOKEN is missing from .env")

        if not space_id:
            raise RuntimeError("HF_SPACE_ID is missing from .env")

        self.client = Client(
            space_id,
            token=token,
        )

    def analyze_text(self, text: str):
        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        return self.client.predict(
            text,
            api_name="/analyze_text",
        )