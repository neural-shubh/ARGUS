import os 
from gradio_client import Client
from pydantic import BaseModel

SPACE = "https://neural-shubh-argus-inference.hf.space"

class Claim(BaseModel):
    claims: str
    attribution: str | None=None
    evidence: str

_client = None

def extract_claims(text: str, source: str = "an unknown source") -> list[Claim]:
    global _client
    if _client in None:
        _client = Client(SPACE, token = os.getenv('HF_TOKEN'))
    result = _client.predict([text], [source], api_name="/extract_batch")
    return [Claim(**c) for c in result[0]["claims"]]