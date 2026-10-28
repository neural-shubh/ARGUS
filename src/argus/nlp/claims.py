import os
import re
from datetime import datetime, timedelta

import ollama
from pydantic import BaseModel

MODEL = os.getenv("ARGUS_MODEL", "llama3.2:3b")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class Claim(BaseModel):
    claim: str
    attribution: str | None = None
    evidence: str


class Claims(BaseModel):
    claims: list[Claim]


PROMPT = (
    "Extract every factual claim from the passage below, published by {source}. "
    "Include each number, score, date and quantity it reports, even when several "
    "appear in one sentence.\n"
    "- Write each claim as a self-contained statement that names its subject "
    "(never 'the company', 'it', 'the figure').\n"
    "- Keep 'X said' out of the claim; put the speaker in attribution. If the passage "
    "states it as plain fact, attribution is {source}.\n"
    "- evidence is one exact sentence copied from the passage.\n"
    "- Skip pure opinion. Do not add anything the passage does not say.\n\nPassage:\n"
)


def resolve_dates(claim: str, published: str | None) -> str:
    """Make weekday names and 'next month' absolute, relative to the publication date."""
    if not published:
        return claim
    pub = datetime.fromisoformat(published)

    def weekday(m):  # news says "on Thursday" about the most recent Thursday
        back = (pub.weekday() - WEEKDAYS.index(m.group(1))) % 7
        return f"{m.group(1)} ({pub - timedelta(days=back):%Y-%m-%d})"

    if not re.search(r"\b(will|plans?|scheduled|upcoming)\b", claim, re.I):  # future tense: leave alone
        claim = re.sub(r"\b(" + "|".join(WEEKDAYS) + r")\b", weekday, claim)
    next_month = (pub.replace(day=1) + timedelta(days=32)).replace(day=1)
    return re.sub(r"\bnext month\b", f"in {next_month:%B %Y}", claim)


SCHEMA_HINT = '\n\nReturn only JSON like {"claims": [{"claim": "...", "attribution": "...", "evidence": "..."}]}'


def _generate(prompt: str, temperature: float) -> str:
    """The model's JSON reply as text: Gemini when ARGUS_MODEL starts with 'gemini:', otherwise Ollama."""
    model = os.getenv("ARGUS_MODEL", MODEL)
    if model.startswith("gemini:"):
        import time

        import requests

        for attempt in range(4):
            reply = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model[7:]}:generateContent",
                headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=60,
                json={"contents": [{"parts": [{"text": prompt + SCHEMA_HINT}]}],
                      "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}},
            )
            if reply.status_code == 429:  # free-tier rate limit: wait and retry
                time.sleep(8 * (attempt + 1))
                continue
            reply.raise_for_status()
            return reply.json()["candidates"][0]["content"]["parts"][0]["text"]
        raise RuntimeError("Gemini rate limit reached; try again in a minute.")
    reply = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=Claims.model_json_schema(),  # decoding is constrained, so the JSON is always valid
        options={"temperature": temperature},
    )
    return reply["message"]["content"]


def extract_claims(text: str, source: str = "an unknown source", published: str | None = None) -> list[Claim]:
    flat = " ".join(text.split())  # ignore line breaks when checking evidence
    for temperature in (0, 0.4):  # if the model returns nothing usable, retry once with some variation
        raw = _generate(PROMPT.format(source=source) + text, temperature)
        body = re.search(r"\{.*\}", raw, re.DOTALL)
        claims = Claims.model_validate_json(body.group(0) if body else '{"claims": []}').claims
        kept = [c for c in claims if " ".join(c.evidence.split()) in flat]
        if len(kept) < len(claims):
            print(f"  dropped {len(claims) - len(kept)} claim(s) with ungrounded evidence")
        if kept:
            break
    return [c.model_copy(update={"claim": resolve_dates(c.claim, published)}) for c in kept]
