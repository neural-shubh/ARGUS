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


def extract_claims(text: str, source: str = "an unknown source", published: str | None = None) -> list[Claim]:
    flat = " ".join(text.split())  # ignore line breaks when checking evidence
    for temperature in (0, 0.4):  # if the model returns nothing usable, retry once with some variation
        reply = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(source=source) + text}],
            format=Claims.model_json_schema(),  # decoding is constrained, so the JSON is always valid
            options={"temperature": temperature},
        )
        claims = Claims.model_validate_json(reply["message"]["content"]).claims
        kept = [c for c in claims if " ".join(c.evidence.split()) in flat]
        if len(kept) < len(claims):
            print(f"  dropped {len(claims) - len(kept)} claim(s) with ungrounded evidence")
        if kept:
            break
    return [c.model_copy(update={"claim": resolve_dates(c.claim, published)}) for c in kept]