
from argus.nlp.claims import extract_claims

text = (
    "On 12 March 2025, Acme Motors announced that it sold "
    "12,000 vehicles during the previous quarter."
)

claims = extract_claims(text, source="Example News")

for claim in claims:
    print(claim.model_dump())
