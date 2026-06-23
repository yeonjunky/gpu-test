"""Task C scorer: Needle in a Haystack, exact substring match.

Strict per the project spec -- no fuzzy/partial credit, consistent with NIAH
convention (the model either reproduced the exact secret or it didn't).
"""


def score_sample(model_output_text: str, secret_value: str) -> dict:
    found = secret_value in model_output_text
    return {"found": found, "score": 1.0 if found else 0.0}
