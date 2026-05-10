import json
from datasets import load_dataset

# Gretel stores full language names; normalize to ISO 639-1 codes
_LANG_NORM = {
    "English": "en",
    "Spanish": "es",
    "Swedish": "sv",
    "German": "de",
    "Italian": "it",
    "Dutch": "nl",
    "French": "fr",
    "France": "fr",  # Gretel data quirk — "France" appears instead of "French"
}


def load_gretel(token: str | None = None) -> list[dict]:
    ds = load_dataset(
        "gretelai/synthetic_pii_finance_multilingual",
        split="train",
        token=token,
    )

    records = []
    for row in ds:
        raw_spans = row["pii_spans"]
        # pii_spans is a JSON string — must parse
        try:
            spans = json.loads(raw_spans)
        except (json.JSONDecodeError, TypeError):
            continue

        normalized = [
            {"start": s["start"], "end": s["end"], "label": s["label"]}
            for s in spans
            if "start" in s and "end" in s and "label" in s
        ]

        raw_lang = row.get("language", "en")
        lang = _LANG_NORM.get(raw_lang, raw_lang)

        records.append({
            "text": row["generated_text"],
            "spans": normalized,
            "source": "gretel",
            "language": lang,
        })

    return records
