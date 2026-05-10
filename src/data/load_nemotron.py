from datasets import load_dataset

FINANCE_DOMAINS = {
    "Finance",
    "Identity Verification Services",
    "Real Estate",
    "Insurance",
    "Ecommerce",
    "Retail",
    "Identity Management",
}


def load_nemotron(token: str | None = None) -> list[dict]:
    ds = load_dataset(
        "nvidia/Nemotron-PII",
        split="train",
        token=token,
    )

    records = []
    for row in ds:
        if row.get("domain") not in FINANCE_DOMAINS:
            continue

        # spans is already a list of dicts — no json.loads() needed
        raw_spans = row.get("spans", [])
        normalized = [
            {"start": s["start"], "end": s["end"], "label": s["label"]}
            for s in raw_spans
            if "start" in s and "end" in s and "label" in s
        ]

        records.append({
            "text": row["text"],
            "spans": normalized,
            "source": "nemotron",
            "language": "en",
        })

    return records
