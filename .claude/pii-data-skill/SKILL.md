---
name: pii-data-pipeline
description: Use this skill when loading, parsing, harmonizing, or validating PII detection datasets for token classification training. Covers Gretel (gretelai/synthetic_pii_finance_multilingual), NVIDIA Nemotron-PII (nvidia/Nemotron-PII), and AI4Privacy formats; their specific schema quirks (JSON-string spans, unordered spans, label naming differences); harmonizing labels across taxonomies into a unified schema; span-alignment validation; train/val/test splitting; and pushing prepared datasets to HF Hub. Trigger when working with PII or NER datasets in JSONL format with character-level span annotations.
---

# PII Data Pipeline Skill

This skill captures the operational details of building a unified PII training dataset from multiple public sources. The schemas look similar at first glance but each has quirks that cause silent failures.

## Source 1: `gretelai/synthetic_pii_finance_multilingual`

**License**: Apache 2.0 (commercial use OK)
**Size**: 55,940 records (50,346 train + 5,594 test)
**Languages**: English (28.9k), Spanish (4.6k), Swedish (4.5k), German (4.5k), Italian (4.5k), Dutch (4.4k), French (4.4k)

### Schema specifics (CRITICAL)

```python
from datasets import load_dataset

ds = load_dataset("gretelai/synthetic_pii_finance_multilingual", split="train")
example = ds[0]
```

**Key fields**:
- `generated_text` (string) — the document text
- `pii_spans` (string) — **JSON-encoded string**, NOT a parsed list
- `language` (string) — one of "English", "Spanish", "Swedish", "German", "Italian", "Dutch", "French"
- `document_type` (string) — e.g., "FpML", "MT940", "SWIFT Message", "Insurance Policy"
- `language_description` (string) — adds locale info (e.g., "English language as spoken in the United States, the UK, or Canada")

### THE #1 GOTCHA: pii_spans is a JSON string

```python
# WRONG — will fail or return characters
spans = example["pii_spans"][0]  # gets first character of the string

# CORRECT
import json
spans = json.loads(example["pii_spans"])
# Now spans is: [{"start": 119, "end": 141, "label": "date"}, ...]
```

### Other Gretel quirks

1. **Spans can be unordered** — sort by `start` before any iteration:
   ```python
   spans = sorted(json.loads(example["pii_spans"]), key=lambda s: s["start"])
   ```

2. **Empty spans are valid** — `pii_spans = "[]"` means no PII in this document. Keep these as negative examples.

3. **Some examples have NO `text` field in spans** — Gretel only includes `start`, `end`, `label`. Don't assume `span["text"]` exists.

4. **Label naming has underscores in unexpected places**: `swift_bic_code` (not `swift_bic`), `credit_card_number` (not `credit_debit_card`), `account_pin` (not `pin`).

5. **Quality scores filter**: each example has `quality_score`, `conformance_score`, `bias_score`, `toxicity_score`, `groundedness_score`. Gretel has already filtered to scores > 80 for quality, but you may want to filter further (e.g., `quality_score >= 90`) for training.

### Gretel's 29 labels

```
account_pin, api_key, bank_routing_number, bban, company,
credit_card_number, credit_card_security_code, customer_id,
date, date_of_birth, date_time, driver_license_number,
email, employee_id, first_name, iban, ipv4, ipv6, last_name,
local_latlng, name, passport_number, password, phone_number,
ssn, street_address, swift_bic_code, time, user_name
```

Note: `name`, `first_name`, `last_name` are SEPARATE labels in Gretel (most data uses `name`).

### Loading example

```python
import json
from datasets import load_dataset

def load_gretel(split="train"):
    ds = load_dataset("gretelai/synthetic_pii_finance_multilingual", split=split)
    examples = []
    for row in ds:
        spans = json.loads(row["pii_spans"])
        spans = sorted(spans, key=lambda s: s["start"])
        examples.append({
            "text": row["generated_text"],
            "spans": spans,
            "language": row["language"],
            "document_type": row["document_type"],
            "source": "gretel",
        })
    return examples
```

## Source 2: `nvidia/Nemotron-PII`

**License**: CC-BY-4.0 (commercial use OK with attribution)
**Size**: 200k rows (100k train + 100k test)
**Languages**: English only

### Schema specifics

```python
ds = load_dataset("nvidia/Nemotron-PII", split="train")
example = ds[0]
```

**Key fields**:
- `text` (string) — the document
- `spans` (list) — **already parsed as a list of dicts** (NOT a JSON string, unlike Gretel)
- `text_tagged` (string) — text with inline `[entity]label` markup (alternative format, ignore for training)
- `domain` (string) — e.g., "Finance", "Healthcare", "Identity Verification Services", "Real Estate"
- `document_type` (string) — e.g., "Bill of Lading", "Tax Return Form", "Equity Allocation Report"
- `locale` (string) — usually "us"
- `document_format` (string) — "structured" or "unstructured"
- `uid` (string) — unique identifier

### Filter to fintech domains

For a fintech-focused fine-tune, filter Nemotron to:
```python
FINTECH_DOMAINS = {
    "Finance", "Identity Verification Services", "Real Estate",
    "Insurance", "Ecommerce", "Retail", "Identity Management"
}

def load_nemotron_fintech(split="train"):
    ds = load_dataset("nvidia/Nemotron-PII", split=split)
    examples = []
    for row in ds:
        if row["domain"] not in FINTECH_DOMAINS:
            continue
        spans = sorted(row["spans"], key=lambda s: s["start"])
        examples.append({
            "text": row["text"],
            "spans": spans,
            "domain": row["domain"],
            "document_type": row["document_type"],
            "source": "nemotron",
        })
    return examples
```

Expected size after filter: ~30-40k examples.

### Nemotron's 55 labels

Identity (17): `first_name`, `last_name`, `user_name`, `age`, `gender`, `race_ethnicity`, `sexuality`, `religious_belief`, `political_view`, `marital_status`, `nationality`, `education_level`, `occupation`, `employment_status`, `language`, `blood_type`, `biometric_identifier`

Contact (4): `email`, `phone_number`, `fax_number`, `url`

Address (7): `street_address`, `city`, `county`, `state`, `country`, `postcode`, `coordinate`

Dates (4): `date`, `date_of_birth`, `date_time`, `time`

Government IDs (3): `ssn`, `national_id`, `tax_id`

Financial (7): `account_number`, `bank_routing_number`, `swift_bic`, `credit_debit_card`, `cvv`, `pin`, `password`

Healthcare (2): `medical_record_number`, `health_plan_beneficiary_number`

Enterprise IDs (4): `customer_id`, `employee_id`, `unique_id`, `certificate_license_number`

Vehicle (2): `license_plate`, `vehicle_identifier`

Digital (6): `ipv4`, `ipv6`, `mac_address`, `device_identifier`, `api_key`, `http_cookie`

Note: `swift_bic` (NO underscore between bic and code, unlike Gretel's `swift_bic_code`).

## Label harmonization

The biggest engineering challenge is mapping Gretel's 29 labels and Nemotron's 55 labels into a unified schema.

### Recommended approach: build on Nemotron's 55, add fintech extensions

Use Nemotron's 55 labels as the foundation (since you're fine-tuning from OpenMed-nemotron which uses these). Map Gretel's labels onto Nemotron's, plus add new fintech labels.

```python
# Gretel → Nemotron label mapping
GRETEL_TO_UNIFIED = {
    "account_pin": "pin",
    "api_key": "api_key",
    "bank_routing_number": "bank_routing_number",
    "bban": "bban",  # NEW (not in Nemotron)
    "company": "company_name",  # Note: Nemotron uses "company_name" sometimes
    "credit_card_number": "credit_debit_card",
    "credit_card_security_code": "cvv",
    "customer_id": "customer_id",
    "date": "date",
    "date_of_birth": "date_of_birth",
    "date_time": "date_time",
    "driver_license_number": "certificate_license_number",  # closest match
    "email": "email",
    "employee_id": "employee_id",
    "first_name": "first_name",
    "iban": "iban",  # NEW (not in Nemotron)
    "ipv4": "ipv4",
    "ipv6": "ipv6",
    "last_name": "last_name",
    "local_latlng": "coordinate",
    "name": "first_name",  # WARNING: lossy. Gretel "name" is full name; we map to first_name as fallback.
                            # Better: split heuristically into first_name + last_name
    "passport_number": "passport_number",  # NEW (not in Nemotron — closest is national_id)
    "password": "password",
    "phone_number": "phone_number",
    "ssn": "ssn",
    "street_address": "street_address",
    "swift_bic_code": "swift_bic",  # underscore difference!
    "time": "time",
    "user_name": "user_name",
}

# New fintech labels added on top of Nemotron's 55
NEW_FINTECH_LABELS = [
    "iban",        # International Bank Account Number
    "bban",        # Basic Bank Account Number
    "lei",         # Legal Entity Identifier
    "isin",        # International Securities ID
    "cusip",       # North American securities ID
    "swift_mt_ref", # SWIFT message reference numbers
    "policy_number",  # insurance
    "vat_number",
    "loan_number",
    "passport_number",  # not in Nemotron base
    "crypto_address",  # optional/stretch
]
```

**Document EVERY mapping decision** in `docs/label_taxonomy.md`. Interview gold.

### The "name" label problem

Gretel's `name` label is full name (e.g., "Jann N. Butte"). Nemotron has separate `first_name` and `last_name`. Three options:

1. **Lossy mapping**: Map all `name` to `first_name` (or to a new `full_name` label). Simple, accepts information loss.
2. **Heuristic split**: Split on whitespace, label first token as `first_name`, last as `last_name`. Wrong sometimes (middle initials, hyphenated names).
3. **Add `full_name` to taxonomy**: Cleanest. Now you have `first_name`, `last_name`, AND `full_name` as distinct labels.

Recommendation: **Option 3** — add `full_name` as a label. It's honest about the source data and gives the model the right vocabulary.

## Span-alignment validation (MANDATORY)

This is the silent killer of NER training. ALWAYS run this:

```python
def validate_spans(examples, name="dataset"):
    errors = []
    for i, ex in enumerate(examples):
        text = ex["text"]
        for span in ex["spans"]:
            # Check bounds
            if span["start"] < 0 or span["end"] > len(text):
                errors.append(f"{name}[{i}]: span out of bounds {span}")
                continue
            # Check ordering
            if span["start"] >= span["end"]:
                errors.append(f"{name}[{i}]: start >= end {span}")
                continue
            # If span has expected text, verify
            extracted = text[span["start"]:span["end"]]
            if "text" in span and span["text"] != extracted:
                errors.append(
                    f"{name}[{i}]: text mismatch. "
                    f"Expected {span['text']!r}, got {extracted!r}"
                )
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} errors")
        for e in errors[:10]:  # first 10
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        raise ValueError("Span validation failed")
    print(f"OK: {name} ({len(examples)} examples)")
```

Run this AFTER loading each source AND after harmonization (in case label remapping introduced errors).

## Train/val/test splitting

For the fintech project, since both Gretel and Nemotron already have train/test splits, use:
- **Train**: Gretel-train + Nemotron-train (filtered)
- **Val**: 10% holdout from train (for early stopping during training)
- **Test**: Gretel-test + Nemotron-test (filtered) — never seen during training

```python
import random

def make_splits(examples, val_fraction=0.1, seed=42):
    random.seed(seed)
    random.shuffle(examples)
    n_val = int(len(examples) * val_fraction)
    return examples[n_val:], examples[:n_val]  # train, val
```

## Writing JSONL for opf

```python
import json

def write_jsonl(examples, path):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps({"text": ex["text"], "spans": ex["spans"]}) + "\n")
```

`opf train` only needs `text` and `spans`. Drop language/source/domain metadata for the training file (keep separately for analysis).

## Pushing to HF Hub

```python
from datasets import Dataset, DatasetDict

train_ds = Dataset.from_list(train_examples)
val_ds = Dataset.from_list(val_examples)
test_ds = Dataset.from_list(test_examples)

dd = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

dd.push_to_hub(
    "your-username/fintech-pii-unified",
    private=False,  # or True if you want it private during dev
)
```

Add a `README.md` to the dataset repo documenting:
- Sources (Gretel + Nemotron with versions/dates)
- Licenses (Apache 2.0 + CC-BY-4.0)
- Label taxonomy
- Train/val/test sizes
- Language distribution
- Citation requirements

## Pre-flight checklist before fine-tuning

1. ✅ Both source datasets load without errors
2. ✅ `pii_spans` from Gretel parsed via `json.loads`
3. ✅ All spans sorted by `start`
4. ✅ Label harmonization applied; no spans reference unmapped labels
5. ✅ Span-alignment validator passes on all examples
6. ✅ Empty-span examples retained as negatives
7. ✅ Train/val/test splits don't overlap (check by uid where available)
8. ✅ Final dataset pushed to HF Hub
9. ✅ JSONL files written to disk for `opf train` to consume
10. ✅ Label space JSON file created with `O` first
