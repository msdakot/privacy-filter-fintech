"""
Maps source-specific labels to the unified 65-label taxonomy.
All mapping decisions are documented in docs/label_taxonomy.md.
"""

GRETEL_TO_UNIFIED = {
    # Direct renames
    "swift_bic_code": "swift_bic",
    "credit_card_number": "credit_debit_card",
    "credit_card_security_code": "cvv",
    "account_pin": "pin",
    "local_latlng": "coordinate",
    "driver_license_number": "certificate_license_number",
    "passport_number": "national_id",
    # Lossy mappings — documented in label_taxonomy.md
    "name": "first_name",       # full name → first_name (conservative, see docs)
    "company": "unique_id",     # company name → unique_id (closest enterprise ID)
    # 1:1 pass-throughs (listed explicitly for completeness)
    "iban": "iban",
    "bban": "bban",
    "api_key": "api_key",
    "bank_routing_number": "bank_routing_number",
    "customer_id": "customer_id",
    "employee_id": "employee_id",
    "date": "date",
    "date_of_birth": "date_of_birth",
    "date_time": "date_time",
    "email": "email",
    "first_name": "first_name",
    "ipv4": "ipv4",
    "ipv6": "ipv6",
    "last_name": "last_name",
    "password": "password",
    "phone_number": "phone_number",
    "ssn": "ssn",
    "street_address": "street_address",
    "time": "time",
    "user_name": "user_name",
}

# Nemotron already uses the OpenMed-nemotron 55-label space — all 1:1
NEMOTRON_TO_UNIFIED: dict[str, str] = {}

# Synthetic labels are new fintech labels — direct pass-through
SYNTHETIC_LABELS = {
    "lei", "isin", "cusip", "bban", "iban",
    "vat_number", "swift_mt_ref", "policy_number",
    "loan_number", "crypto_address",
}


def harmonize_record(record: dict) -> dict:
    source = record["source"]
    spans = record["spans"]

    if source == "gretel":
        mapping = GRETEL_TO_UNIFIED
    elif source == "nemotron":
        mapping = NEMOTRON_TO_UNIFIED
    else:
        # synthetic — labels already in unified taxonomy
        return record

    new_spans = []
    for span in spans:
        label = span["label"]
        unified = mapping.get(label, label)  # unmapped → keep as-is
        new_spans.append({**span, "label": unified})

    return {**record, "spans": new_spans}


def harmonize(records: list[dict]) -> list[dict]:
    return [harmonize_record(r) for r in records]
