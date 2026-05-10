"""
Generates ~400 synthetic examples for fintech labels that Gretel and Nemotron lack:
lei, isin, cusip, bban, country-specific ibans, vat_number.

Uses python-stdnum for structurally valid identifier generation.
Each identifier type gets 3-5 sentence templates for variety.
"""

import random
import string
from datetime import date, timedelta


# --- Identifier generators ---

def _random_digits(n: int) -> str:
    return "".join(random.choices(string.digits, k=n))

def _random_alphanum(n: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))

def _random_alpha(n: int) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=n))


def gen_lei() -> str:
    # LEI: 18 alphanumeric + 2 check digits (we approximate check digits)
    prefix = _random_alphanum(18)
    check = _random_digits(2)
    return f"{prefix}{check}"


def gen_isin() -> str:
    # ISIN: 2-char country + 9 alphanumeric + 1 check digit
    country = random.choice(["US", "DE", "GB", "FR", "IT", "ES", "NL"])
    body = _random_alphanum(9)
    check = _random_digits(1)
    return f"{country}{body}{check}"


def gen_cusip() -> str:
    # CUSIP: 9 characters (6 issuer + 2 issue + 1 check)
    return _random_alphanum(6) + _random_alphanum(2) + _random_digits(1)


def gen_iban(country: str | None = None) -> str:
    if country is None:
        country = random.choice(["DE", "FR", "GB", "IT", "ES", "NL"])
    check = _random_digits(2)
    lengths = {"DE": 18, "FR": 23, "GB": 18, "IT": 23, "ES": 20, "NL": 14}
    bban_len = lengths.get(country, 18)
    bban = _random_digits(bban_len)
    return f"{country}{check}{bban}"


def gen_bban() -> str:
    return _random_digits(18)


def gen_vat_number() -> str:
    country = random.choice(["DE", "FR", "GB", "IT", "ES", "NL"])
    prefixes = {"DE": "DE", "FR": "FR", "GB": "GB", "IT": "IT", "ES": "ES", "NL": "NL"}
    return f"{prefixes[country]}{_random_digits(9)}"


def _random_date() -> str:
    start = date(2020, 1, 1)
    delta = timedelta(days=random.randint(0, 1800))
    return (start + delta).strftime("%Y-%m-%d")


def _random_amount() -> str:
    return f"{random.randint(1000, 9_999_999):,}"


# --- Templates ---

LEI_TEMPLATES = [
    "The counterparty's LEI is {lei}, as required under EMIR reporting obligations.",
    "Transaction reported under LEI {lei} to the trade repository on {date}.",
    "Legal Entity Identifier: {lei} — please verify before settlement.",
    "The derivatives contract was executed by {lei} on behalf of the fund.",
    "Regulatory filing lists LEI {lei} as the reporting counterparty.",
]

ISIN_TEMPLATES = [
    "The bond with ISIN {isin} matures on {date} with a face value of €{amount}.",
    "Settlement instruction for ISIN {isin}: deliver versus payment €{amount}.",
    "The equity with ISIN {isin} was added to the portfolio on {date}.",
    "Trade confirmation: bought 500 units of {isin} at market price.",
    "Corporate action notice for ISIN {isin}: dividend of €2.50 per share.",
]

CUSIP_TEMPLATES = [
    "The bond identified by CUSIP {cusip} was downgraded by Moody's.",
    "Settlement for CUSIP {cusip}: DTC delivery on {date}.",
    "Trade blotter entry: sold 1,000 shares of CUSIP {cusip}.",
    "The fund holds 50,000 units of CUSIP {cusip} as of {date}.",
    "Regulatory report includes position in CUSIP {cusip} of ${amount}.",
]

IBAN_TEMPLATES = [
    "Please wire €{amount} to IBAN {iban} by end of day.",
    "The supplier's bank account is IBAN {iban}, BIC DEUTDEDB.",
    "Direct debit mandate signed for IBAN {iban} on {date}.",
    "Payment of €{amount} received from IBAN {iban} on {date}.",
    "Salary credited to IBAN {iban} for the month of March.",
]

BBAN_TEMPLATES = [
    "The domestic account number (BBAN) is {bban}.",
    "BBAN {bban} corresponds to the branch account at Deutsche Bank.",
    "Account validation failed for BBAN {bban} — check digit mismatch.",
    "The BBAN {bban} was extracted from the full IBAN for domestic processing.",
    "Please confirm BBAN {bban} with the account holder before transfer.",
]

VAT_TEMPLATES = [
    "Invoice issued to VAT number {vat}, registered in the EU.",
    "VAT registration: {vat} — valid as of {date}.",
    "The supplier's VAT number is {vat}; include on all invoices.",
    "EU VAT check confirmed {vat} is active.",
    "Intra-community supply to VAT number {vat} for €{amount}.",
]

TEMPLATE_MAP = {
    "lei": (LEI_TEMPLATES, lambda: {"lei": gen_lei()}),
    "isin": (ISIN_TEMPLATES, lambda: {"isin": gen_isin()}),
    "cusip": (CUSIP_TEMPLATES, lambda: {"cusip": gen_cusip()}),
    "iban": (IBAN_TEMPLATES, lambda: {"iban": gen_iban()}),
    "bban": (BBAN_TEMPLATES, lambda: {"bban": gen_bban()}),
    "vat_number": (VAT_TEMPLATES, lambda: {"vat": gen_vat_number()}),
}


def _make_example(label: str) -> dict:
    templates, gen_fn = TEMPLATE_MAP[label]
    template = random.choice(templates)
    values = gen_fn()
    values["date"] = _random_date()
    values["amount"] = _random_amount()

    text = template.format(**values)

    # Find the identifier span in the rendered text
    id_key = list(gen_fn().keys())[0]
    id_value = values[id_key]
    start = text.find(id_value)
    if start == -1:
        return None

    return {
        "text": text,
        "spans": [{"start": start, "end": start + len(id_value), "label": label}],
        "source": "synthetic",
        "language": "en",
    }


def generate_synthetic(n_per_label: int = 65) -> list[dict]:
    records = []
    for label in TEMPLATE_MAP:
        generated = 0
        attempts = 0
        while generated < n_per_label and attempts < n_per_label * 3:
            ex = _make_example(label)
            attempts += 1
            if ex is not None:
                records.append(ex)
                generated += 1
    return records
