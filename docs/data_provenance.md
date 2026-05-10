# Data Provenance

Records the origin, license, and usage terms for every data source in this project.

---

## gretelai/synthetic_pii_finance_multilingual

- **License**: Apache 2.0
- **Origin**: Gretel AI — synthetically generated
- **HF path**: `gretelai/synthetic_pii_finance_multilingual`
- **Size used**: All 55,940 records (50,346 train + 5,594 test)
- **Languages**: en, es, sv, de, it, nl, fr
- **Usage**: Primary training source; all records retained

## nvidia/Nemotron-PII

- **License**: Creative Commons Attribution 4.0 (CC-BY-4.0)
- **Origin**: NVIDIA — synthetically generated
- **HF path**: `nvidia/Nemotron-PII`
- **Size used**: Finance-domain filtered subset (~30-40k of 200k total)
- **Filter applied**: `domain` ∈ {Finance, Identity Verification Services, Real Estate, Insurance, Ecommerce, Retail, Identity Management}
- **Usage**: Secondary training source; filtered by domain

## Synthetic supplement (this project)

- **License**: Apache 2.0 (same as this project)
- **Origin**: Generated using `python-stdnum` for valid identifier generation
- **Size**: ~400 examples
- **Labels covered**: `lei`, `isin`, `cusip`, `bban`, `iban` (country-specific), `vat_number`
- **Generation tool**: `src/data/synthetic.py`

---

## Derivative dataset

Final harmonized dataset published as `msdakot/fintech-privacy-pii` under Apache 2.0. All source licenses permit this use.
