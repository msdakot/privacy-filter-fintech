# Label Taxonomy — Fintech Privacy Filter

This document records every label mapping decision across the three data sources and explains the rationale. The goal is a unified 65-label taxonomy that extends OpenMed/privacy-filter-nemotron's 55 labels with 10 fintech-specific identifiers.

---

## Unified Taxonomy

### Inherited from OpenMed/privacy-filter-nemotron (55 labels — unchanged)

| Group | Labels |
|---|---|
| Identity (17) | `first_name`, `last_name`, `user_name`, `age`, `gender`, `race_ethnicity`, `sexuality`, `religious_belief`, `political_view`, `marital_status`, `nationality`, `education_level`, `occupation`, `employment_status`, `language`, `blood_type`, `biometric_identifier` |
| Contact (4) | `email`, `phone_number`, `fax_number`, `url` |
| Address (7) | `street_address`, `city`, `county`, `state`, `country`, `postcode`, `coordinate` |
| Dates (4) | `date`, `date_of_birth`, `date_time`, `time` |
| Government IDs (3) | `ssn`, `national_id`, `tax_id` |
| Financial (7) | `account_number`, `bank_routing_number`, `swift_bic`, `credit_debit_card`, `cvv`, `pin`, `password` |
| Healthcare (2) | `medical_record_number`, `health_plan_beneficiary_number` |
| Enterprise IDs (4) | `customer_id`, `employee_id`, `unique_id`, `certificate_license_number` |
| Vehicle (2) | `license_plate`, `vehicle_identifier` |
| Digital (6) | `ipv4`, `ipv6`, `mac_address`, `device_identifier`, `api_key`, `http_cookie` |

### New fintech-specific labels (10)

| Label | Description | Source |
|---|---|---|
| `iban` | International Bank Account Number (ISO 13616) | Gretel (native), Synthetic |
| `bban` | Basic Bank Account Number (IBAN domestic component) | Gretel (native), Synthetic |
| `lei` | Legal Entity Identifier (ISO 17442) — required for derivatives | Synthetic only |
| `isin` | International Securities Identification Number (ISO 6166) | Synthetic only |
| `cusip` | North American securities identifier (CUSIP 9-char) | Synthetic only |
| `swift_mt_ref` | SWIFT message reference numbers (MT103, MT202 format) | Synthetic only |
| `policy_number` | Insurance policy references | Synthetic only |
| `vat_number` | EU VAT identification numbers | Synthetic only |
| `loan_number` | Mortgage and loan reference numbers | Synthetic only |
| `crypto_address` | Bitcoin/Ethereum/Solana wallet addresses | Synthetic (stretch goal) |

---

## Label Mappings by Source

### Gretel → Unified

| Gretel label | Unified label | Rationale |
|---|---|---|
| `swift_bic_code` | `swift_bic` | Nemotron uses `swift_bic`; standardize |
| `credit_card_number` | `credit_debit_card` | Match nemotron's broader label |
| `credit_card_security_code` | `cvv` | Semantic match |
| `account_pin` | `pin` | Semantic match |
| `local_latlng` | `coordinate` | Semantic match |
| `driver_license_number` | `certificate_license_number` | Broader enterprise ID category |
| `passport_number` | `national_id` | Passport is a national identity document |
| `name` | `first_name` | Gretel uses a single `name` label for full names; treat as first_name for now. **Decision**: splitting into first/last heuristically is error-prone; mapping to first_name is conservative and avoids span corruption. Document as a known limitation. |
| `company` | `unique_id` | No exact nemotron match; company names are not PII in the traditional sense but may be sensitive in fintech context. **Decision**: map to `unique_id` as the closest enterprise identifier. **Revisit if needed.** |
| `iban` | `iban` | Direct (new fintech label) |
| `bban` | `bban` | Direct (new fintech label) |
| `api_key` | `api_key` | 1:1 match |
| `bank_routing_number` | `bank_routing_number` | 1:1 match |
| `customer_id` | `customer_id` | 1:1 match |
| `employee_id` | `employee_id` | 1:1 match |
| `date` | `date` | 1:1 match |
| `date_of_birth` | `date_of_birth` | 1:1 match |
| `date_time` | `date_time` | 1:1 match |
| `email` | `email` | 1:1 match |
| `first_name` | `first_name` | 1:1 match |
| `ipv4` | `ipv4` | 1:1 match |
| `ipv6` | `ipv6` | 1:1 match |
| `last_name` | `last_name` | 1:1 match |
| `password` | `password` | 1:1 match |
| `phone_number` | `phone_number` | 1:1 match |
| `ssn` | `ssn` | 1:1 match |
| `street_address` | `street_address` | 1:1 match |
| `time` | `time` | 1:1 match |
| `user_name` | `user_name` | 1:1 match |

### Nemotron → Unified

Nemotron already uses the OpenMed-nemotron 55-label space. All 55 labels map 1:1 to the inherited taxonomy above. No remapping needed.

### Synthetic → Unified

All synthetic examples use new fintech labels (`lei`, `isin`, `cusip`, `bban`, `iban`, `vat_number`, `swift_mt_ref`, `policy_number`, `loan_number`) — direct mapping, no harmonization needed.

---

## Known Limitations

- **`name` → `first_name`**: Gretel's `name` label covers full names but we map to `first_name`. Some examples will have a full name span tagged as `first_name` — a known imprecision.
- **`company`**: Mapped to `unique_id` as a conservative fallback. Company names are borderline PII in fintech contexts (may identify counterparties in confidential transactions). This mapping may need refinement.
- **Language coverage**: Only English, Spanish, Swedish, German, Italian, Dutch, French (from Gretel). No Hindi, Telugu, Portuguese, or other languages.
