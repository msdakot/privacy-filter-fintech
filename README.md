# Fintech Privacy Filter

A multilingual, fintech-specialized PII detection model fine-tuned from [`OpenMed/privacy-filter-nemotron`](https://huggingface.co/OpenMed/privacy-filter-nemotron). Extends the base model's 55-label taxonomy with 10 financial-domain entity types (IBAN, LEI, ISIN, CUSIP, VAT numbers, etc.) and adds multilingual coverage across 7 European languages.

> **Status**: Training in progress — results will be updated after the Colab training run completes.

---

## What this project does

Standard PII detection models handle common identifiers (names, emails, SSNs) but treat financial identifiers coarsely or miss them entirely. This model is fine-tuned specifically for fintech use cases — compliance pipelines, document redaction, and data governance — where entities like Legal Entity Identifiers (LEI), ISIN securities codes, and country-specific IBANs matter.

**Key improvements over the base model:**
- 10 new fintech-specific entity labels (LEI, ISIN, CUSIP, BBAN, SWIFT MT refs, VAT numbers, policy numbers, loan numbers)
- Multilingual coverage: English, Spanish, Swedish, German, Italian, Dutch, French
- Trained on real-world financial document formats: FpML, derivative trade confirmations, financial statements, insurance policies

---

## Results

*To be filled in after training. Target: improved F1 on fintech labels with no regression on general PII.*

| Label group | OpenMed-nemotron F1 | This model F1 | Delta |
|---|---|---|---|
| Fintech labels (new) | — | TBD | — |
| Shared financial labels | TBD | TBD | TBD |
| General PII (names, emails) | TBD | TBD | TBD |

**Per-language F1 (this model vs baseline):**

| Language | Baseline F1 | This model F1 |
|---|---|---|
| English (en) | TBD | TBD |
| Spanish (es) | TBD | TBD |
| Swedish (sv) | TBD | TBD |
| German (de) | TBD | TBD |
| Italian (it) | TBD | TBD |
| Dutch (nl) | TBD | TBD |
| French (fr) | TBD | TBD |

---

## Usage

```python
from transformers import pipeline

pipe = pipeline(
    "token-classification",
    model="msdakot/fintech-privacy-filter-v0",
    aggregation_strategy="simple",
)

text = "Wire transfer of €50,000 to IBAN DE89370400440532013000, LEI 529900T8BM49AURSDO55."
results = pipe(text)
for r in results:
    print(f"{r['word']!r:40s} → {r['entity_group']}")
```

---

## Label taxonomy

65-label unified taxonomy: 55 inherited from `OpenMed/privacy-filter-nemotron` + 10 new fintech labels.

**New fintech labels:**

| Label | Description |
|---|---|
| `iban` | International Bank Account Number (ISO 13616) |
| `bban` | Basic Bank Account Number (domestic IBAN component) |
| `lei` | Legal Entity Identifier (ISO 17442) |
| `isin` | International Securities Identification Number (ISO 6166) |
| `cusip` | North American securities identifier |
| `swift_mt_ref` | SWIFT MT message reference numbers |
| `policy_number` | Insurance policy references |
| `vat_number` | EU VAT identification numbers |
| `loan_number` | Mortgage and loan references |
| `crypto_address` | Bitcoin/Ethereum/Solana wallet addresses |

Full taxonomy and mapping rationale: [`docs/label_taxonomy.md`](docs/label_taxonomy.md)

---

## Data sources

| Dataset | License | Records used |
|---|---|---|
| [`gretelai/synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | Apache 2.0 | 55,940 (all) |
| [`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII) | CC-BY-4.0 | ~35k (finance-domain filtered) |
| Synthetic supplement (this project) | Apache 2.0 | ~400 |

Final harmonized dataset: [`msdakot/fintech-privacy-pii`](https://huggingface.co/datasets/msdakot/fintech-privacy-pii)

Full provenance: [`docs/data_provenance.md`](docs/data_provenance.md)

---

## Training

Base model: `OpenMed/privacy-filter-nemotron` (achieves F1 0.95 across 55 labels)  
Framework: [`opf`](https://github.com/openai/privacy-filter) CLI  
Hardware: Google Colab T4 (~3 hours)

Hyperparameters (Vasanth's recipe):
```
learning_rate: 1e-4
epochs: 3
batch_size: 2
grad_accumulation_steps: 4
dtype: bf16
```

Full recipe: [`docs/training_recipe.md`](docs/training_recipe.md)

**Why full fine-tuning over LoRA:** LoRA was considered for parameter efficiency, but full fine-tuning was chosen because (a) `opf` natively supports it with a proven recipe, (b) the label delta is small enough (10 new labels on top of 55) that catastrophic forgetting on general PII is a manageable and measurable risk, and (c) the goal was learning the end-to-end fine-tuning workflow. The regression guardrail in eval (general PII F1 vs baseline) explicitly checks for forgetting.

---

## Reproduce

```bash
git clone https://github.com/msdakot/privacy-filter-fintech
cd privacy-filter-fintech

# 1. Build the dataset (local, CPU)
pip install -e .
bash scripts/prepare_data.sh

# 2. Train (Google Colab — open the notebook)
# notebooks/02_train_colab.ipynb

# 3. Evaluate (local, after training)
bash scripts/run_evaluation.sh
```

---

## Limitations

- Language coverage limited to the 7 Gretel languages (en, es, sv, de, it, nl, fr). Not tested on other languages.
- `name` label from Gretel (full name spans) is mapped to `first_name` in the unified taxonomy — a known simplification.
- Trained on synthetic data only; not validated on real financial documents.
- Compute budget: single T4 run, 3 epochs. Not hyperparameter-tuned.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Training data licenses: Apache 2.0 (Gretel), CC-BY-4.0 (Nemotron).
