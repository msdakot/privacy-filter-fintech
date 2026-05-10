# Fintech PII Detection Model — Project Handoff

## Context for Claude Code

You are taking over a fine-tuning project that has been thoroughly scoped and planned. **Do not redesign the plan.** Execute it. If you find a concrete blocker, raise it explicitly and propose a fix; don't silently change scope.

This is a portfolio/learning project for an ML engineer. The goal is to ship a working fine-tuned model with a clean repo, a public HuggingFace artifact, and a strong narrative — all within ~2 days of work, using Google Colab for the GPU-bound training step.

---

## Project Goal

Build a **fintech-specialized PII detection model** by fine-tuning `OpenMed/privacy-filter-nemotron` on combined public fintech datasets, with an expanded label taxonomy that handles financial-specific identifiers (IBAN, LEI, ISIN, country-specific banking IDs) that the baseline model treats coarsely.

**Resume bullet (the target outcome)**:
> Fine-tuned a multilingual fintech-specialized PII detection model from OpenMed/privacy-filter-nemotron, combining Gretel finance-multilingual (7 languages) and Nemotron-PII into a unified 65-label taxonomy. Achieved [X]% F1 improvement on financial entity types and extended language coverage from English-only to 7 European languages. Open-sourced model, dataset, and reproducible pipeline.

---

## Decisions Already Made (Do Not Revisit)

These were debated extensively. Treat as fixed constraints.

### 1. Fine-tune, don't benchmark
The user wants "I fine-tuned a model" on their resume. A benchmark project would have lower risk but doesn't serve the resume goal.

### 2. Fintech domain (not secret scanning)
Fintech has two strong public datasets with clean licenses. Secret scanning has license-blocked data (`bigcode/bigcode-pii-dataset` prohibits derivative model publishing). Fintech career narrative is also broader (banking, insurance, compliance).

### 3. Fine-tune from `OpenMed/privacy-filter-nemotron`, NOT from `openai/privacy-filter` or `OpenMed/privacy-filter-multilingual`
OpenMed-nemotron already expanded the label space from 8 to 55 labels and proved the recipe works (F1 0.95 across all 55 labels). Building on top means a smaller, more defensible delta and less compute. Final model has both general PII coverage AND fintech specialization.

**Why NOT `OpenMed/privacy-filter-multilingual`** (despite Gretel being multilingual):
- Less battle-tested (released ~3 weeks ago, sparse public documentation)
- Exact label taxonomy unverified — may not match nemotron's 55 labels, which would complicate harmonization
- The underlying base architecture (o200k_base tokenizer) is already multilingual-capable — the model has multilingual *capability* in its substrate; only the supervised PII labeling was English-specialized
- The multilingual story is achievable from the nemotron base by training on Gretel's multilingual data — the tokenizer handles the languages, our training teaches the PII labels in those languages

**Project narrative on multilinguality**: "Started from OpenMed/privacy-filter-nemotron (English-specialized) and extended both domain (fintech) and language coverage (7 languages) by training on Gretel's multilingual fintech data, leveraging the underlying base model's multilingual tokenizer." This is a stronger story than starting from a multilingual base because it demonstrates architectural understanding of how the layered fine-tuning works.

### 4. Public data only
Use `gretelai/synthetic_pii_finance_multilingual` (Apache-2.0) + `nvidia/Nemotron-PII` filtered to finance domains (CC-BY-4.0). Add ~300-500 synthetic examples for international identifiers Gretel/Nemotron lack (LEI, ISIN, country-specific IBAN). DO NOT undertake a synthetic data generation marathon.

### 5. Pipeline-driven, not one-off scripts
Every step is config-driven (YAML), every script reproducible. The user wants a *training pipeline*, not a single training run.

### 6. Colab T4 budget ~$15
Colab Pro at $10/month + ~$5 of compute units. Training run targeted at 2-4 hours on T4.

---

## Architecture: How Local + Colab + HuggingFace Connect

```
[User's Laptop + Claude Code]              [HuggingFace Hub]              [Google Colab]
         |                                        |                              |
         |-- data pipeline (CPU work)             |                              |
         |-- push prepared dataset -------->      |                              |
         |-- write Colab notebook                 |                              |
         |                                        |                              |
         |  (user opens notebook in Colab)        |                              |
         |                                        | <-- pulls dataset -----------|
         |                                        | <-- pulls base model --------|
         |                                        |    (training ~3 hrs)         |
         |                                        | <-- pushes trained model ----|
         |                                        |                              |
         |  <-- pulls trained model --------------|                              |
         |-- runs eval locally                    |                              |
         |-- pushes results to GitHub             |                              |
```

**Key insight**: HuggingFace Hub is the handoff point between local prep and cloud training. Claude Code does NOT need direct Colab access.

---

## Label Taxonomy

### Inherited from OpenMed-nemotron (keep all 55)
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

### NEW: Add for fintech specialization (~10 labels)
- `iban` — International Bank Account Number (with country variants validated)
- `bban` — Basic Bank Account Number (IBAN's domestic component)
- `lei` — Legal Entity Identifier (ISO 17442, required for derivatives reporting)
- `isin` — International Securities ID (ISO 6166)
- `cusip` — North American securities ID
- `swift_mt_ref` — SWIFT message reference numbers
- `policy_number` — insurance policies
- `vat_number` — EU VAT IDs
- `loan_number` — mortgage/loan refs
- `crypto_address` — Bitcoin/Ethereum/Solana wallet addresses (stretch goal, optional)

**Total**: ~65 labels in the unified taxonomy.

**Document the harmonization decisions in `docs/label_taxonomy.md`** — this is the most defensible part of the project for interview discussion.

---

## Datasets

### Primary: `gretelai/synthetic_pii_finance_multilingual`
- **HF path**: `gretelai/synthetic_pii_finance_multilingual`
- **License**: Apache 2.0
- **Size**: 55,940 records (50,346 train + 5,594 test)
- **Languages**: English (28.9k), Spanish (4.6k), Swedish (4.5k), German (4.5k), Italian (4.5k), Dutch (4.4k), French (4.4k)
- **IMPORTANT**: Keep ALL languages. Do NOT filter to English-only. The OpenMed-nemotron base model uses the o200k_base tokenizer (same as GPT-4o) which natively handles all these languages — the multilingual training data extends the model's PII labeling coverage to languages the tokenizer already understands.
- **Schema**: `generated_text` (string) + `pii_spans` (JSON string with `start`, `end`, `label`)
- **Labels (29)**: `account_pin`, `api_key`, `bank_routing_number`, `bban`, `company`, `credit_card_number`, `credit_card_security_code`, `customer_id`, `date`, `date_of_birth`, `date_time`, `driver_license_number`, `email`, `employee_id`, `first_name`, `iban`, `ipv4`, `ipv6`, `last_name`, `local_latlng`, `name`, `passport_number`, `password`, `phone_number`, `ssn`, `street_address`, `swift_bic_code`, `time`, `user_name`
- **Document types**: 100 distinct financial formats including FpML, derivative trades, financial statements, IT support tickets

### Secondary: `nvidia/Nemotron-PII` (filtered)
- **HF path**: `nvidia/Nemotron-PII`
- **License**: CC-BY-4.0
- **Size**: 200k rows (100k train + 100k test) — but you'll filter to finance/insurance domains
- **Schema**: `text` (string) + `spans` (JSON list with `start`, `end`, `text`, `label`)
- **Filter to these domains**: `Finance`, `Identity Verification Services`, `Real Estate`, `Insurance`, `Ecommerce`, `Retail`, `Identity Management`
- **Expected after filter**: ~30-40k rows
- **Labels (55)**: see OpenMed-nemotron — Nemotron uses the same 55-label space

### Synthetic supplement (you generate this)
- ~300-500 examples for `lei`, `isin`, `cusip`, `bban`, country-specific IBANs (DE, FR, GB, IT, ES), `vat_number`, `crypto_address`
- Use `python-stdnum` library for valid identifier generation (it has `lei`, `iso6166` for ISIN, `iban` with country support)
- Embed each in a realistic sentence template (e.g., "The wire transfer to LEI {lei} was completed on {date}")

---

## Repo Structure (Build This Exactly)

```
fintech-privacy-filter/
├── README.md                    # Project narrative, results, usage
├── LICENSE                      # Apache-2.0
├── pyproject.toml               # Dependencies, package metadata
├── .gitignore
│
├── configs/
│   ├── data_config.yaml         # Dataset paths, filters, label mapping
│   ├── train_config.yaml        # Hyperparameters, model paths
│   └── eval_config.yaml         # Eval settings, baseline model
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_gretel.py       # Load + parse Gretel finance dataset
│   │   ├── load_nemotron.py     # Load + filter Nemotron to finance
│   │   ├── synthetic.py         # Generate LEI/ISIN/IBAN synth examples
│   │   ├── harmonize.py         # Unify label taxonomies
│   │   └── build_dataset.py     # Main orchestrator → final HF dataset
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   └── train.py             # Wraps `opf train` CLI with config
│   │
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py           # Span-level F1 via seqeval
│       ├── compare.py           # Side-by-side vs OpenMed baseline
│       └── run_eval.py          # Main eval script
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Look at the datasets
│   ├── 02_train_colab.ipynb         # The Colab notebook to launch training
│   └── 03_evaluation.ipynb          # Generate the final eval report
│
├── scripts/
│   ├── prepare_data.sh          # End-to-end data prep
│   ├── launch_training.sh       # End-to-end training (calls opf)
│   └── run_evaluation.sh        # End-to-end eval
│
├── data/                        # Gitignored; document structure in README
│   ├── raw/                     # Downloaded datasets
│   ├── processed/               # Harmonized + filtered
│   └── synthetic/               # Generated supplements
│
├── outputs/                     # Gitignored
│   ├── checkpoints/
│   ├── eval_results/
│   └── logs/
│
└── docs/
    ├── label_taxonomy.md        # Documents the label schema + mapping rationale
    ├── data_provenance.md       # Where each example came from
    └── training_recipe.md       # Actual hyperparameters used
```

---

## Execution Plan (Ordered Tasks)

### Phase 0: Pre-flight checks (Do FIRST, before writing any code)

This phase exists to fail fast on infrastructure issues.

- [ ] Verify Python 3.10+ available locally
- [ ] Verify HF token works: `huggingface-cli login`
- [ ] Verify git + GitHub access
- [ ] **CRITICAL SMOKE TEST**: Try loading `OpenMed/privacy-filter-nemotron` in a quick script. Verify it works with `transformers` (not just MLX). If this fails, fall back plan: fine-tune from `openai/privacy-filter` base instead. Document the fallback decision in `docs/training_recipe.md`.
- [ ] Try loading both `gretelai/synthetic_pii_finance_multilingual` and `nvidia/Nemotron-PII` (just first 10 rows). Confirm schemas match what's documented above.

**Stop and report to the user if any of these fail.** Do not proceed silently.

### Phase 1: Repo scaffolding (30 min)

- [ ] Create the repo structure exactly as specified above
- [ ] Initialize git, push initial commit to GitHub
- [ ] Set up `pyproject.toml` with dependencies: `transformers`, `datasets`, `huggingface_hub`, `python-stdnum`, `seqeval`, `pyyaml`, `pandas`. The `opf` package goes via git: `opf @ git+https://github.com/openai/privacy-filter.git`
- [ ] Write a brief stub `README.md` (will be filled in at the end)
- [ ] Apache-2.0 LICENSE file

### Phase 2: Data pipeline (3-4 hours of active work)

- [ ] **`src/data/load_gretel.py`**: Loads Gretel dataset, parses `pii_spans` (it's a JSON string!), returns a list of `{text, spans, source: "gretel"}` dicts. Spans should be normalized to `{start, end, label}` format.
- [ ] **`src/data/load_nemotron.py`**: Loads Nemotron, filters to finance-relevant domains, returns same normalized format with `source: "nemotron"`. The Nemotron `domain` column is what you filter on.
- [ ] **`src/data/synthetic.py`**: Generates the 300-500 supplement examples. Use `python-stdnum.lei.generate()`, `python-stdnum.iso6166.generate()` for ISIN, `python-stdnum.iban` for country-specific IBANs. Crypto addresses can be hand-templated (or use a small library). For each generated identifier, wrap in 3-5 different sentence templates so the model sees variety.
- [ ] **`src/data/harmonize.py`**: This is the MOST IMPORTANT file. Maps Gretel's labels to the unified taxonomy + maps Nemotron's labels to the unified taxonomy. Examples of mappings:
  - Gretel `swift_bic_code` → unified `swift_bic`
  - Gretel `credit_card_number` → unified `credit_debit_card`
  - Gretel `name` (single full-name label) → split heuristically or just keep as `private_person`-equivalent
  - Nemotron labels mostly map 1:1
  - Synthetic labels (LEI, ISIN, etc.) are new — direct mapping
- [ ] **`src/data/build_dataset.py`**: Orchestrator. Loads all three sources, harmonizes, deduplicates, splits 80/10/10 (train/val/test), pushes to user's HF Hub as `<username>/fintech-privacy-pii`.
- [ ] **`docs/label_taxonomy.md`**: Document every mapping decision and the rationale. This is interview gold.
- [ ] **Test**: After running `build_dataset.py`, eyeball 20 random examples from the final dataset to verify spans align with text correctly. **DO NOT skip this** — span misalignment is the #1 silent killer of NER training runs.

### Phase 3: Training notebook (1-2 hours)

- [ ] **`configs/train_config.yaml`**: Hyperparameters. Starting point (Vasanth's recipe, which worked):
  - learning_rate: 1e-4
  - epochs: 3
  - batch_size: 2
  - grad_accumulation_steps: 4
  - dtype: bf16
  - weight_decay: 0.0
  - base_model: `OpenMed/privacy-filter-nemotron` (or fallback to `openai/privacy-filter`)
- [ ] **`notebooks/02_train_colab.ipynb`**: Self-contained Colab notebook that:
  1. Mounts Drive (for checkpoint persistence across disconnects)
  2. `pip install`s dependencies (including `opf` from git)
  3. `huggingface-cli login` with token from secrets
  4. Pulls the dataset from HF Hub (the one prepared in Phase 2)
  5. Runs `opf train` with config from YAML
  6. Pushes trained checkpoint to HF Hub as `<username>/fintech-privacy-filter-v0`
- [ ] **Document in the notebook**: expected runtime (~2-4 hours), what to do if Colab disconnects (resume from latest Drive checkpoint), how to use Colab Pro secrets for the HF token

### Phase 4: Evaluation (2-3 hours, AFTER training completes)

- [ ] **`src/eval/metrics.py`**: Span-level F1, precision, recall using `seqeval`. Per-label breakdown.
- [ ] **`src/eval/compare.py`**: Loads both your trained model and `OpenMed/privacy-filter-nemotron`, runs both on the held-out test split, produces a markdown comparison table.
- [ ] **`src/eval/run_eval.py`**: Main script. Outputs results to `outputs/eval_results/`.
- [ ] **`notebooks/03_evaluation.ipynb`**: Pretty visualization of the comparison — bar chart per label, headline metrics.
- [ ] **Critical check**: Compute F1 specifically on the new fintech labels (LEI, ISIN, BBAN, etc.). Compute F1 on labels both models share (IBAN, SWIFT, account_number). Compute F1 on general PII (names, emails) as a guardrail — the model should not have regressed there.
- [ ] **Per-language F1 breakdown**: This is a key result. Show your model's F1 across all 7 Gretel languages (en, es, sv, de, it, nl, fr). Run OpenMed-nemotron baseline on the same per-language splits — it should degrade significantly on non-English data (this is one of your strongest comparison points). Report both in the results table.
- [ ] **Honest scope**: Don't claim support for languages not in Gretel (Hindi, Telugu, Portuguese). Document this in the limitations section.

### Phase 5: Polish & ship (1-2 hours)

- [ ] Write the real `README.md` with:
  - Project narrative
  - Headline results table
  - Usage example (loading the model from HF Hub)
  - Data sources + licenses
  - Training recipe summary
  - Honest limitations section
- [ ] Write the HF model card (similar but model-focused)
- [ ] Push final code to GitHub
- [ ] Push final model to HF Hub
- [ ] (Optional) Brief LinkedIn/blog post

---

## Iteration Cadence with the User

The user prefers oversight on major decisions, not micromanagement. Specifically:

- **Before each phase**: Briefly state what you're about to do and why. Get user nod.
- **Within a phase**: Just execute. Don't ask permission for routine stuff.
- **When you hit a real decision point** (e.g., a label mapping is ambiguous, a dataset has a quirk you didn't expect): Stop and ask.
- **When you hit an error**: Try to fix it twice. If still stuck, surface to user with the specific traceback and your hypothesis.

---

## Things to Watch For

### 1. The OpenMed-nemotron loading smoke test (Phase 0)
This is the highest-risk infrastructure piece. If `AutoModelForTokenClassification.from_pretrained("OpenMed/privacy-filter-nemotron", trust_remote_code=True)` doesn't work cleanly in standard `transformers`, you may need to use the `opf` package's loader instead. If both fail, fall back to fine-tuning from `openai/privacy-filter` (the base model, 8 labels) — same datasets, simpler base, slightly weaker final result but still complete project.

### 2. Span alignment after harmonization
When you remap labels and merge datasets, character offsets must stay correct relative to the text. Write a quick validator: for every `(start, end, label)` span, assert `text[start:end]` matches the expected entity. **Run this on every example before training**.

### 3. Label imbalance
Gretel has 89,642 `name` instances and only 922 `api_key` instances. Without rebalancing, the model will be biased. Either: (a) accept it and document, (b) downsample dominant classes, or (c) use class-weighted loss. Discuss with user before choosing.

### 4. Don't expand scope
If you find yourself wanting to: add a Gradio demo, build an adversarial benchmark too, train a second model, optimize for inference latency — STOP. Ship the core first. The user can do those as Project 2.

### 5. The `opf` CLI is recent
The OpenAI Privacy Filter package was released ~Nov 2025. If you're uncertain about its API, check `https://github.com/openai/privacy-filter` directly. Don't guess from training data.

### 6. Don't redesign the label taxonomy
The taxonomy in this doc is the result of careful research and discussion. If a concrete blocker emerges (e.g., Gretel doesn't actually have a label we listed), surface it; don't silently restructure.

---

## Success Criteria

The project is "done" when:

1. ✅ Trained model is published on HuggingFace Hub under user's account
2. ✅ Dataset (harmonized fintech PII) is published on HuggingFace Hub under user's account
3. ✅ GitHub repo has all code, configs, and docs
4. ✅ README.md has a results table showing F1 vs baseline (OpenMed-nemotron)
5. ✅ Headline metric is positive: F1 improvement on fintech-specific labels, no significant regression on general PII labels
6. ✅ The pipeline is reproducible: someone can clone the repo, run `prepare_data.sh` + the Colab notebook + `run_evaluation.sh` and get the same results

---

## What Not to Do

- ❌ Don't ask the user to choose hyperparameters — start with Vasanth's recipe (lr 1e-4, 3 epochs, bs 2, grad-accum 4, bf16) and only deviate if you have a concrete reason.
- ❌ Don't skip the smoke tests in Phase 0.
- ❌ Don't redesign the label taxonomy unless a concrete blocker emerges.
- ❌ Don't add features the plan doesn't specify (no Gradio demo, no benchmark, no extra models).
- ❌ Don't use synthetic data generation as the primary source — it's a *supplement* to the public datasets only.
- ❌ Don't ignore span-alignment validation.
- ❌ Don't ship without the comparison vs OpenMed-nemotron baseline.

---

## First Action

1. Read this entire document
2. Confirm understanding to the user (briefly — one paragraph)
3. Begin **Phase 0 pre-flight checks** before writing any code
4. Report Phase 0 results before proceeding to Phase 1

If anything in this plan is unclear or seems wrong based on what you discover during pre-flight, surface it to the user immediately rather than making assumptions.

Good luck. The user has done the upstream thinking carefully — the execution should be the easier half of this project.
