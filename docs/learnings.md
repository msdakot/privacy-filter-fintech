# Project Learnings Log

Running log of discoveries, mistakes, and decisions made during implementation. Updated in-session as things emerge.

---

## Phase 0 — Pre-flight (2026-05-10)

### Schema discovery: Gretel vs Nemotron spans are different types
- **Gretel** `pii_spans`: JSON **string** — must call `json.loads()` before use
- **Nemotron** `spans`: already a **list of dicts** — use directly
- `load_gretel.py` and `load_nemotron.py` must handle these differently; do NOT apply the same parsing logic to both

### OpenMed model uses BIOES tagging (not BIO)
- Label count is 221, not 55 — it's BIOES-tagged: B/I/E/S per entity + O
- 55 entities × 4 tags + 1 O = 221 total
- Training data must be formatted to match the BIOES scheme `opf train` expects
- Config confirmed via `AutoConfig.from_pretrained("OpenMed/privacy-filter-nemotron")`

### PyTorch not installed locally — not a blocker
- Data pipeline (Phases 1–2) runs on CPU only; no model weights needed locally
- Model loading for eval (Phase 4) will use the same venv with PyTorch added
- Training runs on Colab T4 — no local GPU needed

### Nemotron `domain` column confirmed
- Column exists and is a string per row (e.g., `"Finance"`, `"Healthcare"`)
- Filter targets: `Finance`, `Identity Verification Services`, `Real Estate`, `Insurance`, `Ecommerce`, `Retail`, `Identity Management`

---

## Phase 1 — Scaffolding

No issues.

---

## Phase 2 — Data Pipeline

### Nemotron finance-filtered size: 24,866 rows (plan estimated 30-40k)
- Filter domains: Finance, Identity Verification Services, Real Estate, Insurance, Ecommerce, Retail, Identity Management
- 24,866 is at the low end of estimates but still substantial — not a blocker
- Combined with Gretel (50,346) and synthetic (390) gives 68,902 valid records

### 91 invalid spans dropped (0.13% of records)
- Span alignment validator caught 91 records where `start`/`end` offsets were out of bounds relative to text length
- All from Gretel source (Nemotron and synthetic passed cleanly)
- 0.13% drop rate is acceptable — document in data_provenance.md if asked

### Language column was inconsistent across sources — caught post-push
- Gretel stores full language names: `"English"`, `"Spanish"`, `"France"` (not `"French"`)
- Nemotron/synthetic use ISO codes: `"en"`
- Also: Gretel uses `"France"` as a data quirk instead of `"French"`
- Fix: normalization map in `load_gretel.py` (`_LANG_NORM`) → all values now ISO 639-1
- **Lesson**: always check `.unique()` or `Counter()` on categorical columns right after loading, before pushing

### Synthetic: 390 of targeted 400 (gen failures ~2.5%)
- Template rendering occasionally fails to find the identifier in the rendered text (due to formatting edge cases)
- Retry logic caps at 3× attempts per label — acceptable loss, no action needed

---

## Phase 3 — Training

### opf train CLI is very different from what the plan assumed
The plan assumed flags like `--model`, `--dataset <hf_path>`, `--output <hf_repo>`. Actual CLI:
- Takes **local JSONL files** only — no HF Hub paths
- Base model via `--checkpoint <local_dir>` — needs `snapshot_download` first
- Output is a **local directory** via `--output-dir` — HF push is a separate manual step
- `--output-param-dtype bf16` (not `--dtype`)
- `--grad-accum-steps` (not `--grad-accumulation-steps`)
- Custom labels via `--label-space-json` — expects `{"span_class_names": ["O", ...]}`

### Our HF dataset format ≠ opf JSONL format — needs export step
- Our format: `"spans": [{"start": 18, "end": 40, "label": "iban"}]` (list of objects)
- opf format: `"spans": {"iban": [[18, 40]]}` (dict of label → list of [start, end] pairs)
- Added `src/data/export_opf.py` to handle conversion; Colab notebook does this inline

### base model has `company_name` not in the plan's taxonomy
- Plan listed `marital_status` and `nationality` as base labels — neither exists in the actual model
- Base model has `company_name` (plan omitted this)
- Fixed: `harmonize.py` now maps Gretel's `company` → `company_name` (was `unique_id`)
- `configs/label_space.json` uses the actual 55 base labels verified from the model config

### Label space JSON format for opf
- Only need `span_class_names` (entity names without BIOES prefixes)
- opf auto-expands to BIOES token labels internally
- Must include `"O"` as first entry
- 66 total entries = `"O"` + 55 base + 10 fintech

---

## Phase 4 — Evaluation

*(Add entries here as they arise)*
