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

### opf train manages its own base model — do not use --checkpoint for fresh fine-tuning
- `--checkpoint` in opf is for resuming from a checkpoint opf itself produced, not for specifying a base model
- For a fresh fine-tune, run `opf train` without `--checkpoint` — opf downloads its own base to `/root/.opf/privacy_filter`
- The HuggingFace-published model (`openai/privacy-filter`) is in inference format and cannot be used as a `--checkpoint` argument
- Pre-download and patch `/root/.opf/privacy_filter/config.json` before training to avoid download mid-run

### openai/privacy-filter config.json is missing fields opf train requires
- Error: `ValueError: Checkpoint config field encoding must be a non-empty string`
- The HuggingFace version is missing `encoding` and `bidirectional_context` fields that opf's runtime requires
- Fix: patch after download — `cfg.setdefault('encoding', 'o200k_base')` and `cfg.setdefault('bidirectional_context', True)`
- Use `setdefault` so existing values are preserved

### Checkpoint write OOMs on standard Colab RAM (exit code -9 = SIGKILL)
- Training completes fine but bf16 checkpoint serialization spikes CPU RAM over 12.7 GB limit
- OOM killer fires instantly — no stderr output, only config.json written to output dir
- Fix: use High RAM runtime (~25 GB) in Colab Pro — Runtime → Change runtime type → High RAM

### OpenMed/privacy-filter-nemotron is incompatible with opf train as a base checkpoint
- Error 1: `ValueError: Checkpoint config field encoding must be a non-empty string` — patching config.json with `encoding` and `bidirectional_context` got past this
- Error 2: `ValueError: num_labels=221 does not match known encoder label spaces (v2:33, v4:57, v7:101)` — opf hardcodes valid checkpoint sizes; OpenMed's 221-label BIOES taxonomy (55 entities × 4 + O) is not in the allowed set
- Root cause: OpenMed was saved in transformers format with a label count opf doesn't recognize. Not patchable.
- Fix: fall back to `openai/privacy-filter` (v2:33, the canonical opf base). Output head expands 8→65 labels via `--label-space-json`.
- Impact: training still covers all 65 labels; the 8 existing opf labels get fine-tuned, the 57 new ones get freshly initialized output heads
- Narrative: "Fine-tuned from openai/privacy-filter and extended to 65 fintech-specialized labels" — still a strong story; base model choice was forced by opf's checkpoint format requirements

### opf train does NOT write incremental checkpoints
- Checkpoint is written at end of training only — mid-run disconnect loses all progress
- The Drive mount is for the final output, not resumption
- Colab Pro requires browser to stay open; Pro+ ($50) adds true background execution
- Corrected misleading "safe to disconnect" note in the notebook

### Label space JSON format for opf
- Only need `span_class_names` (entity names without BIOES prefixes)
- opf auto-expands to BIOES token labels internally
- Must include `"O"` as first entry
- 66 total entries = `"O"` + 55 base + 10 fintech

---

### Training runtime was 5.9 hours, not 2-4 hours
- Estimate was based on OpenMed-nemotron training (55 labels, fine-tuned from opf base)
- Our run starts from 8-label base with 257 randomly initialized output heads — more learning per step
- 55k examples × 3 epochs on T4 = 5.9 hours actual
- Future estimate: plan for 6-8 hours when starting from a low-label base checkpoint

### Val loss still decreasing at epoch 3 — model not fully converged
- Epoch 1→2 val loss drop: 0.093 → 0.084 (large)
- Epoch 2→3 val loss drop: 0.084 → 0.079 (smaller, diminishing returns)
- 1-2 more epochs would likely improve entity-level F1, especially on thin fintech labels
- Resume by pointing `--checkpoint` at the Drive output dir and running `--epochs 2`

---

## Phase 4 — Evaluation

*(Add entries here as they arise)*
