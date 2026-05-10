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

### Synthetic: 390 of targeted 400 (gen failures ~2.5%)
- Template rendering occasionally fails to find the identifier in the rendered text (due to formatting edge cases)
- Retry logic caps at 3× attempts per label — acceptable loss, no action needed

---

## Phase 3 — Training

*(Add entries here as they arise)*

---

## Phase 4 — Evaluation

*(Add entries here as they arise)*
