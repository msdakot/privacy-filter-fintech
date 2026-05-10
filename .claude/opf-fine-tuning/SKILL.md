---
name: opf-fine-tuning
description: Use this skill when training, evaluating, or running inference with OpenAI's Privacy Filter via the `opf` CLI. Covers the `opf train`, `opf eval`, and `opf` (redact) commands; dataset schema requirements; custom label spaces with `--label-space-json`; eval modes (typed vs untyped) for cross-taxonomy comparison; output checkpoint structure; and common errors. Trigger when the task involves fine-tuning OpenAI Privacy Filter or any of its community fine-tunes (OpenMed/privacy-filter-nemotron, Vasanth155/privacy-filter-india-v5, etc.), evaluating PII detection F1 scores, or comparing models across different label taxonomies.
---

# OpenAI Privacy Filter (opf) Fine-Tuning Skill

This skill captures the operational details of `opf`, OpenAI's CLI for the Privacy Filter model. The `opf` package was released ~Nov 2025 and is recent enough that general LLM training data is sparse on its specifics. This skill is the canonical reference.

## Installation

```bash
# Install from git
pip install 'opf @ git+https://github.com/openai/privacy-filter.git'

# Or from a local clone (preferred if iterating)
git clone https://github.com/openai/privacy-filter.git
cd privacy-filter
pip install -e .
```

After install, the `opf` command is available globally. Verify:
```bash
opf --help
```

Three subcommands exist:
- `opf` (default = `opf redact`) — run inference / redaction
- `opf eval` — evaluate against a labeled dataset
- `opf train` — fine-tune

## Critical environment variables

- `OPF_CHECKPOINT` — path to model checkpoint. Default: `~/.opf/privacy_filter`. If not present, auto-downloads `openai/privacy-filter` from HF.
- `OPF_MOE_TRITON=0` — **REQUIRED on non-CUDA hosts** (CPU, MPS/Apple Silicon). Disables Triton MoE kernels which are CUDA-only. Forgetting this causes cryptic kernel errors on Mac/CPU.

## Dataset schema (CRITICAL)

`opf train` and `opf eval` consume **JSONL** files (one JSON object per line) with this schema:

```json
{"text": "Alice was born on 1990-01-02.", "spans": [{"start": 0, "end": 5, "label": "private_person"}, {"start": 18, "end": 28, "label": "private_date"}]}
```

Required fields:
- `text` (string) — the input text
- `spans` (list of objects) OR `label` — character-level annotations

Each span must have:
- `start` (int, inclusive) — character offset
- `end` (int, exclusive) — character offset
- `label` (string) — must match the label space

**SILENT FAILURE MODES** to watch for:
1. **`text[start:end]` MUST match the entity exactly.** Off-by-one or whitespace-shifted spans look fine to the loader but produce a broken model. ALWAYS write a validator.
2. **Spans can be unordered in source data** (e.g., Gretel datasets). Sort them before any span-arithmetic operation.
3. **Spans can overlap.** `opf train` may handle overlap silently or error — verify behavior on a small batch first.
4. **Empty spans list (`[]`) is valid** — these are negative examples (text with no PII). Keep them.
5. **`label` must exist in the label space.** If using `--label-space-json` and a span references a label not in the JSON, training fails late with a confusing error.

Quick validator to run BEFORE training:
```python
import json

def validate_jsonl(path):
    with open(path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            text = ex["text"]
            for span in ex.get("spans", []):
                actual = text[span["start"]:span["end"]]
                if "text" in span:
                    assert actual == span["text"], (
                        f"Line {i}: span text mismatch. "
                        f"Expected {span['text']!r}, got {actual!r}"
                    )
    print(f"OK: {path}")
```

## Custom label spaces

To train with a label taxonomy different from the base 8 categories, use `--label-space-json`:

```json
{
  "category_version": "fintech_v1",
  "span_class_names": [
    "O",
    "private_person",
    "private_email",
    "private_phone",
    "iban",
    "lei",
    "isin",
    "swift_bic",
    "...etc..."
  ]
}
```

**Rules**:
- `O` MUST be the first entry (background class)
- `span_class_names` is the preferred field name (some older docs may show alternatives — use this one)
- Total class count = `len(span_class_names)` — including `O`
- Output head will have `4 * (N-1) + 1` BIOES classes where N is len of span_class_names

**Critical gotcha**: If you fine-tune from a base model with 55 labels (e.g., OpenMed-nemotron) but provide a `--label-space-json` with only 65 labels, you're REPLACING the head, not extending it. The base model's 55-label knowledge is reinitialized for any non-overlapping label. Plan accordingly:
- If the new label space is a SUPERSET of the old, list the old labels FIRST in the same order, then append new ones — this preserves head weights for matching labels.
- If labels are reordered, head weights for moved labels are scrambled (you'd need to manually remap the classification head).

## Training

Minimal command:
```bash
opf train /path/to/train.jsonl --output-dir /path/to/checkpoint
```

Recommended for production fine-tunes:
```bash
opf train /path/to/train.jsonl \
  --validation-dataset /path/to/validation.jsonl \
  --label-space-json /path/to/custom_label_space.json \
  --checkpoint /path/to/base_model_checkpoint \
  --output-dir /path/to/finetuned_checkpoint
```

Common flag patterns observed in successful fine-tunes (e.g., Vasanth's india-v5):
- 3-5 epochs
- batch_size 2 with grad-accum 4 (effective batch 8) — fits on T4 16GB
- learning_rate 1e-4
- bf16 weights
- weight_decay 0.0

For exact CLI flag names, run `opf train --help` — names may evolve between versions.

## Output artifacts

`--output-dir` produces:
- `config.json` — model architecture + label space
- `model.safetensors` — weights (~2.6-2.8 GB)
- `finetune_summary.json` — training metrics, loss curves
- `USAGE.txt` — quick-reference for using the checkpoint

Push to HF Hub:
```python
from huggingface_hub import upload_folder
upload_folder(
    folder_path="/path/to/finetuned_checkpoint",
    repo_id="your-username/your-model-name",
    repo_type="model",
)
```

## Evaluation: typed vs untyped modes

This is the most subtle and important part of `opf eval`.

### `opf eval --eval-mode typed`
Use when ground truth labels match the model's label space exactly. Reports per-category F1, precision, recall.

### `opf eval --eval-mode untyped`
Use when ground truth labels use a DIFFERENT taxonomy than the model's. This:
- Ignores category identity during matching
- Reports `detection_metrics` (just span-level — was a span detected at all?)
- Reports `ground_truth_label_recall` — for each ground truth label, what fraction of its span text was recalled by ANY predicted span?

**WHY THIS MATTERS for our project**: When comparing your fine-tuned fintech model (65 labels) against the OpenMed-nemotron baseline (55 labels), use `--eval-mode untyped`. This sidesteps the label-mapping headache entirely and gives a fair comparison of "did either model find this PII span?".

```bash
# Compare YOUR model on YOUR labels (per-category metrics)
opf eval test.jsonl --checkpoint /path/to/your_model --eval-mode typed

# Compare BASELINE on YOUR labels (cross-taxonomy fair comparison)
opf eval test.jsonl --checkpoint OpenMed/privacy-filter-nemotron --eval-mode untyped
```

## Predictions output

For deeper analysis, write predictions to JSONL:
```bash
opf eval test.jsonl \
  --checkpoint /path/to/model \
  --predictions-out predictions.jsonl
```

Each line:
```json
{
  "example_id": "...",
  "text": "...",
  "predicted_spans": {"private_person: Alice": [[0, 5]]}
}
```

Note the format: `"label_name: span_text": [[start, end]]`. Multiple spans of the same label/text get bundled into the list.

## Inference (one-off)

```bash
# Single example
opf "Alice was born on 1990-01-02."

# File
opf -f /path/to/file.txt

# Pipe
cat file.txt | opf

# Override checkpoint
opf --checkpoint /path/to/model "text here"

# CPU mode
opf --device cpu "text"

# Output mode: typed (default, shows real labels) or redacted (collapses to one label)
opf --output-mode typed "text"
opf --output-mode redacted "text"
```

JSON output structure (per example):
```json
{
  "schema_version": 1,
  "summary": {"output_mode": "typed", "span_count": 2, "by_label": {...}},
  "text": "...",
  "detected_spans": [
    {"label": "private_person", "start": 0, "end": 5, "text": "Alice", "placeholder": "<PRIVATE_PERSON>"}
  ],
  "redacted_text": "<PRIVATE_PERSON> was born on <PRIVATE_DATE>."
}
```

## Python API (for programmatic use)

Some community models (e.g., Vasanth155/privacy-filter-india-v5) recommend the Python API over CLI:

```python
import os
os.environ['OPF_MOE_TRITON'] = '0'  # required on non-CUDA hosts

from opf import OPF
from huggingface_hub import snapshot_download

local_path = snapshot_download(repo_id='OpenMed/privacy-filter-nemotron')

model = OPF(
    model=local_path,
    device='cuda',  # or 'mps' or 'cpu'
    output_mode='typed',
    decode_mode='viterbi',  # IMPORTANT: viterbi >> argmax for span coherence
)

result = model.redact("Alice's IBAN is GB82WEST12345698765432")
print(result.redacted_text)
for span in result.detected_spans:
    print(f"{span.label}: {span.text} [{span.start}:{span.end}]")
```

**`decode_mode='viterbi'` is the recommended setting** — produces cleaner span boundaries than `argmax`. Both `opf` CLI and OpenMed wrappers default to viterbi.

## Loading via standard transformers (alternative)

OpenMed-nemotron also loads via standard HF Transformers:

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("OpenMed/privacy-filter-nemotron", trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(
    "OpenMed/privacy-filter-nemotron",
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to("cuda")
```

`trust_remote_code=True` is required because the architecture is custom (`openai_privacy_filter` model_type).

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `Triton kernel not available` | CUDA-only kernel on non-CUDA host | `export OPF_MOE_TRITON=0` |
| `Label not in label space` | Span uses a label not declared in `--label-space-json` | Audit dataset; either remove the span or add the label |
| `Span text mismatch` | `text[start:end]` doesn't match expected entity | Span-alignment bug — re-derive offsets after any text normalization |
| `OOM during training` on T4 | Batch too large or sequences too long | Reduce `--per-device-batch-size` to 1, increase `--gradient-accumulation-steps`, or truncate to max_length=512 |
| `trust_remote_code` warning | Custom architecture | Pass `trust_remote_code=True` explicitly |
| Model loads but predictions are nonsense | Likely BIOES/Viterbi decoder bypassed | Use `decode_mode='viterbi'` not raw argmax |

## Pre-flight checklist before training

ALWAYS run these checks before launching a training run:

1. ✅ Train and val JSONL files exist and parse (every line is valid JSON)
2. ✅ Span alignment validator passes on all examples (`text[start:end]` matches)
3. ✅ All span labels appear in `--label-space-json` (or the base model's label space)
4. ✅ `O` is the first entry in `span_class_names`
5. ✅ Base model checkpoint loads cleanly (smoke test: run `opf` with `--checkpoint` on a single example)
6. ✅ `--output-dir` is writable and has ~10 GB free
7. ✅ For Colab: Drive is mounted for checkpoint persistence

## Reference links

- Repo: https://github.com/openai/privacy-filter
- Fine-tuning guide: https://github.com/openai/privacy-filter/blob/main/FINETUNING.md
- Eval modes: https://github.com/openai/privacy-filter/blob/main/EVAL_AND_OUTPUT_MODES.md
- Output schemas: https://github.com/openai/privacy-filter/blob/main/OUTPUT_SCHEMAS.md
- Demo scripts: `examples/scripts/finetuning/finetune_secret_demo.sh` and `finetune_custom_label_demo.sh` in the repo
- Community fine-tune examples (read these for working recipes):
  - https://huggingface.co/OpenMed/privacy-filter-nemotron
  - https://huggingface.co/Vasanth155/privacy-filter-india-v5
