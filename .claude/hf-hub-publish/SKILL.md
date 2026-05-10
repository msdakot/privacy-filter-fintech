---
name: hf-publishing
description: Use this skill when pushing datasets or models to HuggingFace Hub, writing model cards or dataset cards, handling HF authentication, or managing repo metadata. Covers `huggingface_hub` library usage, `push_to_hub` patterns for datasets and models, ModelCard/DatasetCard APIs, repo creation, license metadata, and the YAML frontmatter conventions HF Hub expects. Trigger when the task involves publishing ML artifacts to HF Hub or writing the documentation that accompanies them.
---

# HuggingFace Hub Publishing Skill

This skill captures the operational details of pushing datasets and models to HuggingFace Hub with proper metadata, model cards, and licensing.

## Authentication

### Get a token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **write** permission (read-only won't allow pushes)
3. Copy the token (shown only once)

### Local authentication

```bash
# Interactive (recommended for dev machines)
huggingface-cli login
# Paste token when prompted

# Or via environment variable (for CI/scripts)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
```

### In Python

```python
from huggingface_hub import login
login(token="hf_xxx")  # not recommended — token in code

# Better: rely on cached credentials from huggingface-cli login
# OR use HF_TOKEN env var
```

### In Colab

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
# Then standard HF library calls work
```

Add `HF_TOKEN` as a Colab Secret (key icon in left sidebar) — never paste tokens in notebook cells.

## Pushing a dataset

### From a `Dataset` or `DatasetDict` object

```python
from datasets import Dataset, DatasetDict

train = Dataset.from_list(train_examples)
val = Dataset.from_list(val_examples)
test = Dataset.from_list(test_examples)

dd = DatasetDict({
    "train": train,
    "validation": val,
    "test": test,
})

dd.push_to_hub(
    "your-username/fintech-pii-unified",
    private=False,  # True during dev if you want
    token=os.environ.get("HF_TOKEN"),  # auto-detected if logged in
)
```

This automatically:
- Creates the repo if it doesn't exist
- Converts to Parquet (efficient for large datasets)
- Generates a basic dataset card stub

### Dataset card YAML frontmatter

After pushing, edit the README.md on the Hub (or push it via API) with proper YAML frontmatter:

```yaml
---
license: apache-2.0  # or other valid SPDX identifier
language:
  - en
  - es
  - de
  - it
  - nl
  - fr
  - sv
task_categories:
  - token-classification
size_categories:
  - 10K<n<100K
tags:
  - pii
  - finance
  - fintech
  - ner
  - privacy
source_datasets:
  - gretelai/synthetic_pii_finance_multilingual
  - nvidia/Nemotron-PII
pretty_name: Fintech PII Unified Dataset
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: validation
        path: data/validation-*
      - split: test
        path: data/test-*
---
```

The frontmatter drives HF Hub's filtering UI — get it right or your dataset is hard to find.

### Pushing the dataset card programmatically

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="path/to/local/README.md",
    path_in_repo="README.md",
    repo_id="your-username/fintech-pii-unified",
    repo_type="dataset",
)
```

## Pushing a model

### Method 1: Local checkpoint folder

After `opf train` produces `/path/to/finetuned_checkpoint/`:

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="/path/to/finetuned_checkpoint",
    repo_id="your-username/fintech-privacy-filter-v0",
    repo_type="model",
    commit_message="Initial fine-tuned checkpoint",
)
```

This uploads everything: `config.json`, `model.safetensors`, `tokenizer.json`, etc.

### Method 2: Via `transformers` (if model loaded in memory)

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load locally trained model
model = AutoModelForTokenClassification.from_pretrained("/path/to/checkpoint", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint", trust_remote_code=True)

# Push
model.push_to_hub("your-username/fintech-privacy-filter-v0")
tokenizer.push_to_hub("your-username/fintech-privacy-filter-v0")
```

### Repo creation in advance (optional)

```python
from huggingface_hub import create_repo

create_repo(
    repo_id="your-username/fintech-privacy-filter-v0",
    repo_type="model",
    private=False,
    exist_ok=True,  # don't error if it already exists
)
```

## Model card structure (CRITICAL)

A good model card is the difference between "interesting project" and "polished portfolio piece." Anatomy:

### YAML frontmatter

```yaml
---
license: apache-2.0
base_model: OpenMed/privacy-filter-nemotron
language:
  - en
  - es
  - de
  - it
  - nl
  - fr
  - sv
pipeline_tag: token-classification
library_name: transformers
tags:
  - pii
  - fintech
  - ner
  - privacy
  - finance
datasets:
  - gretelai/synthetic_pii_finance_multilingual
  - nvidia/Nemotron-PII
metrics:
  - f1
  - precision
  - recall
model-index:
  - name: fintech-privacy-filter-v0
    results:
      - task:
          type: token-classification
        dataset:
          type: gretelai/synthetic_pii_finance_multilingual
          name: Gretel Finance Multilingual (test split)
        metrics:
          - type: f1
            value: 0.XX
            name: Span-level F1 (typed)
---
```

### Body sections (in order)

1. **Model name + one-paragraph summary** — what is this, why does it exist
2. **Quick start** — copy-paste-able usage example
3. **Performance** — headline metrics table comparing to baseline
4. **Label space** — full taxonomy with examples
5. **Training data** — sources, sizes, languages, licenses
6. **Training procedure** — hyperparameters, hardware, runtime
7. **Limitations** — honest section on what it doesn't do well
8. **License + citation**
9. **Acknowledgements** — base model, dataset providers

### Example "Quick start" section that actually works

Always include both the OPF Python API path AND the standard transformers path. Users will try both.

```markdown
## Quick start

### Via OPF (recommended)

\`\`\`bash
pip install 'opf @ git+https://github.com/openai/privacy-filter.git'
\`\`\`

\`\`\`python
import os
os.environ['OPF_MOE_TRITON'] = '0'  # only on non-CUDA hosts

from opf import OPF
from huggingface_hub import snapshot_download

local_path = snapshot_download(repo_id='your-username/fintech-privacy-filter-v0')
model = OPF(model=local_path, device='cuda', decode_mode='viterbi')

text = "Wire payment from IBAN GB82WEST12345698765432 to LEI 549300JKLM01..."
result = model.redact(text)
print(result.redacted_text)
\`\`\`

### Via transformers

\`\`\`python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("your-username/fintech-privacy-filter-v0", trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(
    "your-username/fintech-privacy-filter-v0",
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to("cuda")
\`\`\`
```

### Performance section that lands well

Include three tables:
1. **Headline**: overall F1 vs baseline (one number per model)
2. **Per-label**: F1 breakdown for fintech-specific labels (where you should win)
3. **Per-language**: F1 by language (your multilingual story)

Example structure:
```markdown
## Performance

### Overall (Gretel test split, n=5,594)

| Model | Span-level F1 | Precision | Recall |
|---|---|---|---|
| OpenMed/privacy-filter-nemotron (baseline) | 0.XX | 0.XX | 0.XX |
| **fintech-privacy-filter-v0 (this model)** | **0.XX** | **0.XX** | **0.XX** |

### Fintech-specific labels (where this model adds capability)

| Label | Baseline F1 | This model F1 | Δ |
|---|---|---|---|
| iban | 0.00 (not in baseline) | 0.XX | +0.XX |
| lei | 0.00 (not in baseline) | 0.XX | +0.XX |
| ... | ... | ... | ... |

### Per-language F1

| Language | Examples | Baseline | This model |
|---|---|---|---|
| English | 2,891 | 0.XX | 0.XX |
| Spanish | 461 | 0.XX | 0.XX |
| ... | ... | ... | ... |
```

### Limitations section (DON'T SKIP)

Honest limitations are a credibility signal. For this project:

```markdown
## Limitations

- Trained primarily on synthetic data from Gretel Navigator and NVIDIA NeMo Data Designer.
  Real production financial documents (e.g., actual SWIFT messages, real KYC forms) may show
  different surface forms not seen during training.
- Languages with limited training data (Hindi, Telugu, Portuguese — not present in Gretel)
  are not supported. Despite OpenMed-nemotron's o200k_base tokenizer being multilingual,
  no fine-tuning data was available for these languages in the fintech domain.
- The model adds 10 fintech-specific labels (IBAN, LEI, ISIN, etc.) but does not validate
  identifiers against checksums (e.g., a syntactically valid but checksum-invalid IBAN
  will still be detected as IBAN). Pair with `python-stdnum` for validation.
- This is a redaction aid, not a compliance certification. Inherits the same risks as
  OpenAI Privacy Filter — see https://huggingface.co/openai/privacy-filter for full
  bias/risks documentation.
```

## License compatibility check

When combining datasets from multiple sources, verify license compatibility:

| Dataset | License | Commercial use | Sharing derivatives |
|---|---|---|---|
| nvidia/Nemotron-PII | CC-BY-4.0 | ✅ with attribution | ✅ |
| gretelai/synthetic_pii_finance_multilingual | Apache 2.0 | ✅ | ✅ |
| openai/privacy-filter | Apache 2.0 | ✅ | ✅ |
| OpenMed/privacy-filter-nemotron | Apache 2.0 | ✅ | ✅ |

All compatible. Final model can be released under Apache 2.0. Required: attribution to NVIDIA, Gretel, OpenAI, and OpenMed in the model card.

**Model cards MUST cite all upstream sources.** Example citation block:

```markdown
## Citation

\`\`\`bibtex
@misc{your-fintech-pii-2026,
  author = {Your Name},
  title = {Fintech Privacy Filter v0},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/your-username/fintech-privacy-filter-v0}
}

@misc{openai-privacy-filter-2026,
  author = {OpenAI},
  title = {OpenAI Privacy Filter},
  year = {2026},
  url = {https://huggingface.co/openai/privacy-filter}
}

@dataset{nemotron-pii-2025,
  author = {NVIDIA},
  title = {Nemotron-PII},
  year = {2025},
  url = {https://huggingface.co/datasets/nvidia/Nemotron-PII}
}

@software{gretel-pii-finance-2024,
  author = {Watson, Alex and others},
  title = {Synthetic-PII-Financial-Documents},
  year = {2024},
  url = {https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual}
}
\`\`\`
```

## Pushing the model card

```python
from huggingface_hub import ModelCard

card = ModelCard.load("path/to/local/MODEL_CARD.md")
card.push_to_hub("your-username/fintech-privacy-filter-v0")
```

Or just `upload_file` for the README.md directly:

```python
api.upload_file(
    path_or_fileobj="MODEL_CARD.md",
    path_in_repo="README.md",
    repo_id="your-username/fintech-privacy-filter-v0",
    repo_type="model",
)
```

## Versioning strategy

For an ongoing portfolio project:
- `v0` — initial training run (probably what you ship in 2 days)
- `v1` — after first round of feedback / additional data
- `v2` — major label space changes, etc.

Use git tags on the HF repo for versioning. The HF repo is itself a git repo:
```bash
git clone https://huggingface.co/your-username/fintech-privacy-filter-v0
# ... commit/push as normal git
```

## Common pitfalls

| Issue | Cause | Fix |
|---|---|---|
| `Authorization error` | Token missing or read-only | Re-login with write token |
| YAML frontmatter not rendering | Indentation or invalid SPDX license | Validate at https://huggingface.co/docs/hub/model-cards |
| `base_model` not linking | Repo doesn't exist or typo | Verify the full path `org/name` |
| Dataset shows as "Unable to load" | Schema inconsistency across splits | Use `Dataset.from_list` per split, ensure same columns |
| Model loading fails for users | Missing `trust_remote_code` note in card | Always include this in Quick Start examples |
| License flagged as missing | No `license:` in frontmatter | Add it; without this, HF Hub treats the artifact as "all rights reserved" |
| `push_to_hub` hangs | Large file without git-lfs | `huggingface_hub` handles this automatically; if not, ensure `huggingface_hub>=0.20` |

## Pre-flight checklist before publishing

### Dataset

1. ✅ All splits have the same columns
2. ✅ License is set in frontmatter (`apache-2.0` for our case)
3. ✅ All upstream sources cited in `source_datasets`
4. ✅ `language` covers all included languages
5. ✅ Sample row validates (load it back via `load_dataset` and check)
6. ✅ Dataset card has: purpose, schema, sizes, examples, citation

### Model

1. ✅ All checkpoint files uploaded (config.json, model.safetensors, tokenizer.json)
2. ✅ Model card has: quick start (BOTH `opf` and `transformers`), performance table, limitations, citation
3. ✅ `base_model` field links correctly
4. ✅ License matches base model (apache-2.0 inherited)
5. ✅ `pipeline_tag: token-classification` set
6. ✅ Tags include domain (`fintech`, `pii`)
7. ✅ Test the published model: download fresh in a new env, run inference, verify output matches local
